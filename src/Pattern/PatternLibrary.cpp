// SPDX-License-Identifier: GPL-3.0-or-later
// Clunk — LLVM IR superoptimiser
// Copyright (C) 2025 Clunk contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

/*
 * Clunk Pattern Library — persistent store of optimisation patterns.
 * Self-improving: discovered patterns are stored and reused, and
 * the pattern matcher itself is Clunk-optimised.
 */
#include "clunk/Pattern/PatternLibrary.h"

#include "clunk/IR/Verify.h"
#include "clunk/IR/Clone.h"  // for ir::validate_function
#include "clunk/Parser/IRParser.h"  // reparse persisted pattern IR into functions

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

namespace {
// The persistence format is line-based, but a pattern's source/replacement
// IR is a multi-line `define ... { ... }`. Escape newlines (and backslashes)
// so each value stays on one line; reverse on load.
std::string escape_ir(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}
std::string unescape_ir(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char n = s[i + 1];
            if (n == 'n')  { o += '\n'; ++i; continue; }
            if (n == '\\') { o += '\\'; ++i; continue; }
        }
        o += s[i];
    }
    return o;
}
// Best-effort: parse a full-function IR string into a Function. Returns null
// for the built-in patterns' non-IR templates (e.g. "%r = mul %x, <pow2>"),
// which simply remain text-only and never match.
std::shared_ptr<clunk::ir::Function> parse_pattern_fn(const std::string& ir_text) {
    if (ir_text.find("define") == std::string::npos) return nullptr;
    clunk::parser::IRParser p;
    auto mod = p.parse_string(ir_text);
    if (!mod) return nullptr;
    for (auto& f : mod->functions())
        if (f && !f->blocks().empty()) return f;
    return nullptr;
}
}  // namespace

namespace clunk::pattern {

// ── ArchDescriptor::distance ────────────────────────────────────────────────

double ArchDescriptor::distance(const ArchDescriptor& other) const {
    double d = 0.0;

    // Different vendor is a significant difference
    if (vendor != other.vendor) {
        d += 0.3;
    }

    // CPU vs GPU is a very large difference
    if (is_gpu != other.is_gpu) {
        d += 0.5;
    }

    // Vector width difference — scale by ratio difference
    {
        double max_w = std::max(static_cast<double>(vector_width),
                                static_cast<double>(other.vector_width));
        double min_w = std::min(static_cast<double>(vector_width),
                                static_cast<double>(other.vector_width));
        if (max_w > 0.0) {
            d += (1.0 - min_w / max_w) * 0.1;
        }
    }

    // Cache size differences — scaled contributions
    {
        auto cache_dist = [](unsigned a, unsigned b) -> double {
            double max_c = std::max(static_cast<double>(a), static_cast<double>(b));
            double min_c = std::min(static_cast<double>(a), static_cast<double>(b));
            if (max_c > 0.0) return (1.0 - min_c / max_c) * 0.05;
            return 0.0;
        };
        d += cache_dist(l1_cache_kb, other.l1_cache_kb);
        d += cache_dist(l2_cache_kb, other.l2_cache_kb);
        d += cache_dist(l3_cache_kb, other.l3_cache_kb);
    }

    // GPU-specific: compute capability difference
    if (is_gpu && other.is_gpu) {
        if (compute_capability != other.compute_capability) {
            double diff = std::abs(static_cast<double>(compute_capability) -
                                   static_cast<double>(other.compute_capability));
            d += std::min(diff / 100.0, 0.1);  // Scale: 10 CC difference = 0.01
        }
        // Warp size difference
        if (warp_size != other.warp_size) {
            d += 0.05;
        }
        // Shared memory size difference
        {
            double max_s = std::max(static_cast<double>(shared_mem_kb),
                                    static_cast<double>(other.shared_mem_kb));
            double min_s = std::min(static_cast<double>(shared_mem_kb),
                                    static_cast<double>(other.shared_mem_kb));
            if (max_s > 0.0) {
                d += (1.0 - min_s / max_s) * 0.03;
            }
        }
    }

    // Feature flag differences
    if (has_fma != other.has_fma)     d += 0.02;
    if (has_avx2 != other.has_avx2)   d += 0.02;
    if (has_avx512 != other.has_avx512) d += 0.02;
    if (has_sve != other.has_sve)     d += 0.02;

    return std::min(d, 1.0);
}

// ── Constructor ─────────────────────────────────────────────────────────────

PatternLibrary::PatternLibrary() {
    seed_builtin_patterns();
}

// ── load / save ─────────────────────────────────────────────────────────────

bool PatternLibrary::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;

    // Simple line-based format:
    // Each pattern is a block of key=value lines separated by a blank line.
    // Keys: id, name, description, source_ir, replacement_ir,
    //        discovered_arch_name, discovered_arch_vendor, discovered_arch_is_gpu,
    //        avg_speedup, scope, tags (comma-separated)

    OptimisationPattern current;
    bool in_pattern = false;

    auto flush_pattern = [&]() {
        if (in_pattern && !current.id.empty()) {
            // Reconstruct the pattern functions from the persisted IR so the
            // matcher can use them. Best-effort: non-IR
            // templates parse to null and remain text-only.
            if (!current.source_function)
                current.source_function = parse_pattern_fn(current.source_ir);
            if (!current.replacement_function)
                current.replacement_function = parse_pattern_fn(current.replacement_ir);
            add_pattern(current);
        }
        current = OptimisationPattern{};
        in_pattern = false;
    };

    std::string line;
    while (std::getline(ifs, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            flush_pattern();
            continue;
        }
        line = line.substr(start);
        // Skip comments
        if (!line.empty() && line[0] == '#') continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        in_pattern = true;

        if (key == "id") {
            current.id = val;
        } else if (key == "name") {
            current.name = val;
        } else if (key == "description") {
            current.description = val;
        } else if (key == "source_ir") {
            current.source_ir = unescape_ir(val);
        } else if (key == "replacement_ir") {
            current.replacement_ir = unescape_ir(val);
        } else if (key == "discovered_arch_name") {
            current.discovered_arch.name = val;
        } else if (key == "discovered_arch_vendor") {
            current.discovered_arch.vendor = val;
        } else if (key == "discovered_arch_is_gpu") {
            current.discovered_arch.is_gpu = (val == "1" || val == "true");
        } else if (key == "avg_speedup") {
            current.avg_speedup = std::stod(val);
        } else if (key == "scope") {
            if (val == "InstructionLevel") {
                current.scope = OptimisationPattern::Scope::InstructionLevel;
            } else if (val == "BlockLevel") {
                current.scope = OptimisationPattern::Scope::BlockLevel;
            } else if (val == "FunctionLevel") {
                current.scope = OptimisationPattern::Scope::FunctionLevel;
            } else if (val == "KernelLevel") {
                current.scope = OptimisationPattern::Scope::KernelLevel;
            }
        } else if (key == "tags") {
            std::istringstream ts(val);
            std::string tag;
            while (std::getline(ts, tag, ',')) {
                // Trim
                size_t b = tag.find_first_not_of(" \t");
                size_t e = tag.find_last_not_of(" \t");
                if (b != std::string::npos && e != std::string::npos) {
                    current.tags.push_back(tag.substr(b, e - b + 1));
                }
            }
        }
    }
    flush_pattern();

    library_path_ = path;
    return true;
}

bool PatternLibrary::save(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;

    for (auto& [id, pat] : patterns_) {
        ofs << "id=" << pat.id << "\n";
        ofs << "name=" << pat.name << "\n";
        ofs << "description=" << pat.description << "\n";
        ofs << "source_ir=" << escape_ir(pat.source_ir) << "\n";
        ofs << "replacement_ir=" << escape_ir(pat.replacement_ir) << "\n";
        ofs << "discovered_arch_name=" << pat.discovered_arch.name << "\n";
        ofs << "discovered_arch_vendor=" << pat.discovered_arch.vendor << "\n";
        ofs << "discovered_arch_is_gpu=" << (pat.discovered_arch.is_gpu ? "1" : "0") << "\n";
        ofs << "avg_speedup=" << pat.avg_speedup << "\n";

        const char* scope_str = "InstructionLevel";
        switch (pat.scope) {
            case OptimisationPattern::Scope::InstructionLevel: scope_str = "InstructionLevel"; break;
            case OptimisationPattern::Scope::BlockLevel:       scope_str = "BlockLevel"; break;
            case OptimisationPattern::Scope::FunctionLevel:    scope_str = "FunctionLevel"; break;
            case OptimisationPattern::Scope::KernelLevel:      scope_str = "KernelLevel"; break;
        }
        ofs << "scope=" << scope_str << "\n";

        ofs << "tags=";
        for (size_t i = 0; i < pat.tags.size(); ++i) {
            if (i > 0) ofs << ",";
            ofs << pat.tags[i];
        }
        ofs << "\n";

        ofs << "\n";  // Blank line separator
    }

    return true;
}

// ── add / remove ────────────────────────────────────────────────────────────

void PatternLibrary::index_pattern_tags(const std::string& pattern_id,
                                         const std::vector<std::string>& tags) {
    for (const auto& t : tags) {
        auto& bucket = tag_index_[t];
        // Guard against duplicate tags within a single pattern: the original
        // find_by_tag returned each matching pattern at most once, so we
        // preserve that invariant by not inserting the id twice.
        if (std::find(bucket.begin(), bucket.end(), pattern_id) == bucket.end()) {
            bucket.push_back(pattern_id);
        }
    }
}

void PatternLibrary::rebuild_tag_index() {
    tag_index_.clear();
    for (const auto& [id, pat] : patterns_) {
        index_pattern_tags(id, pat.tags);
    }
}

void PatternLibrary::add_pattern(const OptimisationPattern& pattern) {
    // If a pattern with the same id already exists, remove its old tags from
    // the index before we overwrite it.
    auto existing = patterns_.find(pattern.id);
    if (existing != patterns_.end()) {
        for (const auto& t : existing->second.tags) {
            auto bucket_it = tag_index_.find(t);
            if (bucket_it == tag_index_.end()) continue;
            auto& bucket = bucket_it->second;
            bucket.erase(std::remove(bucket.begin(), bucket.end(), pattern.id), bucket.end());
            if (bucket.empty()) tag_index_.erase(bucket_it);
        }
    }
    patterns_[pattern.id] = pattern;
    index_pattern_tags(pattern.id, pattern.tags);
}

bool PatternLibrary::remove_pattern(const std::string& pattern_id) {
    auto it = patterns_.find(pattern_id);
    if (it == patterns_.end()) return false;
    // Remove this pattern's tags from the index.
    for (const auto& t : it->second.tags) {
        auto bucket_it = tag_index_.find(t);
        if (bucket_it == tag_index_.end()) continue;
        auto& bucket = bucket_it->second;
        bucket.erase(std::remove(bucket.begin(), bucket.end(), pattern_id), bucket.end());
        if (bucket.empty()) tag_index_.erase(bucket_it);
    }
    patterns_.erase(it);
    return true;
}

// ── find_by_tag / find_by_name ──────────────────────────────────────────────

std::vector<const OptimisationPattern*> PatternLibrary::find_by_tag(const std::string& tag) const {
    std::vector<const OptimisationPattern*> results;
    auto idx_it = tag_index_.find(tag);
    if (idx_it != tag_index_.end()) {
        // Fast path: use the tag index.
        for (const auto& id : idx_it->second) {
            auto pat_it = patterns_.find(id);
            if (pat_it != patterns_.end()) {
                results.push_back(&pat_it->second);
            }
        }
        return results;
    }
    // Fallback: linear scan over all patterns (used if tag is not in the index,
    // e.g. if the index was not maintained for some reason).
    for (auto& [id, pat] : patterns_) {
        for (auto& t : pat.tags) {
            if (t == tag) {
                results.push_back(&pat);
                break;
            }
        }
    }
    return results;
}

std::vector<const OptimisationPattern*> PatternLibrary::find_by_name(const std::string& name) const {
    std::vector<const OptimisationPattern*> results;
    for (auto& [id, pat] : patterns_) {
        if (pat.name == name) {
            results.push_back(&pat);
        }
    }
    return results;
}

// ── Statistics ──────────────────────────────────────────────────────────────

void PatternLibrary::record_application(const std::string& pattern_id) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    auto it = patterns_.find(pattern_id);
    if (it != patterns_.end()) {
        it->second.application_count++;
    }
}

void PatternLibrary::record_verification(const std::string& pattern_id) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    auto it = patterns_.find(pattern_id);
    if (it != patterns_.end()) {
        it->second.verification_count++;
    }
}

// ── match ───────────────────────────────────────────────────────────────────

std::vector<PatternMatch> PatternLibrary::match(const ir::Function& fn,
                                                  const ArchDescriptor& target_arch) const {
    std::vector<PatternMatch> results;

    for (auto& [id, pat] : patterns_) {
        // Check architecture compatibility
        if (!is_compatible(pat, target_arch)) continue;

        // If the pattern has no source_function, we cannot match
        if (!pat.source_function) continue;

        // Try to match against each basic block
        for (auto& block : fn.blocks()) {
            if (!block) continue;

            // Try every starting position in the block
            size_t pattern_len = 0;
            if (pat.source_function->entry_block()) {
                pattern_len = pat.source_function->entry_block()->size();
            }
            if (pattern_len == 0) continue;

            for (size_t start = 0; start + pattern_len <= block->size(); ++start) {
                double confidence = 0.0;
                if (match_instruction_sequence(*block, start, *pat.source_function, confidence)) {
                    PatternMatch m;
                    m.pattern_id = pat.id;
                    m.block_name = block->name();
                    m.instruction_start = start;
                    m.instruction_end = start + pattern_len;
                    m.confidence = confidence;
                    m.estimated_speedup = pat.avg_speedup;
                    results.push_back(m);
                }
            }
        }
    }

    return results;
}

// ── apply ───────────────────────────────────────────────────────────────────

std::shared_ptr<ir::Function> PatternLibrary::apply(
    const ir::Function& fn,
    const PatternMatch& match,
    const ArchDescriptor& target_arch) const
{
    auto it = patterns_.find(match.pattern_id);
    if (it == patterns_.end()) return nullptr;

    const OptimisationPattern& pat = it->second;

    // If discovered on a different architecture, try to adapt
    const OptimisationPattern* effective_pat = &pat;
    OptimisationPattern adapted;

    if (pat.discovered_arch.distance(target_arch) > 0.01) {
        auto maybe_adapted = adapt_pattern(pat, target_arch);
        if (maybe_adapted) {
            adapted = std::move(*maybe_adapted);
            effective_pat = &adapted;
        }
        // If adaptation failed, we proceed with the original — the caller
        // can decide whether to trust an incompatible pattern.
    }

    if (!effective_pat->replacement_function) return nullptr;
    if (!effective_pat->source_function) return nullptr;

    auto src_entry = effective_pat->source_function->entry_block();
    auto repl_entry = effective_pat->replacement_function->entry_block();
    if (!src_entry || !repl_entry) return nullptr;

    // Build an operand binding map from the matched
    // source pattern's operand names to the target function's operand
    // values. The replacement instructions may reference the source
    // pattern's operand names (e.g. `add_zero_elim` source is
    // `add %x, 0`; replacement is `ret %x`). Without rewriting,
    // pasting the replacement verbatim would leave dangling references
    // to `%x` (which exists only in the pattern's source function).
    //
    // We walk the matched instruction range in parallel: pattern source
    // inst at position i ↔ target inst at position match.instruction_start + i.
    // For each operand position, if the pattern's operand is a NAMED
    // value, we bind pattern_name → target_value (a shared_ptr<Value>).
    // If the pattern's operand is a ConstantInt, we don't bind it —
    // the replacement's constant operands stay as constants.
    std::unordered_map<std::string, std::shared_ptr<ir::Value>> operand_map;

    auto target_block = fn.block(match.block_name);
    if (!target_block) return nullptr;

    size_t pat_len = src_entry->size();
    if (match.instruction_start + pat_len > target_block->size()) return nullptr;

    // Check that the matched range still matches (it should — match()
    // just produced this PatternMatch). We also build the operand_map.
    for (size_t i = 0; i < pat_len; ++i) {
        auto pat_inst = src_entry->instruction(i);
        auto tgt_inst = target_block->instruction(match.instruction_start + i);
        if (!pat_inst || !tgt_inst) return nullptr;
        if (pat_inst->opcode() != tgt_inst->opcode()) return nullptr;
        if (pat_inst->num_operands() != tgt_inst->num_operands()) return nullptr;

        for (size_t op_i = 0; op_i < pat_inst->num_operands(); ++op_i) {
            auto pat_op = pat_inst->operand(op_i);
            auto tgt_op = tgt_inst->operand(op_i);
            if (!pat_op || !tgt_op) return nullptr;
            // Bind named pattern operands to the corresponding target
            // operand value. Constants are not bound — the replacement's
            // constants pass through unchanged.
            if (pat_op->has_name()) {
                operand_map[pat_op->name()] = tgt_op;
            }
        }
    }

    // Deep-copy the function
    // For the prototype, we create a new function with the same name and type,
    // then copy all blocks, replacing the matched instruction range.
    auto result = std::make_shared<ir::Function>(fn.name(), fn.function_type(), fn.linkage());

    // Copy arguments
    for (auto& arg : fn.arguments()) {
        result->add_argument(arg.type, arg.name);
    }

    // Copy attributes
    for (auto& [k, v] : fn.attributes()) {
        result->set_attribute(k, v);
    }

    // Helper: rewrite a replacement instruction's operands using the
    // binding map. Returns a fresh instruction with rewritten operands.
    auto rewrite_replacement = [&](const std::shared_ptr<ir::Instruction>& repl_inst)
        -> std::shared_ptr<ir::Instruction>
    {
        if (!repl_inst) return nullptr;

        // Patterns are AUTHORED at one width (every seed pattern is i32),
        // but must apply at whatever integer width the matched subject
        // actually is. Resolve operands FIRST, then derive the result
        // type from them where LLVM requires it to track the operand:
        // `ret` always returns its operand's type, and a same-width
        // binary op's result is its operands' type. Blindly copying the
        // pattern's own authored type here used to leave a `ret i32 %x`
        // wrapped around an i8/i64 %x whenever a pattern matched a
        // non-i32 function — invalid IR that only surfaced once the
        // (recently added) verifier or an LLVM parse actually checked it.
        // Casts, select, icmp and phi are left alone: their result width
        // is a deliberate choice of the pattern, not implied by operand 0.
        std::vector<std::shared_ptr<ir::Value>> new_operands;
        new_operands.reserve(repl_inst->num_operands());
        for (auto& op : repl_inst->operands()) {
            if (op && op->has_name()) {
                auto it2 = operand_map.find(op->name());
                if (it2 != operand_map.end()) {
                    // Rewrite to the bound target value.
                    new_operands.push_back(it2->second);
                    continue;
                }
                // Name not in the map — this is either a value defined
                // BY the replacement itself (e.g. a pattern that introduces
                // a new temporary) or a dangling reference. Pass it
                // through unchanged; validate_function will catch the
                // latter case below.
            }
            new_operands.push_back(op);
        }

        auto new_type = repl_inst->type();
        const ir::Opcode op = repl_inst->opcode();
        const bool width_tracks_operand0 =
            op == ir::Opcode::Add || op == ir::Opcode::Sub || op == ir::Opcode::Mul ||
            op == ir::Opcode::UDiv || op == ir::Opcode::SDiv ||
            op == ir::Opcode::URem || op == ir::Opcode::SRem ||
            op == ir::Opcode::And || op == ir::Opcode::Or || op == ir::Opcode::Xor ||
            op == ir::Opcode::Shl || op == ir::Opcode::LShr || op == ir::Opcode::AShr;
        if (op == ir::Opcode::Ret && !new_operands.empty() && new_operands[0] && new_operands[0]->type()) {
            new_type = new_operands[0]->type();
        } else if (width_tracks_operand0 && !new_operands.empty() &&
                   new_operands[0] && new_operands[0]->type() && new_operands[0]->type()->is_integer()) {
            new_type = new_operands[0]->type();
        }

        auto new_inst = std::make_shared<ir::Instruction>(op, new_type, repl_inst->name());
        for (auto& v : new_operands) new_inst->add_operand(v);
        for (auto& [k, v] : repl_inst->metadata()) {
            new_inst->set_metadata(k, v);
        }
        new_inst->binop_flags() = repl_inst->binop_flags();
        if (repl_inst->alignment()) new_inst->set_alignment(repl_inst->alignment().value());
        new_inst->set_volatile(repl_inst->is_volatile());
        return new_inst;
    };

    // Copy blocks
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        auto& new_block = result->add_block(block->name());

        if (block->name() == match.block_name) {
            // Replace the matched instruction range with the replacement pattern
            // Instructions before the match
            for (size_t i = 0; i < match.instruction_start && i < block->size(); ++i) {
                new_block.add_instruction(block->instruction(i));
            }

            // Replacement instructions from the pattern (with operand rewriting)
            for (auto& repl_inst : repl_entry->instructions()) {
                auto rewritten = rewrite_replacement(repl_inst);
                if (rewritten) {
                    new_block.add_instruction(rewritten);
                }
            }

            // Instructions after the match
            for (size_t i = match.instruction_end; i < block->size(); ++i) {
                new_block.add_instruction(block->instruction(i));
            }
        } else {
            // Copy all instructions unchanged
            for (auto& inst : block->instructions()) {
                new_block.add_instruction(inst);
            }
        }
    }

    // Validate the result. If the replacement left
    // dangling references (e.g. the pattern's replacement references a
    // name not bound in the source), reject the application by returning
    // nullptr. This prevents the search from accepting corrupt IR.
    if (!ir::validate_function(*result)) {
        return nullptr;
    }

    // Second, stronger check: operand counts, integer typing (the bug this
    // guards against directly), return type, phi consistency, dominance.
    // validate_function above only checks "is this name defined earlier in
    // layout order?" — it would have waved the i32/i8 mismatch straight
    // through. A pattern-authoring bug here must never reach the caller as
    // silently-invalid IR.
    if (!ir::verify_function(*result).empty()) {
        return nullptr;
    }

    return result;
}

// ── adapt_pattern ───────────────────────────────────────────────────────────

std::optional<OptimisationPattern> PatternLibrary::adapt_pattern(
    const OptimisationPattern& pattern,
    const ArchDescriptor& target_arch) const
{
    double dist = pattern.discovered_arch.distance(target_arch);

    if (dist < 0.3) {
        // Close enough — return as-is
        return pattern;
    }

    if (dist < 0.7) {
        // Minor adjustments needed
        OptimisationPattern adapted = pattern;

        // Adjust the estimated speedup — less confident on a different arch
        adapted.avg_speedup = 1.0 + (pattern.avg_speedup - 1.0) * (1.0 - dist * 0.5);

        // If the target has different cache sizes, note that memory-layer
        // assumptions may differ.  We adjust the description to reflect this.
        adapted.description += " [adapted from " + pattern.discovered_arch.name + "]";

        return adapted;
    }

    // Too different to safely adapt
    return std::nullopt;
}

// ── is_compatible ───────────────────────────────────────────────────────────

bool PatternLibrary::is_compatible(const OptimisationPattern& pattern,
                                    const ArchDescriptor& target) const {
    // A pattern is compatible if:
    // 1. It was discovered on the same architecture class (CPU/GPU)
    // 2. Or the distance is small enough to adapt

    if (pattern.discovered_arch.is_gpu != target.is_gpu) {
        // CPU pattern on GPU target or vice versa — incompatible
        // unless the pattern is explicitly tagged as cross-platform
        for (auto& tag : pattern.tags) {
            if (tag == "cross-platform" || tag == "portable") return true;
        }
        return false;
    }

    // If same vendor, always compatible
    if (pattern.discovered_arch.vendor == target.vendor) return true;

    // Different vendor but same class (e.g. AMD vs Intel CPU)
    // Check distance — if close enough, compatible
    double dist = pattern.discovered_arch.distance(target);
    return dist < 0.7;
}

// ── match_instruction_sequence ──────────────────────────────────────────────

bool PatternLibrary::match_instruction_sequence(
    const ir::BasicBlock& block,
    size_t start,
    const ir::Function& pattern_fn,
    double& confidence) const
{
    auto pat_entry = pattern_fn.entry_block();
    if (!pat_entry) {
        confidence = 0.0;
        return false;
    }

    size_t pat_len = pat_entry->size();
    if (start + pat_len > block.size()) {
        confidence = 0.0;
        return false;
    }

    // Stricter matching requirements:
    //   - All opcodes must match.
    //   - Operand arity must match.
    //   - Operand kind must match: if the pattern's operand is a
    //     ConstantInt, the target's operand must also be a ConstantInt;
    //     if the pattern's operand is a named Value, the target's
    //     operand must also be a named Value.
    //   - For ConstantInt operands in the pattern, we ALSO require the
    //     target's constant value to match (this is what makes
    //     `add_zero_elim` actually match only `add %x, 0` rather than
    //     `add %x, 5`). Constant values for arithmetic identity
    //     patterns are part of the pattern's identity, not a free
    //     variable.
    size_t opcode_matches = 0;
    for (size_t i = 0; i < pat_len; ++i) {
        auto block_inst = block.instruction(start + i);
        auto pat_inst = pat_entry->instruction(i);
        if (!block_inst || !pat_inst) {
            confidence = 0.0;
            return false;
        }
        if (block_inst->opcode() != pat_inst->opcode()) {
            confidence = 0.0;
            return false;
        }
        opcode_matches++;

        // Operand arity check.
        if (block_inst->num_operands() != pat_inst->num_operands()) {
            confidence = 0.0;
            return false;
        }

        // Operand kind + constant-value check.
        for (size_t op_i = 0; op_i < pat_inst->num_operands(); ++op_i) {
            auto pat_op = pat_inst->operand(op_i);
            auto blk_op = block_inst->operand(op_i);
            if (!pat_op || !blk_op) {
                confidence = 0.0;
                return false;
            }
            auto* pat_ci = dynamic_cast<ir::ConstantInt*>(pat_op.get());
            auto* blk_ci = dynamic_cast<ir::ConstantInt*>(blk_op.get());
            if (pat_ci && !blk_ci) {
                confidence = 0.0;
                return false;
            }
            if (!pat_ci && blk_ci) {
                confidence = 0.0;
                return false;
            }
            // For constants, require the values to match (so
            // `add_zero_elim`'s pattern operand `0` matches only
            // target operands that are ConstantInt(0)).
            if (pat_ci && blk_ci) {
                if (pat_ci->value() != blk_ci->value()) {
                    confidence = 0.0;
                    return false;
                }
            }
        }
    }

    double match_ratio = static_cast<double>(opcode_matches) / static_cast<double>(pat_len);
    confidence = match_ratio;

    return match_ratio >= 1.0;
}

// ── seed_builtin_patterns ───────────────────────────────────────────────────

void PatternLibrary::seed_builtin_patterns() {
    // ── Pattern 1: strength_reduce_mul ──────────────────────────────────────
    {
        OptimisationPattern pat;
        pat.id = "strength_reduce_mul";
        pat.name = "Strength Reduce Multiply";
        pat.description = "Replace multiplication by a power of 2 with a left shift";
        pat.source_ir = "%r = mul i32 %x, <power_of_2>";
        pat.replacement_ir = "%r = shl i32 %x, <log2>";
        pat.discovered_arch = ArchDescriptor{"x86_64", "intel", false, 0, 256};
        pat.avg_speedup = 1.3;
        pat.scope = OptimisationPattern::Scope::InstructionLevel;
        pat.tags = {"arithmetic", "simplification", "strength-reduction"};

        // Build source function
        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                ir::IntegerType::i32(),
                std::vector<std::shared_ptr<ir::Type>>{ir::IntegerType::i32(), ir::IntegerType::i32()}
            );
            auto fn = std::make_shared<ir::Function>("__strength_reduce_mul_src", fn_type);
            fn->add_argument(ir::IntegerType::i32(), "x");
            fn->add_argument(ir::IntegerType::i32(), "c");
            auto& bb = fn->add_block("entry");
            auto x_val = std::make_shared<ir::Value>(ir::IntegerType::i32(), "x");
            auto c_val = std::make_shared<ir::Value>(ir::IntegerType::i32(), "c");
            bb.add_instruction(ir::inst::make_mul(x_val, c_val, "r"));
            bb.add_instruction(ir::inst::make_ret(std::make_shared<ir::Value>(ir::IntegerType::i32(), "r")));
            pat.source_function = fn;
        }
        // Build replacement function
        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                ir::IntegerType::i32(),
                std::vector<std::shared_ptr<ir::Type>>{ir::IntegerType::i32(), ir::IntegerType::i32()}
            );
            auto fn = std::make_shared<ir::Function>("__strength_reduce_mul_rep", fn_type);
            fn->add_argument(ir::IntegerType::i32(), "x");
            fn->add_argument(ir::IntegerType::i32(), "shift");
            auto& bb = fn->add_block("entry");
            auto x_val = std::make_shared<ir::Value>(ir::IntegerType::i32(), "x");
            auto s_val = std::make_shared<ir::Value>(ir::IntegerType::i32(), "shift");
            // `shift` (log2 of the multiplier) has no binding in the source, so
            // apply() and the e-graph both refuse to instantiate this: the
            // "power of two" side condition is not expressible structurally.
            // It used to be an operand-less `shl`, which they instantiated as
            // invalid IR for every multiply.
            auto shl = std::make_shared<ir::Instruction>(ir::Opcode::Shl, ir::IntegerType::i32(), "r");
            shl->add_operand(x_val);
            shl->add_operand(s_val);
            bb.add_instruction(shl);
            bb.add_instruction(ir::inst::make_ret(std::make_shared<ir::Value>(ir::IntegerType::i32(), "r")));
            pat.replacement_function = fn;
        }

        add_pattern(pat);
    }

    // ── Pattern 2: constant_fold ────────────────────────────────────────────
    {
        OptimisationPattern pat;
        pat.id = "constant_fold";
        pat.name = "Constant Folding";
        pat.description = "Fold constant binary expressions at compile time";
        pat.source_ir = "%r = <binop> <const_a>, <const_b>";
        pat.replacement_ir = "%r = <computed_const>";
        pat.discovered_arch = ArchDescriptor{"generic", "any", false, 0, 0};
        pat.avg_speedup = 2.0;
        pat.scope = OptimisationPattern::Scope::InstructionLevel;
        pat.tags = {"arithmetic", "simplification", "constant-fold"};

        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                ir::IntegerType::i32(),
                std::vector<std::shared_ptr<ir::Type>>{}
            );
            auto fn = std::make_shared<ir::Function>("__constant_fold_src", fn_type);
            auto& bb = fn->add_block("entry");
            ir::TypeContext tmp_ctx;
            auto a = ir::ConstantInt::get(tmp_ctx, 3);
            auto b = ir::ConstantInt::get(tmp_ctx, 5);
            bb.add_instruction(ir::inst::make_add(a, b, "r"));
            bb.add_instruction(ir::inst::make_ret(std::make_shared<ir::Value>(ir::IntegerType::i32(), "r")));
            pat.source_function = fn;
        }
        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                ir::IntegerType::i32(),
                std::vector<std::shared_ptr<ir::Type>>{}
            );
            auto fn = std::make_shared<ir::Function>("__constant_fold_rep", fn_type);
            auto& bb = fn->add_block("entry");
            ir::TypeContext tmp_ctx2;
            auto result = ir::ConstantInt::get(tmp_ctx2, 8);
            bb.add_instruction(ir::inst::make_ret(result));
            pat.replacement_function = fn;
        }

        add_pattern(pat);
    }

    // ── Pattern 3: dead_code_elim ──────────────────────────────────────────
    {
        OptimisationPattern pat;
        pat.id = "dead_code_elim";
        pat.name = "Dead Code Elimination";
        pat.description = "Remove computation whose result is never used";
        pat.source_ir = "%unused = <binop> %a, %b  ; result never used";
        pat.replacement_ir = "; (removed)";
        pat.discovered_arch = ArchDescriptor{"generic", "any", false, 0, 0};
        pat.avg_speedup = 1.1;
        pat.scope = OptimisationPattern::Scope::InstructionLevel;
        pat.tags = {"simplification", "dead-code", "elimination"};

        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                std::make_shared<ir::VoidType>(),
                std::vector<std::shared_ptr<ir::Type>>{}
            );
            auto fn = std::make_shared<ir::Function>("__dead_code_elim_src", fn_type);
            auto& bb = fn->add_block("entry");
            auto a = std::make_shared<ir::Value>(ir::IntegerType::i32(), "a");
            auto b = std::make_shared<ir::Value>(ir::IntegerType::i32(), "b");
            bb.add_instruction(ir::inst::make_add(a, b, "unused"));
            bb.add_instruction(ir::inst::make_ret_void());
            pat.source_function = fn;
        }
        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                std::make_shared<ir::VoidType>(),
                std::vector<std::shared_ptr<ir::Type>>{}
            );
            auto fn = std::make_shared<ir::Function>("__dead_code_elim_rep", fn_type);
            auto& bb = fn->add_block("entry");
            bb.add_instruction(ir::inst::make_ret_void());
            pat.replacement_function = fn;
        }

        add_pattern(pat);
    }

    // ── Pattern 4: add_zero_elim ───────────────────────────────────────────
    {
        OptimisationPattern pat;
        pat.id = "add_zero_elim";
        pat.name = "Add Zero Elimination";
        pat.description = "Replace add %x, 0 with just %x";
        pat.source_ir = "%r = add %x, 0";
        pat.replacement_ir = "%r = %x  ; (identity)";
        pat.discovered_arch = ArchDescriptor{"generic", "any", false, 0, 0};
        pat.avg_speedup = 1.5;
        pat.scope = OptimisationPattern::Scope::InstructionLevel;
        pat.tags = {"arithmetic", "simplification", "identity"};

        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                ir::IntegerType::i32(),
                std::vector<std::shared_ptr<ir::Type>>{ir::IntegerType::i32()}
            );
            auto fn = std::make_shared<ir::Function>("__add_zero_elim_src", fn_type);
            fn->add_argument(ir::IntegerType::i32(), "x");
            auto& bb = fn->add_block("entry");
            auto x_val = std::make_shared<ir::Value>(ir::IntegerType::i32(), "x");
            ir::TypeContext tmp_ctx3;
            auto zero = ir::ConstantInt::get(tmp_ctx3, 0);
            bb.add_instruction(ir::inst::make_add(x_val, zero, "r"));
            bb.add_instruction(ir::inst::make_ret(std::make_shared<ir::Value>(ir::IntegerType::i32(), "r")));
            pat.source_function = fn;
        }
        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                ir::IntegerType::i32(),
                std::vector<std::shared_ptr<ir::Type>>{ir::IntegerType::i32()}
            );
            auto fn = std::make_shared<ir::Function>("__add_zero_elim_rep", fn_type);
            fn->add_argument(ir::IntegerType::i32(), "x");
            auto& bb = fn->add_block("entry");
            auto x_val = std::make_shared<ir::Value>(ir::IntegerType::i32(), "x");
            bb.add_instruction(ir::inst::make_ret(x_val));
            pat.replacement_function = fn;
        }

        add_pattern(pat);
    }

    // ── Pattern 5: mul_one_elim ────────────────────────────────────────────
    {
        OptimisationPattern pat;
        pat.id = "mul_one_elim";
        pat.name = "Multiply by One Elimination";
        pat.description = "Replace mul %x, 1 with just %x";
        pat.source_ir = "%r = mul %x, 1";
        pat.replacement_ir = "%r = %x  ; (identity)";
        pat.discovered_arch = ArchDescriptor{"generic", "any", false, 0, 0};
        pat.avg_speedup = 1.5;
        pat.scope = OptimisationPattern::Scope::InstructionLevel;
        pat.tags = {"arithmetic", "simplification", "identity"};

        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                ir::IntegerType::i32(),
                std::vector<std::shared_ptr<ir::Type>>{ir::IntegerType::i32()}
            );
            auto fn = std::make_shared<ir::Function>("__mul_one_elim_src", fn_type);
            fn->add_argument(ir::IntegerType::i32(), "x");
            auto& bb = fn->add_block("entry");
            auto x_val = std::make_shared<ir::Value>(ir::IntegerType::i32(), "x");
            ir::TypeContext tmp_ctx4;
            auto one = ir::ConstantInt::get(tmp_ctx4, 1);
            bb.add_instruction(ir::inst::make_mul(x_val, one, "r"));
            bb.add_instruction(ir::inst::make_ret(std::make_shared<ir::Value>(ir::IntegerType::i32(), "r")));
            pat.source_function = fn;
        }
        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                ir::IntegerType::i32(),
                std::vector<std::shared_ptr<ir::Type>>{ir::IntegerType::i32()}
            );
            auto fn = std::make_shared<ir::Function>("__mul_one_elim_rep", fn_type);
            fn->add_argument(ir::IntegerType::i32(), "x");
            auto& bb = fn->add_block("entry");
            auto x_val = std::make_shared<ir::Value>(ir::IntegerType::i32(), "x");
            bb.add_instruction(ir::inst::make_ret(x_val));
            pat.replacement_function = fn;
        }

        add_pattern(pat);
    }

    // ── Pattern 6: double_negation ─────────────────────────────────────────
    {
        OptimisationPattern pat;
        pat.id = "double_negation";
        pat.name = "Double Negation Elimination";
        pat.description = "Replace xor (xor %x, -1), -1 with %x";
        pat.source_ir = "%t = xor %x, -1\n%r = xor %t, -1";
        pat.replacement_ir = "%r = %x  ; (double negation cancels)";
        pat.discovered_arch = ArchDescriptor{"generic", "any", false, 0, 0};
        pat.avg_speedup = 2.0;
        pat.scope = OptimisationPattern::Scope::InstructionLevel;
        pat.tags = {"arithmetic", "simplification", "bitwise", "identity"};

        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                ir::IntegerType::i32(),
                std::vector<std::shared_ptr<ir::Type>>{ir::IntegerType::i32()}
            );
            auto fn = std::make_shared<ir::Function>("__double_negation_src", fn_type);
            fn->add_argument(ir::IntegerType::i32(), "x");
            auto& bb = fn->add_block("entry");
            auto x_val = std::make_shared<ir::Value>(ir::IntegerType::i32(), "x");
            ir::TypeContext tmp_ctx5;
            auto neg1 = ir::ConstantInt::get(tmp_ctx5, -1);
            bb.add_instruction(std::make_shared<ir::Instruction>(ir::Opcode::Xor, ir::IntegerType::i32(), "t"));
            bb.add_instruction(std::make_shared<ir::Instruction>(ir::Opcode::Xor, ir::IntegerType::i32(), "r"));
            bb.add_instruction(ir::inst::make_ret(std::make_shared<ir::Value>(ir::IntegerType::i32(), "r")));
            pat.source_function = fn;
        }
        {
            auto fn_type = std::make_shared<ir::FunctionType>(
                ir::IntegerType::i32(),
                std::vector<std::shared_ptr<ir::Type>>{ir::IntegerType::i32()}
            );
            auto fn = std::make_shared<ir::Function>("__double_negation_rep", fn_type);
            fn->add_argument(ir::IntegerType::i32(), "x");
            auto& bb = fn->add_block("entry");
            auto x_val = std::make_shared<ir::Value>(ir::IntegerType::i32(), "x");
            bb.add_instruction(ir::inst::make_ret(x_val));
            pat.replacement_function = fn;
        }

        add_pattern(pat);
    }
}

} // namespace clunk::pattern