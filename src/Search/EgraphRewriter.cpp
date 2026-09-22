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
 * Clunk EgraphRewriter — implementation.
 *
 * See include/clunk/Search/EgraphRewriter.h for the design overview.
 *
 * a self-contained e-graph + equality-saturation driver. The e-graph
 * supports hash-consing, union-find, and bounded congruence-closure
 * rebuild. The driver lowers an ir::Function into the e-graph, applies
 * pattern-library rewrites as e-graph rules, saturates, and extracts
 * the cheapest representative per e-class as a fresh ir::Function.
 *
 * Self-contained: no egg, no Boost, no new link-time deps. C++17.
 */
#include "clunk/Search/EgraphRewriter.h"

#include "clunk/IR/Verify.h"
#include "clunk/Search/StochasticSearch.h"  // for Candidate + structural_hash

#include <algorithm>
#include <cstdint>
#include <functional>
#include <sstream>
#include <utility>

namespace clunk::egraph {

// ─────────────────────────────────────────────────────────────────────────
//  ENode hashing & equality
// ─────────────────────────────────────────────────────────────────────────

// Two nodes are the same VALUE only if they also have the same TYPE. A leaf
// has no children, and a cast's children do not fix its destination, so
// (opcode, children) alone cannot tell `i8 1` from `i64 1` or
// `zext i8 %x to i32` from `zext i8 %x to i64`. Hash-consing them together
// made extraction emit one width where the other was needed — an
// operand-width mismatch LLVM rejects.
static bool same_type(const std::shared_ptr<ir::Type>& a,
                      const std::shared_ptr<ir::Type>& b) {
    if (a == b) return true;
    if (!a || !b) return false;
    return a->type_id() == b->type_id() && a->bit_width() == b->bit_width() &&
           a->to_string() == b->to_string();
}

bool ENode::operator==(const ENode& other) const {
    if (opcode != other.opcode) return false;
    if (!same_type(result_type, other.result_type)) return false;
    if (is_constant != other.is_constant) return false;
    if (is_argument != other.is_argument) return false;
    if (is_constant && constant_value != other.constant_value) return false;
    if (is_argument && argument_name != other.argument_name) return false;
    if (children != other.children) return false;
    if (predicate != other.predicate) return false;
    if (flags.nuw != other.flags.nuw) return false;
    if (flags.nsw != other.flags.nsw) return false;
    if (flags.exact != other.flags.exact) return false;
    return true;
}

// boost::hash_combine-style mixing.
static inline void hash_combine(size_t& h, size_t v) {
    h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
}

size_t ENodeHash::operator()(const ENode& n) const noexcept {
    size_t h = std::hash<int>{}(static_cast<int>(n.opcode));
    hash_combine(h, std::hash<bool>{}(n.is_constant));
    hash_combine(h, std::hash<bool>{}(n.is_argument));
    if (n.is_constant) {
        hash_combine(h, std::hash<int64_t>{}(n.constant_value));
    }
    if (n.is_argument) {
        hash_combine(h, std::hash<std::string>{}(n.argument_name));
    }
    for (EClassId c : n.children) {
        hash_combine(h, std::hash<EClassId>{}(c));
    }
    hash_combine(h, std::hash<unsigned>{}(static_cast<unsigned>(n.predicate)));
    hash_combine(h, std::hash<bool>{}(n.flags.nuw));
    hash_combine(h, std::hash<bool>{}(n.flags.nsw));
    hash_combine(h, std::hash<bool>{}(n.flags.exact));
    if (n.result_type) {  // equal nodes have equal types, so this is hash-consistent
        hash_combine(h, std::hash<int>{}(static_cast<int>(n.result_type->type_id())));
        hash_combine(h, std::hash<size_t>{}(static_cast<size_t>(n.result_type->bit_width())));
    }
    return h;
}

// ─────────────────────────────────────────────────────────────────────────
//  EGraph
// ─────────────────────────────────────────────────────────────────────────

EGraph::EGraph() = default;

EClassId EGraph::add(const ENode& node) {
    // Canonicalise children.
    ENode canon = node;
    for (EClassId& c : canon.children) {
        c = find(c);
    }

    auto it = hash_cons_.find(canon);
    if (it != hash_cons_.end()) {
        return find(it->second);
    }

    EClassId id = static_cast<EClassId>(classes_.size());
    EClass cls;
    cls.id = id;
    cls.nodes.push_back(canon);
    classes_.push_back(std::move(cls));
    parent_.push_back(id);

    // Register as parent of each child (best-effort; rebuild() does a
    // full scan too).
    for (EClassId c : canon.children) {
        EClassId cc = find(c);
        if (cc < classes_.size()) {
            classes_[cc].parents.push_back(id);
        }
    }

    hash_cons_[canon] = id;
    return id;
}

void EGraph::link(EClassId a, EClassId b) {
    // Union-find link: make `a` the parent of `b`. Does NOT trigger
    // rebuild — used by rebuild() itself.
    EClassId ra = find(a), rb = find(b);
    if (ra == rb) return;
    parent_[rb] = ra;
    // Move all e-nodes from rb into ra.
    auto& ra_nodes = classes_[ra].nodes;
    auto& rb_nodes = classes_[rb].nodes;
    for (auto& n : rb_nodes) {
        ra_nodes.push_back(std::move(n));
    }
    rb_nodes.clear();
    // Merge parent lists (best-effort).
    auto& ra_parents = classes_[ra].parents;
    auto& rb_parents = classes_[rb].parents;
    for (EClassId p : rb_parents) {
        ra_parents.push_back(p);
    }
    rb_parents.clear();
}

void EGraph::merge(EClassId a, EClassId b) {
    link(a, b);
    if (!in_rebuild_) {
        rebuild();
    }
}

void EGraph::rebuild() {
    if (in_rebuild_) return;
    in_rebuild_ = true;

    size_t merges_this_rebuild = 0;
    bool changed = true;
    while (changed && merges_this_rebuild < kMaxRebuildMerges) {
        changed = false;

        // 1. Canonicalise all e-node children in place.
        for (auto& cls : classes_) {
            for (auto& node : cls.nodes) {
                for (EClassId& c : node.children) {
                    c = find(c);
                }
            }
        }

        // 2. Rebuild the hash-cons; merge congruent e-classes.
        hash_cons_.clear();
        for (EClassId cid = 0; cid < classes_.size(); ++cid) {
            if (find(cid) != cid) continue;  // skip non-canonical classes
            auto& cls = classes_[cid];
            for (ENode& node : cls.nodes) {
                auto it = hash_cons_.find(node);
                if (it == hash_cons_.end()) {
                    hash_cons_[node] = cid;
                } else if (it->second != cid) {
                    // Congruent! Merge cid into it->second.
                    link(it->second, cid);
                    ++merges_this_rebuild;
                    changed = true;
                    if (merges_this_rebuild >= kMaxRebuildMerges) break;
                }
            }
            if (merges_this_rebuild >= kMaxRebuildMerges) break;
        }
    }

    in_rebuild_ = false;
}

EClassId EGraph::find(EClassId id) const {
    if (id >= parent_.size()) return id;
    EClassId root = id;
    while (parent_[root] != root) root = parent_[root];
    // Path compression.
    EClassId cur = id;
    while (parent_[cur] != root) {
        EClassId next = parent_[cur];
        parent_[cur] = root;
        cur = next;
    }
    return root;
}

// ─────────────────────────────────────────────────────────────────────────
//  Lowering helpers
// ─────────────────────────────────────────────────────────────────────────

namespace {

bool is_function_arg(const ir::Function& fn, const std::string& name) {
    for (auto& arg : fn.arguments()) {
        if (arg.name == name) return true;
    }
    return false;
}

bool as_constant_int(const std::shared_ptr<ir::Value>& v, int64_t& out) {
    auto ci = std::dynamic_pointer_cast<ir::ConstantInt>(v);
    if (!ci) return false;
    out = ci->value();
    return true;
}

// The e-graph hash-conses by (opcode, children): that is only sound for
// pure computations whose value depends on nothing but their operands.
// Loads (memory state), calls (effects), allocas (identity), phis (their
// value depends on the incoming EDGE, which is not part of a node) and
// friends must stay opaque: two structurally identical loads are NOT the
// same value, and a phi cannot be rebuilt without its incoming blocks.
bool is_pure_value_op(ir::Opcode op) {
    switch (op) {
        case ir::Opcode::Add: case ir::Opcode::Sub: case ir::Opcode::Mul:
        case ir::Opcode::UDiv: case ir::Opcode::SDiv: case ir::Opcode::URem: case ir::Opcode::SRem:
        case ir::Opcode::And: case ir::Opcode::Or: case ir::Opcode::Xor:
        case ir::Opcode::Shl: case ir::Opcode::LShr: case ir::Opcode::AShr:
        case ir::Opcode::ICmp: case ir::Opcode::Select:
        case ir::Opcode::ZExt: case ir::Opcode::SExt: case ir::Opcode::Trunc:
            return true;
        default:
            return false;
    }
}

ir::CmpPredicate predicate_from_inst(const ir::Instruction& inst) {
    auto& md = inst.metadata();
    auto it = md.find("pred");
    if (it == md.end()) return ir::CmpPredicate::EQ;
    try {
        unsigned val = static_cast<unsigned>(std::stoul(it->second));
        return static_cast<ir::CmpPredicate>(val);
    } catch (...) {
        return ir::CmpPredicate::EQ;
    }
}

} // namespace

LoweringResult lower_to_egraph(const ir::Function& fn) {
    LoweringResult result;
    result.egraph = std::make_unique<EGraph>();
    result.source_function = &fn;

    // Pre-populate function arguments as argument e-nodes.
    for (auto& arg : fn.arguments()) {
        if (arg.name.empty()) continue;
        ENode node;
        node.is_argument = true;
        node.argument_name = arg.name;
        node.result_type = arg.type;
        EClassId id = result.egraph->add(node);
        result.value_to_class[arg.name] = id;
    }

    // Walk blocks in order; lower each non-terminator instruction.
    std::optional<EClassId> return_class;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst) continue;

            // Ret: record the returned value's e-class.
            if (inst->opcode() == ir::Opcode::Ret) {
                if (inst->num_operands() == 1) {
                    auto op = inst->operand(0);
                    if (op && op->has_name()) {
                        auto it = result.value_to_class.find(op->name());
                        if (it != result.value_to_class.end()) {
                            return_class = it->second;
                        }
                    } else if (op) {
                        int64_t cv = 0;
                        if (as_constant_int(op, cv)) {
                            ENode cnode;
                            cnode.is_constant = true;
                            cnode.constant_value = cv;
                            cnode.result_type = op->type();
                            EClassId cid = result.egraph->add(cnode);
                            return_class = cid;
                        }
                    }
                }
                continue;
            }

            // Skip terminators (Br, Switch, etc.) — they don't produce values.
            if (inst->is_terminator()) continue;

            // Skip instructions with no name (no result) — e.g. Store.
            if (!inst->has_name()) continue;

            // Anything that is not a pure value computation stays opaque:
            // extraction copies it verbatim and later uses see it as a leaf.
            if (!is_pure_value_op(inst->opcode())) continue;

            // Resolve operands to e-class ids.
            ENode node;
            node.opcode = inst->opcode();
            node.flags = inst->binop_flags();
            node.result_type = inst->type();
            if (inst->is_cmp()) {
                node.predicate = predicate_from_inst(*inst);
            }

            bool ok = true;
            for (auto& op : inst->operands()) {
                if (!op) { ok = false; break; }
                int64_t cv = 0;
                if (as_constant_int(op, cv)) {
                    ENode cnode;
                    cnode.is_constant = true;
                    cnode.constant_value = cv;
                    cnode.result_type = op->type();
                    EClassId cid = result.egraph->add(cnode);
                    node.children.push_back(cid);
                } else if (op->has_name()) {
                    auto it = result.value_to_class.find(op->name());
                    if (it == result.value_to_class.end()) {
                        // Unknown named value — treat as opaque argument
                        // leaf so the structure is preserved.
                        ENode onode;
                        onode.is_argument = true;
                        onode.argument_name = op->name();
                        onode.result_type = op->type();
                        // Hash-consing already gives every reference to this
                        // name the same leaf class. It must NOT be recorded in
                        // value_to_class: that map means "lowered instruction",
                        // and extraction would then skip emitting the (opaque)
                        // instruction that defines this value.
                        EClassId cid = result.egraph->add(onode);
                        node.children.push_back(cid);
                    } else {
                        node.children.push_back(it->second);
                    }
                } else {
                    // Unnamed non-constant — opaque. Skip lowering.
                    ok = false;
                    break;
                }
            }

            if (!ok) continue;

            EClassId id = result.egraph->add(node);
            result.value_to_class[inst->name()] = id;
        }
    }

    result.return_class = return_class;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────
//  Pattern matching (e-matching, simplified)
// ─────────────────────────────────────────────────────────────────────────

namespace {

struct MatchBinding {
    std::unordered_map<std::string, EClassId> args;
    std::unordered_map<std::string, EClassId> inst_results;
};

std::unordered_map<std::string, ir::Instruction*>
build_name_map(const ir::Function& fn) {
    std::unordered_map<std::string, ir::Instruction*> map;
    if (!fn.entry_block()) return map;
    for (auto& inst : fn.entry_block()->instructions()) {
        if (inst && inst->has_name()) {
            map[inst->name()] = inst.get();
        }
    }
    return map;
}

bool opcodes_compatible(const ir::Instruction& inst, const ENode& node) {
    if (inst.opcode() != node.opcode) return false;
    // Patterns are written at one width (i32); applying one to a value of
    // another width would splice pattern-typed nodes into the subject.
    if (!same_type(inst.type(), node.result_type)) return false;
    if (inst.is_cmp() && node.predicate != predicate_from_inst(inst)) return false;
    if (inst.is_binary_op()) {
        if (inst.binop_flags().nuw != node.flags.nuw) return false;
        if (inst.binop_flags().nsw != node.flags.nsw) return false;
        if (inst.binop_flags().exact != node.flags.exact) return false;
    }
    return true;
}

bool match_value(const std::shared_ptr<ir::Value>& value,
                 EClassId cls,
                 const ir::Function& src_fn,
                 const std::unordered_map<std::string, ir::Instruction*>& src_names,
                 MatchBinding& binding,
                 const EGraph& eg);

bool match_instruction(ir::Instruction& inst,
                       const ENode& node,
                       EClassId cls,
                       const ir::Function& src_fn,
                       const std::unordered_map<std::string, ir::Instruction*>& src_names,
                       MatchBinding& binding,
                       const EGraph& eg) {
    if (!opcodes_compatible(inst, node)) return false;
    if (inst.num_operands() != node.children.size()) return false;

    if (inst.has_name()) {
        auto it = binding.inst_results.find(inst.name());
        if (it != binding.inst_results.end()) {
            if (eg.find(it->second) != eg.find(cls)) return false;
        } else {
            binding.inst_results[inst.name()] = cls;
        }
    }

    for (size_t i = 0; i < inst.num_operands(); ++i) {
        if (!match_value(inst.operand(i), node.children[i],
                         src_fn, src_names, binding, eg)) {
            return false;
        }
    }
    return true;
}

bool match_value(const std::shared_ptr<ir::Value>& value,
                 EClassId cls,
                 const ir::Function& src_fn,
                 const std::unordered_map<std::string, ir::Instruction*>& src_names,
                 MatchBinding& binding,
                 const EGraph& eg) {
    if (!value) return false;
    EClassId canon = eg.find(cls);

    // ConstantInt operand in the pattern.
    int64_t cv = 0;
    if (as_constant_int(value, cv)) {
        for (auto& node : eg.classes()[canon].nodes) {
            if (node.is_constant && node.constant_value == cv &&
                same_type(node.result_type, value->type())) return true;
        }
        return false;
    }

    if (value->has_name()) {
        const std::string& name = value->name();

        if (is_function_arg(src_fn, name)) {
            auto it = binding.args.find(name);
            if (it != binding.args.end()) {
                return eg.find(it->second) == canon;
            }
            binding.args[name] = cls;
            return true;
        }

        auto it = src_names.find(name);
        if (it != src_names.end() && it->second) {
            auto bound = binding.inst_results.find(name);
            if (bound != binding.inst_results.end()) {
                return eg.find(bound->second) == canon;
            }
            // Try each e-node in the e-class.
            for (auto& node : eg.classes()[canon].nodes) {
                MatchBinding trial = binding;
                if (match_instruction(*it->second, node, canon, src_fn, src_names,
                                      trial, eg)) {
                    binding = trial;
                    return true;
                }
            }
            return false;
        }

        return false;
    }

    return false;
}

std::optional<EClassId> resolve_value(
    const std::shared_ptr<ir::Value>& value,
    const MatchBinding& binding,
    EGraph& eg) {
    if (!value) return std::nullopt;
    int64_t cv = 0;
    if (as_constant_int(value, cv)) {
        ENode cnode;
        cnode.is_constant = true;
        cnode.constant_value = cv;
        cnode.result_type = value->type();
        return eg.add(cnode);
    }
    if (value->has_name()) {
        const std::string& name = value->name();
        auto a = binding.args.find(name);
        if (a != binding.args.end()) return eg.find(a->second);
        auto r = binding.inst_results.find(name);
        if (r != binding.inst_results.end()) return eg.find(r->second);
    }
    return std::nullopt;
}

std::optional<EClassId> build_replacement(
    const ir::Function& repl_fn,
    MatchBinding& binding,
    EGraph& eg) {
    auto repl_names = build_name_map(repl_fn);
    if (!repl_fn.entry_block()) return std::nullopt;

    for (auto& inst : repl_fn.entry_block()->instructions()) {
        if (!inst) continue;
        if (inst->opcode() == ir::Opcode::Ret) {
            if (inst->num_operands() != 1) return std::nullopt;
            return resolve_value(inst->operand(0), binding, eg);
        }
        if (inst->is_terminator()) continue;
        if (!inst->has_name()) continue;

        // A replacement instruction with too few operands is a malformed
        // pattern (the seed strength_reduce_mul used to build an operand-less
        // `shl`). Merging it into the subject's e-class would let extraction
        // emit `%r = shl i32` — invalid IR — so refuse the whole rewrite.
        if (inst->num_operands() < ir::verify_detail::min_operands(inst->opcode()))
            return std::nullopt;

        ENode node;
        node.opcode = inst->opcode();
        node.flags = inst->binop_flags();
        node.result_type = inst->type();
        if (inst->is_cmp()) node.predicate = predicate_from_inst(*inst);

        bool ok = true;
        for (auto& op : inst->operands()) {
            auto cid = resolve_value(op, binding, eg);
            if (!cid) { ok = false; break; }
            node.children.push_back(*cid);
        }
        if (!ok) continue;

        EClassId id = eg.add(node);
        binding.inst_results[inst->name()] = id;
    }

    return std::nullopt;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────
//  EgraphRewriter
// ─────────────────────────────────────────────────────────────────────────

EgraphRewriter::EgraphRewriter(const pattern::PatternLibrary* lib,
                               const pattern::ArchDescriptor& arch)
    : lib_(lib), arch_(arch), egraph_(std::make_unique<EGraph>()) {}

size_t EgraphRewriter::add_pattern_rules() {
    if (!lib_ || !egraph_) return 0;

    size_t rules_added = 0;
    for (auto& [pid, pat] : lib_->patterns()) {
        if (!pat.source_function || !pat.replacement_function) continue;
        if (pat.source_function->entry_block() == nullptr) continue;

        auto src_names = build_name_map(*pat.source_function);

        // The pattern's "root" is the operand of the source's ret.
        std::shared_ptr<ir::Value> src_root;
        for (auto& inst : pat.source_function->entry_block()->instructions()) {
            if (inst && inst->opcode() == ir::Opcode::Ret && inst->num_operands() == 1) {
                src_root = inst->operand(0);
                break;
            }
        }
        if (!src_root) continue;

        // Snapshot num_classes() to avoid iterating over newly added
        // classes (they will be covered in the next saturation iteration).
        size_t n = egraph_->num_classes();
        for (EClassId cid = 0; cid < n; ++cid) {
            EClassId canon = egraph_->find(cid);
            if (canon != cid) continue;  // already merged away

            MatchBinding binding;
            if (!match_value(src_root, canon, *pat.source_function,
                             src_names, binding, *egraph_)) {
                continue;
            }

            auto repl_root = build_replacement(*pat.replacement_function,
                                               binding, *egraph_);
            if (!repl_root) continue;

            EClassId repl_canon = egraph_->find(*repl_root);
            if (repl_canon != canon) {
                egraph_->merge(canon, repl_canon);
                ++rewrites_applied_;
                ++rules_added;
            }
        }
    }
    return rules_added;
}

size_t EgraphRewriter::saturate(size_t max_iterations) {
    size_t total_rewrites = 0;
    for (size_t iter = 0; iter < max_iterations; ++iter) {
        size_t n = add_pattern_rules();
        total_rewrites += n;
        if (n == 0) break;  // fixpoint
    }
    return total_rewrites;
}

// ─────────────────────────────────────────────────────────────────────────
//  Extraction: e-graph -> ir::Function (cost-based)
// ─────────────────────────────────────────────────────────────────────────

namespace {

double node_cost(const ENode& node, const evaluator::CostModel& cm) {
    if (node.is_constant || node.is_argument) return 0.0;
    return cm.cost(node.opcode).latency_cycles;
}

std::shared_ptr<ir::Value> resolve_operand(
    const std::shared_ptr<ir::Value>& op,
    const std::unordered_map<std::string, std::shared_ptr<ir::Value>>& binding) {
    if (!op) return nullptr;
    int64_t cv = 0;
    if (as_constant_int(op, cv)) return op;  // constants are immutable
    if (op->has_name()) {
        auto it = binding.find(op->name());
        if (it != binding.end()) return it->second;
    }
    return op;
}

std::shared_ptr<ir::Instruction> build_instruction(
    const ENode& node,
    const std::vector<std::shared_ptr<ir::Value>>& resolved_children,
    const std::string& name) {
    // Prefer the type recorded at lowering time — it's the ONLY reliable
    // source of truth. Guessing from operand types (the old behaviour)
    // is simply wrong for any opcode whose result type isn't "same as
    // its first operand": Alloca's result is a pointer to the allocated
    // type (unrelated to any operand), Load's result is the pointee
    // type (not the pointer operand's type), GetElementPtr's result is
    // an indexed pointer type, Select's first operand is an i1
    // condition, casts change the type by definition, etc. Building an
    // Instruction with the wrong Type subclass here is a memory-safety
    // bug, not just a correctness one: e.g. Instruction::to_string()
    // static_casts an Alloca's type to PointerType* unconditionally, so
    // an Alloca stamped with IntegerType reads a PointerType::pointee_
    // shared_ptr out of memory that was never a PointerType, corrupting
    // an unrelated shared_ptr control block on the next copy.
    std::shared_ptr<ir::Type> ty = node.result_type;
    if (!ty) {
        // Defensive fallback for the (should-not-happen) case of a node
        // with no recorded type — better than leaving type_ null.
        ty = resolved_children.empty() ? ir::IntegerType::i32()
                                        : resolved_children[0]->type();
    }
    auto inst = std::make_shared<ir::Instruction>(node.opcode, ty, name);
    for (auto& v : resolved_children) {
        inst->add_operand(v);
    }
    inst->binop_flags() = node.flags;
    if (node.opcode == ir::Opcode::ICmp || node.opcode == ir::Opcode::FCmp) {
        inst->set_metadata("pred", std::to_string(static_cast<unsigned>(node.predicate)));
    }
    return inst;
}

// Extraction context — bundles all the state needed by the recursive
// materialise() helper so we don't have to thread 8 arguments through.
struct ExtractCtx {
    const EGraph& eg;
    const evaluator::CostModel& cm;
    const std::unordered_map<EClassId, ENode>& best_node;
    std::unordered_map<EClassId, std::shared_ptr<ir::Value>> memo;
    // Block each memoised INSTRUCTION was emitted into (nullptr for leaves).
    // A value may only be reused where its definition is guaranteed to
    // dominate: the same block, or the entry block (which dominates all).
    std::unordered_map<EClassId, const ir::BasicBlock*> memo_block;
    std::unordered_map<std::string, std::shared_ptr<ir::Value>> binding;
    ir::BasicBlock* current_bb = nullptr;
    const ir::BasicBlock* entry_bb = nullptr;
    size_t counter = 0;
};

std::shared_ptr<ir::Value> materialise(EClassId cls, ExtractCtx& ctx) {
    EClassId canon = ctx.eg.find(cls);
    auto m = ctx.memo.find(canon);
    if (m != ctx.memo.end()) {
        const ir::BasicBlock* def_bb = ctx.memo_block[canon];
        if (!def_bb || def_bb == ctx.current_bb || def_bb == ctx.entry_bb) return m->second;
        // Defined in a block that need not dominate this one (e.g. the same
        // expression computed in both arms of a branch): rebuild it here.
    }

    auto bn = ctx.best_node.find(canon);
    if (bn == ctx.best_node.end()) return nullptr;
    const ENode& node = bn->second;

    std::shared_ptr<ir::Value> result;

    if (node.is_argument) {
        auto it = ctx.binding.find(node.argument_name);
        if (it != ctx.binding.end()) {
            result = it->second;
        } else {
            std::shared_ptr<ir::Type> ty =
                node.result_type ? node.result_type : ir::IntegerType::i32();
            result = std::make_shared<ir::Value>(ty, node.argument_name);
            ctx.binding[node.argument_name] = result;
        }
    } else if (node.is_constant) {
        auto int_ty = std::dynamic_pointer_cast<ir::IntegerType>(node.result_type);
        if (!int_ty) int_ty = ir::IntegerType::i32();
        result = std::make_shared<ir::ConstantInt>(int_ty, node.constant_value);
    } else {
        std::vector<std::shared_ptr<ir::Value>> children;
        for (EClassId c : node.children) {
            auto v = materialise(c, ctx);
            if (!v) return nullptr;
            children.push_back(v);
        }
        std::string name = "_eg_" + std::to_string(ctx.counter++);
        auto inst = build_instruction(node, children, name);
        if (ctx.current_bb) {
            ctx.current_bb->add_instruction(inst);
        }
        result = inst;
        ctx.memo[canon] = result;
        ctx.memo_block[canon] = ctx.current_bb;
        return result;
    }

    ctx.memo[canon] = result;
    ctx.memo_block[canon] = nullptr;  // argument / constant leaf
    return result;
}

} // namespace

std::shared_ptr<ir::Function> EgraphRewriter::extract(
    const LoweringResult& lowering,
    const evaluator::EvaluationEngine& eval) const {

    if (!lowering.source_function || !egraph_) return nullptr;
    const ir::Function& src = *lowering.source_function;

    // ── 1. Cost-based extraction: cheapest e-node per e-class ──
    std::unordered_map<EClassId, double> best_cost;
    std::unordered_map<EClassId, ENode> best_node;
    const evaluator::CostModel& cm = eval.cost_model();

    bool changed = true;
    int passes = 0;
    while (changed && passes < 16) {
        changed = false;
        ++passes;
        for (auto& cls : egraph_->classes()) {
            EClassId canon = egraph_->find(cls.id);
            for (auto& node : cls.nodes) {
                double c = node_cost(node, cm);
                bool ok = true;
                for (EClassId child : node.children) {
                    EClassId cc = egraph_->find(child);
                    auto it = best_cost.find(cc);
                    if (it == best_cost.end()) { ok = false; break; }
                    c += it->second;
                }
                if (!ok) continue;
                auto it = best_cost.find(canon);
                if (it == best_cost.end() || c < it->second) {
                    best_cost[canon] = c;
                    best_node[canon] = node;
                    changed = true;
                }
            }
        }
    }

    // ── 2. Re-materialise IR ──
    auto out = std::make_shared<ir::Function>(src.name(), src.function_type(), src.linkage());
    for (auto& arg : src.arguments()) {
        out->add_argument(arg.type, arg.name);
    }
    for (auto& [k, v] : src.attributes()) {
        out->set_attribute(k, v);
    }

    ExtractCtx ctx{*egraph_, cm, best_node, {}, {}, {}, nullptr, nullptr, 0};
    for (auto& arg : out->arguments()) {
        ctx.binding[arg.name] = std::make_shared<ir::Value>(arg.type, arg.name);
    }

    // Instructions we copy rather than rebuild (terminators, phis, memory,
    // calls, ...). Copies preserve EVERYTHING — including alignment and
    // volatile, which the old hand-rolled copies silently dropped.
    std::vector<std::shared_ptr<ir::Instruction>> copies;
    auto copy_verbatim = [&](ir::BasicBlock& bb, const std::shared_ptr<ir::Instruction>& inst) {
        auto copy = std::make_shared<ir::Instruction>(inst->opcode(), inst->type(), inst->name());
        for (auto& op : inst->operands()) {
            copy->add_operand(resolve_operand(op, ctx.binding));
        }
        for (auto& [k, v] : inst->metadata()) {
            copy->set_metadata(k, v);
        }
        copy->binop_flags() = inst->binop_flags();
        if (inst->alignment()) copy->set_alignment(inst->alignment().value());
        copy->set_volatile(inst->is_volatile());
        bb.add_instruction(copy);
        if (inst->has_name()) ctx.binding[inst->name()] = copy;
        copies.push_back(copy);
    };

    for (auto& block : src.blocks()) {
        if (!block) continue;
        auto& new_bb = out->add_block(block->name());
        if (!ctx.entry_bb) ctx.entry_bb = &new_bb;
        ctx.current_bb = &new_bb;

        for (auto& inst : block->instructions()) {
            if (!inst) continue;

            // Ret.
            if (inst->opcode() == ir::Opcode::Ret) {
                if (inst->num_operands() == 1) {
                    auto op = inst->operand(0);
                    std::shared_ptr<ir::Value> resolved;
                    int64_t cv = 0;
                    if (as_constant_int(op, cv)) {
                        resolved = op;
                    } else if (op && op->has_name()) {
                        auto it = ctx.binding.find(op->name());
                        if (it != ctx.binding.end()) {
                            resolved = it->second;
                        } else {
                            auto lc = lowering.value_to_class.find(op->name());
                            if (lc != lowering.value_to_class.end()) {
                                resolved = materialise(lc->second, ctx);
                            }
                        }
                    }
                    if (!resolved) resolved = op;
                    new_bb.add_instruction(ir::inst::make_ret(resolved));
                } else {
                    new_bb.add_instruction(ir::inst::make_ret_void());
                }
                continue;
            }

            // Other terminators, and anything that was not lowered (phis,
            // memory, calls, unnamed effects): copy verbatim.
            auto lc = lowering.value_to_class.find(inst->name());
            if (inst->is_terminator() || lc == lowering.value_to_class.end()) {
                copy_verbatim(new_bb, inst);
                continue;
            }

            // Lowered: materialise the cheapest e-node for this e-class.
            auto v = materialise(lc->second, ctx);
            if (v) {
                ctx.binding[inst->name()] = v;
            } else {
                copy_verbatim(new_bb, inst);  // fallback
            }
        }

        ctx.current_bb = nullptr;
    }

    // A copy may name a value that extraction only (re)defined LATER under a
    // new `_eg_N` name — the loop back-edge of a phi is the classic case.
    // Now that every original name is bound, resolve the copies again.
    for (auto& c : copies) {
        for (size_t i = 0; i < c->num_operands(); ++i) {
            c->set_operand(i, resolve_operand(c->operand(i), ctx.binding));
        }
    }

    return out;
}

} // namespace clunk::egraph

// ─────────────────────────────────────────────────────────────────────────
//  Driver: clunk::search::egraph_rewrite
// ─────────────────────────────────────────────────────────────────────────

namespace clunk::search {

std::optional<Candidate> egraph_rewrite(
    const ir::Function& fn,
    const pattern::PatternLibrary& lib,
    const pattern::ArchDescriptor& arch,
    const evaluator::EvaluationEngine& eval) {

    // 1. Lower the function into an e-graph.
    auto lower = clunk::egraph::lower_to_egraph(fn);
    if (!lower.egraph) return std::nullopt;

    // 2. Construct the rewriter and hand over the e-graph.
    clunk::egraph::EgraphRewriter rw(&lib, arch);
    rw.take_egraph(std::move(lower.egraph));

    // 3. Add pattern rules and saturate.
    rw.add_pattern_rules();
    rw.saturate(30);

    // 4. Extract the cheapest candidate. We pass `lower` (whose egraph
    // field is now null because we moved it into the rewriter). extract()
    // uses the rewriter's own egraph_ for traversal; it only reads
    // lowering.value_to_class, lowering.return_class, and
    // lowering.source_function from the LoweringResult.
    std::shared_ptr<ir::Function> candidate = rw.extract(lower, eval);
    if (!candidate) return std::nullopt;

    // Last line of defence: candidates from here are flagged sound and skip
    // the prover, so nothing malformed may ever leave this function.
    if (!ir::verify_function(*candidate).empty()) return std::nullopt;

    // 5. Score and compare.
    double ratio = eval.score_candidate(fn, *candidate);
    if (ratio <= 1.0) return std::nullopt;

    // 6. Build the Candidate. Fix D: marked sound=true by the caller
    // (egraph_phase) because all pattern-library rewrites are sound-by-
    // construction (algebraic identities / SMT-proven miner patterns).
    Candidate c;
    c.function = candidate;
    c.score = eval.analyse(*candidate).score;
    c.iteration_found = 0;
    c.description = "egraph_rewrite";
    c.sound = false;  // caller (egraph_phase) overrides to true
    c.structural_hash = search::StochasticSearch::structural_hash(*candidate);
    return c;
}

} // namespace clunk::search
