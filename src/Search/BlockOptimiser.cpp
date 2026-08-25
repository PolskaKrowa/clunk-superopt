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
 * Clunk Block-Level Optimiser — implementation.
 * See include/clunk/Search/BlockOptimiser.h for the contract.
 *
 * Strategy:
 *   optimize(fn):
 *     work = deep_copy(fn)
 *     status = optimise_node(work, anchor=fn, bb_name, 0, work.size())
 *     if status == SYNTHETIC: return work
 *     else: return nullptr
 *
 *   optimise_node(work, anchor, bb, start, end):
 *     (See the header for the recursive algorithm.)
 *
 *   try_optimise_range(work, anchor, bb, start, end):
 *     1. Collect range instructions; bail if any is not an integer binop.
 *     2. Compute inputs (live-in) and the single output (live-out).
 *     3. Enumerate candidate replacement sequences of depth 1..max_depth
 *        (9 binops × (n_inputs + 16 consts + prior)² per step).
 *     4. For each candidate:
 *        a. Build a candidate function: deep_copy(work), replace the range.
 *        b. Validate SSA.
 *        c. Test-vector pre-filter (sound).
 *        d. Score; skip if not cheaper than work.
 *        e. SMT-verify against anchor.
 *        f. If equivalent and cheaper → adopt (work = candidate), return true.
 */

#include "clunk/Search/BlockOptimiser.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "clunk/IR/Clone.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Search/StochasticSearch.h"  // for structural_hash reuse

namespace clunk::search {

namespace {

// ── Opcode pool ────────────────────────────────────────────────────────────
// The set of integer binops the synthesiser will consider. Deliberately
// small — every opcode here is one the SMT verifier can soundly encode.
constexpr std::array<ir::Opcode, 9> kBinops = {
    ir::Opcode::Add, ir::Opcode::Sub, ir::Opcode::Mul,
    ir::Opcode::And, ir::Opcode::Or,  ir::Opcode::Xor,
    ir::Opcode::Shl, ir::Opcode::LShr, ir::Opcode::AShr,
};

// ── Constant pool ──────────────────────────────────────────────────────────
// A small but "interesting" set of constants (Massalin §3.2).
constexpr std::array<int64_t, 16> kConstants = {
    0, 1, -1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 255, 256,
};

// ── Test-vector probe set ──────────────────────────────────────────────────
constexpr std::array<int64_t, 16> kProbeValues = {
    0, 1, -1, 2, -2, 3, 7, 8, 15, 16, 17, 127, 128, 255, 256, 1023,
};

// Is `fn` a candidate this optimiser can attempt? Returns false for
// anything outside the scope documented in the header.
bool is_in_scope(const ir::Function& fn) {
    if (fn.blocks().size() != 1) return false;
    if (fn.argument_count() > 4) return false;  // keep enumeration tractable
    auto ret_ty = fn.return_type();
    if (!ret_ty || !ret_ty->is_integer()) return false;
    for (const auto& arg : fn.arguments()) {
        if (!arg.type || !arg.type->is_integer()) return false;
    }
    return true;
}

// Look up the integer bit-width used by the function (return type and
// all args assumed to share the same width — verified by is_in_scope).
unsigned common_int_width(const ir::Function& fn) {
    auto ret_ty = fn.return_type();
    auto* it = dynamic_cast<const ir::IntegerType*>(ret_ty.get());
    return it ? it->bits() : 32u;
}

// One operand slot in the candidate's operand pool. Either an input
// value by index, a small constant by index, or a reference to a prior
// instruction's result by index (0-based, relative to the candidate
// sequence).
struct PoolRef {
    enum Kind : uint8_t { Input, Const, Prior };
    Kind kind;
    uint16_t index;
};

struct HoleStep {
    ir::Opcode op;
    PoolRef lhs;
    PoolRef rhs;
};

// Resolve a PoolRef to a concrete shared_ptr<Value> given the candidate's
// environment.
std::shared_ptr<ir::Value> resolve_ref(
    const PoolRef& ref,
    const std::vector<std::shared_ptr<ir::Value>>& inputs,
    const std::vector<std::shared_ptr<ir::ConstantInt>>& consts,
    const std::vector<std::shared_ptr<ir::Value>>& prior_results) {
    switch (ref.kind) {
    case PoolRef::Input: return inputs.at(ref.index);
    case PoolRef::Const: return consts.at(ref.index);
    case PoolRef::Prior: return prior_results.at(ref.index);
    }
    return nullptr;
}

// Is `op` a commutative integer binop? (Used to prune symmetric duplicates.)
bool is_commutative(ir::Opcode op) {
    return op == ir::Opcode::Add || op == ir::Opcode::Mul ||
           op == ir::Opcode::And || op == ir::Opcode::Or ||
           op == ir::Opcode::Xor;
}

} // namespace

// ── Constructor ────────────────────────────────────────────────────────────

BlockOptimiser::BlockOptimiser(evaluator::EvaluationEngine* engine,
                                 const BlockOptimiserConfig& config)
    : engine_(engine), config_(config), smt_(SMTConfig{}) {
    if (config_.smt_timeout_ms > 0) {
        smt_.config().timeout_ms = config_.smt_timeout_ms;
    }
}

bool BlockOptimiser::time_up() const {
    if (!has_deadline_) return false;
    return std::chrono::steady_clock::now() >= deadline_;
}

// ── optimize (public entry point) ──────────────────────────────────────────

std::shared_ptr<ir::Function> BlockOptimiser::optimize(const ir::Function& fn,
                                                         bool* proven) {
    if (proven) *proven = false;
    ++stats_.functions_seen;

    if (!is_in_scope(fn)) {
        ++stats_.functions_skipped;
        return nullptr;
    }
    if (fn.instruction_count() > config_.max_function_instructions) {
        ++stats_.functions_skipped;
        return nullptr;
    }
    if (!SMTVerifier::is_z3_available()) {
        // Without a prover we cannot soundly adopt any rewrite (unless
        // the user opted into trust_unverified).
        if (!config_.trust_unverified) {
            ++stats_.functions_skipped;
            return nullptr;
        }
    }

    // Set the wall-clock deadline.
    if (config_.time_budget_seconds > 0.0) {
        deadline_ = std::chrono::steady_clock::now() +
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(config_.time_budget_seconds));
        has_deadline_ = true;
    } else {
        has_deadline_ = false;
    }

    // Work on a deep copy; `fn` is the immutable soundness anchor.
    auto work = ir::deep_copy_function(fn);
    if (!work || work->blocks().empty()) {
        ++stats_.functions_skipped;
        return nullptr;
    }

    const std::string bb_name = work->entry_block()->name();
    const size_t bb_size = work->entry_block()->size();

    // The last instruction of the block is the terminator (ret, br, etc.).
    // The block optimiser only replaces integer binop sequences, so the
    // terminator must be EXCLUDED from the optimisable range — otherwise
    // the root call always bails (the terminator is not a binary op) and
    // no whole-block optimisation is ever attempted.
    const size_t optimisable = (bb_size > 0) ? bb_size - 1 : 0;
    if (optimisable < config_.min_block_size) {
        // Too few non-terminator instructions to even try.
        ++stats_.functions_skipped;
        return nullptr;
    }

    auto t0 = std::chrono::steady_clock::now();

    NodeResult result = optimise_node(*work, fn, bb_name, 0, optimisable);

    stats_.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    if (result.status == Status::Synthetic) {
        if (proven) *proven = true;
        ++stats_.proven;
        return work;
    }
    return nullptr;
}

// ── optimise_node (recursive divide-and-conquer) ──────────────────────────

BlockOptimiser::NodeResult BlockOptimiser::optimise_node(
    ir::Function& work, const ir::Function& anchor,
    const std::string& bb_name, size_t start, size_t end)
{
    const size_t size = end - start;
    if (size < config_.min_block_size) {
        // Too small to even try — declare natural without incrementing
        // blocks_tried (we didn't actually attempt optimisation).
        return {Status::Natural, 0};
    }

    if (time_up()) return {Status::Natural, 0};

    ++stats_.blocks_tried;

    // Step 1: try to optimise this whole range. If it improves, we're done.
    int delta = try_optimise_range(work, anchor, bb_name, start, end);
    if (delta != 0) {
        ++stats_.blocks_optimised;
        return {Status::Synthetic, delta};
    }

    // Step 2: if we're at the minimum block size and couldn't improve,
    // this block is "naturally optimal".
    if (size <= config_.min_block_size) {
        ++stats_.blocks_natural;
        return {Status::Natural, 0};
    }

    // Step 3: split into two halves and recurse.
    size_t mid = start + size / 2;
    NodeResult left  = optimise_node(work, anchor, bb_name, start, mid);

    // ── Index-shift propagation ──────────────────────────────────────
    // If the left child adopted a rewrite, the instruction count may have
    // changed. The right child's range [mid, end) must be adjusted by
    // the left child's size delta — otherwise the right child would
    // operate on stale indices (the instructions shifted when the left
    // range was shortened).
    mid += static_cast<ptrdiff_t>(left.size_delta);
    end += static_cast<ptrdiff_t>(left.size_delta);

    NodeResult right = optimise_node(work, anchor, bb_name, mid, end);

    // Adjust end for the right child's adoption (the parent's range
    // [start, end) must reflect the current block state).
    end += static_cast<ptrdiff_t>(right.size_delta);

    int total_delta = left.size_delta + right.size_delta;

    // Step 4: decide the parent's status based on the children's.
    if (left.status == Status::Natural && right.status == Status::Natural) {
        // All children (and grandchildren) are naturally optimal →
        // the parent is naturally optimal too. No need to re-try.
        return {Status::Natural, total_delta};
    }

    if (left.status == Status::Synthetic && right.status == Status::Synthetic) {
        // Both children were synthetically optimised → the parent's
        // context has changed (both halves are now different). Re-try
        // the parent: the new context might unlock a whole-range
        // optimisation that wasn't available before.
        int parent_delta = try_optimise_range(work, anchor, bb_name, start, end);
        if (parent_delta != 0) {
            ++stats_.blocks_optimised;
            return {Status::Synthetic, total_delta + parent_delta};
        }
    }

    // At least one child was synthetic (otherwise we'd have returned
    // NATURAL above). The parent is synthetic: something in its subtree
    // changed, even if the parent itself wasn't re-optimised.
    return {Status::Synthetic, total_delta};
}

// ── try_optimise_range ─────────────────────────────────────────────────────

int BlockOptimiser::try_optimise_range(
    ir::Function& work, const ir::Function& anchor,
    const std::string& bb_name, size_t start, size_t end)
{
    auto bb = work.block(bb_name);
    if (!bb) return 0;
    auto& instrs = bb->instructions();
    if (end > instrs.size() || start >= end) return 0;

    const size_t range_size = end - start;
    if (range_size < config_.min_block_size) return 0;

    // ── Collect the range's instructions ─────────────────────────────
    std::vector<std::shared_ptr<ir::Instruction>> range_instrs(
        instrs.begin() + static_cast<ptrdiff_t>(start),
        instrs.begin() + static_cast<ptrdiff_t>(end));

    // Bail if any instruction is not an integer binop. The synthesiser
    // only enumerates integer binops, so any range containing casts,
    // comparisons, memory ops, etc. is out of scope.
    for (auto& inst : range_instrs) {
        if (!inst || !inst->is_binary_op()) return 0;
        if (!inst->type() || !inst->type()->is_integer()) return 0;
    }

    // ── Determine range defs (SSA names defined within the range) ────
    std::unordered_set<std::string> range_defs;
    for (auto& inst : range_instrs) {
        if (inst && inst->has_name()) range_defs.insert(inst->name());
    }

    // ── Determine inputs (live-in): operands defined OUTSIDE the range ──
    std::vector<std::shared_ptr<ir::Value>> inputs;
    std::unordered_set<std::string> seen_inputs;
    for (auto& inst : range_instrs) {
        if (!inst) continue;
        for (auto& op : inst->operands()) {
            if (!op || !op->has_name()) continue;
            if (range_defs.count(op->name())) continue;
            if (seen_inputs.count(op->name())) continue;
            seen_inputs.insert(op->name());
            inputs.push_back(op);
        }
    }

    // All inputs must be integer (the synthesiser only handles integers).
    for (auto& v : inputs) {
        if (!v->type() || !v->type()->is_integer()) return 0;
    }
    // Cap the input count to keep enumeration tractable.
    if (inputs.size() > 4) return 0;

    // ── Determine outputs (live-out): range defs used AFTER the range ──
    // For single-block functions, we only need to check instructions
    // after `end` in this block.
    std::unordered_set<std::string> output_names;
    for (size_t i = end; i < instrs.size(); ++i) {
        auto& inst = instrs[i];
        if (!inst) continue;
        for (auto& op : inst->operands()) {
            if (!op || !op->has_name()) continue;
            if (range_defs.count(op->name())) {
                output_names.insert(op->name());
            }
        }
    }

    // Only handle single-output ranges. Zero outputs = dead code (DCE
    // handles it). Multiple outputs = too complex for the synthesiser
    // (the candidate would need to produce multiple named results).
    if (output_names.size() != 1) return 0;
    const std::string output_name = *output_names.begin();

    // ── Determine the integer width ──────────────────────────────────
    unsigned width = common_int_width(work);
    // Prefer the width of the output instruction.
    for (auto& inst : range_instrs) {
        if (inst && inst->has_name() && inst->name() == output_name) {
            auto* it = dynamic_cast<const ir::IntegerType*>(inst->type().get());
            if (it) width = it->bits();
            break;
        }
    }

    // ── Build the constant pool ──────────────────────────────────────
    std::vector<std::shared_ptr<ir::ConstantInt>> consts;
    consts.reserve(kConstants.size());
    for (int64_t c : kConstants) {
        consts.push_back(ir::ConstantInt::get(type_ctx_, c, width));
    }

    // ── Score the current working baseline ───────────────────────────
    const double baseline_score = engine_->analyse(work).score;

    // ── Helper: build a candidate function from a spec ───────────────
    // The spec is a sequence of `depth` binop steps. The last step's
    // result is renamed to `output_name` so later references resolve.
    auto build_candidate = [&](const std::vector<HoleStep>& steps)
        -> std::shared_ptr<ir::Function>
    {
        auto cand = ir::deep_copy_function(work);
        if (!cand) return nullptr;  // lambda return — not the function's return
        auto cbb = cand->block(bb_name);
        if (!cbb) return nullptr;
        auto& cinstrs = cbb->instructions();
        if (end > cinstrs.size()) return nullptr;

        // Build a name → Value map for the candidate function (for
        // resolving input refs).
        std::unordered_map<std::string, std::shared_ptr<ir::Value>> def_map;
        for (auto& arg : cand->arguments()) {
            def_map[arg.name] = std::make_shared<ir::Value>(arg.type, arg.name);
        }
        for (auto& block : cand->blocks()) {
            for (auto& inst : block->instructions()) {
                if (inst && inst->has_name()) def_map[inst->name()] = inst;
            }
        }

        // Resolve input values in the candidate.
        std::vector<std::shared_ptr<ir::Value>> cand_inputs;
        cand_inputs.reserve(inputs.size());
        for (auto& v : inputs) {
            auto it = def_map.find(v->name());
            if (it == def_map.end()) return nullptr;
            cand_inputs.push_back(it->second);
        }

        // Erase the range [start, end) from the candidate's block.
        cinstrs.erase(cinstrs.begin() + static_cast<ptrdiff_t>(start),
                       cinstrs.begin() + static_cast<ptrdiff_t>(end));

        // Build the replacement sequence and insert at `start`.
        std::vector<std::shared_ptr<ir::Value>> prior_results;
        prior_results.reserve(steps.size());
        for (size_t i = 0; i < steps.size(); ++i) {
            const auto& step = steps[i];
            auto lhs = resolve_ref(step.lhs, cand_inputs, consts, prior_results);
            auto rhs = resolve_ref(step.rhs, cand_inputs, consts, prior_results);
            if (!lhs || !rhs) return nullptr;
            if (!lhs->type() || !lhs->type()->is_integer()) return nullptr;
            if (!rhs->type() || !rhs->type()->is_integer()) return nullptr;
            std::string name = (i == steps.size() - 1)
                ? output_name
                : ("_bo_" + std::to_string(i));
            auto inst = ir::inst::make_binop(step.op, lhs, rhs, name);
            cbb->insert_instruction(start + i, inst);
            prior_results.push_back(inst);
        }

        // ── Remap operands by name ───────────────────────────────────
        // After erasing the range and inserting the replacement, instructions
        // AFTER the range (and in successor blocks) still hold shared_ptr
        // operands pointing to the ERASED range instructions. The erased
        // instructions kept the right NAME (e.g. output_name), but are no
        // longer in the block — the Interpreter/SMT verifier may resolve
        // operands by pointer identity, not just by name, so we must
        // re-point every operand that names a value defined in this
        // function to the instruction that CURRENTLY defines that name.
        // (This is the same two-pass remap deep_copy_function does.)
        {
            std::unordered_map<std::string, std::shared_ptr<ir::Value>> defs;
            for (auto& arg : cand->arguments()) {
                defs[arg.name] = std::make_shared<ir::Value>(arg.type, arg.name);
            }
            for (auto& block : cand->blocks()) {
                for (auto& inst : block->instructions()) {
                    if (inst && inst->has_name()) defs[inst->name()] = inst;
                }
            }
            for (auto& block : cand->blocks()) {
                for (auto& inst : block->instructions()) {
                    if (!inst) continue;
                    for (size_t i = 0; i < inst->num_operands(); ++i) {
                        auto op = inst->operand(i);
                        if (!op || !op->has_name()) continue;
                        auto it = defs.find(op->name());
                        if (it != defs.end() && it->second != op) {
                            inst->set_operand(i, it->second);
                        }
                    }
                }
            }
        }

        if (!ir::validate_function(*cand)) return nullptr;
        return cand;
    };

    // ── Helper: test-vector pre-filter ───────────────────────────────
    // Returns true iff `cand` agrees with `anchor` on all probe vectors.
    // Sound: only ever widens "not equivalent".
    auto passes_test_vectors = [&](const ir::Function& cand) -> bool {
        const size_t nargs = anchor.argument_count();
        const size_t nvecs = std::min(config_.test_vector_count, kProbeValues.size());
        for (size_t i = 0; i < nvecs; ++i) {
            std::vector<int64_t> args(nargs, kProbeValues[i]);
            for (size_t j = 1; j < nargs; ++j) {
                args[j] = kProbeValues[(i + j) % kProbeValues.size()];
            }
            auto orig_r = evaluator::Interpreter::interpret(anchor, args);
            if (!orig_r) return false;  // can't verify → be conservative
            auto cand_r = evaluator::Interpreter::interpret(cand, args);
            if (!cand_r) return false;
            if (*orig_r != *cand_r) return false;
        }
        return true;
    };

    // ── Enumerate candidates of depth 1 .. min(max_depth, range_size) ──
    const size_t max_d = std::min(config_.max_depth, range_size);

    // Recursive enumeration of `depth` steps. Each step picks an opcode
    // and two pool refs. The pool at position `pos` (0-based) contains:
    //   inputs (0..n_inputs-1) + consts (0..n_consts-1) + prior (0..pos-1)
    const size_t n_inputs = inputs.size();
    const size_t n_consts = consts.size();

    // Cap on candidates per depth (matches HoleSynth's historical limit).
    // Without this, the depth-3 search space is ~1B candidates for a
    // typical 2-input range — infeasible even with the test-vector
    // pre-filter. The cap ensures each range's enumeration is tractable.
    constexpr size_t kMaxCandidatesAtDepth = 50000;

    // Build the pool of PoolRefs available at position `pos`.
    auto pool_at = [&](size_t pos) -> std::vector<PoolRef> {
        std::vector<PoolRef> p;
        p.reserve(n_inputs + n_consts + pos);
        for (size_t i = 0; i < n_inputs; ++i)
            p.push_back({PoolRef::Input, static_cast<uint16_t>(i)});
        for (size_t i = 0; i < n_consts; ++i)
            p.push_back({PoolRef::Const, static_cast<uint16_t>(i)});
        for (size_t i = 0; i < pos; ++i)
            p.push_back({PoolRef::Prior, static_cast<uint16_t>(i)});
        return p;
    };

    std::vector<HoleStep> current;
    current.reserve(max_d);

    // `adopted` is set when an improvement is spliced into `work`.
    // `aborted` is set when the cap or time budget is hit (abort the
    // entire enumeration, but DON'T treat it as an improvement).
    // `adopted_delta` captures the instruction-count change (replacement
    // length minus original range size, always <= 0).
    bool adopted = false;
    bool aborted = false;
    int adopted_delta = 0;

    // Returns true iff the caller should unwind (improvement adopted or
    // abort requested). `adopted` / `aborted` distinguish the two cases.
    std::function<bool(size_t, size_t&)> dfs = [&](size_t depth, size_t& enumerated) -> bool {
        if (current.size() == depth) {
            if (enumerated >= kMaxCandidatesAtDepth) { aborted = true; return true; }
            ++enumerated;
            ++stats_.candidates_enumerated;
            if (time_up()) { aborted = true; return true; }

            auto cand = build_candidate(current);
            if (!cand) return false;

            // Test-vector pre-filter.
            if (!passes_test_vectors(*cand)) {
                ++stats_.candidates_test_vector_pruned;
                return false;
            }

            // Score: must be strictly cheaper than the working baseline.
            const double cand_score = engine_->analyse(*cand).score;
            if (cand_score <= baseline_score) {
                ++stats_.rejected_by_score;
                return false;
            }

            // SMT-verify against the anchor.
            ++stats_.candidates_smt_verified;
            auto vr = smt_.verify(anchor, *cand);
            if (vr.is_safe()) {
                ++stats_.candidates_equivalent;
                // Adopt: replace `work` with the candidate.
                work = *cand;
                adopted = true;
                // The replacement has `depth` instructions; the original
                // range had `range_size` instructions. The delta is
                // depth - range_size (always <= 0 since depth <= range_size).
                adopted_delta = static_cast<int>(depth) - static_cast<int>(range_size);
                return true;  // improvement adopted — unwind
            }
            return false;
        }

        // Extend the current spec by one step.
        const size_t pos = current.size();
        auto pool = pool_at(pos);
        for (ir::Opcode op : kBinops) {
            for (const auto& lhs : pool) {
                for (const auto& rhs : pool) {
                    // Prune commutative duplicates: for commutative ops,
                    // only emit (lhs, rhs) with lhs <= rhs (by kind, then index).
                    if (is_commutative(op)) {
                        if (lhs.kind > rhs.kind) continue;
                        if (lhs.kind == rhs.kind && lhs.index > rhs.index) continue;
                    }
                    // Prune `x op x` for non-Add/Mul ops (wasteful: sub/xor/
                    // and/or/shift of a value with itself is 0, 0, x, x, 0,
                    // x, x — all of which are cheaper to express directly).
                    if (lhs.kind == rhs.kind &&
                        lhs.index == rhs.index &&
                        op != ir::Opcode::Add && op != ir::Opcode::Mul) {
                        continue;
                    }
                    if (time_up()) { aborted = true; return true; }

                    current.push_back({op, lhs, rhs});
                    if (dfs(depth, enumerated)) {
                        current.pop_back();
                        return true;  // unwind (adopted or aborted)
                    }
                    current.pop_back();
                }
            }
        }
        return false;
    };

    for (size_t d = 1; d <= max_d; ++d) {
        current.clear();
        size_t enumerated = 0;
        adopted = false;
        aborted = false;
        adopted_delta = 0;
        dfs(d, enumerated);
        if (adopted) return adopted_delta;  // improvement spliced — return the delta
        if (aborted) break;                 // cap or time — stop trying deeper depths
    }

    return 0;  // no improvement found
}

} // namespace clunk::search
