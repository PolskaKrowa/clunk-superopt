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

#pragma once
/*
 * Clunk Block-Level Optimiser — divide-and-conquer superoptimisation.
 *
 * AFTER the whole-function refinement loop fails to find an improvement,
 * this phase splits the function's single basic block into a binary tree
 * of contiguous instruction ranges (halving at each level until the
 * minimum block size of 2 instructions is reached) and tries to
 * SMT-prove a cheaper equivalent for each range.
 *
 * ── Recursive algorithm (per the user's spec) ─────────────────────────────
 *
 *   optimise_node(range [start, end)):
 *     size = end - start
 *     if size < min_block_size: return NATURAL      // too small to try
 *     if try_optimise_range([start, end)): return SYNTHETIC   // whole range improved
 *     if size == min_block_size: return NATURAL     // at min, can't improve → natural
 *     mid = start + size / 2
 *     left  = optimise_node([start, mid))
 *     right = optimise_node([mid,   end))
 *     if left == NATURAL and right == NATURAL:
 *         return NATURAL                             // all children natural → parent natural
 *     if left == SYNTHETIC and right == SYNTHETIC:
 *         if try_optimise_range([start, end)):       // both children changed → re-try parent
 *             return SYNTHETIC
 *     return SYNTHETIC                               // at least one child changed
 *
 *   NATURAL  = the block (and all its descendants) could not be optimised
 *              further; it is in its "naturally optimal" form.
 *   SYNTHETIC = the block (or some descendant) was actually optimised by
 *              the program; it is "synthetically optimal".
 *
 * ── Data-flow preservation ─────────────────────────────────────────────────
 *
 * When `try_optimise_range` replaces the instructions in [start, end) with
 * a synthesised sequence, it must preserve data flow:
 *   - INPUTS  (live-in):  operands of range instructions that are defined
 *                         OUTSIDE the range (function args or earlier
 *                         instructions). The synthesised sequence may use
 *                         these as operands.
 *   - OUTPUTS (live-out): range instruction results that are used AFTER
 *                         the range. The synthesised sequence's final
 *                         instruction is renamed to the (single) output
 *                         name, so later references resolve correctly.
 *
 * Only ranges with EXACTLY ONE output are handled (the common case: a
 * chain of instructions whose intermediate results are dead, and only
 * the final result escapes). Ranges with zero outputs (dead code) are
 * left to DCE; ranges with multiple outputs are skipped (the whole-
 * function search or other phases handle them).
 *
 * ── Soundness ──────────────────────────────────────────────────────────────
 *
 * Every adopted range rewrite is SMT-proven equivalent to the INPUT
 * function (the baseline at the start of the phase). By transitivity
 * (the input is itself equivalent to the refinement chain's original),
 * the final result is equivalent to the original. The returned candidate
 * carries `proven = true`.
 *
 * ── Scope ──────────────────────────────────────────────────────────────────
 *
 * Single-block integer functions (same scope as HoleSynth). The phase is
 * inert on functions outside this scope (multi-block, FP, memory, etc.).
 */
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "clunk/IR/Function.h"
#include "clunk/IR/Type.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Search/SMTVerifier.h"

namespace clunk::search {

struct BlockOptimiserConfig {
    // Minimum block size (in instructions). Blocks at or below this size
    // that cannot be optimised are declared "naturally optimal" without
    // further splitting. Must be >= 2 (a 1-instruction block has nothing
    // to synthesise — the instruction IS the optimal form).
    size_t min_block_size = 2;

    // Maximum enumeration depth for candidate replacement sequences.
    // A replacement of depth D has D instructions. Capped at the range
    // size, so larger ranges don't get over-long replacements. Most
    // superoptimiser wins live at depth <= 3 (Massalin).
    size_t max_depth = 3;

    // Wall-clock budget in seconds for a single optimize() call.
    // 0 = no internal cap (rely on the pipeline-level time budget).
    double time_budget_seconds = 5.0;

    // Per-candidate SMT timeout in milliseconds.
    unsigned smt_timeout_ms = 5000;

    // Return unproven rewrites (test-vector-passing + cheaper) when SMT
    // is unavailable or returns Unknown. Off by default: no proof, no
    // rewrite.
    bool trust_unverified = false;

    // Number of random test vectors used as a pre-filter before SMT.
    // A candidate that does not agree with the input on every test
    // vector is skipped without an SMT call (sound: the pre-filter only
    // ever widens "not equivalent", never blesses a wrong candidate).
    size_t test_vector_count = 16;

    // Cap on the function's instruction count. Functions above this cap
    // are skipped (the block optimiser is most effective on small-to-
    // medium integer functions where the halving tree is shallow).
    size_t max_function_instructions = 64;
};

class BlockOptimiser {
public:
    explicit BlockOptimiser(evaluator::EvaluationEngine* engine,
                             const BlockOptimiserConfig& config = {});

    // Try to optimise `fn` by splitting it into blocks and SMT-proving
    // cheaper equivalents for each. Returns the rewritten function, or
    // nullptr if no block was optimised. `proven` (optional out) is set
    // true iff the returned rewrite carries SMT equivalence proofs.
    //
    // `fn` is NOT modified; a deep copy is made internally.
    std::shared_ptr<ir::Function> optimize(const ir::Function& fn,
                                            bool* proven = nullptr);

    struct Stats {
        size_t functions_seen = 0;
        size_t functions_skipped = 0;       // outside scope (multi-block, FP, ...)
        size_t blocks_tried = 0;            // ranges attempted (whole + split)
        size_t blocks_optimised = 0;        // ranges with an adopted rewrite
        size_t blocks_natural = 0;          // ranges declared naturally optimal
        size_t candidates_enumerated = 0;   // total replacement sequences tried
        size_t candidates_test_vector_pruned = 0;
        size_t candidates_smt_verified = 0;
        size_t candidates_equivalent = 0;   // SMT-proven equivalent
        size_t rejected_by_score = 0;       // verified but not cheaper
        size_t proven = 0;                  // rewrites returned with a proof
        double elapsed_ms = 0.0;
    };
    const Stats& stats() const { return stats_; }

    BlockOptimiserConfig& config() { return config_; }
    const BlockOptimiserConfig& config() const { return config_; }

private:
    evaluator::EvaluationEngine* engine_;
    BlockOptimiserConfig config_;
    ir::TypeContext type_ctx_;
    Stats stats_{};
    SMTVerifier smt_;

    // Internal: the recursive status of a block node.
    enum class Status { Natural, Synthetic };

    // Result of optimise_node: the status (Natural/Synthetic) and the
    // net change in instruction count for this subtree's range. A
    // negative delta means the range was shortened by an adoption; zero
    // means no change. The delta is propagated up so the parent can
    // adjust its child ranges after an adoption shifts the instruction
    // indices.
    struct NodeResult {
        Status status = Status::Natural;
        int size_delta = 0;  // new_range_size - old_range_size
    };

    // Recursive divide-and-conquer. Optimises the range [start, end) in
    // `work` (the working copy), verifying each candidate against `anchor`
    // (the immutable input function). Returns the status and the net
    // instruction-count delta for this subtree.
    NodeResult optimise_node(ir::Function& work, const ir::Function& anchor,
                              const std::string& bb_name,
                              size_t start, size_t end);

    // Try to find a cheaper equivalent for the range [start, end) in
    // `work`'s basic block `bb_name`. If found, splice it into `work`
    // and return the instruction-count delta (new_range_size - old_range_size,
    // always <= 0 since replacements are never longer than the original).
    // Returns 0 if no cheaper equivalent was found (or the range is
    // outside the synthesiser's scope). `anchor` is the soundness anchor
    // for SMT verification.
    int try_optimise_range(ir::Function& work, const ir::Function& anchor,
                            const std::string& bb_name,
                            size_t start, size_t end);

    // Wall-clock deadline for this optimize() call. Set in optimize().
    std::chrono::steady_clock::time_point deadline_{};
    bool has_deadline_ = false;

    // Check whether the deadline has expired.
    bool time_up() const;
};

} // namespace clunk::search
