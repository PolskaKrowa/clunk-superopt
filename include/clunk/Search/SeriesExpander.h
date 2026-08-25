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
 * Clunk Series Expander — closed-form replacement of arithmetic loops.
 *
 * Detects single-block loops that compute arithmetic series and replaces
 * the entire loop with its closed-form expression. This is the classic
 * "superoptimiser series expansion" win: a loop that sums `i` for
 * `i = 0..n` becomes `n*(n-1)/2`, a loop that accumulates a constant
 * becomes `init + c*n`, etc.
 *
 * ── Patterns recognised ───────────────────────────────────────────────
 *
 * The loop must be single-block with exactly two phis:
 *
 *   1. Induction variable `%i`:
 *        %i = phi [ %start, %preheader ], [ %i_next, %loop ]
 *        %i_next = add %i, %step_const
 *
 *   2. Accumulator `%acc`:
 *        %acc = phi [ %init, %preheader ], [ %acc_next, %loop ]
 *
 *      where `%acc_next` is one of:
 *        a) add %acc, %c             — constant accumulation
 *        b) add %acc, %i             — arithmetic series (sum of i)
 *        c) add %acc, (mul %i, %c)   — scaled arithmetic series
 *
 *   The exit condition must be `icmp pred %i, %bound` where `%bound`
 *   is loop-invariant and `pred` is SLT/ULT (i < bound) or SLE/ULE
 *   (i <= bound), with %step_const > 0.
 *
 * ── Closed forms ──────────────────────────────────────────────────────
 *
 * Let n = trip count, s = start, d = step.
 *
 *   trip n = max(0, ceil((bound - s) / d))          for SLT/ULT
 *   trip n = max(0, ceil((bound - s + 1) / d))      for SLE/ULE
 *
 *   a) acc += c:         acc_final = init + c * n
 *   b) acc += i:         acc_final = init + n*s + d * n*(n-1)/2
 *   c) acc += i*c:       acc_final = init + c * (n*s + d * n*(n-1)/2)
 *
 * The division by 2 is exact (n*(n-1) is always even), so `sdiv` is
 * sound without `exact` flag.
 *
 * ── Soundness ─────────────────────────────────────────────────────────
 *
 * The algebra is exact by construction — same trust tier as LICM and
 * LoopOpt. The rewrite is sound for plain `add` (two's-complement
 * wrapping). Loops with `nsw`/`nuw` flags on the accumulator add are
 * REFUSED: the closed form's intermediate multiplications may wrap
 * where the original loop's sequential adds would not, and without
 * the flags the closed form would produce a concrete value where the
 * original produced poison. Generated instructions carry NO flags
 * (matching the wrapping semantics of the un-flagged original).
 *
 * ── Scope ─────────────────────────────────────────────────────────────
 *
 * Single-block loops, integer types only, ≤4 function arguments. The
 * pass is inert on functions outside this scope.
 */
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "clunk/IR/Function.h"
#include "clunk/IR/Type.h"
#include "clunk/Evaluator/EvaluationEngine.h"

namespace clunk::search {

// ── Accumulator update pattern (used internally and by the implementation)
enum class AccPattern {
    None,               // doesn't match any known pattern
    Constant,           // acc += c
    ArithmeticSeries,   // acc += i
    ScaledArithmetic,   // acc += i * c
};

struct SeriesExpanderConfig {
    // Maximum number of instructions in the loop body (excluding phis
    // and terminator). Bodies larger than this are skipped — the
    // pattern matcher expects a minimal body (induction increment,
    // accumulator update, comparison).
    size_t max_body_instructions = 8;

    // Cap on the function's total instruction count. Functions above
    // this cap are skipped.
    size_t max_function_instructions = 128;
};

class SeriesExpander {
public:
    explicit SeriesExpander(evaluator::EvaluationEngine* engine,
                              const SeriesExpanderConfig& config = {});

    // Try to replace a loop in `fn` with a closed-form expression.
    // Returns the rewritten function, or nullptr if no pattern matched.
    // The returned rewrite is sound-by-construction (the algebra is
    // exact), so the candidate's `sound` flag should be set true.
    std::shared_ptr<ir::Function> expand(const ir::Function& fn);

    struct Stats {
        size_t functions_seen = 0;
        size_t functions_skipped = 0;       // outside scope / no loop
        size_t loops_matched = 0;           // pattern recognised
        size_t loops_expanded = 0;          // closed form emitted & adopted
        size_t loops_rejected = 0;          // pattern matched but bail (nsw, etc.)
        double elapsed_ms = 0.0;
    };
    const Stats& stats() const { return stats_; }

    SeriesExpanderConfig& config() { return config_; }
    const SeriesExpanderConfig& config() const { return config_; }

private:
    evaluator::EvaluationEngine* engine_;
    SeriesExpanderConfig config_;
    ir::TypeContext type_ctx_;
    Stats stats_{};

    // Try to expand the single-block loop in `fn`. Returns the rewritten
    // function or nullptr. `fn` is not modified.
    std::shared_ptr<ir::Function> try_expand_loop(const ir::Function& fn);
};

} // namespace clunk::search
