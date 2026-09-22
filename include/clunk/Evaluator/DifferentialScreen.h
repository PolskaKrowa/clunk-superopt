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
 * Clunk DifferentialScreen — run a candidate against the original on
 * concrete inputs BEFORE spending a solver call on it.
 *
 * Almost every candidate a search stage proposes is wrong, and almost
 * every wrong candidate is wrong on an input a person would try first:
 * 0, -1, INT_MIN, a power of two. Evaluating both functions on a few
 * hundred such inputs costs microseconds; a Z3 query costs milliseconds to
 * seconds. Only candidates that survive the screen are worth proving.
 *
 * INPUTS (deterministic for a given seed, canonical for each argument's
 * own bit-width, so an i8 argument is never handed 100000):
 *   1. the cross product of a core set {0, 1, -1, 2, INT_MIN, INT_MAX};
 *   2. one value broadcast to every argument, over the full edge set;
 *   3. single-argument sweeps over the full edge set — every power of two,
 *      every negated power of two, every 2^k-1 mask, the shift-amount
 *      boundaries w-1 and w — with the others at 0, 1 or -1;
 *   4. random vectors: uniform bits, sparse (1-3 bits set), dense (the
 *      complement of sparse) and small-magnitude values.
 *
 * VERDICT follows LLVM's REFINEMENT rule, the same one the SMT encoder
 * uses (Alive2-style), so a screen rejection is never a rejection the
 * prover would have accepted:
 *   - original returns poison or hits UB on an input -> the candidate may
 *     do anything there; the input is skipped;
 *   - original returns a defined value -> the candidate must return the
 *     same value; a different value, poison or UB rejects it.
 *
 * The screen is one-sided by design. `Rejected` is a proof (a concrete
 * counterexample). `Survived` proves nothing and must still go to the
 * prover. `Inconclusive` means nothing could be compared — the function
 * is not interpretable, has loops, returns void, or every input was
 * poison/UB in the original — and must also go to the prover.
 */
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "clunk/IR/Function.h"

namespace clunk::evaluator {

struct ScreenConfig {
    size_t   max_vectors    = 384;        // hard cap on vectors tried
    size_t   random_vectors = 128;        // of which random (the rest are edge cases)
    uint64_t seed           = 0xC1A0DE57; // fixed: screens are reproducible
};

enum class ScreenVerdict {
    Rejected,      // a concrete input shows the candidate is NOT a refinement
    Survived,      // agreed on every comparable input (and there was at least one)
    Inconclusive   // nothing comparable — the prover must decide
};

struct ScreenResult {
    ScreenVerdict verdict = ScreenVerdict::Inconclusive;
    size_t compared = 0;                  // inputs where both sides gave a comparable answer
    size_t skipped  = 0;                  // original poison/UB or unsupported: no verdict
    std::vector<int64_t> counterexample;  // set when Rejected
    std::string reason;
};

class DifferentialScreen {
public:
    explicit DifferentialScreen(ScreenConfig cfg = {}) : cfg_(cfg) {}

    ScreenResult screen(const ir::Function& original,
                        const ir::Function& candidate) const;

    // The input vectors screen() would use for `fn`'s signature (exposed so
    // tests and callers can inspect coverage). Values are canonical for each
    // parameter's width (sign-extended from that width).
    std::vector<std::vector<int64_t>> make_inputs(const ir::Function& fn) const;

    // Edge-case values for a `bits`-wide integer, most valuable first.
    static std::vector<int64_t> edge_values(unsigned bits);

private:
    ScreenConfig cfg_;
};

} // namespace clunk::evaluator
