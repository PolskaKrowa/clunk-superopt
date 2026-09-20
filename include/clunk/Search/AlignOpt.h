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
 * Clunk AlignOpt — provable-alignment upgrades and stack-realignment removal.
 *
 * clunk emits LLVM IR, so `vmovupd` vs `vmovapd` is decided downstream by the
 * `align` attribute on a vector load/store: `align >= vector width` lets the
 * backend pick the aligned move. Every rewrite here is exact by construction
 * (same trust tier as MemOpt/LoopOpt) and runs as a finalisation step over the
 * whole module, because it needs global alignments and gains nothing from the
 * cost-model tournament.
 *
 *   1. Stack realignment removal. An alloca aligned above the stack's
 *      guaranteed alignment (16 on x86-64) forces a dynamic `and rsp, -N`
 *      prologue. Clang over-aligns locals more often than anything reads
 *      them that strictly. For a NON-ESCAPING alloca (every use is a
 *      load/store address, reached through GEP/bitcast, or an
 *      `llvm.lifetime.*` marker) the alignment is lowered to the largest
 *      alignment any access actually needs, floored at the stack alignment —
 *      and only if every access is then RE-PROVEN aligned, otherwise the
 *      alloca is left alone. `stackrealign` attributes are never touched:
 *      they exist because a caller may misalign the stack, which is not
 *      something a callee can prove away.
 *
 *   2. Aligned vector accesses. Each vector load/store's `align` is raised to
 *      min(provable alignment of its pointer, natural vector alignment).
 *      Provable alignment is a greatest-fixpoint over the pointer
 *      def-use graph (so loop-carried pointer phis work), rooted only in
 *      facts nobody can dispute:
 *        - `alloca ..., align N`             (N)
 *        - `@global ..., align N`            (N)
 *        - a parameter marked `align N`      (N)
 *      and combined through GEP (min of base and every index's stride ×
 *      the power of two that provably divides the index), bitcast, phi and
 *      select. Index divisibility comes from constants, KnownBits and a
 *      phi-aware fixpoint over add/sub/mul/shl/and/or/xor/ext/trunc, so
 *      induction variables (`i += 4`) are understood. Everything else
 *      (call results, loaded pointers, inttoptr, addrspacecast) is 1.
 *      An existing `align` is never lowered, and structs are never trusted
 *      for offsets (size_bytes() ignores their padding).
 *
 * Step 1 runs before step 2 so an over-aligned alloca cannot retroactively
 * justify the very accesses that were keeping it over-aligned.
 */
#include <memory>

#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"

namespace clunk::search {

class AlignOptimizer {
public:
    // Returns the rewritten function, or nullptr if nothing changed. `mod`
    // supplies global alignments; `fn` is never mutated.
    std::shared_ptr<ir::Function> optimize(const ir::Function& fn,
                                           const ir::Module& mod);

    struct Stats {
        size_t accesses_raised = 0;  // vector load/store align increased
        size_t allocas_lowered = 0;  // over-aligned allocas relaxed
    };
    const Stats& stats() const { return stats_; }

private:
    Stats stats_{};
};

} // namespace clunk::search
