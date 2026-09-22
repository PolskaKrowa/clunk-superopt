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
 * Clunk LiveOut — SSA liveness, and "which values escape this range?".
 *
 * A value is LIVE-OUT of a block if some path from the block's exit reaches
 * a use of it. This is the question a slice-based optimiser has to answer
 * before it may rewrite an instruction range: only the values that are
 * live out of the range must be preserved; everything else the range
 * defines is free to change or disappear.
 *
 * Values are SSA names (instruction results and function arguments; globals
 * and constants are not tracked). The analysis is the standard backward
 * dataflow, with phis handled per edge:
 *
 *     live_in(b)  = upward_exposed_uses(b)  ∪  (live_out(b) \ defs(b))
 *     live_out(b) = ⋃_{s ∈ succ(b)}  live_in(s)  ∪  phi_operands(s, from b)
 *
 * A phi's incoming value is live-out of THAT predecessor only (never of the
 * phi's own block, and never of the other predecessors), and a phi's result
 * is defined at block entry so it is not live-in. A phi that does not record
 * its incoming blocks is treated conservatively: each operand is live-out of
 * every predecessor (liveness may only be over-, never under-estimated).
 *
 * Sets are std::set so results are deterministic and easy to compare.
 */
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "clunk/IR/Function.h"

namespace clunk::analysis {

struct LiveOutInfo {
    std::unordered_map<std::string, std::set<std::string>> live_in;
    std::unordered_map<std::string, std::set<std::string>> live_out;

    bool is_live_out(const std::string& block, const std::string& name) const {
        auto it = live_out.find(block);
        return it != live_out.end() && it->second.count(name) != 0;
    }
};

LiveOutInfo compute_live_out(const ir::Function& fn);

// The values defined by instructions [first, last) of `block` that are used
// after the range: by a later instruction of the same block, or because they
// are live out of the block. In program order.
std::vector<std::string> range_live_outs(const ir::Function& fn,
                                         const LiveOutInfo& info,
                                         const std::string& block,
                                         size_t first, size_t last);

} // namespace clunk::analysis
