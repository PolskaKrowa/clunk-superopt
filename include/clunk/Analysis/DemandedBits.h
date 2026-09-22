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
 * Clunk DemandedBits — which bits of each integer value can anyone observe?
 *
 * The dual of KnownBits. KnownBits asks "what do we know about this
 * value's bits?"; DemandedBits asks "which of this value's bits could ever
 * change the program's observable behaviour?". A bit nobody demands may be
 * anything, so a rewrite that only differs from the original in undemanded
 * bits is still correct — e.g. `(x & 0xFF) + (y & 0xFF)` stored to an i8
 * never needs the upper 24 bits of `x` and `y` at all.
 *
 * It is a backward fixpoint. Observable behaviour is the ROOTS — returns,
 * stores, calls, address computations, divisions (which can trap) and
 * branch conditions — each of which demands all bits of its integer
 * operands (a branch condition demands its one bit). Demand then flows
 * backwards through pure operations:
 *
 *     and x, C        demands D & C of x          or x, C     D & ~C
 *     xor             D                           add/sub/mul every bit at or below D's top bit
 *     shl x, c        D >> c                      lshr x, c   D << c
 *     ashr x, c       D << c (+ the sign bit if any shifted-in bit is demanded)
 *     variable shift  bits at/below (shl) or at/above (lshr, ashr) D, all of the amount
 *     trunc/zext      D restricted to the source width
 *     sext            likewise, plus the sign bit if a bit above the source is demanded
 *     select          D for both arms, bit 0 of the condition
 *     phi             D for every incoming value
 *     icmp            every bit of both operands (when the result is demanded)
 *
 * Anything without a rule demands every bit of its integer operands.
 *
 * SOUNDNESS CAVEAT (same as LLVM's DemandedBits): this describes VALUE bits
 * only. A rewrite that changes an undemanded bit of an operand must also
 * drop nsw / nuw / exact on the instructions that consume it, because
 * those flags turn a changed bit into poison.
 *
 * Only integer values of up to 64 bits are tracked (instruction results and
 * function arguments); anything else reports "all bits demanded".
 */
#include <cstdint>
#include <string>
#include <unordered_map>

#include "clunk/IR/Function.h"

namespace clunk::analysis {

struct DemandedBitsInfo {
    std::unordered_map<std::string, uint64_t> demanded;   // SSA name -> demanded bit mask
    std::unordered_map<std::string, unsigned> width;      // SSA name -> bit width (tracked values)

    bool is_tracked(const std::string& name) const { return width.count(name) != 0; }

    // Demanded bits of `name`; every bit if the value is not tracked.
    uint64_t of(const std::string& name) const {
        auto it = demanded.find(name);
        return it == demanded.end() ? ~uint64_t{0} : it->second;
    }

    // True if no bit of the value is ever observed (it is dead as a value).
    bool is_dead(const std::string& name) const {
        auto it = demanded.find(name);
        return it != demanded.end() && it->second == 0;
    }
};

DemandedBitsInfo compute_demanded_bits(const ir::Function& fn);

} // namespace clunk::analysis
