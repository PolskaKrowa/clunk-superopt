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
 * Clunk DemandedBits — see the header for the rules and the caveat.
 */
#include "clunk/Analysis/DemandedBits.h"

#include <cstdint>
#include <unordered_set>

#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Instruction.h"

namespace clunk::analysis {

namespace {

using ir::Opcode;

uint64_t width_mask(unsigned w) { return w >= 64 ? ~uint64_t{0} : (uint64_t{1} << w) - 1; }

// All bits at or below the highest set bit of d (what a carry chain can see).
uint64_t bits_up_to_top(uint64_t d) {
    if (d == 0) return 0;
    const unsigned len = 64 - static_cast<unsigned>(__builtin_clzll(d));
    return width_mask(len);
}

// All bits at or above the lowest set bit of d, within `w` bits.
uint64_t bits_from_bottom(uint64_t d, unsigned w) {
    if (d == 0) return 0;
    const unsigned low = static_cast<unsigned>(__builtin_ctzll(d));
    return width_mask(w) & ~width_mask(low);
}

// The value of a constant integer operand, masked to `w` bits.
bool constant_of(const std::shared_ptr<ir::Value>& v, unsigned w, uint64_t& out) {
    if (const auto* c = dynamic_cast<const ir::ConstantInt*>(v.get())) {
        out = static_cast<uint64_t>(c->value()) & width_mask(w);
        return true;
    }
    return false;
}

// Opcodes with a refined demand rule. Everything else is a root or falls
// back to "all bits of every operand".
bool has_rule(Opcode op) {
    switch (op) {
        case Opcode::And: case Opcode::Or: case Opcode::Xor:
        case Opcode::Add: case Opcode::Sub: case Opcode::Mul:
        case Opcode::Shl: case Opcode::LShr: case Opcode::AShr:
        case Opcode::Trunc: case Opcode::ZExt: case Opcode::SExt:
        case Opcode::Select: case Opcode::Phi: case Opcode::ICmp:
            return true;
        default:
            return false;
    }
}

} // namespace

DemandedBitsInfo compute_demanded_bits(const ir::Function& fn) {
    DemandedBitsInfo info;

    // Tracked values: integer arguments and integer instruction results, <= 64 bits.
    auto track = [&](const std::string& name, const std::shared_ptr<ir::Type>& t) {
        if (name.empty() || !t || !t->is_integer() || t->bit_width() == 0 || t->bit_width() > 64) return;
        info.width[name] = static_cast<unsigned>(t->bit_width());
        info.demanded[name] = 0;
    };
    for (const auto& a : fn.arguments()) track(a.name, a.type);
    for (const auto& b : fn.blocks())
        if (b) for (const auto& i : b->instructions())
            if (i && i->has_name()) track(i->name(), i->type());

    bool changed = false;
    // Add `mask` (clipped to the operand's width) to the demand on operand `v`.
    auto demand = [&](const std::shared_ptr<ir::Value>& v, uint64_t mask) {
        if (!v || !v->has_name() || v->is_global()) return;
        auto w = info.width.find(v->name());
        if (w == info.width.end()) return;
        uint64_t& d = info.demanded[v->name()];
        const uint64_t add = mask & width_mask(w->second) & ~d;
        if (add) { d |= add; changed = true; }
    };
    auto demand_all = [&](const std::shared_ptr<ir::Value>& v) { demand(v, ~uint64_t{0}); };

    do {
        changed = false;
        for (auto bit = fn.blocks().rbegin(); bit != fn.blocks().rend(); ++bit) {
            if (!*bit) continue;
            const auto& insts = (*bit)->instructions();
            for (auto it = insts.rbegin(); it != insts.rend(); ++it) {
                const auto& in = *it;
                if (!in) continue;
                const Opcode op = in->opcode();

                // Roots and unknown opcodes: every bit of every integer operand.
                if (!has_rule(op)) {
                    if (op == Opcode::Br && in->num_operands() >= 1) demand(in->operand(0), 1);   // i1 condition
                    else for (const auto& o : in->operands()) demand_all(o);
                    continue;
                }

                // Pure operations: demand flows from the result.
                const unsigned w = in->has_name() && info.width.count(in->name()) ? info.width[in->name()] : 0;
                const uint64_t D = in->has_name() && w ? info.demanded[in->name()] & width_mask(w) : 0;
                const size_t n = in->num_operands();
                auto opd = [&](size_t k) -> std::shared_ptr<ir::Value> { return k < n ? in->operand(k) : nullptr; };

                if (op == Opcode::ICmp) {                       // result is i1: all-or-nothing
                    if (D) for (const auto& o : in->operands()) demand_all(o);
                    continue;
                }
                if (!w) {   // a pure op whose result we do not track (e.g. i128): stay conservative
                    for (const auto& o : in->operands()) demand_all(o);
                    continue;
                }
                uint64_t c = 0;
                switch (op) {
                    case Opcode::And:
                        if (constant_of(opd(1), w, c)) { demand(opd(0), D & c); }
                        else if (constant_of(opd(0), w, c)) { demand(opd(1), D & c); }
                        else { demand(opd(0), D); demand(opd(1), D); }
                        break;
                    case Opcode::Or:
                        if (constant_of(opd(1), w, c)) { demand(opd(0), D & ~c); }
                        else if (constant_of(opd(0), w, c)) { demand(opd(1), D & ~c); }
                        else { demand(opd(0), D); demand(opd(1), D); }
                        break;
                    case Opcode::Xor:
                        demand(opd(0), D); demand(opd(1), D);
                        break;
                    case Opcode::Add: case Opcode::Sub: case Opcode::Mul: {
                        const uint64_t m = bits_up_to_top(D);
                        demand(opd(0), m); demand(opd(1), m);
                        break;
                    }
                    case Opcode::Shl:
                        if (constant_of(opd(1), w, c)) { if (c < w) demand(opd(0), D >> c); }
                        else { demand(opd(0), bits_up_to_top(D)); if (D) demand_all(opd(1)); }
                        break;
                    case Opcode::LShr:
                        if (constant_of(opd(1), w, c)) { if (c < w) demand(opd(0), D << c); }
                        else { demand(opd(0), bits_from_bottom(D, w)); if (D) demand_all(opd(1)); }
                        break;
                    case Opcode::AShr:
                        if (constant_of(opd(1), w, c)) {
                            if (c < w) {
                                uint64_t m = (D << c) & width_mask(w);
                                if (c > 0 && (D & ~width_mask(w - static_cast<unsigned>(c)))) m |= uint64_t{1} << (w - 1);   // sign copies
                                demand(opd(0), m);
                            }
                        } else { demand(opd(0), bits_from_bottom(D, w)); if (D) demand_all(opd(1)); }
                        break;
                    case Opcode::Trunc:
                        demand(opd(0), D);                      // bit j of the source is bit j of the result
                        break;
                    case Opcode::ZExt: {
                        const auto src = opd(0);
                        const unsigned sw = src && src->type() && src->type()->is_integer() ? static_cast<unsigned>(src->type()->bit_width()) : 64;
                        demand(src, D & width_mask(sw));
                        break;
                    }
                    case Opcode::SExt: {
                        const auto src = opd(0);
                        const unsigned sw = src && src->type() && src->type()->is_integer() ? static_cast<unsigned>(src->type()->bit_width()) : 64;
                        uint64_t m = D & width_mask(sw);
                        if (sw > 0 && sw < 64 && (D & ~width_mask(sw))) m |= uint64_t{1} << (sw - 1);   // extension copies the sign
                        demand(src, m);
                        break;
                    }
                    case Opcode::Select:
                        if (D) demand(opd(0), 1);
                        demand(opd(1), D); demand(opd(2), D);
                        break;
                    case Opcode::Phi:
                        for (const auto& o : in->operands()) demand(o, D);
                        break;
                    default: break;
                }
            }
        }
    } while (changed);
    return info;
}

} // namespace clunk::analysis
