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
 * Clunk IR verifier — the checks `validate_function` does not make.
 *
 * validate_function (Clone.h) only asks "is this name defined earlier in
 * layout order?". A candidate generator can satisfy that and still emit IR
 * that LLVM's verifier rejects: an instruction with too few operands, a
 * binary op whose operands have different widths, a `ret` of the wrong
 * type, or a use that is not dominated by its definition (e.g. a value
 * hash-consed across sibling blocks).
 *
 * verify_function() returns "" for a well-formed function and otherwise a
 * one-line reason. It is deliberately CONSERVATIVE: it only reports
 * definite violations for the integer subset it fully understands
 * (arity, integer operand/result widths, casts, icmp/select, ret type,
 * SSA dominance) and stays silent about everything else (pointers,
 * floats, vectors, calls), so a false alarm on real-world IR is not a
 * risk worth worrying about.
 */
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "clunk/IR/Function.h"

namespace clunk::ir {

namespace verify_detail {

// Integer bit-width of a value's type, or 0 if it is not an integer.
inline unsigned int_width(const std::shared_ptr<Value>& v) {
    return v && v->type() && v->type()->is_integer()
               ? static_cast<unsigned>(v->type()->bit_width()) : 0;
}

// Same, for a bare type (an instruction's own result type).
inline unsigned type_width(const std::shared_ptr<Type>& t) {
    return t && t->is_integer() ? static_cast<unsigned>(t->bit_width()) : 0;
}

inline size_t min_operands(Opcode op) {
    switch (op) {
        case Opcode::Add: case Opcode::Sub: case Opcode::Mul:
        case Opcode::UDiv: case Opcode::SDiv: case Opcode::URem: case Opcode::SRem:
        case Opcode::And: case Opcode::Or: case Opcode::Xor:
        case Opcode::Shl: case Opcode::LShr: case Opcode::AShr:
        case Opcode::ICmp: case Opcode::Store: return 2;
        case Opcode::Select: return 3;
        case Opcode::ZExt: case Opcode::SExt: case Opcode::Trunc:
        case Opcode::Load: return 1;
        default: return 0;
    }
}

inline bool is_int_binop(Opcode op) {
    switch (op) {
        case Opcode::Add: case Opcode::Sub: case Opcode::Mul:
        case Opcode::UDiv: case Opcode::SDiv: case Opcode::URem: case Opcode::SRem:
        case Opcode::And: case Opcode::Or: case Opcode::Xor:
        case Opcode::Shl: case Opcode::LShr: case Opcode::AShr: return true;
        default: return false;
    }
}

// Incoming block names of a phi ("phi_blocks" metadata, comma separated).
inline std::vector<std::string> phi_blocks(const Instruction& phi) {
    std::vector<std::string> out;
    auto it = phi.metadata().find("phi_blocks");
    if (it == phi.metadata().end()) return out;
    std::string cur;
    for (char c : it->second) {
        if (c == ',') { out.push_back(cur); cur.clear(); } else cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

} // namespace verify_detail

inline std::string verify_function(const Function& fn) {
    using namespace verify_detail;
    const auto& blocks = fn.blocks();
    const size_t n = blocks.size();
    if (n == 0) return "";  // a declaration

    std::unordered_map<std::string, size_t> bidx;
    for (size_t i = 0; i < n; ++i)
        if (blocks[i]) bidx[blocks[i]->name()] = i;

    // ── Dominators (iterative, one bit-vector per block) ────────────────
    std::vector<std::vector<size_t>> preds(n);
    std::vector<bool> reachable(n, false);
    {
        std::vector<size_t> work{0};
        reachable[0] = true;
        while (!work.empty()) {
            size_t b = work.back(); work.pop_back();
            if (!blocks[b]) continue;
            for (const auto& s : blocks[b]->successors()) {
                auto it = bidx.find(s);
                if (it == bidx.end()) continue;
                preds[it->second].push_back(b);
                if (!reachable[it->second]) { reachable[it->second] = true; work.push_back(it->second); }
            }
        }
    }
    // Dominance is O(n^2) bits; past this many blocks skip it (the arity,
    // typing and phi checks still run) rather than stall on a huge function.
    constexpr size_t kMaxBlocksForDominance = 2048;
    const bool check_dominance = n <= kMaxBlocksForDominance;
    std::vector<std::vector<bool>> dom(check_dominance ? n : 0, std::vector<bool>(check_dominance ? n : 0, true));
    for (size_t j = 0; check_dominance && j < n; ++j) dom[0][j] = (j == 0);
    for (bool changed = check_dominance; changed;) {
        changed = false;
        for (size_t b = 1; b < n; ++b) {
            if (!reachable[b]) continue;
            std::vector<bool> nd(n, true);
            bool any = false;
            for (size_t p : preds[b]) {
                if (!reachable[p]) continue;
                any = true;
                for (size_t j = 0; j < n; ++j) nd[j] = nd[j] && dom[p][j];
            }
            if (!any) std::fill(nd.begin(), nd.end(), false);
            nd[b] = true;
            if (nd != dom[b]) { dom[b] = nd; changed = true; }
        }
    }

    // ── Definitions ─────────────────────────────────────────────────────
    struct Def { size_t block; size_t pos; };
    std::unordered_map<std::string, Def> defs;
    std::unordered_map<std::string, bool> is_arg;
    for (const auto& a : fn.arguments()) if (!a.name.empty()) is_arg[a.name] = true;
    for (size_t b = 0; b < n; ++b) {
        if (!blocks[b]) continue;
        const auto& insts = blocks[b]->instructions();
        for (size_t p = 0; p < insts.size(); ++p) {
            if (!insts[p] || !insts[p]->has_name()) continue;
            if (defs.count(insts[p]->name()) || is_arg.count(insts[p]->name()))
                return "%" + insts[p]->name() + " is defined more than once";
            defs[insts[p]->name()] = {b, p};
        }
    }

    const unsigned ret_w = fn.return_type() && fn.return_type()->is_integer()
                               ? static_cast<unsigned>(fn.return_type()->bit_width()) : 0;

    // A value's definition must dominate the point `at_block`/`at_pos`.
    auto dominated = [&](const std::string& name, size_t at_block, size_t at_pos) -> bool {
        if (is_arg.count(name)) return true;
        auto d = defs.find(name);
        if (d == defs.end()) return false;
        if (d->second.block == at_block) return d->second.pos < at_pos;
        return !check_dominance || dom[at_block][d->second.block];
    };

    for (size_t b = 0; b < n; ++b) {
        if (!blocks[b]) continue;
        if (reachable[b] && !blocks[b]->is_well_formed())
            return "block '" + blocks[b]->name() + "' has no terminator";
        const auto& insts = blocks[b]->instructions();
        for (size_t p = 0; p < insts.size(); ++p) {
            const auto& in = insts[p];
            if (!in) continue;
            const Opcode op = in->opcode();
            const std::string where = in->has_name() ? "%" + in->name() : std::string(Instruction::opcode_name(op));

            if (in->num_operands() < min_operands(op))
                return where + ": " + Instruction::opcode_name(op) + " has " +
                       std::to_string(in->num_operands()) + " operand(s), needs " +
                       std::to_string(min_operands(op));

            // ── Integer typing ──────────────────────────────────────────
            if (is_int_binop(op)) {
                const unsigned a = int_width(in->operand(0)), c = int_width(in->operand(1)), r = type_width(in->type());
                if (a && c && a != c) return where + ": operand widths differ (i" + std::to_string(a) + " vs i" + std::to_string(c) + ")";
                if (a && r && a != r) return where + ": result i" + std::to_string(r) + " but operands i" + std::to_string(a);
            } else if (op == Opcode::ZExt || op == Opcode::SExt || op == Opcode::Trunc) {
                const unsigned s = int_width(in->operand(0)), d = type_width(in->type());
                if (s && d && (op == Opcode::Trunc ? d >= s : d <= s))
                    return where + ": invalid cast i" + std::to_string(s) + " -> i" + std::to_string(d);
            } else if (op == Opcode::ICmp) {
                const unsigned a = int_width(in->operand(0)), c = int_width(in->operand(1));
                if (a && c && a != c) return where + ": icmp operand widths differ";
            } else if (op == Opcode::Select) {
                const unsigned cw = int_width(in->operand(0)), t = int_width(in->operand(1)), f = int_width(in->operand(2));
                if (cw && cw != 1) return where + ": select condition is not i1";
                if (t && f && t != f) return where + ": select arms differ in width";
            } else if (op == Opcode::Ret) {
                if (in->num_operands() == 1) {
                    const unsigned w = int_width(in->operand(0));
                    const unsigned t = type_width(in->type());
                    // The instruction's OWN stored type is what to_string()
                    // prints ("ret iN %x"); it must match its operand, or
                    // printing silently disagrees with what is returned
                    // (e.g. a candidate built by copying a pattern's
                    // authored type instead of deriving it from the
                    // rebound operand). Checking only against the
                    // function's declared return width is not enough: it
                    // would miss exactly this case whenever the operand
                    // happens to already be right and only the
                    // instruction's own type field is stale.
                    if (w && t && w != t)
                        return where + ": ret's own type is i" + std::to_string(t) + " but its operand is i" + std::to_string(w);
                    if (w && ret_w && w != ret_w)
                        return "ret i" + std::to_string(w) + " in a function returning i" + std::to_string(ret_w);
                }
            }

            // ── SSA dominance ───────────────────────────────────────────
            if (!reachable[b]) continue;
            if (op == Opcode::Phi) {
                const auto incoming = phi_blocks(*in);
                if (incoming.size() != in->num_operands())
                    return where + ": phi has " + std::to_string(in->num_operands()) +
                           " incoming value(s) but " + std::to_string(incoming.size()) + " incoming block(s)";
                for (const auto& blk : incoming)
                    if (!bidx.count(blk)) return where + ": phi names unknown block '" + blk + "'";
                for (size_t k = 0; k < in->num_operands(); ++k) {
                    const auto& v = in->operand(k);
                    if (!v || !v->has_name() || v->is_global() || k >= incoming.size()) continue;
                    auto ib = bidx.find(incoming[k]);
                    if (ib == bidx.end() || !reachable[ib->second]) continue;
                    if (!dominated(v->name(), ib->second, blocks[ib->second]->size()))
                        return where + ": phi operand %" + v->name() + " does not dominate its incoming edge";
                }
                continue;
            }
            for (const auto& v : in->operands()) {
                if (!v || !v->has_name() || v->is_global()) continue;   // @globals are not SSA values of this function
                if (in->is_terminator() && bidx.count(v->name()) && !defs.count(v->name())) continue;  // a label
                if (!dominated(v->name(), b, p))
                    return where + ": operand %" + v->name() + " is undefined or does not dominate its use";
            }
        }
    }
    return "";
}

} // namespace clunk::ir
