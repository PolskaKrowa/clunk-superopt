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
 * Clunk LiveOut — see the header for the dataflow equations.
 */
#include "clunk/Analysis/LiveOut.h"

#include <algorithm>
#include <unordered_set>

#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Instruction.h"

namespace clunk::analysis {

namespace {

// Incoming block names of a phi ("phi_blocks" metadata, comma separated).
std::vector<std::string> phi_blocks(const ir::Instruction& phi) {
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

} // namespace

LiveOutInfo compute_live_out(const ir::Function& fn) {
    LiveOutInfo info;
    const auto& blocks = fn.blocks();
    const size_t n = blocks.size();

    // Every SSA value of this function: arguments and named instructions.
    std::unordered_set<std::string> values;
    for (const auto& a : fn.arguments()) if (!a.name.empty()) values.insert(a.name);
    for (const auto& b : blocks)
        if (b) for (const auto& i : b->instructions())
            if (i && i->has_name()) values.insert(i->name());
    auto tracked = [&](const std::shared_ptr<ir::Value>& v) {
        return v && v->has_name() && !v->is_global() && values.count(v->name()) != 0;
    };

    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < n; ++i) if (blocks[i]) index[blocks[i]->name()] = i;

    std::vector<std::vector<size_t>> succ(n), pred(n);
    for (size_t b = 0; b < n; ++b) {
        if (!blocks[b]) continue;
        for (const auto& s : blocks[b]->successors()) {
            auto it = index.find(s);
            if (it == index.end()) continue;
            succ[b].push_back(it->second);
            pred[it->second].push_back(b);
        }
    }

    // Per block: definitions, upward-exposed non-phi uses, and the values
    // each phi takes from a given predecessor.
    std::vector<std::set<std::string>> def(n), use(n);
    std::vector<std::unordered_map<size_t, std::set<std::string>>> phi_from(n);   // block -> pred -> names
    for (size_t b = 0; b < n; ++b) {
        if (!blocks[b]) continue;
        for (const auto& in : blocks[b]->instructions()) {
            if (!in) continue;
            if (in->opcode() == ir::Opcode::Phi) {
                const auto incoming = phi_blocks(*in);
                for (size_t k = 0; k < in->num_operands(); ++k) {
                    const auto& v = in->operand(k);
                    if (!tracked(v)) continue;
                    if (k < incoming.size() && index.count(incoming[k])) {
                        phi_from[b][index[incoming[k]]].insert(v->name());
                    } else {
                        for (size_t p : pred[b]) phi_from[b][p].insert(v->name());   // unknown edge: conservative
                    }
                }
            } else {
                for (const auto& v : in->operands())
                    if (tracked(v) && !def[b].count(v->name())) use[b].insert(v->name());
            }
            if (in->has_name()) def[b].insert(in->name());
        }
    }

    // Backward fixpoint (reverse layout order converges quickly).
    std::vector<std::set<std::string>> in_set(n), out_set(n);
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t b = n; b-- > 0;) {
            if (!blocks[b]) continue;
            std::set<std::string> out;
            for (size_t s : succ[b]) {
                out.insert(in_set[s].begin(), in_set[s].end());
                auto pf = phi_from[s].find(b);
                if (pf != phi_from[s].end()) out.insert(pf->second.begin(), pf->second.end());
            }
            std::set<std::string> in = use[b];
            for (const auto& v : out) if (!def[b].count(v)) in.insert(v);
            if (out != out_set[b] || in != in_set[b]) {
                out_set[b] = std::move(out);
                in_set[b] = std::move(in);
                changed = true;
            }
        }
    }
    for (size_t b = 0; b < n; ++b) {
        if (!blocks[b]) continue;
        info.live_in[blocks[b]->name()] = std::move(in_set[b]);
        info.live_out[blocks[b]->name()] = std::move(out_set[b]);
    }
    return info;
}

std::vector<std::string> range_live_outs(const ir::Function& fn, const LiveOutInfo& info,
                                         const std::string& block, size_t first, size_t last) {
    std::vector<std::string> result;
    const ir::BasicBlock* bb = nullptr;
    for (const auto& b : fn.blocks()) if (b && b->name() == block) { bb = b.get(); break; }
    if (!bb) return result;
    const auto& insts = bb->instructions();
    last = std::min(last, insts.size());

    // Names used by the instructions after the range (phis excluded: their
    // operands are accounted for per edge, in live_out).
    std::unordered_set<std::string> used_after;
    for (size_t i = last; i < insts.size(); ++i) {
        if (!insts[i] || insts[i]->opcode() == ir::Opcode::Phi) continue;
        for (const auto& v : insts[i]->operands())
            if (v && v->has_name() && !v->is_global()) used_after.insert(v->name());
    }
    for (size_t i = first; i < last; ++i) {
        if (!insts[i] || !insts[i]->has_name()) continue;
        const std::string& name = insts[i]->name();
        if (used_after.count(name) || info.is_live_out(block, name)) result.push_back(name);
    }
    return result;
}

} // namespace clunk::analysis
