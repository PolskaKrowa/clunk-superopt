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
 * Clunk DifferentialScreen — see the header for the contract.
 */
#include "clunk/Evaluator/DifferentialScreen.h"

#include <algorithm>
#include <random>
#include <set>
#include <string>
#include <unordered_map>

#include "clunk/Evaluator/Interpreter.h"
#include "clunk/IR/BasicBlock.h"

namespace clunk::evaluator {

namespace {

// Canonical form: sign-extended from `bits`.
int64_t canon(int64_t v, unsigned bits) {
    if (bits == 0 || bits >= 64) return v;
    const uint64_t m = (uint64_t{1} << bits) - 1;
    uint64_t u = static_cast<uint64_t>(v) & m;
    if (u & (uint64_t{1} << (bits - 1))) u |= ~m;
    return static_cast<int64_t>(u);
}

// Bit-width of parameter `i`, 0 if it is not an integer.
unsigned arg_bits(const ir::Function& fn, size_t i) {
    const auto& t = fn.arguments()[i].type;
    return t && t->is_integer() ? static_cast<unsigned>(t->bit_width()) : 0;
}

// Any cycle in the CFG? (Loops can run for the interpreter's whole hop
// budget on every one of ~400 vectors, which costs more than it saves.)
bool has_cycle(const ir::Function& fn) {
    std::unordered_map<std::string, int> colour;   // 0 white, 1 grey, 2 black
    std::unordered_map<std::string, const ir::BasicBlock*> by_name;
    for (const auto& b : fn.blocks()) if (b) by_name[b->name()] = b.get();
    if (fn.blocks().empty() || !fn.blocks().front()) return false;

    struct Frame { const ir::BasicBlock* bb; std::vector<std::string> succ; size_t next; };
    std::vector<Frame> stack;
    auto push = [&](const ir::BasicBlock* bb) {
        colour[bb->name()] = 1;
        stack.push_back({bb, bb->successors(), 0});
    };
    push(fn.blocks().front().get());
    while (!stack.empty()) {
        Frame& f = stack.back();
        if (f.next == f.succ.size()) { colour[f.bb->name()] = 2; stack.pop_back(); continue; }
        const std::string s = f.succ[f.next++];
        auto it = by_name.find(s);
        if (it == by_name.end()) continue;
        const int c = colour[s];
        if (c == 1) return true;
        if (c == 0) push(it->second);
    }
    return false;
}

std::string args_to_string(const std::vector<int64_t>& a) {
    std::string s = "[";
    for (size_t i = 0; i < a.size(); ++i) s += (i ? ", " : "") + std::to_string(a[i]);
    return s + "]";
}

} // namespace

std::vector<int64_t> DifferentialScreen::edge_values(unsigned bits) {
    if (bits == 0 || bits > 64) bits = 64;
    std::vector<int64_t> out;
    std::set<int64_t> seen;
    auto add = [&](int64_t x) {
        x = canon(x, bits);
        if (seen.insert(x).second) out.push_back(x);
    };
    const uint64_t one = 1;
    const int64_t int_min = canon(static_cast<int64_t>(one << (bits - 1)), bits);
    const int64_t int_max = canon(static_cast<int64_t>((one << (bits - 1)) - 1), bits);

    // The values that break the most rewrites, first.
    add(0); add(1); add(-1); add(int_min); add(int_max); add(2); add(3); add(int_min + 1);
    // Powers of two, negated powers of two, and low-bit masks (2^k - 1).
    for (unsigned k = 2; k + 1 < bits; ++k) add(static_cast<int64_t>(one << k));
    for (unsigned k = 1; k + 1 < bits; ++k) add(-static_cast<int64_t>(one << k));
    for (unsigned k = 2; k < bits; ++k)     add(static_cast<int64_t>((one << k) - 1));
    // Shift-amount boundaries: w-1 is the largest legal shift, w is poison.
    add(static_cast<int64_t>(bits) - 1);
    add(static_cast<int64_t>(bits));
    return out;
}

std::vector<std::vector<int64_t>> DifferentialScreen::make_inputs(const ir::Function& fn) const {
    const size_t n = fn.argument_count();
    if (n == 0) return {{}};

    std::vector<unsigned> bits(n);
    std::vector<std::vector<int64_t>> edges(n);
    unsigned widest = 1;
    for (size_t i = 0; i < n; ++i) {
        bits[i] = arg_bits(fn, i);
        if (bits[i] == 0) bits[i] = 64;   // non-integer parameter: the screen refuses these anyway
        edges[i] = edge_values(bits[i]);
        widest = std::max(widest, bits[i]);
    }

    const size_t random_count = std::min(cfg_.random_vectors, cfg_.max_vectors / 3);
    const size_t edge_budget  = cfg_.max_vectors - random_count;

    std::vector<std::vector<int64_t>> out;
    std::set<std::vector<int64_t>> seen;
    auto push = [&](std::vector<int64_t> v, size_t cap) {
        if (out.size() >= cap) return;
        for (size_t i = 0; i < n; ++i) v[i] = canon(v[i], bits[i]);
        if (seen.insert(v).second) out.push_back(std::move(v));
    };

    // 1. Cross product of the core set (odometer), bounded.
    {
        std::vector<std::vector<int64_t>> core(n);
        for (size_t i = 0; i < n; ++i)
            core[i].assign(edges[i].begin(), edges[i].begin() + std::min<size_t>(6, edges[i].size()));
        std::vector<size_t> idx(n, 0);
        size_t emitted = 0;
        for (bool more = true; more && emitted < 128;) {
            std::vector<int64_t> v(n);
            for (size_t i = 0; i < n; ++i) v[i] = core[i][idx[i]];
            push(std::move(v), edge_budget);
            ++emitted;
            size_t d = 0;
            for (; d < n; ++d) {
                if (++idx[d] < core[d].size()) break;
                idx[d] = 0;
            }
            more = d < n;
        }
    }
    // 2. One edge value broadcast to every argument.
    for (int64_t e : edge_values(widest)) push(std::vector<int64_t>(n, e), edge_budget);
    // 3. Single-argument sweeps, interleaved across arguments so that a small
    //    cap still covers every argument's most valuable edge values.
    {
        size_t longest = 0;
        for (auto& e : edges) longest = std::max(longest, e.size());
        for (size_t k = 0; k < longest; ++k)
            for (size_t i = 0; i < n; ++i) {
                if (k >= edges[i].size()) continue;
                for (int64_t base : {int64_t{0}, int64_t{1}, int64_t{-1}}) {
                    std::vector<int64_t> v(n, base);
                    v[i] = edges[i][k];
                    push(std::move(v), edge_budget);
                }
            }
    }
    // 4. Random vectors in four flavours.
    std::mt19937_64 rng(cfg_.seed);
    auto random_word = [&]() -> int64_t {
        switch (rng() % 4) {
            case 0:  return static_cast<int64_t>(rng());                       // uniform bits
            case 1: {                                                          // sparse: 1-3 bits set
                uint64_t x = 0;
                for (size_t b = 1 + rng() % 3; b > 0; --b) x |= uint64_t{1} << (rng() % 64);
                return static_cast<int64_t>(x);
            }
            case 2: {                                                          // dense: complement of sparse
                uint64_t x = 0;
                for (size_t b = 1 + rng() % 3; b > 0; --b) x |= uint64_t{1} << (rng() % 64);
                return static_cast<int64_t>(~x);
            }
            default: return static_cast<int64_t>(rng() % 513) - 256;           // small magnitude
        }
    };
    const size_t cap = out.size() + random_count;
    for (size_t attempts = 0; out.size() < cap && attempts < random_count * 4; ++attempts) {
        std::vector<int64_t> v(n);
        for (auto& x : v) x = random_word();
        push(std::move(v), cap);
    }
    return out;
}

ScreenResult DifferentialScreen::screen(const ir::Function& original,
                                        const ir::Function& candidate) const {
    ScreenResult r;
    auto inconclusive = [&](std::string why) { r.verdict = ScreenVerdict::Inconclusive; r.reason = std::move(why); return r; };

    if (original.argument_count() != candidate.argument_count()) return inconclusive("different signatures");
    if (original.blocks().empty() || candidate.blocks().empty()) return inconclusive("declaration");

    // Only integer signatures are modelled, and both sides must agree on them.
    for (size_t i = 0; i < original.argument_count(); ++i) {
        const unsigned a = arg_bits(original, i), b = arg_bits(candidate, i);
        if (a == 0 || a != b) return inconclusive("non-integer or mismatched parameter");
    }
    const auto ro = original.return_type(), rc = candidate.return_type();
    if (!ro || !rc || !ro->is_integer() || !rc->is_integer() || ro->bit_width() != rc->bit_width())
        return inconclusive("void, non-integer or mismatched return type");

    if (has_cycle(original) || has_cycle(candidate)) return inconclusive("contains a loop");

    size_t unsupported = 0;
    for (const auto& in : make_inputs(original)) {
        const ExecOutcome o = Interpreter::run(original, in);
        if (o.status == ExecStatus::Unsupported) {
            ++r.skipped;
            // Unsupported is almost always structural (a call, a GEP, a float
            // op): if the first few inputs cannot be run, none can.
            if (++unsupported >= 8 && r.compared == 0) return inconclusive("original is not interpretable");
            continue;
        }
        if (o.status != ExecStatus::Value) { ++r.skipped; continue; }   // poison / UB: candidate is free

        const ExecOutcome c = Interpreter::run(candidate, in);
        if (c.status == ExecStatus::Unsupported) { ++r.skipped; continue; }
        if (c.status != ExecStatus::Value || c.value != o.value) {
            r.verdict = ScreenVerdict::Rejected;
            r.counterexample = in;
            r.reason = "input " + args_to_string(in) + ": original returns " + std::to_string(o.value) +
                       ", candidate " + (c.status == ExecStatus::Value ? "returns " + std::to_string(c.value)
                                         : c.status == ExecStatus::Poison ? std::string("returns poison")
                                                                          : std::string("has undefined behaviour"));
            return r;
        }
        ++r.compared;
    }
    if (r.compared == 0) return inconclusive("no input where the original is defined and interpretable");
    r.verdict = ScreenVerdict::Survived;
    r.reason = "agreed on " + std::to_string(r.compared) + " inputs";
    return r;
}

} // namespace clunk::evaluator
