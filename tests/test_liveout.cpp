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
 * LiveOut tests: hand-derived cases, then a cross-check against the
 * textbook PATH definition of liveness ("some path from here reaches a
 * use") on randomly generated diamond-chain CFGs — independent of the
 * dataflow implementation it checks.
 */
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "clunk/Analysis/LiveOut.h"
#include "clunk/Parser/IRParser.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;
using Names = std::set<std::string>;

static std::shared_ptr<ir::Function> fn(const std::string& src) {
    parser::IRParser p;
    return p.parse_string(src.c_str())->function("f");
}

static std::string show(const Names& s) {
    std::string r = "{";
    for (auto& n : s) r += (r.size() > 1 ? ", " : "") + n;
    return r + "}";
}

#define CHECK_SET(actual, expected, msg) do { \
    Names a_ = (actual), e_ = expected; \
    CHECK(a_ == e_, std::string(msg) + ": got " + show(a_) + ", want " + show(e_)); \
} while(0)

// ── hand-derived cases ─────────────────────────────────────────────────────
static void test_straight_line_and_arguments() {
    auto f = fn(R"(
define i32 @f(i32 %a, i32 %b) {
entry:
  %x = add i32 %a, %b
  %y = mul i32 %x, 2
  br label %next
next:
  %z = add i32 %y, %a
  ret i32 %z
}
)");
    auto info = analysis::compute_live_out(*f);
    CHECK_SET(info.live_in["entry"], (Names{"a", "b"}), "entry live-in is the arguments it reads");
    CHECK_SET(info.live_out["entry"], (Names{"a", "y"}), "x is consumed inside entry; y and a escape");
    CHECK_SET(info.live_in["next"], (Names{"a", "y"}), "next reads y and a");
    CHECK_SET(info.live_out["next"], (Names{}), "nothing is live after the return");
}

static void test_diamond_phi_is_per_edge() {
    auto f = fn(R"(
define i32 @f(i32 %a, i32 %b) {
entry:
  %c = icmp eq i32 %a, 0
  %e = add i32 %a, %b
  br i1 %c, label %t, label %el
t:
  %x = add i32 %e, 1
  br label %m
el:
  %y = sub i32 %e, 2
  br label %m
m:
  %p = phi i32 [ %x, %t ], [ %y, %el ]
  %r = add i32 %p, %b
  ret i32 %r
}
)");
    auto info = analysis::compute_live_out(*f);
    CHECK_SET(info.live_out["t"], (Names{"x", "b"}), "x flows to the phi from t; b is used in m");
    CHECK_SET(info.live_out["el"], (Names{"y", "b"}), "y flows to the phi from el");
    CHECK(!info.is_live_out("t", "y"), "y is NOT live out of t (the phi takes it only from el)");
    CHECK(!info.is_live_out("el", "x"), "x is NOT live out of el");
    CHECK_SET(info.live_in["m"], (Names{"b"}), "the phi result is defined in m, so it is not live-in");
    CHECK_SET(info.live_out["entry"], (Names{"e", "b"}), "e and b flow into both arms");
}

static void test_loop_carried_phi() {
    auto f = fn(R"(
define i32 @f(i32 %n, i32 %a) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi i32 [ %a, %entry ], [ %acc2, %loop ]
  %inext = add i32 %i, 1
  %acc2 = add i32 %acc, %i
  %c = icmp ult i32 %inext, %n
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %acc2
}
)");
    auto info = analysis::compute_live_out(*f);
    CHECK_SET(info.live_out["entry"], (Names{"a", "n"}), "the initial phi value a and the bound n leave entry");
    CHECK_SET(info.live_out["loop"], (Names{"n", "inext", "acc2"}), "the back-edge values, the bound, and the exit value");
    CHECK_SET(info.live_in["loop"], (Names{"n"}), "phi results are defined at entry; only n is live-in");
    CHECK_SET(info.live_in["exit"], (Names{"acc2"}), "exit reads acc2");
}

static void test_unused_and_range_live_outs() {
    auto f = fn(R"(
define i32 @f(i32 %a, i32 %b) {
entry:
  %dead = add i32 %a, 1
  %t1 = mul i32 %a, %b
  %t2 = add i32 %t1, 7
  %t3 = xor i32 %t2, %a
  %t4 = add i32 %t3, %t1
  br label %next
next:
  %u = add i32 %t2, 1
  ret i32 %u
}
)");
    auto info = analysis::compute_live_out(*f);
    CHECK(!info.is_live_out("entry", "dead"), "an unused value is live nowhere");
    CHECK(!info.is_live_out("entry", "t4"), "t4 is never read");
    // Range [1,4) = t1, t2, t3: t1 is read by t4 (after the range), t2 by the next block; t3 only by t4.
    auto outs = analysis::range_live_outs(*f, info, "entry", 1, 4);
    CHECK((outs == std::vector<std::string>{"t1", "t2", "t3"}), "all three escape the range (t3 via t4)");
    // Range [1,3) = t1, t2: t1 is used by t3/t4 after, t2 by t3 and by the next block.
    auto outs2 = analysis::range_live_outs(*f, info, "entry", 1, 3);
    CHECK((outs2 == std::vector<std::string>{"t1", "t2"}), "t1 and t2 escape [1,3)");
    // A range whose values are consumed inside it defines no live-outs.
    auto outs3 = analysis::range_live_outs(*f, info, "entry", 0, 5);
    CHECK((outs3 == std::vector<std::string>{"t2"}), "over the whole block only t2 (read by next) escapes: got " + std::to_string(outs3.size()));
    CHECK(analysis::range_live_outs(*f, info, "nope", 0, 1).empty(), "an unknown block yields nothing");
}

static void test_globals_and_constants_are_not_values() {
    auto f = fn(R"(
@g = global i32 0
define i32 @f(i32 %a) {
entry:
  %x = add i32 %a, 5
  br label %next
next:
  %y = add i32 %x, 1
  ret i32 %y
}
)");
    auto info = analysis::compute_live_out(*f);
    CHECK_SET(info.live_out["entry"], (Names{"x"}), "only SSA values are tracked");
}

// ── brute-force cross-check ────────────────────────────────────────────────
static std::vector<std::string> phi_blocks_of(const ir::Instruction& phi) {
    std::vector<std::string> out;
    auto it = phi.metadata().find("phi_blocks");
    if (it == phi.metadata().end()) return out;
    std::string cur;
    for (char c : it->second) { if (c == ',') { out.push_back(cur); cur.clear(); } else cur.push_back(c); }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// The textbook definition: v is live-out of block b iff some path leaving b
// reaches a use of v that is not preceded by a (re)definition of v.
static bool bf_live_out(const ir::Function& f, const std::string& b, const std::string& v) {
    std::unordered_map<std::string, const ir::BasicBlock*> by_name;
    for (auto& bb : f.blocks()) by_name[bb->name()] = bb.get();
    std::set<std::pair<std::string, std::string>> seen;   // (block, arrived-from)
    std::vector<std::pair<std::string, std::string>> work;
    for (auto& s : by_name[b]->successors()) work.push_back({s, b});
    while (!work.empty()) {
        auto [s, from] = work.back(); work.pop_back();
        if (!seen.insert({s, from}).second) continue;
        const ir::BasicBlock* bb = by_name[s];
        bool defined_here = false;
        // phis first: they read their operand on the edge from `from`
        for (auto& in : bb->instructions()) {
            if (in->opcode() != ir::Opcode::Phi) continue;
            auto blocks = phi_blocks_of(*in);
            for (size_t k = 0; k < in->num_operands() && k < blocks.size(); ++k)
                if (blocks[k] == from && in->operand(k)->has_name() && in->operand(k)->name() == v) return true;
            if (in->name() == v) defined_here = true;
        }
        for (auto& in : bb->instructions()) {
            if (in->opcode() == ir::Opcode::Phi) continue;
            for (auto& op : in->operands())
                if (op && op->has_name() && !op->is_global() && op->name() == v && !defined_here) return true;
            if (in->has_name() && in->name() == v) defined_here = true;
        }
        if (defined_here) continue;   // v is (re)defined in s: the older instance dies here
        for (auto& nxt : bb->successors()) work.push_back({nxt, s});
    }
    return false;
}

// A chain of `depth` diamonds; each value only uses values that dominate it.
static std::string random_diamond_chain(std::mt19937_64& rng, int depth) {
    std::ostringstream o;
    std::vector<std::string> visible = {"a", "b"};
    int n = 0;
    auto fresh = [&] { return "v" + std::to_string(n++); };
    auto pick = [&](const std::vector<std::string>& pool) { return "%" + pool[rng() % pool.size()]; };
    o << "define i32 @f(i32 %a, i32 %b) {\nentry:\n";
    std::string cur = "entry";
    for (int d = 0; d < depth; ++d) {
        for (int k = rng() % 3; k >= 0; --k) {   // straight-line work in the dominating block
            auto nm = fresh();
            o << "  %" << nm << " = add i32 " << pick(visible) << ", " << pick(visible) << "\n";
            visible.push_back(nm);
        }
        auto c = fresh();
        o << "  %" << c << " = icmp eq i32 " << pick(visible) << ", 0\n";
        const std::string T = "t" + std::to_string(d), F = "f" + std::to_string(d), M = "m" + std::to_string(d);
        o << "  br i1 %" << c << ", label %" << T << ", label %" << F << "\n";
        std::vector<std::string> arm_out[2];
        const std::string arm_name[2] = {T, F};
        for (int a = 0; a < 2; ++a) {
            o << arm_name[a] << ":\n";
            std::vector<std::string> arm = visible;
            for (int k = rng() % 3; k >= 0; --k) {
                auto nm = fresh();
                o << "  %" << nm << " = sub i32 " << pick(arm) << ", " << pick(arm) << "\n";
                arm.push_back(nm);
            }
            arm_out[a] = arm;
            o << "  br label %" << M << "\n";
        }
        o << M << ":\n";
        auto p = fresh();
        o << "  %" << p << " = phi i32 [ " << pick(arm_out[0]) << ", %" << T << " ], [ " << pick(arm_out[1]) << ", %" << F << " ]\n";
        visible.push_back(p);
        cur = M;
    }
    o << "  %r = add i32 " << pick(visible) << ", " << pick(visible) << "\n  ret i32 %r\n}\n";
    return o.str();
}

static void test_bruteforce_agreement() {
    std::mt19937_64 rng(2024);
    size_t compared = 0, live = 0;
    for (int t = 0; t < 150; ++t) {
        const std::string src = random_diamond_chain(rng, 1 + rng() % 4);
        auto f = fn(src);
        if (!f) { CHECK(false, "generated IR must parse:\n" + src); continue; }
        auto info = analysis::compute_live_out(*f);
        std::set<std::string> values = {"a", "b"};
        for (auto& bb : f->blocks()) for (auto& in : bb->instructions()) if (in->has_name()) values.insert(in->name());
        bool ok = true;
        for (auto& bb : f->blocks())
            for (auto& v : values) {
                const bool got = info.is_live_out(bb->name(), v), want = bf_live_out(*f, bb->name(), v);
                ++compared; live += want;
                if (got != want) {
                    ok = false;
                    std::cerr << "MISMATCH block=" << bb->name() << " value=" << v << " got=" << got << " want=" << want << "\n" << src;
                }
            }
        CHECK(ok, "dataflow liveness equals path liveness on generated CFG #" + std::to_string(t));
    }
    CHECK(compared > 5000, "the cross-check compared many (block, value) pairs: " + std::to_string(compared));
    CHECK(live > 500, "and a meaningful number of them were live (not vacuous): " + std::to_string(live));

    // The loop shapes too.
    for (const char* src : {R"(
define i32 @f(i32 %n, i32 %a) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi i32 [ %a, %entry ], [ %acc2, %loop ]
  %inext = add i32 %i, 1
  %acc2 = add i32 %acc, %i
  %c = icmp ult i32 %inext, %n
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %acc2
}
)", R"(
define i32 @f(i32 %n, i32 %a) {
entry:
  %k = mul i32 %a, 3
  br label %outer
outer:
  %i = phi i32 [ 0, %entry ], [ %inext, %latch ]
  br label %inner
inner:
  %j = phi i32 [ 0, %outer ], [ %jnext, %inner ]
  %jnext = add i32 %j, %k
  %cj = icmp ult i32 %jnext, %n
  br i1 %cj, label %inner, label %latch
latch:
  %inext = add i32 %i, 1
  %ci = icmp ult i32 %inext, %n
  br i1 %ci, label %outer, label %exit
exit:
  %r = add i32 %jnext, %inext
  ret i32 %r
}
)"}) {
        auto f = fn(src);
        auto info = analysis::compute_live_out(*f);
        std::set<std::string> values = {"n", "a"};
        for (auto& bb : f->blocks()) for (auto& in : bb->instructions()) if (in->has_name()) values.insert(in->name());
        bool ok = true;
        for (auto& bb : f->blocks()) for (auto& v : values)
            if (info.is_live_out(bb->name(), v) != bf_live_out(*f, bb->name(), v)) {
                ok = false; std::cerr << "LOOP MISMATCH block=" << bb->name() << " value=" << v << "\n";
            }
        CHECK(ok, "liveness equals path liveness on a loop nest");
    }
}

int main() {
    std::cerr << "test_liveout: SSA liveness and range live-outs\n";
    test_straight_line_and_arguments();
    test_diamond_phi_is_per_edge();
    test_loop_carried_phi();
    test_unused_and_range_live_outs();
    test_globals_and_constants_are_not_values();
    test_bruteforce_agreement();
    std::cerr << "passed " << g_pass << ", failed " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
