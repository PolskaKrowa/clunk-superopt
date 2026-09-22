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
 * DifferentialScreen tests: input coverage, refinement-correct verdicts,
 * and the guarantee that only survivors reach the solver.
 */
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>

#include "clunk/Evaluator/DifferentialScreen.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Parser/IRParser.h"
#include "clunk/Search/SMTVerifier.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;
using evaluator::DifferentialScreen;
using evaluator::ScreenVerdict;

static std::shared_ptr<ir::Function> fn(const std::string& src, const char* name = "f") {
    parser::IRParser p;
    auto m = p.parse_string(src.c_str());
    return m->function(name);
}

// Screen `orig_body` against `cand_body`, both `define i32 @f(i32 %x)` bodies.
static evaluator::ScreenResult screen32(const std::string& orig_body, const std::string& cand_body) {
    auto o = fn("define i32 @f(i32 %x) {\nentry:\n" + orig_body + "}\n");
    auto c = fn("define i32 @f(i32 %x) {\nentry:\n" + cand_body + "}\n");
    return DifferentialScreen().screen(*o, *c);
}

// ── input generation ───────────────────────────────────────────────────────
static void test_edge_values() {
    auto e8 = DifferentialScreen::edge_values(8);
    std::set<int64_t> s8(e8.begin(), e8.end());
    CHECK(s8.size() == e8.size(), "no duplicate edge values");
    for (int64_t v : {0, 1, -1, -128, 127, 2, 4, 8, 16, 32, 64, -2, -4, -8, -16, -32, -64, 3, 7, 15, 31, 63})
        CHECK(s8.count(v), "i8 edge set contains " + std::to_string(v));
    CHECK(s8.count(8) && s8.count(7), "the shift-amount boundaries w-1 and w are present");
    CHECK(std::all_of(e8.begin(), e8.end(), [](int64_t v) { return v >= -128 && v <= 127; }),
          "every i8 edge value is canonical for i8");
    CHECK(e8.front() == 0 && e8[1] == 1 && e8[2] == -1, "the most valuable values come first");

    auto e64 = DifferentialScreen::edge_values(64);
    std::set<int64_t> s64(e64.begin(), e64.end());
    CHECK(s64.count(INT64_MIN) && s64.count(INT64_MAX), "i64 edge set has INT_MIN and INT_MAX");
    CHECK(s64.count(int64_t{1} << 62) && s64.count(-(int64_t{1} << 62)), "i64 edge set has 2^62 and -2^62");
    CHECK(s64.count((int64_t{1} << 40) - 1), "i64 edge set has low-bit masks");

    auto e1 = DifferentialScreen::edge_values(1);
    CHECK(e1.size() == 2 && s64.count(0), "i1 has exactly {0, -1}");
}

static void test_make_inputs() {
    auto f = fn("define i32 @f(i8 %a, i32 %b, i64 %c) {\nentry:\n  ret i32 %b\n}\n");
    DifferentialScreen sc;
    auto in = sc.make_inputs(*f);
    CHECK(!in.empty() && in.size() <= 384, "respects the default cap: " + std::to_string(in.size()));
    bool canon = true, has_int_min32 = false, has_int_min8 = false, has_pow2 = false, has_minus1_all = false;
    for (auto& v : in) {
        if (v.size() != 3) { canon = false; continue; }
        if (v[0] < -128 || v[0] > 127) canon = false;
        if (v[1] < INT32_MIN || v[1] > INT32_MAX) canon = false;
        has_int_min8 |= v[0] == -128;
        has_int_min32 |= v[1] == INT32_MIN;
        has_pow2 |= v[1] == 4096;
        has_minus1_all |= v[0] == -1 && v[1] == -1 && v[2] == -1;
    }
    CHECK(canon, "every value is canonical for its own argument's width");
    CHECK(has_int_min8 && has_int_min32, "INT_MIN of each width appears");
    CHECK(has_pow2, "powers of two appear");
    CHECK(has_minus1_all, "the all -1 vector appears");
    CHECK(sc.make_inputs(*f) == in, "generation is deterministic");

    evaluator::ScreenConfig other; other.seed = 12345;
    CHECK(DifferentialScreen(other).make_inputs(*f) != in, "a different seed changes the random tail");

    evaluator::ScreenConfig tiny; tiny.max_vectors = 8;
    auto small = DifferentialScreen(tiny).make_inputs(*f);
    CHECK(small.size() <= 8, "max_vectors is a hard cap");
    CHECK(std::any_of(small.begin(), small.end(), [](auto& v) { return v[0] == 0 && v[1] == 0 && v[2] == 0; }),
          "even a tiny budget starts with edge cases (all zero)");

    auto nullary = fn("define i32 @f() {\nentry:\n  ret i32 7\n}\n");
    CHECK(sc.make_inputs(*nullary).size() == 1, "a function with no arguments gets exactly one (empty) vector");
}

// ── verdicts ───────────────────────────────────────────────────────────────
static void test_equivalent_survives() {
    auto r = screen32("  %r = mul i32 %x, 2\n  ret i32 %r\n", "  %r = shl i32 %x, 1\n  ret i32 %r\n");
    CHECK(r.verdict == ScreenVerdict::Survived, "x*2 vs x<<1 survives: " + r.reason);
    CHECK(r.compared > 100, "a meaningful number of inputs were compared: " + std::to_string(r.compared));
}

static void test_edge_case_bugs_are_caught() {
    // Wrong on exactly one input each. Uniform random 32-bit testing would
    // essentially never hit any of these; the edge set finds all of them.
    struct Case { const char* name; const char* cand_cmp; int64_t bad; };
    const Case cases[] = {
        {"INT_MIN", "icmp eq i32 %x, -2147483648", INT32_MIN},
        {"a power of two", "icmp eq i32 %x, 4096", 4096},
        {"-1", "icmp eq i32 %x, -1", -1},
        {"0", "icmp eq i32 %x, 0", 0},
        {"INT_MAX", "icmp eq i32 %x, 2147483647", INT32_MAX},
        {"a negated power of two", "icmp eq i32 %x, -1024", -1024},
        {"a low-bit mask", "icmp eq i32 %x, 255", 255},
    };
    for (const auto& c : cases) {
        auto r = screen32("  ret i32 %x\n",
                          std::string("  %c = ") + c.cand_cmp + "\n  %r = select i1 %c, i32 99, i32 %x\n  ret i32 %r\n");
        CHECK(r.verdict == ScreenVerdict::Rejected, std::string("caught the bug at ") + c.name);
        CHECK(r.counterexample.size() == 1 && r.counterexample[0] == c.bad,
              std::string("counterexample is the offending input for ") + c.name);
    }
    auto o = fn("define i64 @f(i64 %x) {\nentry:\n  ret i64 %x\n}\n");
    auto c = fn("define i64 @f(i64 %x) {\nentry:\n  %c = icmp eq i64 %x, -9223372036854775808\n  %r = select i1 %c, i64 1, i64 %x\n  ret i64 %r\n}\n");
    auto r = DifferentialScreen().screen(*o, *c);
    CHECK(r.verdict == ScreenVerdict::Rejected && r.counterexample[0] == INT64_MIN, "caught an INT64_MIN-only bug");
}

static void test_refinement_semantics() {
    // Candidate MAY be more defined than the original...
    auto r = screen32("  %r = shl i32 %x, %x\n  ret i32 %r\n",                         // poison for x >= 32
                      "  %m = and i32 %x, 31\n  %r = shl i32 %x, %m\n  ret i32 %r\n"); // defined there, equal elsewhere? no: x&31 != x for x<32? equal when x<32
    // (For x in [0,31] the two agree; for x >= 32 the original is poison so the candidate is free.)
    CHECK(r.verdict == ScreenVerdict::Survived, "a candidate may replace poison with a value: " + r.reason);

    // ...but never LESS defined.
    auto r2 = screen32("  %m = and i32 %x, 31\n  %r = shl i32 %x, %m\n  ret i32 %r\n",
                       "  %r = shl i32 %x, %x\n  ret i32 %r\n");
    CHECK(r2.verdict == ScreenVerdict::Rejected, "a candidate must not introduce poison: " + r2.reason);
    CHECK(r2.reason.find("poison") != std::string::npos, "the reason says poison: " + r2.reason);

    auto o8 = fn("define i8 @f(i8 %x) {\nentry:\n  %r = add nsw i8 %x, 1\n  ret i8 %r\n}\n");
    auto c8 = fn("define i8 @f(i8 %x) {\nentry:\n  %r = add i8 %x, 1\n  ret i8 %r\n}\n");
    CHECK(DifferentialScreen().screen(*o8, *c8).verdict == ScreenVerdict::Survived, "dropping nsw is a valid refinement");
    auto r3 = DifferentialScreen().screen(*c8, *o8);
    CHECK(r3.verdict == ScreenVerdict::Rejected, "adding nsw is not (poison at 127): " + r3.reason);
    CHECK(r3.counterexample.size() == 1 && r3.counterexample[0] == 127, "the counterexample is INT8_MAX");

    // UB in the original frees the candidate.
    auto ud = fn("define i32 @f(i32 %x, i32 %y) {\nentry:\n  %r = udiv i32 %x, %y\n  ret i32 %r\n}\n");
    auto uc = fn("define i32 @f(i32 %x, i32 %y) {\nentry:\n  %z = icmp eq i32 %y, 0\n  %d = select i1 %z, i32 1, i32 %y\n  %r = udiv i32 %x, %d\n  ret i32 %r\n}\n");
    CHECK(DifferentialScreen().screen(*ud, *uc).verdict == ScreenVerdict::Survived, "udiv by zero is UB: the candidate may do anything there");
    auto sd = fn("define i32 @f(i32 %x, i32 %y) {\nentry:\n  %r = sdiv i32 %x, %y\n  ret i32 %r\n}\n");
    auto sc = fn("define i32 @f(i32 %x, i32 %y) {\nentry:\n  %r = sdiv i32 %x, %y\n  %s = add i32 %r, 0\n  ret i32 %s\n}\n");
    CHECK(DifferentialScreen().screen(*sd, *sc).verdict == ScreenVerdict::Survived, "sdiv INT_MIN, -1 is UB in both: skipped, not compared");
}

static void test_inconclusive_cases() {
    // No comparable input: the original is poison everywhere.
    auto always_poison = fn("define i8 @f(i8 %x) {\nentry:\n  %r = shl i8 %x, 9\n  ret i8 %r\n}\n");
    auto anything = fn("define i8 @f(i8 %x) {\nentry:\n  ret i8 0\n}\n");
    auto r = DifferentialScreen().screen(*always_poison, *anything);
    CHECK(r.verdict == ScreenVerdict::Inconclusive, "always-poison original: nothing to compare: " + r.reason);

    // Different signatures / return widths.
    auto one = fn("define i32 @f(i32 %x) {\nentry:\n  ret i32 %x\n}\n");
    auto two = fn("define i32 @f(i32 %x, i32 %y) {\nentry:\n  ret i32 %x\n}\n");
    CHECK(DifferentialScreen().screen(*one, *two).verdict == ScreenVerdict::Inconclusive, "different arity: leave it to the prover");
    auto wide = fn("define i64 @f(i32 %x) {\nentry:\n  %r = zext i32 %x to i64\n  ret i64 %r\n}\n");
    CHECK(DifferentialScreen().screen(*one, *wide).verdict == ScreenVerdict::Inconclusive, "different return widths: leave it to the prover");

    // Void: nothing observable to compare.
    auto v = fn("define void @f(i32 %x) {\nentry:\n  ret void\n}\n");
    CHECK(DifferentialScreen().screen(*v, *v).verdict == ScreenVerdict::Inconclusive, "void return");

    // Loops are left to the prover (they can burn the whole hop budget per input).
    auto loop = fn(R"(
define i32 @f(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %inext = add i32 %i, 1
  %c = icmp ult i32 %inext, %n
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %inext
}
)");
    auto r2 = DifferentialScreen().screen(*loop, *loop);
    CHECK(r2.verdict == ScreenVerdict::Inconclusive && r2.reason.find("loop") != std::string::npos, "loops: " + r2.reason);

    // Memory through a pointer parameter is not modelled.
    auto mem = fn("define i32 @f(ptr %p) {\nentry:\n  %v = load i32, ptr %p\n  ret i32 %v\n}\n");
    CHECK(DifferentialScreen().screen(*mem, *mem).verdict == ScreenVerdict::Inconclusive, "pointer parameter");
}

static void test_control_flow_and_narrow_types() {
    // Branchy, mixed widths, an i1 result: original vs a correct rewrite.
    auto o = fn(R"(
define i16 @f(i8 %a, i16 %b) {
entry:
  %c = icmp slt i8 %a, 0
  br i1 %c, label %neg, label %pos
neg:
  %s = sext i8 %a to i16
  %r1 = sub i16 %b, %s
  br label %done
pos:
  %z = zext i8 %a to i16
  %r2 = add i16 %b, %z
  br label %done
done:
  %r = phi i16 [ %r1, %neg ], [ %r2, %pos ]
  ret i16 %r
}
)");
    auto same = DifferentialScreen().screen(*o, *o);
    CHECK(same.verdict == ScreenVerdict::Survived, "a function refines itself: " + same.reason);
    auto broken = fn(R"(
define i16 @f(i8 %a, i16 %b) {
entry:
  %s = sext i8 %a to i16
  %r = add i16 %b, %s
  ret i16 %r
}
)");
    auto r = DifferentialScreen().screen(*o, *broken);
    CHECK(r.verdict == ScreenVerdict::Rejected, "the wrong rewrite is caught: " + r.reason);
}

static void test_reproducible() {
    auto a = screen32("  ret i32 %x\n", "  %c = icmp eq i32 %x, 7\n  %r = select i1 %c, i32 0, i32 %x\n  ret i32 %r\n");
    auto b = screen32("  ret i32 %x\n", "  %c = icmp eq i32 %x, 7\n  %r = select i1 %c, i32 0, i32 %x\n  ret i32 %r\n");
    CHECK(a.verdict == b.verdict && a.counterexample == b.counterexample && a.reason == b.reason, "same inputs, same verdict");
}

// ── SMT gateway ────────────────────────────────────────────────────────────
static void test_smt_gateway_only_survivors() {
    auto orig = fn("define i32 @f(i32 %x) {\nentry:\n  ret i32 %x\n}\n");
    auto bad  = fn("define i32 @f(i32 %x) {\nentry:\n  %c = icmp eq i32 %x, -2147483648\n  %r = select i1 %c, i32 1, i32 %x\n  ret i32 %r\n}\n");
    auto good = fn("define i32 @f(i32 %x) {\nentry:\n  %r = add i32 %x, 0\n  ret i32 %r\n}\n");

    search::SMTVerifier v;
    auto rb = v.verify(*orig, *bad);
    CHECK(rb.status == search::VerificationResult::NotEquivalent, "the bad candidate is NotEquivalent");
    CHECK(rb.screened, "...decided by the screen, not the solver");
    CHECK(rb.counterexample.size() == 1 && rb.counterexample[0] == INT32_MIN, "with the real counterexample");
    CHECK(v.stats().screened_out == 1, "screened_out counted");

    auto rg = v.verify(*orig, *good);
    CHECK(!rg.screened, "a survivor is not screened out (it goes on to the solver)");
    CHECK(v.stats().screen_passed == 1, "screen_passed counted");

    // The screen can be switched off.
    search::SMTConfig cfg; cfg.screen_before_solving = false;
    search::SMTVerifier off(cfg);
    CHECK(!off.verify(*orig, *bad).screened, "with the screen disabled nothing is screened");
    CHECK(off.stats().screened_out == 0, "and the counter stays zero");

    // Batch API: screened where rejected, and aligned with its inputs.
    std::vector<ir::Function> cands = {*good, *bad, *good};
    search::SMTVerifier bv;
    auto rs = bv.verify_batch(*orig, cands);
    CHECK(rs.size() == 3, "batch returns one result per candidate");
    if (rs.size() == 3) {
        CHECK(!rs[0].screened && rs[1].screened && !rs[2].screened, "only the bad candidate was screened out");
        CHECK(rs[1].status == search::VerificationResult::NotEquivalent, "and it is NotEquivalent");
    }
}

static void test_assumptions_are_exempt() {
    // Differs from the original only where x == 5; under the assumption
    // x != 5 that is irrelevant, so the screen must NOT reject it.
    auto orig = fn("define i32 @f(i32 %x) {\nentry:\n  ret i32 %x\n}\n");
    auto cand = fn("define i32 @f(i32 %x) {\nentry:\n  %c = icmp eq i32 %x, 5\n  %r = select i1 %c, i32 0, i32 %x\n  ret i32 %r\n}\n");
    search::ArgAssumption a;
    a.predicate = ir::CmpPredicate::NE; a.lhs_arg = 0; a.rhs_arg = -1; a.rhs_const = 5;
    search::SMTVerifier v;
    auto r = v.verify_with_assumptions(*orig, *cand, {a});
    CHECK(!r.screened, "verify_with_assumptions never uses the screen (random inputs violate the path condition)");
    CHECK(v.stats().screened_out == 0, "and nothing was counted");
}

// ── StochasticSearch pre-filter now shares the screen ──────────────────────
static void test_int_min_only_bug_reaches_the_prefilter_too() {
    // (The old private probe table had INT32_MIN but not a width-aware INT8_MIN or powers of two.)
    auto orig = fn("define i8 @f(i8 %x) {\nentry:\n  ret i8 %x\n}\n");
    auto cand = fn("define i8 @f(i8 %x) {\nentry:\n  %c = icmp eq i8 %x, 64\n  %r = select i1 %c, i8 0, i8 %x\n  ret i8 %r\n}\n");
    evaluator::ScreenConfig cfg; cfg.max_vectors = 32;   // STOKE's recommended count
    CHECK(DifferentialScreen(cfg).screen(*orig, *cand).verdict == ScreenVerdict::Rejected,
          "even at STOKE's 32 vectors the power-of-two bug is found");
}

int main() {
    std::cerr << "test_differential_screen: edge + random concrete inputs before the solver\n";
    test_edge_values();
    test_make_inputs();
    test_equivalent_survives();
    test_edge_case_bugs_are_caught();
    test_refinement_semantics();
    test_inconclusive_cases();
    test_control_flow_and_narrow_types();
    test_reproducible();
    test_smt_gateway_only_survivors();
    test_assumptions_are_exempt();
    test_int_min_only_bug_reaches_the_prefilter_too();
    std::cerr << "passed " << g_pass << ", failed " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
