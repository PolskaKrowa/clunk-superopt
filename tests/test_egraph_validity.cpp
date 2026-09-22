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
 * E-graph validity regressions. Each test pins one way the E-graph used to
 * emit invalid (or silently wrong) IR, found by fuzzing it against LLVM's
 * own verifier:
 *
 *   1. the seed strength_reduce_mul pattern built an operand-less `shl`
 *   2. hash-consing ignored types: `i8 1` == `i64 1`, `zext .. to i32`
 *      == `zext .. to i64`
 *   3. phis were rebuilt without their incoming blocks
 *   4. loads/allocas were hash-consed (two loads of one pointer merged)
 *   5. a value computed in both arms of a branch was shared across them
 *   6. verbatim copies dropped volatile / alignment
 *   7. loop back-edge phi operands named values extraction later renamed
 *
 * plus unit tests for ir::verify_function itself.
 */
#include <iostream>
#include <memory>
#include <string>

#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/IR/Verify.h"
#include "clunk/Parser/IRParser.h"
#include "clunk/Pattern/PatternLibrary.h"
#include "clunk/Search/EgraphRewriter.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;

static std::shared_ptr<ir::Module> parse(const std::string& src) {
    parser::IRParser p;
    return p.parse_string(src.c_str());
}

// Run saturation + extraction WITHOUT the cost gate or the final verifier, so
// a test sees exactly what the generator emits.
static std::shared_ptr<ir::Function> extract_raw(const ir::Function& fn) {
    pattern::PatternLibrary lib;
    pattern::ArchDescriptor arch;
    evaluator::EvaluationEngine eval;
    auto lower = egraph::lower_to_egraph(fn);
    egraph::EgraphRewriter rw(&lib, arch);
    rw.take_egraph(std::move(lower.egraph));
    rw.add_pattern_rules();
    rw.saturate(30);
    return rw.extract(lower, eval);
}

static std::string verdict(const ir::Function& f) { return ir::verify_function(f); }

static size_t count_op(const ir::Function& f, ir::Opcode op) {
    size_t n = 0;
    for (auto& bb : f.blocks())
        for (auto& i : bb->instructions())
            if (i && i->opcode() == op) ++n;
    return n;
}

// ── 1. the seed pattern ────────────────────────────────────────────────────
static void test_operandless_shl_seed() {
    auto m = parse(R"(
define i32 @f(i32 %x) {
entry:
  %r = mul i32 %x, %x
  ret i32 %r
}
)");
    auto fn = m->function("f");
    auto out = extract_raw(*fn);
    CHECK(out != nullptr, "extraction succeeds");
    CHECK(verdict(*out).empty(), "mul x,x must not become an operand-less shl: " + verdict(*out));
    CHECK(count_op(*out, ir::Opcode::Shl) == 0, "the seed pattern must not introduce a shl");

    pattern::PatternLibrary lib;
    pattern::ArchDescriptor arch;
    evaluator::EvaluationEngine eval;
    auto cand = search::egraph_rewrite(*fn, lib, arch, eval);
    CHECK(!cand || verdict(*cand->function).empty(), "anything egraph_rewrite returns is well-formed");
}

// ── 2. type-aware congruence ───────────────────────────────────────────────
static void test_types_are_part_of_identity() {
    egraph::EGraph eg;
    auto mk_const = [&](std::shared_ptr<ir::Type> t) {
        egraph::ENode n; n.is_constant = true; n.constant_value = 1; n.result_type = t;
        return eg.add(n);
    };
    CHECK(mk_const(ir::IntegerType::i8()) != mk_const(ir::IntegerType::i64()),
          "`i8 1` and `i64 1` are different values");
    CHECK(mk_const(ir::IntegerType::i32()) == mk_const(ir::IntegerType::i32()),
          "same-typed constants still hash-cons together");

    auto m = parse(R"(
define i64 @f(i8 %x) {
entry:
  %a = zext i8 %x to i32
  %b = zext i8 %x to i64
  %c = sext i32 %a to i64
  %d = add i64 %b, %c
  ret i64 %d
}
)");
    auto out = extract_raw(*m->function("f"));
    CHECK(out && verdict(*out).empty(), "casts to different widths must not merge: " + (out ? verdict(*out) : "null"));
    if (out) CHECK(count_op(*out, ir::Opcode::ZExt) == 2, "both zext instructions survive");
}

// ── 3. phis ────────────────────────────────────────────────────────────────
static void test_phi_keeps_incoming_blocks() {
    auto m = parse(R"(
define i32 @f(i32 %a) {
entry:
  %c = icmp eq i32 %a, 0
  br i1 %c, label %then, label %else
then:
  br label %merge
else:
  br label %merge
merge:
  %r = phi i32 [ 1, %then ], [ 2, %else ]
  ret i32 %r
}
)");
    auto out = extract_raw(*m->function("f"));
    CHECK(out != nullptr, "extraction succeeds");
    if (!out) return;
    CHECK(verdict(*out).empty(), "phi must keep its incoming blocks: " + verdict(*out));
    CHECK(out->to_string().find("<unknown>") == std::string::npos, "no %<unknown> block in the printed IR");
    CHECK(count_op(*out, ir::Opcode::Phi) == 1, "the phi survives");
}

// ── 4. memory ops are not value-numbered ───────────────────────────────────
static void test_loads_are_not_merged() {
    auto m = parse(R"(
define i32 @f(i32 %a, i32 %b) {
entry:
  %p = alloca i32
  store i32 %a, ptr %p
  %v1 = load i32, ptr %p
  store i32 %b, ptr %p
  %v2 = load i32, ptr %p
  %s = sub i32 %v1, %v2
  ret i32 %s
}
)");
    auto fn = m->function("f");
    auto out = extract_raw(*fn);
    CHECK(out && verdict(*out).empty(), "well-formed: " + (out ? verdict(*out) : "null"));
    if (!out) return;
    CHECK(count_op(*out, ir::Opcode::Load) == 2, "both loads survive (they read different memory states)");
    CHECK(count_op(*out, ir::Opcode::Alloca) == 1 && count_op(*out, ir::Opcode::Store) == 2, "alloca and stores survive");
    for (int64_t a : {0, 5, -3}) for (int64_t b : {0, 9, -7}) {
        auto o = evaluator::Interpreter::interpret(*fn, {a, b});
        auto c = evaluator::Interpreter::interpret(*out, {a, b});
        CHECK(o && c && *o == *c, "same result as the original for a=" + std::to_string(a) + " b=" + std::to_string(b));
    }
}

// ── 5. cross-block sharing ─────────────────────────────────────────────────
static void test_expression_in_both_arms_is_not_shared() {
    auto m = parse(R"(
define i32 @f(i32 %a, i32 %b) {
entry:
  %c = icmp eq i32 %a, 0
  br i1 %c, label %then, label %else
then:
  %x = add i32 %a, %b
  br label %merge
else:
  %y = add i32 %a, %b
  br label %merge
merge:
  %r = phi i32 [ %x, %then ], [ %y, %else ]
  ret i32 %r
}
)");
    auto out = extract_raw(*m->function("f"));
    CHECK(out && verdict(*out).empty(), "a value from one arm must not be used in the other: " + (out ? verdict(*out) : "null"));
}

// ── 6. volatile / alignment survive verbatim copies ────────────────────────
static void test_volatile_and_alignment_preserved() {
    // (The parser drops `store volatile` and the rest of its block, so mark
    // the store volatile after parsing.)
    auto m = parse(R"(
define void @f(ptr %p, i32 %x) {
entry:
  %a = add i32 %x, 1
  store i32 %a, ptr %p, align 16
  ret void
}
)");
    for (auto& i : m->function("f")->blocks().front()->instructions())
        if (i && i->opcode() == ir::Opcode::Store) i->set_volatile(true);
    auto out = extract_raw(*m->function("f"));
    CHECK(out != nullptr, "extraction succeeds");
    if (!out) return;
    const ir::Instruction* st = nullptr;
    for (auto& bb : out->blocks()) for (auto& i : bb->instructions())
        if (i && i->opcode() == ir::Opcode::Store) st = i.get();
    CHECK(st != nullptr, "store survives");
    if (st) {
        CHECK(st->is_volatile(), "a volatile store must stay volatile");
        CHECK(st->alignment().value_or(0) == 16, "the store keeps its alignment");
    }
}

// ── 7. loop back-edges ─────────────────────────────────────────────────────
static void test_loop_backedge_operands_follow_renaming() {
    auto m = parse(R"(
define i32 @f(i32 %n, i32 %a) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi i32 [ %a, %entry ], [ %acc2, %loop ]
  %inext = add i32 %i, 1
  %x = xor i32 %i, 3
  %z = xor i32 %i, 3
  %acc2 = add i32 %acc, %x
  %acc3 = add i32 %acc2, %z
  %c = icmp ult i32 %inext, %n
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %acc3
}
)");
    auto fn = m->function("f");
    auto out = extract_raw(*fn);
    CHECK(out && verdict(*out).empty(), "loop back-edge operands must resolve to extracted values: " + (out ? verdict(*out) : "null"));
    if (!out) return;
    for (int64_t n : {1, 2, 5, 20}) for (int64_t a : {0, 7}) {
        auto o = evaluator::Interpreter::interpret(*fn, {n, a});
        auto c = evaluator::Interpreter::interpret(*out, {n, a});
        CHECK(o && c && *o == *c, "loop result matches for n=" + std::to_string(n) + " a=" + std::to_string(a));
    }
}

// ── 8. verify_function itself ──────────────────────────────────────────────
static void test_verifier_accepts_valid_and_rejects_invalid() {
    auto good = parse(R"(
define i32 @f(i32 %a, i1 %c) {
entry:
  br i1 %c, label %t, label %e
t:
  %x = add i32 %a, 1
  br label %m
e:
  %y = sub i32 %a, 1
  br label %m
m:
  %r = phi i32 [ %x, %t ], [ %y, %e ]
  %z = zext i32 %r to i64
  %w = trunc i64 %z to i32
  ret i32 %w
}
)");
    CHECK(verdict(*good->function("f")).empty(), "valid diamond accepted: " + verdict(*good->function("f")));

    // Use of a value that does not dominate the use.
    auto bad_dom = parse(R"(
define i32 @f(i32 %a, i1 %c) {
entry:
  br i1 %c, label %t, label %e
t:
  %x = add i32 %a, 1
  br label %m
e:
  br label %m
m:
  %y = add i32 %x, 1
  ret i32 %y
}
)");
    CHECK(!verdict(*bad_dom->function("f")).empty(), "a use not dominated by its definition is rejected");

    // Operand-less binary op.
    auto m = parse("define i32 @f(i32 %a) {\nentry:\n  ret i32 %a\n}\n");
    auto fn = m->function("f");
    fn->blocks().front()->insert_instruction(0,
        std::make_shared<ir::Instruction>(ir::Opcode::Shl, ir::IntegerType::i32(), "bad"));
    CHECK(verdict(*fn).find("operand") != std::string::npos, "operand-less shl rejected: " + verdict(*fn));

    // Mismatched widths.
    auto mw = parse(R"(
define i32 @f(i32 %a, i8 %b) {
entry:
  %r = add i32 %a, %b
  ret i32 %r
}
)");
    CHECK(verdict(*mw->function("f")).find("widths") != std::string::npos, "mixed-width add rejected: " + verdict(*mw->function("f")));

    // Wrong return width.
    auto mr = parse(R"(
define i64 @f(i32 %a) {
entry:
  ret i32 %a
}
)");
    CHECK(verdict(*mr->function("f")).find("ret") != std::string::npos, "ret of the wrong width rejected: " + verdict(*mr->function("f")));

    // Globals are not SSA values of the function: using one is not "undefined".
    auto mg = parse(R"(
@g = global i32 5
define i32 @f(i32 %a) {
entry:
  %v = load i32, ptr @g
  %r = add i32 %a, %v
  ret i32 %r
}
)");
    CHECK(verdict(*mg->function("f")).empty(), "a function that reads a global verifies: " + verdict(*mg->function("f")));

    // Bad casts.
    auto mc = parse(R"(
define i16 @f(i32 %a) {
entry:
  %r = zext i32 %a to i16
  ret i16 %r
}
)");
    CHECK(verdict(*mc->function("f")).find("cast") != std::string::npos, "zext to a narrower type rejected: " + verdict(*mc->function("f")));
}

int main() {
    std::cerr << "test_egraph_validity: E-graph output must be valid IR\n";
    test_operandless_shl_seed();
    test_types_are_part_of_identity();
    test_phi_keeps_incoming_blocks();
    test_loads_are_not_merged();
    test_expression_in_both_arms_is_not_shared();
    test_volatile_and_alignment_preserved();
    test_loop_backedge_operands_follow_renaming();
    test_verifier_accepts_valid_and_rejects_invalid();
    std::cerr << "passed " << g_pass << ", failed " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
