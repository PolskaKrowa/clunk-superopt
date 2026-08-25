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
 * Clunk Series Expander Tests.
 *
 * Tests the series-expansion loop optimiser that detects single-block
 * loops computing arithmetic series and replaces them with closed-form
 * expressions.
 */
#include <iostream>
#include <memory>
#include <string>

#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"
#include "clunk/Parser/IRParser.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Search/SeriesExpander.h"
#include "clunk/Pipeline.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk;

// ═══════════════════════════════════════════════════════════════════════════
//  Direct SeriesExpander tests
// ═══════════════════════════════════════════════════════════════════════════

// `for (i=0; i<n; i++) sum += i` → `sum = n*(n-1)/2`
// The canonical arithmetic series: 0+1+2+...+(n-1) = n*(n-1)/2.
void test_series_sum_of_i() {
    // IR for:
    //   define i32 @sum_i(i32 %n) {
    //   entry:
    //     br label %loop
    //   loop:
    //     %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
    //     %sum = phi i32 [ 0, %entry ], [ %sum_next, %loop ]
    //     %i_next = add i32 %i, 1
    //     %sum_next = add i32 %sum, %i
    //     %cond = icmp slt i32 %i, %n
    //     br i1 %cond, label %loop, label %exit
    //   exit:
    //     ret i32 %sum
    //   }
    const char* ir = R"(
define i32 @sum_i(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum_next, %loop ]
  %i_next = add i32 %i, 1
  %sum_next = add i32 %sum, %i
  %cond = icmp slt i32 %i, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret i32 %sum
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("sum_i"), "parsed sum-of-i loop");
    if (!mod || !mod->function("sum_i")) return;
    auto fn = mod->function("sum_i");

    evaluator::EvaluationEngine engine;
    search::SeriesExpander expander(&engine);
    auto rewritten = expander.expand(*fn);

    CHECK(rewritten != nullptr, "expander found a closed form for sum-of-i");
    CHECK(expander.stats().loops_expanded > 0, "at least one loop was expanded");
    if (!rewritten) return;

    // The rewritten function should have no back-edge (loop eliminated).
    CHECK(!has_back_edge(*rewritten), "rewritten function has no loop");

    // Verify equivalence on a range of inputs.
    bool all_match = true;
    for (int64_t n = 0; n <= 20; ++n) {
        auto orig_r = evaluator::Interpreter::interpret(*fn, {n});
        auto new_r = evaluator::Interpreter::interpret(*rewritten, {n});
        if (!orig_r || !new_r || *orig_r != *new_r) {
            std::cerr << "  MISMATCH at n=" << n << ": orig="
                      << (orig_r ? std::to_string(*orig_r) : "nullopt")
                      << ", new=" << (new_r ? std::to_string(*new_r) : "nullopt")
                      << "\n";
            all_match = false;
        }
    }
    CHECK(all_match, "closed form matches original for n=0..20");

    // Known value: sum(0..9) = 45.
    auto r10 = evaluator::Interpreter::interpret(*rewritten, {10});
    CHECK(r10 && *r10 == 45, "sum(0..9) = 45");
}

// `for (i=0; i<n; i++) sum += 5` → `sum = 5 * n`
// Constant accumulation.
void test_series_constant_accumulation() {
    const char* ir = R"(
define i32 @sum_c(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum_next, %loop ]
  %i_next = add i32 %i, 1
  %sum_next = add i32 %sum, 5
  %cond = icmp slt i32 %i, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret i32 %sum
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("sum_c"), "parsed constant-accumulation loop");
    if (!mod || !mod->function("sum_c")) return;
    auto fn = mod->function("sum_c");

    evaluator::EvaluationEngine engine;
    search::SeriesExpander expander(&engine);
    auto rewritten = expander.expand(*fn);

    CHECK(rewritten != nullptr, "expander found a closed form for constant accumulation");
    if (!rewritten) return;

    bool all_match = true;
    for (int64_t n = 0; n <= 20; ++n) {
        auto orig_r = evaluator::Interpreter::interpret(*fn, {n});
        auto new_r = evaluator::Interpreter::interpret(*rewritten, {n});
        if (!orig_r || !new_r || *orig_r != *new_r) {
            all_match = false;
        }
    }
    CHECK(all_match, "constant accumulation matches for n=0..20");

    // 5 * 10 = 50.
    auto r10 = evaluator::Interpreter::interpret(*rewritten, {10});
    CHECK(r10 && *r10 == 50, "5*10 = 50");
}

// `for (i=0; i<n; i++) sum += i * 3` → `sum = 3 * n*(n-1)/2`
// Scaled arithmetic series.
void test_series_scaled_arithmetic() {
    const char* ir = R"(
define i32 @sum_scaled(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum_next, %loop ]
  %i_next = add i32 %i, 1
  %term = mul i32 %i, 3
  %sum_next = add i32 %sum, %term
  %cond = icmp slt i32 %i, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret i32 %sum
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("sum_scaled"), "parsed scaled-arithmetic loop");
    if (!mod || !mod->function("sum_scaled")) return;
    auto fn = mod->function("sum_scaled");

    evaluator::EvaluationEngine engine;
    search::SeriesExpander expander(&engine);
    auto rewritten = expander.expand(*fn);

    CHECK(rewritten != nullptr, "expander found a closed form for scaled arithmetic");
    if (!rewritten) return;

    bool all_match = true;
    for (int64_t n = 0; n <= 20; ++n) {
        auto orig_r = evaluator::Interpreter::interpret(*fn, {n});
        auto new_r = evaluator::Interpreter::interpret(*rewritten, {n});
        if (!orig_r || !new_r || *orig_r != *new_r) {
            std::cerr << "  MISMATCH at n=" << n << ": orig="
                      << (orig_r ? std::to_string(*orig_r) : "nullopt")
                      << ", new=" << (new_r ? std::to_string(*new_r) : "nullopt")
                      << "\n";
            all_match = false;
        }
    }
    CHECK(all_match, "scaled arithmetic matches for n=0..20");

    // 3 * (0+1+2+...+9) = 3 * 45 = 135.
    auto r10 = evaluator::Interpreter::interpret(*rewritten, {10});
    CHECK(r10 && *r10 == 135, "3*45 = 135");
}

// A loop that doesn't match any pattern — the expander should return nullptr.
// Here the accumulator multiplies (geometric), which we don't handle.
void test_series_no_match() {
    const char* ir = R"(
define i32 @geo(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %acc = phi i32 [ 1, %entry ], [ %acc_next, %loop ]
  %i_next = add i32 %i, 1
  %acc_next = mul i32 %acc, 2
  %cond = icmp slt i32 %i, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret i32 %acc
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("geo"), "parsed geometric loop");
    if (!mod || !mod->function("geo")) return;
    auto fn = mod->function("geo");

    evaluator::EvaluationEngine engine;
    search::SeriesExpander expander(&engine);
    auto rewritten = expander.expand(*fn);

    // Geometric series (acc *= 2) is NOT in our pattern set — the
    // accumulator update is `mul`, not `add`. The expander should bail.
    CHECK(rewritten == nullptr, "expander bails on geometric (mul) accumulator");
}

// A loop-free function — the expander should skip it.
void test_series_no_loop() {
    const char* ir = R"(
define i32 @straight(i32 %x) {
entry:
  %a = add i32 %x, 1
  %b = mul i32 %a, 2
  ret i32 %b
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("straight"), "parsed loop-free function");
    if (!mod || !mod->function("straight")) return;
    auto fn = mod->function("straight");

    evaluator::EvaluationEngine engine;
    search::SeriesExpander expander(&engine);
    auto rewritten = expander.expand(*fn);

    CHECK(rewritten == nullptr, "expander skips loop-free functions");
    CHECK(expander.stats().functions_skipped > 0, "function counted as skipped");
}

// Loop with nonzero start: `for (i=3; i<n; i++) sum += i`
// sum = 3 + 4 + ... + (n-1) = (sum of 0..n-1) - (0+1+2) = n*(n-1)/2 - 3
// Our formula: n*start + step*n*(n-1)/2 = n*3 + 1*n*(n-1)/2 = 3n + n*(n-1)/2
// For n=10: 30 + 45 = 75. Check: 3+4+5+6+7+8+9 = 42... wait, n=10 means i goes 3..9 (7 iterations).
// Actually for n=10, i < 10, i starts at 3: i = 3,4,5,6,7,8,9 → 7 iterations, sum = 42.
// Trip = (10 - 3) / 1 = 7. Formula: 7*3 + 1*7*6/2 = 21 + 21 = 42. ✓
void test_series_nonzero_start() {
    const char* ir = R"(
define i32 @sum_start3(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 3, %entry ], [ %i_next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum_next, %loop ]
  %i_next = add i32 %i, 1
  %sum_next = add i32 %sum, %i
  %cond = icmp slt i32 %i, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret i32 %sum
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("sum_start3"), "parsed nonzero-start loop");
    if (!mod || !mod->function("sum_start3")) return;
    auto fn = mod->function("sum_start3");

    evaluator::EvaluationEngine engine;
    search::SeriesExpander expander(&engine);
    auto rewritten = expander.expand(*fn);

    CHECK(rewritten != nullptr, "expander handles nonzero start");
    if (!rewritten) return;

    bool all_match = true;
    for (int64_t n = 0; n <= 20; ++n) {
        auto orig_r = evaluator::Interpreter::interpret(*fn, {n});
        auto new_r = evaluator::Interpreter::interpret(*rewritten, {n});
        if (!orig_r || !new_r || *orig_r != *new_r) {
            std::cerr << "  MISMATCH at n=" << n << ": orig="
                      << (orig_r ? std::to_string(*orig_r) : "nullopt")
                      << ", new=" << (new_r ? std::to_string(*new_r) : "nullopt")
                      << "\n";
            all_match = false;
        }
    }
    CHECK(all_match, "nonzero-start series matches for n=0..20");

    // n=10: i = 3..9, sum = 3+4+5+6+7+8+9 = 42.
    auto r10 = evaluator::Interpreter::interpret(*rewritten, {10});
    CHECK(r10 && *r10 == 42, "sum(3..9) = 42");

    // n=3: i < 3, i starts at 3 → 0 iterations, sum = 0.
    auto r3 = evaluator::Interpreter::interpret(*rewritten, {3});
    CHECK(r3 && *r3 == 0, "sum(n=3, start=3) = 0 (zero trips)");

    // n=0: i < 0, i starts at 3 → 0 iterations, sum = 0.
    auto r0 = evaluator::Interpreter::interpret(*rewritten, {0});
    CHECK(r0 && *r0 == 0, "sum(n=0, start=3) = 0 (zero trips)");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Pipeline integration test
// ═══════════════════════════════════════════════════════════════════════════

// End-to-end: the pipeline's series_phase fires and optimises a loop.
void test_pipeline_series_expand() {
    const char* ir = R"(
define i32 @sum_i(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum_next, %loop ]
  %i_next = add i32 %i, 1
  %sum_next = add i32 %sum, %i
  %cond = icmp slt i32 %i, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret i32 %sum
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("sum_i"), "parsed pipeline series input");
    if (!mod || !mod->function("sum_i")) return;
    auto fn = mod->function("sum_i");

    // With series-expand ON (default): the pipeline should eliminate the loop.
    {
        PipelineConfig cfg;
        cfg.opt_level = 2;
        cfg.time_budget = 10.0;
        cfg.enable_series_expand = true;
        Pipeline pipe(cfg);
        auto r = pipe.run_on_function(*fn);
        CHECK(r.optimised != nullptr, "pipeline returns a result");
        if (r.optimised) {
            // The loop should be gone (no back-edge).
            CHECK(!has_back_edge(*r.optimised),
                  "pipeline eliminated the loop via series expansion");
        }
    }

    // With series-expand OFF: the loop should survive (the other phases
    // can't eliminate a variable-trip loop).
    {
        PipelineConfig cfg;
        cfg.opt_level = 2;
        cfg.time_budget = 5.0;
        cfg.enable_series_expand = false;
        Pipeline pipe(cfg);
        auto r = pipe.run_on_function(*fn);
        CHECK(r.optimised != nullptr, "pipeline with series-expand off returns a result");
        if (r.optimised) {
            // The loop survives — no other phase can eliminate a variable-
            // trip arithmetic loop.
            CHECK(has_back_edge(*r.optimised),
                  "loop survives when series-expand is disabled");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "=== Clunk Series Expander Tests ===" << std::endl;

    std::cout << "  Sum of i..." << std::endl;
    test_series_sum_of_i();

    std::cout << "  Constant accumulation..." << std::endl;
    test_series_constant_accumulation();

    std::cout << "  Scaled arithmetic..." << std::endl;
    test_series_scaled_arithmetic();

    std::cout << "  No match (geometric)..." << std::endl;
    test_series_no_match();

    std::cout << "  No loop..." << std::endl;
    test_series_no_loop();

    std::cout << "  Nonzero start..." << std::endl;
    test_series_nonzero_start();

    std::cout << "  Pipeline integration..." << std::endl;
    test_pipeline_series_expand();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
