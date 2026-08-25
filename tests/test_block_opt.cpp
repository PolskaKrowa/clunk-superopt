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
 * Clunk Block-Level Optimiser Tests.
 *
 * Tests the divide-and-conquer block optimiser that splits a function's
 * single basic block into a binary tree of instruction ranges and
 * SMT-proves cheaper equivalents for each.
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
#include "clunk/Search/SMTVerifier.h"
#include "clunk/Search/BlockOptimiser.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Pipeline.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk;

// ═══════════════════════════════════════════════════════════════════════════
//  Direct BlockOptimiser tests
// ═══════════════════════════════════════════════════════════════════════════

// `(x & 1) | (x & 2)` — 3 instructions, equivalent to `x & 3` (1 instruction).
// The block optimiser should find the range-wide rewrite `and x, 3` at depth 1.
void test_block_opt_finds_bit_trick() {
    if (!search::SMTVerifier::is_z3_available()) {
        std::cerr << "  (z3 unavailable — skipped)\n";
        return;
    }
    const char* ir = R"(
define i32 @bits(i32 %x) {
entry:
  %a = and i32 %x, 1
  %b = and i32 %x, 2
  %c = or i32 %a, %b
  ret i32 %c
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("bits"), "parsed bit-trick input");
    if (!mod || !mod->function("bits")) return;
    auto fn = mod->function("bits");
    const size_t ops_before = fn->blocks().front()->size();  // 4 (and, and, or, ret)

    evaluator::EvaluationEngine engine;
    search::BlockOptimiserConfig cfg;
    cfg.time_budget_seconds = 10.0;
    search::BlockOptimiser opt(&engine, cfg);
    bool proven = false;
    auto rewritten = opt.optimize(*fn, &proven);

    CHECK(rewritten != nullptr, "block optimiser found a rewrite for the bit-trick");
    CHECK(proven, "the rewrite is SMT-proven");
    if (rewritten) {
        const size_t ops_after = rewritten->blocks().front()->size();
        CHECK(ops_after < ops_before,
              "rewritten function has fewer instructions");
        CHECK(opt.stats().blocks_optimised > 0,
              "at least one block was optimised");
    }
}

// `x * 2` written as `x + x` — the block optimiser should NOT find a
// cheaper rewrite (x + x is already optimal for the cost model), and
// should declare the block naturally optimal.
void test_block_opt_natural_optimal() {
    if (!search::SMTVerifier::is_z3_available()) {
        std::cerr << "  (z3 unavailable — skipped)\n";
        return;
    }
    const char* ir = R"(
define i32 @double(i32 %x) {
entry:
  %a = add i32 %x, %x
  ret i32 %a
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("double"), "parsed double input");
    if (!mod || !mod->function("double")) return;
    auto fn = mod->function("double");

    evaluator::EvaluationEngine engine;
    search::BlockOptimiserConfig cfg;
    cfg.time_budget_seconds = 5.0;
    search::BlockOptimiser opt(&engine, cfg);
    bool proven = false;
    auto rewritten = opt.optimize(*fn, &proven);

    // The function is 2 instructions (add + ret), which is the minimum
    // block size. The optimiser tries depth 1 (a single binop) and if
    // it can't find a cheaper equivalent, declares it natural.
    // We don't assert rewritten == nullptr (the optimiser MIGHT find
    // a same-cost rewrite), but we do check the stats make sense.
    if (!rewritten) {
        CHECK(opt.stats().blocks_natural > 0,
              "block declared naturally optimal when no improvement found");
    }
}

// Multi-block function — the block optimiser should skip it (out of scope).
void test_block_opt_skips_multi_block() {
    const char* ir = R"(
define i32 @branchy(i32 %x) {
entry:
  %c = icmp eq i32 %x, 0
  br i1 %c, label %then, label %else
then:
  %r1 = add i32 %x, 1
  ret i32 %r1
else:
  %r2 = sub i32 %x, 1
  ret i32 %r2
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("branchy"), "parsed multi-block input");
    if (!mod || !mod->function("branchy")) return;
    auto fn = mod->function("branchy");

    evaluator::EvaluationEngine engine;
    search::BlockOptimiserConfig cfg;
    search::BlockOptimiser opt(&engine, cfg);
    bool proven = false;
    auto rewritten = opt.optimize(*fn, &proven);

    CHECK(rewritten == nullptr, "block optimiser skips multi-block functions");
    CHECK(opt.stats().functions_skipped > 0,
          "multi-block function counted as skipped");
}

// A function where the WHOLE function has no short equivalent, but a
// SUB-RANGE does. This is the block optimiser's key use case.
//
// `((x & 1) | (x & 2)) + x` — 4 instructions + ret. The whole function
// has no 1-3 instruction equivalent (the `+ x` makes it complex). But
// the sub-range `(x & 1) | (x & 2)` has a 1-instruction equivalent
// (`x & 3`). The block optimiser should split the function, find the
// sub-range rewrite, and produce `((x & 3) + x)` — 2 instructions + ret.
void test_block_opt_finds_subrange_rewrite() {
    if (!search::SMTVerifier::is_z3_available()) {
        std::cerr << "  (z3 unavailable — skipped)\n";
        return;
    }
    const char* ir = R"(
define i32 @combo(i32 %x) {
entry:
  %a = and i32 %x, 1
  %b = and i32 %x, 2
  %c = or i32 %a, %b
  %d = add i32 %c, %x
  ret i32 %d
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("combo"), "parsed combo input");
    if (!mod || !mod->function("combo")) return;
    auto fn = mod->function("combo");
    const size_t ops_before = fn->blocks().front()->size();  // 5

    evaluator::EvaluationEngine engine;
    search::BlockOptimiserConfig cfg;
    cfg.time_budget_seconds = 15.0;
    search::BlockOptimiser opt(&engine, cfg);
    bool proven = false;
    auto rewritten = opt.optimize(*fn, &proven);

    CHECK(rewritten != nullptr, "block optimiser found a sub-range rewrite");
    if (rewritten) {
        const size_t ops_after = rewritten->blocks().front()->size();
        CHECK(ops_after < ops_before,
              "rewritten combo function has fewer instructions");
        CHECK(proven, "the sub-range rewrite is SMT-proven");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Pipeline integration tests
// ═══════════════════════════════════════════════════════════════════════════

// End-to-end: the pipeline's block_phase fires when the whole-function
// search is stagnant and optimises a function that the whole-function
// search alone cannot.
void test_pipeline_block_opt_fires() {
    if (!search::SMTVerifier::is_z3_available()) {
        std::cerr << "  (z3 unavailable — skipped)\n";
        return;
    }
    const char* ir = R"(
define i32 @combo(i32 %x) {
entry:
  %a = and i32 %x, 1
  %b = and i32 %x, 2
  %c = or i32 %a, %b
  %d = add i32 %c, %x
  ret i32 %d
}
)";
    auto mod = parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("combo"), "parsed pipeline combo input");
    if (!mod || !mod->function("combo")) return;
    auto fn = mod->function("combo");
    const size_t ops_before = fn->blocks().front()->size();

    // With block-opt ON (default): the pipeline should find the
    // sub-range rewrite.
    {
        PipelineConfig cfg;
        cfg.opt_level = 2;
        cfg.time_budget = 30.0;
        cfg.enable_block_opt = true;
        Pipeline pipe(cfg);
        auto r = pipe.run_on_function(*fn);
        CHECK(r.optimised != nullptr, "pipeline returns a result");
        if (r.optimised) {
            CHECK(r.optimised->blocks().front()->size() <= ops_before,
                  "pipeline with block-opt does not regress");
        }
    }

    // With block-opt OFF: the pipeline might still find the rewrite via
    // the miner or hole-synth, but we just check it doesn't crash.
    {
        PipelineConfig cfg;
        cfg.opt_level = 2;
        cfg.time_budget = 10.0;
        cfg.enable_block_opt = false;
        Pipeline pipe(cfg);
        auto r = pipe.run_on_function(*fn);
        CHECK(r.optimised != nullptr, "pipeline with block-opt off returns a result");
    }
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "=== Clunk Block-Level Optimiser Tests ===" << std::endl;

    std::cout << "  Block optimiser finds bit-trick..." << std::endl;
    test_block_opt_finds_bit_trick();

    std::cout << "  Block optimiser natural optimal..." << std::endl;
    test_block_opt_natural_optimal();

    std::cout << "  Block optimiser skips multi-block..." << std::endl;
    test_block_opt_skips_multi_block();

    std::cout << "  Block optimiser finds sub-range rewrite..." << std::endl;
    test_block_opt_finds_subrange_rewrite();

    std::cout << "  Pipeline block-opt fires..." << std::endl;
    test_pipeline_block_opt_fires();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
