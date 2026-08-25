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
 * Clunk Search Benchmark — measures search performance on increasingly
 * complex functions. Outputs timing data to stdout.
 *
 * Usage: ./clunk_bench [max_instructions]
 *   max_instructions: maximum instruction count for generated functions (default 64)
 */
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <iomanip>

#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"
#include "clunk/IR/IRBuilder.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Search/StochasticSearch.h"
#include "clunk/Search/EvolutionarySearch.h"

using namespace clunk::ir;
using namespace clunk::evaluator;
using namespace clunk::search;
using Clock = std::chrono::high_resolution_clock;

// ═══════════════════════════════════════════════════════════════════════════
//  Generate a function with N arithmetic instructions
// ═══════════════════════════════════════════════════════════════════════════

static std::shared_ptr<Function> generate_compute_function(
    Module& mod, const std::string& name, size_t num_ops)
{
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function(name, fn_type);
    fn.add_argument(ctx.int32(), "x");

    auto& entry = fn.add_block("entry");
    IRBuilder builder(ctx);
    builder.set_insert_point(&entry);

    std::shared_ptr<Value> val = builder.get_int32(5); // placeholder for %x
    std::shared_ptr<Value> accum = val;

    for (size_t i = 0; i < num_ops; ++i) {
        auto operand = builder.get_int32(static_cast<int32_t>(i + 1));
        switch (i % 4) {
            case 0: accum = builder.create_add(accum, operand, "v" + std::to_string(i)); break;
            case 1: accum = builder.create_mul(accum, operand, "v" + std::to_string(i)); break;
            case 2: accum = builder.create_sub(accum, operand, "v" + std::to_string(i)); break;
            case 3: accum = builder.create_add(accum, accum, "v" + std::to_string(i)); break;
        }
    }

    builder.create_ret(accum);
    return mod.function(name);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Generate a function with memory ops
// ═══════════════════════════════════════════════════════════════════════════

static std::shared_ptr<Function> generate_memory_function(
    Module& mod, const std::string& name, size_t num_ops)
{
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{});
    auto& fn = mod.add_function(name, fn_type);

    auto& entry = fn.add_block("entry");
    IRBuilder builder(ctx);
    builder.set_insert_point(&entry);

    auto slot = builder.create_alloca(ctx.int32(), "base", 4);
    auto val = builder.get_int32(0);
    builder.create_store(val, slot, 4);

    for (size_t i = 0; i < num_ops; ++i) {
        auto loaded = builder.create_load(slot, "ld" + std::to_string(i), 4);
        auto inc = builder.create_add(loaded, builder.get_int32(1), "inc" + std::to_string(i));
        builder.create_store(inc, slot, 4);
    }

    auto final_val = builder.create_load(slot, "result", 4);
    builder.create_ret(final_val);

    return mod.function(name);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Benchmark runner
// ═══════════════════════════════════════════════════════════════════════════

struct BenchResult {
    std::string label;
    size_t instruction_count;
    double stochastic_ms;
    double evolutionary_ms;
    size_t stochastic_candidates;
    size_t evolutionary_candidates;
};

static BenchResult run_benchmark(const std::string& label,
                                  std::shared_ptr<Function> fn,
                                  size_t instr_count)
{
    BenchResult result;
    result.label = label;
    result.instruction_count = instr_count;

    EvaluationEngine engine;

    // ── Stochastic Search ─────────────────────────────────────────────
    {
        StochasticConfig config;
        config.max_iterations = 2000;
        config.max_candidates = 50;
        config.seed = 42;
        StochasticSearch search(config, &engine);

        auto start = Clock::now();
        auto candidates = search.search(*fn);
        auto end = Clock::now();

        result.stochastic_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.stochastic_candidates = candidates.size();
    }

    // ── Evolutionary Search ───────────────────────────────────────────
    {
        EvolutionaryConfig config;
        config.population_size = 20;
        config.max_generations = 30;
        config.seed = 42;
        EvolutionarySearch search(config, &engine);

        auto start = Clock::now();
        auto candidates = search.search(*fn);
        auto end = Clock::now();

        result.evolutionary_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.evolutionary_candidates = candidates.size();
    }

    return result;
}

// Helper for calculating doubling numbers up to and including N (only includes N if N is a power of two.)

std::vector<size_t> doublingSequence(size_t N) {
    std::vector<size_t> result;
    if (N < 4) {
        return result;
    }

    // Precompute the count so we allocate exactly once (no reallocations/copies).
    size_t count = 0;
    for (size_t v = 4; v <= N; v <<= 1) {
        ++count;
        if (v > (N >> 1)) break; // guard against overflow on next shift
    }

    result.reserve(count);
    for (size_t v = 4; v <= N; v <<= 1) {
        result.push_back(v);
        if (v > (N >> 1)) break; // stop before v <<= 1 could overflow
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    size_t max_instr = 64;
    if (argc > 1) {
        max_instr = std::stoul(argv[1]);
    }

    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                            Clunk Search Benchmark                            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n\n";

    std::vector<BenchResult> results;

    // Compute-heavy benchmarks at increasing sizes
    std::vector<size_t> sizes = doublingSequence(max_instr);

    for (auto sz : sizes) {
        if (sz > max_instr) break;
        Module mod("bench_compute_" + std::to_string(sz));
        auto fn = generate_compute_function(mod, "compute_" + std::to_string(sz), sz);
        std::cout << "  Running compute benchmark with " << sz << " instructions..." << std::endl;
        results.push_back(run_benchmark("compute_" + std::to_string(sz), fn, sz));
    }

    // Memory-heavy benchmarks
    for (auto sz : sizes) {
        Module mod("bench_mem_" + std::to_string(sz));
        auto fn = generate_memory_function(mod, "mem_" + std::to_string(sz), sz);
        std::cout << "  Running memory benchmark with " << sz << " memory ops..." << std::endl;
        results.push_back(run_benchmark("mem_" + std::to_string(sz), fn, sz * 3 + 3));
    }

    // Print results table
    std::cout << "\n";
    std::cout << "┌─────────────────┬───────────┬─────────────────┬───────────┬─────────────────┬───────────┐\n";
    std::cout << "│ Benchmark       │ Instrs    │ Stochastic (ms) │ Cands     │ Evolutionary(ms)│ Cands     │\n";
    std::cout << "├─────────────────┼───────────┼─────────────────┼───────────┼─────────────────┼───────────┤\n";

    for (auto& r : results) {
        std::cout << "│ "
                  << std::left << std::setw(16) << r.label << "│ "
                  << std::right << std::setw(9) << r.instruction_count << " │ "
                  << std::right << std::setw(15) << std::fixed << std::setprecision(1) << r.stochastic_ms << " │ "
                  << std::right << std::setw(9) << r.stochastic_candidates << " │ "
                  << std::right << std::setw(15) << std::fixed << std::setprecision(1) << r.evolutionary_ms << " │ "
                  << std::right << std::setw(9) << r.evolutionary_candidates << " │\n";
    }

    std::cout << "└─────────────────┴───────────┴─────────────────┴───────────┴─────────────────┴───────────┘\n";

    std::cout << "\nBenchmark complete.\n";
    return 0;
}
