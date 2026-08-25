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
 * Clunk Pipeline — orchestrates the full optimisation pipeline:
 * Pattern Library → Stochastic Search → Evolutionary Search → SMT Verify
 */
#include "clunk/Pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>
#include <unordered_set>

#include "clunk/Evaluator/CostModel.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/IR/Clone.h"
#include "clunk/IR/DataflowPrune.h"
#include "clunk/IR/LoopAnalysis.h"
#include "clunk/IR/Scalarizer.h"
#include "clunk/Analysis/CallGraph.h"
#include "clunk/Search/CrossFunctionPasses.h"
#include "clunk/Search/Inliner.h"
#include "clunk/Search/LoopOpt.h"
#include "clunk/Search/MemOpt.h"
#include "clunk/Search/VectorSynth.h"
#include "clunk/Search/BlockOptimiser.h"
#include "clunk/Search/SeriesExpander.h"

#ifdef CLUNK_HAS_GPU
#include "clunk/GPU/PTXOptimizer.h"
#endif

namespace clunk {

// ── Anonymous-namespace helpers ─────────────────────────────────────────────

// Count total instructions across all basic blocks of a function. Used by
// the function-size gate and the search-budget scaling.
namespace {
size_t count_instructions(const ir::Function& fn) {
    size_t count = 0;
    for (auto& block : fn.blocks()) {
        if (block) count += block->size();
    }
    return count;
}
} // namespace

// ── Constructor ─────────────────────────────────────────────────────────────

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config),
      eval_engine_(),
      searchers_(config, &eval_engine_),
      pattern_lib_()
{
    // Propagate target_arch to the evaluator via a per-architecture
    // CostModel.
    auto cost_model = evaluator::make_cost_model_for_name(config_.target_arch.name);
    eval_engine_.set_cost_model(cost_model);

    // Load pattern library from disk if a path was provided
    if (!config.pattern_library_path.empty()) {
        if (!pattern_lib_.load(config.pattern_library_path)) {
            // Library file didn't exist or was unreadable — seed built-ins
            // already happened in the constructor.
        }
    }

    // ── Persistent SMT rewrite cache ──────────────────────────────
    // Open the file-backed cache and warm it up. The shared_ptr is plumbed
    // into each worker's SMTVerifier via SMTConfig::cache (see the worker
    // lambda in run()). The cache is thread-safe (mutex-guarded).
    if (!config_.smt_cache_path.empty()) {
        rewrite_cache_ = std::make_shared<search::RewriteCache>();
        if (rewrite_cache_->open_file(config_.smt_cache_path)) {
            rewrite_cache_->warmup();
            // Wire it into the shared sequential SMTVerifier. Worker threads
            // create their own SMTVerifier but inherit the same SMTConfig
            // (and thus the same cache shared_ptr).
            searchers_.smt.config().cache = rewrite_cache_;
        } else {
            // File couldn't be opened — disable caching for this run.
            rewrite_cache_.reset();
        }
    }

    // ── STOKE-style moves + test-vector pre-filter ────────────────
    // Propagate the pipeline-level flags into the stochastic search config
    // so both the sequential and per-worker StochasticSearch instances pick
    // them up (the Searchers ctor copies the config by value).
    searchers_.stochastic.config().allow_unsound_mutations =
        config_.allow_unsound_mutations;
    searchers_.stochastic.config().test_vector_count =
        config_.test_vector_count;
}

// ── Deadline bookkeeping ────────────────────────────────────────────────────

double Pipeline::remaining_seconds() const {
    if (!has_deadline_) return std::numeric_limits<double>::infinity();
    return std::chrono::duration<double>(
               deadline_ - std::chrono::steady_clock::now()).count();
}

// ── run ─────────────────────────────────────────────────────────────────────

PipelineResult Pipeline::run(const ir::Module& module) {
    PipelineResult result;
    auto start = std::chrono::high_resolution_clock::now();

    // ── Module-level progress tracking (for the TUI progress bar) ────
    module_start_ = std::chrono::steady_clock::now();
    module_done_functions_.store(0, std::memory_order_relaxed);

    // Arm the shared wall-clock deadline. run_on_function() and the search
    // phases derive their per-round budgets from it.
    if (config_.time_budget > 0.0) {
        deadline_ = std::chrono::steady_clock::now() +
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(config_.time_budget));
        has_deadline_ = true;
    } else {
        has_deadline_ = false;
    }

    // Create the optimised module (deep copy)
    result.optimised_module = std::make_shared<ir::Module>(module.name());
    // Copy target info
    if (module.has_target()) {
        result.optimised_module->set_target(module.target());
    }
    // Copy round-trip-preservation fields (attribute groups, named metadata,
    // metadata defs, module asm, named types, source filename, module flags).
    // Without this, emitting the optimised module drops these — and clang
    // won't recompile the result.
    result.optimised_module->set_source_filename(module.source_filename());
    for (auto& mf : module.module_flags()) result.optimised_module->add_module_flag(mf);
    for (auto& [id, body] : module.attribute_groups()) {
        result.optimised_module->add_attribute_group(id, body);
    }
    for (auto& [name, body] : module.named_metadata()) {
        result.optimised_module->add_named_metadata(name, body);
    }
    for (auto& [id, body] : module.metadata_defs()) {
        result.optimised_module->add_metadata_def(id, body);
    }
    for (auto& asm_line : module.module_asm()) {
        result.optimised_module->add_module_asm(asm_line);
    }
    for (auto& [name, ty] : module.named_types()) {
        result.optimised_module->add_named_type(name, ty);
    }

    // ── Module-level pre-pass: cross-function optimisations ────────────
    // Run BEFORE per-function optimisation. The pre-pass operates on a
    // deep copy of the module so the user's input is untouched. The
    // passes are:
    //   1. Dead Function Elimination — remove module-internal functions
    //      no caller can reach. Frees the per-function pipeline from
    //      wasting rounds on dead code.
    //   2. Interprocedural Constant Propagation — clone a function for
    //      every constant value its callers uniformly pass to a given
    //      argument position, then rewrite callers to the clone. The
    //      clone is then a constant-folding target for the per-function
    //      pipeline. After IPCP, the original function might be dead
    //      (if ALL callers were rewritten); DFE runs again to clean up.
    //
    // Both passes are exact by construction (no SMT needed).
    //
    // Build the work_module as a DEEP copy: a fresh Module with deep-copied
    // Function objects. (Module's default copy constructor is shallow —
    // functions_ holds shared_ptrs, so a shallow copy would share Function
    // objects with the input module and our mutations would leak back.)
    auto work_module = std::make_shared<ir::Module>(module.name());
    if (module.has_target()) work_module->set_target(module.target());
    work_module->set_source_filename(module.source_filename());
    for (auto& mf : module.module_flags()) work_module->add_module_flag(mf);
    for (auto& [id, body] : module.attribute_groups()) {
        work_module->add_attribute_group(id, body);
    }
    for (auto& [name, body] : module.named_metadata()) {
        work_module->add_named_metadata(name, body);
    }
    for (auto& [id, body] : module.metadata_defs()) {
        work_module->add_metadata_def(id, body);
    }
    for (auto& asm_line : module.module_asm()) {
        work_module->add_module_asm(asm_line);
    }
    for (auto& [name, ty] : module.named_types()) {
        work_module->add_named_type(name, ty);
    }
    for (auto& fn : module.functions()) {
        if (!fn) continue;
        work_module->add_function(ir::deep_copy_function(*fn));
    }
    for (auto& gv : module.globals()) {
        work_module->add_global(gv);
    }

    if (config_.enable_cross_function && config_.opt_level >= 2) {
        search::CrossFnConfig xcfg;
        xcfg.enable_dfe = true;
        xcfg.enable_ipcp = true;
        search::CrossFunctionPasses xpass(xcfg);
        bool xchanged = xpass.run(*work_module);
        if (xchanged && config_.verbose) {
            const auto& st = xpass.stats();
            std::cerr << "  [xpass] DFE removed " << st.dfe_removed
                      << " function(s); IPCP created " << st.ipcp_cloned
                      << " clone(s), rewrote " << st.ipcp_callers_rewritten
                      << " caller(s)\n";
        }
    }

    // ── Module-level pre-pass: algorithmic preprocessor ──────────────
    // Walks the call graph and detects functions (or compositions of
    // functions) whose output is a predictable closed form (constant,
    // c*x, c*x+b). When a pattern is detected and SMT-proven, the
    // function's body is rewritten to the minimal closed form —
    // shrinking the work the per-function pipeline has to do.
    if (config_.enable_algo_preprocessor && config_.opt_level >= 2) {
        search::AlgoPreConfig acfg = config_.algo_pre_config;
        if (config_.smt_config.timeout_ms > 0)
            acfg.smt_timeout_ms = config_.smt_config.timeout_ms;
        acfg.trust_unverified = config_.trust_unverified ||
                                config_.skip_smt ||
                                !search::SMTVerifier::is_z3_available();
        search::AlgoPreprocessor ap(acfg);
        bool achanged = ap.run(*work_module);
        if (achanged && config_.verbose) {
            const auto& st = ap.stats();
            std::cerr << "  [algo-pre] scanned " << st.functions_scanned
                      << " function(s); detected "
                      << st.constants_detected << " constant(s), "
                      << st.scalars_detected << " multiplicative, "
                      << st.affines_detected << " affine, "
                      << st.compositions_detected << " composition collapse(s); "
                      << st.rewrites_proven << " proven\n";
        }
    }

    // ── Module-level pre-pass: constant folding ──────────────────────
    // Run prune_dataflow (DCE + known-bits constant/branch folding +
    // unreachable-block elimination) on EVERY function in the module
    // BEFORE any per-function processing. This is critical for IPCP
    // clones: IPCP substitutes constant arguments into a clone, but
    // doesn't fold the resulting constant expressions. Without this
    // pass, a caller's inliner sees the un-folded multi-block callee
    // (e.g. `icmp sge 5, 0; br; mul 5, 2; ret; ret 0`) and rejects it
    // via the cost gate. With this pass, the callee is pre-folded to
    // `ret 10`, and the caller's inliner can inline it trivially.
    //
    // Sound by construction (prune_dataflow is exact). Runs at most
    // 3 iterations per function to reach a fixpoint.
    if (config_.opt_level >= 2) {
        size_t total_folded = 0, total_branches = 0, total_blocks = 0;
        for (auto& fn : work_module->functions()) {
            if (!fn) continue;
            for (int iter = 0; iter < 3; ++iter) {
                ir::PruneStats stats;
                auto pruned = ir::prune_dataflow(*fn, &stats);
                if (!pruned) break;
                // Replace the function's body with the pruned version.
                auto& blocks = const_cast<std::vector<std::shared_ptr<ir::BasicBlock>>&>(
                    fn->blocks());
                blocks.clear();
                for (auto& bb : pruned->blocks()) {
                    if (!bb) continue;
                    auto& nb = fn->add_block(bb->name());
                    for (auto& inst : bb->instructions()) {
                        if (inst) nb.add_instruction(inst);
                    }
                }
                fn->rebuild_block_index();
                fn->compute_predecessors();
                total_folded += stats.values_folded_constant;
                total_branches += stats.branches_simplified;
                total_blocks += stats.blocks_removed;
                if (stats.values_folded_constant == 0 &&
                    stats.branches_simplified == 0 &&
                    stats.blocks_removed == 0) break;
            }
        }
        if ((total_folded > 0 || total_branches > 0 || total_blocks > 0) &&
            config_.verbose) {
            std::cerr << "  [fold] module-level: folded " << total_folded
                      << " constant(s), " << total_branches << " branch(es), "
                      << "removed " << total_blocks << " unreachable block(s)\n";
        }
    }

    // Functions to process. We sort by call-graph topological order
    // (callees before callers) so that when a caller's inliner runs,
    // its callees have already been optimised. This is critical for
    // the cross_fn.ll pattern: `double_if_pos.ipcp_0_5` (the IPCP
    // clone) must be folded to `ret 10` BEFORE `caller_a`'s inliner
    // runs, so the inliner can inline the folded form.
    //
    // The call graph's SCCs are in reverse-topological order (leaves
    // first), which is exactly what we want — process callees (leaves)
    // before callers. We preserve module order within each SCC for
    // determinism.
    //
    // NOTE: we iterate the work_module (post-cross-function-passes),
    // NOT the original `module` — IPCP/DFE may have added/removed
    // functions.
    std::vector<const ir::Function*> fns;
    {
        // Build the call graph on the post-pre-pass module.
        analysis::CallGraph cg;
        std::vector<std::string> entries;
        for (const auto& fn : work_module->functions()) {
            if (fn && (fn->linkage() == ir::Linkage::External ||
                       fn->linkage() == ir::Linkage::Weak)) {
                entries.push_back(fn->name());
            }
        }
        cg.build(*work_module, entries);
        // Map function name → Function*, for lookup after sorting.
        std::unordered_map<std::string, const ir::Function*> fn_by_name;
        for (auto& fn : work_module->functions()) {
            if (fn) fn_by_name[fn->name()] = fn.get();
        }
        // Walk SCCs in reverse-topo order (callees first).
        for (const auto& scc : cg.sccs()) {
            for (const auto& name : scc.members) {
                auto it = fn_by_name.find(name);
                if (it != fn_by_name.end()) fns.push_back(it->second);
            }
        }
        // Safety net: if the call graph missed any function (e.g.
        // indirect calls), append them in module order.
        for (auto& fn : work_module->functions()) {
            if (!fn) continue;
            if (std::find(fns.begin(), fns.end(), fn.get()) == fns.end()) {
                fns.push_back(fn.get());
            }
        }
    }
    const size_t total_fns = fns.size();
    module_total_functions_ = total_fns;
    std::vector<PipelineResult::FunctionResult> fn_results(total_fns);

    // Optimise fns[i] with the given per-thread search state, or pass it
    // through unchanged when the wall-clock budget is exhausted.
    auto process = [&](size_t i, Searchers& s) {
        const ir::Function& fn = *fns[i];
        report_progress("pipeline", fn.name(),
                        static_cast<double>(i) / static_cast<double>(total_fns));
        // Fire a "start" rich progress event so the TUI can mark this
        // function as in-progress before any rounds run. Include the
        // module-level progress (i/total_fns) so the TUI can render an
        // overall progress bar.
        if (rich_progress_cb_) {
            RichProgressEvent ev;
            ev.stage = "start";
            ev.function_name = fn.name();
            ev.progress = static_cast<double>(i) / static_cast<double>(total_fns);
            ev.current_best_ir = fn.to_string();
            ev.round = 0;
            ev.improvements_adopted = 0;
            auto orig_analysis = eval_engine_.analyse(fn);
            ev.score_original = orig_analysis.score;
            ev.score_current = orig_analysis.score;
            ev.improvement_ratio = 1.0;
            ev.verified = false;
            ev.message = "started";
            ev.elapsed_ms = 0.0;
            report_rich_progress(ev);
        }
        if (has_deadline_ && remaining_seconds() <= 0.0) {
            PipelineResult::FunctionResult r;
            r.function_name = fn.name();
            r.original = std::make_shared<ir::Function>(fn);
            r.optimised = std::make_shared<ir::Function>(fn);
            r.score_original = eval_engine_.analyse(fn).score;
            r.score_optimised = r.score_original;
            r.improvement_ratio = 1.0;
            // Fire a "skip" rich progress event so the TUI doesn't
            // leave this function stuck on "pending".
            module_done_functions_.fetch_add(1, std::memory_order_relaxed);
            if (rich_progress_cb_) {
                RichProgressEvent ev;
                ev.stage = "skip";
                ev.function_name = fn.name();
                ev.progress = static_cast<double>(i + 1) / static_cast<double>(total_fns);
                ev.current_best_ir = fn.to_string();
                ev.message = "skipped (time budget exhausted)";
                report_rich_progress(ev);
            }
            fn_results[i] = std::move(r);
            return;
        }

        // ── Interprocedural pre-pass: inline small callees.
        // Two tiers:
        //   1. Single-block inliner (legacy): inlines trivial wrappers
        //      and constant folders — exact by construction and
        //      cost-gated. Removes opaque calls so SMT can verify the
        //      caller.
        //   2. Multi-block inliner (opt-in via enable_multiblock_inliner):
        //      handles multi-block callees, allocas, and recursive call
        //      graphs (with SCC + depth guards). Same cost-gating.
        // Both tiers consult the work_module (post-cross-function-passes)
        // for callee bodies — IPCP/DFE may have changed what's available.
        std::shared_ptr<ir::Function> inlined;
        if (config_.enable_inliner && config_.opt_level >= 1) {
            search::Inliner inliner;
            inlined = inliner.inline_calls(fn, *work_module);
            if (inlined &&
                !(eval_engine_.score_candidate(fn, *inlined) > 1.0)) {
                inlined = nullptr;  // not a win — keep the original
            }
            if (inlined && config_.verbose) {
                std::cerr << "  [inline] " << fn.name() << ": inlined "
                          << inliner.stats().call_sites_inlined
                          << " call site(s)\n";
            }
        }

        // ── Multi-block inliner (cross-function extension) ──────────
        // Runs after the single-block inliner, so it sees callees the
        // single-block path couldn't handle. Cost-gated like above.
        if (config_.enable_multiblock_inliner && config_.opt_level >= 2 &&
            remaining_seconds() > 0.05) {
            const ir::Function& mb_base = inlined ? *inlined : fn;
            search::InlinerConfig mcfg;
            mcfg.enable_multiblock = true;
            search::Inliner mbinliner(mcfg);
            auto mb_inlined = mbinliner.inline_calls_multiblock(
                mb_base, *work_module);
            if (mb_inlined &&
                !(eval_engine_.score_candidate(mb_base, *mb_inlined) > 1.0)) {
                mb_inlined = nullptr;
            }
            if (mb_inlined) {
                if (config_.verbose) {
                    std::cerr << "  [inline-mb] " << fn.name()
                              << ": inlined "
                              << mbinliner.stats().multiblock_inlined
                              << " multi-block call site(s)\n";
                }
                inlined = mb_inlined;
            }
        }

        fn_results[i] = run_on_function(inlined ? *inlined : fn, s);

        // Report honestly against the TRUE original when inlining changed
        // the baseline run_on_function saw.
        if (inlined) {
            auto& r = fn_results[i];
            r.original = std::make_shared<ir::Function>(fn);
            r.score_original = eval_engine_.analyse(fn).score;
            r.improvement_ratio =
                eval_engine_.score_candidate(fn, *r.optimised);
        }
    };

    // Worker count: 0 = auto (hardware_concurrency), clamped to [1,32] and to
    // the number of functions. 1 = sequential (deterministic).
    size_t nthreads = config_.num_threads;
    if (nthreads == 0) {
        nthreads = std::thread::hardware_concurrency();
        if (nthreads == 0) nthreads = 1;
        if (nthreads > 32) nthreads = 32;
    }
    if (total_fns > 0) nthreads = std::min(nthreads, total_fns);

    if (nthreads <= 1) {
        for (size_t i = 0; i < total_fns; ++i) process(i, searchers_);
    } else {
        // ── Shared thread pool for hole-synth work-stealing ───────────
        // Create a shared pool that hole_synth_phase can attach to. The
        // pool's workers are the "idle threads" that help search
        // in-progress functions' codespace when enable_per_function_threads
        // is true. The module-level function dispatch still uses the
        // dedicated-worker pattern below (each worker has its own
        // Searchers, pulls the next function index atomically) — the
        // shared pool is for hole-synth subtasks only.
        //
        // Why not use the shared pool for function dispatch too? The
        // dedicated-worker pattern lets each worker reuse its own
        // Searchers across multiple functions (constructed once, used
        // for all functions that worker processes). Using the pool
        // would require thread_local Searchers, which works but
        // complicates the lifetime story. The dedicated-worker pattern
        // is simpler and proven.
        //
        // The shared pool has `nthreads` workers. When a dedicated
        // worker calls hole_synth_phase, it submits subtasks to this
        // pool. The pool's workers pick them up. When ALL pool workers
        // are busy, the submitting dedicated worker uses run_until() to
        // drain the queue on its own thread.
        if (config_.enable_per_function_threads) {
            shared_pool_ = std::make_unique<search::ThreadPool>(nthreads);
        }

        // Each worker pulls the next function index and optimises it with its
        // OWN search state. The evaluator (thread-safe cache) and pattern
        // library (read-only) are shared.
        std::atomic<size_t> next{0};
        auto worker = [&]() {
            Searchers local(config_, &eval_engine_);
            // Propagate the shared rewrite cache and STOKE-style
            // search flags to this worker's SMTVerifier / search.
            if (rewrite_cache_) {
                local.smt.config().cache = rewrite_cache_;
            }
            local.stochastic.config().allow_unsound_mutations =
                config_.allow_unsound_mutations;
            local.stochastic.config().test_vector_count =
                config_.test_vector_count;
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= total_fns) break;
                process(i, local);
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(nthreads);
        for (size_t t = 0; t < nthreads; ++t) workers.emplace_back(worker);
        for (auto& w : workers) w.join();
        if (config_.verbose) {
            std::cerr << "  [parallel] optimised " << total_fns << " function(s) across "
                      << nthreads << " threads\n";
        }

        // Tear down the shared pool now that the dedicated workers are
        // done — its workers are no longer needed.
        shared_pool_.reset();
    }

    // Assemble the output module in module order.
    for (size_t i = 0; i < total_fns; ++i) {
        result.function_results[fns[i]->name()] = fn_results[i];
        result.optimised_module->add_function(
            fn_results[i].optimised ? fn_results[i].optimised
                                    : std::make_shared<ir::Function>(*fns[i]));
    }

    // ── Post-optimisation cleanup: re-run inlining + folding ────────
    // After per-function optimisation, callees may have been simplified
    // (e.g. an IPCP clone folded to `ret 10`). But the inliner for
    // callers ran BEFORE the callees were optimised (in parallel or
    // just earlier in the sequence). This cleanup pass re-runs the
    // inliner on the now-optimised module, so callers can inline the
    // simplified callees. Then it re-folds to collapse the inlined
    // constants. Repeats up to 3 times to catch multi-level call
    // chains (main → caller_a → double_if_pos.ipcp_0_5).
    //
    // Sound by construction (inlining + prune_dataflow + DFE are all
    // exact). Only runs when the multi-block inliner is enabled.
    if (config_.enable_multiblock_inliner && config_.opt_level >= 2 &&
        result.optimised_module->function_count() > 1) {
        for (int cleanup_round = 0; cleanup_round < 3; ++cleanup_round) {
            bool any_change = false;

            // 1. Re-run the multi-block inliner on each function.
            for (auto& fn : result.optimised_module->functions()) {
                if (!fn) continue;
                // Skip declarations.
                if (fn->blocks().empty()) continue;
                search::InlinerConfig mcfg;
                mcfg.enable_multiblock = true;
                search::Inliner mbinliner(mcfg);
                auto inlined = mbinliner.inline_calls_multiblock(
                    *fn, *result.optimised_module);
                if (!inlined) continue;
                // Cost gate: only adopt if cheaper.
                if (eval_engine_.score_candidate(*fn, *inlined) <= 1.0) continue;
                // Replace the function's body with the inlined version.
                auto& blocks = const_cast<std::vector<std::shared_ptr<ir::BasicBlock>>&>(
                    fn->blocks());
                blocks.clear();
                for (auto& bb : inlined->blocks()) {
                    if (!bb) continue;
                    auto& nb = fn->add_block(bb->name());
                    for (auto& inst : bb->instructions()) {
                        if (inst) nb.add_instruction(inst);
                    }
                }
                fn->rebuild_block_index();
                fn->compute_predecessors();
                any_change = true;
                if (config_.verbose) {
                    std::cerr << "  [cleanup-inline] " << fn->name()
                              << ": inlined "
                              << mbinliner.stats().multiblock_inlined
                              << " call site(s)\n";
                }
            }

            // 2. Re-run prune_dataflow on each function to fold the
            // inlined constants. Also merge blocks connected by
            // unconditional branches (the inliner produces chains of
            // `br label %X` blocks that need to be collapsed).
            for (auto& fn : result.optimised_module->functions()) {
                if (!fn) continue;
                if (fn->blocks().empty()) continue;

                // Merge blocks connected by unconditional branches.
                // A block whose only instruction is `br label %dest`
                // can be replaced: all predecessors that branch to it
                // are rewritten to branch to %dest instead, then the
                // block is removed. Iterate to a fixpoint.
                for (int merge_iter = 0; merge_iter < 10; ++merge_iter) {
                    bool merged_any = false;
                    auto& blocks = const_cast<std::vector<std::shared_ptr<ir::BasicBlock>>&>(
                        fn->blocks());
                    // Build a map: block_name → replacement block_name
                    // (for chains like A→B→C, A should be replaced by C).
                    std::unordered_map<std::string, std::string> merge_map;
                    for (auto& bb : blocks) {
                        if (!bb || bb->size() != 1) continue;
                        auto term = bb->terminator();
                        if (!term || term->opcode() != ir::Opcode::Br) continue;
                        // Unconditional branch has metadata "dest_bb".
                        auto it = term->metadata().find("dest_bb");
                        if (it == term->metadata().end()) continue;
                        if (it->second == bb->name()) continue;  // self-loop
                        merge_map[bb->name()] = it->second;
                    }
                    if (merge_map.empty()) break;
                    // Resolve chains: A→B→C becomes A→C.
                    for (auto& [from, to] : merge_map) {
                        std::string cur = to;
                        while (true) {
                            auto mit = merge_map.find(cur);
                            if (mit == merge_map.end()) break;
                            // Check for cycles.
                            if (mit->second == from || mit->second == to) break;
                            cur = mit->second;
                        }
                        to = cur;
                    }
                    // Rewrite all branch targets in remaining blocks.
                    for (auto& bb : blocks) {
                        if (!bb) continue;
                        for (auto& inst : bb->instructions()) {
                            if (!inst || inst->opcode() != ir::Opcode::Br) continue;
                            auto& md = const_cast<std::unordered_map<std::string, std::string>&>(
                                inst->metadata());
                            for (auto& [key, val] : md) {
                                if (key == "dest_bb" || key == "true_bb" ||
                                    key == "false_bb") {
                                    auto mit = merge_map.find(val);
                                    if (mit != merge_map.end()) {
                                        val = mit->second;
                                        merged_any = true;
                                    }
                                }
                            }
                        }
                    }
                    // Rewrite phi node incoming blocks.
                    for (auto& bb : blocks) {
                        if (!bb) continue;
                        for (auto& inst : bb->instructions()) {
                            if (!inst || inst->opcode() != ir::Opcode::Phi) continue;
                            auto& md = const_cast<std::unordered_map<std::string, std::string>&>(
                                inst->metadata());
                            auto pit = md.find("phi_blocks");
                            if (pit == md.end()) continue;
                            std::string new_blocks;
                            bool changed = false;
                            // Split comma-separated list, replace each.
                            std::string s = pit->second;
                            size_t start = 0;
                            while (start < s.size()) {
                                size_t comma = s.find(',', start);
                                std::string token = (comma == std::string::npos)
                                    ? s.substr(start) : s.substr(start, comma - start);
                                // Trim whitespace.
                                while (!token.empty() && isspace(token.front())) token.erase(0, 1);
                                while (!token.empty() && isspace(token.back())) token.pop_back();
                                auto mit = merge_map.find(token);
                                if (mit != merge_map.end()) {
                                    token = mit->second;
                                    changed = true;
                                }
                                if (!new_blocks.empty()) new_blocks += ",";
                                new_blocks += token;
                                if (comma == std::string::npos) break;
                                start = comma + 1;
                            }
                            if (changed) pit->second = new_blocks;
                        }
                    }
                    // Remove the merged-away blocks.
                    std::vector<std::string> to_remove;
                    for (auto& bb : blocks) {
                        if (!bb) continue;
                        if (merge_map.count(bb->name())) {
                            to_remove.push_back(bb->name());
                        }
                    }
                    for (auto& name : to_remove) {
                        fn->remove_block(name);
                        any_change = true;
                    }
                    fn->rebuild_block_index();
                    fn->compute_predecessors();
                    if (!merged_any) break;
                }

                // Now run prune_dataflow to fold constants.
                for (int iter = 0; iter < 3; ++iter) {
                    ir::PruneStats stats;
                    auto pruned = ir::prune_dataflow(*fn, &stats);
                    if (!pruned) break;
                    auto& blocks = const_cast<std::vector<std::shared_ptr<ir::BasicBlock>>&>(
                        fn->blocks());
                    blocks.clear();
                    for (auto& bb : pruned->blocks()) {
                        if (!bb) continue;
                        auto& nb = fn->add_block(bb->name());
                        for (auto& inst : bb->instructions()) {
                            if (inst) nb.add_instruction(inst);
                        }
                    }
                    fn->rebuild_block_index();
                    fn->compute_predecessors();
                    if (stats.values_folded_constant > 0 ||
                        stats.branches_simplified > 0 ||
                        stats.blocks_removed > 0) {
                        any_change = true;
                    }
                    if (stats.values_folded_constant == 0 &&
                        stats.branches_simplified == 0 &&
                        stats.blocks_removed == 0) break;
                }
            }

            // 3. Re-run DFE to remove functions that are no longer
            // called after inlining.
            {
                search::CrossFnConfig xcfg;
                xcfg.enable_dfe = true;
                xcfg.enable_ipcp = false;
                search::CrossFunctionPasses xpass(xcfg);
                if (xpass.run(*result.optimised_module)) {
                    any_change = true;
                    if (config_.verbose) {
                        std::cerr << "  [cleanup-dfe] removed "
                                  << xpass.stats().dfe_removed
                                  << " dead function(s)\n";
                    }
                }
            }

            if (!any_change) break;
        }

        // Update function_results with the post-cleanup versions.
        for (auto& [name, fr] : result.function_results) {
            auto fn = result.optimised_module->function(name);
            if (fn && fr.optimised) {
                fr.optimised = std::make_shared<ir::Function>(*fn);
                fr.score_optimised = eval_engine_.analyse(*fn).score;
                // Recompute improvement ratio.
                if (fr.score_original != 0.0) {
                    double so = fr.score_original, sp = fr.score_optimised;
                    constexpr double EPS = 1e-9;
                    if (std::abs(sp - so) < EPS) fr.improvement_ratio = 1.0;
                    else if (so <= 0.0 && sp <= 0.0)
                        fr.improvement_ratio = (std::abs(sp) < EPS) ? 1e6 :
                                                std::abs(so) / std::abs(sp);
                    else if (so >= 0.0 && sp >= 0.0)
                        fr.improvement_ratio = (std::abs(so) < EPS) ? 1e6 :
                                                sp / so;
                    else fr.improvement_ratio = sp / so;
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.total_functions_processed = result.function_results.size();

    // Compute aggregate statistics
    double total_improvement = 0.0;
    size_t improved_count = 0;
    for (auto& [name, fr] : result.function_results) {
        if (fr.improvement_ratio > 1.0) {
            total_improvement += fr.improvement_ratio;
            ++improved_count;
        }
    }
    result.total_optimised = improved_count;
    result.avg_improvement = (improved_count > 0)
        ? total_improvement / static_cast<double>(improved_count)
        : 1.0;

    // Copy globals to the output module
    for (auto& gv : module.globals()) {
        result.optimised_module->add_global(gv);
    }

    report_progress("pipeline", "done", 1.0);

    // Disarm the deadline so later direct run_on_function() calls arm
    // their own fresh budget.
    has_deadline_ = false;

    return result;
}

Pipeline::~Pipeline() = default;

// ── run_on_function ─────────────────────────────────────────────────────────

PipelineResult::FunctionResult Pipeline::run_on_function(const ir::Function& fn) {
    // Public entry point: use the shared sequential search state.
    return run_on_function(fn, searchers_);
}

PipelineResult::FunctionResult Pipeline::run_on_function(const ir::Function& fn,
                                                         Searchers& s) {
    PipelineResult::FunctionResult result;
    result.function_name = fn.name();
    result.original = std::make_shared<ir::Function>(fn);

    // Declarations (no body) have nothing to optimise — pass through
    // without burning search rounds on them (clang modules typically
    // declare a handful of intrinsics).
    if (fn.blocks().empty()) {
        result.optimised = std::make_shared<ir::Function>(fn);
        result.improvement_ratio = 1.0;
        // Fire a "skip" rich progress event so the TUI doesn't leave
        // declarations stuck on "pending".
        module_done_functions_.fetch_add(1, std::memory_order_relaxed);
        if (rich_progress_cb_) {
            RichProgressEvent ev;
            ev.stage = "skip";
            ev.function_name = fn.name();
            ev.progress = 1.0;
            ev.message = "declaration (no body)";
            report_rich_progress(ev);
        }
        return result;
    }

    // ── Function-size gate ────────────────────────────────────────
    // The mutation search's cost is super-linear in function size and the
    // SMT encoding goes exponential on branch-heavy CFGs, so functions above
    // max_function_size don't get the full search stack. But slice
    // HARVESTING scales linearly, so instead of passing such functions
    // through entirely untouched, they get a miner-only refinement path.
    // Whole-function SMT re-verification cannot model functions this large,
    // so miner wins are only adoptable when the user opted into
    // trust_unverified; without it (or above the mining cap) the function
    // is passed through unchanged. Sound either way: we never emit an
    // unproven candidate beyond what the trust level allows.
    const size_t inst_count = count_instructions(fn);
    const size_t SKIP_THRESHOLD = config_.max_function_size;
    const bool oversized = inst_count > SKIP_THRESHOLD;
    if (oversized) {
        const bool miner_can_run =
            config_.opt_level >= 2 && config_.enable_peephole_miner &&
            !config_.skip_smt && search::SMTVerifier::is_z3_available() &&
            config_.max_mining_function_size > 0 &&
            inst_count <= config_.max_mining_function_size &&
            (config_.trust_unverified ||
             (inst_count <= config_.smt_config.max_instructions_for_smt &&
              fn.blocks().size() <= config_.smt_config.max_blocks_for_smt));
        if (!miner_can_run) {
            result.optimised = std::make_shared<ir::Function>(fn);
            result.score_original = eval_engine_.analyse(fn).score;
            result.score_optimised = result.score_original;
            result.improvement_ratio = 1.0;
            result.time_spent_ms = 0.0;
            if (config_.verbose) {
                std::cerr << "  [skip] " << fn.name() << " (" << inst_count
                          << " instructions > " << SKIP_THRESHOLD << ")\n";
            }
            // Fire a "skip" rich progress event so the TUI doesn't
            // leave this function stuck on "pending".
            module_done_functions_.fetch_add(1, std::memory_order_relaxed);
            if (rich_progress_cb_) {
                RichProgressEvent ev;
                ev.stage = "skip";
                ev.function_name = fn.name();
                ev.progress = 1.0;
                ev.current_best_ir = fn.to_string();
                ev.message = "skipped (oversized: " +
                             std::to_string(inst_count) + " > " +
                             std::to_string(SKIP_THRESHOLD) + ")";
                ev.elapsed_ms = 0.0;
                report_rich_progress(ev);
            }
            return result;
        }
        if (config_.verbose) {
            std::cerr << "  [miner-only] " << fn.name() << " (" << inst_count
                      << " instructions > " << SKIP_THRESHOLD
                      << " — slice mining only)\n";
        }
    }

    auto fn_start = std::chrono::high_resolution_clock::now();

    // Reset the per-function guards.
    s.last_mined_hash = 0;
    s.last_egraph_hash = 0;
    s.last_vector_hash = 0;
    s.last_loop_hash = 0;
    s.last_mem_hash = 0;
    s.last_prune_hash = 0;
    s.last_pow2_hash = 0;
    s.last_hole_hash = 0;
    s.last_block_hash = 0;
    s.last_series_hash = 0;
    s.rejected_hashes.clear();

    // Fire a "start" rich progress event so the TUI can mark this
    // function as in-progress before any rounds run.
    if (rich_progress_cb_) {
        RichProgressEvent ev;
        ev.stage = "start";
        ev.function_name = fn.name();
        ev.progress = 0.0;
        ev.current_best_ir = fn.to_string();
        ev.round = 0;
        ev.improvements_adopted = 0;
        auto orig_analysis = eval_engine_.analyse(fn);
        ev.score_original = orig_analysis.score;
        ev.score_current = orig_analysis.score;
        ev.improvement_ratio = 1.0;
        ev.verified = false;
        ev.message = "started";
        ev.elapsed_ms = 0.0;
        report_rich_progress(ev);
    }

    // Arm a local deadline if run_on_function() was called directly
    // (not via run(), which arms the shared one).
    bool local_deadline = false;
    if (!has_deadline_ && config_.time_budget > 0.0) {
        deadline_ = std::chrono::steady_clock::now() +
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(config_.time_budget));
        has_deadline_ = true;
        local_deadline = true;
    }

    // Stage 1: Apply patterns from the pattern library (iteratively).
    // Skipped on the miner-only path: pattern rewrites are adopted on the
    // evaluator's word alone, which is a trust level the oversized path
    // hasn't earned — the miner's per-slice SMT proofs have.
    report_progress("patterns", fn.name(), 0.0);
    auto after_patterns =
        oversized ? nullptr : apply_patterns(fn, result);

    // The refinement baseline: best-known-so-far version of the function.
    std::shared_ptr<ir::Function> current =
        after_patterns ? after_patterns : std::make_shared<ir::Function>(fn);

    // ── Continuous refinement rounds (souper/minotaur style) ────────────
    // Each round runs the full search stack from `current`, verifies the
    // top candidates against the ORIGINAL function (the soundness anchor
    // for the whole chain), and adopts the best improvement as the new
    // baseline. There is no fixed optimisation cap: rounds continue until
    // `convergence_rounds` consecutive rounds fail to improve, until
    // `max_rounds` (if set), or until the time budget runs out.
    if (config_.opt_level >= 2) {
        const size_t convergence =
            std::max<size_t>(1, config_.convergence_rounds);
        size_t round = 0;
        size_t stagnant = 0;

        while (stagnant < convergence) {
            if (config_.max_rounds > 0 && round >= config_.max_rounds) break;
            // Leave a little headroom; a round that can't run for at
            // least ~50ms won't find anything.
            if (remaining_seconds() <= 0.05) break;

            // ── Per-function time cap ─────────────────────────────────
            // If max_time_per_function is set, stop refining this function
            // once we've spent that long on it. This prevents one stuck
            // function from monopolizing the budget while other functions
            // starve.
            if (config_.max_time_per_function > 0.0) {
                double fn_elapsed = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - fn_start).count();
                if (fn_elapsed >= config_.max_time_per_function) {
                    if (config_.verbose) {
                        std::cerr << "  [cap] " << fn.name() << ": hit per-function time cap ("
                                  << config_.max_time_per_function << "s)\n";
                    }
                    break;
                }
            }

            report_progress("round", fn.name(),
                            1.0 - 1.0 / static_cast<double>(round + 2));

            // Stage 1.5: vector-intrinsic synthesis (SMT-proved SIMD
            // rewrites on functions containing vector operations; inert
            // otherwise). Runs FIRST: it is deterministic and cheap (one
            // idiom scan + one SMT proof), and on vector functions its
            // candidate is usually the decisive one — running it after the
            // mutation searches would let them exhaust the round budget
            // before it ever fires.
            auto vector_candidates = vector_phase(*current, s);

            // Stage 1.6: loop optimisation (LICM + constant-trip full
            // unrolling), Stage 1.7: alias-aware memory optimisation,
            // Stage 1.8: dataflow pruning (DCE + same-block CSE +
            // unreachable-block elimination + known-bits constant/branch
            // folding), and Stage 1.9: powers-of-two strength reduction
            // (`x urem y` -> `x and (y-1)` when y is provably a nonzero
            // power of two, including at runtime). All four are
            // deterministic, sound-by-construction, and cheap — and their
            // rewrites COMPOUND with the rest of the round: an unrolled
            // loop or a forwarded load becomes straight-line integer code
            // the miner can then prove rewrites for, and a smaller pruned
            // baseline is cheaper for every phase for the REST of this
            // round and every round after.
            {
                auto loop_candidates = loop_phase(*current, s);
                vector_candidates.insert(
                    vector_candidates.end(),
                    std::make_move_iterator(loop_candidates.begin()),
                    std::make_move_iterator(loop_candidates.end()));
                auto mem_candidates = mem_phase(*current, s);
                vector_candidates.insert(
                    vector_candidates.end(),
                    std::make_move_iterator(mem_candidates.begin()),
                    std::make_move_iterator(mem_candidates.end()));
                auto prune_candidates = dataflow_prune_phase(*current, s);
                vector_candidates.insert(
                    vector_candidates.end(),
                    std::make_move_iterator(prune_candidates.begin()),
                    std::make_move_iterator(prune_candidates.end()));
                auto pow2_candidates = pow2_phase(*current, s);
                vector_candidates.insert(
                    vector_candidates.end(),
                    std::make_move_iterator(pow2_candidates.begin()),
                    std::make_move_iterator(pow2_candidates.end()));
                // Series expansion: detect arithmetic-series loops and
                // replace with closed forms (e.g. sum-of-i → n*(n-1)/2).
                // Sound by construction; inert on loop-free functions.
                auto series_candidates = series_phase(*current, s);
                vector_candidates.insert(
                    vector_candidates.end(),
                    std::make_move_iterator(series_candidates.begin()),
                    std::make_move_iterator(series_candidates.end()));
            }

            // Stage 2: Stochastic search from the current baseline.
            // (Miner-only path: the mutation search is super-linear in
            // function size, so oversized functions skip straight to the
            // linear-cost slice mining below.)
            auto stochastic_candidates =
                oversized ? std::vector<search::Candidate>{}
                          : stochastic_phase(*current, remaining_seconds(), s);

            // Stage 3: Evolutionary search seeded with stochastic results
            auto all_candidates =
                oversized ? std::vector<search::Candidate>{}
                          : evolutionary_phase(*current, stochastic_candidates,
                                               remaining_seconds(), s);

            // Merge (evolutionary results first: usually higher quality)
            all_candidates.insert(all_candidates.end(),
                                  std::make_move_iterator(stochastic_candidates.begin()),
                                  std::make_move_iterator(stochastic_candidates.end()));
            stochastic_candidates.clear();
            stochastic_candidates.shrink_to_fit();

            // Stage 3.5: SMT-verified peephole mining from the current baseline.
            // This is the lever that finds wins the mutation search (a subset of
            // provably-cheaper integer slice rewrites.
            auto mining_candidates = mining_phase(*current, s);
            all_candidates.insert(all_candidates.end(),
                                  std::make_move_iterator(mining_candidates.begin()),
                                  std::make_move_iterator(mining_candidates.end()));

            // Merge the Stage-1.5 vector-synthesis candidate (collected
            // before the mutation searches so it cannot be starved).
            all_candidates.insert(all_candidates.end(),
                                  std::make_move_iterator(vector_candidates.begin()),
                                  std::make_move_iterator(vector_candidates.end()));

            // ── Equality-saturation candidate generation ──────────────
            // Guarded by last_egraph_hash — skips re-running on an
            // unchanged baseline (eliminates the repeated-identical-candidate
            // pathology). Skipped on the miner-only path
            // (e-graph lowering cost grows with function size, and its
            // candidates are pattern-derived, not slice-proven).
            if (!oversized) {
                auto egraph_candidates = egraph_phase(*current, s);
                all_candidates.insert(all_candidates.end(),
                                      std::make_move_iterator(egraph_candidates.begin()),
                                      std::make_move_iterator(egraph_candidates.end()));
            }

            // ── Hole-based progressive-deepening synthesis ─────────────
            // Runs AFTER the mutation searches and the miner/e-graph:
            // hole-synth is deterministic and finds the SHORTEST
            // equivalent form, so it's the natural "last resort" when
            // the probabilistic phases didn't find a 1-2 instruction
            // equivalent. Skipped on the miner-only path (it's a whole-
            // function search and needs the function small enough to
            // enumerate).
            if (!oversized) {
                auto hole_candidates = hole_synth_phase(*current, s);
                all_candidates.insert(all_candidates.end(),
                                      std::make_move_iterator(hole_candidates.begin()),
                                      std::make_move_iterator(hole_candidates.end()));
            }

            // Stage 4: Verify against the ORIGINAL, select vs `current`
            auto best = verify_and_select(fn, *current, all_candidates, result, s);

            // ── Block-level divide-and-conquer fallback ──────────────────
            // If no whole-function phase (stochastic, evolutionary, miner,
            // hole-synth, etc.) produced an improving candidate this round,
            // try the block optimiser: split the function into a binary
            // tree of instruction ranges and SMT-prove cheaper equivalents
            // for each. This is the "fallback" that finds wins the whole-
            // function search misses (where the whole function has no short
            // equivalent, but a sub-range does).
            //
            // The block optimiser is deterministic (guarded by
            // last_block_hash) — a given baseline is block-optimised at
            // most once. The returned candidate is sound-by-construction
            // (every adopted range rewrite is SMT-proven equivalent to the
            // input function), so verify_and_select adopts it without
            // re-verification.
            if (!best && !oversized && config_.enable_block_opt) {
                auto block_candidates = block_phase(*current, s);
                if (!block_candidates.empty()) {
                    // Re-run verify_and_select with JUST the block candidate.
                    // `fn` is the original (soundness anchor), `*current` is
                    // the baseline the block candidate must beat.
                    best = verify_and_select(fn, *current, block_candidates,
                                             result, s);
                }
            }

            // Free candidate memory promptly — each candidate holds a
            // deep copy of the function, and the next round rebuilds its
            // own pool.
            all_candidates.clear();
            all_candidates.shrink_to_fit();

            ++round;
            if (best) {
                current = best;
                stagnant = 0;
                ++result.improvements_adopted;
                if (config_.verbose && should_log_verbose("round", fn.name())) {
                    std::cerr << "  [round " << round << "] " << fn.name()
                              << ": adopted improvement"
                              << (result.verified ? " (verified)" : " (unverified)")
                              << "\n";
                }
                // Fire a rich progress event so the TUI (and any other
                // machine consumer) can refresh its display with the
                // new current-best IR snapshot.
                if (rich_progress_cb_) {
                    RichProgressEvent ev;
                    ev.stage = "round";
                    ev.function_name = fn.name();
                    ev.progress = 1.0 - 1.0 / static_cast<double>(round + 2);
                    ev.current_best_ir = current->to_string();
                    ev.round = round;
                    ev.improvements_adopted = result.improvements_adopted;
                    ev.verified = result.verified;
                    auto cur_analysis = eval_engine_.analyse(*current);
                    ev.score_current = cur_analysis.score;
                    auto orig_analysis = eval_engine_.analyse(fn);
                    ev.score_original = orig_analysis.score;
                    double so = orig_analysis.score, sp = cur_analysis.score;
                    constexpr double EPS = 1e-9;
                    if (std::abs(sp - so) < EPS) ev.improvement_ratio = 1.0;
                    else if (so <= 0.0 && sp <= 0.0)
                        ev.improvement_ratio = (std::abs(sp) < EPS) ? 1e6 :
                                                std::abs(so) / std::abs(sp);
                    else if (so >= 0.0 && sp >= 0.0)
                        ev.improvement_ratio = (std::abs(so) < EPS) ? 1e6 :
                                                sp / so;
                    else ev.improvement_ratio = sp / so;
                    ev.message = "adopted improvement";
                    auto fn_elapsed = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - fn_start).count();
                    ev.elapsed_ms = fn_elapsed;
                    report_rich_progress(ev);
                }
            } else {
                ++stagnant;
            }
        }
        result.rounds_run = round;
    }

    result.optimised = current;

    if (local_deadline) has_deadline_ = false;

    // Stage 5 (if GPU): GPU-specific optimisation
#ifdef CLUNK_HAS_GPU
    if (config_.enable_gpu_opt && result.optimised->is_gpu_kernel()) {
        report_progress("gpu", fn.name(), 0.9);
        auto gpu_result = gpu_optimise(*result.optimised);
        if (gpu_result) {
            result.optimised = gpu_result;
        }
    }
#endif

    auto fn_end = std::chrono::high_resolution_clock::now();
    result.time_spent_ms = std::chrono::duration<double, std::milli>(fn_end - fn_start).count();

    // Compute scores
    auto orig_analysis = eval_engine_.analyse(fn);
    auto opt_analysis = eval_engine_.analyse(*result.optimised);
    result.score_original = orig_analysis.score;
    result.score_optimised = opt_analysis.score;

    // Fire a "done" rich progress event so the TUI can show the final
    // optimised IR for this function.
    module_done_functions_.fetch_add(1, std::memory_order_relaxed);
    if (rich_progress_cb_) {
        RichProgressEvent ev;
        ev.stage = "done";
        ev.function_name = fn.name();
        ev.progress = 1.0;
        ev.current_best_ir = result.optimised->to_string();
        ev.round = result.rounds_run;
        ev.improvements_adopted = result.improvements_adopted;
        ev.score_original = result.score_original;
        ev.score_current = result.score_optimised;
        ev.improvement_ratio = result.improvement_ratio;
        ev.verified = result.verified;
        ev.message = "done";
        ev.elapsed_ms = result.time_spent_ms;
        report_rich_progress(ev);
    }

    // Compute the improvement ratio, sign-aware.
    //
    // Convention (matches EvaluationEngine::score_candidate_with_cached_orig):
    //   ratio > 1.0 → improvement (higher score = better)
    //   ratio < 1.0 → regression
    //
    // Scores usually follow score = -cost ≤ 0, where |orig|/|opt| works.
    // But bonus-heavy functions can score POSITIVE, where |orig|/|opt|
    // inverts the meaning (an improvement showed as ratio < 1). Handle
    // the three sign cases explicitly; in every case ratio > 1 iff
    // score_optimised > score_original.
    const double so = result.score_original;
    const double sp = result.score_optimised;
    constexpr double EPS = 1e-9;
    if (std::abs(sp - so) < EPS) {
        result.improvement_ratio = 1.0;
    } else if (so <= 0.0 && sp <= 0.0) {
        // Both are costs: classic cost ratio.
        double orig_abs = std::abs(so), opt_abs = std::abs(sp);
        result.improvement_ratio =
            (opt_abs < EPS) ? 1e6 : orig_abs / opt_abs;
    } else if (so >= 0.0 && sp >= 0.0) {
        // Both are bonuses: bigger is better.
        result.improvement_ratio =
            (so < EPS) ? 1e6 : sp / so;
    } else {
        // Mixed signs: map the score delta into a ratio-like number
        // centred at 1 (monotone in the delta, bounded to (0, 2)).
        result.improvement_ratio =
            1.0 + (sp - so) / (std::abs(so) + std::abs(sp));
    }
    if (!std::isfinite(result.improvement_ratio) ||
        result.improvement_ratio < 0.0) {
        result.improvement_ratio = 1.0;
    }

    return result;
}

// ── apply_patterns ──────────────────────────────────────────────────────────

std::shared_ptr<ir::Function> Pipeline::apply_patterns(
    const ir::Function& fn,
    PipelineResult::FunctionResult& result)
{
    if (config_.opt_level == 0) return nullptr;

    // Apply patterns ITERATIVELY: after one pattern rewrites the function,
    // new patterns may match the result. Keep applying the best match as
    // long as the evaluator agrees it improves, up to a small cap (which
    // also guards against A→B / B→A pattern ping-pong).
    constexpr size_t MAX_PATTERN_PASSES = 8;

    std::shared_ptr<ir::Function> current;
    double current_score = eval_engine_.analyse(fn).score;

    for (size_t pass = 0; pass < MAX_PATTERN_PASSES; ++pass) {
        const ir::Function& base = current ? *current : fn;

        auto matches = pattern_lib_.match(base, config_.target_arch);
        if (matches.empty()) break;

        // Apply the best-matching pattern (highest estimated speedup first)
        std::sort(matches.begin(), matches.end(),
                  [](const pattern::PatternMatch& a, const pattern::PatternMatch& b) {
                      return a.estimated_speedup > b.estimated_speedup;
                  });

        auto& best_match = matches[0];
        auto optimised = pattern_lib_.apply(base, best_match, config_.target_arch);
        if (!optimised) break;

        // Only keep the rewrite if the evaluator scores it strictly better
        // (prevents ping-pong between mutually inverse patterns).
        double new_score = eval_engine_.analyse(*optimised).score;
        if (new_score <= current_score) break;

        current = optimised;
        current_score = new_score;
        ++result.patterns_applied;
        pattern_lib_.record_application(best_match.pattern_id);
    }

    return current;
}

// ── stochastic_phase ────────────────────────────────────────────────────────
//
// opt_level semantics (continuous-refinement era):
//   opt_level 0 → no patterns, no search, no SMT  (cheapest; pass-through)
//   opt_level 1 → patterns only                  (cheap; library-driven)
//   opt_level 2 → continuous refinement rounds   (default)
//   opt_level 3 → continuous refinement with larger per-round budgets
//
// The level no longer bounds how much optimisation a function receives —
// the round loop in run_on_function keeps going until convergence or the
// time budget runs out. The level only sizes each individual round.
//
// User-provided budgets are RESPECTED (capped, not overridden): the level
// preset acts as an upper bound so the unbounded struct defaults stay
// tractable per round, but a smaller user config wins.

std::vector<search::Candidate> Pipeline::stochastic_phase(
    const ir::Function& fn, double remaining, Searchers& s) {
    if (config_.opt_level < 2) return {};
    if (remaining <= 0.0) return {};

    search::StochasticConfig sconfig = config_.stochastic_config;
    const size_t iter_cap = (config_.opt_level >= 3) ? 2000 : 1000;
    const size_t cand_cap = (config_.opt_level >= 3) ? 30 : 20;
    sconfig.max_iterations = std::min(sconfig.max_iterations, iter_cap);
    sconfig.max_candidates = std::min(sconfig.max_candidates, cand_cap);

    // Scale the iteration / candidate counts DOWN for larger
    // functions. Each applied mutation deep-copies the function, so
    // per-iteration cost is O(inst_count). We scale the budget so the
    // *total work* (iterations × inst_count) stays roughly constant.
    const size_t inst_count = count_instructions(fn);
    const size_t cap = config_.max_function_size > 0
                           ? config_.max_function_size
                           : size_t(200);
    // scale ∈ [0.1, 1.0]: 1.0 for tiny fns, ~0.1 for fns near 4× the cap
    // (we already skip anything >cap, so the floor at 0.1 keeps the
    // search from becoming useless for borderline-large functions).
    double scale = std::max(0.1, 1.0 - static_cast<double>(inst_count) /
                                       static_cast<double>(cap * 4));
    sconfig.max_iterations = static_cast<size_t>(
        static_cast<double>(sconfig.max_iterations) * scale);
    sconfig.max_candidates = static_cast<size_t>(
        static_cast<double>(sconfig.max_candidates) * scale);
    // Floors keep the search non-trivial, but never above what the user
    // asked for.
    sconfig.max_iterations = std::max(
        sconfig.max_iterations,
        std::min<size_t>(100, config_.stochastic_config.max_iterations));
    sconfig.max_candidates = std::max(
        sconfig.max_candidates,
        std::min<size_t>(5, config_.stochastic_config.max_candidates));

    // Enable stagnation restarts by default: when the SA chain stops
    // improving for a quarter of the round's budget, re-heat and restart
    // from the baseline. (The struct default of 0 = disabled meant this
    // escape hatch was never used.)
    if (sconfig.stagnation_limit == 0) {
        sconfig.stagnation_limit = std::max<size_t>(50, sconfig.max_iterations / 4);
    }

    // Hand the phase the true remaining budget (still capped at 30s per
    // phase so one round can't starve the rest of the module).
    if (std::isfinite(remaining)) {
        sconfig.time_budget_seconds = std::min(remaining, 30.0);
    }

    s.stochastic.config() = sconfig;
    return s.stochastic.search(fn);
}

// ── evolutionary_phase ──────────────────────────────────────────────────────

std::vector<search::Candidate> Pipeline::evolutionary_phase(
    const ir::Function& fn,
    const std::vector<search::Candidate>& stochastic_candidates,
    double remaining, Searchers& s)
{
    if (config_.opt_level < 2) return {};
    if (remaining <= 0.0) return {};

    // Per-round budget caps by opt level (user-provided smaller budgets
    // are respected; the preset is only an upper bound).
    search::EvolutionaryConfig econfig = config_.evolutionary_config;
    const size_t pop_cap = (config_.opt_level >= 3) ? 30 : 20;
    const size_t gen_cap = (config_.opt_level >= 3) ? 50 : 30;
    econfig.population_size = std::min(econfig.population_size, pop_cap);
    econfig.max_generations = std::min(econfig.max_generations, gen_cap);

    // Scale population/generation counts down for larger
    // functions, and propagate the time budget. Each individual holds a
    // full deep-copy of the function; keeping the population small bounds
    // peak memory and per-generation wall-clock time.
    const size_t inst_count = count_instructions(fn);
    const size_t cap = config_.max_function_size > 0
                           ? config_.max_function_size
                           : size_t(200);
    double scale = std::max(0.1, 1.0 - static_cast<double>(inst_count) /
                                       static_cast<double>(cap * 4));
    econfig.population_size = static_cast<size_t>(
        static_cast<double>(econfig.population_size) * scale);
    econfig.max_generations = static_cast<size_t>(
        static_cast<double>(econfig.max_generations) * scale);
    // Floor: at least 5 individuals / 2 generations so the search is
    // non-trivial, but never above what the user asked for.
    econfig.population_size = std::max(
        econfig.population_size,
        std::min<size_t>(5, config_.evolutionary_config.population_size));
    econfig.max_generations = std::max(
        econfig.max_generations,
        std::min<size_t>(2, config_.evolutionary_config.max_generations));

    if (std::isfinite(remaining)) {
        econfig.time_budget_seconds = std::min(remaining, 30.0);
    }

    s.evolutionary.config() = econfig;
    return s.evolutionary.search(fn, stochastic_candidates);
}

// ── mining_phase ──────────────────────────────────────────────────────────────

std::vector<search::Candidate> Pipeline::mining_phase(const ir::Function& fn,
                                                      Searchers& s) {
    if (config_.opt_level < 2) return {};
    if (!config_.enable_peephole_miner) return {};
    // The miner proves each rewrite with SMT; without a real prover it would be
    // reduced to unsound random testing, so it is disabled in that case (the
    // whole point is a *proven* superoptimisation).
    if (config_.skip_smt || !search::SMTVerifier::is_z3_available()) return {};
    if (remaining_seconds() <= 0.05) return {};

    // Mine each distinct baseline at most once. The miner is deterministic and
    // applies all its non-overlapping wins in a single pass, so re-running it on
    // an unchanged function only burns the round's budget.
    const uint64_t h = search::StochasticSearch::structural_hash(fn);
    if (h == s.last_mined_hash) return {};
    s.last_mined_hash = h;

    search::MinerConfig mcfg = config_.peephole_config;
    if (config_.smt_config.timeout_ms > 0)
        mcfg.smt_timeout_ms = config_.smt_config.timeout_ms;
    // Hand the miner the true remaining wall-clock budget (capped like the
    // other phases so one function can't starve the rest of the module).
    // Without this the slice loop runs unbounded — each slice can spend
    // many SMT calls, which on large functions blows far past the deadline.
    const double rem = remaining_seconds();
    if (std::isfinite(rem)) {
        mcfg.time_budget_seconds = std::max(0.05, std::min(rem, 30.0));
    }
    if (config_.max_time_per_function > 0.0) {
        mcfg.time_budget_seconds = (mcfg.time_budget_seconds > 0.0)
            ? std::min(mcfg.time_budget_seconds, config_.max_time_per_function)
            : config_.max_time_per_function;
    }
    // The miner does its own whole-function SMT re-verification (against `fn`,
    // the current baseline), so the returned rewrite is already proven; mark it
    // sound below rather than re-verifying in verify_and_select. Memory/loop
    // functions the verifier can't model as a whole are only accepted when the
    // pipeline is in best-effort (trust_unverified) mode.
    mcfg.require_smt_reverify = true;
    mcfg.trust_unverified_slices = config_.trust_unverified;

    search::PeepholeMiner miner(&eval_engine_, mcfg);
    bool whole_fn_proven = false;
    // The straight-line window pass gets a third of the phase budget; the
    // harvest/select pass below is the higher-yield path on -O3 code and
    // must not be starved on large functions.
    if (mcfg.time_budget_seconds > 0.0)
        miner.config().time_budget_seconds = mcfg.time_budget_seconds / 3.0;
    auto rewritten = miner.mine_and_rewrite(fn, &whole_fn_proven);

    // ── Souper-style harvesting fallback ────────────────────────────
    // If the straight-line slice miner (mine_and_rewrite) found no win,
    // try the more general harvesting path, which lifts every integer
    // instruction (replacing unsupported operands with opaque vars) and
    // can find wins on functions with memory ops, calls, and phi nodes.
    // With use_path_conditions (default), each slice is additionally mined
    // under the branch conditions dominating its block — the Souper-class
    // lever that finds wins the mutation search cannot. Both paths are SMT-verified.
    if ((!rewritten || !*rewritten) && config_.enable_harvest_miner) {
        // Refresh the budget from the true remaining time for this pass.
        const double rem2 = remaining_seconds();
        if (std::isfinite(rem2))
            miner.config().time_budget_seconds =
                std::max(0.05, std::min(rem2, 30.0));
        rewritten = mcfg.use_path_conditions
                        ? miner.mine_with_path_conditions(fn, &whole_fn_proven)
                        : miner.harvest_and_rewrite(fn, &whole_fn_proven);
        if (rewritten && *rewritten && config_.verbose &&
            should_log_verbose("harvest", fn.name())) {
            std::cerr << "  [harvest] " << fn.name() << ": harvesting found "
                      << miner.stats().harvest_rewrites_applied << " splice(s) ("
                      << miner.stats().pc_slices_mined << " under path conditions)\n";
        }
    }

    if (!rewritten || !*rewritten) return {};

    search::Candidate cand;
    cand.function = *rewritten;
    cand.score = eval_engine_.analyse(**rewritten).score;
    cand.iteration_found = 0;
    cand.description = "peephole-miner (SMT-verified slice rewrite)";
    cand.structural_hash = search::StochasticSearch::structural_hash(**rewritten);
    // Sound iff the miner's whole-function SMT re-verification proved the
    // rewrite Equivalent end-to-end. Rewrites accepted through the
    // trust_unverified_slices escape hatch (memory/loop functions the
    // verifier can't model as a whole) carry per-slice proofs only, so they
    // go through verify_and_select's normal Unknown handling and are
    // reported honestly as unverified.
    cand.sound = whole_fn_proven;

    if (config_.verbose && should_log_verbose("mine", fn.name())) {
        std::cerr << "  [mine] " << fn.name() << ": peephole miner applied "
                  << miner.stats().rewrites_applied << " slice rewrite(s)\n";
    }

    std::vector<search::Candidate> out;
    out.push_back(std::move(cand));
    return out;
}

// ── vector_phase (vector-intrinsic synthesis) ───────────────────────────────

std::vector<search::Candidate> Pipeline::vector_phase(const ir::Function& fn,
                                                       Searchers& s) {
    if (config_.opt_level < 2) return {};
    if (!config_.enable_vector_synth) return {};
    if (remaining_seconds() <= 0.05) return {};
    // Cheap pre-filter: every synthesised form consumes vector values.
    if (!ir::function_has_vector_ops(fn)) return {};
    // Like the miner, the synthesiser is deterministic — run each distinct
    // baseline at most once.
    const uint64_t h = search::StochasticSearch::structural_hash(fn);
    if (h == s.last_vector_hash) return {};
    s.last_vector_hash = h;

    search::VectorSynthConfig vcfg = config_.vector_synth_config;
    if (config_.smt_config.timeout_ms > 0)
        vcfg.smt_timeout_ms = config_.smt_config.timeout_ms;
    // Without a prover the pass returns rewrites only in best-effort mode
    // (they then go through verify_and_select's normal Unknown handling).
    vcfg.trust_unverified =
        config_.trust_unverified || config_.skip_smt ||
        !search::SMTVerifier::is_z3_available();

    search::VectorSynthesizer synth(&eval_engine_, vcfg);
    bool proven = false;
    auto rewritten = synth.synthesize(fn, &proven);
    if (!rewritten) return {};

    search::Candidate cand;
    cand.function = rewritten;
    cand.score = eval_engine_.analyse(*rewritten).score;
    cand.iteration_found = 0;
    cand.description = proven
        ? "vector-intrinsic synthesis (SMT-verified)"
        : "vector-intrinsic synthesis (unverified)";
    cand.structural_hash = search::StochasticSearch::structural_hash(*rewritten);
    cand.sound = proven;

    if (config_.verbose && should_log_verbose("vector", fn.name())) {
        std::cerr << "  [vector] " << fn.name() << ": synthesised "
                  << synth.stats().lane_fusions << " vector op(s), "
                  << synth.stats().reductions << " reduction intrinsic(s), "
                  << synth.stats().shuffle_folds << " shuffle fold(s)"
                  << (proven ? " [proven]" : " [unproven]") << "\n";
    }

    std::vector<search::Candidate> out;
    out.push_back(std::move(cand));
    return out;
}

// ── hole_synth_phase (hole-based progressive-deepening synthesis) ───────────

std::vector<search::Candidate> Pipeline::hole_synth_phase(const ir::Function& fn,
                                                            Searchers& s) {
    if (config_.opt_level < 2) return {};
    if (!config_.enable_hole_synth) return {};
    if (remaining_seconds() <= 0.05) return {};

    // The synthesiser is deterministic for a given (fn, config) — run
    // each distinct baseline at most once.
    const uint64_t h = search::StochasticSearch::structural_hash(fn);
    if (h == s.last_hole_hash) return {};
    s.last_hole_hash = h;

    search::HoleSynthConfig hcfg = config_.hole_synth_config;
    if (config_.smt_config.timeout_ms > 0)
        hcfg.smt_timeout_ms = config_.smt_config.timeout_ms;
    // Cap the hole-synth's internal time budget at the pipeline's
    // remaining wall-clock budget — prevents hole-synth from
    // monopolising the round when no equivalent exists at any depth.
    double rem = remaining_seconds();
    if (rem > 0.0 && rem < hcfg.time_budget_seconds) {
        hcfg.time_budget_seconds = rem;
    }
    // Without a prover the pass returns rewrites only in best-effort mode
    // (they then go through verify_and_select's normal Unknown handling).
    hcfg.trust_unverified = config_.trust_unverified ||
                            config_.skip_smt ||
                            !search::SMTVerifier::is_z3_available();
    // Engage the work-stealing parallel path when a shared pool is
    // available (i.e. num_threads > 1 and enable_per_function_threads).
    // The pool is shared with module-level dispatch — idle module
    // workers help search this function's codespace.
    hcfg.parallel_search = config_.enable_per_function_threads;

    search::HoleSynthesizer synth(&eval_engine_, hcfg);
    if (hcfg.parallel_search && shared_pool_) {
        synth.set_thread_pool(shared_pool_.get());
    }
    bool proven = false;
    auto rewritten = synth.synthesize(fn, &proven);
    if (!rewritten) return {};

    search::Candidate cand;
    cand.function = rewritten;
    cand.score = eval_engine_.analyse(*rewritten).score;
    cand.iteration_found = 0;
    cand.description = proven
        ? "hole-based progressive-deepening synthesis (SMT-verified)"
        : "hole-based progressive-deepening synthesis (unverified)";
    cand.structural_hash = search::StochasticSearch::structural_hash(*rewritten);
    cand.sound = proven;

    if (config_.verbose && should_log_verbose("hole", fn.name())) {
        std::cerr << "  [hole] " << fn.name() << ": synthesised "
                  << synth.stats().candidates_enumerated << " candidate(s), "
                  << synth.stats().candidates_smt_verified << " SMT-checked, "
                  << synth.stats().candidates_equivalent << " equivalent; "
                  << "best depth = " << synth.stats().best_depth
                  << (proven ? " [proven]" : " [unproven]");
        if (synth.stats().parallel_depths_run > 0) {
            std::cerr << " [parallel: " << synth.stats().parallel_depths_run
                      << " depth(s), " << synth.stats().parallel_work_items_dispatched
                      << " work items, ~"
                      << synth.stats().parallel_candidates_per_work_item_avg
                      << " cands/item]";
        }
        std::cerr << "\n";
    }

    std::vector<search::Candidate> out;
    out.push_back(std::move(cand));
    return out;
}

// ── block_phase (block-level divide-and-conquer optimisation) ──────────────

std::vector<search::Candidate> Pipeline::block_phase(const ir::Function& fn,
                                                      Searchers& s) {
    if (config_.opt_level < 2) return {};
    if (!config_.enable_block_opt) return {};
    if (remaining_seconds() <= 0.05) return {};

    // The block optimiser is deterministic for a given (fn, config) —
    // run each distinct baseline at most once.
    const uint64_t h = search::StochasticSearch::structural_hash(fn);
    if (h == s.last_block_hash) return {};
    s.last_block_hash = h;

    search::BlockOptimiserConfig bcfg = config_.block_opt_config;
    if (config_.smt_config.timeout_ms > 0)
        bcfg.smt_timeout_ms = config_.smt_config.timeout_ms;
    // Cap the block optimiser's internal time budget at the pipeline's
    // remaining wall-clock budget — prevents it from monopolising the
    // round when no block equivalent exists.
    double rem = remaining_seconds();
    if (rem > 0.0 && rem < bcfg.time_budget_seconds) {
        bcfg.time_budget_seconds = rem;
    }
    // Without a prover the optimiser returns rewrites only in best-effort
    // mode (they then go through verify_and_select's normal Unknown
    // handling).
    bcfg.trust_unverified = config_.trust_unverified ||
                            config_.skip_smt ||
                            !search::SMTVerifier::is_z3_available();

    search::BlockOptimiser opt(&eval_engine_, bcfg);
    bool proven = false;
    auto rewritten = opt.optimize(fn, &proven);
    if (!rewritten) return {};

    search::Candidate cand;
    cand.function = rewritten;
    cand.score = eval_engine_.analyse(*rewritten).score;
    cand.iteration_found = 0;
    cand.description = proven
        ? "block-level divide-and-conquer (SMT-verified)"
        : "block-level divide-and-conquer (unverified)";
    cand.structural_hash = search::StochasticSearch::structural_hash(*rewritten);
    cand.sound = proven;

    if (config_.verbose && should_log_verbose("block", fn.name())) {
        std::cerr << "  [block] " << fn.name() << ": tried "
                  << opt.stats().blocks_tried << " block(s), optimised "
                  << opt.stats().blocks_optimised << ", natural "
                  << opt.stats().blocks_natural << " ("
                  << opt.stats().candidates_enumerated << " candidate(s), "
                  << opt.stats().candidates_smt_verified << " SMT-checked, "
                  << opt.stats().candidates_equivalent << " equivalent)"
                  << (proven ? " [proven]" : " [unproven]") << "\n";
    }

    std::vector<search::Candidate> out;
    out.push_back(std::move(cand));
    return out;
}

// ── series_phase (series-expansion loop optimisation) ──────────────────────

std::vector<search::Candidate> Pipeline::series_phase(const ir::Function& fn,
                                                        Searchers& s) {
    if (config_.opt_level < 2) return {};
    if (!config_.enable_series_expand) return {};
    if (remaining_seconds() <= 0.05) return {};
    // Cheap pre-filter: only run on functions with back-edges (loops).
    if (!ir::has_back_edge(fn)) return {};

    // The expander is deterministic for a given (fn, config) — run each
    // distinct baseline at most once.
    const uint64_t h = search::StochasticSearch::structural_hash(fn);
    if (h == s.last_series_hash) return {};
    s.last_series_hash = h;

    search::SeriesExpander expander(&eval_engine_, config_.series_expand_config);
    auto rewritten = expander.expand(fn);
    if (!rewritten) return {};

    search::Candidate cand;
    cand.function = rewritten;
    cand.score = eval_engine_.analyse(*rewritten).score;
    cand.iteration_found = 0;
    cand.description = "series expansion (closed-form loop replacement, sound)";
    cand.structural_hash = search::StochasticSearch::structural_hash(*rewritten);
    // Exact by construction: the closed-form algebra is mathematically
    // equivalent to the loop's accumulation (same trust tier as loop_phase
    // / LICM). The expander refuses loops with nsw/nuw flags on the
    // accumulator add, so wrapping semantics match exactly.
    cand.sound = true;

    if (config_.verbose && should_log_verbose("series", fn.name())) {
        std::cerr << "  [series] " << fn.name() << ": expanded loop to closed form ("
                  << expander.stats().loops_expanded << " loop(s), "
                  << expander.stats().loops_matched << " matched, "
                  << expander.stats().loops_rejected << " rejected)\n";
    }

    std::vector<search::Candidate> out;
    out.push_back(std::move(cand));
    return out;
}

// ── loop_phase (LICM + constant-trip full unrolling) ────────────────────────

std::vector<search::Candidate> Pipeline::loop_phase(const ir::Function& fn,
                                                     Searchers& s) {
    if (config_.opt_level < 2) return {};
    if (!config_.enable_loop_opt) return {};
    if (remaining_seconds() <= 0.05) return {};
    if (!ir::has_back_edge(fn)) return {};  // cheap pre-filter

    const uint64_t h = search::StochasticSearch::structural_hash(fn);
    if (h == s.last_loop_hash) return {};
    s.last_loop_hash = h;

    search::LoopOptimizer lopt;
    std::vector<search::Candidate> out;

    auto push = [&](std::shared_ptr<ir::Function> rewritten,
                    const char* what) {
        if (!rewritten) return;
        search::Candidate cand;
        cand.function = rewritten;
        cand.score = eval_engine_.analyse(*rewritten).score;
        cand.iteration_found = 0;
        cand.description = what;
        cand.structural_hash =
            search::StochasticSearch::structural_hash(*rewritten);
        // Exact by construction (pure-instruction hoisting with stripped
        // poison flags / simulation-proved trip counts) — the class of
        // candidate verify_and_select adopts without a prover, which is
        // the only option here: the SMT verifier refuses back-edges.
        cand.sound = true;
        out.push_back(std::move(cand));
    };

    auto hoisted = lopt.hoist_invariants(fn);
    push(hoisted, "loop-invariant code motion (sound)");

    // Unroll from the hoisted version when available — LICM shrinks the
    // body the unroller must clone.
    auto unrolled = lopt.unroll_constant_loops(hoisted ? *hoisted : fn);
    push(unrolled, "constant-trip loop unrolling (sound)");

    if (config_.verbose && !out.empty() && should_log_verbose("loop", fn.name())) {
        std::cerr << "  [loop] " << fn.name() << ": hoisted "
                  << lopt.stats().instructions_hoisted << " instruction(s), unrolled "
                  << lopt.stats().loops_unrolled << " loop(s) ("
                  << lopt.stats().iterations_expanded << " iterations)\n";
    }
    return out;
}

// ── mem_phase (alias-aware memory optimisation) ─────────────────────────────

std::vector<search::Candidate> Pipeline::mem_phase(const ir::Function& fn,
                                                    Searchers& s) {
    if (config_.opt_level < 2) return {};
    if (!config_.enable_mem_opt) return {};
    if (remaining_seconds() <= 0.05) return {};

    const uint64_t h = search::StochasticSearch::structural_hash(fn);
    if (h == s.last_mem_hash) return {};
    s.last_mem_hash = h;

    search::MemOptimizer mopt;
    auto rewritten = mopt.optimize(fn);
    if (!rewritten) return {};

    search::Candidate cand;
    cand.function = rewritten;
    cand.score = eval_engine_.analyse(*rewritten).score;
    cand.iteration_found = 0;
    cand.description = "alias-aware memory optimisation (sound)";
    cand.structural_hash = search::StochasticSearch::structural_hash(*rewritten);
    cand.sound = true;  // exact under the conservative alias oracle

    if (config_.verbose && should_log_verbose("mem", fn.name())) {
        std::cerr << "  [mem] " << fn.name() << ": forwarded "
                  << mopt.stats().loads_forwarded << " store(s)-to-load, eliminated "
                  << mopt.stats().loads_eliminated << " redundant load(s), "
                  << mopt.stats().stores_eliminated << " dead store(s)\n";
    }

    std::vector<search::Candidate> out;
    out.push_back(std::move(cand));
    return out;
}

// ── dataflow_prune_phase ─────────────────────────────────────────────────
// DCE + unreachable-block elimination + known-bits-driven constant/branch
// folding (see clunk/IR/DataflowPrune.h). Sound by construction — same
// trust tier as loop_phase/mem_phase above.

std::vector<search::Candidate> Pipeline::dataflow_prune_phase(const ir::Function& fn,
                                                               Searchers& s) {
    if (config_.opt_level < 2) return {};
    if (!config_.enable_dataflow_prune) return {};
    if (remaining_seconds() <= 0.05) return {};

    const uint64_t h = search::StochasticSearch::structural_hash(fn);
    if (h == s.last_prune_hash) return {};
    s.last_prune_hash = h;

    ir::PruneStats stats;
    auto pruned = ir::prune_dataflow(fn, &stats);
    if (!pruned) return {};

    search::Candidate cand;
    cand.function = pruned;
    cand.score = eval_engine_.analyse(*pruned).score;
    cand.iteration_found = 0;
    cand.description = "dataflow pruning (DCE + known-bits simplify, sound)";
    cand.structural_hash = search::StochasticSearch::structural_hash(*pruned);
    // Every rewrite prune_dataflow performs is exact by construction (dead
    // code has no observable effect by definition; known-bits folds and
    // the branch/block pruning they enable are standard sound peephole
    // transforms) — same trust tier as loop_phase/mem_phase.
    cand.sound = true;

    if (config_.verbose && should_log_verbose("prune", fn.name())) {
        std::cerr << "  [prune] " << fn.name() << ": removed "
                  << stats.instructions_removed << " dead instruction(s), "
                  << stats.blocks_removed << " unreachable block(s); folded "
                  << stats.values_folded_constant << " known-constant value(s), "
                  << stats.comparisons_folded << " comparison(s), "
                  << stats.redundant_masks_removed << " redundant mask(s), "
                  << stats.branches_simplified << " branch(es); deduplicated "
                  << stats.subexpressions_eliminated << " common subexpression(s)\n";
    }

    std::vector<search::Candidate> out;
    out.push_back(std::move(cand));
    return out;
}

// ── pow2_phase (powers-of-two strength reduction) ────────────────────────
// `x urem y` -> `x and (y - 1)` wherever the PowersOfTwo analysis (see
// clunk/Analysis/PowersOfTwo.h) proves y is a nonzero power of two — see
// clunk/IR/StrengthReduce.h for the full soundness argument. Sound by
// construction — same trust tier as loop_phase/mem_phase/
// dataflow_prune_phase above.

std::vector<search::Candidate> Pipeline::pow2_phase(const ir::Function& fn,
                                                     Searchers& s) {
    if (config_.opt_level < 2) return {};
    if (!config_.enable_pow2_strength_reduce) return {};
    if (remaining_seconds() <= 0.05) return {};

    const uint64_t h = search::StochasticSearch::structural_hash(fn);
    if (h == s.last_pow2_hash) return {};
    s.last_pow2_hash = h;

    ir::StrengthReduceStats stats;
    auto rewritten = ir::simplify_pow2_strength_reduce(fn, &stats);
    if (!rewritten) return {};

    search::Candidate cand;
    cand.function = rewritten;
    cand.score = eval_engine_.analyse(*rewritten).score;
    cand.iteration_found = 0;
    cand.description = "powers-of-two strength reduction (urem -> and, sound)";
    cand.structural_hash = search::StochasticSearch::structural_hash(*rewritten);
    // Exact by construction: `x urem y == x and (y-1)` for every possible
    // x whenever y truly is a nonzero power of two, which is exactly what
    // the PowersOfTwo analysis proved before this rewrite fired.
    cand.sound = true;

    if (config_.verbose && should_log_verbose("pow2", fn.name())) {
        std::cerr << "  [pow2] " << fn.name() << ": rewrote "
                  << stats.urem_to_and
                  << " urem-by-power-of-two into and-mask form\n";
    }

    std::vector<search::Candidate> out;
    out.push_back(std::move(cand));
    return out;
}

// ── egraph_phase ────────────────────────────────────────────────────────

std::vector<search::Candidate> Pipeline::egraph_phase(const ir::Function& fn,
                                                        Searchers& s) {
    if (!config_.enable_egraph_phase) return {};
    if (config_.opt_level < 2) return {};
    if (remaining_seconds() <= 0.01) return {};
    // The e-graph rewriter consumes pattern-library rewrites; if the library
    // is empty (or only has built-in seed patterns that don't apply), there
    // is nothing to do.
    if (pattern_lib_.size() == 0) return {};

    // ── Skip e-graph on unchanged baseline ───────────────────────
    // The e-graph is deterministic: given the same input function and the
    // same pattern library, it produces the same output. If the baseline
    // hasn't changed since the last e-graph run, re-running it would
    // produce the same candidate that was already rejected (or adopted),
    // wasting cycles. Skip it.
    const uint64_t h = search::StochasticSearch::structural_hash(fn);
    if (h == s.last_egraph_hash) return {};
    s.last_egraph_hash = h;

    // egraph_rewrite() returns nullopt if the saturated+extracted function
    // is not strictly cheaper than the input.
    auto opt = search::egraph_rewrite(fn, pattern_lib_,
                                       config_.target_arch, eval_engine_);
    if (!opt) return {};

    // ── E-graph candidates are sound-by-construction ──────────────
    // The e-graph only applies pattern-library rewrites. All pattern-library
    // patterns are sound: the built-in seed patterns (add_zero_elim,
    // mul_one_elim, double_negation, strength_reduce_mul, constant_fold,
    // dead_code_elim) are algebraic identities / compile-time folds, and
    // miner-discovered patterns are SMT-proven. The e-graph's merge
    // operation preserves soundness transitively: if A≡B and B≡C are both
    // sound, then A≡C is sound. So the extracted cheapest representative
    // is sound-by-construction.
    //
    // This is CRUCIAL for real-world code: without it, e-graph wins on
    // functions with memory ops or loops (where SMT bails to Unknown) are
    // never adopted — the candidate goes through SMT, gets Unknown, and is
    // rejected. Marking it sound=true lets verify_and_select adopt it
    // without SMT, which is the only way the e-graph can help on code
    // (where most functions have memory ops).
    opt->sound = true;

    if (config_.verbose && should_log_verbose("egraph", fn.name())) {
        std::cerr << "  [egraph] " << fn.name()
                  << ": equality-saturation produced a candidate (score="
                  << opt->score << ")\n";
    }

    std::vector<search::Candidate> out;
    out.push_back(std::move(*opt));
    return out;
}

// ── verify_and_select ───────────────────────────────────────────────────────

std::shared_ptr<ir::Function> Pipeline::verify_and_select(
    const ir::Function& original,
    const ir::Function& current,
    const std::vector<search::Candidate>& candidates,
    PipelineResult::FunctionResult& result,
    Searchers& s)
{
    if (candidates.empty()) return nullptr;

    // Score the current baseline — a candidate must beat THIS to be worth
    // adopting (the baseline itself already beats the original).
    double baseline_score = eval_engine_.analyse(current).score;

    // Evaluate each candidate
    struct ScoredCandidate {
        const search::Candidate* candidate;
        double score;
    };

    std::vector<ScoredCandidate> scored;
    scored.reserve(candidates.size());
    for (auto& cand : candidates) {
        if (!cand.function) continue;
        double cand_score = eval_engine_.analyse(*cand.function).score;
        // Only consider candidates strictly better than the baseline
        // (less negative = better)
        if (cand_score > baseline_score) {
            // ── Skip candidates already SMT-rejected this round-chain ──
            // The candidate's structural hash uniquely identifies it (up to
            // SSA renaming). If we already tried and rejected this exact
            // candidate in a previous round (when the baseline was the same),
            // don't waste another SMT call on it.
            uint64_t chash = cand.structural_hash;
            if (chash == 0) {
                chash = search::StochasticSearch::structural_hash(*cand.function);
            }
            if (s.rejected_hashes.count(chash)) {
                if (config_.verbose && should_log_verbose("verify-skip", original.name())) {
                    std::cerr << "  [verify] " << original.name()
                              << ": skipping already-rejected candidate (hash="
                              << chash << ")\n";
                }
                continue;
            }
            scored.push_back({&cand, cand_score});
        }
    }

    if (scored.empty()) return nullptr;

    // Sort by score (higher = better, since scores are negative)
    std::sort(scored.begin(), scored.end(),
              [](const ScoredCandidate& a, const ScoredCandidate& b) {
                  return a.score > b.score;
              });

    // ── llvm-mca final ranking (opt-in, Minotaur-style) ────────────────
    // Re-rank the top candidates by MEASURED cycles against the current
    // baseline: candidates the machine model can measure and says are
    // NOT faster are dropped; measured candidates order by their cycle
    // ratio (best first) ahead of unmeasurable ones (which keep the
    // built-in cost-model order as the fallback signal). Bounded to the
    // top 8 — each measurement is two subprocess launches (cached).
    if (config_.use_mca_ranker && evaluator::MCACostModel::is_available()) {
        struct MCAScored {
            ScoredCandidate sc;
            double ratio;  // >1 = faster than baseline; 0 = unmeasurable
        };
        std::vector<MCAScored> kept;
        kept.reserve(scored.size());
        size_t measured_budget = 8;
        for (auto& sc : scored) {
            double ratio = 0.0;
            if (measured_budget > 0) {
                --measured_budget;
                ratio = mca_.compare(current, *sc.candidate->function);
            }
            if (ratio > 0.0 && ratio <= 1.0) {
                if (config_.verbose && should_log_verbose("mca-drop", original.name())) {
                    std::cerr << "  [mca] " << original.name()
                              << ": dropped a candidate llvm-mca measures at "
                              << ratio << "x baseline\n";
                }
                continue;  // measured, not actually faster
            }
            kept.push_back({sc, ratio});
        }
        std::stable_sort(kept.begin(), kept.end(),
                         [](const MCAScored& a, const MCAScored& b) {
                             return a.ratio > b.ratio;
                         });
        scored.clear();
        for (auto& k : kept) scored.push_back(k.sc);
        if (scored.empty()) return nullptr;
    }

    // Adopt in order of score. Three tiers of trust:
    //   1. sound-by-construction candidates (derived from the baseline
    //      purely via semantics-preserving rewrites — DCE, CSE, folding,
    //      identities, strength reduction, guarded swaps) need no prover
    //      at all. This is what lets clunk optimise real clang IR, where
    //      memory ops and loops put SMT permanently out of reach.
    //   2. SMT-verified candidates — always proved against the ORIGINAL
    //      function, never against `current`, so every adopted baseline
    //      in the refinement chain is independently anchored.
    //   3. (opt-in) candidates the prover returned Unknown for.
    //      SOUNDNESS GATE: these candidates MUST pass the test-vector
    //      differential (the original is interpretable AND the candidate
    //      agrees on all probes). Without this gate, `trust_unverified`
    //      would adopt unsound rewrites on uninterpretable functions
    //      (e.g. functions with residual calls the inliner couldn't
    //      eliminate) — the Interpreter can't evaluate them, SMT also
    //      returns Unknown, and there's no verification path at all.
    // Candidates PROVED NotEquivalent are never adopted, in any mode.
    bool z3_available = !config_.skip_smt &&
                        search::SMTVerifier::is_z3_available();

    // Helper: is it safe to adopt `candidate` via the trust_unverified
    // tier? Requires the Interpreter differential to pass — the original
    // must be interpretable AND the candidate must agree on all probes.
    // Returns false when the original is uninterpretable (the Interpreter
    // returns nullopt), because in that case SMT also returns Unknown
    // and there is NO verification path. Sound-by-construction candidates
    // bypass this check (they're adopted in tier 1 regardless).
    auto safe_for_trust_unverified = [&](const ir::Function& orig,
                                          const std::shared_ptr<ir::Function>& cand)
        -> bool {
        // Probe values: a small set of "interesting" integers.
        static const int64_t kProbe[] = {
            0, 1, -1, 2, -2, 3, -3, 7, -7, 8, 15, 16, 17,
            127, 128, 255, 256, 1023, 1024,
        };
        constexpr size_t kProbeCount = sizeof(kProbe) / sizeof(kProbe[0]);
        size_t nargs = orig.argument_count();
        size_t n_vectors = config_.test_vector_count > 0
                              ? std::min(config_.test_vector_count, kProbeCount)
                              : kProbeCount;

        for (size_t i = 0; i < n_vectors; ++i) {
            std::vector<int64_t> args(nargs, kProbe[i]);
            for (size_t j = 1; j < nargs; ++j) {
                args[j] = kProbe[(i + j) % kProbeCount];
            }
            auto orig_r = evaluator::Interpreter::interpret(orig, args);
            if (!orig_r) {
                // Original is uninterpretable — NO verification path.
                // SMT also returns Unknown for functions with calls /
                // memory / FP / loops. Reject.
                return false;
            }
            auto cand_r = evaluator::Interpreter::interpret(*cand, args);
            if (!cand_r) {
                // Candidate is uninterpretable but original IS — the
                // candidate introduced an unsupported op. Reject.
                return false;
            }
            if (*orig_r != *cand_r) return false;
        }
        return true;
    };

    if (!z3_available) {
        // No prover available (or explicitly skipped): adopt the best
        // candidate that EITHER is sound-by-construction OR passes the
        // test-vector differential. Unsound candidates on uninterpretable
        // originals are rejected (no verification path exists).
        for (auto& sc : scored) {
            if (sc.candidate->sound) {
                result.verified = true;
                return sc.candidate->function;
            }
            if (safe_for_trust_unverified(original, sc.candidate->function)) {
                result.verified = false;
                return sc.candidate->function;
            }
        }
        return nullptr;
    }

    std::shared_ptr<ir::Function> unknown_fallback;
    size_t smt_attempts = 0;
    // Per-round SMT attempt cap. The historical hard-coded 5 is the
    // default; --max-smt-attempts overrides it. 0 falls back to 5.
    const size_t MAX_SMT_ATTEMPTS =
        config_.max_smt_attempts > 0 ? config_.max_smt_attempts : 5;
    for (auto& sc : scored) {
        if (sc.candidate->sound) {
            result.verified = true;
            return sc.candidate->function;
        }
        if (smt_attempts >= MAX_SMT_ATTEMPTS) continue;
        ++smt_attempts;
        auto vr = s.smt.verify(original, *sc.candidate->function);
        if (vr.is_safe()) {
            // ── Optional Alive2 second opinion ───────────────────────────
            // SMT said this candidate is safe. If --alive2 is enabled,
            // get a second, independent verdict from alive-tv before
            // trusting it. alive-tv can prove things clunk's own encoder
            // structurally cannot (memory, calls, more of the poison/UB
            // model) and parses the candidate through LLVM's real .ll
            // grammar, so a REFUTED verdict here means either an SMT
            // soundness bug or an invalid-IR-emission bug — either way,
            // this candidate must NOT be adopted even though SMT blessed
            // it. Unknown/Error/NotAvailable are not held against the
            // candidate: SMT's proof stands on its own in that case,
            // exactly as it did before --alive2 existed.
            if (config_.enable_alive2) {
                auto ar = s.alive.verify(original, *sc.candidate->function);
                if (ar.status == search::AliveResult::Refuted) {
                    if (config_.verbose) {
                        std::cerr << "  [alive2] " << original.name()
                                  << ": *** SOUNDNESS BUG *** SMT proved "
                                     "Equivalent but alive-tv REFUTED refinement "
                                     "— rejecting candidate ("
                                  << ar.message << ")\n";
                    }
                    uint64_t chash = sc.candidate->structural_hash;
                    if (chash == 0) {
                        chash = search::StochasticSearch::structural_hash(
                            *sc.candidate->function);
                    }
                    s.rejected_hashes.insert(chash);
                    continue;
                }
                if (config_.verbose && ar.status != search::AliveResult::Verified) {
                    std::cerr << "  [alive2] " << original.name() << ": "
                              << ar.message << " (adopting on SMT proof alone)\n";
                }
            }
            result.verified = true;
            return sc.candidate->function;
        }
        // ── Record the rejection so we don't re-try this candidate ──
        uint64_t chash = sc.candidate->structural_hash;
        if (chash == 0) {
            chash = search::StochasticSearch::structural_hash(*sc.candidate->function);
        }
        s.rejected_hashes.insert(chash);

        if (vr.status == search::VerificationResult::NotEquivalent) {
            continue;
        }
        // Unknown / Error: eligible for best-effort adoption below.
        // SOUNDNESS GATE: only accept as fallback if the test-vector
        // differential passes. When the original is uninterpretable
        // (calls / memory / FP / loops), both SMT and the Interpreter
        // return Unknown/nullopt — there is NO verification path, and
        // adopting would be unsound.
        if (config_.trust_unverified && !unknown_fallback &&
            safe_for_trust_unverified(original, sc.candidate->function)) {
            unknown_fallback = sc.candidate->function;
        }
    }
    if (unknown_fallback) {
        result.verified = false;
        return unknown_fallback;
    }
    // Souper-style soundness (default): improving candidates existed but
    // none carried a proof — keep the current baseline.
    if (config_.verbose && should_log_verbose("verify-unprovable", original.name())) {
        std::cerr << "  [verify] " << original.name() << ": " << scored.size()
                  << " improving candidate(s), none provable — rerun with "
                     "--trust-unverified to adopt best-effort\n";
    }
    return nullptr;
}

#ifdef CLUNK_HAS_GPU
// ── gpu_optimise ────────────────────────────────────────────────────────────

std::shared_ptr<ir::Function> Pipeline::gpu_optimise(const ir::Function& fn) {
    gpu::PTXOptimizer ptx_opt(config_.target_arch);

    // Run PTX-level optimisation
    auto optimised = ptx_opt.optimise(fn);

    if (optimised && config_.enable_launch_opt) {
        // Tune kernel launch configuration
        gpu::KernelLaunchOptimizer launch_opt(config_.target_arch);
        gpu::LaunchConfig default_config;
        auto launch_result = launch_opt.optimise_launch(*optimised, default_config);

        // Store the launch config as function attributes
        optimised->set_attribute("gpu.block_x",
                                 std::to_string(launch_result.optimised.block_x));
        optimised->set_attribute("gpu.block_y",
                                 std::to_string(launch_result.optimised.block_y));
        optimised->set_attribute("gpu.block_z",
                                 std::to_string(launch_result.optimised.block_z));
        optimised->set_attribute("gpu.estimated_speedup",
                                 std::to_string(launch_result.estimated_speedup));
        optimised->set_attribute("gpu.launch_rationale",
                                 launch_result.rationale);
    }

    return optimised;
}
#endif

// ── report_progress ─────────────────────────────────────────────────────────

void Pipeline::report_progress(const std::string& stage,
                                const std::string& fn_name,
                                double progress) {
    if (progress_cb_) {
        // Serialised: workers report progress concurrently in the parallel
        // module path, and the user's callback need not be thread-safe.
        std::lock_guard<std::mutex> lock(progress_mutex_);
        progress_cb_(stage, fn_name, progress);
    }
    // Fire the rich callback too (if set) with the minimal info we have
    // at this call site. Callers that want to send the full event
    // (with current_best_ir, round, scores, ...) call report_rich_progress
    // directly.
    if (rich_progress_cb_) {
        RichProgressEvent ev;
        ev.stage = stage;
        ev.function_name = fn_name;
        ev.progress = progress;
        std::lock_guard<std::mutex> lock(progress_mutex_);
        rich_progress_cb_(ev);
    }
}

// ── report_rich_progress (full event, for TUI / machine consumers) ─────────
//
// Call this from any pipeline stage that has a meaningful "current best"
// snapshot to share. It fires the rich callback with the full event,
// and also forwards a stripped-down version to the legacy simple
// callback (so existing stderr progress prints keep working).

void Pipeline::report_rich_progress(const RichProgressEvent& ev) {
    // Fill in the module-level fields (total/done function counts,
    // elapsed time, time budget) so the TUI can render an overall
    // progress bar. We make a mutable copy because the event is passed
    // by const ref.
    RichProgressEvent filled = ev;
    fill_module_progress(filled);

    if (rich_progress_cb_) {
        std::lock_guard<std::mutex> lock(progress_mutex_);
        rich_progress_cb_(filled);
    }
    if (progress_cb_) {
        std::lock_guard<std::mutex> lock(progress_mutex_);
        progress_cb_(filled.stage, filled.function_name, filled.progress);
    }
}

// ── fill_module_progress ───────────────────────────────────────────────────
//
// Populates the module_* fields on a RichProgressEvent. Called on every
// rich event so the TUI always has up-to-date module-level stats for its
// overall progress bar.

void Pipeline::fill_module_progress(RichProgressEvent& ev) {
    ev.module_total_functions = module_total_functions_;
    ev.module_done_functions = module_done_functions_.load(std::memory_order_relaxed);
    if (module_start_ != std::chrono::steady_clock::time_point{}) {
        ev.module_elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - module_start_).count();
    }
    if (config_.time_budget > 0.0) {
        ev.module_time_budget_ms = config_.time_budget * 1000.0;
    }
}

// ── should_log_verbose ───────────────────────────────────────────────────
// See the declaration in Pipeline.h and PipelineConfig::verbose_interval_
// seconds for the rationale (long runs otherwise flood the console with
// one near-duplicate line per phase per round).

bool Pipeline::should_log_verbose(const std::string& stage, const std::string& fn_name) {
    if (config_.verbose_interval_seconds <= 0.0) return true;  // throttling off
    const std::string key = stage + "|" + fn_name;
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(verbose_mutex_);
    auto it = last_verbose_emit_.find(key);
    if (it == last_verbose_emit_.end() ||
        std::chrono::duration<double>(now - it->second).count() >=
            config_.verbose_interval_seconds) {
        last_verbose_emit_[key] = now;
        return true;
    }
    return false;
}

} // namespace clunk