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
 * Clunk Pipeline — orchestrates the full optimisation pipeline:
 * Pattern Library → Stochastic Search → Evolutionary Search → SMT Verify
 */
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include "clunk/IR/Module.h"
#include "clunk/IR/Function.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Evaluator/MCACostModel.h"
#include "clunk/Search/StochasticSearch.h"
#include "clunk/Search/EvolutionarySearch.h"
#include "clunk/Search/SMTVerifier.h"
#include "clunk/Search/AliveVerifier.h"
#include "clunk/Search/PeepholeMiner.h"
#include "clunk/Search/EgraphRewriter.h"
#include "clunk/Search/RewriteCache.h"
#include "clunk/Search/VectorSynth.h"
#include "clunk/Search/HoleSynth.h"
#include "clunk/Search/AlgoPreprocessor.h"
#include "clunk/Search/BlockOptimiser.h"
#include "clunk/Search/SeriesExpander.h"
#include "clunk/Search/ThreadPool.h"
#include "clunk/Pattern/PatternLibrary.h"
#include "clunk/IR/StrengthReduce.h"

namespace clunk {

// ── Pipeline configuration ──────────────────────────────────────────────
struct PipelineConfig {
    // Optimisation level:
    //   0 = none (pass-through)
    //   1 = pattern library only (cheap, no search)
    //   2 = continuous search (patterns + stochastic + evolutionary + SMT)
    //   3 = continuous search with larger per-round budgets
    // At level >= 2 the pipeline runs souper/minotaur-style refinement
    // ROUNDS: each round searches from the current best, verifies the top
    // candidates against the ORIGINAL function with SMT, and adopts the
    // best verified improvement as the new baseline. Rounds continue until
    // `convergence_rounds` consecutive rounds find no improvement, until
    // `max_rounds` is reached, or until the time budget is exhausted —
    // there is no fixed cap on how far a function can be optimised.
    unsigned opt_level = 2;

    // Time budget in seconds (0 = no limit)
    double time_budget = 0.0;

    // Worker threads for MODULE-level parallelism: functions are independent,
    // so they are optimised concurrently. 0 = auto (hardware_concurrency),
    // 1 = sequential (deterministic — search RNG isn't shared across threads,
    // so results can differ by thread count). The shared evaluator (thread-safe
    // cache) and read-only pattern library are shared; each worker gets its own
    // search RNG / stats / Z3 context.
    size_t num_threads = 0;

    // ── Per-function multithreading (work-stealing) ──────────────────────
    // When true, the pipeline's module-level worker pool is ALSO shared
    // with the hole-synth per-depth search: idle threads (from finished
    // functions, or a small program with fewer functions than threads)
    // help search the in-progress function's codespace via the
    // work-stealing HoleSynth path. Each depth's search space is split
    // into "first-step" work items dispatched to the pool; workers
    // collect ALL verified candidates (NOT short-circuiting on first
    // hit) and the synthesiser picks the lowest-score one at the end
    // of each depth. This implements the user's "per-function
    // multithreading" requirement — when multiple threads find
    // different optimal candidates at the same depth, the score
    // arbitrates.
    //
    // Only takes effect when enable_hole_synth is true AND num_threads
    // > 1 (the pool needs at least 2 workers for parallel dispatch).
    // The pool's run_until() helper lets a worker that submitted
    // subtasks participate in draining the queue, avoiding the
    // nested-parallelism deadlock. Default ON.
    bool enable_per_function_threads = true;

    // ── Continuous-refinement controls ─────────────────────────────────
    // Maximum refinement rounds per function. 0 = unlimited (bounded only
    // by convergence detection and the time budget).
    size_t max_rounds = 0;

    // Stop refining a function after this many consecutive rounds without
    // an adopted improvement. Minimum 1.
    size_t convergence_rounds = 3;

    // ── Per-function time cap ───────────────────────────────────
    // Maximum wall-clock seconds to spend on ANY ONE function. 0 = no cap
    // (bounded only by the module-level time_budget). When set, forces the
    // search to move on from a stuck function so other functions get their
    // fair share of the budget. Default 0 (off) for backward compat; the
    // CLI sets a sensible default.
    double max_time_per_function = 0.0;

    // Souper-style soundness (default): when a working SMT prover is
    // available, only adopt candidates it PROVES equivalent to the
    // original. Set true to also adopt the best-scoring candidate the
    // prover returned Unknown for (functions with memory ops, floats, or
    // loops cannot currently be proved) — best-effort, like the legacy
    // behaviour, except candidates proved NotEquivalent are still always
    // rejected. Ignored when no prover is available (the evaluator is
    // trusted in that case).
    bool trust_unverified = false;

    // Target architecture
    pattern::ArchDescriptor target_arch;

    // Search configurations
    search::StochasticConfig stochastic_config;
    search::EvolutionaryConfig evolutionary_config;
    search::SMTConfig smt_config;

    // ── Alive2 (alive-tv) second-opinion verification ───────────────────
    // OFF by default: alive-tv is an external binary (see
    // AliveVerifier.h), not something clunk ships or dlopens, so this is
    // opt-in. When enabled, every candidate that SMTVerifier (or a
    // sound-by-construction rewrite) already blessed is additionally
    // checked with alive-tv before being adopted. alive-tv covers more of
    // LLVM's real semantics than clunk's own encoder (memory, calls,
    // more of the poison/UB model) and — critically — runs the candidate
    // through LLVM's actual .ll parser, so it also catches cases where
    // clunk's own printer emitted invalid IR. A candidate alive-tv
    // REFUTES is never adopted, even if SMT said Equivalent (that
    // combination means clunk's own SMT encoding has a soundness bug).
    // If no Alive2 backend is found (libAlive2.so or alive-tv binary),
    // this silently degrades to a no-op — see AliveVerifier::is_available().
    bool enable_alive2 = false;
    search::AliveConfig alive_config;

    // SMT-verified peephole miner (Souper-style enumerative superoptimisation).
    // At opt_level >= 2 the miner runs as one more candidate source per round:
    // it extracts pure-integer straight-line slices from the current baseline,
    // proves cheaper equivalents with SMT, and splices them back — the lever
    // that finds wins the mutation search cannot. Disabled
    // automatically when SMT is unavailable/skipped (it is unsound without a
    // real prover).
    bool enable_peephole_miner = true;
    search::MinerConfig peephole_config;

    // ── Souper/Minotaur-style harvesting ──────────────────────────────
    // When true, the mining_phase additionally calls
    // PeepholeMiner::harvest_and_rewrite() on the current baseline, which
    // harvests every integer instruction (replacing unsupported operands
    // with opaque vars) and SMT-proves cheaper rewrites. This is the
    // generalisation of mine_and_rewrite that can find wins on functions
    // with memory ops, calls, and (eventually) loops. Default ON at
    // opt_level >= 2.
    bool enable_harvest_miner = true;

    // ── llvm-mca final ranking (Minotaur-style) ──────────────────────────
    // When true and llc + llvm-mca are on PATH, verify_and_select re-ranks
    // improving candidates by MEASURED cycle count (llc → llvm-mca) before
    // adoption, and drops "improvements" the machine model says are not
    // faster. Refinement only — soundness never depends on it, and any
    // candidate the toolchain cannot lower falls back to the built-in
    // cost-model ranking. Off by default (each measurement costs two
    // subprocess launches). Enable with --mca.
    bool use_mca_ranker = false;

    // ── Loop optimisation (LICM + constant-trip full unrolling) ─────────
    // Sound-by-construction loop rewrites (the SMT verifier refuses
    // back-edges, so these carry no proof but are exact by construction):
    // loop-invariant code motion into the preheader, and full unrolling of
    // single-block loops whose trip count folds to a constant — which
    // converts an unverifiable loop function into straight-line code the
    // SMT-verified phases can then attack. The cost model arbitrates the
    // unrolling tradeoff like any other candidate. Default ON.
    bool enable_loop_opt = true;

    // ── Alias-aware memory optimisation ──────────────────────────────────
    // Block-local store-to-load forwarding, redundant-load elimination and
    // dead-store elimination over a conservative alias oracle (distinct
    // allocas, noalias arguments, disjoint constant GEPs). Sound by
    // construction. Default ON.
    bool enable_mem_opt = true;

    // ── Alignment finalisation (vmovupd -> vmovapd, no stack realignment) ─
    // Module-level finishing pass, run once after every function is
    // optimised (see clunk/Search/AlignOpt.h): raises vector load/store
    // `align` to what is PROVABLE (so the backend can emit aligned moves)
    // and relaxes over-aligned, non-escaping allocas that would otherwise
    // force a dynamic `and rsp, -N` prologue. Sound by construction.
    // Default ON at opt_level >= 1.
    bool enable_align_opt = true;

    // ── Dataflow pruning (dead-code + same-block CSE + unreachable-block
    //     elimination + known-bits-driven constant/branch folding) ───────
    // Runs ir::prune_dataflow() on the current baseline every round,
    // before the mutation searches see it: dead-code elimination (def-use
    // closure — no CFG traversal needed, correct across loops), same-block
    // common-subexpression elimination (every duplicate pure computation
    // in a block is deduplicated, exhaustively — see clunk/IR/
    // DataflowPrune.h's simplify_cse), removal of blocks unreachable from
    // entry (with Phi fixup), and a lightweight Souper-style "infer known
    // bits" pass (see clunk/Analysis/KnownBits.h) that folds
    // provably-constant values, drops no-op AND masks, resolves
    // provably-true/false comparisons, and turns a conditional Br with a
    // now-constant condition into an unconditional one (which is what
    // feeds the unreachable-block pass its work). Sound by construction —
    // same trust tier as LoopOpt/MemOpt. This doesn't just shrink the
    // FINAL output: a smaller, cleaner baseline is cheaper for every later
    // phase this round and every round after (fewer instructions to
    // mutate, lower to the e-graph, or measure). Default ON.
    bool enable_dataflow_prune = true;

    // ── Powers-of-two strength reduction ─────────────────────────────────
    // Runs ir::simplify_pow2_strength_reduce() on the current baseline
    // every round: rewrites `x urem y` to `x and (y - 1)` whenever the
    // PowersOfTwo analysis (clunk/Analysis/PowersOfTwo.h, itself layered
    // on KnownBits) proves y is a nonzero power of two — including when y
    // is a RUNTIME value, not just a compile-time constant (constant
    // divisors are already folded elsewhere). Sound by construction —
    // same trust tier as LoopOpt/MemOpt/dataflow pruning. Default ON.
    bool enable_pow2_strength_reduce = true;

    // ── Interprocedural inlining ─────────────────────────────────────────
    // Inline small single-block module-internal callees before optimising
    // each function. Removes opaque calls — which also makes the caller
    // SMT-verifiable — and is only kept when the cost model scores the
    // inlined body cheaper. Default ON.
    bool enable_inliner = true;

    // ── Cross-function optimisation (module-level passes) ───────────────
    // Run before per-function superoptimisation. Two passes:
    //  - Dead Function Elimination (DFE): remove module-internal
    //    functions that no caller can reach. Frees the per-function
    //    pipeline from wasting rounds on dead code.
    //  - Interprocedural Constant Propagation (IPCP): for every function
    //    argument where ALL direct callers pass the SAME constant,
    //    clone the function with the constant substituted and rewrite
    //    callers to invoke the clone. The clone is then a target for
    //    constant folding by the per-function pipeline.
    // Both passes are exact by construction (no SMT needed). Default ON
    // at opt_level >= 2.
    bool enable_cross_function = true;
    // ── Multi-block inliner ─────────────────────────────────────────────
    // Extends the inliner beyond single-block callees: handles multi-block
    // callees (CFG cloned verbatim), allocas (hoisted to caller entry),
    // and recursive call graphs (refuses to inline a callee into itself
    // via SCC check; transitive depth cap). Default ON at opt_level >= 2.
    bool enable_multiblock_inliner = true;

    // ── Vector-intrinsic synthesis (Minotaur-style) ──────────────────────
    // When true, run_on_function runs a vector_phase() each round on
    // functions containing vector operations: it recognises scalar idioms
    // over vector values (lane-wise op groups, horizontal-reduction trees,
    // shuffle chains) and re-expresses them as vector instructions and
    // clunk.vector.reduce.* intrinsics. Every rewrite is gated by the cost
    // model AND an SMT equivalence proof (the verifier lane-blasts vector
    // functions via ir::Scalarizer). Default ON at opt_level >= 2; inert
    // on vector-free functions.
    //
    // The pass now ALSO runs a width-cascade pre-pass that tries
    // AVX-512 → AVX2 → AVX → scalar in order, picking the widest tier
    // that yields a verified cheaper rewrite. Lane decomposition and
    // surrounding-code rewriting (scalar load → vector load) are applied
    // at each tier. See VectorSynth.h for the full design.
    bool enable_vector_synth = true;
    search::VectorSynthConfig vector_synth_config;

    // ── Equality-saturation candidate generation ─────────────────────
    // When true, the pipeline runs a new egraph_phase() before the mining
    // phase each round: it lowers the current baseline to an e-graph,
    // applies pattern-library rewrites as e-graph rules, saturates, and
    // extracts the cheapest representative as a Candidate (sound=false:
    // pattern-library rewrites need SMT verification). Solves the
    // pattern-ordering problem for free. Default OFF (the pattern library
    // must be non-empty for this to be useful).
    bool enable_egraph_phase = false;

    // ── Hole-based progressive-deepening synthesis (Massalin-style) ───────
    // When true, run_on_function runs a hole_synth_phase() each round:
    // it replaces the function body with a "hole" and enumerates
    // candidate fillings in order of increasing size (1 instruction,
    // then 2, then 3, ...), SMT-verifying each. The first verified
    // candidate that scores strictly cheaper than the original is
    // returned. Complements the stochastic / evolutionary phases — a
    // deterministic, completeness-bounded search that finds the
    // SHORTEST equivalent form. Default ON at opt_level >= 2; inert
    // on functions outside its scope (multi-block, FP, >4 args, etc.).
    bool enable_hole_synth = true;
    search::HoleSynthConfig hole_synth_config;

    // ── Algorithmic preprocessor (module-level pre-pass) ─────────────────
    // When true, Pipeline::run() invokes AlgoPreprocessor on the module
    // BEFORE per-function superoptimisation: it walks the call graph and
    // detects functions (or compositions of functions) whose output is a
    // predictable closed form (constant, c*x, c*x+b). When a pattern is
    // detected and SMT-proven, the function's body is rewritten to the
    // minimal closed form, shrinking the work the per-function pipeline
    // has to do. Default ON at opt_level >= 2.
    bool enable_algo_preprocessor = true;
    search::AlgoPreConfig algo_pre_config;

    // ── Block-level divide-and-conquer optimisation ─────────────────────
    // When true, the refinement loop runs a block_phase() each round
    // AFTER the whole-function phases (stochastic, evolutionary, miner,
    // hole-synth, etc.) — but ONLY if none of those phases produced an
    // improving candidate this round (i.e. the round was otherwise
    // stagnant). The block phase splits the function's single basic
    // block into a binary tree of contiguous instruction ranges
    // (halving at each level, minimum size 2 instructions) and tries to
    // SMT-prove a cheaper equivalent for each range, preserving data
    // flow between ranges. This finds wins the whole-function search
    // misses (where the whole function has no short equivalent, but a
    // sub-range does). Sound: every adopted rewrite is SMT-proven.
    // Default ON at opt_level >= 2; inert on functions outside its
    // scope (multi-block, FP, memory, etc.).
    bool enable_block_opt = true;
    search::BlockOptimiserConfig block_opt_config;

    // ── Series-expansion loop optimisation ───────────────────────────
    // When true, the refinement loop runs a series_phase() each round
    // on functions containing loops: it detects single-block loops
    // that compute arithmetic series (sum of i, sum of a+i*d, constant
    // accumulation, scaled arithmetic) and replaces the entire loop
    // with its closed-form expression. This is the classic
    // superoptimiser "series expansion" win — a loop that sums i for
    // i=0..n becomes n*(n-1)/2, turning O(n) iterations into O(1)
    // arithmetic. Sound by construction (the algebra is exact, same
    // trust tier as LICM/LoopOpt). Default ON at opt_level >= 2;
    // inert on functions without matching loops.
    bool enable_series_expand = true;
    search::SeriesExpanderConfig series_expand_config;

    // ── STOKE-style stochastic search moves ────────────────────────────
    // When true, the stochastic search may use the unsound STOKE-style
    // mutation kinds (OpcodeReplace, OperandReplace, OperandSwap,
    // InstructionInsert, InstructionReplace). Candidates produced via
    // these moves are NOT sound-by-construction and MUST be SMT-verified
    // before adoption. Default OFF (preserves today's sound-by-construction
    // search behaviour).
    bool allow_unsound_mutations = false;

    // Number of random test vectors the stochastic search uses as a
    // pre-filter before sending candidates to SMT. 0 = disabled (today's
    // behaviour). 32 = Massalin's recommended value. Default 0 — enable
    // explicitly with --test-vectors N when the search is producing many
    // non-equivalent candidates that SMT is rejecting.
    size_t test_vector_count = 0;

    // ── Persistent SMT rewrite cache ──────────────────────────────────
    // When non-empty, the SMT verifier consults this file-backed cache
    // before calling Z3 and writes the result back after. The cache is
    // human-readable (one entry per line, tab-separated). Default empty
    // (no caching).
    std::string smt_cache_path;

    // Pattern library path (empty = in-memory only)
    std::string pattern_library_path;

    // GPU optimisation
    bool enable_gpu_opt = true;
    bool enable_launch_opt = true;

    // Reporting
    bool verbose = false;
    bool dump_candidates = false;
    std::string dump_dir;

    // ── Verbose output throttling ────────────────────────────────────────
    // The per-round diagnostic lines emitted when verbose is on (one per
    // phase that fired: "[mem] ...", "[prune] ...", "[round N] adopted",
    // etc.) are genuinely useful for a short debugging run, but for
    // long-running superoptimisation (thousands of rounds) printing every
    // single one floods the console with near-duplicate lines and the
    // signal that matters — is this STILL making progress? — gets buried.
    // Each diagnostic site is independently rate-limited (per stage, per
    // function) to at most one line every verbose_interval_seconds; the
    // very first line for a given (stage, function) always prints
    // immediately, and the aggregate counts are always available at the
    // end from the per-function summary regardless of how much was
    // throttled along the way. 0 disables throttling entirely (every line
    // prints, i.e. the old unthrottled firehose behaviour) — useful when
    // debugging a specific short-lived issue where every line matters.
    // Default: 1 line/second/stage, which is plenty to see live progress
    // without scrolling thousands of lines on a long run.
    double verbose_interval_seconds = 1.0;

    // ── Scale-control fields ──────────────────────────────────────────
    // Maximum function size (instruction count) to attempt superoptimisation.
    // Functions larger than this are passed through unchanged. Default is
    // 512 — site-driven mutation keeps the per-iteration cost low on big
    // functions (it was 200 back when every iteration deep-copied the
    // whole function on a blind mutation guess). Tunable via
    // --max-function-size.
    size_t max_function_size = 512;

    // ── Oversized-function mining ───────────────────────────────────────
    // Functions above max_function_size are passed through for the full\n    // search stack but still get a miner-only refinement path instead of\n    // a skip, up to this second cap (0 = disabled). Because the
    // whole-function SMT re-verification cannot model functions this large,
    // miner wins on them are only adoptable in trust_unverified mode (each
    // slice still carries its own SMT proof).
    size_t max_mining_function_size = 8192;

    // If true, skip SMT verification entirely (always return Unknown).
    // Useful for very large inputs where SMT would never terminate.
    // Tunable via --skip-smt.
    bool skip_smt = false;

    // ── SMT-attempt cap ─────────────────────────────────────────────────
    // Per-round cap on the number of SMT verify() calls verify_and_select
    // makes before giving up on the candidate batch. The historical hard-
    // coded default was 5; raising it lets more candidates get proven at
    // the cost of more wall-clock time per round. Tunable via
    // --max-smt-attempts. 0 = use the built-in default (5).
    size_t max_smt_attempts = 5;
};

// ── Pipeline result ─────────────────────────────────────────────────────
struct PipelineResult {
    std::shared_ptr<ir::Module> optimised_module;

    // Per-function results
    struct FunctionResult {
        std::string function_name;
        std::shared_ptr<ir::Function> original;
        std::shared_ptr<ir::Function> optimised;
        double score_original = 0.0;
        double score_optimised = 0.0;
        double improvement_ratio = 1.0;
        bool verified = false;        // SMT verified
        size_t patterns_applied = 0;  // Patterns from library
        size_t rounds_run = 0;        // Refinement rounds executed
        size_t improvements_adopted = 0;  // Rounds that improved the baseline
        double time_spent_ms = 0.0;
    };

    std::map<std::string, FunctionResult> function_results;

    // Aggregate statistics
    double total_time_ms = 0.0;
    size_t total_functions_processed = 0;
    size_t total_optimised = 0;
    double avg_improvement = 1.0;
};

// ── Progress callback ───────────────────────────────────────────────────
using ProgressCallback = std::function<void(const std::string& stage,
                                             const std::string& function_name,
                                             double progress)>;

// ── Rich progress event (for the TUI / machine consumers) ────────────────
//
// A richer progress event that includes the current best IR snapshot,
// the round number, and per-stage stats. The pipeline emits these
// alongside the simple ProgressCallback above (the legacy callback
// stays for backward compat). Consumers that need the IR — like the
// ncurses TUI — subscribe via set_rich_progress_callback().
struct RichProgressEvent {
    std::string stage;            // "patterns", "inline", "round", "vector",
                                  // "hole", "algo-pre", "mem", "prune",
                                  // "pow2", "mining", "egraph", "verify",
                                  // "done", "pipeline", "skip", "cap"
    std::string function_name;    // function being processed
    double progress = 0.0;        // 0..1 (module-level fraction)

    // ── Current best snapshot ───────────────────────────────────────────
    // The IR text of the current best function (i.e. the baseline the
    // search is mutating from in this round). Empty when the stage
    // doesn't update the baseline (e.g. "pipeline" / "skip").
    std::string current_best_ir;

    // ── Round / improvement counters ────────────────────────────────────
    size_t round = 0;             // refinement round (0-indexed)
    size_t improvements_adopted = 0;
    double score_original = 0.0;
    double score_current = 0.0;
    double improvement_ratio = 1.0;
    bool verified = false;        // was the current best SMT-verified?

    // ── Stage-specific message (human-readable) ────────────────────────
    // e.g. "vector: synthesised 4 vector op(s)" or "mining: 3 slice(s) proved".
    std::string message;

    // ── Elapsed time so far on this function (ms) ──────────────────────
    double elapsed_ms = 0.0;

    // ── Module-level progress (for the overall progress bar) ───────────
    size_t module_total_functions = 0;   // total functions in the module
    size_t module_done_functions = 0;    // functions that reached done/skip
    double module_elapsed_ms = 0.0;      // total elapsed wall-clock
    double module_time_budget_ms = 0.0;  // 0 = no budget
};

using RichProgressCallback = std::function<void(const RichProgressEvent&)>;

// ── The Pipeline ────────────────────────────────────────────────────────
class Pipeline final {
public:
    explicit Pipeline(const PipelineConfig& config = {});
    ~Pipeline();

    // Run the full optimisation pipeline on a module
    PipelineResult run(const ir::Module& module);

    // Run on a single function
    PipelineResult::FunctionResult run_on_function(const ir::Function& fn);

    // Set progress callback
    void set_progress_callback(ProgressCallback cb) { progress_cb_ = std::move(cb); }

    // Set rich progress callback (for TUI / machine consumers).
    // The rich callback fires on the same stages as the simple
    // callback, but includes the current best IR snapshot and per-stage
    // stats. Both callbacks can be set simultaneously.
    void set_rich_progress_callback(RichProgressCallback cb) {
        rich_progress_cb_ = std::move(cb);
    }

    // Access sub-components
    evaluator::EvaluationEngine& evaluation_engine() { return eval_engine_; }
    search::StochasticSearch& stochastic_search() { return searchers_.stochastic; }
    search::EvolutionarySearch& evolutionary_search() { return searchers_.evolutionary; }
    search::SMTVerifier& smt_verifier() { return searchers_.smt; }
    pattern::PatternLibrary& pattern_library() { return pattern_lib_; }

    // Configuration
    PipelineConfig& config() { return config_; }
    const PipelineConfig& config() const { return config_; }

private:
    // Per-thread search state. The evaluator (thread-safe cache) and pattern
    // library (read-only match/apply) are shared across workers, but each
    // worker needs its own RNG, statistics, and Z3 context — those live here.
    // One Searchers is used for sequential runs; the parallel path gives each
    // worker thread its own.
    struct Searchers {
        search::StochasticSearch stochastic;
        search::EvolutionarySearch evolutionary;
        search::SMTVerifier smt;
        search::AliveVerifier alive;    // second-opinion gate; see enable_alive2
        uint64_t last_mined_hash = 0;   // peephole-mining guard, reset per fn
        uint64_t last_egraph_hash = 0;  // e-graph guard, reset per fn
        uint64_t last_vector_hash = 0;  // vector-synthesis guard, reset per fn
        uint64_t last_loop_hash = 0;    // loop-opt guard, reset per fn
        uint64_t last_mem_hash = 0;     // mem-opt guard, reset per fn
        uint64_t last_prune_hash = 0;   // dataflow-prune guard, reset per fn
        uint64_t last_pow2_hash = 0;    // pow2 strength-reduce guard, reset per fn
        uint64_t last_hole_hash = 0;    // hole-synth guard, reset per fn
        uint64_t last_block_hash = 0;   // block-opt guard, reset per fn
        uint64_t last_series_hash = 0;  // series-expand guard, reset per fn
        // Rejected-candidate cache: structural hashes of candidates
        // that were SMT-rejected this round-chain. Prevents re-proposing
        // and re-rejecting the same candidate on unchanged baselines.
        std::unordered_set<uint64_t> rejected_hashes;
        Searchers(const PipelineConfig& cfg, evaluator::EvaluationEngine* eng)
            : stochastic(cfg.stochastic_config, eng),
              evolutionary(cfg.evolutionary_config, eng),
              smt(cfg.smt_config),
              alive(cfg.alive_config) {}
    };

    // Optimise one function using the given per-thread search state. The public
    // run_on_function() delegates to this with the shared sequential Searchers.
    PipelineResult::FunctionResult run_on_function(const ir::Function& fn,
                                                   Searchers& s);

    // Pipeline stages
    std::shared_ptr<ir::Function> apply_patterns(
        const ir::Function& fn,
        PipelineResult::FunctionResult& result);

    std::vector<search::Candidate> stochastic_phase(
        const ir::Function& fn, double remaining_seconds, Searchers& s);

    std::vector<search::Candidate> evolutionary_phase(
        const ir::Function& fn,
        const std::vector<search::Candidate>& stochastic_candidates,
        double remaining_seconds, Searchers& s);

    // SMT-verified peephole mining of `fn` (the current baseline). Returns the
    // miner's whole-function rewrite as a single sound candidate, or empty if
    // nothing was proven. Each distinct baseline is mined at most once
    // (structural-hash guard in `s`) since the miner is deterministic.
    std::vector<search::Candidate> mining_phase(const ir::Function& fn,
                                                Searchers& s);

    // ── Equality-saturation candidate generation ────────────────────
    // Lower `fn` (the current baseline) to an e-graph, apply pattern-library
    // rewrites as e-graph rules, saturate, and extract the cheapest
    // representative as a Candidate (sound=false: pattern-library rewrites
    // need SMT verification). Returns empty if disabled or no improvement.
    // Guarded by last_egraph_hash — skips re-running on an unchanged
    // baseline (eliminates the repeated-identical-candidate pathology).
    std::vector<search::Candidate> egraph_phase(const ir::Function& fn,
                                                 Searchers& s);

    // ── Vector-intrinsic synthesis phase ─────────────────────────────────
    // Runs VectorSynthesizer on the current baseline when it contains
    // vector operations. The synthesiser SMT-proves its rewrite internally
    // (Candidate::sound reflects the proof), so verify_and_select can adopt
    // it without re-verification. Guarded by last_vector_hash — a given
    // baseline is synthesised at most once (the pass is deterministic).
    std::vector<search::Candidate> vector_phase(const ir::Function& fn,
                                                 Searchers& s);

    // ── Hole-based progressive-deepening synthesis phase ───────────────
    // Runs HoleSynthesizer on the current baseline: replaces the
    // function body with a "hole" and enumerates 1-instruction, then
    // 2-instruction, then 3-instruction equivalents, SMT-verifying
    // each. Returns the first verified cheaper candidate. The
    // candidate carries `sound = proven` (verified rewrites are
    // sound-by-construction once SMT says Equivalent). Guarded by
    // last_hole_hash — a given baseline is synthesised at most once.
    std::vector<search::Candidate> hole_synth_phase(const ir::Function& fn,
                                                     Searchers& s);

    // ── Block-level divide-and-conquer optimisation phase ─────────────
    // Runs BlockOptimiser on the current baseline: splits the single
    // basic block into a binary tree of contiguous instruction ranges
    // (halving at each level, minimum size 2) and SMT-proves cheaper
    // equivalents for each range, preserving data flow between ranges.
    // The candidate carries `sound = proven` (every adopted range
    // rewrite is SMT-proven equivalent to the input function). Guarded
    // by last_block_hash — a given baseline is block-optimised at most
    // once (the optimiser is deterministic for a given baseline).
    //
    // This phase is only called by the refinement loop when no whole-
    // function phase produced an improving candidate this round (i.e.
    // the round was otherwise stagnant) — it is the "fallback" that
    // finds wins the whole-function search misses.
    std::vector<search::Candidate> block_phase(const ir::Function& fn,
                                                Searchers& s);

    // ── Series-expansion loop optimisation phase ─────────────────────
    // Runs SeriesExpander on the current baseline: detects single-block
    // loops computing arithmetic series and replaces them with closed-
    // form expressions (e.g. `for(i=0;i<n;i++) sum+=i` → `n*(n-1)/2`).
    // The candidate carries `sound = true` (the algebra is exact by
    // construction, same trust tier as loop_phase). Guarded by
    // last_series_hash — a given baseline is expanded at most once.
    std::vector<search::Candidate> series_phase(const ir::Function& fn,
                                                  Searchers& s);

    // ── Loop-optimisation phase ──────────────────────────────────────────
    // LICM + constant-trip full unrolling of the current baseline. Both
    // rewrites are exact by construction (Candidate::sound = true); the
    // cost model decides adoption. Guarded by last_loop_hash.
    std::vector<search::Candidate> loop_phase(const ir::Function& fn,
                                               Searchers& s);

    // ── Memory-optimisation phase ────────────────────────────────────────
    // Alias-aware store-to-load forwarding / RLE / DSE on the current
    // baseline. Exact by construction. Guarded by last_mem_hash.
    std::vector<search::Candidate> mem_phase(const ir::Function& fn,
                                              Searchers& s);

    // ── Dataflow-pruning phase ───────────────────────────────────────────
    // ir::prune_dataflow(): DCE + unreachable-block elimination + known-
    // bits-driven constant/branch folding on the current baseline. Exact
    // by construction. Guarded by last_prune_hash.
    std::vector<search::Candidate> dataflow_prune_phase(const ir::Function& fn,
                                                         Searchers& s);

    // ── Powers-of-two strength-reduction phase ───────────────────────────
    // ir::simplify_pow2_strength_reduce(): `x urem y` -> `x and (y - 1)`
    // wherever the PowersOfTwo analysis proves y is a nonzero power of
    // two, including at runtime. Exact by construction. Guarded by
    // last_pow2_hash.
    std::vector<search::Candidate> pow2_phase(const ir::Function& fn,
                                               Searchers& s);

    // Select the best candidate that beats `current`, verifying against
    // `original` (the soundness anchor for the whole refinement chain).
    // Sets result.verified when the selection was SMT-proved.
    std::shared_ptr<ir::Function> verify_and_select(
        const ir::Function& original,
        const ir::Function& current,
        const std::vector<search::Candidate>& candidates,
        PipelineResult::FunctionResult& result,
        Searchers& s);

    // Seconds until the active deadline (+inf when no deadline is set).
    double remaining_seconds() const;

#ifdef CLUNK_HAS_GPU
    std::shared_ptr<ir::Function> gpu_optimise(
        const ir::Function& fn);
#endif

    void report_progress(const std::string& stage,
                          const std::string& fn_name,
                          double progress);

    // Fire a rich progress event (for TUI / machine consumers).
    // The simple progress callback (if set) is also fired with the
    // event's stage/name/progress fields, so existing stderr prints
    // keep working.
    void report_rich_progress(const RichProgressEvent& ev);

    // Fill in the module-level fields (total/done function counts,
    // elapsed time, time budget) on a RichProgressEvent before
    // dispatching it. Called by report_rich_progress and by the
    // inline sites that build events.
    void fill_module_progress(RichProgressEvent& ev);

    // ── Verbose-output throttling ─────────────────────────────────────
    // Gate for every per-round `if (config_.verbose) { std::cerr << ... }`
    // diagnostic site: returns true at most once every
    // config_.verbose_interval_seconds for a given (stage, fn_name) pair
    // (always true the first time a pair is seen, and always true when
    // verbose_interval_seconds <= 0, i.e. throttling disabled). See
    // PipelineConfig::verbose_interval_seconds for the rationale. Callers
    // still gate on config_.verbose themselves first — this only decides
    // WHETHER a verbose-eligible line prints this time, not whether
    // verbose mode is on at all.
    bool should_log_verbose(const std::string& stage, const std::string& fn_name);

    PipelineConfig config_;
    evaluator::EvaluationEngine eval_engine_;
    // llvm-mca final ranker (opt-in, see PipelineConfig::use_mca_ranker).
    // One shared instance so the measurement cache spans rounds and
    // worker threads (its methods are thread-safe).
    evaluator::MCACostModel mca_;
    Searchers searchers_;              // shared sequential search state
    pattern::PatternLibrary pattern_lib_;
    ProgressCallback progress_cb_;
    RichProgressCallback rich_progress_cb_;
    std::mutex progress_mutex_;        // guards progress_cb_ across workers

    // Last-emitted timestamp per "stage|fn_name" key, for
    // should_log_verbose() above. Guarded by verbose_mutex_ since worker
    // threads superoptimise different functions concurrently and must
    // not throttle each other's independent (stage, fn_name) lines.
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_verbose_emit_;
    std::mutex verbose_mutex_;

    // ── Persistent SMT rewrite cache ──────────────────────────────────
    // Shared across all worker threads. Created lazily in the constructor
    // when config_.smt_cache_path is non-empty. Plumbed into each worker's
    // SMTVerifier via SMTConfig::cache.
    std::shared_ptr<search::RewriteCache> rewrite_cache_;

    // Wall-clock deadline shared by run() and run_on_function() so each
    // refinement round sees the true remaining budget.
    std::chrono::steady_clock::time_point deadline_{};
    bool has_deadline_ = false;

    // ── Module-level progress tracking (for the TUI progress bar) ────
    // Set at the start of run(), read by fill_module_progress() on
    // every rich event. module_total_functions_ is the count of
    // functions to process; module_done_functions_ is incremented
    // whenever a function reaches "done" or "skip".
    std::chrono::steady_clock::time_point module_start_{};
    size_t module_total_functions_ = 0;
    std::atomic<size_t> module_done_functions_{0};

    // ── Shared thread pool (module-level + per-function) ──────────────
    // Lazily created in run() when num_threads > 1. The pool is shared
    // between module-level function dispatch AND the hole-synth per-
    // depth search — so idle module-level workers (from a small program
    // or finished functions) help search the in-progress function's
    // codespace via the work-stealing HoleSynth path. The pool's
    // run_until() helper lets a worker that submitted subtasks
    // participate in draining the queue, avoiding the nested-
    // parallelism deadlock (all N workers blocked on subtasks, no one
    // left to run them). Null when num_threads <= 1.
    std::unique_ptr<search::ThreadPool> shared_pool_;
};

} // namespace clunk