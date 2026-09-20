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
 * Clunk CLI — the `clunk` command-line superoptimiser.
 *
 * Usage: clunk [options] <input.ll>
 */
#include "clunk/Pipeline.h"
#include "clunk/Parser/IRParser.h"
#include "clunk/Search/PeepholeMiner.h"

#ifdef CLUNK_HAS_TUI
#include "clunk/cli/TUI.h"
#endif

#include <atomic>
#include <cstdlib>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

static const char* CLUNK_VERSION = "0.1.0";

// ── Command-line options ────────────────────────────────────────────────────

struct CliOptions {
    std::string input_file;
    std::string output_file;       // empty = stdout
    unsigned opt_level = 3;        // numeric level (0-3); 3 = highest tier
    // `max` is the new DEFAULT: opt_level=3 AND every optional stage
    // enabled AND raised SMT/search limits. Users who want a faster,
    // lighter run can pass --opt-level 0|1|2|3 explicitly.
    bool opt_level_max = true;     // true iff --opt-level max (or default)
    bool opt_level_explicitly_set = false;  // user passed --opt-level
    std::string target_triple;     // empty = host
    double time_budget = 30.0;     // default 30s wall-clock budget
    bool time_budget_explicitly_set = false;  // user passed --time-budget
    bool no_z3 = false;
    bool no_gpu = false;
    bool no_miner = false;         // --no-miner: disable the in-loop peephole miner
    bool no_vector_synth = false;  // --no-vector-synth: disable vector-intrinsic synthesis
    bool no_hole_synth = false;    // --no-hole-synth: disable hole-based progressive-deepening synthesis
    bool no_block_opt = false;     // --no-block-opt: disable block-level divide-and-conquer optimisation
    bool no_series_expand = false; // --no-series-expand: disable series-expansion loop optimisation
    bool no_align_opt = false;     // --no-align-opt: disable the alignment finalisation pass
    bool no_algo_preprocessor = false;  // --no-algo-preprocessor: disable module-level algo pre-pass
    std::string vector_width = "auto";  // --vector-width <avx512|avx2|avx|auto>
    bool tui = false;             // --tui: launch the ncurses TUI
    bool use_mca = false;          // --mca: rank final candidates with llvm-mca
    bool verbose = false;
    double verbose_interval = 1.0;  // --verbose-interval: throttle seconds (0 = unthrottled)
    bool mine = false;             // --mine: SMT-verified peephole mining -> pattern library
    bool report_json = false;      // --report-json: emit per-function stats as JSON to stdout
    std::string dump_candidates_dir;
    std::string pattern_library_path;
    bool show_help = false;
    bool show_version = false;

    // ── Search-engine flags ─────────────────────────────────────────────
    // All default to "use the engine default" so existing behaviour is
    // preserved when the user does not pass them.
    bool no_parallel_eval = false;          // --no-parallel-eval
    bool no_pattern_library = false;        // --no-pattern-library
    double diversity_radius = -1.0;         // --diversity-radius <r>; <0 = default
    size_t max_results = 0;                 // --max-results <n>; 0 = unlimited
    size_t eval_threads = 0;                // --eval-threads <n>; 0 = auto
    size_t jobs = 0;                        // --jobs/-j <n>; module-level worker threads (0 = auto)
    size_t stagnation_limit = 0;            // --stagnation-limit <n>; 0 = default
    bool no_per_function_threads = false;   // --no-per-function-threads: disable work-stealing HoleSynth

    // ── Scale-control flags ───────────────────────────────────────────────
    size_t max_function_size = 512;         // --max-function-size <n>
    bool max_function_size_explicitly_set = false;
    bool skip_smt = false;                  // --skip-smt (disable SMT entirely)

    // ── Continuous-refinement flags ───────────────────────────────────
    size_t max_rounds = 0;                  // --max-rounds <n>; 0 = unlimited
    size_t convergence_rounds = 0;          // --convergence-rounds <n>; 0 = default
    bool convergence_rounds_explicitly_set = false;
    bool trust_unverified = false;          // --trust-unverified

    // ── Per-function time cap ─────────────────────────────────────────────
    double max_time_per_function = 0.0;     // --max-time-per-fn <seconds>; 0 = no cap

    // ── Superoptimiser-strategy flags ──────────────────────────────────
    bool allow_unsound_mutations = false;   // --stoke-moves
    bool allow_unsound_mutations_explicitly_set = false;
    size_t test_vector_count = 0;           // --test-vectors <n>
    bool test_vector_count_explicitly_set = false;
    bool enable_egraph_phase = false;       // --egraph
    bool enable_egraph_phase_explicitly_set = false;
    bool no_harvest_miner = false;          // --no-harvest-miner
    std::string smt_cache_path;             // --cache-path <path>
    bool no_honor_binop_flags = false;      // --no-honor-binop-flags
    bool no_path_conditions = false;        // --no-path-conditions
    size_t max_mining_function_size = 8192; // --max-mining-function-size <n>
    bool max_mining_function_size_explicitly_set = false;

    // ── SMT tuning flags ───────────────────────────────────────────────
    unsigned smt_timeout_ms = 0;            // --smt-timeout <ms>; 0 = default (30000)
    size_t smt_max_blocks = 0;              // --smt-max-blocks <n>; 0 = default (20)
    size_t smt_max_instructions = 0;        // --smt-max-instructions <n>; 0 = default (100)
    size_t max_smt_attempts = 0;            // --max-smt-attempts <n>; 0 = default (5)
    bool smt_bounded_unrolling = false;     // --smt-bounded-unrolling
    // Track which SMT flags were explicitly set so max-mode can override
    // only the ones the user did NOT pass.
    bool smt_timeout_explicitly_set = false;
    bool smt_max_blocks_explicitly_set = false;
    bool smt_max_instructions_explicitly_set = false;
    bool max_smt_attempts_explicitly_set = false;
    bool smt_bounded_unrolling_explicitly_set = false;
    bool use_mca_explicitly_set = false;

    // ── Cross-function flags (new) ─────────────────────────────────────
    bool no_cross_function = false;         // --no-cross-function
    bool no_multiblock_inliner = false;     // --no-multiblock-inliner

    // ── Alive2 second-opinion verification ──────────────────────────────
    bool alive2 = true;                     // alive2
    std::string alive_tv_path = "alive-tv"; // --alive-tv-path <path>
    unsigned alive_timeout_ms = 30000;      // --alive-timeout <ms>
};

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <input.ll>\n"
              << "\n"
              << "Clunk — LLVM superoptimiser\n"
              << "\n"
              << "Options:\n"
              << "  --opt-level <0-3|max>     0 = off, 1 = patterns only, 2+ = continuous\n"
              << "                            search (level sizes each round, not the total;\n"
              << "                            refinement continues until convergence or the\n"
              << "                            time budget runs out). `max` (= level 3) also\n"
              << "                            enables every optional stage (mca, stoke-moves,\n"
              << "                            bounded unrolling, cross-function, multi-block\n"
              << "                            inliner) and raises SMT/search limits to\n"
              << "                            maximise optimisation opportunities. (e-graph is\n"
              << "                            NOT auto-enabled — pass --egraph explicitly.)\n"
              << "                            DEFAULT: max (just run `clunk input.ll` and get\n"
              << "                            a faster program out).\n"
              << "  --max-rounds <n>          Cap refinement rounds per function (0 = unlimited)\n"
              << "  --convergence-rounds <n>  Stop after n rounds without improvement (default 3)\n"
              << "  --trust-unverified        Also adopt candidates the prover returned Unknown\n"
              << "                            for (needed to optimise functions with memory ops,\n"
              << "                            floats, or loops; proven-wrong candidates are\n"
              << "                            still rejected)\n"
              << "  --max-time-per-fn <secs>  Cap wall-clock time spent on any one function (0 = no cap)\n"
              << "                            — prevents one stuck function from monopolizing the budget\n"
              << "  --output <file>           Output file (default: stdout)\n"
              << "  --target <triple>         Target triple (default: host)\n"
              << "  --time-budget <seconds>   Time budget in seconds (default: 30; 0 = no limit)\n"
              << "  --no-z3                   Disable Z3 verification\n"
              << "  --no-alive2               Disable Alive2 verification\n"
              << "  --alive-tv-path <path>    Path to the alive-tv binary (default: alive-tv on PATH)\n"
              << "  --alive-timeout <ms>      Per-candidate alive-tv timeout in ms (default 30000)\n"
              << "  --no-gpu                  Disable GPU optimisation\n"
              << "  --no-miner                Disable the in-loop SMT-verified peephole miner\n"
              << "  --no-vector-synth         Disable SMT-verified vector-intrinsic synthesis\n"
              << "  --no-hole-synth           Disable hole-based progressive-deepening synthesis\n"
              << "                            (replaces fn body with a hole, enumerates 1-inst, then 2,\n"
              << "                            then 3 ... equivalents, SMT-verifies each)\n"
              << "  --no-block-opt            Disable block-level divide-and-conquer optimisation\n"
              << "                            (splits fn into halving ranges, SMT-proves cheaper\n"
              << "                            equivalents for each; fallback when whole-fn search\n"
              << "                            finds nothing)\n"
              << "  --no-align-opt            Disable alignment finalisation (aligned vector\n"
              << "                            moves where provable; no needless stack realignment)\n"
              << "  --no-series-expand        Disable series-expansion loop optimisation\n"
              << "                            (detects arithmetic-series loops, replaces with\n"
              << "                            closed forms: sum-of-i → n*(n-1)/2, etc.)\n"
              << "  --no-algo-preprocessor    Disable module-level algorithmic preprocessor\n"
              << "                            (detects f(x)=C, f(x)=c*x, f(x)=c*x+b, g(f(x)) collapses)\n"
              << "  --vector-width <tier>     Widest vector tier to attempt (avx512|avx2|avx|auto)\n"
              << "                            Default: auto (= avx512 with cascade fallback)\n"
              << "  --tui                     Launch an ncurses TUI showing live superoptimiser\n"
              << "                            progress (function list + current-best IR preview).\n"
              << "                            Requires a tty; ignored (with a warning) otherwise.\n"
              << "                            Keys: ↑/↓ nav, Tab/p pin, r toggle raw IR, q quit.\n"
              << "  --mca                     Rank final candidates by measured cycles\n"
              << "                            (llc + llvm-mca; ignored when not installed)\n"
              << "  --verbose                 Verbose output\n"
              << "  --verbose-interval <s>    Throttle verbose progress lines to at most one\n"
              << "                            per stage every <s> seconds (default 1; 0 = print\n"
              << "                            every line, unthrottled — useful for short runs)\n"
              << "  --report-json             Emit per-function stats as a JSON array to stdout\n"
              << "                            (score_original/optimised, improvement, verified);\n"
              << "                            optimised IR then goes only to --output, not stdout\n"
              << "  --dump-candidates <dir>   Dump all candidates to directory\n"
              << "  --pattern-library <path>  Path to pattern library file\n"
              << "  --mine                    Mine SMT-verified peephole rewrites from the\n"
              << "                            input's integer functions and add them to the\n"
              << "                            --pattern-library file (Souper-style; does not\n"
              << "                            emit optimised IR). Pairs with --report-json.\n"
              << "  --no-parallel-eval        Disable parallel population evaluation\n"
              << "  --no-pattern-library      Disable pattern-guided mutation\n"
              << "  --diversity-radius <r>    Fitness-sharing radius (0 = disabled)\n"
              << "  --max-results <n>         Limit final candidate count\n"
              << "  --jobs, -j <n>            Worker threads to optimise functions in parallel (0 = auto)\n"
              << "  --eval-threads <n>        Worker threads for parallel eval (0 = auto)\n"
              << "  --no-per-function-threads Disable work-stealing hole-synth (idle workers don't help\n"
              << "                            search in-progress functions' codespace)\n"
              << "  --stagnation-limit <n>    Generations before restart (0 = default)\n"
              << "  --max-function-size <n>   Skip superoptimisation for functions larger than this (default 512)\n"
              << "  --skip-smt                Disable SMT verification entirely (sound: returns Unknown)\n"
              << "  --stoke-moves             Enable STOKE-style unsound search moves (opcode/operand/swap/insert/replace)\n"
              << "                            — candidates are SMT-verified before adoption.\n"
              << "  --test-vectors <n>        Pre-filter candidates with N random test vectors before SMT\n"
              << "  --egraph                  Run equality-saturation candidate generation each round\n"
              << "  --no-harvest-miner        Disable Souper-style harvest_and_rewrite fallback miner\n"
              << "  --no-path-conditions      Mine slices without dominating-branch assumptions\n"
              << "  --max-mining-function-size <n>  Upper size cap for the miner-only path on functions\n"
              << "                            above --max-function-size (default 8192; 0 = disable)\n"
              << "  --cache-path <path>       Persistent SMT rewrite cache (file-backed LRU)\n"
              << "  --no-honor-binop-flags    Ignore nsw/nuw/exact flags in SMT encoding (sound fallback)\n"
              << "  --smt-timeout <ms>        Per-call Z3 timeout in ms (default 30000)\n"
              << "  --smt-max-blocks <n>      Refuse SMT on functions with more than n blocks (default 20)\n"
              << "  --smt-max-instructions <n>  Refuse SMT on functions with more than n instructions (default 100)\n"
              << "  --max-smt-attempts <n>    Max SMT verify calls per verify_and_select (default 5)\n"
              << "  --smt-bounded-unrolling   Pre-unroll constant-trip single-block loops before SMT (sound)\n"
              << "  --no-cross-function       Disable module-level DFE + IPCP pre-pass\n"
              << "  --no-multiblock-inliner   Disable multi-block (CFG-aware) inliner\n"
              << "\n"
              << "  --version                 Print version\n"
              << "  --help                    Print this help\n"
              << "\n";
}

static CliOptions parse_args(int argc, char* argv[]) {
    CliOptions opts;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            opts.show_help = true;
        } else if (arg == "--version") {
            opts.show_version = true;
        } else if (arg == "--opt-level") {
            if (i + 1 < argc) {
                std::string val = argv[++i];
                opts.opt_level_explicitly_set = true;
                // Case-insensitive comparison for "max".
                std::string lower_val;
                lower_val.reserve(val.size());
                for (char c : val) lower_val.push_back(
                    static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                if (lower_val == "max") {
                    opts.opt_level_max = true;
                    opts.opt_level = 3;
                } else {
                    opts.opt_level_max = false;
                    try {
                        opts.opt_level = static_cast<unsigned>(std::stoul(val));
                    } catch (...) {
                        std::cerr << "Error: --opt-level requires 0, 1, 2, 3, or max\n";
                        opts.show_help = true;
                        continue;
                    }
                    if (opts.opt_level > 3) opts.opt_level = 3;
                }
            } else {
                std::cerr << "Error: --opt-level requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                opts.output_file.assign(argv[++i]);
            } else {
                std::cerr << "Error: --output requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--target") {
            if (i + 1 < argc) {
                opts.target_triple.assign(argv[++i]);
            } else {
                std::cerr << "Error: --target requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--time-budget") {
            if (i + 1 < argc) {
                opts.time_budget = std::stod(argv[++i]);
                opts.time_budget_explicitly_set = true;
            } else {
                std::cerr << "Error: --time-budget requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--no-z3") {
            opts.no_z3 = true;
        } else if (arg == "--no-alive2") {
            opts.alive2 = false;
        } else if (arg == "--alive-tv-path") {
            if (i + 1 < argc) {
                opts.alive_tv_path = argv[++i];
            } else {
                std::cerr << "Error: --alive-tv-path requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--alive-timeout") {
            if (i + 1 < argc) {
                opts.alive_timeout_ms = static_cast<unsigned>(std::stoul(argv[++i]));
            } else {
                std::cerr << "Error: --alive-timeout requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--no-gpu") {
            opts.no_gpu = true;
        } else if (arg == "--no-miner") {
            opts.no_miner = true;
        } else if (arg == "--no-vector-synth") {
            opts.no_vector_synth = true;
        } else if (arg == "--no-hole-synth") {
            opts.no_hole_synth = true;
        } else if (arg == "--no-block-opt") {
            opts.no_block_opt = true;
        } else if (arg == "--no-align-opt") {
            opts.no_align_opt = true;
        } else if (arg == "--no-series-expand") {
            opts.no_series_expand = true;
        } else if (arg == "--no-algo-preprocessor") {
            opts.no_algo_preprocessor = true;
        } else if (arg == "--vector-width") {
            if (i + 1 < argc) {
                opts.vector_width = argv[++i];
            } else {
                std::cerr << "Error: --vector-width requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--tui") {
            opts.tui = true;
        } else if (arg == "--mca") {
            opts.use_mca = true;
            opts.use_mca_explicitly_set = true;
        } else if (arg == "--verbose" || arg == "-v") {
            opts.verbose = true;
        } else if (arg == "--verbose-interval") {
            if (i + 1 < argc) {
                opts.verbose_interval = std::stod(argv[++i]);
            } else {
                std::cerr << "Error: --verbose-interval requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--mine") {
            opts.mine = true;
        } else if (arg == "--report-json") {
            opts.report_json = true;
        } else if (arg == "--dump-candidates") {
            if (i + 1 < argc) {
                opts.dump_candidates_dir.assign(argv[++i]);
            } else {
                std::cerr << "Error: --dump-candidates requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--pattern-library") {
            if (i + 1 < argc) {
                opts.pattern_library_path.assign(argv[++i]);
            } else {
                std::cerr << "Error: --pattern-library requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--no-parallel-eval") {
            opts.no_parallel_eval = true;
        } else if (arg == "--no-per-function-threads") {
            opts.no_per_function_threads = true;
        } else if (arg == "--no-pattern-library") {
            opts.no_pattern_library = true;
        } else if (arg == "--diversity-radius") {
            if (i + 1 < argc) {
                opts.diversity_radius = std::stod(argv[++i]);
            } else {
                std::cerr << "Error: --diversity-radius requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--max-results") {
            if (i + 1 < argc) {
                opts.max_results = static_cast<size_t>(std::stoull(argv[++i]));
            } else {
                std::cerr << "Error: --max-results requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--eval-threads") {
            if (i + 1 < argc) {
                opts.eval_threads = static_cast<size_t>(std::stoull(argv[++i]));
            } else {
                std::cerr << "Error: --eval-threads requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--jobs" || arg == "-j") {
            if (i + 1 < argc) {
                opts.jobs = static_cast<size_t>(std::stoull(argv[++i]));
            } else {
                std::cerr << "Error: --jobs requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--stagnation-limit") {
            if (i + 1 < argc) {
                opts.stagnation_limit = static_cast<size_t>(std::stoull(argv[++i]));
            } else {
                std::cerr << "Error: --stagnation-limit requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--max-function-size") {
            if (i + 1 < argc) {
                opts.max_function_size = static_cast<size_t>(std::stoull(argv[++i]));
                opts.max_function_size_explicitly_set = true;
            } else {
                std::cerr << "Error: --max-function-size requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--skip-smt") {
            opts.skip_smt = true;
        } else if (arg == "--max-rounds") {
            if (i + 1 < argc) {
                opts.max_rounds = static_cast<size_t>(std::stoull(argv[++i]));
            } else {
                std::cerr << "Error: --max-rounds requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--convergence-rounds") {
            if (i + 1 < argc) {
                opts.convergence_rounds = static_cast<size_t>(std::stoull(argv[++i]));
                opts.convergence_rounds_explicitly_set = true;
            } else {
                std::cerr << "Error: --convergence-rounds requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--trust-unverified") {
            opts.trust_unverified = true;
        } else if (arg == "--max-time-per-fn") {
            if (i + 1 < argc) {
                opts.max_time_per_function = std::stod(argv[++i]);
            } else {
                std::cerr << "Error: --max-time-per-fn requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--stoke-moves") {
            opts.allow_unsound_mutations = true;
            opts.allow_unsound_mutations_explicitly_set = true;
        } else if (arg == "--test-vectors") {
            if (i + 1 < argc) {
                opts.test_vector_count = static_cast<size_t>(std::stoull(argv[++i]));
                opts.test_vector_count_explicitly_set = true;
            } else {
                std::cerr << "Error: --test-vectors requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--egraph") {
            opts.enable_egraph_phase = true;
            opts.enable_egraph_phase_explicitly_set = true;
        } else if (arg == "--no-harvest-miner") {
            opts.no_harvest_miner = true;
        } else if (arg == "--no-path-conditions") {
            opts.no_path_conditions = true;
        } else if (arg == "--max-mining-function-size") {
            if (i + 1 < argc) {
                opts.max_mining_function_size =
                    static_cast<size_t>(std::stoull(argv[++i]));
                opts.max_mining_function_size_explicitly_set = true;
            } else {
                std::cerr << "Error: --max-mining-function-size requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--cache-path") {
            if (i + 1 < argc) {
                opts.smt_cache_path.assign(argv[++i]);
            } else {
                std::cerr << "Error: --cache-path requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--no-honor-binop-flags") {
            opts.no_honor_binop_flags = true;
        } else if (arg == "--smt-timeout") {
            if (i + 1 < argc) {
                opts.smt_timeout_ms = static_cast<unsigned>(std::stoul(argv[++i]));
                opts.smt_timeout_explicitly_set = true;
            } else {
                std::cerr << "Error: --smt-timeout requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--smt-max-blocks") {
            if (i + 1 < argc) {
                opts.smt_max_blocks = static_cast<size_t>(std::stoull(argv[++i]));
                opts.smt_max_blocks_explicitly_set = true;
            } else {
                std::cerr << "Error: --smt-max-blocks requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--smt-max-instructions") {
            if (i + 1 < argc) {
                opts.smt_max_instructions = static_cast<size_t>(std::stoull(argv[++i]));
                opts.smt_max_instructions_explicitly_set = true;
            } else {
                std::cerr << "Error: --smt-max-instructions requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--max-smt-attempts") {
            if (i + 1 < argc) {
                opts.max_smt_attempts = static_cast<size_t>(std::stoull(argv[++i]));
                opts.max_smt_attempts_explicitly_set = true;
            } else {
                std::cerr << "Error: --max-smt-attempts requires an argument\n";
                opts.show_help = true;
            }
        } else if (arg == "--smt-bounded-unrolling") {
            opts.smt_bounded_unrolling = true;
            opts.smt_bounded_unrolling_explicitly_set = true;
        } else if (arg == "--no-cross-function") {
            opts.no_cross_function = true;
        } else if (arg == "--no-multiblock-inliner") {
            opts.no_multiblock_inliner = true;
        } else if (arg[0] == '-') {
            std::cerr << "Error: unknown option: " << arg << "\n";
            opts.show_help = true;
        } else {
            // Positional argument — input file
            if (opts.input_file.empty()) {
                opts.input_file.assign(arg);
            } else {
                std::cerr << "Error: multiple input files specified\n";
                opts.show_help = true;
            }
        }
    }

    return opts;
}

// ── Derive ArchDescriptor from target triple ────────────────────────────────

static clunk::pattern::ArchDescriptor arch_from_triple(const std::string& triple) {
    clunk::pattern::ArchDescriptor arch;

    if (triple.empty()) {
        // Default to x86_64 host
        arch.name = "x86_64";
        arch.vendor = "intel";
        arch.is_gpu = false;
        arch.vector_width = 256;
        arch.has_avx2 = true;
        arch.l1_cache_kb = 32;
        arch.l2_cache_kb = 256;
        arch.l3_cache_kb = 8192;
        return arch;
    }

    // Simple heuristic parsing
    if (triple.find("nvptx") != std::string::npos ||
        triple.find("amdgcn") != std::string::npos) {
        arch.is_gpu = true;
        arch.warp_size = 32;  // Default for NVIDIA; AMD is wavefront=64
        if (triple.find("amdgcn") != std::string::npos) {
            arch.warp_size = 64;
            arch.vendor = "amd";
            arch.name = "amdgcn";
        } else {
            arch.vendor = "nvidia";
            arch.name = "nvptx64";
        }
        arch.shared_mem_kb = 48;
        arch.compute_capability = 80;  // Default assumption
    } else if (triple.find("aarch64") != std::string::npos ||
               triple.find("arm64") != std::string::npos) {
        arch.name = "aarch64";
        arch.vendor = "arm";
        arch.vector_width = 128;
        arch.has_sve = true;
        arch.l1_cache_kb = 64;
        arch.l2_cache_kb = 256;
        arch.l3_cache_kb = 4096;
    } else if (triple.find("x86_64") != std::string::npos ||
               triple.find("amd64") != std::string::npos) {
        arch.name = "x86_64";
        arch.vendor = "intel";
        arch.vector_width = 256;
        arch.has_avx2 = true;
        arch.l1_cache_kb = 32;
        arch.l2_cache_kb = 256;
        arch.l3_cache_kb = 8192;
    } else if (triple.find("i386") != std::string::npos ||
               triple.find("i686") != std::string::npos) {
        arch.name = "x86";
        arch.vendor = "intel";
        arch.vector_width = 128;
        arch.l1_cache_kb = 32;
        arch.l2_cache_kb = 256;
        arch.l3_cache_kb = 4096;
    } else if (triple.find("riscv") != std::string::npos) {
        arch.name = "riscv64";
        arch.vendor = "riscv";
        arch.vector_width = 128;
        arch.l1_cache_kb = 32;
        arch.l2_cache_kb = 128;
    } else {
        // Unknown — use generic defaults
        arch.name = "generic";
        arch.vendor = "unknown";
    }

    return arch;
}

// ── JSON helpers (for --report-json) ────────────────────────────────────────

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// Emit the per-function pipeline results as a JSON array to `os`. Fields come
// straight from PipelineResult::FunctionResult so the harness can decide
// "stronger after -O3" = (improvement_ratio > 1 && verified).
static void emit_report_json(std::ostream& os, const clunk::PipelineResult& result) {
    os << std::setprecision(10);
    os << "[";
    bool first = true;
    for (auto& [name, fr] : result.function_results) {
        os << (first ? "\n" : ",\n");
        first = false;
        os << "  {"
           << "\"function\":\"" << json_escape(name) << "\","
           << "\"score_original\":" << fr.score_original << ","
           << "\"score_optimised\":" << fr.score_optimised << ","
           << "\"improvement_ratio\":" << fr.improvement_ratio << ","
           << "\"verified\":" << (fr.verified ? "true" : "false") << ","
           << "\"patterns_applied\":" << fr.patterns_applied << ","
           << "\"rounds_run\":" << fr.rounds_run << ","
           << "\"improvements_adopted\":" << fr.improvements_adopted
           << "}";
    }
    os << (first ? "]\n" : "\n]\n");
}

// ── main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    auto opts = parse_args(argc, argv);

    if (opts.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    if (opts.show_version) {
        std::cout << "clunk version " << CLUNK_VERSION << "\n";
        return 0;
    }

    if (opts.input_file.empty()) {
        std::cerr << "Error: no input file specified\n";
        print_usage(argv[0]);
        return 1;
    }

    // ── Parse the input .ll file ────────────────────────────────────────
    if (opts.verbose) {
        std::cerr << "Parsing " << opts.input_file << " ...\n";
    }

    clunk::parser::IRParser parser;
    auto module = parser.parse_file(opts.input_file);

    if (!module) {
        std::cerr << "Error: failed to parse " << opts.input_file << "\n";
        return 1;
    }

    // Report any parser warnings
    for (auto& warning : parser.warnings()) {
        std::cerr << "Warning: " << warning << "\n";
    }

    if (opts.verbose) {
        std::cerr << "Parsed module: " << module->name()
                  << " (" << module->function_count() << " functions)\n";
    }

    // ── Large-input warning for --opt-level max ────────────────────────
    // Max mode runs every stage with raised limits — on a large input
    // that can take a very long time (or exhaust memory). Warn the user
    // before the pipeline starts so they can Ctrl-C and re-run with a
    // lower opt-level or a tighter time budget. The thresholds are
    // heuristic: ~5000 instructions or ~500KB of source IR is roughly
    // where max mode starts to feel slow on a typical workstation.
    if (opts.opt_level_max) {
        size_t total_insts = module->instruction_count();
        size_t fn_count = module->function_count();
        // Also check the file size on disk — a module with few functions
        // but huge per-function bodies still warrants a warning.
        std::ifstream size_check(opts.input_file, std::ios::binary | std::ios::ate);
        size_t file_bytes = 0;
        if (size_check.is_open()) {
            file_bytes = static_cast<size_t>(size_check.tellg());
            size_check.close();
        }
        const size_t INST_THRESHOLD = 5000;
        const size_t FILE_BYTES_THRESHOLD = 500 * 1024;  // 500 KB
        const size_t FN_THRESHOLD = 200;
        if (total_insts > INST_THRESHOLD ||
            file_bytes > FILE_BYTES_THRESHOLD ||
            fn_count > FN_THRESHOLD) {
            std::cerr << "Warning: --opt-level max on a large input ("
                      << total_insts << " instructions, "
                      << fn_count << " functions, "
                      << (file_bytes / 1024) << " KB).\n"
                      << "         This may take a long time.\n"
                      << "         Continuing with max mode...\n";
        }
    }

    // ── Mining mode (--mine): SMT-verified peephole discovery ───────────
    // Enumerative superoptimisation over the input's integer functions;
    // proven-cheaper rewrites are appended to the --pattern-library file for
    // the normal pipeline to reuse via PatternGuided. This is the lever that
    // can beat -O3 on integer code (see scripts/beat_o3.sh for measurement).
    if (opts.mine) {
        clunk::evaluator::EvaluationEngine engine;
        clunk::search::PeepholeMiner miner(&engine);
        auto arch = arch_from_triple(opts.target_triple);
        auto wins = miner.mine_module(*module);

        clunk::pattern::PatternLibrary lib;
        if (!opts.pattern_library_path.empty()) {
            lib.load(opts.pattern_library_path);  // extend an existing library
        }
        for (auto& w : wins) {
            lib.add_pattern(clunk::search::PeepholeMiner::to_pattern(w, arch));
        }
        if (!opts.pattern_library_path.empty()) {
            lib.save(opts.pattern_library_path);
        }

        const auto& st = miner.stats();
        std::cerr << "\n=== Clunk Mine Summary ===\n"
                  << "Functions seen:      " << st.functions_seen << "\n"
                  << "Eligible (integer):  " << st.eligible << "\n"
                  << "Proven rewrites:     " << st.mined << "\n"
                  << "Candidates enumerated: " << st.candidates_tried << "\n"
                  << "SMT checks:          " << st.smt_checks << "\n";
        if (!opts.pattern_library_path.empty()) {
            std::cerr << "Pattern library:     " << opts.pattern_library_path
                      << " (" << lib.size() << " patterns)\n";
        }

        if (opts.report_json) {
            std::cout << std::setprecision(10) << "[";
            for (size_t i = 0; i < wins.size(); ++i) {
                auto& w = wins[i];
                std::cout << (i ? ",\n" : "\n") << "  {"
                          << "\"function\":\"" << json_escape(w.source->name()) << "\","
                          << "\"source_score\":" << w.source_score << ","
                          << "\"replacement_score\":" << w.replacement_score << ","
                          << "\"smt_checks\":" << w.smt_checks << "}";
            }
            std::cout << (wins.empty() ? "]\n" : "\n]\n");
        }
        return 0;
    }

    // ── Create PipelineConfig from CLI options ──────────────────────────
    clunk::PipelineConfig config;
    config.opt_level = opts.opt_level;
    config.time_budget = opts.time_budget;
    config.target_arch = arch_from_triple(opts.target_triple);
    config.enable_gpu_opt = !opts.no_gpu;
    config.enable_launch_opt = !opts.no_gpu;
    config.enable_peephole_miner = !opts.no_miner;
    config.enable_vector_synth = !opts.no_vector_synth;
    config.enable_hole_synth = !opts.no_hole_synth;
    config.enable_block_opt = !opts.no_block_opt;
    config.enable_series_expand = !opts.no_series_expand;
    config.enable_align_opt = !opts.no_align_opt;
    config.enable_algo_preprocessor = !opts.no_algo_preprocessor;

    // ── Vector width tier ─────────────────────────────────────────────
    // Parse --vector-width into VectorSynthConfig.widest_tier. "auto"
    // (default) means try every tier starting from AVX-512 and fall back
    // to narrower tiers when wider ones don't yield a win.
    {
        std::string vw = opts.vector_width;
        // Case-insensitive comparison.
        std::string lower_vw;
        lower_vw.reserve(vw.size());
        for (char c : vw) lower_vw.push_back(
            static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        if (lower_vw == "auto" || lower_vw.empty()) {
            config.vector_synth_config.widest_tier = clunk::search::VectorWidth::AVX512;
        } else if (lower_vw == "avx512" || lower_vw == "avx-512") {
            config.vector_synth_config.widest_tier = clunk::search::VectorWidth::AVX512;
        } else if (lower_vw == "avx2" || lower_vw == "avx-2") {
            config.vector_synth_config.widest_tier = clunk::search::VectorWidth::AVX2;
        } else if (lower_vw == "avx" || lower_vw == "avx-1" || lower_vw == "avx1") {
            config.vector_synth_config.widest_tier = clunk::search::VectorWidth::AVX;
        } else {
            std::cerr << "Error: --vector-width must be one of: "
                         "avx512, avx2, avx, auto\n";
            opts.show_help = true;
        }
    }
    config.use_mca_ranker = opts.use_mca;
    config.verbose = opts.verbose;
    config.verbose_interval_seconds = opts.verbose_interval;
    config.dump_candidates = !opts.dump_candidates_dir.empty();
    config.dump_dir = opts.dump_candidates_dir;
    config.pattern_library_path = opts.pattern_library_path;

    // ── Wire search-engine flags ────────────────────────────────────────
    // Each flag overrides the engine default only when the user passed
    // it; otherwise the default from StochasticConfig / EvolutionaryConfig
    // is preserved.
    if (opts.no_parallel_eval) {
        config.evolutionary_config.parallel_evaluation = false;
    }
    if (opts.no_pattern_library) {
        config.stochastic_config.use_pattern_library = false;
        config.evolutionary_config.use_pattern_library = false;
    }
    if (opts.diversity_radius >= 0.0) {
        config.evolutionary_config.diversity_radius = opts.diversity_radius;
    }
    if (opts.max_results > 0) {
        config.evolutionary_config.max_results = opts.max_results;
        config.stochastic_config.max_candidates =
            std::min(config.stochastic_config.max_candidates, opts.max_results);
    }
    config.num_threads = opts.jobs;
    config.enable_per_function_threads = !opts.no_per_function_threads;
    if (opts.eval_threads > 0) {
        config.evolutionary_config.num_eval_threads = opts.eval_threads;
    }
    if (opts.stagnation_limit > 0) {
        config.evolutionary_config.stagnation_limit = opts.stagnation_limit;
        config.stochastic_config.stagnation_limit = opts.stagnation_limit;
    }
    // If the user set a time budget on the CLI, propagate it to both
    // search engines so they honour it.
    if (opts.time_budget > 0.0) {
        config.stochastic_config.time_budget_seconds = opts.time_budget;
        config.evolutionary_config.time_budget_seconds = opts.time_budget;
    }
    // Propagate the target architecture to the stochastic search so
    // PatternLibrary::match filters patterns by compatibility.
    config.stochastic_config.target_arch = config.target_arch;

    // --no-z3 disables SMT verification but keeps the search running.
    if (opts.no_z3) {
        config.skip_smt = true;
    }

    // ── Alive2 second-opinion verification ──────────────────────────────
    config.enable_alive2 = opts.alive2;
    config.alive_config.alive_tv_path = opts.alive_tv_path;
    config.alive_config.timeout_ms = opts.alive_timeout_ms;
    // Reuse the module's own target info (if the input .ll specified one)
    // so alive-tv reasons about the same triple/datalayout clang used.
    config.alive_config.target_triple =
        module->has_target() ? module->target().triple : "";
    config.alive_config.target_datalayout =
        module->has_target() ? module->target().datalayout : "";

    // ── Scale-control flags ───────────────────────────────────────────────
    config.max_function_size = opts.max_function_size;
    if (opts.skip_smt) config.skip_smt = true;

    // ── Continuous-refinement flags ────────────────────────────────────
    config.max_rounds = opts.max_rounds;
    if (opts.convergence_rounds > 0) {
        config.convergence_rounds = opts.convergence_rounds;
    }
    config.trust_unverified = opts.trust_unverified;

    // ── Superoptimiser-strategy flags ──────────────────────────────────
    config.allow_unsound_mutations = opts.allow_unsound_mutations;
    config.test_vector_count = opts.test_vector_count;
    config.enable_egraph_phase = opts.enable_egraph_phase;
    config.enable_harvest_miner = !opts.no_harvest_miner;
    config.peephole_config.use_path_conditions = !opts.no_path_conditions;
    config.max_mining_function_size = opts.max_mining_function_size;
    config.smt_cache_path = opts.smt_cache_path;
    if (opts.no_honor_binop_flags) {
        config.smt_config.honor_binop_flags = false;
    }

    // ── SMT tuning flags ───────────────────────────────────────────────
    if (opts.smt_timeout_ms > 0) {
        config.smt_config.timeout_ms = opts.smt_timeout_ms;
    }
    if (opts.smt_max_blocks > 0) {
        config.smt_config.max_blocks_for_smt = opts.smt_max_blocks;
    }
    if (opts.smt_max_instructions > 0) {
        config.smt_config.max_instructions_for_smt = opts.smt_max_instructions;
    }
    config.smt_config.sound_bounded_unrolling = opts.smt_bounded_unrolling;
    if (opts.max_smt_attempts > 0) {
        config.max_smt_attempts = opts.max_smt_attempts;
    }

    // ── Cross-function flags ───────────────────────────────────────────
    config.enable_cross_function = !opts.no_cross_function;
    config.enable_multiblock_inliner = !opts.no_multiblock_inliner;

    // ── Per-function time cap ─────────────────────────────────────────────
    config.max_time_per_function = opts.max_time_per_function;

    // ── `--opt-level max` overrides ────────────────────────────────────
    // When max mode is active (the default, or via --opt-level max), turn
    // on every optional optimisation stage and raise every limit so the
    // pipeline finds the most provable optimisations it can within the
    // time budget. Each override respects an explicit user flag — e.g.
    // `--opt-level max --no-egraph` keeps e-graph OFF even in max mode.
    if (opts.opt_level_max && !opts.no_z3 && !opts.skip_smt) {
        config.opt_level = 3;

        // ── Enable every optional stage (each gated by its explicit-set
        // flag so the user can still turn any of them OFF).
        // NOTE: e-graph is NOT enabled by default in max mode — it has a
        // known issue where call instructions can be extracted as
        // `call @<unknown>()` (the e-graph doesn't model callees). Users
        // who want e-graph can still pass --egraph explicitly.
        if (!opts.enable_egraph_phase_explicitly_set) {
            config.enable_egraph_phase = false;
        }
        if (!opts.use_mca_explicitly_set) {
            // MCA ranker requires llc + llvm-mca on PATH; the pipeline
            // checks is_available() before using it, so turning the flag
            // on is safe even when the tools aren't installed.
            config.use_mca_ranker = true;
        }
        if (!opts.allow_unsound_mutations_explicitly_set) {
            // STOKE moves are unsound-by-construction but every candidate
            // is SMT-verified before adoption. Safe in max mode (which
            // keeps SMT on); we already bailed above if --no-z3.
            config.allow_unsound_mutations = true;
        }
        if (!opts.test_vector_count_explicitly_set) {
            // Massalin's recommended 32 test vectors as a pre-filter
            // before SMT — catches most non-equivalent candidates cheaply.
            config.test_vector_count = 64;
        }
        if (!opts.smt_bounded_unrolling_explicitly_set) {
            config.smt_config.sound_bounded_unrolling = true;
        }
        // Cross-function passes are ON by default already; ensure max
        // mode keeps them on unless the user explicitly disabled them.
        // (No action needed — handled by the no_cross_function /
        // no_multiblock_inliner flags above.)

        // ── Raise SMT limits so more functions are verifiable ────────
        if (!opts.smt_timeout_explicitly_set) {
            config.smt_config.timeout_ms = 60000;  // 1 min per Z3 call
        }
        if (!opts.smt_max_blocks_explicitly_set) {
            config.smt_config.max_blocks_for_smt = 100;  // 5x default
        }
        if (!opts.smt_max_instructions_explicitly_set) {
            config.smt_config.max_instructions_for_smt = 500;  // 5x default
        }
        if (!opts.max_smt_attempts_explicitly_set) {
            config.max_smt_attempts = 80;  // prove more candidates per round
        }

        // ── Raise scale-control so bigger functions get the full stack ─
        if (!opts.max_function_size_explicitly_set) {
            config.max_function_size = 4096;  // 8x default
        }
        if (!opts.max_mining_function_size_explicitly_set) {
            config.max_mining_function_size = 65536;  // 8x default
        }

        // ── More patience: convergence_rounds 1024 (default 3) so the
        // pipeline keeps searching longer before giving up on a function.
        if (!opts.convergence_rounds_explicitly_set) {
            config.convergence_rounds = 1024;
        }
        // max_rounds stays 0 (unlimited) — the time budget bounds wall-clock.
        // max_time_per_function stays 0 (no per-fn cap) — the time budget
        // bounds wall-clock and we want every function to get its fair share.

        // ── Time budget: if the user did NOT explicitly set one, set
        // to an unlimited time budget.
        if (!opts.time_budget_explicitly_set) {
            config.time_budget = 0.0;
            // Propagate to the search engines.
            config.stochastic_config.time_budget_seconds = 0.0;
            config.evolutionary_config.time_budget_seconds = 0.0;
        }
    }


    // ── Create Pipeline and run ─────────────────────────────────────────
    clunk::Pipeline pipeline(config);

    // Set a progress callback that prints to stderr. The "round" stage
    // fires once per refinement round — which can be thousands of times
    // on a long-running superoptimisation — so it's throttled the same
    // way Pipeline throttles its own internal verbose diagnostics (see
    // PipelineConfig::verbose_interval_seconds); one-off stage
    // transitions ("patterns"/"gpu"/"pipeline") always print immediately.
    if (opts.verbose) {
        auto last_emit = std::make_shared<
            std::unordered_map<std::string, std::chrono::steady_clock::time_point>>();
        const double interval = opts.verbose_interval;
        pipeline.set_progress_callback(
            [last_emit, interval](const std::string& stage,
               const std::string& function_name,
               double progress) {
                if (stage == "round" && interval > 0.0) {
                    const std::string key = stage + "|" + function_name;
                    const auto now = std::chrono::steady_clock::now();
                    auto it = last_emit->find(key);
                    if (it != last_emit->end() &&
                        std::chrono::duration<double>(now - it->second).count() < interval) {
                        return;  // too soon since the last "round" tick for this function
                    }
                    (*last_emit)[key] = now;
                }
                std::cerr << "[" << stage << "] " << function_name
                          << " (" << static_cast<int>(progress * 100) << "%)\n";
            });
    }

    if (opts.verbose) {
        std::cerr << "Running pipeline at opt_level=" << config.opt_level
                  << (opts.opt_level_max ? " (max)" : "")
                  << " target=" << config.target_arch.name << " ...\n";
    }

    // ── Run the pipeline, optionally under the TUI ──────────────────────
    // When --tui is passed:
    //   - Launch the pipeline on a worker thread.
    //   - Install the TUI's rich progress callback on the pipeline.
    //   - Run the TUI's render loop on the main thread.
    //   - When the worker joins, mark the TUI as "done" so the user
    //     can inspect the final state and press q to exit.
    //
    // Without --tui (or when the TUI can't be initialised — e.g. stdin
    // isn't a tty), fall back to the synchronous path: just call
    // pipeline.run(*module) directly.
    clunk::PipelineResult result;
#ifdef CLUNK_HAS_TUI
    if (opts.tui) {
        // The TUI owns ncurses state; the pipeline must NOT print to
        // stderr/stdout while it's running, or the output will corrupt
        // the screen. Suppress verbose during the TUI run.
        config.verbose = false;
        pipeline.config() = config;

        clunk::cli::TUI tui;
        if (!tui.init()) {
            std::cerr << "Warning: --tui could not initialise ncurses "
                         "(not a tty?) — falling back to non-TUI mode.\n";
            result = pipeline.run(*module);
        } else {
            // Install the TUI's rich progress callback.
            pipeline.set_rich_progress_callback(tui.make_callback());

            // Run the pipeline on a worker thread.
            std::atomic<bool> pipeline_running{true};
            std::thread worker([&]() {
                try {
                    result = pipeline.run(*module);
                } catch (...) {
                    // Swallow — the main thread will see pipeline_running
                    // go false and report the failure.
                }
                pipeline_running = false;
            });

            // Run the TUI render loop on the main thread.
            tui.run_loop(pipeline_running);

            // If the user pressed q before the pipeline finished, wait
            // for the worker to drain (the pipeline's time budget will
            // expire on the next round). Detach is unsafe (the captured
            // references would dangle), so join unconditionally.
            if (worker.joinable()) worker.join();

            // Mark the TUI as done so it shows the final state.
            std::ostringstream summary;
            summary << "Functions processed: " << result.total_functions_processed
                    << "  Optimised: " << result.total_optimised
                    << "  Avg improvement: " << result.avg_improvement << "x"
                    << "  Total time: " << result.total_time_ms << " ms";
            tui.mark_done(summary.str());

            // Render one final frame so the user sees the "done" state,
            // then wait for them to press q to exit.
            std::atomic<bool> dummy_running{false};
            tui.run_loop(dummy_running);
        }
    } else {
        result = pipeline.run(*module);
    }
#else
    if (opts.tui) {
        std::cerr << "Warning: --tui requested but ncurses was not available "
                     "at build time — running without the TUI.\n";
    }
    result = pipeline.run(*module);
#endif

    // ── Output the optimised module ─────────────────────────────────────
    if (result.optimised_module) {
        std::string output = result.optimised_module->to_string();

        if (opts.output_file.empty()) {
            // Write to stdout — unless --report-json owns stdout, in which
            // case the IR is only emitted when an explicit --output is given.
            if (!opts.report_json) {
                std::cout << output;
            }
        } else {
            std::ofstream ofs(opts.output_file);
            if (!ofs.is_open()) {
                std::cerr << "Error: cannot open output file: " << opts.output_file << "\n";
                return 1;
            }
            ofs << output;
            if (opts.verbose) {
                std::cerr << "Output written to " << opts.output_file << "\n";
            }
        }
    }

    // ── Print summary statistics to stderr ──────────────────────────────
    std::cerr << "\n=== Clunk Summary ===\n";
    std::cerr << "Functions processed: " << result.total_functions_processed << "\n";
    std::cerr << "Functions optimised: " << result.total_optimised << "\n";
    std::cerr << "Average improvement:  " << result.avg_improvement << "x\n";
    std::cerr << "Total time:           " << result.total_time_ms << " ms\n";

    if (opts.verbose) {
        std::cerr << "\n--- Per-function results ---\n";
        for (auto& [name, fr] : result.function_results) {
            std::cerr << "  " << name << ":\n"
                      << "    original score: " << fr.score_original << "\n"
                      << "    optimised score: " << fr.score_optimised << "\n"
                      << "    improvement: " << fr.improvement_ratio << "x\n"
                      << "    verified: " << (fr.verified ? "yes" : "no") << "\n"
                      << "    patterns applied: " << fr.patterns_applied << "\n"
                      << "    refinement rounds: " << fr.rounds_run
                      << " (" << fr.improvements_adopted << " adopted)\n"
                      << "    time: " << fr.time_spent_ms << " ms\n";
        }
    }

    // ── Machine-readable per-function report (--report-json) ────────────
    if (opts.report_json) {
        emit_report_json(std::cout, result);
    }

    // ── Save pattern library if it was loaded from disk ─────────────────
    auto& pattern_lib = pipeline.pattern_library();
    if (!opts.pattern_library_path.empty() && pattern_lib.size() > 0) {
        // Save any newly discovered patterns back
        pattern_lib.save(opts.pattern_library_path);
        if (opts.verbose) {
            std::cerr << "Pattern library saved (" << pattern_lib.size() << " patterns)\n";
        }
    }

    return 0;
}