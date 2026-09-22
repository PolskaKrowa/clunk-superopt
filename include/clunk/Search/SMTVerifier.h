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
 * Clunk SMT Verifier — Z3-based equivalence checking.
 * Proves that optimised candidates are semantically equivalent
 * to the original program, ensuring optimisations are safe.
 *
 * Soundness policy:
 *   The verifier NEVER reports `Equivalent` for a pair it cannot soundly
 *   model. Functions containing memory operations, floating-point ops,
 *   function calls, or unbounded loops are reported as `Unknown` rather
 *   than risk a false `Equivalent` (which would bless an incorrect
 *   optimisation). The simulation fallback (used when Z3 is not
 *   available) follows the same policy.
 *
 * Key features:
 *   - `SMTConfig::honor_binop_flags`: when true, the encoder reads
 *     `inst->binop_flags()` (nuw/nsw/exact) and wraps each flagged binop's
 *     result in `ite(overflow_condition, POISON, normal_result)` where
 *     `POISON` is a single shared free Z3 constant per bit-width. This
 *     lets the verifier distinguish `add nsw` (poison on signed overflow)
 *     from `add` (always defined) and prove/disprove equivalence of
 *     flag-aware rewrites — instead of silently treating them the same.
 *   - `synthesize_with_cegis`: CEGIS-style symbolic-constant synthesis.
 *     Given an original and a candidate template containing named
 *     placeholder `ConstantInt`s, finds concrete values for the
 *     placeholders that make the candidate equivalent to the original.
 *     Implemented via constant-pool enumeration (the brief's documented
 *     fallback when Z3 quantifier-based exists-forall is too risky to
 *     wire through the dlopen shim).
 *   - `prune_batch_with_unsat_core`: batch pre-check using
 *     `Z3_solver_check_assumptions` to short-circuit candidates that are
 *     jointly unequivalent in a single SAT check, avoiding N full
 *     `verify()` calls when an entire batch is wrong.
 *   - `SMTConfig::cache`: if non-null, `verify()` consults the
 *     `RewriteCache` before calling Z3, and stores the result after.
 */
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <utility>
#include "clunk/Evaluator/DifferentialScreen.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"  // ir::CmpPredicate (ArgAssumption)

namespace clunk::search {

// ── Forward declarations ─────────────────────────────────────────────────
class RewriteCache;  // from clunk/Search/RewriteCache.h

// ── Verification result ─────────────────────────────────────────────────
struct VerificationResult {
    enum Status {
        Equivalent,       // Proved equivalent
        NotEquivalent,    // Proved NOT equivalent
        Unknown,          // Could not determine (timeout, unsupported ops, etc.)
        Error             // Z3 error
    };

    Status status;
    std::string message;
    double solve_time_ms = 0.0;    // Time spent in Z3

    // Counterexample inputs (populated when status == NotEquivalent and Z3
    // returned a model). Each entry corresponds to a function argument in
    // positional order. Empty if no model was extracted.
    std::vector<int64_t> counterexample;

    // Human-readable reason for Unknown/Error (e.g. "function contains
    // memory operations", "Z3 timeout", "Z3 error: ...").
    std::string z3_reason;

    // True when the concrete-input screen (see SMTConfig::screen_before_
    // solving) decided this result: no solver was invoked and the
    // counterexample is a real input. Callers that budget solver calls
    // should not charge a screened result against that budget.
    bool screened = false;

    bool is_safe() const { return status == Equivalent; }
};

// ── One path-condition conjunct ───────────────────────────────────────────
//
// A constraint over the SHARED arguments of the (original, candidate) pair
// being verified: `lhs <predicate> rhs`, where each side is either an
// argument (by POSITION — the same positional Z3 constants both functions
// are encoded against) or a 64-bit constant. `negated` flips the whole
// predicate (assume NOT(lhs pred rhs)).
//
// Used by verify_with_assumptions to prove equivalence UNDER a path
// condition: the caller (PeepholeMiner::mine_with_path_conditions) maps the
// branch conditions dominating a harvested slice onto the slice's argument
// positions and passes them here. Dropping an inexpressible conjunct is
// always sound (it only widens the input space the proof must cover).
struct ArgAssumption {
    ir::CmpPredicate predicate = ir::CmpPredicate::EQ;
    int lhs_arg = -1;        // >= 0: argument index; < 0: use lhs_const
    int rhs_arg = -1;        // >= 0: argument index; < 0: use rhs_const
    int64_t lhs_const = 0;
    int64_t rhs_const = 0;
    bool negated = false;    // assume NOT(lhs pred rhs) instead
};

// ── CEGIS synthesis result ────────────────────────────────────────────────
//
// Returned by `SMTVerifier::synthesize_with_cegis`. Mirrors Souper's
// `ConstantSynthesis::synthesize` 2-query loop result: on success,
// `model` contains the synthesised concrete values for
// each placeholder; on failure, `success == false` and `message` explains
// why (no model found, Z3 returned Unknown, or template was malformed).
//
struct SynthesisResult {
    // True iff a model was found AND a final `verify()` confirmed
    // equivalence under that model (the second query in Souper's 2-query
    // CEGIS loop). When false, `verification.status` may still carry
    // diagnostic information (e.g. NotEquivalent for the last-attempted
    // instantiation, or Unknown if Z3 bailed).
    bool success = false;

    // The verification result of the final (best-guess) instantiation.
    // On `success == true`, this is `Equivalent`. On `success == false`,
    // it's the result of the last attempted constant — useful for
    // diagnostics.
    VerificationResult verification;

    // placeholder name → synthesised concrete value. Empty on failure.
    // The order matches `placeholder_names` passed to `synthesize_with_cegis`.
    std::vector<std::pair<std::string, int64_t>> model;

    // Human-readable summary ("found model in 3 tries", "no model in
    // constant pool", "Z3 unavailable", etc.).
    std::string message;

    // Wall-clock time spent in the synthesiser (ms). Includes the final
    // `verify()` call.
    double solve_time_ms = 0.0;
};

// ── SMT configuration ──────────────────────────────────────────────────
struct SMTConfig {
    // ── Concrete-input screen ───────────────────────────────────────────
    // When true (default), verify() first runs both functions on a few
    // hundred edge-case and random inputs (0, -1, INT_MIN, powers of two,
    // ... — see evaluator::DifferentialScreen). A candidate that is not a
    // refinement on some input is answered NotEquivalent immediately, with
    // that input as the counterexample; only survivors reach the solver.
    // The screen follows the same refinement semantics as the encoder, so
    // it never rejects anything the solver would have accepted.
    // verify_with_assumptions() is exempt: random inputs would violate the
    // path condition it is proving under.
    bool screen_before_solving = true;
    evaluator::ScreenConfig screen;

    unsigned timeout_ms = 30000;    // Z3 solver timeout (wall-clock)
    bool simplify_before = true;    // Run Z3 simplification first
    bool use_bitvectors = true;     // Use BV theory instead of integers
    unsigned max_unrolling = 4;     // Max loop unrolling depth for encoding

    // ── Backward-compatible additions ─────────────────────────────────────
    // Resource limit (deterministic, in millions of instructions). Acts as
    // a backup to timeout_ms which is wall-clock and can be flaky under
    // load. 0 = unlimited.
    unsigned rlimit = 0;

    // If true, fall back to `Unknown` for any function containing memory
    // operations (Load/Store/Alloca/GEP/Call) rather than attempting the
    // (incomplete) array-theory encoding. Sound. Default: true.
    bool sound_memory_fallback = true;

    // If true, fall back to `Unknown` for any function containing
    // floating-point operations rather than using the (unsound) raw-BV
    // encoding. Sound. Default: true.
    bool sound_float_fallback = true;

    // If true, fall back to `Unknown` for any function containing loops
    // (back-edges in the CFG) rather than silently mis-encoding them.
    // Sound. Proper bounded unrolling is a future improvement.
    // When set to false, loops are encoded as if executed exactly once
    // (UNSOUND — for testing only).
    bool sound_loop_fallback = true;

    // ── Bounded loop unrolling (implements the historical TODO) ────────
    // When true, `verify_with_z3` pre-unrolls constant-trip single-block
    // loops in BOTH the original and the candidate before SMT encoding.
    // The unroller (search::LoopOptimizer::unroll_constant_loops) is
    // sound-by-construction — it abstractly interprets the loop to
    // determine the trip count, and refuses to unroll anything it
    // cannot prove constant. So a function that previously triggered
    // `sound_loop_fallback` may now be SMT-verifiable after unrolling.
    //
    // Trip count is capped by `max_unrolling` (default 4) — loops with
    // larger trip counts (or non-constant trip counts) still trigger
    // `sound_loop_fallback` and return Unknown. Default: false (opt-in
    // via --smt-bounded-unrolling), to preserve backward compatibility.
    bool sound_bounded_unrolling = false;

    // ── BinOp-flags (nsw/nuw/exact) SMT encoding ───────────────────────
    // When true, the encoder inspects `inst->binop_flags()` for each
    // binary op and wraps the result in `ite(overflow_condition, POISON,
    // normal_result)` where POISON is a shared free Z3 constant of the
    // appropriate bit-width. This lets the verifier distinguish `add nsw`
    // (poison on signed overflow) from plain `add` (always defined) and
    // prove/disprove equivalence of flag-aware rewrites. Default: true.
    // When false, the verifier falls back to the plain-integer encoding
    // (today's behaviour — flags are silently ignored, which is unsound
    // for flag-exploiting rewrites).
    bool honor_binop_flags = true;

    // ── Alive2-style refinement semantics ──────────────────────────────
    // When true (default), verify() checks REFINEMENT rather than exact
    // equivalence, using per-value poison flags (the Alive2 model, minus
    // undef/memory): the candidate may replace the original iff on every
    // input where the original is well-defined, the candidate is also
    // well-defined and produces the same value. Where the original is
    // poison (nsw/nuw/exact violation, shift amount >= bit-width), the
    // candidate may do anything — which is precisely LLVM's contract and
    // what lets flag-dropping rewrites (`add nsw x,y` -> `add x,y`) be
    // proven. Poison-INTRODUCING candidates are still rejected (their
    // poison flag is checked on the candidate side).
    //
    // When false, the legacy exact-equivalence encoding is used (flagged
    // binops substitute a shared free POISON constant into the value).
    // Note: the legacy encoding does not model shift-out-of-range poison.
    bool refinement_semantics = true;

    // ── Unsat-core batch pruning ────────────────────────────────────────
    // When true, `verify_batch_with_z3` runs a single batch pre-check
    // using `Z3_solver_check_assumptions` before paying for per-candidate
    // `verify()` calls. If the batch check is SAT (all candidates jointly
    // unequivalent), the entire batch is short-circuited to
    // `NotEquivalent`. Default: true. Sound: SAT ⇒ all candidates are
    // definitely unequivalent.
    bool use_unsat_core_batch = true;

    // ── Optional rewrite cache ───────────────────────────────────────────
    // If non-null, `verify()` consults the cache before calling Z3 (using
    // `StochasticSearch::structural_hash` of the LHS and RHS as keys) and
    // stores the result after. Null = no caching (default).
    std::shared_ptr<RewriteCache> cache;

    // ── Scale-control fields ─────────────────────────────────────────────
    // Sound fallback for large/complex functions. The SMT path-condition
    // encoding is exponential in the number of basic blocks (path_pc(B) =
    // OR over preds P of (path_pc(P) AND edge_cond(P->B)), which explodes
    // for branch-heavy / diamond CFGs). We refuse to verify functions
    // that exceed these limits (returning Unknown rather than risk a
    // memory blowup or hours of solver time). Sound: Unknown never
    // blesses an incorrect optimisation.
    bool sound_large_function_fallback = true;
    size_t max_blocks_for_smt = 20;
    size_t max_instructions_for_smt = 100;
};

// ── SMT Verifier ────────────────────────────────────────────────────────
class SMTVerifier final {
public:
    explicit SMTVerifier(const SMTConfig& config = {});
    ~SMTVerifier();

    // Verify that two functions are semantically equivalent
    VerificationResult verify(const ir::Function& original,
                              const ir::Function& candidate);

    // ── Equivalence under a path condition ─────────────────────────────
    // Prove the two functions equivalent FOR ALL inputs satisfying the
    // conjunction of `assumptions` (each an icmp-style constraint over the
    // shared positional arguments). UNSAT of (assumptions ∧ ret_o ≠ ret_c)
    // establishes equivalence on the constrained input space — which is
    // exactly what a rewrite spliced into a branch-dominated block needs.
    //
    // Differences from verify():
    //   - the rewrite cache is BYPASSED (a cached unconditional verdict
    //     answers a different query than a conditional one);
    //   - there is no simulation fallback (returns Unknown without Z3).
    // With no assumptions at all this degenerates to verify().
    //
    // `condition_fn` (optional): an arbitrary i1-returning single-block
    // integer mini-function over the SAME positional/naming argument
    // convention as the two functions. Its encoded result is asserted true
    // (or false when `condition_negated`) alongside the ArgAssumption
    // conjuncts. This expresses conditions ArgAssumption cannot — e.g. a
    // select's condition `(x & 1) == 0`, whose operand is itself a derived
    // expression — which is the shape that dominates real -O3 output
    // (branches become selects). If the condition cannot be soundly
    // encoded, the result is Unknown (never a silently-weakened proof).
    VerificationResult verify_with_assumptions(
        const ir::Function& original,
        const ir::Function& candidate,
        const std::vector<ArgAssumption>& assumptions,
        const ir::Function* condition_fn = nullptr,
        bool condition_negated = false);

    // Batch verification — verify multiple candidates against the same original
    // (existing signature, preserved for backward compatibility).
    std::vector<VerificationResult> verify_batch(
        const ir::Function& original,
        const std::vector<std::shared_ptr<ir::Function>>& candidates);

    // Batch verification — overload taking Functions by value.
    // Uses Z3 push/pop to share the original's encoding across candidates
    // when Z3 is available; falls back to looping verify() otherwise.
    std::vector<VerificationResult> verify_batch(
        const ir::Function& original,
        const std::vector<ir::Function>& candidates);

    // ── CEGIS for symbolic-constant synthesis ───────────────────────────
    // Synthesise concrete values for `placeholder_names` in
    // `candidate_template` so that the resulting function is equivalent
    // to `original`. Mirrors Souper's
    // lib/Infer/ConstantSynthesis.cpp:synthesize (2-query
    // synthesis+verification loop). Returns SynthesisResult::success=false
    // if no model exists or Z3 returns Unknown.
    //
    // `placeholder_names`: names of `ConstantInt`s in `candidate_template`
    //   (set via `ConstantInt::set_name(...)`) whose values should be
    //   synthesised.
    // `placeholder_widths`: bit-widths of each placeholder, parallel to
    //   `placeholder_names` (used for the constant-pool enumeration). The
    //   synthesiser picks values that fit `placeholder_widths[i]` bits.
    //
    // Implementation: tries the real Z3 quantifier-based
    // exists-forall CEGIS first (synthesize_with_z3_quantifiers), which
    // can find ANY 64-bit placeholder value. If Z3 quantifier symbols are
    // unavailable or CEGIS doesn't converge, falls back to constant-pool
    // enumeration over {0,1,2,3,5,7,8,15,16,31,32,63,64,-1,255,256}.
    SynthesisResult synthesize_with_cegis(
        const ir::Function& original,
        const ir::Function& candidate_template,
        const std::vector<std::string>& placeholder_names,
        const std::vector<int64_t>& placeholder_widths);

    // Check if Z3 is available at runtime
    static bool is_z3_available();

    // Configuration
    SMTConfig& config() { return config_; }
    const SMTConfig& config() const { return config_; }

    // Statistics
    struct Stats {
        size_t verifications_run = 0;
        size_t equivalent = 0;
        size_t not_equivalent = 0;
        size_t unknown = 0;
        size_t errors = 0;
        size_t screened_out = 0;    // rejected by the concrete screen (no solver call)
        size_t screen_passed = 0;   // survived the screen and went on to the solver
        double total_time_ms = 0.0;
        double avg_time_ms = 0.0;
    };
    const Stats& stats() const { return stats_; }

    // ── Z3 encoding (public for access by encode helper) ────────────────
    // Encode a function's semantics as a Z3 formula
    struct SMTEncoding;

private:

    VerificationResult verify_with_z3(
        const ir::Function& original,
        const ir::Function& candidate,
        const std::vector<ArgAssumption>* assumptions = nullptr,
        const ir::Function* condition_fn = nullptr,
        bool condition_negated = false);
    VerificationResult verify_with_simulation(const ir::Function& original,
                                               const ir::Function& candidate);

    // Incremental batch verification using Z3 push/pop. Returns false if
    // Z3 is unavailable or the original cannot be encoded (caller should
    // fall back to looping verify()).
    bool verify_batch_with_z3(const ir::Function& original,
                               const std::vector<ir::Function>& candidates,
                               std::vector<VerificationResult>& results);

    // ── Unsat-core / assumption-based batch pruning ─────────────────────
    //
    // Cheap pre-check: encodes the original and ALL candidates into a
    // single Z3 solver, asserts each candidate's unequivalence (negation
    // of `cand_return == orig_return`) guarded by a fresh Bool assumption
    // literal `a_i`, and runs ONE `Z3_solver_check_assumptions` call with
    // all `a_i`'s as assumptions.
    //
    // If the check is SAT, ALL candidates are jointly unequivalent (a
    // single model witnesses a counterexample for every candidate) — we
    // short-circuit them all to `NotEquivalent` (with the model's
    // counterexample inputs) and skip per-candidate `verify()`. Sound:
    // SAT ⇒ every candidate has a counterexample.
    //
    // If the check is UNSAT (at least one candidate is equivalent or
    // Unknown), or `Z3_solver_check_assumptions` is unavailable, no
    // pruning is performed — `results` is left empty and the caller falls
    // back to `verify_batch_with_z3`'s normal per-candidate loop.
    //
    // `pruned_indices` (if non-null) receives the indices of candidates
    // that were pruned (in ascending order); the caller can use this to
    // skip them in the per-candidate loop.
    void prune_batch_with_unsat_core(const ir::Function& original,
                                      const std::vector<ir::Function>& candidates,
                                      std::vector<VerificationResult>& results,
                                      std::vector<size_t>* pruned_indices = nullptr);

    // ── Z3 quantifier-based exists-forall CEGIS ─────────────────────────
    //
    // Real CEGIS via Z3's quantifier support: alternates between
    //   (1) synthesis:  exists placeholders. forall cex. orig(cex) == cand(cex, placeholders)
    //   (2) verify:     exists inputs. orig(inputs) != cand(inputs, model_placeholders)
    // until verify returns UNSAT (the model is sound) or the iteration
    // cap is hit. This is the Souper `lib/Infer/ConstantSynthesis.cpp`
    // algorithm, with the same 2-query structure as the constant-pool
    // enumeration fallback but able to find ANY 64-bit placeholder value,
    // not just the 16-element pool.
    //
    // Returns false (caller falls back to enumeration) if Z3 quantifier
    // symbols are unavailable, Z3 returns Unknown on either query, or
    // the iteration cap is hit without convergence. On success, `out_model`
    // receives the synthesised placeholder values and the function
    // returns true.
    bool synthesize_with_z3_quantifiers(
        const ir::Function& original,
        const ir::Function& candidate_template,
        const std::vector<std::string>& placeholder_names,
        const std::vector<int64_t>& placeholder_widths,
        std::vector<std::pair<std::string, int64_t>>& out_model,
        std::string& out_message);

    // Lazily create and cache a Z3 context (creation is 1-10ms).
    // The context lives for the lifetime of the verifier (Z3 is dlopen'd
    // globally and never unloaded). Solvers are created fresh per verify()
    // call — solvers are cheap, contexts are not.
    void* get_z3_context() const;

    SMTConfig config_;
    Stats stats_;
    bool z3_available_;

    // Cached Z3 context (opaque — Z3_context is only defined in the .cpp).
    mutable void* z3_ctx_ = nullptr;
};

} // namespace clunk::search
