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
 * Clunk Algorithmic Preprocessor — implementation.
 * See include/clunk/Search/AlgoPreprocessor.h for the contract.
 *
 * Walks the module's functions in module order. For each in-scope
 * function (single-block, integer args/return, ≤64 instructions) it:
 *
 *   1. Probes the function with the Interpreter on a set of "interesting"
 *      input vectors.
 *   2. Fits the probe results to one of the closed forms:
 *        - constant       f(x) ≡ C
 *        - multiplicative f(x) ≡ c * x    (single-arg affine with b=0)
 *        - affine         f(x) ≡ c * x + b
 *   3. If a consistent fit is found, builds the minimal closed-form
 *      candidate function and SMT-proves it against the original.
 *   4. On proof success, replaces the function's body with the closed form.
 *
 * For composition collapse (g(f(x))), the pass additionally walks the
 * call graph looking for single-arg callers that call exactly one
 * single-arg callee, probes the composition, and rewrites the caller's
 * call site to use the detected closed form directly.
 */

#include "clunk/Search/AlgoPreprocessor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "clunk/Analysis/CallGraph.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/IR/Clone.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

namespace clunk::search {

namespace {

// Interesting probe inputs (small magnitudes + boundary values).
constexpr std::array<int64_t, 12> kProbes = {
    0, 1, -1, 2, -2, 3, 5, 7, 8, 15, 16, 31,
};

// Is `fn` in scope for this pass?
bool in_scope(const ir::Function& fn) {
    if (fn.blocks().size() != 1) return false;
    if (fn.argument_count() == 0 || fn.argument_count() > 4) return false;
    auto ret_ty = fn.return_type();
    if (!ret_ty || !ret_ty->is_integer()) return false;
    for (const auto& arg : fn.arguments()) {
        if (!arg.type || !arg.type->is_integer()) return false;
    }
    // Refuse functions that touch memory / calls / FP — the SMT verifier
    // refuses them anyway, but probing the interpreter would return
    // nullopt and we'd waste time building candidates.
    for (const auto& bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto& inst : bb->instructions()) {
            if (!inst) continue;
            if (inst->is_memory_op()) return false;
            if (inst->opcode() == ir::Opcode::Call) return false;
            switch (inst->opcode()) {
            case ir::Opcode::FAdd: case ir::Opcode::FSub:
            case ir::Opcode::FMul: case ir::Opcode::FDiv:
            case ir::Opcode::FRem: case ir::Opcode::FCmp:
                return false;
            default: break;
            }
        }
    }
    return true;
}

unsigned common_int_width(const ir::Function& fn) {
    auto ret_ty = fn.return_type();
    auto* it = dynamic_cast<const ir::IntegerType*>(ret_ty.get());
    return it ? it->bits() : 32u;
}

// Probe the function on the probe set. Returns nullopt if the
// interpreter refuses any probe (we need a complete probe set to fit
// a closed form).
struct Probe {
    std::vector<int64_t> args;
    int64_t result;
};
std::optional<std::vector<Probe>> probe_function(const ir::Function& fn,
                                                   size_t max_probes) {
    std::vector<Probe> out;
    size_t nargs = fn.argument_count();
    size_t n_probes = std::min(max_probes, kProbes.size());

    if (nargs == 1) {
        for (size_t i = 0; i < n_probes; ++i) {
            std::vector<int64_t> args = {kProbes[i]};
            auto r = evaluator::Interpreter::interpret(fn, args);
            if (!r) return std::nullopt;
            out.push_back({std::move(args), *r});
        }
    } else {
        // For multi-arg: probe along the "diagonal" + a few off-diagonal
        // points. We can only fit closed forms over a single input, so
        // multi-arg detection is restricted to constant-output (where
        // the same value appears on every probe).
        for (size_t i = 0; i < n_probes; ++i) {
            std::vector<int64_t> args(nargs, kProbes[i]);
            for (size_t j = 1; j < nargs; ++j) {
                args[j] = kProbes[(i + j) % kProbes.size()];
            }
            auto r = evaluator::Interpreter::interpret(fn, args);
            if (!r) return std::nullopt;
            out.push_back({std::move(args), *r});
        }
    }
    return out;
}

// ── Pattern detection ─────────────────────────────────────────────────────
//
// Each detector returns the (c, b) coefficients if the pattern fits the
// probe set. They return nullopt if the probe set is inconsistent with
// the pattern.

// f(x) ≡ C? Returns (0, C) on success (i.e. c=0, b=C).
struct Coeffs { int64_t c; int64_t b; };
std::optional<Coeffs> detect_constant(const std::vector<Probe>& probes) {
    if (probes.empty()) return std::nullopt;
    int64_t v = probes[0].result;
    for (const auto& p : probes) {
        if (p.result != v) return std::nullopt;
    }
    return Coeffs{0, v};
}

// f(x) ≡ c*x? Returns (c, 0) on success. Requires single-arg probes.
std::optional<Coeffs> detect_multiplicative(const std::vector<Probe>& probes) {
    if (probes.empty()) return std::nullopt;
    // Need at least one non-zero input to identify c.
    int64_t c = 0;
    bool c_set = false;
    for (const auto& p : probes) {
        if (p.args.empty() || p.args[0] == 0) continue;
        // f(x) / x should equal c for every probe.
        // Avoid 64-bit INT64_MIN division issues by checking via multiplication.
        // Check: x * (f(x) / x) == f(x)  &&  f(x) % x == 0.
        if (p.args[0] == INT64_MIN) continue;  // INT64_MIN / -1 is UB
        int64_t this_c;
        if (p.args[0] != 0) {
            // Use the candidate c and verify with multiplication.
            // First, derive this_c = f(x) / x  (must divide evenly).
            if (p.args[0] == -1) {
                this_c = -p.result;
            } else {
                if (p.result % p.args[0] != 0) return std::nullopt;
                this_c = p.result / p.args[0];
            }
        } else {
            continue;
        }
        if (!c_set) { c = this_c; c_set = true; }
        else if (c != this_c) return std::nullopt;
    }
    // Verify: f(0) must equal 0 (since c*0 = 0).
    for (const auto& p : probes) {
        if (!p.args.empty() && p.args[0] == 0 && p.result != 0) {
            return std::nullopt;
        }
    }
    if (!c_set) return std::nullopt;
    return Coeffs{c, 0};
}

// f(x) ≡ c*x + b? Returns (c, b) on success. Single-arg probes only.
std::optional<Coeffs> detect_affine(const std::vector<Probe>& probes) {
    if (probes.size() < 3) return std::nullopt;
    // b = f(0). Find a probe with x=0.
    int64_t b = 0;
    bool b_set = false;
    for (const auto& p : probes) {
        if (!p.args.empty() && p.args[0] == 0) {
            b = p.result;
            b_set = true;
            break;
        }
    }
    if (!b_set) return std::nullopt;
    // c = (f(1) - b). Need a probe with x=1.
    int64_t c = 0;
    bool c_set = false;
    for (const auto& p : probes) {
        if (!p.args.empty() && p.args[0] == 1) {
            c = p.result - b;
            c_set = true;
            break;
        }
    }
    if (!c_set) return std::nullopt;
    // Verify: every probe must satisfy f(x) == c*x + b.
    for (const auto& p : probes) {
        if (p.args.empty()) continue;
        int64_t x = p.args[0];
        // Compute c*x + b with overflow-safe builtins. If the multiply
        // or add overflows, the probe doesn't fit our closed form (which
        // is exact, not modular — the interpreter masks results but the
        // pattern detector requires the closed form to hold without
        // overflow).
        int64_t cx = 0, cxb = 0;
        if (__builtin_mul_overflow(c, x, &cx)) return std::nullopt;
        if (__builtin_add_overflow(cx, b, &cxb)) return std::nullopt;
        if (cxb != p.result) return std::nullopt;
    }
    // Reject the trivial c=0, b=0 case (it's already caught by detect_constant).
    if (c == 0) return std::nullopt;
    return Coeffs{c, b};
}

// ── Candidate materialisation ─────────────────────────────────────────────
//
// Build a fresh `ir::Function` with the same signature as `orig`, a
// single "entry" block, and a body that computes c*x + b (or just `ret
// C` when c == 0).

std::shared_ptr<ir::Function> make_hole_skeleton(const ir::Function& src) {
    auto fn_type = src.function_type();
    auto out = std::make_shared<ir::Function>(src.name(), fn_type, src.linkage());
    for (const auto& arg : src.arguments()) {
        out->add_argument(arg.type, arg.name, arg.attrs);
    }
    out->add_block("entry");
    return out;
}

// Build the closed-form candidate `ret (c * arg0 + b)` for the given
// coefficients. If `c == 0`, this is just `ret b`. If `b == 0`, this is
// `ret (c * arg0)`.
//
// For single-arg functions only — multi-arg functions are only ever
// detected as constant (where the closed form is `ret C` regardless of
// args).
std::shared_ptr<ir::Function> build_affine_candidate(
    const ir::Function& orig, int64_t c, int64_t b,
    ir::TypeContext& ctx) {

    unsigned width = common_int_width(orig);
    auto out = make_hole_skeleton(orig);
    auto bb = out->entry_block();

    // The first argument's value (we'll use it as the input).
    if (orig.argument_count() == 0) {
        // No args — only constant detection can succeed here.
        auto ret_val = ir::ConstantInt::get(ctx, b, width);
        bb->add_instruction(ir::inst::make_ret(ret_val));
        if (!ir::validate_function(*out)) return nullptr;
        return out;
    }

    // arg0 as a Value (referenced by name).
    auto arg0 = std::make_shared<ir::Value>(
        orig.arguments()[0].type, orig.arguments()[0].name);

    std::shared_ptr<ir::Value> result;

    if (c == 0) {
        // ret b
        result = ir::ConstantInt::get(ctx, b, width);
    } else {
        // Compute c * arg0 (+ b)
        auto c_const = ir::ConstantInt::get(ctx, c, width);
        auto mul = ir::inst::make_mul(arg0, c_const, "scaled");
        bb->add_instruction(mul);
        if (b != 0) {
            auto b_const = ir::ConstantInt::get(ctx, b, width);
            auto add = ir::inst::make_add(mul, b_const, "shifted");
            bb->add_instruction(add);
            result = add;
        } else {
            result = mul;
        }
    }

    bb->add_instruction(ir::inst::make_ret(result));
    if (!ir::validate_function(*out)) return nullptr;
    return out;
}

// ── Composition analysis ──────────────────────────────────────────────────
//
// Walk `caller`'s calls. If it has exactly one call to a single-arg
// callee whose argument is caller's arg0, and the caller's body is
// otherwise just `ret <call_result>`, then we have a pure wrapper
// g(f(x)) and can collapse it.

struct CompositionChain {
    std::vector<std::string> callee_names;  // in call order, outermost first
};

// Identify whether `caller` is a single-arg wrapper around a chain of
// single-arg callees. Returns the chain (outermost first) on success.
std::optional<CompositionChain> detect_wrapper_chain(const ir::Function& caller) {
    if (caller.argument_count() != 1) return std::nullopt;
    if (caller.blocks().size() != 1) return std::nullopt;
    auto bb = caller.entry_block();
    if (!bb) return std::nullopt;

    CompositionChain chain;
    // Walk the block: each call to a single-arg callee feeding the next
    // instruction (or the ret) is part of the chain.
    auto current_input_name = caller.arguments()[0].name;

    for (const auto& inst : bb->instructions()) {
        if (!inst) continue;
        if (inst->opcode() == ir::Opcode::Call) {
            // Must take exactly 1 arg, and that arg must be the
            // current_input_name.
            if (inst->num_operands() != 1) return std::nullopt;
            auto arg = inst->operand(0);
            if (!arg || !arg->has_name()) return std::nullopt;
            if (arg->name() != current_input_name) return std::nullopt;
            auto it = inst->metadata().find("callee");
            if (it == inst->metadata().end()) return std::nullopt;
            chain.callee_names.push_back(it->second);
            current_input_name = inst->name();
        } else if (inst->opcode() == ir::Opcode::Ret) {
            if (inst->num_operands() != 1) return std::nullopt;
            auto ret_val = inst->operand(0);
            if (!ret_val || !ret_val->has_name()) return std::nullopt;
            if (ret_val->name() != current_input_name) {
                // The ret doesn't return the chain's result — not a
                // pure wrapper.
                return std::nullopt;
            }
            return chain;
        } else {
            // Any other instruction means the wrapper does extra work
            // — not a pure composition.
            return std::nullopt;
        }
    }
    return std::nullopt;
}

} // anonymous namespace

// ── Public driver ────────────────────────────────────────────────────────

bool AlgoPreprocessor::run(ir::Module& module) {
    bool any_change = false;

    // Build the call graph once for composition analysis.
    analysis::CallGraph cg;
    std::vector<std::string> entries;
    for (const auto& fn : module.functions()) {
        if (fn && (fn->linkage() == ir::Linkage::External ||
                   fn->linkage() == ir::Linkage::Weak)) {
            entries.push_back(fn->name());
        }
    }
    cg.build(module, entries);

    // SMT verifier for the proof step.
    SMTConfig scfg;
    scfg.timeout_ms = config_.smt_timeout_ms;
    SMTVerifier verifier(scfg);
    const bool z3_available = SMTVerifier::is_z3_available();
    if (!z3_available && !config_.trust_unverified) {
        // Nothing to do without a prover — bail early.
        return false;
    }

    ir::TypeContext ctx;

    // Walk functions in module order. (CallGraph::sccs() gives
    // reverse-topo for composition, but for direct closed-form
    // detection any order works since each function is proven
    // independently against its own original.)
    for (auto& fn_ptr : module.functions()) {
        if (!fn_ptr) continue;
        ir::Function& fn = *fn_ptr;
        ++stats_.functions_scanned;

        if (!in_scope(fn) ||
            fn.instruction_count() > config_.max_function_instructions) {
            ++stats_.functions_skipped;
            continue;
        }

        // Probe the function.
        auto probes = probe_function(fn, config_.max_probes);
        if (!probes || probes->empty()) {
            ++stats_.functions_skipped;
            continue;
        }

        // Try the closed-form detectors in order: constant, then
        // multiplicative, then affine.
        std::optional<Coeffs> coeffs;
        enum class Pattern { Constant, Multiplicative, Affine, None };
        Pattern detected = Pattern::None;

        if (auto c = detect_constant(*probes)) {
            coeffs = c;
            detected = Pattern::Constant;
        } else if (fn.argument_count() == 1) {
            if (auto c = detect_multiplicative(*probes)) {
                coeffs = c;
                detected = Pattern::Multiplicative;
            } else if (auto c = detect_affine(*probes)) {
                coeffs = c;
                detected = Pattern::Affine;
            }
        }

        if (detected == Pattern::None) continue;

        // Build the candidate.
        auto candidate = build_affine_candidate(fn, coeffs->c, coeffs->b, ctx);
        if (!candidate) continue;

        // SMT-prove equivalence.
        bool verified = false;
        if (z3_available) {
            auto res = verifier.verify(fn, *candidate);
            if (res.status == VerificationResult::Equivalent) {
                verified = true;
            } else {
                ++stats_.rewrites_rejected_by_smt;
            }
        }
        if (!verified && !config_.trust_unverified) continue;

        // Replace the function's body with the candidate's body.
        // We rebuild the Function in place: clear its blocks and copy
        // the candidate's blocks over.
        //
        // Approach: since the Function API doesn't expose "clear blocks",
        // we mutate the existing first block's instruction list and
        // drop any extra blocks.
        auto& blocks = const_cast<std::vector<std::shared_ptr<ir::BasicBlock>>&>(
            fn.blocks());
        // Replace the entry block's instructions with the candidate's.
        if (!blocks.empty()) {
            auto& entry_instrs = blocks[0]->instructions();
            entry_instrs.clear();
            for (const auto& inst : candidate->entry_block()->instructions()) {
                if (inst) entry_instrs.push_back(inst);
            }
        }
        // Drop any additional blocks.
        while (blocks.size() > 1) {
            blocks.pop_back();
        }
        fn.rebuild_block_index();
        fn.compute_predecessors();

        // Update stats.
        switch (detected) {
        case Pattern::Constant:      ++stats_.constants_detected; break;
        case Pattern::Multiplicative: ++stats_.scalars_detected; break;
        case Pattern::Affine:        ++stats_.affines_detected; break;
        default: break;
        }
        ++stats_.rewrites_proven;
        any_change = true;
    }

    // ── Composition collapse ────────────────────────────────────────────
    //
    // For each function that is a pure wrapper (calls a chain of
    // single-arg callees and returns the result), probe the composition
    // and try to detect a closed form. If found, inline the closed form
    // into the caller.
    //
    // We re-walk the module after the per-function rewrites, because
    // the callees may now be in their minimal closed form (so the
    // composition's closed form is simpler).
    if (any_change) {
        for (auto& fn_ptr : module.functions()) {
            if (!fn_ptr) continue;
            ir::Function& fn = *fn_ptr;

            auto chain = detect_wrapper_chain(fn);
            if (!chain || chain->callee_names.empty()) continue;
            if (chain->callee_names.size() > config_.max_composition_depth) continue;

            // Build a synthetic function that inlines the chain and
            // SMT-prove the collapsed form.

            // For now, restrict to single-callee compositions (the
            // common case).
            if (chain->callee_names.size() != 1) continue;

            auto callee = module.function(chain->callee_names[0]);
            if (!callee) continue;
            if (callee->argument_count() != 1) continue;

            // Probe the composition: for each probe x, evaluate
            // callee(x) then wrap it (the wrapper IS the composition).
            std::vector<Probe> composed_probes;
            bool ok = true;
            for (int64_t x : kProbes) {
                auto intermediate = evaluator::Interpreter::interpret(
                    *callee, {x});
                if (!intermediate) { ok = false; break; }
                // The wrapper just returns the callee's result, so
                // composed(x) = intermediate.
                composed_probes.push_back({{x}, *intermediate});
            }
            if (!ok || composed_probes.empty()) continue;

            // Detect a closed form.
            std::optional<Coeffs> coeffs;
            enum class Pattern { Constant, Multiplicative, Affine, None };
            Pattern detected = Pattern::None;
            if (auto c = detect_constant(composed_probes)) {
                coeffs = c; detected = Pattern::Constant;
            } else if (auto c = detect_multiplicative(composed_probes)) {
                coeffs = c; detected = Pattern::Multiplicative;
            } else if (auto c = detect_affine(composed_probes)) {
                coeffs = c; detected = Pattern::Affine;
            }
            if (detected == Pattern::None) continue;

            // Build the candidate.
            auto candidate = build_affine_candidate(fn, coeffs->c, coeffs->b, ctx);
            if (!candidate) continue;

            // SMT-prove. The SMT verifier needs the callee to be in
            // the same module — it is.
            bool verified = false;
            if (z3_available) {
                auto res = verifier.verify(fn, *candidate);
                if (res.status == VerificationResult::Equivalent) {
                    verified = true;
                }
            }
            if (!verified && !config_.trust_unverified) continue;

            // Inline the closed form into the caller.
            auto& blocks = const_cast<std::vector<std::shared_ptr<ir::BasicBlock>>&>(
                fn.blocks());
            if (!blocks.empty()) {
                auto& entry_instrs = blocks[0]->instructions();
                entry_instrs.clear();
                for (const auto& inst : candidate->entry_block()->instructions()) {
                    if (inst) entry_instrs.push_back(inst);
                }
            }
            while (blocks.size() > 1) {
                blocks.pop_back();
            }
            fn.rebuild_block_index();
            fn.compute_predecessors();

            ++stats_.compositions_detected;
            ++stats_.rewrites_proven;
            any_change = true;
        }
    }

    return any_change;
}

} // namespace clunk::search
