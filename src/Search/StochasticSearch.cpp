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
 * Clunk Stochastic Search — random exploration of the optimisation space.
 * Quickly finds good candidate instruction sequences using simulated annealing.
 *
 */
#include "clunk/Search/StochasticSearch.h"
#include "clunk/Search/MutationScope.h"
#include "clunk/IR/Clone.h"
#include "clunk/Evaluator/DifferentialScreen.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Pattern/PatternLibrary.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace clunk::search {

// ── Helper: count total instructions in a function ──────────────────────────
static size_t count_instructions(const ir::Function& fn) {
    size_t count = 0;
    for (auto& block : fn.blocks()) {
        if (block) count += block->size();
    }
    return count;
}

// ── Helper: check if a value is a ConstantInt ──────────────────────────────
static bool is_constant_int(const std::shared_ptr<ir::Value>& v) {
    return v && dynamic_cast<const ir::ConstantInt*>(v.get()) != nullptr;
}

static ir::ConstantInt* as_constant_int(const std::shared_ptr<ir::Value>& v) {
    return dynamic_cast<ir::ConstantInt*>(v.get());
}

// ── Helper: check if an integer is a power of 2 ────────────────────────────
static bool is_power_of_two(int64_t v) {
    return v > 0 && (v & (v - 1)) == 0;
}

static unsigned log2_of_power_of_two(int64_t v) {
    unsigned shift = 0;
    while (v > 1) {
        v >>= 1;
        ++shift;
    }
    return shift;
}

// ── Helper: integer semantics for sound constant folding ───────────────────

// Wrap an int64 result to `bw`-bit two's complement (sign-extended back to
// int64). LLVM integer ops wrap at the type's bit width; folding with raw
// int64 arithmetic would otherwise produce out-of-range constants.
static int64_t wrap_to_width(int64_t v, unsigned bw) {
    if (bw == 0 || bw >= 64) return v;
    const uint64_t mask = (uint64_t(1) << bw) - 1;
    uint64_t r = static_cast<uint64_t>(v) & mask;
    if (r & (uint64_t(1) << (bw - 1))) r |= ~mask;  // sign-extend
    return static_cast<int64_t>(r);
}

// Interpret a stored constant as an unsigned bw-bit value (for udiv/urem/
// lshr semantics).
static uint64_t as_unsigned(int64_t v, unsigned bw) {
    if (bw == 0 || bw >= 64) return static_cast<uint64_t>(v);
    return static_cast<uint64_t>(v) & ((uint64_t(1) << bw) - 1);
}

// True iff the constant is the all-ones pattern for its bit width
// (i.e. -1 in two's complement).
static bool is_all_ones(const ir::ConstantInt& ci) {
    unsigned bw = ci.bit_width();
    if (bw == 0 || bw >= 64) return ci.value() == -1;
    const uint64_t mask = (uint64_t(1) << bw) - 1;
    return (static_cast<uint64_t>(ci.value()) & mask) == mask;
}

// ── Helper: memory / reordering safety ──────────────────────────────────────

// True iff the instruction reads or writes memory, or has side effects
// that pin it in program order.
static bool touches_memory(const ir::Instruction& inst) {
    switch (inst.opcode()) {
        case ir::Opcode::Load:
        case ir::Opcode::Store:
        case ir::Opcode::Call:
        case ir::Opcode::Invoke:
        case ir::Opcode::Fence:
        case ir::Opcode::Alloca:
        case ir::Opcode::VAArg:
            return true;
        default:
            return inst.is_volatile();
    }
}

// True iff the instruction may WRITE memory / have externally visible
// effects (stores, calls, fences — and volatile loads, which must not be
// reordered).
static bool may_write_memory(const ir::Instruction& inst) {
    switch (inst.opcode()) {
        case ir::Opcode::Store:
        case ir::Opcode::Call:
        case ir::Opcode::Invoke:
        case ir::Opcode::Fence:
            return true;
        case ir::Opcode::Load:
            return inst.is_volatile();
        default:
            return false;
    }
}

// Is it safe (semantics-preserving) to swap two ADJACENT instructions,
// assuming SSA dependences between them were already ruled out?
static bool swap_memory_safe(const ir::Instruction& a, const ir::Instruction& b) {
    // PHI nodes must stay at the head of their block, in order.
    if (a.opcode() == ir::Opcode::Phi || b.opcode() == ir::Opcode::Phi) return false;
    // A writer never moves past anything that touches memory, and
    // vice versa. (Load/load reordering remains allowed.)
    if (may_write_memory(a) && touches_memory(b)) return false;
    if (may_write_memory(b) && touches_memory(a)) return false;
    return true;
}

// ── Helper: algebraic identities ─────────────────────────────────────────────

// Result of classifying an instruction against the identity table.
//   keep_operand >= 0 → replace all uses of the result with operand(keep_operand)
//   keep_operand == -1 → replace all uses with `constant`
struct IdentityRewrite {
    int keep_operand = -2;   // -2 = not applicable
    int64_t constant = 0;
    bool applicable() const { return keep_operand >= -1; }
};

// Classify integer binops against a table of always-valid algebraic
// identities. Only rewrites that are unconditionally semantics-preserving
// (or refine poison to a defined value, which is sound) are listed.
static IdentityRewrite classify_identity(const ir::Instruction& inst) {
    IdentityRewrite r;
    if (!inst.is_binary_op()) return r;
    if (inst.num_operands() < 2) return r;
    if (!inst.type() || inst.type()->bit_width() == 0) return r;

    // ── Floating point: ONLY identities that are bit-exact under IEEE-754
    // default semantics, valid without any fast-math flags. Each preserves
    // NaN propagation, infinities, AND the sign of zero:
    //   fmul x, 1.0  → x   (exact product)
    //   fdiv x, 1.0  → x   (exact quotient)
    //   fadd x, -0.0 → x   (+0 + -0 = +0, -0 + -0 = -0 — adding -0.0 is
    //                       the true FP additive identity; x + +0.0 is NOT:
    //                       -0.0 + +0.0 = +0.0 flips the zero sign)
    //   fsub x, +0.0 → x   (x - +0.0 ≡ x + -0.0, same reasoning)
    // Anything else (reassociation, x*0, x-x, ...) stays off the table.
    {
        const auto fop = inst.opcode();
        const bool is_fp_binop =
            fop == ir::Opcode::FAdd || fop == ir::Opcode::FSub ||
            fop == ir::Opcode::FMul || fop == ir::Opcode::FDiv ||
            fop == ir::Opcode::FRem;
        if (is_fp_binop) {
            auto f0 = dynamic_cast<ir::ConstantFP*>(inst.operand(0).get());
            auto f1 = dynamic_cast<ir::ConstantFP*>(inst.operand(1).get());
            auto is_pos_zero = [](const ir::ConstantFP* c) {
                return c && c->value() == 0.0 && !std::signbit(c->value());
            };
            auto is_neg_zero = [](const ir::ConstantFP* c) {
                return c && c->value() == 0.0 && std::signbit(c->value());
            };
            auto is_one = [](const ir::ConstantFP* c) {
                return c && c->value() == 1.0;
            };
            switch (fop) {
                case ir::Opcode::FMul:
                    if (is_one(f1))      r.keep_operand = 0;
                    else if (is_one(f0)) r.keep_operand = 1;
                    break;
                case ir::Opcode::FDiv:
                    if (is_one(f1)) r.keep_operand = 0;
                    break;
                case ir::Opcode::FAdd:
                    if (is_neg_zero(f1))      r.keep_operand = 0;
                    else if (is_neg_zero(f0)) r.keep_operand = 1;
                    break;
                case ir::Opcode::FSub:
                    if (is_pos_zero(f1)) r.keep_operand = 0;
                    break;
                default: break;
            }
            return r;  // no integer identity applies to an FP binop
        }
    }

    auto op0 = inst.operand(0);
    auto op1 = inst.operand(1);
    if (!op0 || !op1) return r;
    auto c0 = dynamic_cast<ir::ConstantInt*>(op0.get());
    auto c1 = dynamic_cast<ir::ConstantInt*>(op1.get());
    const bool same_var = !c0 && !c1 &&
                          op0->has_name() && op1->has_name() &&
                          op0->name() == op1->name();
    const unsigned bw = inst.type()->bit_width();

    auto is_zero = [bw](ir::ConstantInt* c) {
        return c && as_unsigned(c->value(), bw) == 0;
    };
    auto is_one = [bw](ir::ConstantInt* c) {
        return c && as_unsigned(c->value(), bw) == 1;
    };

    switch (inst.opcode()) {
        case ir::Opcode::Add:
            if (is_zero(c1)) r.keep_operand = 0;
            else if (is_zero(c0)) r.keep_operand = 1;
            break;
        case ir::Opcode::Sub:
            if (is_zero(c1)) r.keep_operand = 0;
            else if (same_var) { r.keep_operand = -1; r.constant = 0; }
            break;
        case ir::Opcode::Mul:
            if (is_one(c1)) r.keep_operand = 0;
            else if (is_one(c0)) r.keep_operand = 1;
            else if (is_zero(c0) || is_zero(c1)) { r.keep_operand = -1; r.constant = 0; }
            break;
        case ir::Opcode::UDiv:
        case ir::Opcode::SDiv:
            if (is_one(c1)) r.keep_operand = 0;
            break;
        case ir::Opcode::URem:
        case ir::Opcode::SRem:
            if (is_one(c1)) { r.keep_operand = -1; r.constant = 0; }
            break;
        case ir::Opcode::And:
            if (same_var) r.keep_operand = 0;
            else if (is_zero(c0) || is_zero(c1)) { r.keep_operand = -1; r.constant = 0; }
            else if (c1 && is_all_ones(*c1)) r.keep_operand = 0;
            else if (c0 && is_all_ones(*c0)) r.keep_operand = 1;
            break;
        case ir::Opcode::Or:
            if (same_var) r.keep_operand = 0;
            else if (is_zero(c1)) r.keep_operand = 0;
            else if (is_zero(c0)) r.keep_operand = 1;
            else if ((c1 && is_all_ones(*c1)) || (c0 && is_all_ones(*c0))) {
                r.keep_operand = -1;
                r.constant = wrap_to_width(-1, bw);
            }
            break;
        case ir::Opcode::Xor:
            if (same_var) { r.keep_operand = -1; r.constant = 0; }
            else if (is_zero(c1)) r.keep_operand = 0;
            else if (is_zero(c0)) r.keep_operand = 1;
            break;
        case ir::Opcode::Shl:
        case ir::Opcode::LShr:
        case ir::Opcode::AShr:
            if (is_zero(c1)) r.keep_operand = 0;
            // 0 shifted by anything in-range is 0. (Out-of-range shift
            // amounts produce poison; refining poison to 0 is sound.)
            else if (is_zero(c0)) { r.keep_operand = -1; r.constant = 0; }
            break;
        default:
            break;
    }
    return r;
}

// ── Helper: CSE keys ─────────────────────────────────────────────────────────

// Build a structural key for same-block common-subexpression elimination,
// or an empty string when the instruction is not CSE-eligible. Eligible:
// pure, deterministic, non-memory instructions. The key covers opcode,
// binop flags, cmp predicate (metadata "pred"), result type, and the
// operand list (constants by value, everything else by name).
static std::string cse_key(const ir::Instruction& inst) {
    switch (inst.opcode()) {
        case ir::Opcode::Add: case ir::Opcode::Sub: case ir::Opcode::Mul:
        case ir::Opcode::UDiv: case ir::Opcode::SDiv:
        case ir::Opcode::URem: case ir::Opcode::SRem:
        case ir::Opcode::And: case ir::Opcode::Or: case ir::Opcode::Xor:
        case ir::Opcode::Shl: case ir::Opcode::LShr: case ir::Opcode::AShr:
        case ir::Opcode::ICmp: case ir::Opcode::Select:
        case ir::Opcode::Trunc: case ir::Opcode::ZExt: case ir::Opcode::SExt:
        case ir::Opcode::PtrToInt: case ir::Opcode::IntToPtr:
        case ir::Opcode::BitCast:
        case ir::Opcode::GetElementPtr:
            break;  // eligible
        default:
            return {};  // memory ops, calls, phis, FP (NaN quirks), etc.
    }
    if (!inst.has_name()) return {};  // result must be nameable to reuse

    std::string key;
    key.reserve(64);
    key += std::to_string(static_cast<unsigned>(inst.opcode()));
    key += '|';
    key += inst.binop_flags().to_string();
    key += '|';
    // Cmp predicate / other semantic metadata.
    auto& md = inst.metadata();
    auto it = md.find("pred");
    if (it != md.end()) { key += it->second; }
    key += '|';
    if (inst.type()) key += inst.type()->to_string();
    for (auto& op : inst.operands()) {
        key += '|';
        if (!op) { key += "<null>"; continue; }
        if (auto ci = dynamic_cast<ir::ConstantInt*>(op.get())) {
            key += '#';
            key += std::to_string(ci->value());
            key += ':';
            key += std::to_string(ci->bit_width());
        } else if (op->has_name()) {
            key += '%';
            key += op->name();
        } else {
            return {};  // unnamed non-constant operand — not comparable
        }
    }
    return key;
}

// ── Helper: count uses of a named value across the whole function ──────────
// Used by combine_instructions to check inst_a has no uses other than inst_b
// before removing it.
static size_t count_uses_excluding(const ir::Function& fn,
                                     const std::string& value_name,
                                     const ir::Instruction* exclude_inst) {
    size_t uses = 0;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            if (inst.get() == exclude_inst) continue;
            for (auto& op : inst->operands()) {
                if (op && op->has_name() && op->name() == value_name) {
                    ++uses;
                }
            }
        }
    }
    return uses;
}

// ── Helper: replace all uses of `old_name` with `new_value` across fn ──────
// Used by fold_constant to rewrite operand references when we replace a
// binop with a ConstantInt.
static void replace_all_uses(ir::Function& fn,
                              const std::string& old_name,
                              std::shared_ptr<ir::Value> new_value) {
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->num_operands(); ++i) {
                auto& op = inst->operands()[i];
                if (op && op->has_name() && op->name() == old_name) {
                    inst->set_operand(i, new_value);
                }
            }
        }
    }
}

// ── Helper: deep-copy a single instruction ─────────────────────────────────
// Same field-for-field copy the whole-function deep_copy_function performs,
// but for one instruction. The in-place rewrite path clones only the handful
// of instructions it actually touches, instead of the entire function.
static std::shared_ptr<ir::Instruction> clone_instruction(const ir::Instruction& inst) {
    auto ni = std::make_shared<ir::Instruction>(inst.opcode(), inst.type(), inst.name());
    for (auto& op : inst.operands()) ni->add_operand(op);
    for (auto& [k, v] : inst.metadata()) ni->set_metadata(k, v);
    ni->binop_flags() = inst.binop_flags();
    if (inst.alignment()) ni->set_alignment(inst.alignment().value());
    ni->set_volatile(inst.is_volatile());
    return ni;
}

// ── Helper: transactional replace_all_uses ──────────────────────────────────
// In-place counterpart to replace_all_uses. Rather than mutating operands of
// live instructions (which a MutationScope cannot undo — it records
// instruction-level swaps, not operand edits), it swaps each USING instruction
// for a clone with the operand substituted and records a Replace so rejection
// restores the original pointer. Only instructions that reference `old_name`
// are cloned (usually one or two), so this stays far cheaper than a
// whole-function deep copy.
static void replace_all_uses_scoped(MutationScope& scope,
                                    const std::string& old_name,
                                    const std::shared_ptr<ir::Value>& new_value) {
    ir::Function& fn = scope.function();
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (size_t i = 0; i < block->size(); ++i) {
            auto inst = block->instruction(i);
            if (!inst) continue;
            bool uses = false;
            for (auto& op : inst->operands()) {
                if (op && op->has_name() && op->name() == old_name) { uses = true; break; }
            }
            if (!uses) continue;
            auto repl = clone_instruction(*inst);
            for (size_t k = 0; k < repl->num_operands(); ++k) {
                auto op = repl->operand(k);
                if (op && op->has_name() && op->name() == old_name) {
                    repl->set_operand(k, new_value);
                }
            }
            block->replace_instruction(i, repl);
            scope.record_replace(block->name(), i, inst);
        }
    }
}

// ── Shared mutation compute (soundness-critical; used by both paths) ─────────

// Build the strength-reduced replacement for `inst` (mul/udiv/urem by a power
// of two → shl/lshr/and), or nullptr if the reduction does not apply. The
// replacement carries the same result name, so uses resolve unchanged.
// True iff dividing by the FP constant `c` can be rewritten as multiplying
// by its reciprocal with BIT-EXACT results: c must be ±2^k with 1/c finite
// (and, for float-typed constants, exactly representable in float). Both
// operations then round identically for every input, including subnormals,
// infinities and NaNs.
static bool fp_reciprocal_is_exact(double c, bool is_float_ty, double& recip) {
    if (c == 0.0 || !std::isfinite(c)) return false;
    int exp = 0;
    const double mant = std::frexp(std::fabs(c), &exp);
    if (mant != 0.5) return false;  // not a power of two
    recip = 1.0 / c;
    if (!std::isfinite(recip) || 1.0 / recip != c) return false;
    if (is_float_ty &&
        static_cast<double>(static_cast<float>(recip)) != recip) {
        return false;
    }
    return true;
}

static std::shared_ptr<ir::Instruction> build_strength_reduced(
    const ir::Instruction& inst) {
    if (inst.num_operands() < 2) return nullptr;
    const ir::Opcode op = inst.opcode();

    // ── FP: fdiv x, ±2^k → fmul x, ±2^-k (bit-exact, no fast-math) ─────
    if (op == ir::Opcode::FDiv) {
        auto cfp = dynamic_cast<ir::ConstantFP*>(inst.operand(1).get());
        if (!cfp || !inst.type() ||
            !(inst.type()->is_float() || inst.type()->is_double())) {
            return nullptr;
        }
        double recip = 0.0;
        if (!fp_reciprocal_is_exact(cfp->value(), inst.type()->is_float(), recip))
            return nullptr;
        auto repl = std::make_shared<ir::Instruction>(ir::Opcode::FMul,
                                                      inst.type(), inst.name());
        repl->add_operand(inst.operand(0));
        repl->add_operand(std::make_shared<ir::ConstantFP>(inst.type(), recip));
        return repl;
    }

    if (op != ir::Opcode::Mul && op != ir::Opcode::UDiv &&
        op != ir::Opcode::URem) {
        return nullptr;
    }
    if (!inst.type() || inst.type()->bit_width() == 0) return nullptr;
    const unsigned bw = inst.type()->bit_width();

    auto op0 = inst.operand(0);
    auto op1 = inst.operand(1);

    int64_t constant = 0;
    size_t var_operand = 0;  // index of the non-constant operand

    if (op == ir::Opcode::Mul) {
        // mul is commutative — the constant may be on either side.
        if (is_constant_int(op1)) {
            constant = as_constant_int(op1)->value();
            var_operand = 0;
        } else if (is_constant_int(op0)) {
            constant = as_constant_int(op0)->value();
            var_operand = 1;
        } else {
            return nullptr;
        }
    } else {
        // udiv/urem: the divisor (op1) must be the constant.
        if (!is_constant_int(op1)) return nullptr;
        constant = static_cast<int64_t>(
            as_unsigned(as_constant_int(op1)->value(), bw));
        var_operand = 0;
    }

    if (!is_power_of_two(constant)) return nullptr;
    const unsigned shift_amount = log2_of_power_of_two(constant);

    auto var = inst.operand(var_operand);
    if (!var || !var->type()) return nullptr;
    auto operand_ty = std::make_shared<ir::IntegerType>(
        var->type()->bit_width() > 0 ? var->type()->bit_width() : 32);

    std::shared_ptr<ir::Instruction> repl;
    if (op == ir::Opcode::Mul) {
        repl = std::make_shared<ir::Instruction>(ir::Opcode::Shl, inst.type(), inst.name());
        repl->add_operand(var);
        repl->add_operand(std::make_shared<ir::ConstantInt>(
            operand_ty, static_cast<int64_t>(shift_amount)));
    } else if (op == ir::Opcode::UDiv) {
        repl = std::make_shared<ir::Instruction>(ir::Opcode::LShr, inst.type(), inst.name());
        repl->add_operand(var);
        repl->add_operand(std::make_shared<ir::ConstantInt>(
            operand_ty, static_cast<int64_t>(shift_amount)));
    } else {  // URem
        repl = std::make_shared<ir::Instruction>(ir::Opcode::And, inst.type(), inst.name());
        repl->add_operand(var);
        repl->add_operand(std::make_shared<ir::ConstantInt>(
            operand_ty, wrap_to_width(constant - 1, bw)));
    }
    return repl;
}

// Fold a binop with two ConstantInt operands into the resulting ConstantInt,
// applying the LLVM type's wrapping / unsigned semantics and refusing UB
// (div-by-zero, INT_MIN/-1, out-of-range shifts). Returns nullptr if `inst`
// is not a foldable both-constant binop.
static std::shared_ptr<ir::ConstantInt> build_folded_constant(
    const ir::Instruction& inst) {
    if (!inst.is_binary_op()) return nullptr;
    if (inst.num_operands() < 2) return nullptr;

    auto op0 = inst.operand(0);
    auto op1 = inst.operand(1);
    if (!is_constant_int(op0) || !is_constant_int(op1)) return nullptr;

    auto ci0 = as_constant_int(op0);
    auto ci1 = as_constant_int(op1);

    const unsigned bw = ci0->bit_width() > 0 ? ci0->bit_width() : 32;
    const int64_t s0 = wrap_to_width(ci0->value(), bw);
    const int64_t s1 = wrap_to_width(ci1->value(), bw);
    const uint64_t u0 = as_unsigned(ci0->value(), bw);
    const uint64_t u1 = as_unsigned(ci1->value(), bw);
    int64_t result = 0;

    switch (inst.opcode()) {
        case ir::Opcode::Add:  result = static_cast<int64_t>(u0 + u1); break;
        case ir::Opcode::Sub:  result = static_cast<int64_t>(u0 - u1); break;
        case ir::Opcode::Mul:  result = static_cast<int64_t>(u0 * u1); break;
        case ir::Opcode::SDiv:
            if (s1 == 0 || (s0 == std::numeric_limits<int64_t>::min() && s1 == -1))
                return nullptr;
            result = s0 / s1;
            break;
        case ir::Opcode::UDiv:
            if (u1 == 0) return nullptr;
            result = static_cast<int64_t>(u0 / u1);
            break;
        case ir::Opcode::SRem:
            if (s1 == 0 || (s0 == std::numeric_limits<int64_t>::min() && s1 == -1))
                return nullptr;
            result = s0 % s1;
            break;
        case ir::Opcode::URem:
            if (u1 == 0) return nullptr;
            result = static_cast<int64_t>(u0 % u1);
            break;
        case ir::Opcode::And:  result = s0 & s1; break;
        case ir::Opcode::Or:   result = s0 | s1; break;
        case ir::Opcode::Xor:  result = s0 ^ s1; break;
        case ir::Opcode::Shl:
            if (u1 >= bw) return nullptr;
            result = static_cast<int64_t>(u0 << u1);
            break;
        case ir::Opcode::LShr:
            if (u1 >= bw) return nullptr;
            result = static_cast<int64_t>(u0 >> u1);
            break;
        case ir::Opcode::AShr:
            if (u1 >= bw) return nullptr;
            result = s0 >> u1;
            break;
        default:
            return nullptr;
    }
    result = wrap_to_width(result, bw);
    return std::make_shared<ir::ConstantInt>(
        std::make_shared<ir::IntegerType>(bw), result);
}

// Build the replacement value for an algebraic-identity simplification
// (x+0 → x, x*0 → 0, ...), or nullptr if no identity applies. A keep_operand
// rewrite returns the surviving operand; a constant rewrite returns a fresh
// ConstantInt.
static std::shared_ptr<ir::Value> build_identity_replacement(
    const ir::Instruction& inst) {
    IdentityRewrite rw = classify_identity(inst);
    if (!rw.applicable()) return nullptr;
    if (rw.keep_operand >= 0) {
        return inst.operand(static_cast<size_t>(rw.keep_operand));
    }
    const unsigned bw = (inst.type() && inst.type()->bit_width() > 0)
                            ? inst.type()->bit_width() : 32;
    return std::make_shared<ir::ConstantInt>(
        std::make_shared<ir::IntegerType>(bw), rw.constant);
}

// True iff the instruction at `idx` in `bb_name` is provably dead and safe to
// remove (no side effects, result unused anywhere in `fn`). Shared by the
// copy and in-place delete paths.
static bool is_dead_instruction(const ir::Function& fn, size_t idx,
                                const std::string& bb_name) {
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return false;
    auto inst = block->instruction(idx);
    if (!inst) return false;

    if (inst->is_terminator()) return false;
    if (inst->opcode() == ir::Opcode::Store ||
        inst->opcode() == ir::Opcode::Call ||
        inst->opcode() == ir::Opcode::Fence ||
        (inst->opcode() == ir::Opcode::Load && inst->is_volatile())) {
        return false;
    }

    if (!inst->has_name()) {
        // Unnamed instructions produce no referenceable value; be
        // conservative about allocas, which may be used later.
        if (inst->opcode() == ir::Opcode::Alloca) return false;
        return true;
    }

    // Named result: removable only if nothing else references it.
    return count_uses_excluding(fn, inst->name(), inst.get()) == 0;
}

// ── Opcode equivalence classes & operand pools ─────────────────────────────
//
// STOKE's OpcodeReplace move swaps an opcode for another from the same
// "equivalence class" — opcodes that take the same number and shape of
// operands and produce the same type (STOKE ASPLOS'13 §4.3). For clunk's
// integer-only STOKE-style moves we model one class: the integer binary
// ops. The replacement is NOT sound (Add→Sub changes semantics); the
// candidate must be SMT-verified.

static const std::vector<ir::Opcode>& int_binop_class() {
    static const std::vector<ir::Opcode> k = {
        ir::Opcode::Add,  ir::Opcode::Sub,  ir::Opcode::Mul,
        ir::Opcode::UDiv, ir::Opcode::SDiv, ir::Opcode::URem, ir::Opcode::SRem,
        ir::Opcode::And,  ir::Opcode::Or,   ir::Opcode::Xor,
        ir::Opcode::Shl,  ir::Opcode::LShr, ir::Opcode::AShr,
    };
    return k;
}

// Return the equivalence class for `op`, or an empty vector if `op` is
// not in any swappable class (terminators, memory ops, casts, …).
static const std::vector<ir::Opcode>& opcode_equivalence_class(ir::Opcode op) {
    switch (op) {
        case ir::Opcode::Add: case ir::Opcode::Sub: case ir::Opcode::Mul:
        case ir::Opcode::UDiv: case ir::Opcode::SDiv:
        case ir::Opcode::URem: case ir::Opcode::SRem:
        case ir::Opcode::And: case ir::Opcode::Or: case ir::Opcode::Xor:
        case ir::Opcode::Shl: case ir::Opcode::LShr: case ir::Opcode::AShr:
            return int_binop_class();
        default:
            break;
    }
    static const std::vector<ir::Opcode> kEmpty;
    return kEmpty;
}

// Build a replacement instruction for OpcodeReplace: same name, same
// operands, same type, same metadata, but a DIFFERENT opcode drawn from
// the same equivalence class. Returns nullptr if `inst` is not in any
// swappable class. The caller's RNG picks the new opcode.
static std::shared_ptr<ir::Instruction> build_opcode_replacement(
    const ir::Instruction& inst, std::mt19937& rng) {
    const auto& klass = opcode_equivalence_class(inst.opcode());
    if (klass.size() < 2) return nullptr;
    std::uniform_int_distribution<size_t> dist(0, klass.size() - 1);
    size_t pick = dist(rng);
    while (klass[pick] == inst.opcode()) {
        pick = dist(rng);
    }
    auto repl = std::make_shared<ir::Instruction>(klass[pick], inst.type(), inst.name());
    for (auto& op : inst.operands()) repl->add_operand(op);
    for (auto& [k, v] : inst.metadata()) repl->set_metadata(k, v);
    repl->binop_flags() = inst.binop_flags();
    if (inst.alignment()) repl->set_alignment(inst.alignment().value());
    repl->set_volatile(inst.is_volatile());
    return repl;
}

// Collect the pool of named SSA values visible at position `idx` in
// `block` whose type matches `target_ty`. Includes:
//   - Instructions earlier in the block (def-before-use, SSA-correct).
//   - Function arguments (visible throughout the function).
// Used by OperandReplace, InstructionInsert, InstructionReplace.
static void collect_operand_pool(const ir::Function& fn,
                                  const ir::BasicBlock& block,
                                  size_t idx,
                                  const ir::Type& target_ty,
                                  std::vector<std::shared_ptr<ir::Value>>& out) {
    for (size_t i = 0; i < idx && i < block.size(); ++i) {
        auto prev = block.instruction(i);
        if (!prev || !prev->has_name() || !prev->type()) continue;
        if (*prev->type() == target_ty) out.push_back(prev);
    }
    for (auto& arg : fn.arguments()) {
        if (arg.name.empty() || !arg.type) continue;
        if (*arg.type == target_ty) {
            // Arguments are not first-class Values in clunk's IR — wrap
            // them so the candidate instruction can reference them by name.
            out.push_back(std::make_shared<ir::Value>(arg.type, arg.name));
        }
    }
}

// Build a fresh random binary-int-op instruction over two operands drawn
// from `pool`. Returns nullptr if the pool has fewer than 2 candidates
// or the result type is unsuitable. `name` is the new instruction's SSA
// name. Used by InstructionInsert and InstructionReplace.
static std::shared_ptr<ir::Instruction> build_random_binop(
    const std::vector<std::shared_ptr<ir::Value>>& pool,
    const std::shared_ptr<ir::Type>& result_ty,
    const std::string& name,
    std::mt19937& rng) {
    if (pool.size() < 2 || !result_ty) return nullptr;
    std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);
    auto lhs = pool[dist(rng)];
    auto rhs = pool[dist(rng)];
    static const ir::Opcode ops[] = {
        ir::Opcode::Add, ir::Opcode::Sub, ir::Opcode::Mul,
        ir::Opcode::And, ir::Opcode::Or,  ir::Opcode::Xor,
    };
    std::uniform_int_distribution<int> op_dist(0, static_cast<int>(sizeof(ops) / sizeof(ops[0])) - 1);
    auto inst = std::make_shared<ir::Instruction>(ops[op_dist(rng)], result_ty, name);
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    return inst;
}

// ── Constructor ─────────────────────────────────────────────────────────────

StochasticSearch::StochasticSearch(const StochasticConfig& config,
                                    evaluator::EvaluationEngine* engine)
    : StochasticSearch(config, engine, nullptr) {}

StochasticSearch::StochasticSearch(const StochasticConfig& config,
                                    evaluator::EvaluationEngine* engine,
                                    pattern::PatternLibrary* lib)
    : config_(config), engine_(engine), pattern_lib_(lib) {
    if (config_.seed == 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(config_.seed);
    }
    stats_ = {};
    // Bump very small max_instruction_count defaults to 1024.
    // Values <= 64 are assumed to be unintentionally low.
    if (config_.max_instruction_count <= 64) {
        config_.max_instruction_count = 1024;
    }
}

// ── Reset for reuse ─────────────────────────────────────────────────────────

void StochasticSearch::reset(unsigned new_seed) {
    rng_.seed(new_seed);
    stats_ = Stats{};
    original_ = nullptr;
    original_analysis_cached_ = false;
    score_cache_.clear();
    next_stoke_name_ = 0;
}

// ── Lazy original-function analysis ──────────────────────────────────────────

const evaluator::FunctionAnalysis& StochasticSearch::original_analysis() const {
    if (!original_analysis_cached_ && engine_ && original_) {
        cached_original_analysis_ = engine_->analyse(*original_);
        original_analysis_cached_ = true;
    }
    return cached_original_analysis_;
}

// ── Structural hash ─────────────────────────────────────────────────────────

uint64_t StochasticSearch::structural_hash(const ir::Function& fn) {
    // FNV-1a 64-bit. We hash opcode + operand-name stream per block,
    // then XOR-combine blocks. Two functions with the same hash are
    // structurally identical for scoring purposes.
    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME  = 0x100000001b3ULL;

    uint64_t h = FNV_OFFSET;
    auto mix = [&h](uint64_t x) {
        h ^= x;
        h *= FNV_PRIME;
    };

    mix(fn.blocks().size());
    for (auto& block : fn.blocks()) {
        if (!block) {
            mix(0);
            continue;
        }
        // Mix block name length + size to distinguish blocks.
        mix(block->name().size());
        for (char c : block->name()) mix(static_cast<uint64_t>(c));
        mix(block->size());
        for (auto& inst : block->instructions()) {
            if (!inst) {
                mix(0);
                continue;
            }
            mix(static_cast<uint64_t>(inst->opcode()));
            mix(inst->num_operands());
            for (auto& op : inst->operands()) {
                if (!op) {
                    mix(0);
                    continue;
                }
                // Distinguish ConstantInt by value, named values by name.
                if (auto ci = dynamic_cast<ir::ConstantInt*>(op.get())) {
                    mix(0xC0FFEEULL);  // tag: "this is a constant"
                    mix(static_cast<uint64_t>(ci->value()));
                    mix(static_cast<uint64_t>(ci->bit_width()));
                } else {
                    mix(0xBA5EBA11ULL);  // tag: "this is a named value"
                    mix(op->name().size());
                    for (char c : op->name()) mix(static_cast<uint64_t>(c));
                }
            }
        }
    }
    return h;
}

// ── Score with cache ─────────────────────────────────────────────────────────

double StochasticSearch::score_with_cache(const ir::Function& fn, uint64_t* hash_out) {
    if (!engine_) return 0.0;
    // Key the cache by canonical_structural_hash so renaming-
    // equivalent functions collapse to one entry.
    uint64_t h = canonical_structural_hash(fn);
    if (hash_out) *hash_out = h;
    auto it = score_cache_.find(h);
    if (it != score_cache_.end()) {
        stats_.score_cache_hits++;
        stats_.canonical_cache_hits++;
        return it->second;
    }
    double s = engine_->score_candidate_with_cached_orig(original_analysis(), fn);
    score_cache_[h] = s;
    return s;
}

// ── Canonical structural hash ───────────────────────────────────────────────
//
// Walk the function in block order, then instruction order within each
// block. Assign each new SSA name a sequential canonical name (%v0, %v1,
// …) and each distinct constant (per bitwidth) a canonical name (%c0,
// %c1, …). Hash the (canonical-name, opcode, operand-canonical-names)
// stream. Two functions that differ only in SSA value names or in the
// order constants appear produce the SAME hash, dramatically improving
// the score-cache hit rate.

uint64_t StochasticSearch::canonical_structural_hash(const ir::Function& fn) {
    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME  = 0x100000001b3ULL;

    uint64_t h = FNV_OFFSET;
    auto mix = [&h](uint64_t x) {
        h ^= x;
        h *= FNV_PRIME;
    };

    // Constants and named values share the same id space but are tagged
    // differently so a constant and a named value with the same numeric
    // id do not collide.
    constexpr uint64_t kConstTag = 0x8000000000000000ULL;
    constexpr uint64_t kValueTag = 0x4000000000000000ULL;

    std::unordered_map<std::string, uint64_t> name_to_canon;
    // Per-bitwidth constant table: value -> canonical id.
    std::unordered_map<unsigned,
                       std::unordered_map<int64_t, uint64_t>> const_to_canon;
    uint64_t next_value_id = 0;
    uint64_t next_const_id = 0;

    mix(fn.blocks().size());
    mix(fn.argument_count());

    // Function arguments get canonical ids first (they're "defined" at
    // function entry, before any instruction).
    for (auto& arg : fn.arguments()) {
        if (!arg.name.empty()) {
            name_to_canon[arg.name] = next_value_id++;
        }
    }

    auto canon_for_value = [&](const std::shared_ptr<ir::Value>& v) -> uint64_t {
        if (!v) return 0xDEADBEEFULL;
        if (auto ci = dynamic_cast<ir::ConstantInt*>(v.get())) {
            unsigned bw = ci->bit_width() > 0 ? ci->bit_width() : 64;
            int64_t val = ci->value();
            auto& map = const_to_canon[bw];
            auto it = map.find(val);
            if (it != map.end()) return it->second | kConstTag;
            uint64_t id = next_const_id++;
            map[val] = id;
            return id | kConstTag;
        }
        if (v->has_name()) {
            auto it = name_to_canon.find(v->name());
            if (it != name_to_canon.end()) return it->second | kValueTag;
            // Undefined operand — shouldn't happen in well-formed SSA,
            // but be defensive so the hash doesn't crash.
            return 0xBADBADBADULL;
        }
        return 0xCAFEF00DULL;
    };

    for (auto& block : fn.blocks()) {
        if (!block) { mix(0); continue; }
        // The block NAME itself is not mixed — two functions with
        // identically-structured blocks but different block names should
        // canonicalise equal (the canonical form is name-agnostic).
        mix(block->size());
        for (auto& inst : block->instructions()) {
            if (!inst) { mix(0); continue; }
            mix(static_cast<uint64_t>(inst->opcode()));
            mix(inst->num_operands());
            // Mix binop flags so Add{nsw} and Add{} hash differently
            // (they have different semantics).
            mix(static_cast<uint64_t>(inst->binop_flags().nuw ? 1 : 0) |
                (static_cast<uint64_t>(inst->binop_flags().nsw ? 1 : 0) << 1) |
                (static_cast<uint64_t>(inst->binop_flags().exact ? 1 : 0) << 2));
            for (auto& op : inst->operands()) {
                mix(canon_for_value(op));
            }
            // Define this instruction's result name (if any).
            if (inst->has_name()) {
                name_to_canon[inst->name()] = next_value_id++;
            }
        }
    }
    return h;
}

// ── Test-vector pre-filter ─────────────────────────────────────────────────
//
// Run the original and candidate through up to `num_vectors` concrete input
// vectors via the shared evaluator::DifferentialScreen: edge cases first
// (0, -1, INT_MIN, INT_MAX, every power of two, ...), then random vectors,
// each canonical for its argument's width. Returns false iff the candidate
// is provably NOT a refinement of the original (a concrete counterexample).
// Anything the interpreter cannot decide (memory ops, floats, calls, loops,
// a void return, ...) is left to SMT — this is a strict pre-filter, never a
// soundness gate. The screen is deterministic (fixed seed), which
// test_stochastic_seed_reproducibility relies on.

bool StochasticSearch::passes_test_vectors(const ir::Function& original,
                                            const ir::Function& candidate,
                                            size_t num_vectors) {
    if (num_vectors == 0) return true;
    evaluator::ScreenConfig cfg;
    cfg.max_vectors = num_vectors;
    const auto r = evaluator::DifferentialScreen(cfg).screen(original, candidate);
    if (r.verdict == evaluator::ScreenVerdict::Rejected) {
        stats_.candidates_rejected_by_test_vectors++;
        return false;
    }
    return true;
}

// ── Time budget check ──────────────────────────────────────────────────────

bool StochasticSearch::time_budget_exceeded() const {
    if (!config_.time_budget_seconds.has_value()) return false;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = now - start_time_;
    return elapsed.count() >= *config_.time_budget_seconds;
}

// ── Mutation description ─────────────────────────────────────────────────────

std::string StochasticSearch::describe_mutation(const Mutation& m) {
    const char* kind_name = "unknown";
    switch (m.kind) {
        case MutationKind::StrengthReduce:       kind_name = "strength_reduce"; break;
        case MutationKind::FoldConstant:         kind_name = "fold_constant"; break;
        case MutationKind::CombineInstructions:  kind_name = "combine_instructions"; break;
        case MutationKind::DeleteInstruction:    kind_name = "delete_instruction"; break;
        case MutationKind::SwapInstructions:     kind_name = "swap_instructions"; break;
        case MutationKind::ReplaceInstruction:   kind_name = "replace_instruction"; break;
        case MutationKind::SubstituteVariable:   kind_name = "substitute_variable"; break;
        case MutationKind::PatternGuided:        kind_name = "pattern_guided"; break;
        case MutationKind::SimplifyIdentity:     kind_name = "simplify_identity"; break;
        case MutationKind::EliminateCommonSubexpr: kind_name = "cse"; break;
        // STOKE-style moves.
        case MutationKind::OpcodeReplace:        kind_name = "stoke:opcode_replace"; break;
        case MutationKind::OperandReplace:       kind_name = "stoke:operand_replace"; break;
        case MutationKind::OperandSwap:          kind_name = "stoke:operand_swap"; break;
        case MutationKind::InstructionInsert:    kind_name = "stoke:instruction_insert"; break;
        case MutationKind::InstructionReplace:   kind_name = "stoke:instruction_replace"; break;
        // MutationKind::UnrollLoop falls through to default "unknown" to
        // match the original random_mutation switch behaviour.
        default: break;
    }
    return std::string(kind_name) + " @" + m.block_name + ":" + std::to_string(m.instruction_index);
}

// ── Main search loop ───────────────────────────────────────────────────────

std::vector<Candidate> StochasticSearch::search(const ir::Function& original) {
    std::vector<Candidate> candidates;

    // Cache the original function pointer for lazy analysis.
    // The original is immutable for the lifetime of this search() call.
    original_ = &original;
    original_analysis_cached_ = false;
    score_cache_.clear();

    // Start the wall-clock timer.
    start_time_ = std::chrono::steady_clock::now();

    // Clone the original as the current best
    auto current_fn = ir::deep_copy_function(original);
    double baseline_score = 0.0;
    double current_score = 0.0;

    if (engine_) {
        baseline_score = engine_->score_candidate_with_cached_orig(original_analysis(), *current_fn);
        current_score = baseline_score;
    }

    double temperature = config_.temperature;
    stats_.best_score = baseline_score;

    // Set of structural hashes of candidates already accepted.
    std::unordered_set<uint64_t> seen_hashes;
    seen_hashes.reserve(config_.max_candidates);

    // Stagnation tracking.
    double last_best_score = stats_.best_score;
    size_t stagnation_counter = 0;

    // Instruction count of current_fn, maintained across iterations so the
    // hot loop doesn't re-count (O(n)) twice per iteration. Recomputed only
    // when a mutation is accepted.
    size_t cur_inst_count = count_instructions(*current_fn);

    // Applicable-mutation sites for current_fn. Rejected mutations leave
    // the function untouched, so the O(n) scan only re-runs on accept.
    // The two-phase flag is captured in `effective_allow_unsound`
    // below; collect_mutation_sites uses it to include / exclude the
    // STOKE-style unsound sites.
    bool effective_allow_unsound = config_.allow_unsound_mutations ||
                                    config_.two_phase_mode;
    std::vector<MutationSite> sites = collect_mutation_sites(*current_fn,
                                                              effective_allow_unsound);

    // Phase-1 iteration count. In two-phase mode the first
    // `phase1_iters` iterations use unsound mutations (synthesis); the
    // remainder use sound-only (optimization). Phase 2 starts from the
    // best phase-1 candidate, which is whatever `current_fn` happens to
    // be at the phase boundary.
    size_t phase1_iters = 0;
    if (config_.two_phase_mode) {
        double split = config_.two_phase_split;
        if (split < 0.0) split = 0.0;
        if (split > 1.0) split = 1.0;
        phase1_iters = static_cast<size_t>(
            static_cast<double>(config_.max_iterations) * split);
    }

    // Is current_fn related to the original solely via semantics-
    // preserving rewrites? (Adopting a pattern-library rewrite clears
    // this; see Candidate::sound.)
    bool current_sound = true;

    // `current_fn_for_filter` is a raw pointer to whichever function the
    // current iteration mutated (current_fn for the in-place path,
    // candidate_fn for the deep-copy path). It's set each iteration
    // BEFORE consider_candidate is called, so passes_test_vectors sees
    // the just-mutated function. (Captured by reference into the lambda
    // below.)
    const ir::Function* current_fn_for_filter = current_fn.get();

    // Insert an improving, novel candidate into the results list. When the
    // list is full, the current worst entry is replaced so the final list
    // keeps the BEST max_candidates seen, not the first. `materialise`
    // yields the shared_ptr to STORE and is invoked only once we've decided
    // to keep the candidate — so a rejected or duplicate mutation never pays
    // for the snapshot copy. (In-place mutation shares the live current_fn,
    // so its materialise deep-copies; the deep-copy path already holds an
    // independent function.)
    auto consider_candidate = [&](double candidate_score, uint64_t cand_hash,
                                  const Mutation& mut, bool candidate_sound,
                                  size_t iter, auto&& materialise) {
        if (candidate_score <= baseline_score) return;
        if (seen_hashes.count(cand_hash) != 0) return;

        // Test-vector pre-filter. Cheap µs-cost interpreter pass
        // that rejects candidates disagreeing with the original on any
        // of `test_vector_count` random inputs. Skipped when
        // test_vector_count == 0 (legacy behaviour) or when the
        // interpreter cannot evaluate either function (memory ops,
        // floats, …). A rejected candidate is NOT inserted into the
        // list — Stats::candidates_rejected_by_test_vectors is
        // incremented inside passes_test_vectors.
        if (config_.test_vector_count > 0) {
            if (!passes_test_vectors(original, *current_fn_for_filter,
                                      config_.test_vector_count)) {
                return;
            }
        }

        Candidate cand;
        cand.score = candidate_score;
        cand.iteration_found = iter;
        cand.description = describe_mutation(mut);
        cand.structural_hash = cand_hash;
        cand.sound = candidate_sound;

        if (candidates.size() < config_.max_candidates) {
            cand.function = materialise();
            seen_hashes.insert(cand_hash);
            candidates.push_back(std::move(cand));
        } else if (!candidates.empty()) {
            auto worst = std::min_element(
                candidates.begin(), candidates.end(),
                [](const Candidate& a, const Candidate& b) {
                    return a.score < b.score;
                });
            if (candidate_score > worst->score) {
                cand.function = materialise();
                seen_hashes.insert(cand_hash);
                *worst = std::move(cand);
            }
        }

        if (candidate_score > stats_.best_score) {
            stats_.best_score = candidate_score;
        }
        stats_.candidates_found = candidates.size();
    };

    // `current_fn_for_filter` was declared above (before the lambda) so
    // it can be captured by reference. The loop body updates it each
    // iteration BEFORE consider_candidate is called, so passes_test_vectors
    // sees the just-mutated function.

    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        stats_.iterations_run = iter + 1;

        // Phase-2 transition: switch from synthesis to sound-only
        // mutations. The phase-1 best is the phase-2 starting point;
        // current_sound may already be false from phase-1 unsound
        // mutations, so phase-2 candidates MUST still be SMT-verified
        // before adoption.
        if (config_.two_phase_mode && iter == phase1_iters &&
            iter > 0) {
            effective_allow_unsound = false;
            sites = collect_mutation_sites(*current_fn, effective_allow_unsound);
        }

        // Stop if temperature is too low
        if (temperature < config_.min_temperature) {
            stats_.final_temperature = temperature;
            break;
        }

        // Time-budget check. Check every 32 iterations to
        // keep the syscall overhead negligible.
        if ((iter & 31) == 0 && time_budget_exceeded()) {
            stats_.final_temperature = temperature;
            break;
        }

        // Target-score early stop.
        if (config_.target_score.has_value() &&
            stats_.best_score >= *config_.target_score) {
            stats_.final_temperature = temperature;
            break;
        }

        // Skip if the function is too large. Rather than rolling back to
        // the original (which throws away accumulated optimisations),
        // we skip this iteration — the next iteration might pick a
        // DeleteInstruction that brings us back under the cap.
        if (current_fn && cur_inst_count > config_.max_instruction_count) {
            // Force a delete-dead-code mutation to try to bring it back under.
            Mutation forced;
            forced.kind = MutationKind::DeleteInstruction;
            forced.instruction_index = 0;
            if (!current_fn->blocks().empty()) {
                auto& first_block = current_fn->blocks().front();
                if (first_block && !first_block->empty()) {
                    forced.block_name = first_block->name();
                    auto candidate_fn = apply_mutation_impl(*current_fn, forced,
                                                            cur_inst_count);
                    if (candidate_fn) {
                        double candidate_score = engine_
                            ? score_with_cache(*candidate_fn) : 0.0;
                        if (accept(current_score, candidate_score, temperature)) {
                            current_fn = candidate_fn;
                            current_score = candidate_score;
                            cur_inst_count = count_instructions(*current_fn);
                            sites = collect_mutation_sites(*current_fn,
                                                            effective_allow_unsound);
                            stats_.mutations_accepted++;
                        }
                    }
                }
            }
            temperature *= config_.temperature_decay;
            stats_.final_temperature = temperature;
            continue;
        }

        // Pick a mutation from the cached applicable-site list (with a
        // small exploration slice).
        auto mut = pick_mutation(*current_fn, sites, effective_allow_unsound);
        stats_.mutations_tried++;
        // Track how often STOKE-style mutations fire for stats.
        if (!is_sound_kind(mut.kind)) {
            stats_.stoke_moves_tried++;
        }

        const bool candidate_sound = current_sound && is_sound_kind(mut.kind);

        if (config_.use_in_place_mutation && is_in_place_kind(mut.kind)) {
            // ── In-place path ──────────────────────────────────────────
            // Mutate current_fn under a transactional scope. On rejection
            // the scope's destructor restores the function exactly, so the
            // (usual) rejected mutation costs no deep copy. A copy is paid
            // only to preserve an improving, novel candidate.
            MutationScope scope(current_fn);
            if (!apply_mutation_in_place(scope, mut, cur_inst_count)) {
                // Not applicable — nothing recorded, nothing to undo.
                temperature *= config_.temperature_decay;
                continue;
            }

            // Early-terminate scoring: short-circuit accept for
            // improvements (skip the Metropolis RNG draw entirely).
            // Stats::score_evaluations_short_circuited counts
            // improvements that skipped the RNG.
            current_fn_for_filter = current_fn.get();
            double candidate_score = 0.0;
            uint64_t cand_hash = 0;
            if (engine_) {
                candidate_score = score_with_cache(*current_fn, &cand_hash);
            }
            // If the candidate improves on current_score, accept()
            // returns true without drawing the RNG.
            const bool short_circuit = (candidate_score >= current_score);
            const bool accepted =
                accept(current_score, candidate_score, temperature);
            if (short_circuit && accepted) {
                stats_.score_evaluations_short_circuited++;
            }

            // Snapshot an improving candidate BEFORE a possible rollback.
            if (candidate_score > baseline_score) {
                if (cand_hash == 0) cand_hash = canonical_structural_hash(*current_fn);
                consider_candidate(candidate_score, cand_hash, mut,
                                   candidate_sound, iter,
                                   [&] { return ir::deep_copy_function(*current_fn); });
            }

            if (accepted) {
                scope.commit();  // keep the in-place edit
                current_score = candidate_score;
                current_sound = candidate_sound;
                cur_inst_count = count_instructions(*current_fn);
                sites = collect_mutation_sites(*current_fn, effective_allow_unsound);
                stats_.mutations_accepted++;
            }
            // scope destructs here — undo iff not committed.
        } else {
            // ── Deep-copy path (PatternGuided, or in-place disabled) ─────
            auto candidate_fn = apply_mutation_impl(*current_fn, mut, cur_inst_count);
            if (!candidate_fn) {
                // Mutation was not applicable, continue
                temperature *= config_.temperature_decay;
                continue;
            }

            // Evaluate the candidate. score_with_cache also hands back the
            // structural hash so the candidate-insertion path below doesn't
            // hash the function a second time.
            current_fn_for_filter = candidate_fn.get();
            double candidate_score = 0.0;
            uint64_t cand_hash = 0;
            if (engine_) {
                candidate_score = score_with_cache(*candidate_fn, &cand_hash);
            }

            // Same short-circuit accounting as the in-place path.
            const bool short_circuit = (candidate_score >= current_score);
            // Accept or reject based on simulated annealing
            if (accept(current_score, candidate_score, temperature)) {
                if (short_circuit) stats_.score_evaluations_short_circuited++;
                current_fn = candidate_fn;
                current_score = candidate_score;
                current_sound = candidate_sound;
                cur_inst_count = count_instructions(*current_fn);
                sites = collect_mutation_sites(*current_fn, effective_allow_unsound);
                stats_.mutations_accepted++;
            }

            if (candidate_score > baseline_score) {
                if (cand_hash == 0) cand_hash = canonical_structural_hash(*candidate_fn);
                // When in-place mutation is enabled, current_fn may later be
                // edited in place; if this candidate was just adopted as
                // current_fn, storing the same pointer would let that edit
                // corrupt the stored result. Snapshot in that case; when
                // in-place is off, current_fn is never mutated in place, so
                // sharing the (disposable) candidate_fn is safe and cheap.
                consider_candidate(candidate_score, cand_hash, mut,
                                   candidate_sound, iter, [&] {
                                       return config_.use_in_place_mutation
                                                  ? ir::deep_copy_function(candidate_fn)
                                                  : candidate_fn;
                                   });
            }
        }

        // Stagnation detection.
        if (stats_.best_score > last_best_score) {
            last_best_score = stats_.best_score;
            stagnation_counter = 0;
        } else {
            ++stagnation_counter;
        }
        if (config_.stagnation_limit > 0 &&
            stagnation_counter >= config_.stagnation_limit) {
            // Re-seed from a fresh clone of the original, but keep the
            // best_score / candidates list. This is the SA equivalent of
            // a "restart" — escape the current basin.
            stats_.stagnation_restarts++;
            current_fn = ir::deep_copy_function(original);
            cur_inst_count = count_instructions(*current_fn);
            sites = collect_mutation_sites(*current_fn, effective_allow_unsound);
            current_sound = true;
            if (engine_) {
                current_score = engine_->score_candidate_with_cached_orig(
                    original_analysis(), *current_fn);
            }
            temperature = config_.temperature;  // re-heat
            stagnation_counter = 0;
        }

        // Decay temperature
        temperature *= config_.temperature_decay;
        stats_.final_temperature = temperature;
    }

    // Record total elapsed time.
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time_;
    stats_.elapsed_seconds = elapsed.count();

    // Sort candidates by score (descending)
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    return candidates;
}

// ── Mutation-site discovery ─────────────────────────────────────────────────
//
// Blind random (block, index, kind) picks almost never land on an
// applicable rewrite in a real (clang-sized) function, so the search
// burned its whole budget on inapplicable mutations. Scanning for
// concrete sites first makes every iteration meaningful; the scan is
// O(n) and the hot loop only re-runs it when the current function
// actually changes.

std::vector<MutationSite> StochasticSearch::collect_mutation_sites(
    const ir::Function& fn) const {
    return collect_mutation_sites(fn, config_.allow_unsound_mutations);
}

std::vector<MutationSite> StochasticSearch::collect_mutation_sites(
    const ir::Function& fn, bool allow_unsound) const {
    std::vector<MutationSite> sites;

    // Function-wide use counts (for dead-code and combine safety).
    std::unordered_map<std::string, size_t> use_count;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            for (auto& op : inst->operands()) {
                if (op && op->has_name()) use_count[op->name()]++;
            }
        }
    }

    for (auto& block : fn.blocks()) {
        if (!block) continue;
        const std::string& bb = block->name();

        // Per-block CSE table: structural key → index of first occurrence.
        std::unordered_map<std::string, size_t> cse_table;
        // Per-block add/sub-with-constant defs: result name → index.
        std::unordered_map<std::string, size_t> addsub_defs;

        const size_t n = block->size();
        for (size_t i = 0; i < n; ++i) {
            auto inst = block->instruction(i);
            if (!inst) continue;

            // ── Dead code ────────────────────────────────────────────
            if (!inst->is_terminator() &&
                inst->opcode() != ir::Opcode::Store &&
                inst->opcode() != ir::Opcode::Call &&
                inst->opcode() != ir::Opcode::Fence &&
                !(inst->opcode() == ir::Opcode::Load && inst->is_volatile()) &&
                !(inst->opcode() == ir::Opcode::Alloca && !inst->has_name())) {
                if (!inst->has_name() || use_count[inst->name()] == 0) {
                    sites.push_back({MutationKind::DeleteInstruction, bb, i, 0});
                }
            }

            const bool is_int_binop =
                inst->is_binary_op() && inst->type() &&
                inst->type()->bit_width() > 0;

            auto op0 = inst->num_operands() > 0 ? inst->operand(0) : nullptr;
            auto op1 = inst->num_operands() > 1 ? inst->operand(1) : nullptr;
            auto c0 = op0 ? dynamic_cast<ir::ConstantInt*>(op0.get()) : nullptr;
            auto c1 = op1 ? dynamic_cast<ir::ConstantInt*>(op1.get()) : nullptr;

            // ── Constant folding ─────────────────────────────────────
            if (is_int_binop && c0 && c1) {
                sites.push_back({MutationKind::FoldConstant, bb, i, 0});
            }

            // ── Strength reduction (mul/udiv/urem by power of two) ───
            if (is_int_binop) {
                const unsigned bw = inst->type()->bit_width();
                if (inst->opcode() == ir::Opcode::Mul &&
                    ((c1 && is_power_of_two(c1->value())) ||
                     (c0 && is_power_of_two(c0->value())))) {
                    sites.push_back({MutationKind::StrengthReduce, bb, i, 0});
                } else if ((inst->opcode() == ir::Opcode::UDiv ||
                            inst->opcode() == ir::Opcode::URem) &&
                           c1 &&
                           is_power_of_two(static_cast<int64_t>(
                               as_unsigned(c1->value(), bw)))) {
                    sites.push_back({MutationKind::StrengthReduce, bb, i, 0});
                }
            }

            // ── FP strength reduction (fdiv by ±2^k → fmul by ±2^-k) ─
            // Bit-exact for power-of-two divisors, so it belongs to the
            // sound mutation family despite being floating point. The
            // 18-cycle divide becomes a 5-cycle multiply.
            if (inst->opcode() == ir::Opcode::FDiv && op1) {
                if (auto cfp = dynamic_cast<ir::ConstantFP*>(op1.get())) {
                    double recip_unused = 0.0;
                    if (inst->type() &&
                        (inst->type()->is_float() || inst->type()->is_double()) &&
                        fp_reciprocal_is_exact(cfp->value(),
                                               inst->type()->is_float(),
                                               recip_unused)) {
                        sites.push_back({MutationKind::StrengthReduce, bb, i, 0});
                    }
                }
            }

            // ── Algebraic identities ─────────────────────────────────
            if (classify_identity(*inst).applicable()) {
                sites.push_back({MutationKind::SimplifyIdentity, bb, i, 0});
            }

            // ── Same-block CSE ───────────────────────────────────────
            {
                std::string key = cse_key(*inst);
                if (!key.empty()) {
                    auto [it, inserted] = cse_table.emplace(std::move(key), i);
                    if (!inserted) {
                        sites.push_back({MutationKind::EliminateCommonSubexpr,
                                         bb, i, it->second});
                    }
                }
            }

            // ── Constant-chain combining (a: x±C1, b: a±C2 → x+C) ────
            // For Sub, only the x-minus-constant form is combinable
            // (C-minus-x flips the sign of x; rewriting it as an Add
            // would be wrong).
            if (is_int_binop &&
                (inst->opcode() == ir::Opcode::Add ||
                 inst->opcode() == ir::Opcode::Sub)) {
                const bool const_ok =
                    (inst->opcode() == ir::Opcode::Add)
                        ? (c0 != nullptr) != (c1 != nullptr)  // exactly one const
                        : (c1 && !c0);                        // sub: x - C only
                if (const_ok) {
                    std::shared_ptr<ir::Value> var = c1 ? op0 : op1;
                    // This inst can be the "b" of an earlier "a".
                    if (var && var->has_name()) {
                        auto def = addsub_defs.find(var->name());
                        if (def != addsub_defs.end() &&
                            use_count[var->name()] == 1) {
                            sites.push_back({MutationKind::CombineInstructions,
                                             bb, def->second, i});
                        }
                    }
                    // And it can be the "a" of a later combinable use.
                    if (inst->has_name()) {
                        addsub_defs[inst->name()] = i;
                    }
                }
            }

            // ── Legal adjacent swaps (exploration) ───────────────────
            if (i + 1 < n) {
                auto next = block->instruction(i + 1);
                if (next && !inst->is_terminator() && !next->is_terminator() &&
                    swap_memory_safe(*inst, *next)) {
                    bool ssa_dep = false;
                    if (inst->has_name()) {
                        for (auto& op : next->operands()) {
                            if (op && op->has_name() &&
                                op->name() == inst->name()) {
                                ssa_dep = true;
                                break;
                            }
                        }
                    }
                    if (!ssa_dep) {
                        sites.push_back({MutationKind::SwapInstructions, bb, i, 0});
                    }
                }
            }

            // ── STOKE-style unsound sites ───────────────────────
            // Only enumerated when allow_unsound is true (config flag
            // or two-phase synthesis). All five kinds produce
            // Candidate::sound == false; the Pipeline must SMT-verify
            // before adoption. Sites here mirror the per-kind rules:
            //   OpcodeReplace: every non-terminator instruction whose
            //     opcode is in a swappable equivalence class.
            //   OperandReplace: every non-terminator instruction with
            //     at least one operand. aux_index carries the operand
            //     slot to replace.
            //   OperandSwap: every non-terminator binary instruction.
            //   InstructionReplace: every non-terminator instruction.
            //   InstructionInsert: every position [0..size] in every
            //     block (a position BETWEEN instructions, or at the
            //     start/end). instruction_index encodes the position.
            if (allow_unsound && !inst->is_terminator()) {
                if (!opcode_equivalence_class(inst->opcode()).empty()) {
                    sites.push_back({MutationKind::OpcodeReplace, bb, i, 0});
                    sites.push_back({MutationKind::InstructionReplace, bb, i, 0});
                    if (inst->is_binary_op()) {
                        sites.push_back({MutationKind::OperandSwap, bb, i, 0});
                    }
                }
                for (size_t op_i = 0; op_i < inst->num_operands(); ++op_i) {
                    sites.push_back({MutationKind::OperandReplace, bb, i, op_i});
                }
            }
        }

        // InstructionInsert sites: one per position [0..n] in the block.
        // (Position 0 = before the first instruction; position n = after
        // the last. Both are valid insertion points.)
        if (allow_unsound) {
            for (size_t pos = 0; pos <= n; ++pos) {
                sites.push_back({MutationKind::InstructionInsert, bb, pos, 0});
            }
        }
    }

    return sites;
}

// ── Random mutation generation ─────────────────────────────────────────────

Mutation StochasticSearch::blind_random_mutation(const ir::Function& fn,
                                                   bool allow_unsound) {
    Mutation mut;

    if (fn.blocks().empty()) {
        mut.kind = MutationKind::FoldConstant;
        mut.instruction_index = 0;
        mut.block_name = "";
        return mut;
    }

    // Pick a random basic block
    std::uniform_int_distribution<size_t> block_dist(0, fn.blocks().size() - 1);
    size_t block_idx = block_dist(rng_);
    auto& block = fn.blocks()[block_idx];
    mut.block_name = block ? block->name() : "";

    // Pick a random instruction index
    if (!block || block->empty()) {
        mut.kind = MutationKind::FoldConstant;
        mut.instruction_index = 0;
        return mut;
    }

    std::uniform_int_distribution<size_t> inst_dist(0, block->size() - 1);
    mut.instruction_index = inst_dist(rng_);

    // The kind pool depends on whether unsound mutations are allowed.
    // When allowed (two-phase synthesis), the STOKE-style kinds are
    // mixed in. We use a single static table for each pool to avoid
    // per-call allocation.
    static const MutationKind kSoundKinds[] = {
        MutationKind::StrengthReduce,
        MutationKind::FoldConstant,
        MutationKind::CombineInstructions,
        MutationKind::DeleteInstruction,
        MutationKind::SwapInstructions,
        MutationKind::ReplaceInstruction,
        MutationKind::SubstituteVariable,
        MutationKind::SimplifyIdentity,
    };
    static const MutationKind kUnsoundKinds[] = {
        MutationKind::OpcodeReplace,
        MutationKind::OperandReplace,
        MutationKind::OperandSwap,
        MutationKind::InstructionInsert,
        MutationKind::InstructionReplace,
    };

    // Mix: when unsound is on, ~50% sound / ~50% unsound. This matches
    // STOKE's roughly-even move-class weighting (paper Figure 10).
    if (allow_unsound) {
        std::uniform_int_distribution<int> pool_dist(0, 1);
        if (pool_dist(rng_) == 0) {
            std::uniform_int_distribution<int> kind_dist(
                0, static_cast<int>(sizeof(kSoundKinds) / sizeof(kSoundKinds[0])) - 1);
            mut.kind = kSoundKinds[kind_dist(rng_)];
        } else {
            std::uniform_int_distribution<int> kind_dist(
                0, static_cast<int>(sizeof(kUnsoundKinds) / sizeof(kUnsoundKinds[0])) - 1);
            mut.kind = kUnsoundKinds[kind_dist(rng_)];
        }
    } else {
        std::uniform_int_distribution<int> kind_dist(
            0, static_cast<int>(sizeof(kSoundKinds) / sizeof(kSoundKinds[0])) - 1);
        mut.kind = kSoundKinds[kind_dist(rng_)];
    }

    return mut;
}

Mutation StochasticSearch::pick_mutation(const ir::Function& fn,
                                          const std::vector<MutationSite>& sites,
                                          bool allow_unsound) {
    // Small exploration slice: pattern-guided rewrites (when a library is
    // wired) and blind randomness keep the search from being purely
    // greedy over the discovered sites.
    std::uniform_real_distribution<double> roll(0.0, 1.0);
    if (pattern_lib_ && config_.use_pattern_library && pattern_lib_->size() > 0 &&
        roll(rng_) < 0.10) {
        Mutation mut;
        mut.kind = MutationKind::PatternGuided;
        mut.instruction_index = 0;
        if (!fn.blocks().empty() && fn.blocks().front()) {
            mut.block_name = fn.blocks().front()->name();
        }
        return mut;
    }

    if (sites.empty() || roll(rng_) < 0.05) {
        return blind_random_mutation(fn, allow_unsound);
    }

    std::uniform_int_distribution<size_t> site_dist(0, sites.size() - 1);
    // When unsound is OFF, the site list itself is already filtered (no
    // unsound sites were enumerated), so any site we pick is sound. When
    // ON, the list contains a mix and we sample uniformly. (We don't
    // re-filter here — collect_mutation_sites already did the gating.)
    const MutationSite& s = sites[site_dist(rng_)];
    Mutation mut;
    mut.kind = s.kind;
    mut.block_name = s.block_name;
    mut.instruction_index = s.instruction_index;
    mut.aux_index = s.aux_index;
    return mut;
}

Mutation StochasticSearch::random_mutation(const ir::Function& fn) {
    return pick_mutation(fn,
                          collect_mutation_sites(fn, config_.allow_unsound_mutations),
                          config_.allow_unsound_mutations);
}

// ── Apply mutation ─────────────────────────────────────────────────────────

bool StochasticSearch::mutation_allowed_near_cap(MutationKind kind,
                                                 size_t inst_count) const {
    // If the function is already very large (within 90% of the configured
    // instruction cap), only allow destructive / size-neutral mutations —
    // not constructive ones (swap, replace, unroll, pattern) which would
    // keep churn high on every iteration.
    const size_t cap = config_.max_instruction_count;
    const bool near_cap = (cap > 10) && (inst_count * 10 >= cap * 9);
    if (!near_cap) return true;
    switch (kind) {
        case MutationKind::DeleteInstruction:
        case MutationKind::FoldConstant:
        case MutationKind::SimplifyIdentity:
        case MutationKind::EliminateCommonSubexpr:
        case MutationKind::StrengthReduce:
        case MutationKind::CombineInstructions:
            return true;
        default:
            return false;
    }
}

std::shared_ptr<ir::Function> StochasticSearch::apply_mutation(
    const ir::Function& fn, const Mutation& mut) {
    return apply_mutation_impl(fn, mut, count_instructions(fn));
}

// ── In-place mutation dispatch ──────────────────────────────────────────────
//
// Transactional counterpart to apply_mutation_impl: applies the mutation
// directly to the live function held by `scope`, recording undo actions so
// the SA loop can reject (destructor undoes) without the deep copy that the
// legacy path paid on every applicable-but-rejected mutation.

bool StochasticSearch::apply_mutation_in_place(MutationScope& scope,
                                               const Mutation& mut,
                                               size_t inst_count) {
    if (!mutation_allowed_near_cap(mut.kind, inst_count)) return false;

    switch (mut.kind) {
        case MutationKind::StrengthReduce:
            return strength_reduce_in_place(scope, mut.instruction_index, mut.block_name);
        case MutationKind::FoldConstant:
            return fold_constant_in_place(scope, mut.instruction_index, mut.block_name);
        case MutationKind::CombineInstructions: {
            size_t idx_b = (mut.aux_index > mut.instruction_index)
                               ? mut.aux_index
                               : mut.instruction_index + 1;
            return combine_instructions_in_place(scope, mut.instruction_index,
                                                 idx_b, mut.block_name);
        }
        case MutationKind::DeleteInstruction:
            return delete_dead_code_in_place(scope, mut.instruction_index, mut.block_name);
        case MutationKind::SimplifyIdentity:
            return simplify_identity_in_place(scope, mut.instruction_index, mut.block_name);
        case MutationKind::EliminateCommonSubexpr:
            return eliminate_cse_in_place(scope, mut.instruction_index,
                                          mut.aux_index, mut.block_name);
        case MutationKind::SwapInstructions:
            return swap_instructions_in_place(scope, mut.instruction_index, mut.block_name);
        case MutationKind::ReplaceInstruction:
        case MutationKind::SubstituteVariable:
        case MutationKind::UnrollLoop:
            // Prototype fallbacks — mirror apply_mutation_impl, which routes
            // these to a dead-code delete.
            return delete_dead_code_in_place(scope, mut.instruction_index, mut.block_name);
        // ── STOKE-style unsound mutations ────────────────────────
        case MutationKind::OpcodeReplace:
            return opcode_replace_in_place(scope, mut.instruction_index, mut.block_name);
        case MutationKind::OperandReplace:
            return operand_replace_in_place(scope, mut.instruction_index,
                                             mut.aux_index, mut.block_name);
        case MutationKind::OperandSwap:
            return operand_swap_in_place(scope, mut.instruction_index, mut.block_name);
        case MutationKind::InstructionInsert:
            return instruction_insert_in_place(scope, mut.instruction_index, mut.block_name);
        case MutationKind::InstructionReplace:
            return instruction_replace_in_place(scope, mut.instruction_index, mut.block_name);
        default:
            // PatternGuided (and anything unhandled) cannot be undone
            // transactionally; the caller uses the deep-copy path for it.
            return false;
    }
}

bool StochasticSearch::strength_reduce_in_place(
    MutationScope& scope, size_t idx, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return false;
    auto inst = block->instruction(idx);
    if (!inst) return false;

    auto repl = build_strength_reduced(*inst);
    if (!repl) return false;

    // Uses resolve by name, and the replacement carries the same name.
    block->replace_instruction(idx, repl);
    scope.record_replace(bb_name, idx, inst);
    return true;
}

bool StochasticSearch::fold_constant_in_place(
    MutationScope& scope, size_t idx, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return false;
    auto inst = block->instruction(idx);
    if (!inst) return false;

    auto const_result = build_folded_constant(*inst);
    if (!const_result) return false;

    // Rewrite every use of the binop's result to the folded constant, then
    // remove the now-dead binop. Both edits are recorded for undo; the
    // remove is recorded last so undo re-inserts before restoring the
    // rewritten users (reverse order keeps positions valid).
    if (inst->has_name()) {
        replace_all_uses_scoped(scope, inst->name(), const_result);
    }
    block->remove_instruction(idx);
    scope.record_remove(bb_name, idx, inst);
    return true;
}

bool StochasticSearch::combine_instructions_in_place(
    MutationScope& scope, size_t idx_a, size_t idx_b, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    auto combined_inst = build_combined_instruction(fn, idx_a, idx_b, bb_name);
    if (!combined_inst) return false;

    auto block = fn.block(bb_name);
    if (!block || idx_b >= block->size()) return false;

    auto old_b = block->instruction(idx_b);
    block->replace_instruction(idx_b, combined_inst);
    scope.record_replace(bb_name, idx_b, old_b);

    // Remove inst_a (dead — verified by build_combined_instruction).
    auto old_a = block->instruction(idx_a);
    block->remove_instruction(idx_a);
    scope.record_remove(bb_name, idx_a, old_a);
    return true;
}

bool StochasticSearch::delete_dead_code_in_place(
    MutationScope& scope, size_t idx, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    if (!is_dead_instruction(fn, idx, bb_name)) return false;

    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return false;
    auto removed = block->instruction(idx);
    block->remove_instruction(idx);
    scope.record_remove(bb_name, idx, removed);
    return true;
}

bool StochasticSearch::simplify_identity_in_place(
    MutationScope& scope, size_t idx, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return false;
    auto inst = block->instruction(idx);
    if (!inst) return false;

    auto replacement = build_identity_replacement(*inst);
    if (!replacement) return false;

    if (inst->has_name()) {
        replace_all_uses_scoped(scope, inst->name(), replacement);
    }
    block->remove_instruction(idx);
    scope.record_remove(bb_name, idx, inst);
    return true;
}

bool StochasticSearch::eliminate_cse_in_place(
    MutationScope& scope, size_t idx_dup, size_t idx_orig, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    if (idx_orig >= idx_dup) return false;
    auto block = fn.block(bb_name);
    if (!block || idx_dup >= block->size()) return false;

    auto dup = block->instruction(idx_dup);
    auto orig = block->instruction(idx_orig);
    if (!dup || !orig) return false;
    if (!dup->has_name() || !orig->has_name()) return false;

    // Re-verify structural equality (the site list may be stale).
    std::string k1 = cse_key(*dup);
    if (k1.empty() || k1 != cse_key(*orig)) return false;

    // Rewire uses of the duplicate to the earlier result, then delete it.
    replace_all_uses_scoped(scope, dup->name(), orig);
    block->remove_instruction(idx_dup);
    scope.record_remove(bb_name, idx_dup, dup);
    return true;
}

bool StochasticSearch::swap_instructions_in_place(
    MutationScope& scope, size_t idx, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    auto bb = fn.block(bb_name);
    if (!bb || idx + 1 >= bb->size()) return false;
    auto inst_a = bb->instruction(idx);
    auto inst_b = bb->instruction(idx + 1);
    if (!inst_a || !inst_b) return false;
    if (inst_a->is_terminator() || inst_b->is_terminator()) return false;
    // Don't reorder memory/side-effect ops or PHI nodes.
    if (!swap_memory_safe(*inst_a, *inst_b)) return false;
    // Don't swap if b depends on a's result.
    if (inst_a->has_name()) {
        for (auto& op : inst_b->operands()) {
            if (op && op->has_name() && op->name() == inst_a->name()) return false;
        }
    }
    // Swap the two shared_ptrs, recording each slot as a Replace.
    bb->replace_instruction(idx, inst_b);
    scope.record_replace(bb_name, idx, inst_a);
    bb->replace_instruction(idx + 1, inst_a);
    scope.record_replace(bb_name, idx + 1, inst_b);
    return true;
}

std::shared_ptr<ir::Function> StochasticSearch::apply_mutation_impl(
    const ir::Function& fn, const Mutation& mut, size_t inst_count) {
    // Near the instruction cap, only destructive / size-
    // neutral mutations are allowed so per-iteration work stays bounded.
    if (!mutation_allowed_near_cap(mut.kind, inst_count)) return nullptr;

    switch (mut.kind) {
        case MutationKind::StrengthReduce:
            return strength_reduce(fn, mut.instruction_index, mut.block_name);
        case MutationKind::FoldConstant:
            return fold_constant(fn, mut.instruction_index, mut.block_name);
        case MutationKind::CombineInstructions: {
            // aux_index carries the second (use) instruction; legacy
            // callers leave it unset, meaning the adjacent pair.
            size_t idx_b = (mut.aux_index > mut.instruction_index)
                               ? mut.aux_index
                               : mut.instruction_index + 1;
            return combine_instructions(fn, mut.instruction_index, idx_b,
                                        mut.block_name);
        }
        case MutationKind::DeleteInstruction:
            return delete_dead_code(fn, mut.instruction_index, mut.block_name);
        case MutationKind::SimplifyIdentity:
            return simplify_identity(fn, mut.instruction_index, mut.block_name);
        case MutationKind::EliminateCommonSubexpr:
            return eliminate_cse(fn, mut.instruction_index, mut.aux_index,
                                 mut.block_name);
        case MutationKind::PatternGuided:
            return pattern_guided_mutate(fn);
        case MutationKind::SwapInstructions: {
            // Swap two adjacent independent instructions. Precheck on the
            // original — deep-copy only when the swap is actually legal
            // (most random picks are not, and cloning first wasted most
            // of the search time).
            auto bb = fn.block(mut.block_name);
            if (!bb || mut.instruction_index + 1 >= bb->size()) return nullptr;
            size_t i = mut.instruction_index;
            auto inst_a = bb->instruction(i);
            auto inst_b = bb->instruction(i + 1);
            if (!inst_a || !inst_b) return nullptr;
            // Don't swap terminators
            if (inst_a->is_terminator() || inst_b->is_terminator()) return nullptr;
            // Don't reorder memory/side-effect operations or PHI nodes
            // (SSA name checks alone let a load move across a store).
            if (!swap_memory_safe(*inst_a, *inst_b)) return nullptr;
            // Don't swap if b depends on a's result
            if (inst_a->has_name()) {
                for (auto& op : inst_b->operands()) {
                    if (op && op->has_name() && op->name() == inst_a->name()) {
                        return nullptr;
                    }
                }
            }
            auto copy = ir::deep_copy_function(fn);
            auto cbb = copy->block(mut.block_name);
            if (!cbb || i + 1 >= cbb->size()) return nullptr;
            auto c_a = cbb->instruction(i);
            auto c_b = cbb->instruction(i + 1);
            cbb->replace_instruction(i, c_b);
            cbb->replace_instruction(i + 1, c_a);
            return copy;
        }
        case MutationKind::ReplaceInstruction:
        case MutationKind::SubstituteVariable:
        case MutationKind::UnrollLoop:
            // These are more complex mutations; for the prototype, we fall back
            // to deleting dead code as a safe default
            return delete_dead_code(fn, mut.instruction_index, mut.block_name);
        // ── STOKE-style unsound mutations (deep-copy path) ──────────
        case MutationKind::OpcodeReplace:
            return opcode_replace(fn, mut.instruction_index, mut.block_name);
        case MutationKind::OperandReplace:
            return operand_replace(fn, mut.instruction_index, mut.aux_index,
                                    mut.block_name);
        case MutationKind::OperandSwap:
            return operand_swap(fn, mut.instruction_index, mut.block_name);
        case MutationKind::InstructionInsert:
            return instruction_insert(fn, mut.instruction_index, mut.block_name);
        case MutationKind::InstructionReplace:
            return instruction_replace_full(fn, mut.instruction_index, mut.block_name);
        default:
            return nullptr;
    }
}

// ── Simulated annealing acceptance ──────────────────────────────────────────

bool StochasticSearch::accept(double current_score, double candidate_score,
                               double temperature) {
    // Always accept improvements (higher score is better)
    if (candidate_score >= current_score) {
        return true;
    }

    // Accept worse solutions with probability exp((candidate - current) / temperature)
    if (temperature <= 0.0) {
        return false;
    }

    double delta = candidate_score - current_score;
    double probability = std::exp(delta / temperature);

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_) < probability;
}

// ── Strength reduction ──────────────────────────────────────────────────────
//   mul  x, 2^k → shl  x, k
//   udiv x, 2^k → lshr x, k          (unsigned only — sdiv rounds toward 0)
//   urem x, 2^k → and  x, 2^k - 1

std::shared_ptr<ir::Function> StochasticSearch::strength_reduce(
    const ir::Function& fn, size_t idx, const std::string& bb_name) {
    // Precheck applicability on the original — deep-copy only if the
    // mutation actually applies. Operand shared_ptrs are shared between
    // the original and the copy (see Clone.h), so values extracted here
    // are valid to reference from the copy's new instruction.
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return nullptr;
    auto inst = block->instruction(idx);
    if (!inst) return nullptr;

    auto repl = build_strength_reduced(*inst);
    if (!repl) return nullptr;

    // Applicable — now pay for the deep copy and swap the instruction in.
    // Uses of the original result resolve by name, and the replacement
    // carries the same name — no rewrite needed.
    auto copy = ir::deep_copy_function(fn);
    auto cblock = copy->block(bb_name);
    if (!cblock || idx >= cblock->size()) return nullptr;
    cblock->replace_instruction(idx, repl);
    return copy;
}

// ── Constant folding ──────────────────────────────────────────────────────────

std::shared_ptr<ir::Function> StochasticSearch::fold_constant(
    const ir::Function& fn, size_t idx, const std::string& bb_name) {
    // Precheck on the original — deep-copy only when both operands are
    // constants and the opcode is foldable. build_folded_constant applies
    // the LLVM type's wrapping / unsigned semantics and refuses UB (div by
    // zero, INT_MIN/-1, out-of-range shifts).
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return nullptr;

    auto inst = block->instruction(idx);
    if (!inst) return nullptr;

    // Emit a real ConstantInt and rewrite all uses of the
    // binop's result name to point at it, then remove the binop.
    auto const_result = build_folded_constant(*inst);
    if (!const_result) return nullptr;

    // Applicable — now pay for the deep copy and rewrite it.
    auto copy = ir::deep_copy_function(fn);
    auto cblock = copy->block(bb_name);
    if (!cblock || idx >= cblock->size()) return nullptr;

    if (inst->has_name()) {
        replace_all_uses(*copy, inst->name(), const_result);
    }
    // Remove the now-dead binop. remove_instruction is safe because we
    // just rewrote every use to point at const_result.
    cblock->remove_instruction(idx);

    return copy;
}

// ── Combine instructions ─────────────────────────────────────────────────────

std::shared_ptr<ir::Instruction> StochasticSearch::build_combined_instruction(
    const ir::Function& fn, size_t idx_a, size_t idx_b, const std::string& bb_name) {
    // Pattern: %a = add/sub i32 %x, C1; ...; %b = add/sub i32 %a, C2
    //        → %b = add i32 %x, (±C1 ± C2)
    // The two instructions no longer need to be adjacent — only in the
    // same block, with %a's single use being %b (pure SSA ops, so any
    // instructions in between cannot interfere).
    //
    // Soundness note: only `x ± C` forms combine. `sub C, x` computes
    // C - x — the sign of x is flipped, and folding it into an Add would
    // be WRONG.
    auto block = fn.block(bb_name);
    if (idx_a >= idx_b) return nullptr;
    if (!block || idx_b >= block->size()) return nullptr;

    auto inst_a = block->instruction(idx_a);
    auto inst_b = block->instruction(idx_b);
    if (!inst_a || !inst_b) return nullptr;

    if (inst_a->opcode() != ir::Opcode::Add && inst_a->opcode() != ir::Opcode::Sub) return nullptr;
    if (inst_b->opcode() != ir::Opcode::Add && inst_b->opcode() != ir::Opcode::Sub) return nullptr;

    if (!inst_a->has_name()) return nullptr;
    if (inst_a->num_operands() < 2 || inst_b->num_operands() < 2) return nullptr;

    // Check that inst_b uses inst_a as an operand
    bool uses_a = false;
    for (auto& op : inst_b->operands()) {
        if (op && op->has_name() && op->name() == inst_a->name()) {
            uses_a = true;
            break;
        }
    }
    if (!uses_a) return nullptr;

    // Check that inst_a's result is not used by any
    // instruction OTHER than inst_b. If it is, removing inst_a would
    // leave dangling references. Bail (return nullptr) so the SA loop
    // tries a different mutation.
    size_t other_uses = count_uses_excluding(fn, inst_a->name(), inst_b.get());
    if (other_uses > 0) {
        stats_.mutations_rejected_by_validation++;
        return nullptr;
    }

    auto a_op0 = inst_a->operand(0);
    auto a_op1 = inst_a->operand(1);
    auto b_op0 = inst_b->operand(0);
    auto b_op1 = inst_b->operand(1);

    if (!a_op0 || !a_op1 || !b_op0 || !b_op1) return nullptr;

    // Extract inst_a as (x, effective constant): add x,C / add C,x → +C;
    // sub x,C → -C; sub C,x → NOT combinable.
    std::shared_ptr<ir::Value> a_var;
    int64_t a_const = 0;

    if (is_constant_int(a_op1) && !is_constant_int(a_op0)) {
        a_var = a_op0;
        a_const = as_constant_int(a_op1)->value();
        if (inst_a->opcode() == ir::Opcode::Sub) a_const = -a_const;
    } else if (is_constant_int(a_op0) && !is_constant_int(a_op1) &&
               inst_a->opcode() == ir::Opcode::Add) {
        a_var = a_op1;
        a_const = as_constant_int(a_op0)->value();
    } else {
        return nullptr;
    }

    // Extract inst_b the same way; its variable operand must be %a.
    std::shared_ptr<ir::Value> b_var;
    int64_t b_const = 0;

    if (is_constant_int(b_op1) && !is_constant_int(b_op0)) {
        b_var = b_op0;
        b_const = as_constant_int(b_op1)->value();
        if (inst_b->opcode() == ir::Opcode::Sub) b_const = -b_const;
    } else if (is_constant_int(b_op0) && !is_constant_int(b_op1) &&
               inst_b->opcode() == ir::Opcode::Add) {
        b_var = b_op1;
        b_const = as_constant_int(b_op0)->value();
    } else {
        return nullptr;
    }

    if (!b_var || !b_var->has_name() || b_var->name() != inst_a->name()) {
        return nullptr;
    }

    // Combine the constants with wrapping semantics at the result width.
    const unsigned bw = (inst_b->type() && inst_b->type()->bit_width() > 0)
                            ? inst_b->type()->bit_width() : 32;
    int64_t combined = wrap_to_width(a_const + b_const, bw);

    // Create the combined instruction: %b = add iN %x, combined
    auto combined_const = std::make_shared<ir::ConstantInt>(
        std::make_shared<ir::IntegerType>(bw), combined);

    auto combined_inst = std::make_shared<ir::Instruction>(
        ir::Opcode::Add, inst_b->type(), inst_b->name());
    combined_inst->add_operand(a_var);
    combined_inst->add_operand(combined_const);
    return combined_inst;
}

std::shared_ptr<ir::Function> StochasticSearch::combine_instructions(
    const ir::Function& fn, size_t idx_a, size_t idx_b, const std::string& bb_name) {
    // Precheck the whole pattern on the original — deep-copy only when
    // the combine is actually applicable and safe.
    auto combined_inst = build_combined_instruction(fn, idx_a, idx_b, bb_name);
    if (!combined_inst) return nullptr;

    // Applicable — now pay for the deep copy and rewrite it.
    auto copy = ir::deep_copy_function(fn);
    auto cblock = copy->block(bb_name);
    if (!cblock || idx_b >= cblock->size()) return nullptr;

    // Replace inst_b with the combined instruction
    cblock->replace_instruction(idx_b, combined_inst);

    // Remove inst_a (it's now dead — verified by the use-check above).
    // idx_a < idx_b, so the combined instruction shifts down by one.
    cblock->remove_instruction(idx_a);

    return copy;
}

// ── Delete dead code ───────────────────────────────────────────────────────

std::shared_ptr<ir::Function> StochasticSearch::delete_dead_code(
    const ir::Function& fn, size_t idx, const std::string& bb_name) {
    // All safety checks run on the original; the deep copy happens only
    // when the instruction is provably dead.
    if (!is_dead_instruction(fn, idx, bb_name)) return nullptr;

    // Safe to remove — now pay for the deep copy.
    auto copy = ir::deep_copy_function(fn);
    auto cblock = copy->block(bb_name);
    if (!cblock || idx >= cblock->size()) return nullptr;
    cblock->remove_instruction(idx);
    return copy;
}

// ── Algebraic identity simplification ───────────────────────────────────────
// x+0 → x, x*1 → x, x*0 → 0, x-x → 0, x^x → 0, x&x → x, x|0 → x, ... (see
// classify_identity for the full table). Replaces every use of the result
// with the surviving operand (or a constant) and deletes the instruction.

std::shared_ptr<ir::Function> StochasticSearch::simplify_identity(
    const ir::Function& fn, size_t idx, const std::string& bb_name) {
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return nullptr;

    auto inst = block->instruction(idx);
    if (!inst) return nullptr;

    auto replacement = build_identity_replacement(*inst);
    if (!replacement) return nullptr;

    // Applicable — now pay for the deep copy and rewrite it.
    auto copy = ir::deep_copy_function(fn);
    auto cblock = copy->block(bb_name);
    if (!cblock || idx >= cblock->size()) return nullptr;

    if (inst->has_name()) {
        replace_all_uses(*copy, inst->name(), replacement);
    }
    cblock->remove_instruction(idx);
    return copy;
}

// ── Same-block common-subexpression elimination ─────────────────────────────
// Two structurally identical pure instructions in one block always compute
// the same value; the later one's uses are rewired to the earlier result
// and the duplicate is deleted. Sound by construction for the pure opcode
// set admitted by cse_key (no loads: those need alias analysis).

std::shared_ptr<ir::Function> StochasticSearch::eliminate_cse(
    const ir::Function& fn, size_t idx_dup, size_t idx_orig,
    const std::string& bb_name) {
    if (idx_orig >= idx_dup) return nullptr;
    auto block = fn.block(bb_name);
    if (!block || idx_dup >= block->size()) return nullptr;

    auto dup = block->instruction(idx_dup);
    auto orig = block->instruction(idx_orig);
    if (!dup || !orig) return nullptr;
    if (!dup->has_name() || !orig->has_name()) return nullptr;

    // Re-verify structural equality (the site list may be stale).
    std::string k1 = cse_key(*dup);
    if (k1.empty() || k1 != cse_key(*orig)) return nullptr;

    // Applicable — now pay for the deep copy and rewrite it.
    auto copy = ir::deep_copy_function(fn);
    auto cblock = copy->block(bb_name);
    if (!cblock || idx_dup >= cblock->size()) return nullptr;

    auto corig = cblock->instruction(idx_orig);
    if (!corig) return nullptr;

    // Rewire all uses of the duplicate's result to the earlier result,
    // then delete the duplicate.
    replace_all_uses(*copy, dup->name(), corig);
    cblock->remove_instruction(idx_dup);
    return copy;
}

// ── Pattern-guided mutation ────────────────────────────────────────────────────

std::shared_ptr<ir::Function> StochasticSearch::pattern_guided_mutate(
    const ir::Function& fn) {
    stats_.pattern_guided_attempts++;
    if (!pattern_lib_ || pattern_lib_->size() == 0) return nullptr;

    auto matches = pattern_lib_->match(fn, config_.target_arch);
    if (matches.empty()) return nullptr;

    // Pick a random match
    std::uniform_int_distribution<size_t> dist(0, matches.size() - 1);
    auto& chosen = matches[dist(rng_)];

    auto result = pattern_lib_->apply(fn, chosen, config_.target_arch);
    if (result) {
        stats_.pattern_guided_applied++;
        // Record the application so the library can track success rate.
        pattern_lib_->record_application(chosen.pattern_id);
    }
    return result;
}

// ── STOKE-style unsound mutations (deep-copy path) ──────────────────────────
//
// Each helper prechecks applicability on the original (so a non-applicable
// site costs no deep copy), then deep-copies the function and applies the
// edit. These are the legacy deep-copy counterparts to the in-place
// variants below; the search loop uses the in-place path by default (the
// deep-copy path is only used when use_in_place_mutation is false, or by

std::shared_ptr<ir::Function> StochasticSearch::opcode_replace(
    const ir::Function& fn, size_t idx, const std::string& bb_name) {
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return nullptr;
    auto inst = block->instruction(idx);
    if (!inst || inst->is_terminator()) return nullptr;
    auto repl = build_opcode_replacement(*inst, rng_);
    if (!repl) return nullptr;
    auto copy = ir::deep_copy_function(fn);
    auto cbb = copy->block(bb_name);
    if (!cbb || idx >= cbb->size()) return nullptr;
    cbb->replace_instruction(idx, repl);
    return copy;
}

std::shared_ptr<ir::Function> StochasticSearch::operand_replace(
    const ir::Function& fn, size_t idx, size_t op_idx, const std::string& bb_name) {
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return nullptr;
    auto inst = block->instruction(idx);
    if (!inst || inst->is_terminator()) return nullptr;
    if (op_idx >= inst->num_operands()) return nullptr;
    auto target = inst->operand(op_idx);
    if (!target || !target->type()) return nullptr;

    std::vector<std::shared_ptr<ir::Value>> pool;
    collect_operand_pool(fn, *block, idx, *target->type(), pool);
    // Remove the original operand from the pool so we always pick a
    // different value (a no-op replace isn't a mutation).
    pool.erase(std::remove_if(pool.begin(), pool.end(),
        [&](const std::shared_ptr<ir::Value>& v) {
            return v.get() == target.get() ||
                   (v->has_name() && target->has_name() &&
                    v->name() == target->name());
        }), pool.end());
    if (pool.empty()) return nullptr;

    std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);
    auto new_op = pool[dist(rng_)];
    auto repl = clone_instruction(*inst);
    repl->set_operand(op_idx, new_op);

    auto copy = ir::deep_copy_function(fn);
    auto cbb = copy->block(bb_name);
    if (!cbb || idx >= cbb->size()) return nullptr;
    cbb->replace_instruction(idx, repl);
    return copy;
}

std::shared_ptr<ir::Function> StochasticSearch::operand_swap(
    const ir::Function& fn, size_t idx, const std::string& bb_name) {
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return nullptr;
    auto inst = block->instruction(idx);
    if (!inst || !inst->is_binary_op() || inst->num_operands() < 2) return nullptr;
    // Build the swapped instruction on the original (no RNG draw needed
    // — the swap is deterministic).
    auto repl = clone_instruction(*inst);
    auto op0 = repl->operand(0);
    auto op1 = repl->operand(1);
    repl->set_operand(0, op1);
    repl->set_operand(1, op0);
    auto copy = ir::deep_copy_function(fn);
    auto cbb = copy->block(bb_name);
    if (!cbb || idx >= cbb->size()) return nullptr;
    cbb->replace_instruction(idx, repl);
    return copy;
}

std::shared_ptr<ir::Function> StochasticSearch::instruction_insert(
    const ir::Function& fn, size_t pos, const std::string& bb_name) {
    auto block = fn.block(bb_name);
    if (!block || pos > block->size()) return nullptr;
    // Need a non-empty pool of integer-typed values to build a random
    // binop over. The pool comes from earlier instructions + arguments.
    std::shared_ptr<ir::Type> common_ty;
    // Use the first named integer-typed instruction's type as the
    // common type. If none exists, bail.
    for (size_t i = 0; i < block->size() && !common_ty; ++i) {
        auto inst = block->instruction(i);
        if (inst && inst->has_name() && inst->type() &&
            inst->type()->is_integer()) {
            common_ty = inst->type();
        }
    }
    if (!common_ty) {
        // Fall back to first argument's type.
        for (auto& arg : fn.arguments()) {
            if (arg.type && arg.type->is_integer()) {
                common_ty = arg.type;
                break;
            }
        }
    }
    if (!common_ty) return nullptr;

    std::vector<std::shared_ptr<ir::Value>> pool;
    collect_operand_pool(fn, *block, pos, *common_ty, pool);
    if (pool.size() < 2) return nullptr;

    std::string name = "_stoke_" + std::to_string(next_stoke_name_++);
    auto new_inst = build_random_binop(pool, common_ty, name, rng_);
    if (!new_inst) return nullptr;

    auto copy = ir::deep_copy_function(fn);
    auto cbb = copy->block(bb_name);
    if (!cbb || pos > cbb->size()) return nullptr;
    cbb->insert_instruction(pos, new_inst);
    return copy;
}

std::shared_ptr<ir::Function> StochasticSearch::instruction_replace_full(
    const ir::Function& fn, size_t idx, const std::string& bb_name) {
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return nullptr;
    auto inst = block->instruction(idx);
    if (!inst || inst->is_terminator()) return nullptr;
    if (!inst->type()) return nullptr;
    // Build a random binop with the SAME result type and name as the
    // original (so downstream uses still resolve).
    std::vector<std::shared_ptr<ir::Value>> pool;
    collect_operand_pool(fn, *block, idx, *inst->type(), pool);
    if (pool.size() < 2) return nullptr;

    auto repl = build_random_binop(pool, inst->type(), inst->name(), rng_);
    if (!repl) return nullptr;

    auto copy = ir::deep_copy_function(fn);
    auto cbb = copy->block(bb_name);
    if (!cbb || idx >= cbb->size()) return nullptr;
    cbb->replace_instruction(idx, repl);
    return copy;
}

// ── STOKE-style unsound mutations (in-place path) ────────────────────────────
//
// Transactional counterparts to the deep-copy helpers above. Each edits
// the live function held by `scope` and records an undo action; rejection
// (scope destructor without commit()) restores the function exactly. Used
// by the search loop when use_in_place_mutation is true (the default) and
// the kind is amenable to in-place application (all five new kinds are).

bool StochasticSearch::opcode_replace_in_place(
    MutationScope& scope, size_t idx, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return false;
    auto inst = block->instruction(idx);
    if (!inst || inst->is_terminator()) return false;
    auto repl = build_opcode_replacement(*inst, rng_);
    if (!repl) return false;
    block->replace_instruction(idx, repl);
    scope.record_replace(bb_name, idx, inst);
    return true;
}

bool StochasticSearch::operand_replace_in_place(
    MutationScope& scope, size_t idx, size_t op_idx, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return false;
    auto inst = block->instruction(idx);
    if (!inst || inst->is_terminator()) return false;
    if (op_idx >= inst->num_operands()) return false;
    auto target = inst->operand(op_idx);
    if (!target || !target->type()) return false;

    std::vector<std::shared_ptr<ir::Value>> pool;
    collect_operand_pool(fn, *block, idx, *target->type(), pool);
    pool.erase(std::remove_if(pool.begin(), pool.end(),
        [&](const std::shared_ptr<ir::Value>& v) {
            return v.get() == target.get() ||
                   (v->has_name() && target->has_name() &&
                    v->name() == target->name());
        }), pool.end());
    if (pool.empty()) return false;

    std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);
    auto new_op = pool[dist(rng_)];
    auto repl = clone_instruction(*inst);
    repl->set_operand(op_idx, new_op);
    block->replace_instruction(idx, repl);
    scope.record_replace(bb_name, idx, inst);
    return true;
}

bool StochasticSearch::operand_swap_in_place(
    MutationScope& scope, size_t idx, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return false;
    auto inst = block->instruction(idx);
    if (!inst || !inst->is_binary_op() || inst->num_operands() < 2) return false;
    auto repl = clone_instruction(*inst);
    auto op0 = repl->operand(0);
    auto op1 = repl->operand(1);
    repl->set_operand(0, op1);
    repl->set_operand(1, op0);
    block->replace_instruction(idx, repl);
    scope.record_replace(bb_name, idx, inst);
    return true;
}

bool StochasticSearch::instruction_insert_in_place(
    MutationScope& scope, size_t pos, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    auto block = fn.block(bb_name);
    if (!block || pos > block->size()) return false;

    std::shared_ptr<ir::Type> common_ty;
    for (size_t i = 0; i < block->size() && !common_ty; ++i) {
        auto inst = block->instruction(i);
        if (inst && inst->has_name() && inst->type() &&
            inst->type()->is_integer()) {
            common_ty = inst->type();
        }
    }
    if (!common_ty) {
        for (auto& arg : fn.arguments()) {
            if (arg.type && arg.type->is_integer()) {
                common_ty = arg.type;
                break;
            }
        }
    }
    if (!common_ty) return false;

    std::vector<std::shared_ptr<ir::Value>> pool;
    collect_operand_pool(fn, *block, pos, *common_ty, pool);
    if (pool.size() < 2) return false;

    std::string name = "_stoke_" + std::to_string(next_stoke_name_++);
    auto new_inst = build_random_binop(pool, common_ty, name, rng_);
    if (!new_inst) return false;

    block->insert_instruction(pos, new_inst);
    scope.record_insert(bb_name, pos);
    return true;
}

bool StochasticSearch::instruction_replace_in_place(
    MutationScope& scope, size_t idx, const std::string& bb_name) {
    ir::Function& fn = scope.function();
    auto block = fn.block(bb_name);
    if (!block || idx >= block->size()) return false;
    auto inst = block->instruction(idx);
    if (!inst || inst->is_terminator() || !inst->type()) return false;

    std::vector<std::shared_ptr<ir::Value>> pool;
    collect_operand_pool(fn, *block, idx, *inst->type(), pool);
    if (pool.size() < 2) return false;

    auto repl = build_random_binop(pool, inst->type(), inst->name(), rng_);
    if (!repl) return false;

    block->replace_instruction(idx, repl);
    scope.record_replace(bb_name, idx, inst);
    return true;
}

} // namespace clunk::search
