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
 * Clunk Series Expander — implementation.
 * See include/clunk/Search/SeriesExpander.h for the contract.
 *
 * Strategy:
 *   1. Find single-block natural loops.
 *   2. Pattern-match the two phis (induction + accumulator).
 *   3. Classify the accumulator update (constant / arithmetic / scaled).
 *   4. Compute the trip count symbolically.
 *   5. Emit the closed-form arithmetic (no loop).
 *   6. Validate and return.
 */

#include "clunk/Search/SeriesExpander.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "clunk/IR/Clone.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/LoopAnalysis.h"
#include "clunk/IR/Value.h"

namespace clunk::search {

namespace {

// ── Helpers ────────────────────────────────────────────────────────────────

unsigned int_bits(const std::shared_ptr<ir::Type>& t) {
    return t && t->is_integer() ? static_cast<unsigned>(t->bit_width()) : 0;
}

// Extract a constant int64 from a Value (returns nullopt for non-constants).
std::optional<int64_t> as_const_int(const std::shared_ptr<ir::Value>& v) {
    if (!v) return std::nullopt;
    auto* ci = dynamic_cast<const ir::ConstantInt*>(v.get());
    if (!ci) return std::nullopt;
    return ci->value();
}

// Split "a,b,c" (a phi's phi_blocks metadata) into labels.
std::vector<std::string> split_phi_blocks(const ir::Instruction& phi) {
    std::vector<std::string> out;
    auto it = phi.metadata().find("phi_blocks");
    if (it == phi.metadata().end()) return out;
    std::istringstream ss(it->second);
    std::string b;
    while (std::getline(ss, b, ',')) out.push_back(b);
    return out;
}

// All value names defined inside the given block set.
std::unordered_set<std::string> block_defs(const ir::Function& fn,
                                             const std::unordered_set<std::string>& blocks) {
    std::unordered_set<std::string> defs;
    for (auto& name : blocks) {
        auto bb = fn.block(name);
        if (!bb) continue;
        for (auto& inst : bb->instructions()) {
            if (inst && inst->has_name()) defs.insert(inst->name());
        }
    }
    return defs;
}

// Is `v` loop-invariant? (defined outside the loop's blocks, or a constant,
// or a function argument.)
bool is_loop_invariant(const std::shared_ptr<ir::Value>& v,
                        const std::unordered_set<std::string>& loop_defs) {
    if (!v) return true;
    if (!v->has_name()) return true;  // constant
    return !loop_defs.count(v->name());
}

// ── Phi info ───────────────────────────────────────────────────────────────
struct PhiInfo {
    std::shared_ptr<ir::Instruction> phi;
    std::shared_ptr<ir::Value> init;      // value from the preheader
    std::shared_ptr<ir::Value> carried;   // value from the latch (self-loop)
    std::string name;                     // phi's result name
};

// Parse a phi instruction. Returns nullopt if the phi doesn't have exactly
// 2 incoming values from [preheader, loop_header].
std::optional<PhiInfo> parse_phi(const ir::Instruction& phi,
                                   const std::string& preheader,
                                   const std::string& header) {
    if (phi.opcode() != ir::Opcode::Phi) return std::nullopt;
    if (phi.num_operands() != 2) return std::nullopt;
    auto blocks = split_phi_blocks(phi);
    if (blocks.size() != 2) return std::nullopt;

    PhiInfo info;
    info.phi = std::make_shared<ir::Instruction>(phi);  // shallow copy is fine
    info.name = phi.name();
    for (size_t k = 0; k < 2; ++k) {
        if (blocks[k] == preheader) info.init = phi.operand(k);
        else if (blocks[k] == header) info.carried = phi.operand(k);
    }
    if (!info.init || !info.carried || info.name.empty()) return std::nullopt;
    return info;
}

// ── Accumulator pattern classification ─────────────────────────────────────
//
// Given the accumulator's carried value (`acc_next`), classify the update:
//   a) add %acc, %c             → Constant, c = the addend
//   b) add %acc, %i             → ArithmeticSeries
//   c) add %acc, (mul %i, %c)   → ScaledArithmetic, c = the multiplier
//
// `acc_name` is the accumulator phi's name (to identify the self-reference).
// `iv_name` is the induction variable's name.
struct AccClassification {
    AccPattern pattern = AccPattern::None;
    std::shared_ptr<ir::Value> c;  // the constant/multiplier (for Constant, ScaledArithmetic)
};

AccClassification classify_acc_update(const std::shared_ptr<ir::Value>& acc_next,
                                        const std::string& acc_name,
                                        const std::string& iv_name) {
    AccClassification result;
    auto* inst = dynamic_cast<const ir::Instruction*>(acc_next.get());
    if (!inst || inst->opcode() != ir::Opcode::Add) return result;

    // The add must NOT have nsw/nuw flags (see soundness note in the header).
    if (inst->binop_flags().nuw || inst->binop_flags().nsw) return result;

    // One operand must be the accumulator self-reference (%acc).
    // The other operand is the "addend".
    std::shared_ptr<ir::Value> addend = nullptr;
    bool found_self = false;
    for (size_t k = 0; k < inst->num_operands(); ++k) {
        auto op = inst->operand(k);
        if (op && op->has_name() && op->name() == acc_name) {
            found_self = true;
        } else {
            addend = op;
        }
    }
    if (!found_self || !addend) return result;

    // Case b: addend is the induction variable itself.
    if (addend->has_name() && addend->name() == iv_name) {
        result.pattern = AccPattern::ArithmeticSeries;
        return result;
    }

    // Case c: addend is (mul %i, %c).
    if (auto* mul_inst = dynamic_cast<const ir::Instruction*>(addend.get())) {
        if (mul_inst->opcode() == ir::Opcode::Mul &&
            !mul_inst->binop_flags().nuw && !mul_inst->binop_flags().nsw) {
            // One operand must be %i, the other is %c.
            bool found_iv = false;
            std::shared_ptr<ir::Value> c_val = nullptr;
            for (size_t k = 0; k < mul_inst->num_operands(); ++k) {
                auto op = mul_inst->operand(k);
                if (op && op->has_name() && op->name() == iv_name) {
                    found_iv = true;
                } else {
                    c_val = op;
                }
            }
            if (found_iv && c_val) {
                result.pattern = AccPattern::ScaledArithmetic;
                result.c = c_val;
                return result;
            }
        }
    }

    // Case a: addend is a constant or loop-invariant value (NOT %i).
    // The addend must not reference the induction variable or accumulator.
    if (addend->has_name() && addend->name() == iv_name) return result;  // already handled
    result.pattern = AccPattern::Constant;
    result.c = addend;
    return result;
}

} // anonymous namespace

// ── Constructor ────────────────────────────────────────────────────────────

SeriesExpander::SeriesExpander(evaluator::EvaluationEngine* engine,
                                 const SeriesExpanderConfig& config)
    : engine_(engine), config_(config) {}

// ── expand (public entry point) ────────────────────────────────────────────

std::shared_ptr<ir::Function> SeriesExpander::expand(const ir::Function& fn) {
    ++stats_.functions_seen;
    auto t0 = std::chrono::steady_clock::now();

    auto result = try_expand_loop(fn);

    stats_.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return result;
}

// ── try_expand_loop ────────────────────────────────────────────────────────

std::shared_ptr<ir::Function> SeriesExpander::try_expand_loop(const ir::Function& fn) {
    // ── Scope gate ───────────────────────────────────────────────────
    if (fn.argument_count() > 4) { ++stats_.functions_skipped; return nullptr; }
    auto ret_ty = fn.return_type();
    if (!ret_ty || !ret_ty->is_integer()) { ++stats_.functions_skipped; return nullptr; }
    for (const auto& arg : fn.arguments()) {
        if (!arg.type || !arg.type->is_integer()) {
            ++stats_.functions_skipped;
            return nullptr;
        }
    }
    if (fn.instruction_count() > config_.max_function_instructions) {
        ++stats_.functions_skipped;
        return nullptr;
    }

    // ── Find natural loops ───────────────────────────────────────────
    auto loops = ir::find_natural_loops(fn);
    if (loops.empty()) { ++stats_.functions_skipped; return nullptr; }

    // Try each loop (typically there's only one in a single-block function).
    for (auto& loop : loops) {
        if (!loop.is_single_block()) continue;
        if (loop.preheader.empty()) continue;

        auto header = fn.block(loop.header);
        if (!header) continue;
        auto preheader = fn.block(loop.preheader);
        if (!preheader) continue;

        // ── Shape: single-block loop, conditional branch to {self, exit}.
        auto term = header->terminator();
        if (!term || term->opcode() != ir::Opcode::Br ||
            term->num_operands() != 1) {
            continue;
        }
        auto md_t = term->metadata().find("true_bb");
        auto md_f = term->metadata().find("false_bb");
        if (md_t == term->metadata().end() || md_f == term->metadata().end())
            continue;
        const std::string t_bb = md_t->second, f_bb = md_f->second;
        // One of the branch targets must be the header (back-edge), the
        // other is the exit.
        if ((t_bb == loop.header) == (f_bb == loop.header)) continue;
        const std::string exit_bb = (t_bb == loop.header) ? f_bb : t_bb;

        // ── Collect phis and body instructions ──────────────────────
        std::vector<PhiInfo> phis;
        std::vector<std::shared_ptr<ir::Instruction>> body;
        bool ok = true;
        for (size_t i = 0; i + 1 < header->size(); ++i) {  // excl. term
            auto inst = header->instruction(i);
            if (!inst) { ok = false; break; }
            if (inst->opcode() == ir::Opcode::Phi) {
                if (!body.empty()) { ok = false; break; }  // phis must lead
                auto pi = parse_phi(*inst, loop.preheader, loop.header);
                if (!pi) { ok = false; break; }
                phis.push_back(*pi);
            } else {
                if (inst->opcode() == ir::Opcode::Invoke) { ok = false; break; }
                body.push_back(inst);
            }
        }
        if (!ok || phis.size() != 2) continue;
        if (body.size() > config_.max_body_instructions) continue;

        // ── Identify the induction variable and accumulator ──────────
        // The induction variable: phi whose carried value is `add %i, step_const`.
        // The accumulator: the other phi.
        auto loop_def_set = block_defs(fn, loop.blocks);

        size_t iv_idx = 2, acc_idx = 2;
        std::shared_ptr<ir::Value> step_val;
        for (size_t a = 0; a < 2; ++a) {
            for (size_t b = 0; b < 2; ++b) {
                if (a == b) continue;
                // Is phi[a]'s carried value `add phi[a], step_const`?
                auto carried = phis[a].carried;
                auto* inst = dynamic_cast<const ir::Instruction*>(carried.get());
                if (!inst || inst->opcode() != ir::Opcode::Add) continue;
                if (inst->binop_flags().nuw || inst->binop_flags().nsw) continue;
                // One operand must be phi[a]'s name, the other a constant.
                bool found_self = false;
                std::shared_ptr<ir::Value> other = nullptr;
                for (size_t k = 0; k < inst->num_operands(); ++k) {
                    auto op = inst->operand(k);
                    if (op && op->has_name() && op->name() == phis[a].name) {
                        found_self = true;
                    } else {
                        other = op;
                    }
                }
                if (!found_self || !other) continue;
                // `other` must be a positive constant (the step).
                auto step_c = as_const_int(other);
                if (!step_c || *step_c <= 0) continue;
                // phi[b] must NOT have the same shape (it's the accumulator).
                iv_idx = a;
                acc_idx = b;
                step_val = other;
                break;
            }
            if (iv_idx < 2) break;
        }
        if (iv_idx >= 2) continue;  // no induction variable found

        const auto& iv = phis[iv_idx];
        const auto& acc = phis[acc_idx];
        const int64_t step = *as_const_int(step_val);
        if (step <= 0) continue;

        // ── The accumulator's carried value must be `add %acc, <addend>`.
        auto acc_class = classify_acc_update(acc.carried, acc.name, iv.name);
        if (acc_class.pattern == AccPattern::None) {
            ++stats_.loops_rejected;
            continue;
        }

        // ── Find the exit condition: `icmp pred %i, %bound` ──────────
        // The condition operand must be an icmp whose LHS or RHS is the
        // induction variable, and the other side is loop-invariant.
        auto cond_val = term->operand(0);
        if (!cond_val) continue;
        auto* icmp_inst = dynamic_cast<const ir::Instruction*>(cond_val.get());
        if (!icmp_inst || icmp_inst->opcode() != ir::Opcode::ICmp) continue;
        if (icmp_inst->num_operands() != 2) continue;

        auto pred_it = icmp_inst->metadata().find("pred");
        if (pred_it == icmp_inst->metadata().end()) continue;
        auto pred = static_cast<ir::CmpPredicate>(std::stoul(pred_it->second));

        // Identify which operand is %i and which is %bound.
        std::shared_ptr<ir::Value> bound_val;
        for (size_t k = 0; k < 2; ++k) {
            auto op = icmp_inst->operand(k);
            if (op && op->has_name() && op->name() == iv.name) {
                bound_val = icmp_inst->operand(1 - k);
            }
        }
        if (!bound_val) continue;

        // `bound_val` must be loop-invariant.
        if (!is_loop_invariant(bound_val, loop_def_set)) continue;

        // `iv.init` (start) must be loop-invariant.
        if (!is_loop_invariant(iv.init, loop_def_set)) continue;

        // `acc.init` must be loop-invariant.
        if (!is_loop_invariant(acc.init, loop_def_set)) continue;

        // Only handle SLT/ULT and SLE/ULE with positive step.
        bool is_strict = (pred == ir::CmpPredicate::SLT || pred == ir::CmpPredicate::ULT);
        bool is_nonstrict = (pred == ir::CmpPredicate::SLE || pred == ir::CmpPredicate::ULE);
        if (!is_strict && !is_nonstrict) {
            ++stats_.loops_rejected;
            continue;
        }

        // ── Pattern fully matched — emit the closed form ─────────────
        ++stats_.loops_matched;

        unsigned width = int_bits(iv.phi->type());
        if (width == 0) { ++stats_.loops_rejected; continue; }
        auto int_ty = type_ctx_.get_int(width);

        // Helper: resolve a loop-invariant value in the rewritten function.
        // We'll build the replacement in a deep copy of `fn`, then replace
        // the loop header's contents with the closed form and redirect
        // the preheader to branch to the exit.
        auto work = ir::deep_copy_function(fn);
        if (!work) continue;

        // Build a name → Value map for the work copy (for resolving
        // loop-invariant operands like start, bound, init, c).
        std::unordered_map<std::string, std::shared_ptr<ir::Value>> def_map;
        for (auto& arg : work->arguments()) {
            def_map[arg.name] = std::make_shared<ir::Value>(arg.type, arg.name);
        }
        for (auto& block : work->blocks()) {
            for (auto& inst : block->instructions()) {
                if (inst && inst->has_name()) def_map[inst->name()] = inst;
            }
        }
        auto resolve = [&](const std::shared_ptr<ir::Value>& v)
            -> std::shared_ptr<ir::Value>
        {
            if (!v || !v->has_name()) return v;  // constant
            auto it = def_map.find(v->name());
            return (it != def_map.end()) ? it->second : v;
        };

        // Resolve the loop-invariant values in the work copy.
        auto start_v = resolve(iv.init);
        auto bound_v = resolve(bound_val);
        auto init_v = resolve(acc.init);
        std::shared_ptr<ir::Value> c_v = nullptr;
        if (acc_class.pattern == AccPattern::Constant ||
            acc_class.pattern == AccPattern::ScaledArithmetic) {
            c_v = resolve(acc_class.c);
        }

        // Constant helpers.
        auto const_int = [&](int64_t v) {
            return ir::ConstantInt::get(type_ctx_, v, width);
        };

        // ── Build the trip count: n = max(0, ceil((bound - start + adj) / step))
        //   adj = 0 for SLT/ULT, 1 for SLE/ULE
        //   ceil(a/b) for b > 0, a > 0: (a + b - 1) / b
        //   For a <= 0: trip = 0 (guard with select)
        //
        // We emit:
        //   %diff = sub bound, start
        //   %diff_adj = add %diff, adj            ; adj = 0 or 1
        //   %is_pos = icmp sgt %diff_adj, 0
        //   %numerator = add %diff_adj, %step_minus_1
        //   %ceil = sdiv %numerator, %step
        //   %trip = select %is_pos, %ceil, 0

        auto work_header = work->block(loop.header);
        auto work_preheader = work->block(loop.preheader);
        auto work_exit = work->block(exit_bb);
        if (!work_header || !work_preheader || !work_exit) continue;

        // Clear the header — we'll rebuild it as a straight-line block.
        work_header->instructions().clear();

        // Use the IRBuilder pattern (manual, since we need fine control).
        auto emit = [&](ir::Opcode op, std::shared_ptr<ir::Value> lhs,
                        std::shared_ptr<ir::Value> rhs,
                        const std::string& name) -> std::shared_ptr<ir::Value> {
            auto inst = ir::inst::make_binop(op, lhs, rhs, name);
            work_header->add_instruction(inst);
            return inst;
        };

        int64_t adj = is_strict ? 0 : 1;
        int64_t step_minus_1 = step - 1;

        auto diff = emit(ir::Opcode::Sub, bound_v, start_v, "_se_diff");
        auto diff_adj = emit(ir::Opcode::Add, diff,
                              const_int(adj), "_se_diff_adj");
        // icmp sgt %diff_adj, 0
        auto is_pos_inst = ir::inst::make_icmp(ir::CmpPredicate::SGT,
                                                  diff_adj, const_int(0), "_se_is_pos");
        work_header->add_instruction(is_pos_inst);
        auto is_pos = is_pos_inst;

        auto numerator = emit(ir::Opcode::Add, diff_adj,
                               const_int(step_minus_1), "_se_num");
        auto ceil_div = emit(ir::Opcode::SDiv, numerator, step_val, "_se_ceil");
        // select %is_pos, %ceil, 0
        auto trip_inst = ir::inst::make_select(is_pos, ceil_div,
                                                  const_int(0), "_se_trip");
        work_header->add_instruction(trip_inst);
        auto trip = trip_inst;

        // ── Build the closed-form sum ────────────────────────────────
        // The "series sum" (the amount to add to acc.init):
        //   Constant:        c * n
        //   ArithmeticSeries: n*start + step * n*(n-1)/2
        //   ScaledArithmetic: c * (n*start + step * n*(n-1)/2)
        //
        // We factor out the common subexpression:
        //   %n_start = mul %trip, %start
        //   %n1 = sub %trip, 1
        //   %n_n1 = mul %trip, %n1
        //   %half = sdiv %n_n1, 2           ; exact: n*(n-1) is even
        //   %step_half = mul %step, %half
        //   %arith = add %n_start, %step_half    ; = n*start + step*n*(n-1)/2
        //
        // Then:
        //   Constant:        %sum = mul %c, %trip
        //   ArithmeticSeries: %sum = %arith
        //   ScaledArithmetic: %sum = mul %c, %arith

        std::shared_ptr<ir::Value> series_sum;

        if (acc_class.pattern == AccPattern::Constant) {
            // sum = c * n
            series_sum = emit(ir::Opcode::Mul, c_v, trip, "_se_sum");
        } else {
            // Common: compute the arithmetic series sum.
            auto n_start = emit(ir::Opcode::Mul, trip, start_v, "_se_nstart");
            auto n1 = emit(ir::Opcode::Sub, trip, const_int(1), "_se_n1");
            auto n_n1 = emit(ir::Opcode::Mul, trip, n1, "_se_nn1");
            auto half = emit(ir::Opcode::SDiv, n_n1, const_int(2), "_se_half");
            auto step_half = emit(ir::Opcode::Mul, step_val, half, "_se_stephalf");
            auto arith = emit(ir::Opcode::Add, n_start, step_half, "_se_arith");

            if (acc_class.pattern == AccPattern::ArithmeticSeries) {
                series_sum = arith;
            } else {
                // ScaledArithmetic: sum = c * arith
                series_sum = emit(ir::Opcode::Mul, c_v, arith, "_se_sum");
            }
        }

        // ── result = acc.init + series_sum ───────────────────────────
        auto result = emit(ir::Opcode::Add, init_v, series_sum, acc.name);

        // Redirect: preheader branches to exit, header is dead

        // Add the unconditional branch to the exit.
        auto br_to_exit = ir::inst::make_br_uncond(exit_bb);
        work_header->add_instruction(br_to_exit);

        // Fix up the exit block's phis: any phi in the exit block that
        // references the header should use `%acc` (our result) as the
        // incoming value.
        for (auto& inst : work_exit->instructions()) {
            if (!inst || inst->opcode() != ir::Opcode::Phi) continue;
            auto blocks = split_phi_blocks(*inst);
            for (size_t k = 0; k < inst->num_operands() && k < blocks.size(); ++k) {
                if (blocks[k] == loop.header) {
                    inst->set_operand(k, result);
                }
            }
        }

        // The preheader already branches to the header (that's the
        // definition of a preheader). No change needed there.

        // ── Remap operands by name (like deep_copy does) ────────────
        // After rebuilding the header, downstream instructions in the
        // exit block may still hold stale operand pointers to the old
        // loop instructions. Remap everything by name.
        {
            std::unordered_map<std::string, std::shared_ptr<ir::Value>> defs;
            for (auto& arg : work->arguments()) {
                defs[arg.name] = std::make_shared<ir::Value>(arg.type, arg.name);
            }
            for (auto& block : work->blocks()) {
                for (auto& inst : block->instructions()) {
                    if (inst && inst->has_name()) defs[inst->name()] = inst;
                }
            }
            for (auto& block : work->blocks()) {
                for (auto& inst : block->instructions()) {
                    if (!inst) continue;
                    for (size_t i = 0; i < inst->num_operands(); ++i) {
                        auto op = inst->operand(i);
                        if (!op || !op->has_name()) continue;
                        auto it = defs.find(op->name());
                        if (it != defs.end() && it->second != op) {
                            inst->set_operand(i, it->second);
                        }
                    }
                }
            }
        }

        if (!ir::validate_function(*work)) {
            ++stats_.loops_rejected;
            continue;
        }

        ++stats_.loops_expanded;
        return work;
    }

    ++stats_.functions_skipped;
    return nullptr;
}

} // namespace clunk::search
