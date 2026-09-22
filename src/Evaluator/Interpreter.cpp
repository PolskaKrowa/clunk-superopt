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
 * Clunk Interpreter — implementation.
 *
 * See Interpreter.h for scope and semantics. This is a deliberately small
 * AST-walking interpreter; it follows LLVM's integer semantics (poison for
 * flag violations and oversized shifts, UB for division traps) so it can be
 * trusted as a differential-testing oracle.
 */
#include "clunk/Evaluator/Interpreter.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"

namespace clunk::evaluator {

namespace {

using ir::BasicBlock;
using ir::Function;
using ir::Instruction;
using ir::Opcode;
using ir::Type;
using ir::Value;

// Mask off the low `bits` of v, returning a value in [0, 2^bits).
uint64_t mask_unsigned(uint64_t v, unsigned bits) {
    if (bits == 0 || bits >= 64) return v;
    return v & ((uint64_t{1} << bits) - 1);
}

// Sign-extend the low `bits` of v to a 64-bit signed integer.
int64_t sign_extend(uint64_t v, unsigned bits) {
    if (bits == 0 || bits >= 64) return static_cast<int64_t>(v);
    uint64_t m = uint64_t{1} << (bits - 1);
    if (v & m) v |= ~((uint64_t{1} << bits) - 1);
    return static_cast<int64_t>(v);
}

// Canonical form of a value of width `bits`: sign-extended from that width.
int64_t canon(int64_t v, unsigned bits) {
    return sign_extend(mask_unsigned(static_cast<uint64_t>(v), bits), bits);
}

// Resolve the bit-width of a type if it is an integer type.
std::optional<unsigned> integer_bit_width(const Type& t) {
    if (t.is_integer()) return static_cast<unsigned>(t.bit_width());
    return std::nullopt;
}

// Runtime context for a single run() call.
class Context {
public:
    Context(const Function& fn, const std::vector<int64_t>& args)
        : fn_(fn), args_(args), next_handle_(1) {}

    void bind(const Value& v, int64_t val, bool poison = false) {
        values_[key(v)] = val;
        if (poison) poison_.insert(key(v)); else poison_.erase(key(v));
    }

    std::optional<int64_t> lookup(const Value& v) const {
        auto it = values_.find(key(v));
        if (it == values_.end()) return std::nullopt;
        return it->second;
    }

    bool is_poison(const Value& v) const { return poison_.count(key(v)) != 0; }

    int64_t alloc_handle() { return next_handle_++; }

    void store(int64_t handle, int64_t val, bool poison) {
        memory_[handle] = val;
        if (poison) poison_mem_.insert(handle); else poison_mem_.erase(handle);
    }

    int64_t load(int64_t handle) const {
        auto it = memory_.find(handle);
        return it == memory_.end() ? 0 : it->second;
    }

    bool mem_poison(int64_t handle) const { return poison_mem_.count(handle) != 0; }

    const Function& fn() const { return fn_; }
    const std::vector<int64_t>& args() const { return args_; }

private:
    // Key by raw pointer identity: SSA values are unique within a function.
    uintptr_t key(const Value& v) const { return reinterpret_cast<uintptr_t>(&v); }

    const Function& fn_;
    const std::vector<int64_t>& args_;
    int64_t next_handle_;
    std::unordered_map<uintptr_t, int64_t> values_;
    std::unordered_set<uintptr_t> poison_;
    std::unordered_map<int64_t, int64_t> memory_;
    std::unordered_set<int64_t> poison_mem_;
};

// Resolve a Value to a canonical int64_t (constants are canonicalised to
// their own type's width). Returns nullopt on unsupported value kinds.
std::optional<int64_t> resolve(Context& ctx, const Value& v) {
    if (auto ci = dynamic_cast<const ir::ConstantInt*>(&v)) {
        auto bits = ci->type() ? integer_bit_width(*ci->type()) : std::nullopt;
        return bits ? canon(ci->value(), *bits) : ci->value();
    }
    return ctx.lookup(v);
}

bool operand_poison(Context& ctx, const Instruction& inst) {
    for (auto& op : inst.operands())
        if (op && ctx.is_poison(*op)) return true;
    return false;
}

// Apply an integer binary opcode. `lhs`/`rhs` are canonical values of the
// result type's width. Sets `poison` for flag violations / oversized shifts
// and `ub` for division traps. Returns nullopt if unsupported or UB.
std::optional<int64_t> apply_binop(Opcode op, int64_t lhs, int64_t rhs,
                                    const Type& result_type,
                                    const ir::BinOpFlags& fl,
                                    bool& poison, bool& ub) {
    const unsigned bits = integer_bit_width(result_type).value_or(64);
    const uint64_t mask = bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);
    const uint64_t ul = static_cast<uint64_t>(lhs) & mask;   // unsigned views
    const uint64_t ur = static_cast<uint64_t>(rhs) & mask;
    const int64_t  sl = sign_extend(ul, bits);               // signed views
    const int64_t  sr = sign_extend(ur, bits);
    const __int128 smin = -(static_cast<__int128>(1) << (bits - 1));
    const __int128 smax = (static_cast<__int128>(1) << (bits - 1)) - 1;
    const auto signed_overflow = [&](__int128 r) { return r < smin || r > smax; };

    uint64_t res = 0;
    switch (op) {
        case Opcode::Add:
            res = (ul + ur) & mask;
            if (fl.nuw && (static_cast<unsigned __int128>(ul) + ur) > mask) poison = true;
            if (fl.nsw && signed_overflow(static_cast<__int128>(sl) + sr)) poison = true;
            break;
        case Opcode::Sub:
            res = (ul - ur) & mask;
            if (fl.nuw && ul < ur) poison = true;
            if (fl.nsw && signed_overflow(static_cast<__int128>(sl) - sr)) poison = true;
            break;
        case Opcode::Mul:
            res = (ul * ur) & mask;
            if (fl.nuw && (static_cast<unsigned __int128>(ul) * ur) > mask) poison = true;
            if (fl.nsw && signed_overflow(static_cast<__int128>(sl) * sr)) poison = true;
            break;
        case Opcode::And: res = ul & ur; break;
        case Opcode::Or:  res = ul | ur; break;
        case Opcode::Xor: res = ul ^ ur; break;
        case Opcode::Shl:
            if (ur >= bits) { poison = true; break; }
            res = (ul << ur) & mask;
            if (fl.nuw && (res >> ur) != ul) poison = true;
            if (fl.nsw && (sign_extend(res, bits) >> ur) != sl) poison = true;
            break;
        case Opcode::LShr:
            if (ur >= bits) { poison = true; break; }
            res = ul >> ur;
            if (fl.exact && ((res << ur) & mask) != ul) poison = true;
            break;
        case Opcode::AShr:
            if (ur >= bits) { poison = true; break; }
            res = static_cast<uint64_t>(sl >> ur) & mask;
            if (fl.exact && ((res << ur) & mask) != ul) poison = true;
            break;
        case Opcode::UDiv:
            if (ur == 0) { ub = true; return std::nullopt; }
            res = ul / ur;
            if (fl.exact && ul % ur != 0) poison = true;
            break;
        case Opcode::URem:
            if (ur == 0) { ub = true; return std::nullopt; }
            res = ul % ur;
            break;
        case Opcode::SDiv:
        case Opcode::SRem: {
            if (sr == 0 || (static_cast<__int128>(sl) == smin && sr == -1)) { ub = true; return std::nullopt; }
            const int64_t q = sl / sr;   // INT_MIN / -1 was excluded above
            const int64_t r = sl % sr;
            if (op == Opcode::SDiv) {
                res = static_cast<uint64_t>(q) & mask;
                if (fl.exact && r != 0) poison = true;
            } else {
                res = static_cast<uint64_t>(r) & mask;
            }
            break;
        }
        default:
            return std::nullopt;
    }
    return sign_extend(res, bits);
}

// Apply an ICMP predicate. The result is 0 or 1 (i1).
std::optional<int64_t> apply_icmp(const Instruction& inst, int64_t lhs, int64_t rhs) {
    auto it = inst.metadata().find("pred");
    if (it == inst.metadata().end()) return std::nullopt;
    int pred = 0;
    try { pred = std::stoi(it->second); } catch (...) { return std::nullopt; }

    // Operands are canonical (sign-extended). Signed predicates compare them
    // directly; unsigned ones first mask to the operand width.
    unsigned bits = 64;
    if (inst.num_operands() >= 1 && inst.operand(0) && inst.operand(0)->type() &&
        inst.operand(0)->type()->is_integer()) {
        bits = static_cast<unsigned>(inst.operand(0)->type()->bit_width());
    }
    uint64_t ul = mask_unsigned(static_cast<uint64_t>(lhs), bits);
    uint64_t ur = mask_unsigned(static_cast<uint64_t>(rhs), bits);
    int64_t  sl = lhs;
    int64_t  sr = rhs;
    bool result = false;

    using P = ir::CmpPredicate;
    switch (static_cast<P>(pred)) {
        case P::EQ:  result = (sl == sr); break;
        case P::NE:  result = (sl != sr); break;
        case P::UGT: result = (ul >  ur); break;
        case P::UGE: result = (ul >= ur); break;
        case P::ULT: result = (ul <  ur); break;
        case P::ULE: result = (ul <= ur); break;
        case P::SGT: result = (sl >  sr); break;
        case P::SGE: result = (sl >= sr); break;
        case P::SLT: result = (sl <  sr); break;
        case P::SLE: result = (sl <= sr); break;
        default: return std::nullopt; // FP predicates unsupported
    }
    return result ? -1 : 0;   // canonical i1: true is all-ones, like every other width
}

struct ExecResult {
    bool ok = true;           // false = unsupported / fatal
    bool ub = false;          // executed undefined behaviour
    bool is_ret = false;
    bool ret_poison = false;
    int64_t ret_value = 0;
    bool is_br = false;
    std::string next_block;   // empty if not a br
};

ExecResult execute(Context& ctx, const Instruction& inst, const std::string& prev_block) {
    ExecResult r;
    Opcode op = inst.opcode();

    switch (op) {
        // ── Terminators ───────────────────────────────────────────────
        case Opcode::Ret: {
            if (inst.num_operands() == 0) { r.is_ret = true; return r; }
            auto v = resolve(ctx, *inst.operand(0));
            if (!v) { r.ok = false; return r; }
            r.is_ret = true;
            r.ret_value = *v;
            r.ret_poison = ctx.is_poison(*inst.operand(0));
            return r;
        }
        case Opcode::Br: {
            if (inst.num_operands() > 0) {
                auto cond_v = resolve(ctx, *inst.operand(0));
                if (!cond_v) { r.ok = false; return r; }
                if (ctx.is_poison(*inst.operand(0))) { r.ub = true; r.ok = false; return r; }
                bool taken = (*cond_v != 0);
                auto it_true  = inst.metadata().find("true_bb");
                auto it_false = inst.metadata().find("false_bb");
                if (it_true == inst.metadata().end() || it_false == inst.metadata().end()) {
                    r.ok = false; return r;
                }
                r.is_br = true;
                r.next_block = taken ? it_true->second : it_false->second;
                return r;
            }
            auto it = inst.metadata().find("dest_bb");
            if (it == inst.metadata().end()) { r.ok = false; return r; }
            r.is_br = true;
            r.next_block = it->second;
            return r;
        }
        case Opcode::Unreachable:
            r.ok = false; r.ub = true;
            return r;

        // ── Memory ────────────────────────────────────────────────────
        case Opcode::Alloca: {
            ctx.bind(inst, ctx.alloc_handle());
            return r;
        }
        case Opcode::Load: {
            if (inst.num_operands() < 1) { r.ok = false; return r; }
            auto ptr_v = resolve(ctx, *inst.operand(0));
            if (!ptr_v) { r.ok = false; return r; }
            if (ctx.is_poison(*inst.operand(0))) { r.ub = true; r.ok = false; return r; }
            int64_t v = ctx.load(*ptr_v);
            auto bits = integer_bit_width(*inst.type());
            if (bits) v = canon(v, *bits);
            ctx.bind(inst, v, ctx.mem_poison(*ptr_v));
            return r;
        }
        case Opcode::Store: {
            if (inst.num_operands() < 2) { r.ok = false; return r; }
            auto val_v  = resolve(ctx, *inst.operand(0));
            auto ptr_v  = resolve(ctx, *inst.operand(1));
            if (!val_v || !ptr_v) { r.ok = false; return r; }
            if (ctx.is_poison(*inst.operand(1))) { r.ub = true; r.ok = false; return r; }
            ctx.store(*ptr_v, *val_v, ctx.is_poison(*inst.operand(0)));
            return r;
        }

        // ── Integer binary ops ────────────────────────────────────────
        case Opcode::Add: case Opcode::Sub: case Opcode::Mul:
        case Opcode::UDiv: case Opcode::SDiv:
        case Opcode::URem: case Opcode::SRem:
        case Opcode::And: case Opcode::Or:  case Opcode::Xor:
        case Opcode::Shl: case Opcode::LShr: case Opcode::AShr: {
            if (inst.num_operands() < 2) { r.ok = false; return r; }
            auto lhs = resolve(ctx, *inst.operand(0));
            auto rhs = resolve(ctx, *inst.operand(1));
            if (!lhs || !rhs) { r.ok = false; return r; }
            if (!inst.type()) { r.ok = false; return r; }
            bool poison = operand_poison(ctx, inst), ub = false;
            // A poison divisor is UB; a poison anything-else just propagates.
            if (poison && (op == Opcode::UDiv || op == Opcode::SDiv ||
                           op == Opcode::URem || op == Opcode::SRem)) {
                if (ctx.is_poison(*inst.operand(1))) { r.ub = true; r.ok = false; return r; }
            }
            auto v = apply_binop(op, *lhs, *rhs, *inst.type(), inst.binop_flags(), poison, ub);
            if (ub) { r.ub = true; r.ok = false; return r; }
            if (!v) { r.ok = false; return r; }
            ctx.bind(inst, *v, poison);
            return r;
        }

        // ── Compare ───────────────────────────────────────────────────
        case Opcode::ICmp: {
            if (inst.num_operands() < 2) { r.ok = false; return r; }
            auto lhs = resolve(ctx, *inst.operand(0));
            auto rhs = resolve(ctx, *inst.operand(1));
            if (!lhs || !rhs) { r.ok = false; return r; }
            auto v = apply_icmp(inst, *lhs, *rhs);
            if (!v) { r.ok = false; return r; }
            ctx.bind(inst, *v, operand_poison(ctx, inst));
            return r;
        }

        // ── Select ────────────────────────────────────────────────────
        case Opcode::Select: {
            if (inst.num_operands() < 3) { r.ok = false; return r; }
            auto cond = resolve(ctx, *inst.operand(0));
            auto tv   = resolve(ctx, *inst.operand(1));
            auto fv   = resolve(ctx, *inst.operand(2));
            if (!cond || !tv || !fv) { r.ok = false; return r; }
            const bool cond_poison = ctx.is_poison(*inst.operand(0));
            const bool taken = *cond != 0;
            // Only the CHOSEN arm's poison matters (LangRef).
            const bool arm_poison = ctx.is_poison(*inst.operand(taken ? 1 : 2));
            ctx.bind(inst, taken ? *tv : *fv, cond_poison || arm_poison);
            return r;
        }

        // ── Phi ───────────────────────────────────────────────────────
        case Opcode::Phi: {
            auto it = inst.metadata().find("phi_blocks");
            if (it == inst.metadata().end()) { r.ok = false; return r; }
            std::vector<std::string> blocks;
            {
                std::string cur;
                for (char c : it->second) {
                    if (c == ',') { blocks.push_back(cur); cur.clear(); }
                    else cur.push_back(c);
                }
                if (!cur.empty()) blocks.push_back(cur);
            }
            for (size_t i = 0; i < blocks.size() && i < inst.num_operands(); ++i) {
                if (blocks[i] == prev_block) {
                    auto v = resolve(ctx, *inst.operand(i));
                    if (!v) { r.ok = false; return r; }
                    ctx.bind(inst, *v, ctx.is_poison(*inst.operand(i)));
                    return r;
                }
            }
            // No matching incoming block — pick operand 0 as a fallback.
            if (inst.num_operands() >= 1) {
                auto v = resolve(ctx, *inst.operand(0));
                if (!v) { r.ok = false; return r; }
                ctx.bind(inst, *v, ctx.is_poison(*inst.operand(0)));
                return r;
            }
            r.ok = false;
            return r;
        }

        // ── Casts (integer-only) ──────────────────────────────────────
        case Opcode::Trunc: case Opcode::ZExt: case Opcode::SExt:
        case Opcode::BitCast: case Opcode::PtrToInt: case Opcode::IntToPtr: {
            if (inst.num_operands() < 1) { r.ok = false; return r; }
            auto v = resolve(ctx, *inst.operand(0));
            if (!v) { r.ok = false; return r; }
            auto bits = integer_bit_width(*inst.type());
            if (!bits) { r.ok = false; return r; }
            const bool poison = ctx.is_poison(*inst.operand(0));
            if (op == Opcode::Trunc) {
                ctx.bind(inst, canon(*v, *bits), poison);
            } else if (op == Opcode::ZExt) {
                auto src_ty = inst.operand(0)->type();
                auto src_bits = src_ty ? integer_bit_width(*src_ty) : std::nullopt;
                if (!src_bits || *src_bits >= *bits) { r.ok = false; return r; }
                ctx.bind(inst, static_cast<int64_t>(
                    mask_unsigned(static_cast<uint64_t>(*v), *src_bits)), poison);
            } else { // SExt, BitCast, PtrToInt, IntToPtr
                ctx.bind(inst, *v, poison);
            }
            return r;
        }

        default:
            r.ok = false;   // floats, calls, vectors, ...
            return r;
    }
}

} // namespace

ExecOutcome Interpreter::run(const ir::Function& fn, const std::vector<int64_t>& args) {
    ExecOutcome out;   // Unsupported by default

    if (args.size() != fn.argument_count()) return out;
    if (fn.blocks().empty()) return out;

    // Refuse to interpret non-integer return types.
    auto ret_ty = fn.return_type();
    if (ret_ty && !ret_ty->is_integer() && !ret_ty->is_void()) return out;

    // Canonicalise arguments to each parameter's width.
    std::vector<int64_t> canon_args(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        const auto& t = fn.arguments()[i].type;
        if (t && t->is_pointer()) { canon_args[i] = args[i]; continue; }   // memory handle
        if (!t || !t->is_integer()) return out;                              // float / vector parameter
        canon_args[i] = canon(args[i], static_cast<unsigned>(t->bit_width()));
    }

    Context ctx(fn, canon_args);

    auto resolve_argument = [&](const ir::Value& v) -> std::optional<int64_t> {
        if (!v.has_name()) return std::nullopt;
        const auto& fa = fn.arguments();
        for (size_t i = 0; i < fa.size(); ++i)
            if (fa[i].name == v.name()) return canon_args[i];
        return std::nullopt;
    };

    std::string current_block_name = fn.blocks().front()->name();
    std::string prev_block_name;
    size_t hops = 0;

    while (hops < kMaxBlockHops) {
        ++hops;
        auto bb = fn.block(current_block_name);
        if (!bb) return out;

        bool branched = false;
        for (auto& inst : bb->instructions()) {
            // Pre-bind any operands that are arguments.
            for (auto& op : inst->operands()) {
                if (op && op->has_name() && !dynamic_cast<const ir::ConstantInt*>(op.get())) {
                    if (!ctx.lookup(*op)) {
                        auto av = resolve_argument(*op);
                        if (av) ctx.bind(*op, *av);
                    }
                }
            }

            ExecResult er = execute(ctx, *inst, prev_block_name);
            if (er.ub) { out.status = ExecStatus::UB; return out; }
            if (!er.ok) return out;
            if (er.is_ret) {
                out.value = er.ret_value;
                if (ret_ty && ret_ty->is_integer() && ret_ty->bit_width() == 1)
                    out.value = out.value != 0 ? 1 : 0;   // i1 is reported as 0/1
                out.status = er.ret_poison ? ExecStatus::Poison : ExecStatus::Value;
                return out;
            }
            if (er.is_br) {
                prev_block_name = current_block_name;
                current_block_name = er.next_block;
                branched = true;
                break;
            }
        }
        if (!branched) return out;   // block ended without a terminator
    }
    return out;   // exceeded hop budget
}

std::optional<int64_t> Interpreter::interpret(
    const ir::Function& fn,
    const std::vector<int64_t>& args) {
    auto r = run(fn, args);
    if (r.status == ExecStatus::Value || r.status == ExecStatus::Poison) return r.value;
    return std::nullopt;
}

} // namespace clunk::evaluator
