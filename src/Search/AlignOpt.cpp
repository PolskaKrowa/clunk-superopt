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
 * Clunk AlignOpt — see include/clunk/Search/AlignOpt.h for the soundness contract.
 */
#include "clunk/Search/AlignOpt.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "clunk/Analysis/KnownBits.h"
#include "clunk/IR/Clone.h"

namespace clunk::search {

namespace {

using ir::Opcode;
using ir::TypeID;
using ValuePtr = std::shared_ptr<ir::Value>;
using KnownEnv = std::unordered_map<std::string, analysis::KnownBits>;

constexpr uint64_t kTop = uint64_t(1) << 32;  // "unconstrained": fixpoint start and cap on every product
constexpr uint64_t kStackAlign = 16;          // SysV x86-64 (datalayout S128)

uint64_t pow2_div(uint64_t v) { return v ? std::min(v & (0 - v), kTop) : kTop; }
bool is_pow2(uint64_t v) { return v && !(v & (v - 1)); }
uint64_t mul_cap(uint64_t a, uint64_t b) { return a >= kTop || b >= kTop ? kTop : std::min(a * b, kTop); }

// A usable alignment fact: a power of two, else no fact at all (1).
uint64_t pow2_or_1(uint64_t v) { return is_pow2(v) && v < kTop ? v : 1; }

// Leading decimal digits of `s` from `i` (0 when there are none); `end` gets the stop index.
uint64_t digits(const std::string& s, size_t i, size_t& end) {
    uint64_t v = 0;
    for (end = i; end < s.size() && s[end] >= '0' && s[end] <= '9' && v < kTop; ++end) v = v * 10 + uint64_t(s[end] - '0');
    return v;
}

// The parser never fills GlobalValue::alignment: `, align N` rides at the tail of init_value.
uint64_t global_align(const ir::GlobalValue& g) {
    if (g.alignment) return pow2_or_1(g.alignment);
    static const std::string key = ", align ";
    const auto pos = g.init_value.rfind(key);
    if (pos == std::string::npos) return 1;
    size_t end;
    const uint64_t v = digits(g.init_value, pos + key.size(), end);
    return end == pos + key.size() || (end < g.init_value.size() && g.init_value[end] != ',') ? 1 : pow2_or_1(v);
}

// A scalar whose size_bytes() is exact up to power-of-two rounding.
bool byte_scalar(const ir::Type& t) {
    switch (t.type_id()) {
        case TypeID::Float: case TypeID::Double: case TypeID::Pointer: return true;
        case TypeID::Integer: return t.bit_width() % 8 == 0;
        default: return false;  // i1 & friends are bit-packed inside vectors
    }
}

// ABI alignment of a fixed vector of byte-sized scalars with a power-of-two
// size (which is its size); 0 for anything else.
uint64_t vector_align(const ir::Type& t) {
    if (!t.is_vector()) return 0;
    const auto& v = static_cast<const ir::VectorType&>(t);
    return byte_scalar(*v.element_type()) && is_pow2(v.size_bytes()) ? v.size_bytes() : 0;
}

// ABI alignment for the types an alloca's users may access; 0 = not modelled.
uint64_t abi_align(const ir::Type& t) {
    if (byte_scalar(t)) return is_pow2(t.size_bytes()) ? t.size_bytes() : 0;
    return vector_align(t);
}

// Largest power of two dividing the alloc size of `t`; 1 when we cannot vouch
// for it (structs: size_bytes() ignores their padding).
uint64_t stride_align(const ir::Type& t) {
    if (byte_scalar(t)) return pow2_div(t.size_bytes());
    if (t.type_id() == TypeID::Array) {
        const auto& a = static_cast<const ir::ArrayType&>(t);
        return mul_cap(pow2_div(a.count()), stride_align(*a.element_type()));
    }
    if (t.is_vector()) {
        const auto& v = static_cast<const ir::VectorType&>(t);
        return byte_scalar(*v.element_type()) ? pow2_div(v.size_bytes()) : 1;
    }
    return 1;
}

bool is_ptr_transparent(const ir::Instruction& i) {
    if (!i.type() || !i.type()->is_pointer()) return false;
    const auto op = i.opcode();
    return op == Opcode::GetElementPtr || op == Opcode::BitCast || op == Opcode::Phi || op == Opcode::Select;
}

// Integer defs whose divisibility we track through the fixpoint (so that
// induction variables such as `i += 4` are understood; KnownBits gives up on
// loop-carried phis).
bool is_int_tracked(const ir::Instruction& i) {
    if (!i.type() || !i.type()->is_integer()) return false;
    switch (i.opcode()) {
        case Opcode::Add: case Opcode::Sub: case Opcode::Mul: case Opcode::Shl: case Opcode::And:
        case Opcode::Or: case Opcode::Xor: case Opcode::ZExt: case Opcode::SExt: case Opcode::Trunc:
        case Opcode::Phi: case Opcode::Select: return true;
        default: return false;
    }
}

// Provable facts for every value in one function, from one greatest fixpoint:
//   pointers → guaranteed alignment in bytes,
//   integers → the power of two that always divides the value.
// Both are "the largest power of two dividing X", so they share a lattice.
class PtrAlign {
public:
    PtrAlign(const ir::Function& fn, const ir::Module& mod)
        : kb_(analysis::analyse_known_bits(fn)) {
        for (const auto& g : mod.globals())  // Module stores "@name"; operands carry "name"
            globals_[!g.name.empty() && g.name[0] == '@' ? g.name.substr(1) : g.name] = global_align(g);
        for (const auto& a : fn.arguments())
            if (a.type && a.type->is_pointer() && !a.name.empty())
                if (auto it = a.attrs.find("align"); it != a.attrs.end()) {
                    size_t end;
                    args_[a.name] = pow2_or_1(digits(it->second, 0, end));
                }
        for (const auto& bb : fn.blocks())
            for (const auto& inst : bb->instructions())
                if (inst && inst->has_name()) {
                    defs_[inst->name()] = inst.get();
                    if (is_ptr_transparent(*inst) || is_int_tracked(*inst)) al_[inst->name()] = kTop;
                }
        // The parser (and deep_copy_function) resolve operands by bare name,
        // so a local `%x` silently captures a global `@x`. With such a
        // collision the def-use graph cannot be trusted: prove nothing.
        for (const auto& [name, def] : defs_)
            if (globals_.count(name)) poisoned_ = true;
        // Greatest fixpoint: values only ever decrease, so this terminates,
        // and it is the strongest inductive invariant (a loop-carried pointer
        // or index phi keeps whatever its entry value and its step preserve).
        for (bool changed = true; changed;) {
            changed = false;
            for (auto& [name, cur] : al_) {
                const uint64_t next = transfer(*defs_.at(name));
                if (next < cur) { cur = next; changed = true; }
            }
        }
    }

    uint64_t of(const ValuePtr& v) const {
        if (poisoned_ || !v) return 1;
        if (const auto* c = dynamic_cast<const ir::ConstantInt*>(v.get()))
            return pow2_div(static_cast<uint64_t>(c->value()));  // -k shares k's trailing zeros
        if (!v->has_name()) return 1;
        if (v->is_global()) {
            auto it = globals_.find(v->name());
            return it == globals_.end() ? 1 : it->second;
        }
        if (auto d = defs_.find(v->name()); d != defs_.end()) {
            if (d->second->opcode() == Opcode::Alloca) return pow2_or_1(d->second->alignment().value_or(0));
            auto it = al_.find(v->name());
            return it == al_.end() ? kb_align(v->name()) : std::max(it->second, kb_align(v->name()));
        }
        auto it = args_.find(v->name());
        return it == args_.end() ? 1 : it->second;
    }

private:
    // Trailing known-zero bits from the (loop-blind) KnownBits pass.
    uint64_t kb_align(const std::string& name) const {
        auto it = kb_.find(name);
        if (it == kb_.end()) return 1;
        const uint64_t not_known_zero = ~it->second.zero;
        return not_known_zero ? std::min(kTop, uint64_t(1) << __builtin_ctzll(not_known_zero)) : kTop;
    }

    // A power of two dividing the byte offset of `g` from its base. The
    // parser records the GEP's SOURCE element type as its result pointee.
    uint64_t gep_offset_align(const ir::Instruction& g) const {
        const ir::Type* ty = static_cast<const ir::PointerType&>(*g.type()).pointee().get();
        if (!ty || g.num_operands() < 1) return 1;
        uint64_t a = kTop;
        for (size_t i = 1; i < g.num_operands(); ++i) {
            if (i > 1) {  // step into the aggregate `ty`
                if (ty->type_id() == TypeID::Array)
                    ty = static_cast<const ir::ArrayType&>(*ty).element_type().get();
                else if (ty->is_vector())
                    ty = static_cast<const ir::VectorType&>(*ty).element_type().get();
                else
                    return 1;  // struct field offsets need padding rules we do not model
            }
            a = std::min(a, mul_cap(of(g.operand(i)), stride_align(*ty)));
        }
        return a;
    }

    uint64_t transfer(const ir::Instruction& i) const {
        // Malformed instructions (e.g. an operand-less `shl` from an unverified search) prove nothing.
        auto val = [&](size_t k) { return k < i.num_operands() ? i.operand(k) : nullptr; };
        auto op = [&](size_t k) { return of(val(k)); };
        switch (i.opcode()) {
            case Opcode::GetElementPtr: return std::min(op(0), gep_offset_align(i));
            case Opcode::BitCast: case Opcode::ZExt: case Opcode::SExt: case Opcode::Trunc: return op(0);
            case Opcode::Add: case Opcode::Sub: case Opcode::Or: case Opcode::Xor: return std::min(op(0), op(1));
            case Opcode::Mul: return mul_cap(op(0), op(1));
            case Opcode::And: return std::max(op(0), op(1));  // a zero bit in either operand is a zero bit in the result
            case Opcode::Shl: {
                const auto* c = dynamic_cast<const ir::ConstantInt*>(val(1).get());
                return c && c->value() >= 0 && c->value() <= 32 ? mul_cap(op(0), uint64_t(1) << c->value()) : op(0);
            }
            default: {                  // Phi / Select: the weakest incoming value
                uint64_t a = kTop;
                for (size_t k = i.opcode() == Opcode::Select ? 1 : 0; k < i.num_operands(); ++k) a = std::min(a, op(k));
                return a;
            }
        }
    }

    KnownEnv kb_;
    bool poisoned_ = false;
    std::unordered_map<std::string, const ir::Instruction*> defs_;
    std::unordered_map<std::string, uint64_t> al_, globals_, args_;
};

// What an alloca's users need from it. `ok` is false when the address may be
// observed in any way other than through a load/store address.
struct AllocaUse {
    bool ok = true;
    uint64_t need = 1;
    std::vector<std::pair<ValuePtr, uint64_t>> accesses;  // (address, alignment the access requires)
};

AllocaUse scan_alloca(const ir::Function& fn, const std::string& root) {
    AllocaUse r;
    std::unordered_set<std::string> derived = {root};
    for (bool grew = true; grew;) {  // GEP/bitcast closure
        grew = false;
        for (const auto& bb : fn.blocks())
            for (const auto& inst : bb->instructions()) {
                if (!inst || !inst->has_name() || derived.count(inst->name())) continue;
                if (inst->opcode() != Opcode::GetElementPtr && inst->opcode() != Opcode::BitCast) continue;
                const auto b = inst->num_operands() ? inst->operand(0) : nullptr;
                if (b && b->has_name() && !b->is_global() && derived.count(b->name())) {
                    derived.insert(inst->name());
                    grew = true;
                }
            }
    }
    auto is_derived = [&](const ValuePtr& v) {
        return v && v->has_name() && !v->is_global() && derived.count(v->name());
    };
    for (const auto& bb : fn.blocks())
        for (const auto& inst : bb->instructions()) {
            if (!inst) continue;
            const auto op = inst->opcode();
            for (size_t i = 0; i < inst->num_operands(); ++i) {
                if (!is_derived(inst->operand(i))) continue;
                if ((op == Opcode::GetElementPtr || op == Opcode::BitCast) && i == 0) continue;
                if (op == Opcode::Call) {
                    const auto it = inst->metadata().find("callee");
                    if (it != inst->metadata().end() && it->second.find("llvm.lifetime.") != std::string::npos) continue;
                }
                const bool load = op == Opcode::Load && i == 0, store = op == Opcode::Store && i == 1;
                if (!load && !store) { r.ok = false; return r; }  // escapes, or is stored/compared/merged
                const ir::Type& ty = load ? *inst->type() : *inst->operand(0)->type();
                const uint64_t req = inst->alignment().value_or(0) ? *inst->alignment() : abi_align(ty);
                if (!req) { r.ok = false; return r; }  // aggregate access: layout not modelled
                r.need = std::max(r.need, req);
                r.accesses.emplace_back(inst->operand(i), req);
            }
        }
    return r;
}

// The address operand and accessed type of a load/store, else {nullptr, nullptr}.
std::pair<ValuePtr, const ir::Type*> access_of(const ir::Instruction& i) {
    if (i.opcode() == Opcode::Load && i.num_operands() >= 1) return {i.operand(0), i.type().get()};
    if (i.opcode() == Opcode::Store && i.num_operands() >= 2) return {i.operand(1), i.operand(0)->type().get()};
    return {nullptr, nullptr};
}

} // anonymous namespace

std::shared_ptr<ir::Function> AlignOptimizer::optimize(const ir::Function& fn, const ir::Module& mod) {
    if (fn.blocks().empty()) return nullptr;
    auto work = ir::deep_copy_function(fn);
    for (const auto& a : fn.function_attributes()) work->add_function_attribute(a);  // dropped by deep_copy_function
    bool changed = false;

    // 1. Relax over-aligned, non-escaping allocas (removes `and rsp, -N`).
    for (const auto& bb : work->blocks())
        for (const auto& inst : bb->instructions()) {
            if (!inst || inst->opcode() != Opcode::Alloca || !inst->has_name()) continue;
            const uint64_t old_align = inst->alignment().value_or(0);
            if (old_align <= kStackAlign) continue;
            const auto use = scan_alloca(*work, inst->name());
            const uint64_t target = std::max(kStackAlign, use.need);
            if (!use.ok || target >= old_align) continue;
            inst->set_alignment(static_cast<unsigned>(target));
            const PtrAlign after(*work, mod);
            const bool still_proven = std::all_of(use.accesses.begin(), use.accesses.end(),
                [&](const auto& a) { return after.of(a.first) >= a.second; });
            if (still_proven) { ++stats_.allocas_lowered; changed = true; }
            else inst->set_alignment(static_cast<unsigned>(old_align));  // an access leaned on the old alignment
        }

    // 2. Raise vector accesses to the alignment we can prove (vmovupd -> vmovapd).
    const PtrAlign al(*work, mod);
    for (const auto& bb : work->blocks())
        for (const auto& inst : bb->instructions()) {
            if (!inst) continue;
            const auto [ptr, ty] = access_of(*inst);
            if (!ty) continue;
            const uint64_t natural = vector_align(*ty);
            if (!natural) continue;
            const uint64_t current = inst->alignment().value_or(0) ? *inst->alignment() : natural;  // omitted = ABI
            const uint64_t proven = std::min(natural, al.of(ptr));
            if (proven > current) {
                inst->set_alignment(static_cast<unsigned>(proven));
                ++stats_.accesses_raised;
                changed = true;
            }
        }

    return changed ? work : nullptr;
}

} // namespace clunk::search
