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
 * Clunk Interpreter — a small AST-walking interpreter for the clunk IR.
 *
 * Scope: a sanity oracle for tests, NOT a production execution engine.
 *   - Single function (no calls).
 *   - Integer types only (i1, i8, i16, i32, i64).
 *   - Operands resolve to int64_t internally; results are masked to
 *     the instruction's result-type bit width.
 *   - Supports: ConstantInt, function arguments, alloca/load/store
 *     (via a simple handle → value map), Add/Sub/Mul/UDiv/SDiv/URem/SRem/
 *     And/Or/Xor/Shl/LShr/AShr, ICmp, Select, Br (cond + uncond),
 *     Ret.
 *   - Phi nodes pick the incoming value from the *previously executed*
 *     block (best-effort; works for the typical test shapes).
 *   - Floats, calls, GEP into struct/array, vector ops, etc. are
 *     unsupported and return std::nullopt.
 *
 * Semantics follow LLVM's LangRef, not C++'s:
 *   - values are canonicalised to their type's width (arguments and
 *     constants included), so an i8 argument of 255 is -1;
 *   - unsigned ops (lshr/udiv/urem/ult...) see the width-masked value;
 *   - `nsw`/`nuw`/`exact` violations and shift amounts >= the bit width
 *     yield POISON; division by zero, INT_MIN / -1, `unreachable` and a
 *     branch on poison are UNDEFINED BEHAVIOUR.
 * run() reports all four outcomes; interpret() keeps its historical
 * shape (poison yields the wrapped value; UB and unsupported are nullopt).
 *
 * The interpreter is intentionally side-effect-free beyond its own
 * internal state: it does not modify the IR, does not allocate OS
 * resources, and is safe to call concurrently on independent
 * ir::Function instances.
 */
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "clunk/IR/Function.h"

namespace clunk::evaluator {

// What happened when a function was run on concrete inputs.
enum class ExecStatus {
    Value,        // returned a fully defined value
    Poison,       // returned poison (`value` is the wrapped result, meaningless)
    UB,           // executed undefined behaviour (div by zero, branch on poison, ...)
    Unsupported   // uses something we cannot model, or exceeded the hop cap
};

struct ExecOutcome {
    ExecStatus status = ExecStatus::Unsupported;
    int64_t value = 0;   // canonical: sign-extended from the return type's width
};

class Interpreter {
public:
    // Run `fn` on `args` (one per parameter; canonicalised to each
    // parameter's width). Never throws.
    static ExecOutcome run(const ir::Function& fn, const std::vector<int64_t>& args);

    // Evaluate `fn` on the supplied integer argument list. Returns
    // std::nullopt if the function is unsupported (non-integer ops,
    // calls, infinite loop guard tripped, …) or the argument count
    // doesn't match `fn.argument_count()`.
    //
    // `args.size()` must equal `fn.argument_count()`.
    static std::optional<int64_t> interpret(
        const ir::Function& fn,
        const std::vector<int64_t>& args);

private:
    // Cap on dynamic block transitions to prevent runaway loops in
    // the oracle. Generous enough for all existing examples/simple_add.ll
    // and the test shapes; tight enough to fail fast on a broken input.
    static constexpr size_t kMaxBlockHops = 1024;
};

} // namespace clunk::evaluator
