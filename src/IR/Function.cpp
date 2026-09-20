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
 * Clunk IR Function — implementation of non-inline methods.
 */
#include "clunk/IR/Function.h"

namespace clunk::ir {

// ── Helper: linkage keyword ──────────────────────────────────────────────
static const char* linkage_name(Linkage l) {
    switch (l) {
        case Linkage::External:           return "";
        case Linkage::Internal:           return "internal ";
        case Linkage::Private:            return "private ";
        case Linkage::LinkOnce:           return "linkonce ";
        case Linkage::Weak:               return "weak ";
        case Linkage::Common:             return "common ";
        case Linkage::Appending:          return "appending ";
        case Linkage::ExternalWeak:       return "extern_weak ";
        case Linkage::LinkOnceODR:        return "linkonce_odr ";
        case Linkage::WeakODR:            return "weak_odr ";
        case Linkage::AvailableExternally:return "available_externally ";
    }
    return "";
}

// ── compute_predecessors() ───────────────────────────────────────────────
void Function::compute_predecessors() {
    for (auto& bb : blocks_) {
        bb->clear_predecessors();
    }
    for (const auto& bb : blocks_) {
        auto succs = bb->successors();
        for (const auto& succ_name : succs) {
            auto succ_bb = block(succ_name);
            if (succ_bb) {
                succ_bb->add_predecessor(bb->name());
            }
        }
    }
}

// ── instruction_count() ─────────────────────────────────────────────────
size_t Function::instruction_count() const {
    size_t count = 0;
    for (const auto& bb : blocks_) {
        count += bb->size();
    }
    return count;
}

// ── to_string() ─────────────────────────────────────────────────────────
std::string Function::to_string() const {
    std::string s;
    s.reserve(256);

    // "define [linkage] <return_type> @<name>(<args>)<attrs> {"
    //
    // LLVM IR syntax places the linkage keyword AFTER "define", not
    // before it (e.g. `define internal i32 @f(...)`, never
    // `internal define i32 @f(...)`). Emitting it in front of "define"
    // produces text that neither clunk's own parser (see
    // IRParser::parse_function_def, which looks for linkage keywords
    // right after "define"), nor clang/opt, nor Alive2's `alive-tv` can
    // parse — see AliveVerifier and scripts/diff_test_alive.sh, which
    // feed this output straight to alive-tv as the "target" module.
    s.append("define ");
    s.append(linkage_name(linkage_));
    s.append(fn_type_->return_type()->to_string());
    s.append(" @");
    s.append(name_);
    s.append("(");

    // Arguments
    for (size_t i = 0; i < args_.size(); ++i) {
        if (i > 0) s.append(", ");
        s.append(args_[i].type->to_string());
        // Keep `align N`: AlignOpt's upgrades are justified by it.
        if (auto a = args_[i].attrs.find("align"); a != args_[i].attrs.end()) s.append(" align " + a->second);
        if (!args_[i].name.empty()) {
            s.append(" %");
            s.append(args_[i].name);
        }
    }

    // Vararg
    if (fn_type_->is_vararg()) {
        if (!args_.empty()) s.append(", ");
        s.append("...");
    }

    s.append(")");

    // Function attributes
    for (const auto& attr : function_attributes_) {
        s.append(" ");
        s.append(attr);
    }

    s.append(" {\n");

    // Basic blocks
    for (const auto& bb : blocks_) {
        s.append(bb->to_string());
    }

    s.append("}\n");
    return s;
}

} // namespace clunk::ir
