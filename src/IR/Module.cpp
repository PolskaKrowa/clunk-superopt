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
 * Clunk IR Module — implementation of non-inline methods.
 */
#include "clunk/IR/Module.h"

#include <algorithm>
#include <vector>

namespace clunk::ir {

// ── Helper: linkage keyword for globals/declarations ────────────────────
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

// ── instruction_count() ─────────────────────────────────────────────────
// Sum of all instructions across every function.
size_t Module::instruction_count() const {
    size_t count = 0;
    for (const auto& fn : functions_) {
        count += fn->instruction_count();
    }
    return count;
}

// ── to_string() ─────────────────────────────────────────────────────────
// Render the complete module as LLVM IR text:
//
//   ; ModuleID = 'module_name'
//   source_filename = "..."
//   target triple = "x86_64-unknown-linux-gnu"
//   target datalayout = "e-m:e-..."
//
//   !llvm.module.flags = !{ ... }
//
//   declare i32 @external_func(i32)
//
//   define i32 @my_func(i32 %x) {
//   ...
//   }
//
std::string Module::to_string() const {
    std::string s;
    s.reserve(1024);

    // Module header
    s.append("; ModuleID = '");
    s.append(name_);
    s.append("'\n");

    // Source filename
    if (!source_filename_.empty()) {
        s.append("source_filename = \"");
        s.append(source_filename_);
        s.append("\"\n");
    }

    // Target information
    if (has_target()) {
        if (!target_.triple.empty()) {
            s.append("target triple = \"");
            s.append(target_.triple);
            s.append("\"\n");
        }
        if (!target_.datalayout.empty()) {
            s.append("target datalayout = \"");
            s.append(target_.datalayout);
            s.append("\"\n");
        }
    }

    // Module flags — required for clang to determine PIC mode, code model, etc.
    // LLVM IR syntax:
    //   !0 = !{ i32 1, !"PIC Level", i32 2 }
    //   !llvm.module.flags = !{ !0, !1 }
    //
    // We emit the flag entries with a leading space inside the braces for
    // readability, matching clang's output style.  The collection line uses
    // `!{ !0, !1 }` (spaces between elements, no space before `}`).
    if (!module_flags_.empty()) {
        s.append("\n");
        for (size_t i = 0; i < module_flags_.size(); ++i) {
            const auto& mf = module_flags_[i];
            s.append("!");
            s.append(std::to_string(i));
            s.append(" = !{ i32 ");
            s.append(std::to_string(mf.behavior));
            s.append(", !\"");
            s.append(mf.key);
            s.append("\", ");
            s.append(mf.value);
            s.append(" }\n");
        }
        s.append("!llvm.module.flags = !{");
        for (size_t i = 0; i < module_flags_.size(); ++i) {
            if (i > 0) s.append(", ");
            s.append("!");   // clang emits `!{!0, !1}` — no space after `{`
            s.append(std::to_string(i));
        }
        s.append("}\n");
    }

    // ── Round-trip preservation: emit captured constructs verbatim ──────
    //
    // Order matters.
    //
    //   1. module asm "..."        (top of file, after module flags)
    //   2. attributes #N = { ... } (after module asm)
    //   3. !name = !{...}          (named metadata, before numbered defs)
    //   4. !N = !{...}             (numbered metadata defs, at end of file)
    //
    // Numbered metadata defs go at the very end because that's where clang
    // emits them and many tools (including older versions of clunk's own
    // parser) expect them there.

    // 1. Module-level inline assembly
    for (const auto& asm_line : module_asm_) {
        s.append("module asm ");
        s.append(asm_line);
        s.append("\n");
    }

    // 2. Attribute groups (must appear before any function that references them)
    // Emit in numeric-id order so output is stable across runs.
    // We sort by parsing the numeric part of "#N" — IDs without a numeric
    // suffix come first lexicographically.
    {
        std::vector<std::pair<std::string, std::string>> sorted_groups(
            attribute_groups_.begin(), attribute_groups_.end());
        std::sort(sorted_groups.begin(), sorted_groups.end(),
            [](const std::pair<std::string, std::string>& a,
               const std::pair<std::string, std::string>& b) {
                // Extract the numeric part of "#N" for comparison.
                auto numeric_part = [](const std::string& id) -> long {
                    if (id.size() < 2 || id[0] != '#') return -1;
                    try { return std::stol(id.substr(1)); }
                    catch (...) { return -1; }
                };
                long na = numeric_part(a.first);
                long nb = numeric_part(b.first);
                if (na >= 0 && nb >= 0) return na < nb;
                return a.first < b.first;
            });
        if (!sorted_groups.empty()) {
            s.append("\n");
        }
        for (const auto& [id, body] : sorted_groups) {
            s.append("attributes ");
            s.append(id);
            s.append(" = { ");
            s.append(body);
            s.append(" }\n");
        }
    }

    // Named types
    for (const auto& [tname, ty] : named_types_) {
        s.append("%");
        s.append(tname);
        s.append(" = type ");
        s.append(ty->to_string());
        s.append("\n");
    }

    // Global variables — emit in canonical LLVM IR form:
    //   @name = [linkage] [constant|global] [type] [init_value] [, align N]
    // Declarations (no initializer, e.g. `@g = external global i32`) are
    // handled specially: LLVM requires the `external`/`extern_weak`
    // keyword whenever there's no initializer value, and the type must
    // NOT be followed by a bare `,` with nothing in between — both of
    // those produce a parse error ("expected value token").
    for (const auto& gv : globals_) {
        s.append(gv.name);
        s.append(" = ");
        if (gv.is_declaration) {
            // `external` has no dedicated Linkage enum value distinct
            // from a normal externally-linked definition (linkage_name()
            // prints "" for Linkage::External, since a *defined* global
            // needs no keyword) — so for declarations we print "external"
            // explicitly unless the linkage is itself a keyword that
            // already implies "no initializer" (e.g. extern_weak).
            if (gv.linkage == Linkage::External) {
                s.append("external ");
            } else {
                s.append(linkage_name(gv.linkage));
            }
        } else {
            s.append(linkage_name(gv.linkage));
        }
        if (gv.is_constant) {
            s.append("constant ");
        } else {
            s.append("global ");
        }
        if (gv.type) {
            s.append(gv.type->to_string());
        } else {
            s.append("ptr");  // fallback if type was lost during parsing
        }
        if (!gv.init_value.empty()) {
            // A captured init_value that starts with ',' is trailing
            // attributes only (align/section/...) with no actual
            // initializer before it — happens for declarations — and
            // must NOT get an extra separating space (that space is what
            // produced the invalid `ptr , align 8` text).
            if (gv.init_value.front() != ',') {
                s += ' ';
            }
            s.append(gv.init_value);
        }
        if (gv.alignment > 0) {
            s.append(", align ");
            s.append(std::to_string(gv.alignment));
        }
        s.append("\n");
    }

    // Separate globals from functions with a blank line if both exist
    if (!globals_.empty() && !functions_.empty()) {
        s.append("\n");
    }

    // Functions — declarations (no basic blocks) vs definitions
    for (const auto& fn : functions_) {
        if (fn->blocks().empty()) {
            // Declaration (external, no body)
            s.append("declare ");
            s.append(fn->return_type()->to_string());
            s.append(" @");
            s.append(fn->name());
            s.append("(");
            const auto& args = fn->arguments();
            for (size_t i = 0; i < args.size(); ++i) {
                if (i > 0) s.append(", ");
                // Emit parameter attributes that round-trip through the
                // parser's collect_decl_param_attrs().  We emit them
                // BEFORE the type, matching LLVM IR convention.
                for (const auto& [k, v] : args[i].attrs) {
                    if (k == "align") {
                        s.append("align ");
                        s.append(v);
                        s.append(" ");
                    } else {
                        s.append(k);
                        s.append(" ");
                    }
                }
                s.append(args[i].type->to_string());
            }
            if (fn->function_type()->is_vararg()) {
                if (!args.empty()) s.append(", ");
                s.append("...");
            }
            s.append(")");
            // Emit function attributes (e.g. ` #0` or ` nounwind`) so the
            // declaration resolves its attribute-group references.
            for (const auto& attr : fn->function_attributes()) {
                s.append(" ");
                s.append(attr);
            }
            s.append("\n\n");
        } else {
            // Definition with body
            s.append(fn->to_string());
            s.append("\n");
        }
    }

    // 3. Named metadata — emit BEFORE numbered metadata definitions so that
    //    the named-metadata references (e.g. `!llvm.ident = !{!0}`) appear
    //    before the `!0 = ...` definitions they point to.  This matches
    //    clang's emission order and is what most .ll consumers expect.
    if (!named_metadata_.empty()) {
        s.append("\n");
        for (const auto& [name, body] : named_metadata_) {
            s.append("!");
            s.append(name);
            s.append(" = ");
            s.append(body);
            s.append("\n");
        }
    }

    // 4. Numbered metadata definitions — emitted at the very end of the
    //    module, matching clang's output convention.  Forward references
    //    from named metadata (above) and from instructions (inside function
    //    bodies) are resolved by LLVM's reader regardless of order, but
    //    keeping these at the end produces output that diffs cleanly
    //    against clang -S -emit-llvm.
    if (!metadata_defs_.empty()) {
        s.append("\n");
        for (const auto& [id, body] : metadata_defs_) {
            s.append("!");
            s.append(id);
            s.append(" = ");
            s.append(body);
            s.append("\n");
        }
    }

    return s;
}

} // namespace clunk::ir
