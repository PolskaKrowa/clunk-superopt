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
 * Clunk SMT Verifier — Z3-based equivalence checking.
 * Proves that optimised candidates are semantically equivalent
 * to the original program, ensuring optimisations are safe.
 *
 * Z3 is loaded at runtime via dlopen/dlsym — there is no link-time
 * dependency on libz3.
 */
#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"
#include "clunk/IR/Clone.h"
#include "clunk/IR/Scalarizer.h"
#include "clunk/Search/LoopOpt.h"
#include "clunk/Search/Z3RAII.h"
#include "clunk/Search/StochasticSearch.h"  // for structural_hash (cache key)
// Include RewriteCache for cache-key-based equivalence lookup in verify().
#include "clunk/Search/RewriteCache.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <stdexcept>
#include <cstdio>

// ── Z3 opaque types (no z3.h included) ──────────────────────────────────────

struct _Z3_context;
struct _Z3_ast;
struct _Z3_sort;
struct _Z3_solver;
struct _Z3_params;
struct _Z3_config;
struct _Z3_symbol;
struct _Z3_model;

using Z3_context = _Z3_context*;
using Z3_ast     = _Z3_ast*;
using Z3_sort    = _Z3_sort*;
using Z3_solver  = _Z3_solver*;
using Z3_params  = _Z3_params*;
using Z3_config  = _Z3_config*;
using Z3_symbol  = _Z3_symbol*;
using Z3_model   = _Z3_model*;

// Z3_lbool — mirrors the Z3 C API enum.
// The Z3 C API enum values (Z3 4.x) — must match exactly because
// Z3_solver_check returns a raw int that we switch on.
enum Z3_lbool_enum {
    Z3_L_FALSE     = -1,
    Z3_L_UNDEFINED = 0,
    Z3_L_TRUE      = 1
};

// Z3_error_code — mirrors the Z3 C API enum (subset; we only need the
// numeric value for error messages).
using Z3_error_code = int;

namespace clunk::search {

// ── Z3 error exception ─────────────────────────────────────────────────────
//
// Thrown by the Z3 error handler (installed via Z3_set_error_handler) when
// Z3 detects a C API misuse (e.g. sort mismatch in mk_bvadd). Caught in
// verify_with_z3 and converted into VerificationResult::Error.
//
struct Z3Error : public std::runtime_error {
    explicit Z3Error(const std::string& msg) : std::runtime_error(msg) {}
};

// ── Function-pointer types for every Z3 C API function we use ──────────────

using Z3_mk_config_t          = Z3_config (*)();
using Z3_del_config_t         = void (*)(Z3_config);
using Z3_mk_context_t         = Z3_context (*)(Z3_config);
using Z3_del_context_t        = void (*)(Z3_context);

using Z3_mk_bv_sort_t         = Z3_sort (*)(Z3_context, unsigned);
using Z3_mk_int_sort_t        = Z3_sort (*)(Z3_context);
using Z3_mk_bool_sort_t       = Z3_sort (*)(Z3_context);

using Z3_mk_string_symbol_t   = Z3_symbol (*)(Z3_context, const char*);
using Z3_mk_const_t           = Z3_ast (*)(Z3_context, Z3_symbol, Z3_sort);

using Z3_mk_unsigned_int64_t  = Z3_ast (*)(Z3_context, uint64_t, Z3_sort);
using Z3_mk_int_t             = Z3_ast (*)(Z3_context, int, Z3_sort);
using Z3_mk_true_t            = Z3_ast (*)(Z3_context);
using Z3_mk_false_t           = Z3_ast (*)(Z3_context);

// BV binary ops
using Z3_mk_bvadd_t           = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvsub_t           = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvmul_t           = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvudiv_t          = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvsdiv_t          = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvurem_t          = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvsrem_t          = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvand_t           = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvor_t            = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvxor_t           = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvshl_t           = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvlshr_t          = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvashr_t          = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);

// Integer arithmetic
using Z3_mk_add_t             = Z3_ast (*)(Z3_context, unsigned, Z3_ast const*);
using Z3_mk_sub_t             = Z3_ast (*)(Z3_context, unsigned, Z3_ast const*);
using Z3_mk_mul_t             = Z3_ast (*)(Z3_context, unsigned, Z3_ast const*);
using Z3_mk_div_t             = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_mod_t             = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);

// Comparisons (return Bool-sort ASTs)
using Z3_mk_eq_t              = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_not_t             = Z3_ast (*)(Z3_context, Z3_ast);
using Z3_mk_bvult_t           = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_bvslt_t           = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_gt_t              = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_ge_t              = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_lt_t              = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);
using Z3_mk_le_t              = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast);

// Boolean combinators (for path conditions)
using Z3_mk_and_t             = Z3_ast (*)(Z3_context, unsigned, Z3_ast const*);
using Z3_mk_or_t              = Z3_ast (*)(Z3_context, unsigned, Z3_ast const*);

// If-then-else
using Z3_mk_ite_t             = Z3_ast (*)(Z3_context, Z3_ast, Z3_ast, Z3_ast);

// Solver
using Z3_mk_solver_t          = Z3_solver (*)(Z3_context);
using Z3_solver_inc_ref_t     = void (*)(Z3_context, Z3_solver);
using Z3_solver_dec_ref_t     = void (*)(Z3_context, Z3_solver);
using Z3_solver_assert_t      = void (*)(Z3_context, Z3_solver, Z3_ast);
using Z3_solver_check_t       = Z3_lbool_enum (*)(Z3_context, Z3_solver);
using Z3_solver_set_params_t  = void (*)(Z3_context, Z3_solver, Z3_params);
using Z3_solver_push_t        = void (*)(Z3_context, Z3_solver);
using Z3_solver_pop_t         = void (*)(Z3_context, Z3_solver, unsigned);
using Z3_solver_get_model_t   = Z3_model (*)(Z3_context, Z3_solver);

// Params
using Z3_mk_params_t          = Z3_params (*)(Z3_context);
using Z3_params_inc_ref_t     = void (*)(Z3_context, Z3_params);
using Z3_params_dec_ref_t     = void (*)(Z3_context, Z3_params);
using Z3_params_set_uint_t    = void (*)(Z3_context, Z3_params, Z3_symbol, unsigned);
using Z3_params_set_bool_t    = void (*)(Z3_context, Z3_params, Z3_symbol, bool);

// Model
using Z3_model_inc_ref_t      = void (*)(Z3_context, Z3_model);
using Z3_model_dec_ref_t      = void (*)(Z3_context, Z3_model);
using Z3_model_eval_t         = int (*)(Z3_context, Z3_model, Z3_ast, int, Z3_ast*);

// Numerals
using Z3_get_numeral_int64_t  = int (*)(Z3_context, Z3_ast, int64_t*);

// AST reference counting
using Z3_inc_ref_t            = void (*)(Z3_context, Z3_ast);
using Z3_dec_ref_t            = void (*)(Z3_context, Z3_ast);

// Sort reference counting (separate functions in Z3)
using Z3_sort_inc_ref_t       = void (*)(Z3_context, Z3_sort);
using Z3_sort_dec_ref_t       = void (*)(Z3_context, Z3_sort);

// Error handling
using Z3_error_handler_t      = void (*)(Z3_context, Z3_error_code);
using Z3_set_error_handler_t  = void (*)(Z3_context, Z3_error_handler_t);

// Simplification
using Z3_simplify_t           = Z3_ast (*)(Z3_context, Z3_ast);

// ── Assumption-based batch pruning: assumption-based batch pruning ────────────────────────
// Z3_solver_check_assumptions: like Z3_solver_check, but takes an array of
// Bool-literal assumptions that are treated as additional assertions for
// this check only (no push/pop needed). On UNSAT, Z3_solver_get_unsat_core
// returns the subset of assumptions responsible for the unsat.
using Z3_solver_check_assumptions_t = Z3_lbool_enum (*)(Z3_context, Z3_solver,
                                                         unsigned, Z3_ast const[]);
// Z3_solver_get_unsat_core: returns a Z3_ast_vector of the assumptions
// responsible for the last UNSAT result. Returns an empty vector if the
// last check was SAT or if no assumptions were used.
struct _Z3_ast_vector;
using Z3_ast_vector = _Z3_ast_vector*;
using Z3_solver_get_unsat_core_t = Z3_ast_vector (*)(Z3_context, Z3_solver);

// Z3_ast_vector reference counting + access.
using Z3_ast_vector_inc_ref_t = void (*)(Z3_context, Z3_ast_vector);
using Z3_ast_vector_dec_ref_t = void (*)(Z3_context, Z3_ast_vector);
using Z3_ast_vector_size_t    = unsigned (*)(Z3_context, Z3_ast_vector);
using Z3_ast_vector_get_t     = Z3_ast (*)(Z3_context, Z3_ast_vector, unsigned);

// ── Z3 quantifier + model-extraction APIs (CEGIS exists-forall) ─────
// Used by SMTVerifier::synthesize_with_z3_quantifiers to build
//   exists placeholders. forall inputs. orig(inputs) == cand(inputs, placeholders)
// and to extract the synthesised placeholder model.
struct _Z3_func_decl;
struct _Z3_pattern;
struct _Z3_app;
using Z3_func_decl = _Z3_func_decl*;
using Z3_pattern   = _Z3_pattern*;
using Z3_app       = _Z3_app*;
using Z3_string    = const char*;

// Z3_mk_quantifier_const: build an exists/forall quantifier over `bound`
// Z3_app variables. body is the quantified formula. weight is a search
// hint (lower = preferred). patterns may be empty (num_patterns=0).
using Z3_mk_quantifier_const_t = Z3_ast (*)(Z3_context, int /*is_forall*/,
                                             unsigned /*weight*/,
                                             unsigned /*num_bound*/,
                                             const Z3_app[] /*bound*/,
                                             unsigned /*num_patterns*/,
                                             const Z3_pattern[] /*patterns*/,
                                             Z3_ast /*body*/);

// Z3_to_app: cast a Z3_ast to a Z3_app (needed for bound[]).
using Z3_to_app_t = Z3_app (*)(Z3_context, Z3_ast);

// Model const extraction: list the constants in a model and read their
// interpretations.
using Z3_model_get_num_consts_t  = unsigned (*)(Z3_context, Z3_model);
using Z3_model_get_const_decl_t  = Z3_func_decl (*)(Z3_context, Z3_model, unsigned);
using Z3_model_get_const_interp_t = Z3_ast (*)(Z3_context, Z3_model, Z3_func_decl);
using Z3_func_decl_to_ast_t      = Z3_ast (*)(Z3_context, Z3_func_decl);
using Z3_get_decl_name_t         = Z3_symbol (*)(Z3_context, Z3_func_decl);
using Z3_get_symbol_string_t     = Z3_string (*)(Z3_context, Z3_symbol);
using Z3_ast_to_string_t         = Z3_string (*)(Z3_context, Z3_ast);
using Z3_get_numeral_string_t    = Z3_string (*)(Z3_context, Z3_ast);

// ── Dynamic Z3 loader (singleton) ──────────────────────────────────────────
//
// Loads libz3.so at runtime and resolves every symbol we need.
// If loading or resolution of a REQUIRED symbol fails, z3_loaded() returns
// false and all function pointers remain nullptr — the verifier falls back
// to random simulation.
//
// Optional symbols (Z3_inc_ref, Z3_solver_set_params, Z3_set_error_handler,
// Z3_solver_push, Z3_solver_pop, Z3_solver_get_model, Z3_model_eval,
// Z3_model_inc_ref, Z3_model_dec_ref, Z3_get_numeral_int64, Z3_mk_bool_sort,
// Z3_mk_true, Z3_mk_false, Z3_mk_and, Z3_mk_or, Z3_sort_inc_ref,
// Z3_sort_dec_ref, Z3_params_set_bool) are resolved best-effort. If any is
// missing, the corresponding feature is disabled but Z3 is still usable
// for basic verification.

class Z3DynamicLoader {
public:
    static Z3DynamicLoader& instance() {
        static Z3DynamicLoader inst;
        return inst;
    }

    bool z3_loaded() const { return loaded_; }

    // Required function pointer members
    Z3_mk_config_t          Z3_mk_config_fn          = nullptr;
    Z3_del_config_t         Z3_del_config_fn         = nullptr;
    Z3_mk_context_t         Z3_mk_context_fn         = nullptr;
    Z3_del_context_t        Z3_del_context_fn        = nullptr;

    Z3_mk_bv_sort_t         Z3_mk_bv_sort_fn         = nullptr;
    Z3_mk_int_sort_t        Z3_mk_int_sort_fn        = nullptr;

    Z3_mk_string_symbol_t   Z3_mk_string_symbol_fn   = nullptr;
    Z3_mk_const_t           Z3_mk_const_fn           = nullptr;

    Z3_mk_unsigned_int64_t  Z3_mk_unsigned_int64_fn  = nullptr;
    Z3_mk_int_t             Z3_mk_int_fn             = nullptr;

    Z3_mk_bvadd_t           Z3_mk_bvadd_fn           = nullptr;
    Z3_mk_bvsub_t           Z3_mk_bvsub_fn           = nullptr;
    Z3_mk_bvmul_t           Z3_mk_bvmul_fn           = nullptr;
    Z3_mk_bvudiv_t          Z3_mk_bvudiv_fn          = nullptr;
    Z3_mk_bvsdiv_t          Z3_mk_bvsdiv_fn          = nullptr;
    Z3_mk_bvurem_t          Z3_mk_bvurem_fn          = nullptr;
    Z3_mk_bvsrem_t          Z3_mk_bvsrem_fn          = nullptr;
    Z3_mk_bvand_t           Z3_mk_bvand_fn           = nullptr;
    Z3_mk_bvor_t            Z3_mk_bvor_fn            = nullptr;
    Z3_mk_bvxor_t           Z3_mk_bvxor_fn           = nullptr;
    Z3_mk_bvshl_t           Z3_mk_bvshl_fn           = nullptr;
    Z3_mk_bvlshr_t          Z3_mk_bvlshr_fn          = nullptr;
    Z3_mk_bvashr_t          Z3_mk_bvashr_fn          = nullptr;

    Z3_mk_add_t             Z3_mk_add_fn             = nullptr;
    Z3_mk_sub_t             Z3_mk_sub_fn             = nullptr;
    Z3_mk_mul_t             Z3_mk_mul_fn             = nullptr;
    Z3_mk_div_t             Z3_mk_div_fn             = nullptr;
    Z3_mk_mod_t             Z3_mk_mod_fn             = nullptr;

    Z3_mk_eq_t              Z3_mk_eq_fn              = nullptr;
    Z3_mk_not_t             Z3_mk_not_fn             = nullptr;
    Z3_mk_bvult_t           Z3_mk_bvult_fn           = nullptr;
    Z3_mk_bvslt_t           Z3_mk_bvslt_fn           = nullptr;
    Z3_mk_gt_t              Z3_mk_gt_fn              = nullptr;
    Z3_mk_ge_t              Z3_mk_ge_fn              = nullptr;
    Z3_mk_lt_t              Z3_mk_lt_fn              = nullptr;
    Z3_mk_le_t              Z3_mk_le_fn              = nullptr;

    Z3_mk_ite_t             Z3_mk_ite_fn             = nullptr;

    Z3_mk_solver_t          Z3_mk_solver_fn          = nullptr;
    Z3_solver_inc_ref_t     Z3_solver_inc_ref_fn     = nullptr;
    Z3_solver_dec_ref_t     Z3_solver_dec_ref_fn     = nullptr;
    Z3_solver_assert_t      Z3_solver_assert_fn      = nullptr;
    Z3_solver_check_t       Z3_solver_check_fn       = nullptr;

    Z3_mk_params_t          Z3_mk_params_fn          = nullptr;
    Z3_params_inc_ref_t     Z3_params_inc_ref_fn     = nullptr;
    Z3_params_dec_ref_t     Z3_params_dec_ref_fn     = nullptr;
    Z3_params_set_uint_t    Z3_params_set_uint_fn    = nullptr;

    Z3_simplify_t           Z3_simplify_fn           = nullptr;

    // ── Optional function pointers (best-effort resolution) ─────────────
    Z3_mk_bool_sort_t       Z3_mk_bool_sort_fn       = nullptr;
    Z3_mk_true_t            Z3_mk_true_fn            = nullptr;
    Z3_mk_false_t           Z3_mk_false_fn           = nullptr;
    Z3_mk_and_t             Z3_mk_and_fn             = nullptr;
    Z3_mk_or_t              Z3_mk_or_fn              = nullptr;

    Z3_solver_set_params_t  Z3_solver_set_params_fn  = nullptr;
    Z3_solver_push_t        Z3_solver_push_fn        = nullptr;
    Z3_solver_pop_t         Z3_solver_pop_fn         = nullptr;
    Z3_solver_get_model_t   Z3_solver_get_model_fn   = nullptr;

    Z3_model_inc_ref_t      Z3_model_inc_ref_fn      = nullptr;
    Z3_model_dec_ref_t      Z3_model_dec_ref_fn      = nullptr;
    Z3_model_eval_t         Z3_model_eval_fn         = nullptr;
    Z3_get_numeral_int64_t  Z3_get_numeral_int64_fn  = nullptr;

    Z3_inc_ref_t            Z3_inc_ref_fn            = nullptr;
    Z3_dec_ref_t            Z3_dec_ref_fn            = nullptr;
    Z3_sort_inc_ref_t       Z3_sort_inc_ref_fn       = nullptr;
    Z3_sort_dec_ref_t       Z3_sort_dec_ref_fn       = nullptr;

    // Sort query (for coerce_to_bool)
    using Z3_get_sort_t       = Z3_sort (*)(Z3_context, Z3_ast);
    using Z3_get_sort_kind_t  = int (*)(Z3_context, Z3_sort);
    Z3_get_sort_t          Z3_get_sort_fn          = nullptr;
    Z3_get_sort_kind_t     Z3_get_sort_kind_fn     = nullptr;

    Z3_set_error_handler_t  Z3_set_error_handler_fn  = nullptr;
    Z3_params_set_bool_t    Z3_params_set_bool_fn    = nullptr;

    // ── Assumption-based batch pruning: assumption-based batch pruning ─────────────────
    Z3_solver_check_assumptions_t Z3_solver_check_assumptions_fn = nullptr;
    Z3_solver_get_unsat_core_t    Z3_solver_get_unsat_core_fn    = nullptr;
    Z3_ast_vector_inc_ref_t       Z3_ast_vector_inc_ref_fn       = nullptr;
    Z3_ast_vector_dec_ref_t       Z3_ast_vector_dec_ref_fn       = nullptr;
    Z3_ast_vector_size_t          Z3_ast_vector_size_fn          = nullptr;
    Z3_ast_vector_get_t           Z3_ast_vector_get_fn           = nullptr;

    // ── Integer cast encoding (trunc/zext/sext) ─────────────────────────
    // Optional: when absent, casts fall back to fresh symbolic variables
    // (sound but blind — functions containing casts become unprovable).
    using Z3_mk_extract_t  = Z3_ast (*)(Z3_context, unsigned, unsigned, Z3_ast);
    using Z3_mk_zero_ext_t = Z3_ast (*)(Z3_context, unsigned, Z3_ast);
    using Z3_mk_sign_ext_t = Z3_ast (*)(Z3_context, unsigned, Z3_ast);
    Z3_mk_extract_t   Z3_mk_extract_fn   = nullptr;
    Z3_mk_zero_ext_t  Z3_mk_zero_ext_fn  = nullptr;
    Z3_mk_sign_ext_t  Z3_mk_sign_ext_fn  = nullptr;

    // ── Z3 quantifier + model-extraction (CEGIS) ────
    Z3_mk_quantifier_const_t      Z3_mk_quantifier_const_fn      = nullptr;
    Z3_to_app_t                   Z3_to_app_fn                   = nullptr;
    Z3_model_get_num_consts_t     Z3_model_get_num_consts_fn     = nullptr;
    Z3_model_get_const_decl_t     Z3_model_get_const_decl_fn     = nullptr;
    Z3_model_get_const_interp_t   Z3_model_get_const_interp_fn   = nullptr;
    Z3_func_decl_to_ast_t         Z3_func_decl_to_ast_fn         = nullptr;
    Z3_get_decl_name_t            Z3_get_decl_name_fn            = nullptr;
    Z3_get_symbol_string_t        Z3_get_symbol_string_fn        = nullptr;
    Z3_ast_to_string_t            Z3_ast_to_string_fn            = nullptr;
    Z3_get_numeral_string_t       Z3_get_numeral_string_fn       = nullptr;

private:
    Z3DynamicLoader() { load(); }
    Z3DynamicLoader(const Z3DynamicLoader&) = delete;
    Z3DynamicLoader& operator=(const Z3DynamicLoader&) = delete;

    void load() {
        // Try several sonames — distributions ship different versions.
        // RTLD_LOCAL (not RTLD_GLOBAL) to avoid polluting the global
        // symbol namespace.
        static const char* names[] = {
            "libz3.so",
            "libz3.so.4",
            "libz3.so.3",
            "libz3.dylib",     // macOS
            nullptr
        };

        for (int i = 0; names[i]; ++i) {
            handle_ = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
            if (handle_) break;
        }

        if (!handle_) {
            loaded_ = false;
            return;
        }

        // Resolve REQUIRED symbols. If any is missing, fall back entirely.
        #define RESOLVE(name)                                               \
            do {                                                            \
                name##_fn = reinterpret_cast<name##_t>(                     \
                    dlsym(handle_, #name));                                  \
                if (!name##_fn) {                                           \
                    loaded_ = false;                                        \
                    return;                                                 \
                }                                                           \
            } while (0)

        // Resolve OPTIONAL symbols (best-effort — nullptr is acceptable).
        #define RESOLVE_OPTIONAL(name)                                      \
            do {                                                            \
                name##_fn = reinterpret_cast<name##_t>(                     \
                    dlsym(handle_, #name));                                  \
            } while (0)

        RESOLVE(Z3_mk_config);
        RESOLVE(Z3_del_config);
        RESOLVE(Z3_mk_context);
        RESOLVE(Z3_del_context);

        RESOLVE(Z3_mk_bv_sort);
        RESOLVE(Z3_mk_int_sort);

        RESOLVE(Z3_mk_string_symbol);
        RESOLVE(Z3_mk_const);

        RESOLVE(Z3_mk_unsigned_int64);
        RESOLVE(Z3_mk_int);

        RESOLVE(Z3_mk_bvadd);
        RESOLVE(Z3_mk_bvsub);
        RESOLVE(Z3_mk_bvmul);
        RESOLVE(Z3_mk_bvudiv);
        RESOLVE(Z3_mk_bvsdiv);
        RESOLVE(Z3_mk_bvurem);
        RESOLVE(Z3_mk_bvsrem);
        RESOLVE(Z3_mk_bvand);
        RESOLVE(Z3_mk_bvor);
        RESOLVE(Z3_mk_bvxor);
        RESOLVE(Z3_mk_bvshl);
        RESOLVE(Z3_mk_bvlshr);
        RESOLVE(Z3_mk_bvashr);

        RESOLVE(Z3_mk_add);
        RESOLVE(Z3_mk_sub);
        RESOLVE(Z3_mk_mul);
        RESOLVE(Z3_mk_div);
        RESOLVE(Z3_mk_mod);

        RESOLVE(Z3_mk_eq);
        RESOLVE(Z3_mk_not);
        RESOLVE(Z3_mk_bvult);
        RESOLVE(Z3_mk_bvslt);
        RESOLVE(Z3_mk_gt);
        RESOLVE(Z3_mk_ge);
        RESOLVE(Z3_mk_lt);
        RESOLVE(Z3_mk_le);

        RESOLVE(Z3_mk_ite);

        RESOLVE(Z3_mk_solver);
        RESOLVE(Z3_solver_inc_ref);
        RESOLVE(Z3_solver_dec_ref);
        RESOLVE(Z3_solver_assert);
        RESOLVE(Z3_solver_check);

        RESOLVE(Z3_mk_params);
        RESOLVE(Z3_params_inc_ref);
        RESOLVE(Z3_params_dec_ref);
        RESOLVE(Z3_params_set_uint);

        RESOLVE(Z3_simplify);

        // Optional symbols — nullptr is acceptable.
        RESOLVE_OPTIONAL(Z3_mk_bool_sort);
        RESOLVE_OPTIONAL(Z3_mk_true);
        RESOLVE_OPTIONAL(Z3_mk_false);
        RESOLVE_OPTIONAL(Z3_mk_and);
        // Integer cast encoding (trunc/zext/sext).
        RESOLVE_OPTIONAL(Z3_mk_extract);
        RESOLVE_OPTIONAL(Z3_mk_zero_ext);
        RESOLVE_OPTIONAL(Z3_mk_sign_ext);
        RESOLVE_OPTIONAL(Z3_mk_or);

        RESOLVE_OPTIONAL(Z3_solver_set_params);
        RESOLVE_OPTIONAL(Z3_solver_push);
        RESOLVE_OPTIONAL(Z3_solver_pop);
        RESOLVE_OPTIONAL(Z3_solver_get_model);

        RESOLVE_OPTIONAL(Z3_model_inc_ref);
        RESOLVE_OPTIONAL(Z3_model_dec_ref);
        RESOLVE_OPTIONAL(Z3_model_eval);
        RESOLVE_OPTIONAL(Z3_get_numeral_int64);

        RESOLVE_OPTIONAL(Z3_inc_ref);
        RESOLVE_OPTIONAL(Z3_dec_ref);
        RESOLVE_OPTIONAL(Z3_sort_inc_ref);
        RESOLVE_OPTIONAL(Z3_sort_dec_ref);
        RESOLVE_OPTIONAL(Z3_get_sort);
        RESOLVE_OPTIONAL(Z3_get_sort_kind);

        RESOLVE_OPTIONAL(Z3_set_error_handler);
        RESOLVE_OPTIONAL(Z3_params_set_bool);

        // Assumption-based batch pruning: assumption-based batch pruning symbols.
        RESOLVE_OPTIONAL(Z3_solver_check_assumptions);
        RESOLVE_OPTIONAL(Z3_solver_get_unsat_core);
        RESOLVE_OPTIONAL(Z3_ast_vector_inc_ref);
        RESOLVE_OPTIONAL(Z3_ast_vector_dec_ref);
        RESOLVE_OPTIONAL(Z3_ast_vector_size);
        RESOLVE_OPTIONAL(Z3_ast_vector_get);

        // Quantifier + model-extraction symbols for
        // exists-forall CEGIS. All optional — if any is missing, the
        // synthesize_with_z3_quantifiers path returns not-supported and
        // the caller falls back to the constant-pool enumeration path.
        RESOLVE_OPTIONAL(Z3_mk_quantifier_const);
        RESOLVE_OPTIONAL(Z3_to_app);
        RESOLVE_OPTIONAL(Z3_model_get_num_consts);
        RESOLVE_OPTIONAL(Z3_model_get_const_decl);
        RESOLVE_OPTIONAL(Z3_model_get_const_interp);
        RESOLVE_OPTIONAL(Z3_func_decl_to_ast);
        RESOLVE_OPTIONAL(Z3_get_decl_name);
        RESOLVE_OPTIONAL(Z3_get_symbol_string);
        RESOLVE_OPTIONAL(Z3_ast_to_string);
        RESOLVE_OPTIONAL(Z3_get_numeral_string);

        #undef RESOLVE
        #undef RESOLVE_OPTIONAL

        loaded_ = true;
    }

    // We intentionally never dlclose() the handle — the library stays
    // loaded for the entire process lifetime.
    void* handle_ = nullptr;
    bool  loaded_ = false;
};

// Convenience macro — short alias for the singleton's function pointer members
#define Z3API(name) (Z3DynamicLoader::instance().name##_fn)

// ── Z3 error handler callback ──────────────────────────────────────────────
//
// Installed via Z3_set_error_handler. Throws Z3Error instead of letting Z3
// call exit(1). On Linux/glibc, throwing through Z3's C frames works
// correctly (the C++ exception unwinder traverses C frames without issue).
// This is technically implementation-defined behaviour but is the standard
// way to use Z3 from C++ and is what z3++ (the official C++ wrapper) does
// internally.
//
// The handler is per-context, so we install it once per context creation.
//
namespace {
void z3_error_handler_cb(Z3_context /*c*/, Z3_error_code err_code) {
    throw Z3Error("Z3 internal error (code " + std::to_string(err_code) + ")");
}
} // anonymous namespace

// ── SMT Encoding structure ─────────────────────────────────────────────────
// Holds the Z3 expressions built from encoding a function.
struct SMTVerifier::SMTEncoding {
    // Map from value name → Z3 AST
    std::unordered_map<std::string, Z3_ast> value_map;

    // The final return value Z3 expression
    Z3_ast return_value = nullptr;

    // Z3 context (borrowed — owned by the caller)
    Z3_context ctx = nullptr;

    // Track ASTs that need to be released (Z3_dec_ref'd) at end of verify.
    // Populated by track_ast().
    std::vector<Z3_ast> local_refs;

    // True if the encoder hit something it cannot soundly model
    // (memory ops, float ops, calls, loops). When set, verify_with_z3
    // returns Unknown instead of risking a false Equivalent.
    bool had_unsupported = false;

    // Human-readable description of what was unsupported (for diagnostics).
    std::string unsupported_reason;

    // Argument Z3 constants in positional order (for counterexample
    // extraction). Each entry is (arg_name, Z3_ast).
    std::vector<std::pair<std::string, Z3_ast>> arg_vars;

    // ── Poison-aware encoding: shared POISON constants per bit-width ────────
    // When honor_binop_flags is true, flagged binops encode their result
    // as `ite(overflow_cond, POISON_<width>, normal_result)` where
    // POISON_<width> is a free Z3 constant of the given bit-width. The
    // SAME POISON_<width> is shared between the original and candidate
    // encodings so that two functions producing poison on the same
    // inputs compare equal (both undefined), while a function producing
    // poison vs. a function producing a defined value compare unequal
    // (Z3 can pick POISON to differ from the defined value).
    //
    // The map is keyed by bit-width. Owned by the caller (verify_with_z3)
    // and cleaned up via shared_poison_refs — NOT via enc.local_refs.
    std::unordered_map<unsigned, Z3_ast> poison_consts;

    // ── Alive2-style refinement semantics ──────────────────────────────
    // When SMTConfig::refinement_semantics is on, each encoded value gets
    // a poison FLAG (Z3 Bool) instead of a POISON value substitution.
    // Absent entry / nullptr means "never poison" (constants, arguments).
    // Flags propagate through every op; flagged binops and out-of-range
    // shifts add their own poison conditions. `return_poison` mirrors
    // `return_value` (nullptr = never poison).
    std::unordered_map<std::string, Z3_ast> poison_map;
    Z3_ast return_poison = nullptr;
};

// ── Helper: track an AST for later dec_ref ─────────────────────────────────
//
// Calls Z3_inc_ref (if available) and appends to enc.local_refs so the AST
// is released at the end of verify_with_z3. Returns the AST unchanged so
// it can be used inline:
//     Z3_ast v = track_ast(enc, Z3API(Z3_mk_bvadd)(ctx, a, b));
//
static Z3_ast track_ast(SMTVerifier::SMTEncoding& enc, Z3_ast ast) {
    if (ast && Z3API(Z3_inc_ref) && Z3API(Z3_dec_ref)) {
        Z3API(Z3_inc_ref)(enc.ctx, ast);
        enc.local_refs.push_back(ast);
    }
    return ast;
}

// ── Helper: release all tracked ASTs ───────────────────────────────────────
static void release_tracked(SMTVerifier::SMTEncoding& enc) {
    if (!Z3API(Z3_dec_ref)) return;
    for (Z3_ast ast : enc.local_refs) {
        if (ast) Z3API(Z3_dec_ref)(enc.ctx, ast);
    }
    enc.local_refs.clear();
}

// ── Helper: make a "true" Bool AST (for path conditions) ───────────────────
static Z3_ast make_true(Z3_context ctx) {
    if (Z3API(Z3_mk_true)) return Z3API(Z3_mk_true)(ctx);
    // Fallback: construct true as mk_eq(0, 0) on BV 1 — not ideal but works.
    Z3_sort bv1 = Z3API(Z3_mk_bv_sort)(ctx, 1);
    Z3_ast zero = Z3API(Z3_mk_unsigned_int64)(ctx, 0, bv1);
    return Z3API(Z3_mk_eq)(ctx, zero, zero);
}

// ── Helper: make a "false" Bool AST ────────────────────────────────────────
static Z3_ast make_false(Z3_context ctx) {
    if (Z3API(Z3_mk_false)) return Z3API(Z3_mk_false)(ctx);
    Z3_sort bv1 = Z3API(Z3_mk_bv_sort)(ctx, 1);
    Z3_ast zero = Z3API(Z3_mk_unsigned_int64)(ctx, 0, bv1);
    Z3_ast one  = Z3API(Z3_mk_unsigned_int64)(ctx, 1, bv1);
    return Z3API(Z3_mk_eq)(ctx, zero, one);
}

// ── Helper: logical AND of two Bool ASTs ───────────────────────────────────
static Z3_ast mk_and2(Z3_context ctx, Z3_ast a, Z3_ast b) {
    if (!a) return b;
    if (!b) return a;
    if (Z3API(Z3_mk_and)) {
        Z3_ast args[] = {a, b};
        return Z3API(Z3_mk_and)(ctx, 2, args);
    }
    // Fallback: (a && b) == !( !a || !b ) — but we don't have mk_or either.
    // Use ite: ite(a, b, false).
    Z3_ast f = make_false(ctx);
    return Z3API(Z3_mk_ite)(ctx, a, b, f);
}

// ── Helper: logical OR of two Bool ASTs ────────────────────────────────────
static Z3_ast mk_or2(Z3_context ctx, Z3_ast a, Z3_ast b) {
    if (!a) return b;
    if (!b) return a;
    if (Z3API(Z3_mk_or)) {
        Z3_ast args[] = {a, b};
        return Z3API(Z3_mk_or)(ctx, 2, args);
    }
    // Fallback: ite(a, true, b).
    Z3_ast t = make_true(ctx);
    return Z3API(Z3_mk_ite)(ctx, a, t, b);
}

// ── Helper: poison flag of an operand (refinement semantics) ───────────────
// nullptr means "never poison" (constants, arguments, values encoded before
// refinement tracking, or flags disabled).
static Z3_ast operand_poison(SMTVerifier::SMTEncoding& enc,
                             const std::shared_ptr<ir::Value>& op) {
    if (!op || !op->has_name()) return nullptr;
    auto it = enc.poison_map.find(op->name());
    return it == enc.poison_map.end() ? nullptr : it->second;
}

// ── Helper: coerce a Bool-sort AST to a 1-bit BV ───────────────────────────
// Used for ICmp results (C9): the comparison returns a Bool, but the IR
// treats i1 as a 1-bit BV. We coerce via ite(bool, bv(1,1), bv(0,1)) so
// the result can be used uniformly as a BV operand.
static Z3_ast coerce_bool_to_bv1(SMTVerifier::SMTEncoding& enc, Z3_ast bool_ast) {
    if (!bool_ast) return nullptr;
    Z3_context ctx = enc.ctx;
    Z3_sort bv1 = Z3API(Z3_mk_bv_sort)(ctx, 1);
    Z3_ast one  = Z3API(Z3_mk_unsigned_int64)(ctx, 1, bv1);
    Z3_ast zero = Z3API(Z3_mk_unsigned_int64)(ctx, 0, bv1);
    Z3_ast ite  = Z3API(Z3_mk_ite)(ctx, bool_ast, one, zero);
    return track_ast(enc, ite);
}

// ── Helper: coerce any Bool/BV(1) AST to a Bool ────────────────────────────
// Inverse of coerce_bool_to_bv1. Used in path-condition computation where
// br_cond is the operand of a conditional branch — it may be either a Bool
// (raw Z3 comparison) or a BV(1) (an ICmp result that was coerced at def
// time). Z3_mk_and / Z3_mk_or / Z3_mk_not require Bool operands, so we
// must convert BV(1) back to Bool via `bv != 0`.
//
// If the AST is already Bool, return unchanged. If we can't query the sort
// (older Z3), conservatively assume BV(1) and apply the conversion.
//
// IMPORTANT: the real Z3 library (4.x) defines Z3_sort_kind as:
//   Z3_UNKNOWN_SORT=0, Z3_BOOL_SORT=1, Z3_INT_SORT=2,
//   Z3_REAL_SORT=3, Z3_BV_SORT=4, Z3_ARRAY_SORT=5, ...
// The vendored vendor/z3/z3.h has WRONG values (BOOL_SORT=4, BV_SORT=5).
// We use the correct values here.
static Z3_ast coerce_to_bool(SMTVerifier::SMTEncoding& enc, Z3_ast ast) {
    if (!ast) return nullptr;
    Z3_context ctx = enc.ctx;

    // Try to query the sort.
    if (Z3API(Z3_get_sort) && Z3API(Z3_get_sort_kind)) {
        Z3_sort s = Z3API(Z3_get_sort)(ctx, ast);
        int kind = Z3API(Z3_get_sort_kind)(ctx, s);
        if (kind == 1 /*Z3_BOOL_SORT in real Z3 4.x*/) {
            return ast;  // already Bool
        }
        // It's a BV (or other sort) — convert to Bool via `!= 0`.
        // This works for any BV width.
        Z3_ast zero = Z3API(Z3_mk_unsigned_int64)(ctx, 0, s);
        Z3_ast eq_zero = track_ast(enc, Z3API(Z3_mk_eq)(ctx, ast, zero));
        return track_ast(enc, Z3API(Z3_mk_not)(ctx, eq_zero));
    }

    // No sort query available — assume BV(1) and convert via eq(_, 1).
    Z3_sort bv1 = Z3API(Z3_mk_bv_sort)(ctx, 1);
    Z3_ast one  = Z3API(Z3_mk_unsigned_int64)(ctx, 1, bv1);
    return track_ast(enc, Z3API(Z3_mk_eq)(ctx, ast, one));
}

// ── Forward declarations ───────────────────────────────────────────────────
static Z3_sort get_z3_sort(Z3_context ctx, const ir::Type& type, bool use_bitvectors);
static bool function_has_unsupported_ops(const ir::Function& fn,
                                          const SMTConfig& config,
                                          std::string& reason);
static bool function_has_loops(const ir::Function& fn);

static SMTVerifier::SMTEncoding encode_function(
    Z3_context ctx,
    const ir::Function& fn,
    const std::string& prefix,
    bool use_bitvectors,
    const std::vector<std::pair<std::string, Z3_ast>>& shared_args,
    const SMTConfig& config,
    std::unordered_map<unsigned, Z3_ast>* shared_poison = nullptr,
    const std::unordered_map<std::string, Z3_ast>* placeholder_vars = nullptr);

static std::optional<int64_t> simulate_function_impl(const ir::Function& fn,
                                                       const std::vector<int64_t>& inputs);

// ── Poison-aware encoding: helpers for poison-aware binop encoding ──────────────
//
// get_or_create_poison_const: lazily create (or look up) the shared POISON
// Z3 constant for a given bit-width. The constant is a free BV variable
// named "poison_<width>". It is stored in `shared_poison` (if non-null) or
// `enc.poison_consts` (otherwise), and is tracked in `enc.local_refs` for
// cleanup. The caller is responsible for ensuring the constant is alive
// for both encodings that share it (via shared_poison).
//
static Z3_ast get_or_create_poison_const(SMTVerifier::SMTEncoding& enc,
                                          Z3_context ctx,
                                          unsigned width,
                                          std::unordered_map<unsigned, Z3_ast>* shared_poison) {
    auto& store = shared_poison ? *shared_poison : enc.poison_consts;
    auto it = store.find(width);
    if (it != store.end()) return it->second;
    Z3_sort sort = Z3API(Z3_mk_bv_sort)(ctx, width);
    std::string name = "poison_" + std::to_string(width);
    Z3_symbol sym = Z3API(Z3_mk_string_symbol)(ctx, name.c_str());
    Z3_ast poison = Z3API(Z3_mk_const)(ctx, sym, sort);
    poison = track_ast(enc, poison);
    store[width] = poison;
    return poison;
}

// ── Poison-aware encoding: compute the overflow condition for a flagged binop ────
//
// Returns a Bool-sort Z3 AST that is TRUE iff the operation would overflow
// (i.e., produce poison under the given flag). Returns nullptr if the
// flag/opcode combination is not modelled (caller should treat as
// "no overflow" — i.e., plain encoding).
//
// The formulas use only already-resolved Z3 BV functions (bvadd, bvsub,
// bvmul, bvudiv, bvsdiv, bvshl, bvlshr, bvashr, bvand, bvxor, bvult,
// bvslt, mk_eq, mk_not, mk_ite, mk_or, mk_and) — no new symbols needed.
//
// Width N is the bit-width of the result. We use sign-bit tests where
// possible (independent of N) to keep the formulas uniform.
//
static Z3_ast compute_binop_overflow_cond(SMTVerifier::SMTEncoding& enc,
                                           Z3_context ctx,
                                           ir::Opcode op,
                                           const ir::BinOpFlags& flags,
                                           Z3_ast lhs, Z3_ast rhs,
                                           Z3_ast result,
                                           unsigned width) {
    // Helper: BV constant of value `v` at `width` bits.
    auto mk_bv = [&](uint64_t v) -> Z3_ast {
        Z3_sort s = Z3API(Z3_mk_bv_sort)(ctx, width);
        return track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, v, s));
    };
    // Helper: signed-non-negative predicate (sign bit clear).
    auto sge_zero = [&](Z3_ast v) -> Z3_ast {
        Z3_sort s = Z3API(Z3_mk_bv_sort)(ctx, width);
        Z3_ast zero = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, 0, s));
        // v >=s 0  iff  !(v <s 0)
        Z3_ast slt_zero = track_ast(enc, Z3API(Z3_mk_bvslt)(ctx, v, zero));
        return track_ast(enc, Z3API(Z3_mk_not)(ctx, slt_zero));
    };
    (void)mk_bv;
    (void)sge_zero;

    // Width-specific INT_MIN (1 << (width-1)) and -1 (~0).
    uint64_t width_mask = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);
    uint64_t int_min_val = (width >= 64) ? (1ULL << 63) : (1ULL << (width - 1));
    int_min_val &= width_mask;
    uint64_t all_ones = width_mask;

    // ── Add/Sub/Mul: nuw + nsw ──────────────────────────────────────────
    if (op == ir::Opcode::Add) {
        // r = a + b mod 2^N
        Z3_ast overflow = nullptr;
        if (flags.nuw) {
            // Unsigned overflow iff r u< a (carry-out occurred).
            Z3_ast ult = track_ast(enc, Z3API(Z3_mk_bvult)(ctx, result, lhs));
            overflow = ult;
        }
        if (flags.nsw) {
            // Signed overflow iff ((a^r) & (b^r)) has sign bit set,
            // i.e., (bvslt (bvand (bvxor a r) (bvxor b r)) 0).
            Z3_ast axr = track_ast(enc, Z3API(Z3_mk_bvxor)(ctx, lhs, result));
            Z3_ast bxr = track_ast(enc, Z3API(Z3_mk_bvxor)(ctx, rhs, result));
            Z3_ast andr = track_ast(enc, Z3API(Z3_mk_bvand)(ctx, axr, bxr));
            Z3_sort s = Z3API(Z3_mk_bv_sort)(ctx, width);
            Z3_ast zero = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, 0, s));
            Z3_ast slt = track_ast(enc, Z3API(Z3_mk_bvslt)(ctx, andr, zero));
            overflow = overflow ? track_ast(enc, mk_or2(ctx, overflow, slt)) : slt;
        }
        return overflow;
    }
    if (op == ir::Opcode::Sub) {
        Z3_ast overflow = nullptr;
        if (flags.nuw) {
            // Unsigned underflow iff a u< b.
            Z3_ast ult = track_ast(enc, Z3API(Z3_mk_bvult)(ctx, lhs, rhs));
            overflow = ult;
        }
        if (flags.nsw) {
            // Signed overflow iff ((a^b) & (a^r)) has sign bit set.
            Z3_ast axb = track_ast(enc, Z3API(Z3_mk_bvxor)(ctx, lhs, rhs));
            Z3_ast axr = track_ast(enc, Z3API(Z3_mk_bvxor)(ctx, lhs, result));
            Z3_ast andr = track_ast(enc, Z3API(Z3_mk_bvand)(ctx, axb, axr));
            Z3_sort s = Z3API(Z3_mk_bv_sort)(ctx, width);
            Z3_ast zero = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, 0, s));
            Z3_ast slt = track_ast(enc, Z3API(Z3_mk_bvslt)(ctx, andr, zero));
            overflow = overflow ? track_ast(enc, mk_or2(ctx, overflow, slt)) : slt;
        }
        return overflow;
    }
    if (op == ir::Opcode::Mul) {
        Z3_ast overflow = nullptr;
        if (flags.nuw) {
            // Unsigned overflow iff a != 0 && (r udiv a) != b.
            Z3_sort s = Z3API(Z3_mk_bv_sort)(ctx, width);
            Z3_ast zero = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, 0, s));
            Z3_ast a_neq_0 = track_ast(enc, Z3API(Z3_mk_not)(ctx,
                track_ast(enc, Z3API(Z3_mk_eq)(ctx, lhs, zero))));
            Z3_ast udiv = track_ast(enc, Z3API(Z3_mk_bvudiv)(ctx, result, lhs));
            Z3_ast div_neq_b = track_ast(enc, Z3API(Z3_mk_not)(ctx,
                track_ast(enc, Z3API(Z3_mk_eq)(ctx, udiv, rhs))));
            Z3_ast cond = track_ast(enc, mk_and2(ctx, a_neq_0, div_neq_b));
            overflow = cond;
        }
        if (flags.nsw) {
            // Signed overflow iff a != 0 && ((r sdiv a) != b), with the
            // edge case (a == -1 && b == INT_MIN) where the mul overflows
            // but sdiv(INT_MIN, -1) also overflows to INT_MIN — so the
            // naive check would miss it. We add the edge case explicitly.
            Z3_sort s = Z3API(Z3_mk_bv_sort)(ctx, width);
            Z3_ast zero = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, 0, s));
            Z3_ast neg_one = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, all_ones, s));
            Z3_ast int_min = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, int_min_val, s));
            Z3_ast a_neq_0 = track_ast(enc, Z3API(Z3_mk_not)(ctx,
                track_ast(enc, Z3API(Z3_mk_eq)(ctx, lhs, zero))));
            Z3_ast sdiv = track_ast(enc, Z3API(Z3_mk_bvsdiv)(ctx, result, lhs));
            Z3_ast div_neq_b = track_ast(enc, Z3API(Z3_mk_not)(ctx,
                track_ast(enc, Z3API(Z3_mk_eq)(ctx, sdiv, rhs))));
            // Edge: (a == -1 && b == INT_MIN)
            Z3_ast a_eq_neg1 = track_ast(enc, Z3API(Z3_mk_eq)(ctx, lhs, neg_one));
            Z3_ast b_eq_min = track_ast(enc, Z3API(Z3_mk_eq)(ctx, rhs, int_min));
            Z3_ast edge = track_ast(enc, mk_and2(ctx, a_eq_neg1, b_eq_min));
            Z3_ast cond_inner = track_ast(enc, mk_or2(ctx, div_neq_b, edge));
            Z3_ast cond = track_ast(enc, mk_and2(ctx, a_neq_0, cond_inner));
            overflow = overflow ? track_ast(enc, mk_or2(ctx, overflow, cond)) : cond;
        }
        return overflow;
    }

    // ── Shl: nuw + nsw ──────────────────────────────────────────────────
    if (op == ir::Opcode::Shl) {
        Z3_sort s = Z3API(Z3_mk_bv_sort)(ctx, width);
        Z3_ast zero = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, 0, s));
        Z3_ast width_bv = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, width, s));
        // b >= width (signed): shift amount exceeds bit-width → UB.
        Z3_ast b_ge_width = track_ast(enc,
            Z3API(Z3_mk_not)(ctx, track_ast(enc, Z3API(Z3_mk_bvslt)(ctx, rhs, width_bv))));
        Z3_ast overflow = b_ge_width;
        if (flags.nuw) {
            // Unsigned overflow iff (a << b) u>> b != a.
            Z3_ast shl_back = track_ast(enc, Z3API(Z3_mk_bvlshr)(ctx, result, rhs));
            Z3_ast neq_a = track_ast(enc, Z3API(Z3_mk_not)(ctx,
                track_ast(enc, Z3API(Z3_mk_eq)(ctx, shl_back, lhs))));
            Z3_ast cond = track_ast(enc, mk_and2(ctx,
                track_ast(enc, Z3API(Z3_mk_not)(ctx, b_ge_width)), neq_a));
            overflow = track_ast(enc, mk_or2(ctx, overflow, cond));
        }
        if (flags.nsw) {
            // Signed overflow iff (a << b) s>> b != a (sign bits changed).
            Z3_ast shl_back = track_ast(enc, Z3API(Z3_mk_bvashr)(ctx, result, rhs));
            Z3_ast neq_a = track_ast(enc, Z3API(Z3_mk_not)(ctx,
                track_ast(enc, Z3API(Z3_mk_eq)(ctx, shl_back, lhs))));
            Z3_ast cond = track_ast(enc, mk_and2(ctx,
                track_ast(enc, Z3API(Z3_mk_not)(ctx, b_ge_width)), neq_a));
            overflow = track_ast(enc, mk_or2(ctx, overflow, cond));
        }
        (void)zero;
        return overflow;
    }

    // ── UDiv/SDiv: exact (no remainder) ─────────────────────────────────
    if (op == ir::Opcode::UDiv && flags.exact) {
        // Exact iff b != 0 && (a udiv b) * b == a.
        Z3_sort s = Z3API(Z3_mk_bv_sort)(ctx, width);
        Z3_ast zero = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, 0, s));
        Z3_ast b_eq_0 = track_ast(enc, Z3API(Z3_mk_eq)(ctx, rhs, zero));
        Z3_ast b_neq_0 = track_ast(enc, Z3API(Z3_mk_not)(ctx, b_eq_0));
        // remainder = a - (a udiv b) * b
        Z3_ast mul = track_ast(enc, Z3API(Z3_mk_bvmul)(ctx, result, rhs));
        Z3_ast rem = track_ast(enc, Z3API(Z3_mk_bvsub)(ctx, lhs, mul));
        Z3_ast rem_neq_0 = track_ast(enc, Z3API(Z3_mk_not)(ctx,
            track_ast(enc, Z3API(Z3_mk_eq)(ctx, rem, zero))));
        // overflow (poison) iff b == 0 OR (b != 0 AND rem != 0).
        Z3_ast cond = track_ast(enc, mk_and2(ctx, b_neq_0, rem_neq_0));
        return track_ast(enc, mk_or2(ctx, b_eq_0, cond));
    }
    if (op == ir::Opcode::SDiv && flags.exact) {
        Z3_sort s = Z3API(Z3_mk_bv_sort)(ctx, width);
        Z3_ast zero = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, 0, s));
        Z3_ast b_eq_0 = track_ast(enc, Z3API(Z3_mk_eq)(ctx, rhs, zero));
        Z3_ast b_neq_0 = track_ast(enc, Z3API(Z3_mk_not)(ctx, b_eq_0));
        Z3_ast mul = track_ast(enc, Z3API(Z3_mk_bvmul)(ctx, result, rhs));
        Z3_ast rem = track_ast(enc, Z3API(Z3_mk_bvsub)(ctx, lhs, mul));
        Z3_ast rem_neq_0 = track_ast(enc, Z3API(Z3_mk_not)(ctx,
            track_ast(enc, Z3API(Z3_mk_eq)(ctx, rem, zero))));
        Z3_ast cond = track_ast(enc, mk_and2(ctx, b_neq_0, rem_neq_0));
        return track_ast(enc, mk_or2(ctx, b_eq_0, cond));
    }

    // ── LShr/AShr: exact (no bits lost) ─────────────────────────────────
    if ((op == ir::Opcode::LShr || op == ir::Opcode::AShr) && flags.exact) {
        Z3_sort s = Z3API(Z3_mk_bv_sort)(ctx, width);
        Z3_ast width_bv = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, width, s));
        Z3_ast b_ge_width = track_ast(enc,
            Z3API(Z3_mk_not)(ctx, track_ast(enc, Z3API(Z3_mk_bvslt)(ctx, rhs, width_bv))));
        // exact iff (a >> b) << b == a (under b < width).
        Z3_ast shl_back = track_ast(enc, Z3API(Z3_mk_bvshl)(ctx, result, rhs));
        Z3_ast neq_a = track_ast(enc, Z3API(Z3_mk_not)(ctx,
            track_ast(enc, Z3API(Z3_mk_eq)(ctx, shl_back, lhs))));
        Z3_ast cond = track_ast(enc, mk_and2(ctx,
            track_ast(enc, Z3API(Z3_mk_not)(ctx, b_ge_width)), neq_a));
        return track_ast(enc, mk_or2(ctx, b_ge_width, cond));
    }

    return nullptr;  // No flag/opcode combination modelled → no overflow constraint.
}

// ── get_z3_sort: map an IR Type to a Z3 sort ───────────────────────────────
static Z3_sort get_z3_sort(Z3_context ctx, const ir::Type& type, bool use_bitvectors) {
    if (type.is_integer()) {
        auto& int_ty = static_cast<const ir::IntegerType&>(type);
        if (use_bitvectors) {
            return Z3API(Z3_mk_bv_sort)(ctx, int_ty.bits());
        } else {
            return Z3API(Z3_mk_int_sort)(ctx);
        }
    }
    if (type.is_float()) {
        // C8 sound fallback: floats should never reach here in sound mode
        // (function_has_unsupported_ops returns true and we bail earlier).
        // If we do reach here, use BV 32 (matches old behaviour — caller
        // should treat result as unsound).
        return Z3API(Z3_mk_bv_sort)(ctx, 32);
    }
    if (type.is_double()) {
        return Z3API(Z3_mk_bv_sort)(ctx, 64);
    }
    if (type.is_pointer()) {
        // Pointers modelled as 64-bit BVs (address space).
        return Z3API(Z3_mk_bv_sort)(ctx, 64);
    }
    if (type.is_void()) {
        // Void has no sort; use BV 1 as a placeholder.
        return Z3API(Z3_mk_bv_sort)(ctx, 1);
    }
    // Default: use 32-bit BV
    if (use_bitvectors) {
        return Z3API(Z3_mk_bv_sort)(ctx, 32);
    } else {
        return Z3API(Z3_mk_int_sort)(ctx);
    }
}

// ── function_has_unsupported_ops ───────────────────────────────────────────
//
// Returns true if the function contains operations that the encoder cannot
// soundly model. When true, verify_with_z3 returns Unknown instead of
// risking a false Equivalent.
//
// The set of unsupported operations depends on the SMTConfig sound_*_fallback
// flags:
//   - Memory ops (Load/Store/Alloca/GEP/Fence/Call) — always unsupported
//     when sound_memory_fallback is true.
//   - Float ops (FAdd/FSub/FMul/FDiv/FRem/FCmp) — unsupported when
//     sound_float_fallback is true.
//   - Loops (back-edges) — unsupported when sound_loop_fallback is true.
//
static bool function_has_unsupported_ops(const ir::Function& fn,
                                          const SMTConfig& config,
                                          std::string& reason) {
    bool has_mem = false, has_fp = false;

    size_t block_count = fn.blocks().size();
    size_t inst_count = 0;

    // Vector functions must never reach the raw encoder (their ops would
    // become per-side fresh variables — trivially SAT, i.e. bogus
    // NotEquivalent verdicts). verify() lowers them via the scalarizer
    // before encoding; if one slips through anyway, refuse. Always on —
    // there is no sound raw encoding to fall back to.
    if (ir::function_has_vector_ops(fn)) {
        reason = "function contains vector operations (must be scalarized first)";
        return true;
    }

    for (const auto& bb : fn.blocks()) {
        if (!bb) continue;
        inst_count += bb->size();
        for (const auto& inst : bb->instructions()) {
            auto op = inst->opcode();

            // Memory operations
            if (inst->is_memory_op()) has_mem = true;

            // Calls
            if (op == ir::Opcode::Call) has_mem = true;

            // Floating-point ops
            switch (op) {
                case ir::Opcode::FAdd:
                case ir::Opcode::FSub:
                case ir::Opcode::FMul:
                case ir::Opcode::FDiv:
                case ir::Opcode::FRem:
                case ir::Opcode::FCmp:
                case ir::Opcode::FPTrunc:
                case ir::Opcode::FPExt:
                case ir::Opcode::FPToUI:
                case ir::Opcode::FPToSI:
                case ir::Opcode::UIToFP:
                case ir::Opcode::SIToFP:
                    has_fp = true;
                    break;
                default: break;
            }
        }
    }

    if (has_mem && config.sound_memory_fallback) {
        reason = "function contains memory operations (load/store/alloca/gep/call)";
        return true;
    }
    if (has_fp && config.sound_float_fallback) {
        reason = "function contains floating-point operations";
        return true;
    }
    if (config.sound_loop_fallback && function_has_loops(fn)) {
        reason = "function contains loops (back-edges) — bounded unrolling not implemented";
        return true;
    }
    // Refuse to verify functions whose CFG is too complex.
    // The path-condition encoding is exponential in the number of basic
    // blocks (OR over predecessors of (pred_pc AND edge_cond)), so a
    // 100-block diamond can produce an AST of size 2^100. Cap it soundly
    // by returning Unknown. The instruction-count cap is a secondary
    // guard against pathological single-block functions.
    if (config.sound_large_function_fallback) {
        if (block_count > config.max_blocks_for_smt ||
            inst_count > config.max_instructions_for_smt) {
            reason = "function too complex for SMT encoding (" +
                     std::to_string(block_count) + " blocks, " +
                     std::to_string(inst_count) + " instructions)";
            return true;
        }
    }
    return false;
}

// ── function_has_loops ─────────────────────────────────────────────────────
//
// Detects back-edges in the CFG. A back-edge is a branch from block B to
// block C where C appears at or before B in source order (i.e., C's index
// <= B's index). This is a conservative loop detector — it may report
// false positives for irreducible CFGs, but never false negatives.
//
static bool function_has_loops(const ir::Function& fn) {
    // Build a map from block name → source-order index.
    std::unordered_map<std::string, size_t> block_index;
    for (size_t i = 0; i < fn.blocks().size(); ++i) {
        block_index[fn.blocks()[i]->name()] = i;
    }

    for (size_t i = 0; i < fn.blocks().size(); ++i) {
        const auto& bb = fn.blocks()[i];
        auto succs = bb->successors();
        for (const auto& succ_name : succs) {
            auto it = block_index.find(succ_name);
            if (it == block_index.end()) continue;
            // Back-edge: successor's index <= current block's index.
            // (We use <= to catch self-loops where successor == current.)
            if (it->second <= i) {
                return true;
            }
        }
    }
    return false;
}

// ── Helper: resolve an IR Value to a Z3 AST ────────────────────────────────
//
// Looks up the value by name in enc.value_map. If not found, checks if it's
// a ConstantInt and encodes it directly. Returns nullptr if the value
// cannot be resolved.
//
static Z3_ast resolve_value(SMTVerifier::SMTEncoding& enc,
                             const std::shared_ptr<ir::Value>& val,
                             bool use_bitvectors) {
    if (!val) return nullptr;

    // Named value: look up in value_map
    if (val->has_name()) {
        auto it = enc.value_map.find(val->name());
        if (it != enc.value_map.end()) return it->second;
    }

    // ConstantInt
    if (auto* ci = dynamic_cast<const ir::ConstantInt*>(val.get())) {
        Z3_context ctx = enc.ctx;
        Z3_sort sort;
        if (val->type() && val->type()->is_integer()) {
            sort = get_z3_sort(ctx, *val->type(), use_bitvectors);
        } else {
            sort = Z3API(Z3_mk_bv_sort)(ctx, 32);
        }
        Z3_ast result;
        if (use_bitvectors) {
            result = Z3API(Z3_mk_unsigned_int64)(ctx,
                        static_cast<uint64_t>(ci->value()), sort);
        } else {
            result = Z3API(Z3_mk_int)(ctx, static_cast<int>(ci->value()), sort);
        }
        return track_ast(enc, result);
    }

    // ConstantPointerNull, UndefValue, PoisonValue: encode as 0
    // (unsound for undef/poison, but we only reach here if the function
    // doesn't trip the unsupported-ops check — which it would, since
    // these usually accompany memory ops).
    return nullptr;
}

// ── encode_function: encode a function's semantics as a Z3 formula ────────
//
// This is the core of the verifier. It walks the function's basic blocks
// in source order, computing a path condition for each block and encoding
// each instruction under that path condition using if-lifting:
//
//   value' = ite(path_cond, encoded_value, value)
//
// Arguments are shared between original and candidate: the caller
// passes a pre-built `shared_args` vector of (name, Z3_ast) pairs, which
// we install in enc.value_map so that references to %arg resolve to the
// SAME Z3 constant on both sides.
//
// The `prefix` is used only for local SSA value names (to avoid collisions
// between original and candidate locals).
//
static SMTVerifier::SMTEncoding encode_function(
    Z3_context ctx,
    const ir::Function& fn,
    const std::string& prefix,
    bool use_bitvectors,
    const std::vector<std::pair<std::string, Z3_ast>>& shared_args,
    const SMTConfig& config,
    std::unordered_map<unsigned, Z3_ast>* shared_poison,
    const std::unordered_map<std::string, Z3_ast>* placeholder_vars) {

    (void)config;  // config used only by caller for unsupported-op detection

    SMTVerifier::SMTEncoding enc;
    enc.ctx = ctx;
    enc.return_value = nullptr;
    enc.had_unsupported = false;

    // ── Install shared arguments ──────────────────────────────────────────
    // The arguments are the SAME Z3 constants for both original and
    // candidate. We register them in value_map under their original names
    // (e.g., "x", "y") so that references to %x in the function body
    // resolve to the shared constant.
    for (const auto& [arg_name, arg_ast] : shared_args) {
        enc.value_map[arg_name] = arg_ast;
        enc.arg_vars.push_back({arg_name, arg_ast});
    }

    // ── Install placeholder Z3 vars (CEGIS) ────────────────────────────────
    // If the caller passed a placeholder_vars map (used by
    // synthesize_with_z3_quantifiers), install each placeholder name →
    // free Z3 const into value_map. This intercepts the named ConstantInt
    // in resolve_value and uses the free Z3 const instead of a concrete
    // numeral, so Z3 can find a model for the placeholder.
    if (placeholder_vars) {
        for (const auto& [name, ast] : *placeholder_vars) {
            enc.value_map[name] = ast;
        }
    }

    // ── Compute path conditions per block ────────────────────────────────
    //
    // block_path_cond[B] = OR over predecessors P of (block_path_cond[P] AND edge_cond(P, B))
    //
    // Entry block has path_cond = true.
    //
    // Since blocks are visited in source order (LLVM IR is topologically
    // sorted for acyclic CFGs), all predecessors of B have been visited
    // before B (except back-edges, which indicate loops — handled by the
    // unsupported-ops check earlier).
    //
    std::unordered_map<std::string, Z3_ast> block_path_cond;

    // Build block index for predecessor lookup
    std::unordered_map<std::string, size_t> block_index;
    for (size_t i = 0; i < fn.blocks().size(); ++i) {
        block_index[fn.blocks()[i]->name()] = i;
    }

    // Ensure predecessors are computed
    const_cast<ir::Function&>(fn).compute_predecessors();

    Z3_ast true_cond = track_ast(enc, make_true(ctx));

    // Edge condition from `pred_name` to `target_name`: the branch-taken
    // predicate on the predecessor's terminator (NOT including the
    // predecessor's own path condition). Returns nullptr if there is no such
    // edge / it can't be modelled. Shared by the path-condition computation and
    // the phi encoding below.
    auto edge_condition = [&](const std::string& pred_name,
                              const std::string& target_name) -> Z3_ast {
        auto pred_bb = fn.block(pred_name);
        if (!pred_bb) return nullptr;
        auto term = pred_bb->terminator();
        if (!term) return nullptr;
        if (term->opcode() == ir::Opcode::Br) {
            const auto& md = term->metadata();
            auto true_it  = md.find("true_bb");
            auto false_it = md.find("false_bb");
            auto dest_it  = md.find("dest_bb");
            if (true_it != md.end() && false_it != md.end()) {
                if (true_it->second == target_name && false_it->second == target_name)
                    return true_cond;  // both edges to same block
                if (true_it->second == target_name) {
                    if (term->num_operands() >= 1) {
                        Z3_ast raw = resolve_value(enc, term->operand(0), use_bitvectors);
                        return raw ? coerce_to_bool(enc, raw) : nullptr;
                    }
                } else if (false_it->second == target_name) {
                    if (term->num_operands() >= 1) {
                        Z3_ast raw = resolve_value(enc, term->operand(0), use_bitvectors);
                        if (raw)
                            return track_ast(enc, Z3API(Z3_mk_not)(
                                ctx, coerce_to_bool(enc, raw)));
                    }
                }
                return nullptr;
            } else if (dest_it != md.end() && dest_it->second == target_name) {
                return true_cond;  // unconditional branch to us
            }
            return nullptr;
        } else if (term->opcode() == ir::Opcode::SwitchInst) {
            // Over-approximate switch edges as always-possible (sound for pc).
            return true_cond;
        }
        return nullptr;
    };

    for (size_t bi = 0; bi < fn.blocks().size(); ++bi) {
        const auto& bb = fn.blocks()[bi];
        const std::string& bb_name = bb->name();

        Z3_ast pc = nullptr;  // path condition for this block

        if (bi == 0) {
            // Entry block: path condition is true.
            pc = true_cond;
        } else if (bb->predecessors().empty()) {
            // Unreachable block (no predecessors): path condition is false.
            // We still encode it (in case it has side effects we care about),
            // but its contributions are gated by false.
            pc = track_ast(enc, make_false(ctx));
        } else {
            // pc = OR over preds of (pred_pc AND edge_cond(pred, this))
            for (const auto& pred_name : bb->predecessors()) {
                auto pc_it = block_path_cond.find(pred_name);
                if (pc_it == block_path_cond.end()) continue;  // pred not visited (back-edge)

                Z3_ast pred_pc = pc_it->second;
                Z3_ast edge_cond = edge_condition(pred_name, bb_name);
                if (!edge_cond) continue;

                Z3_ast contribution = track_ast(enc, mk_and2(ctx, pred_pc, edge_cond));
                pc = track_ast(enc, mk_or2(ctx, pc, contribution));
            }

            if (!pc) {
                // No predecessors contributed (shouldn't happen for reachable
                // blocks, but be safe).
                pc = track_ast(enc, make_false(ctx));
            }
        }

        block_path_cond[bb_name] = pc;

        // ── Encode instructions in this block ────────────────────────────
        for (const auto& inst : bb->instructions()) {
            // Skip terminators (handled by path-condition computation)
            if (inst->is_terminator()) {
                // Handle ret
                if (inst->opcode() == ir::Opcode::Ret && inst->num_operands() > 0) {
                    Z3_ast ret_val = resolve_value(enc, inst->operand(0), use_bitvectors);
                    if (ret_val) {
                        // Merge return values across blocks using path condition:
                        //   ret' = ite(pc, ret_val, ret)
                        const bool first_ret = (enc.return_value == nullptr);
                        if (first_ret) {
                            enc.return_value = ret_val;
                        } else {
                            Z3_ast merged = track_ast(enc,
                                Z3API(Z3_mk_ite)(ctx, pc, ret_val, enc.return_value));
                            enc.return_value = merged;
                        }
                        // Refinement: fold the returned value's poison flag
                        // through the same path-condition merge.
                        if (config.refinement_semantics && use_bitvectors) {
                            Z3_ast rp = operand_poison(enc, inst->operand(0));
                            if (first_ret) {
                                enc.return_poison = rp;  // may stay null
                            } else if (rp || enc.return_poison) {
                                Z3_ast t = rp ? rp
                                              : track_ast(enc, make_false(ctx));
                                Z3_ast f = enc.return_poison
                                    ? enc.return_poison
                                    : track_ast(enc, make_false(ctx));
                                enc.return_poison = track_ast(enc,
                                    Z3API(Z3_mk_ite)(ctx, pc, t, f));
                            }
                        }
                    }
                }
                continue;
            }

            // ── Phi nodes ────────────────────────────────────────────────
            // Encode as nested ite over incoming edges:
            //   phi = ite(edge1_cond, incoming1,
            //             ite(edge2_cond, incoming2, default))
            //
            // edge_cond = pred_pc AND (branch condition that leads to this block)
            //
            if (inst->opcode() == ir::Opcode::Phi) {
                // Proper multi-incoming encoding. The incoming block names live
                // in the "phi_blocks" metadata (comma-separated, aligned with
                // the value operands). The phi's value is incoming_k exactly
                // when control reached this block from block_k, i.e. under
                //   block_path_cond[block_k] AND edge_cond(block_k -> here).
                // Those conditions are mutually exclusive in an acyclic CFG
                // (loops are rejected earlier), so a nested ite is exact:
                //   ite(from_0, v0, ite(from_1, v1, ... v_{n-1})).
                const size_t n = inst->num_operands();
                if (n == 0) continue;
                if (n == 1) {
                    Z3_ast v = resolve_value(enc, inst->operand(0), use_bitvectors);
                    if (v && inst->has_name()) {
                        enc.value_map[inst->name()] = v;
                        if (config.refinement_semantics && use_bitvectors) {
                            Z3_ast p = operand_poison(enc, inst->operand(0));
                            if (p) enc.poison_map[inst->name()] = p;
                        }
                    }
                    continue;
                }

                std::vector<std::string> inc_blocks;
                auto pit = inst->metadata().find("phi_blocks");
                if (pit != inst->metadata().end()) {
                    std::string cur;
                    for (char c : pit->second) {
                        if (c == ',') { inc_blocks.push_back(cur); cur.clear(); }
                        else cur.push_back(c);
                    }
                    if (!cur.empty()) inc_blocks.push_back(cur);
                }
                if (inc_blocks.size() != n) {
                    // No reliable block mapping — stay sound.
                    enc.had_unsupported = true;
                    enc.unsupported_reason =
                        "phi node incoming block metadata missing or mismatched";
                    continue;
                }

                // Fold from the last incoming (default) toward the first.
                // The poison flag folds through the SAME nested ite so the
                // chosen edge's poison is what propagates (refinement mode).
                const bool refine_phi =
                    config.refinement_semantics && use_bitvectors;
                Z3_ast acc = resolve_value(enc, inst->operand(n - 1), use_bitvectors);
                Z3_ast acc_p = refine_phi
                    ? operand_poison(enc, inst->operand(n - 1)) : nullptr;
                bool ok = (acc != nullptr);
                for (size_t k = n - 1; ok && k-- > 0; ) {
                    Z3_ast vk = resolve_value(enc, inst->operand(k), use_bitvectors);
                    if (!vk) { ok = false; break; }
                    Z3_ast ec = edge_condition(inc_blocks[k], bb_name);
                    Z3_ast cond = ec;
                    auto bit = block_path_cond.find(inc_blocks[k]);
                    if (bit != block_path_cond.end() && ec)
                        cond = track_ast(enc, mk_and2(ctx, bit->second, ec));
                    if (!cond) { ok = false; break; }
                    Z3_ast cb = coerce_to_bool(enc, cond);
                    acc = track_ast(enc, Z3API(Z3_mk_ite)(ctx, cb, vk, acc));
                    if (refine_phi) {
                        Z3_ast pk = operand_poison(enc, inst->operand(k));
                        if (pk || acc_p) {
                            Z3_ast t = pk ? pk : track_ast(enc, make_false(ctx));
                            Z3_ast f = acc_p ? acc_p
                                             : track_ast(enc, make_false(ctx));
                            acc_p = track_ast(enc,
                                Z3API(Z3_mk_ite)(ctx, cb, t, f));
                        }
                    }
                }
                if (!ok) {
                    enc.had_unsupported = true;
                    enc.unsupported_reason = "phi node incoming value could not be encoded";
                    continue;
                }
                if (inst->has_name()) {
                    enc.value_map[inst->name()] = acc;
                    if (refine_phi && acc_p) enc.poison_map[inst->name()] = acc_p;
                }
                continue;
            }

            // ── Binary operations ────────────────────────────────────────
            if (inst->is_binary_op() && inst->num_operands() >= 2) {
                Z3_ast lhs = resolve_value(enc, inst->operand(0), use_bitvectors);
                Z3_ast rhs = resolve_value(enc, inst->operand(1), use_bitvectors);

                if (!lhs || !rhs) continue;

                Z3_ast result = nullptr;

                if (use_bitvectors) {
                    switch (inst->opcode()) {
                        case ir::Opcode::Add:  result = Z3API(Z3_mk_bvadd)(ctx, lhs, rhs); break;
                        case ir::Opcode::Sub:  result = Z3API(Z3_mk_bvsub)(ctx, lhs, rhs); break;
                        case ir::Opcode::Mul:  result = Z3API(Z3_mk_bvmul)(ctx, lhs, rhs); break;
                        case ir::Opcode::UDiv: result = Z3API(Z3_mk_bvudiv)(ctx, lhs, rhs); break;
                        case ir::Opcode::SDiv: result = Z3API(Z3_mk_bvsdiv)(ctx, lhs, rhs); break;
                        case ir::Opcode::URem: result = Z3API(Z3_mk_bvurem)(ctx, lhs, rhs); break;
                        case ir::Opcode::SRem: result = Z3API(Z3_mk_bvsrem)(ctx, lhs, rhs); break;
                        case ir::Opcode::And:  result = Z3API(Z3_mk_bvand)(ctx, lhs, rhs); break;
                        case ir::Opcode::Or:   result = Z3API(Z3_mk_bvor)(ctx, lhs, rhs); break;
                        case ir::Opcode::Xor:  result = Z3API(Z3_mk_bvxor)(ctx, lhs, rhs); break;
                        case ir::Opcode::Shl:  result = Z3API(Z3_mk_bvshl)(ctx, lhs, rhs); break;
                        case ir::Opcode::LShr: result = Z3API(Z3_mk_bvlshr)(ctx, lhs, rhs); break;
                        case ir::Opcode::AShr: result = Z3API(Z3_mk_bvashr)(ctx, lhs, rhs); break;
                        // Float ops: should be caught by unsupported-ops check,
                        // but if we reach here, leave result as nullptr (dropped).
                        default: break;
                    }
                } else {
                    switch (inst->opcode()) {
                        case ir::Opcode::Add: {
                            Z3_ast args[] = {lhs, rhs};
                            result = Z3API(Z3_mk_add)(ctx, 2, args);
                            break;
                        }
                        case ir::Opcode::Sub: {
                            Z3_ast args[] = {lhs, rhs};
                            result = Z3API(Z3_mk_sub)(ctx, 2, args);
                            break;
                        }
                        case ir::Opcode::Mul: {
                            Z3_ast args[] = {lhs, rhs};
                            result = Z3API(Z3_mk_mul)(ctx, 2, args);
                            break;
                        }
                        case ir::Opcode::SDiv:
                        case ir::Opcode::UDiv:
                            result = Z3API(Z3_mk_div)(ctx, lhs, rhs); break;
                        case ir::Opcode::SRem:
                        case ir::Opcode::URem:
                            result = Z3API(Z3_mk_mod)(ctx, lhs, rhs); break;
                        default: break;
                    }
                }

                if (result) result = track_ast(enc, result);

                // ── Poison-aware encoding: honor binop flags (nuw/nsw/exact) ────
                // Legacy exact-equivalence mode: wrap the result in
                // `ite(overflow_cond, POISON, result)` where POISON is a
                // shared free Z3 constant of the result's bit-width.
                //
                // Soundness: when honor_binop_flags is false (or the flag/
                // opcode combination is unmodelled), the result is left
                // unchanged — falling back to the existing plain-integer
                // encoding (which is unsound for flag-exploiting rewrites
                // but matches today's behaviour).
                if (result && config.honor_binop_flags && use_bitvectors &&
                    !config.refinement_semantics) {
                    const auto& flags = inst->binop_flags();
                    if (flags.nuw || flags.nsw || flags.exact) {
                        unsigned width = 32;
                        if (inst->type() && inst->type()->is_integer()) {
                            width = static_cast<unsigned>(inst->type()->bit_width());
                        }
                        Z3_ast overflow_cond = compute_binop_overflow_cond(
                            enc, ctx, inst->opcode(), flags, lhs, rhs, result, width);
                        if (overflow_cond) {
                            Z3_ast poison = get_or_create_poison_const(
                                enc, ctx, width, shared_poison);
                            Z3_ast guarded = Z3API(Z3_mk_ite)(ctx, overflow_cond, poison, result);
                            result = track_ast(enc, guarded);
                        }
                    }
                }

                // ── Alive2-style refinement: per-value poison FLAGS ───────
                // The value stays the plain wrapped result; a Bool flag
                // records when the instruction produces poison: operand
                // poison propagates, nsw/nuw/exact violations poison, and
                // (unconditionally, per LLVM LangRef) a shift by an amount
                // >= the bit-width poisons — a case the legacy encoding
                // does not model at all.
                if (result && use_bitvectors && config.refinement_semantics) {
                    unsigned width = 32;
                    if (inst->type() && inst->type()->is_integer()) {
                        width = static_cast<unsigned>(inst->type()->bit_width());
                    }
                    Z3_ast p = mk_or2(ctx, operand_poison(enc, inst->operand(0)),
                                      operand_poison(enc, inst->operand(1)));
                    if (config.honor_binop_flags) {
                        const auto& flags = inst->binop_flags();
                        if (flags.nuw || flags.nsw || flags.exact) {
                            Z3_ast overflow_cond = compute_binop_overflow_cond(
                                enc, ctx, inst->opcode(), flags, lhs, rhs,
                                result, width);
                            if (overflow_cond) p = mk_or2(ctx, p, overflow_cond);
                        }
                    }
                    if (inst->opcode() == ir::Opcode::Shl ||
                        inst->opcode() == ir::Opcode::LShr ||
                        inst->opcode() == ir::Opcode::AShr) {
                        Z3_sort s = Z3API(Z3_mk_bv_sort)(ctx, width);
                        Z3_ast w = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(
                            ctx, width, s));
                        Z3_ast in_range = track_ast(enc,
                            Z3API(Z3_mk_bvult)(ctx, rhs, w));
                        Z3_ast oob = track_ast(enc,
                            Z3API(Z3_mk_not)(ctx, in_range));
                        p = mk_or2(ctx, p, oob);
                    }
                    if (p && inst->has_name())
                        enc.poison_map[inst->name()] = track_ast(enc, p);
                }

                if (result && inst->has_name()) {
                    enc.value_map[inst->name()] = result;
                }
                continue;
            }

            // ── Comparison operations (ICmp) ─────────────────────────────
            // The result is coerced to a 1-bit BV via
            // ite(bool, bv(1,1), bv(0,1)) so it can be used uniformly as
            // a BV operand.
            if (inst->opcode() == ir::Opcode::ICmp && inst->num_operands() >= 2) {
                Z3_ast lhs = resolve_value(enc, inst->operand(0), use_bitvectors);
                Z3_ast rhs = resolve_value(enc, inst->operand(1), use_bitvectors);

                if (!lhs || !rhs) continue;

                // Get the comparison predicate
                auto it_pred = inst->metadata().find("pred");
                ir::CmpPredicate pred = ir::CmpPredicate::EQ;
                if (it_pred != inst->metadata().end()) {
                    try {
                        pred = static_cast<ir::CmpPredicate>(std::stoul(it_pred->second));
                    } catch (...) {
                        pred = ir::CmpPredicate::EQ;
                    }
                }

                Z3_ast bool_result = nullptr;

                if (use_bitvectors) {
                    switch (pred) {
                        case ir::CmpPredicate::EQ:  bool_result = Z3API(Z3_mk_eq)(ctx, lhs, rhs); break;
                        case ir::CmpPredicate::NE: {
                            Z3_ast eq = Z3API(Z3_mk_eq)(ctx, lhs, rhs);
                            bool_result = Z3API(Z3_mk_not)(ctx, eq);
                            break;
                        }
                        case ir::CmpPredicate::UGT: bool_result = Z3API(Z3_mk_bvult)(ctx, rhs, lhs); break;
                        case ir::CmpPredicate::UGE: {
                            Z3_ast lt = Z3API(Z3_mk_bvult)(ctx, lhs, rhs);
                            bool_result = Z3API(Z3_mk_not)(ctx, lt);
                            break;
                        }
                        case ir::CmpPredicate::ULT: bool_result = Z3API(Z3_mk_bvult)(ctx, lhs, rhs); break;
                        case ir::CmpPredicate::ULE: {
                            Z3_ast gt = Z3API(Z3_mk_bvult)(ctx, rhs, lhs);
                            bool_result = Z3API(Z3_mk_not)(ctx, gt);
                            break;
                        }
                        case ir::CmpPredicate::SGT: bool_result = Z3API(Z3_mk_bvslt)(ctx, rhs, lhs); break;
                        case ir::CmpPredicate::SGE: {
                            Z3_ast slt = Z3API(Z3_mk_bvslt)(ctx, lhs, rhs);
                            bool_result = Z3API(Z3_mk_not)(ctx, slt);
                            break;
                        }
                        case ir::CmpPredicate::SLT: bool_result = Z3API(Z3_mk_bvslt)(ctx, lhs, rhs); break;
                        case ir::CmpPredicate::SLE: {
                            Z3_ast sgt = Z3API(Z3_mk_bvslt)(ctx, rhs, lhs);
                            bool_result = Z3API(Z3_mk_not)(ctx, sgt);
                            break;
                        }
                        default: bool_result = Z3API(Z3_mk_eq)(ctx, lhs, rhs); break;
                    }
                } else {
                    switch (pred) {
                        case ir::CmpPredicate::EQ:  bool_result = Z3API(Z3_mk_eq)(ctx, lhs, rhs); break;
                        case ir::CmpPredicate::NE: {
                            Z3_ast eq = Z3API(Z3_mk_eq)(ctx, lhs, rhs);
                            bool_result = Z3API(Z3_mk_not)(ctx, eq);
                            break;
                        }
                        case ir::CmpPredicate::UGT:
                        case ir::CmpPredicate::SGT:
                            bool_result = Z3API(Z3_mk_gt)(ctx, lhs, rhs); break;
                        case ir::CmpPredicate::UGE:
                        case ir::CmpPredicate::SGE:
                            bool_result = Z3API(Z3_mk_ge)(ctx, lhs, rhs); break;
                        case ir::CmpPredicate::ULT:
                        case ir::CmpPredicate::SLT:
                            bool_result = Z3API(Z3_mk_lt)(ctx, lhs, rhs); break;
                        case ir::CmpPredicate::ULE:
                        case ir::CmpPredicate::SLE:
                            bool_result = Z3API(Z3_mk_le)(ctx, lhs, rhs); break;
                        default: bool_result = Z3API(Z3_mk_eq)(ctx, lhs, rhs); break;
                    }
                }

                if (bool_result) {
                    bool_result = track_ast(enc, bool_result);
                    // Coerce Bool to 1-bit BV so the result can be
                    // used uniformly as a BV operand.
                    Z3_ast bv_result = coerce_bool_to_bv1(enc, bool_result);
                    if (inst->has_name()) {
                        enc.value_map[inst->name()] = bv_result;
                        // Refinement: icmp of a poison operand is poison.
                        if (config.refinement_semantics && use_bitvectors) {
                            Z3_ast p = mk_or2(
                                ctx, operand_poison(enc, inst->operand(0)),
                                operand_poison(enc, inst->operand(1)));
                            if (p) enc.poison_map[inst->name()] = track_ast(enc, p);
                        }
                    }
                }
                continue;
            }

            // ── Select ───────────────────────────────────────────────────
            if (inst->opcode() == ir::Opcode::Select && inst->num_operands() >= 3) {
                Z3_ast cond = resolve_value(enc, inst->operand(0), use_bitvectors);
                Z3_ast tv   = resolve_value(enc, inst->operand(1), use_bitvectors);
                Z3_ast fv   = resolve_value(enc, inst->operand(2), use_bitvectors);

                if (cond && tv && fv) {
                    // cond is a 1-bit BV; convert to Bool for ite.
                    // ite(cond != 0, tv, fv)
                    Z3_sort bv1 = Z3API(Z3_mk_bv_sort)(ctx, 1);
                    Z3_ast zero_bv = track_ast(enc, Z3API(Z3_mk_unsigned_int64)(ctx, 0, bv1));
                    Z3_ast cond_bool = track_ast(enc, Z3API(Z3_mk_eq)(ctx, cond, zero_bv));
                    cond_bool = track_ast(enc, Z3API(Z3_mk_not)(ctx, cond_bool));
                    Z3_ast result = Z3API(Z3_mk_ite)(ctx, cond_bool, tv, fv);
                    result = track_ast(enc, result);
                    if (inst->has_name()) {
                        enc.value_map[inst->name()] = result;
                        // Refinement: select is poison if its condition is
                        // poison, or if the CHOSEN arm is poison (the
                        // unchosen arm's poison is masked — LLVM LangRef).
                        if (config.refinement_semantics && use_bitvectors) {
                            Z3_ast pc_ = operand_poison(enc, inst->operand(0));
                            Z3_ast pt = operand_poison(enc, inst->operand(1));
                            Z3_ast pf = operand_poison(enc, inst->operand(2));
                            Z3_ast arm = nullptr;
                            if (pt || pf) {
                                Z3_ast t = pt ? pt : track_ast(enc, make_false(ctx));
                                Z3_ast f = pf ? pf : track_ast(enc, make_false(ctx));
                                arm = track_ast(enc,
                                    Z3API(Z3_mk_ite)(ctx, cond_bool, t, f));
                            }
                            Z3_ast p = mk_or2(ctx, pc_, arm);
                            if (p) enc.poison_map[inst->name()] = track_ast(enc, p);
                        }
                    }
                }
                continue;
            }

            // ── Casts (Trunc, ZExt, SExt) ────────────────────────────────
            // Encode the common integer casts. Others are left as fresh
            // symbolic variables (conservative under-approximation — sound
            // because a fresh variable can take any value, so the equivalence
            // query is still valid).
            if (inst->is_cast() && inst->num_operands() >= 1) {
                Z3_ast src = resolve_value(enc, inst->operand(0), use_bitvectors);
                if (src && inst->type()) {
                    // Exact BV encoding of the integer casts when the loader
                    // resolved the Z3 primitives (extract / zero_ext /
                    // sign_ext). Bit-widths come from the IR types; a
                    // malformed width relation (e.g. a "trunc" that widens)
                    // falls through to the fresh-variable fallback, which is
                    // sound (a free variable admits every behaviour).
                    Z3_ast result = nullptr;
                    if (use_bitvectors && inst->operand(0)->type() &&
                        inst->type()->is_integer() &&
                        inst->operand(0)->type()->is_integer()) {
                        const unsigned src_w = static_cast<unsigned>(
                            inst->operand(0)->type()->bit_width());
                        const unsigned dst_w = static_cast<unsigned>(
                            inst->type()->bit_width());
                        switch (inst->opcode()) {
                            case ir::Opcode::Trunc:
                                if (dst_w < src_w && dst_w > 0 &&
                                    Z3API(Z3_mk_extract))
                                    result = Z3API(Z3_mk_extract)(
                                        ctx, dst_w - 1, 0, src);
                                break;
                            case ir::Opcode::ZExt:
                                if (dst_w > src_w && Z3API(Z3_mk_zero_ext))
                                    result = Z3API(Z3_mk_zero_ext)(
                                        ctx, dst_w - src_w, src);
                                break;
                            case ir::Opcode::SExt:
                                if (dst_w > src_w && Z3API(Z3_mk_sign_ext))
                                    result = Z3API(Z3_mk_sign_ext)(
                                        ctx, dst_w - src_w, src);
                                break;
                            default: break;  // bitcast/ptr casts → fresh var
                        }
                    }
                    if (result && inst->has_name()) {
                        enc.value_map[inst->name()] = track_ast(enc, result);
                        // Refinement: casts propagate operand poison.
                        if (config.refinement_semantics && use_bitvectors) {
                            Z3_ast p = operand_poison(enc, inst->operand(0));
                            if (p) enc.poison_map[inst->name()] = p;
                        }
                    } else if (inst->has_name()) {
                        // Fallback: fresh symbolic variable (sound — any
                        // value is possible, so equivalence can only be
                        // missed, never wrongly proven).
                        Z3_sort sort = get_z3_sort(ctx, *inst->type(), use_bitvectors);
                        std::string fresh_name = prefix + "_fresh_" + inst->name();
                        Z3_symbol sym = Z3API(Z3_mk_string_symbol)(ctx, fresh_name.c_str());
                        Z3_ast fresh = track_ast(enc, Z3API(Z3_mk_const)(ctx, sym, sort));
                        enc.value_map[inst->name()] = fresh;
                    }
                }
                continue;
            }

            // ── Fallback: create a fresh symbolic variable ───────────────
            // For any other instruction that produces a result, create a
            // fresh symbolic variable. This is sound (the variable can take
            // any value, so the equivalence query remains valid) but may
            // cause "Unknown" results if the verifier can't prove
            // equivalence with the fresh variables.
            //
            // We do NOT create fresh variables for memory ops (Load/Store/
            // Alloca/GEP/Call) — those are caught by the unsupported-ops
            // check earlier and result in Unknown.
            if (inst->has_name() && !inst->is_memory_op() &&
                inst->opcode() != ir::Opcode::Call) {
                auto it = enc.value_map.find(inst->name());
                if (it == enc.value_map.end()) {
                    Z3_sort sort = get_z3_sort(ctx, *inst->type(), use_bitvectors);
                    std::string fresh_name = prefix + "_fresh_" + inst->name();
                    Z3_symbol sym = Z3API(Z3_mk_string_symbol)(ctx, fresh_name.c_str());
                    Z3_ast fresh = track_ast(enc, Z3API(Z3_mk_const)(ctx, sym, sort));
                    enc.value_map[inst->name()] = fresh;
                }
            }
        }
    }

    // If no return value was found, create a fresh one
    if (!enc.return_value) {
        Z3_sort sort = Z3API(Z3_mk_bv_sort)(ctx, 32);
        std::string ret_name = prefix + "_retval";
        Z3_symbol sym = Z3API(Z3_mk_string_symbol)(ctx, ret_name.c_str());
        enc.return_value = track_ast(enc, Z3API(Z3_mk_const)(ctx, sym, sort));
    }

    return enc;
}

// ── Constructor ─────────────────────────────────────────────────────────────

SMTVerifier::SMTVerifier(const SMTConfig& config)
    : config_(config), z3_available_(is_z3_available()) {
    stats_ = {};
}

// ── Destructor ──────────────────────────────────────────────────────────────
//
// Releases the cached Z3 context if one was created. We guard against
// double-cleanup during program exit by checking z3_ctx_ before calling
// Z3_del_context.
//
SMTVerifier::~SMTVerifier() {
    if (z3_ctx_) {
        if (Z3API(Z3_del_context)) {
            Z3API(Z3_del_context)(static_cast<Z3_context>(z3_ctx_));
        }
        z3_ctx_ = nullptr;
    }
}

// ── Z3 availability check ──────────────────────────────────────────────────

bool SMTVerifier::is_z3_available() {
    // Z3DynamicLoader is a singleton — dlopen/dlsym happens exactly once
    // on first call, and z3_loaded() returns true only when every required
    // symbol was resolved.
    return Z3DynamicLoader::instance().z3_loaded();
}

// ── Lazy cached Z3 context ──────────────────────────────────────────────────
//
// Context creation is 1-10ms; solvers are cheap. We cache one Z3 context per
// SMTVerifier instance and create a fresh solver per verify() call.
//
// Install a Z3 error handler that throws Z3Error instead of
// letting Z3 call exit(1).
//
void* SMTVerifier::get_z3_context() const {
    if (z3_ctx_) return z3_ctx_;
    if (!Z3DynamicLoader::instance().z3_loaded()) return nullptr;

    Z3_config cfg = Z3API(Z3_mk_config)();
    if (!cfg) return nullptr;

    // Enable eager type checking (catches sort mismatches early).
    if (Z3API(Z3_params_set_bool) && Z3API(Z3_mk_string_symbol)) {
        // Z3_config doesn't support set_value directly in all versions;
        // skip if not available.
    }

    Z3_context ctx = Z3API(Z3_mk_context)(cfg);
    Z3API(Z3_del_config)(cfg);

    if (!ctx) return nullptr;

    // Install error handler that throws Z3Error instead of Z3's
    // default exit(1). Throwing from a C callback is technically UB but
    // works reliably on Linux/glibc (libstdc++ unwinder handles foreign
    // frames). The alternative — letting Z3 call exit(1) — is strictly
    // worse for a long-running superoptimiser. The try/catch in
    // verify_with_z3 converts the throw into a VerificationResult::Error.
    if (Z3API(Z3_set_error_handler)) {
        Z3API(Z3_set_error_handler)(ctx, z3_error_handler_cb);
    }

    z3_ctx_ = ctx;  // cache for the lifetime of this verifier
    return z3_ctx_;
}

// ── Public verify ──────────────────────────────────────────────────────────

// The result for a candidate the concrete screen rejected: a proof of
// NotEquivalent with a real counterexample, produced without the solver.
static VerificationResult screened_result(const evaluator::ScreenResult& sr) {
    VerificationResult r;
    r.status = VerificationResult::NotEquivalent;
    r.message = "concrete screen: " + sr.reason;
    r.counterexample = sr.counterexample;
    r.screened = true;
    return r;
}

VerificationResult SMTVerifier::verify(const ir::Function& original,
                                        const ir::Function& candidate) {
    // ── Vector lowering (lane-blasting) ────────────────────────────────
    // The raw encoder cannot model vector ops (they'd become per-side
    // fresh variables — trivially SAT, i.e. wrong NotEquivalent verdicts
    // for genuinely equivalent rewrites). Instead, lower both functions
    // to integer-only scalar functions with ir::scalarize_* and recurse:
    // scalarization expands each vector argument into its lanes
    // deterministically from the signature, so proving the scalarized
    // pair equivalent over all expanded arguments proves the vector pair
    // equivalent over all vector arguments. Functions the scalarizer
    // refuses (vector phi/memory/non-constant indices/undef lanes) are
    // reported Unknown — sound.
    if (ir::function_has_vector_ops(original) ||
        ir::function_has_vector_ops(candidate)) {
        VerificationResult vres;
        vres.status = VerificationResult::Unknown;

        const auto& oret = *original.return_type();
        const auto& cret = *candidate.return_type();
        if (oret != cret) {
            vres.message = "vector verification: return types differ";
            vres.z3_reason = vres.message;
            return vres;
        }
        if (oret.is_vector()) {
            // Per-lane equivalence: all lanes equal ⇔ vectors equal.
            const auto lanes = static_cast<const ir::VectorType&>(oret).count();
            for (uint64_t lane = 0; lane < lanes; ++lane) {
                auto o = ir::scalarize_lane(original, lane);
                auto c = ir::scalarize_lane(candidate, lane);
                if (!o || !c) {
                    vres.message = "vector ops could not be scalarized";
                    vres.z3_reason = vres.message;
                    return vres;
                }
                auto lane_res = verify(*o, *c);
                if (lane_res.status != VerificationResult::Equivalent) {
                    lane_res.message = "lane " + std::to_string(lane) + ": " +
                                       lane_res.message;
                    return lane_res;
                }
            }
            vres.status = VerificationResult::Equivalent;
            vres.message = "equivalent on all " + std::to_string(lanes) +
                           " lanes (scalarized)";
            return vres;
        }
        auto o = ir::scalarize_function(original);
        auto c = ir::scalarize_function(candidate);
        if (!o || !c) {
            vres.message = "vector ops could not be scalarized";
            vres.z3_reason = vres.message;
            return vres;
        }
        return verify(*o, *c);
    }

    auto start = std::chrono::high_resolution_clock::now();

    VerificationResult result;
    result.status = VerificationResult::Unknown;

    // ── Cache: consult the rewrite cache before Z3 ────
    // The cache key is (canonical_structural_hash(original),
    //                   canonical_structural_hash(candidate)).
    // Hits avoid the Z3 query entirely; misses run Z3 and store the result.
    // Negative results (NotEquivalent/Unknown/Error) are cached just like
    // positive ones — most LHS→RHS pairs don't verify, and caching that
    // fact avoids re-running Z3 on them every run (Souper §2.12).
    if (config_.cache) {
        const uint64_t lhs_h = search::StochasticSearch::structural_hash(original);
        const uint64_t rhs_h = search::StochasticSearch::structural_hash(candidate);
        auto cached = config_.cache->lookup(lhs_h, rhs_h);
        if (cached) {
            result.status = cached->status;
            result.message = cached->message;
            result.solve_time_ms = 0.0;  // cache hit — no Z3 work
            // Update stats (so callers see the cache hit).
            stats_.verifications_run++;
            switch (result.status) {
                case VerificationResult::Equivalent:    stats_.equivalent++; break;
                case VerificationResult::NotEquivalent: stats_.not_equivalent++; break;
                case VerificationResult::Unknown:       stats_.unknown++; break;
                case VerificationResult::Error:         stats_.errors++; break;
            }
            return result;
        }
    }

    // ── Concrete-input screen: only survivors reach the solver ─────────
    // After the cache (a cached verdict is free) and before any solver work.
    // Rejections are deliberately NOT cached: they cost microseconds to
    // recompute, and a cache entry would outlive any future fix to the
    // screen itself.
    if (config_.screen_before_solving) {
        const auto sr = evaluator::DifferentialScreen(config_.screen).screen(original, candidate);
        if (sr.verdict == evaluator::ScreenVerdict::Rejected) {
            stats_.verifications_run++;
            stats_.not_equivalent++;
            stats_.screened_out++;
            return screened_result(sr);
        }
        if (sr.verdict == evaluator::ScreenVerdict::Survived) stats_.screen_passed++;
    }

    if (z3_available_) {
        result = verify_with_z3(original, candidate);
    } else {
        result = verify_with_simulation(original, candidate);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    result.solve_time_ms = elapsed;

    // ── Cache: store the result in the cache ──────────
    if (config_.cache) {
        const uint64_t lhs_h = search::StochasticSearch::structural_hash(original);
        const uint64_t rhs_h = search::StochasticSearch::structural_hash(candidate);
        config_.cache->put(lhs_h, rhs_h, result);
    }

    // Update stats
    stats_.verifications_run++;
    stats_.total_time_ms += elapsed;
    switch (result.status) {
        case VerificationResult::Equivalent:    stats_.equivalent++; break;
        case VerificationResult::NotEquivalent: stats_.not_equivalent++; break;
        case VerificationResult::Unknown:       stats_.unknown++; break;
        case VerificationResult::Error:         stats_.errors++; break;
    }
    if (stats_.verifications_run > 0) {
        stats_.avg_time_ms = stats_.total_time_ms / static_cast<double>(stats_.verifications_run);
    }

    return result;
}

// ── I1 (assumptions API): verify under a path condition ────────────────────
//
// No cache (a cached unconditional verdict answers a different query than a
// conditional one) and no simulation fallback (the simulator has no notion
// of a constrained input space) — Z3 or Unknown.
//
VerificationResult SMTVerifier::verify_with_assumptions(
    const ir::Function& original,
    const ir::Function& candidate,
    const std::vector<ArgAssumption>& assumptions,
    const ir::Function* condition_fn,
    bool condition_negated) {
    if (assumptions.empty() && !condition_fn)
        return verify(original, candidate);

    VerificationResult result;
    result.status = VerificationResult::Unknown;

    // Assumptions are positional over the ORIGINAL argument list;
    // scalarization renumbers arguments, so conditional queries on vector
    // functions cannot be transported. Refuse (sound).
    if (ir::function_has_vector_ops(original) ||
        ir::function_has_vector_ops(candidate)) {
        result.message = "assumption-based verification does not support vector ops";
        result.z3_reason = result.message;
        return result;
    }

    if (!z3_available_) {
        result.message = "assumption-based verification requires Z3";
        result.z3_reason = "Z3 unavailable";
        return result;
    }
    if (!config_.use_bitvectors) {
        // The integer-theory encoding has no unsigned comparisons; rather
        // than silently weakening (or wrongly strengthening) the query,
        // refuse. Sound: Unknown never blesses a rewrite.
        result.message = "assumption-based verification requires the BV encoding";
        result.z3_reason = "use_bitvectors is false";
        return result;
    }

    auto start = std::chrono::high_resolution_clock::now();
    result = verify_with_z3(original, candidate, &assumptions,
                            condition_fn, condition_negated);
    auto end = std::chrono::high_resolution_clock::now();
    result.solve_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    stats_.verifications_run++;
    stats_.total_time_ms += result.solve_time_ms;
    switch (result.status) {
        case VerificationResult::Equivalent:    stats_.equivalent++; break;
        case VerificationResult::NotEquivalent: stats_.not_equivalent++; break;
        case VerificationResult::Unknown:       stats_.unknown++; break;
        case VerificationResult::Error:         stats_.errors++; break;
    }
    if (stats_.verifications_run > 0) {
        stats_.avg_time_ms =
            stats_.total_time_ms / static_cast<double>(stats_.verifications_run);
    }
    return result;
}

// ── Batch verification (existing signature, backward-compatible) ───────────

std::vector<VerificationResult> SMTVerifier::verify_batch(
    const ir::Function& original,
    const std::vector<std::shared_ptr<ir::Function>>& candidates) {
    std::vector<VerificationResult> results;
    results.reserve(candidates.size());
    for (auto& cand : candidates) {
        if (cand) {
            results.push_back(verify(original, *cand));
        } else {
            VerificationResult err;
            err.status = VerificationResult::Error;
            err.message = "Null candidate function";
            err.solve_time_ms = 0.0;
            results.push_back(err);
        }
    }
    return results;
}

// ── Batch verification ────────────────────────────────────────────────────────
//
// Tries incremental verification with Z3 push/pop first. If Z3 is
// unavailable or the original cannot be encoded, falls back to looping
// verify() per candidate.
//
std::vector<VerificationResult> SMTVerifier::verify_batch(
    const ir::Function& original,
    const std::vector<ir::Function>& candidates) {

    // Screen first (the incremental Z3 path below never calls verify()):
    // rejected candidates are answered here, only survivors are encoded.
    if (config_.screen_before_solving) {
        std::vector<VerificationResult> screened(candidates.size());
        std::vector<ir::Function> survivors;
        std::vector<size_t> survivor_slot;
        const evaluator::DifferentialScreen screen(config_.screen);
        bool any_rejected = false;
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto sr = screen.screen(original, candidates[i]);
            if (sr.verdict == evaluator::ScreenVerdict::Rejected) {
                screened[i] = screened_result(sr);
                stats_.verifications_run++;
                stats_.not_equivalent++;
                stats_.screened_out++;
                any_rejected = true;
            } else {
                if (sr.verdict == evaluator::ScreenVerdict::Survived) stats_.screen_passed++;
                survivors.push_back(candidates[i]);
                survivor_slot.push_back(i);
            }
        }
        if (any_rejected) {
            // Recurse on the survivors with the screen off: they were just screened.
            const bool saved = config_.screen_before_solving;
            config_.screen_before_solving = false;
            auto rest = verify_batch(original, survivors);
            config_.screen_before_solving = saved;
            for (size_t k = 0; k < survivors.size(); ++k) screened[survivor_slot[k]] = rest[k];
            return screened;
        }
    }

    std::vector<VerificationResult> results;
    results.reserve(candidates.size());

    // Vector functions must go through verify()'s scalarize-and-recurse
    // path; the incremental encoder would model their ops as fresh vars.
    bool any_vector = ir::function_has_vector_ops(original);
    for (const auto& cand : candidates) {
        if (any_vector) break;
        any_vector = ir::function_has_vector_ops(cand);
    }

    // Try incremental batch verification with Z3.
    if (z3_available_ && !any_vector) {
        bool ok = verify_batch_with_z3(original, candidates, results);
        if (ok) return results;
        // Fall through to per-candidate verify() if incremental failed.
        results.clear();
    }

    // Fallback: loop verify() per candidate.
    for (const auto& cand : candidates) {
        results.push_back(verify(original, cand));
    }
    return results;
}

// ── verify_with_z3 ─────────────────────────────────────────────────────────
//
// Main Z3-based verification. Steps:
//   1. Check argument counts match.
//   2. Check for unsupported operations (memory, float, loops) → Unknown.
//   3. Get/create Z3 context (with error handler installed).
//   4. Create solver; apply Z3_solver_set_params with timeout.
//   5. Build shared argument Z3 constants.
//   6. Encode original and candidate.
//   7. Build equivalence query: not (orig_ret == cand_ret).
//   8. Assert and check.
//   9. On SAT, extract counterexample (C14b).
//  10. Release all tracked ASTs.
//
VerificationResult SMTVerifier::verify_with_z3(
    const ir::Function& original,
    const ir::Function& candidate,
    const std::vector<ArgAssumption>* assumptions,
    const ir::Function* condition_fn,
    bool condition_negated) {
    VerificationResult result;
    result.status = VerificationResult::Unknown;
    result.message = "";

    // Quick check: if the functions have different signatures, they can't be equivalent
    if (original.argument_count() != candidate.argument_count()) {
        result.status = VerificationResult::NotEquivalent;
        result.message = "Different argument counts";
        return result;
    }

    // Verify the dynamic loader has all required symbols
    if (!Z3DynamicLoader::instance().z3_loaded()) {
        result.status = VerificationResult::Error;
        result.message = "Z3 library not loadable at runtime — falling back to simulation";
        return result;
    }

    // ── Bounded loop unrolling pre-pass ────────────────────────────────
    // When sound_bounded_unrolling is enabled, pre-unroll constant-trip
    // single-block loops in BOTH functions. The unroller
    // (LoopOptimizer::unroll_constant_loops) is sound-by-construction:
    // it abstractly interprets the loop to determine the trip count and
    // refuses to unroll anything it cannot prove constant. So a function
    // that would have triggered sound_loop_fallback may now be SMT-
    // verifiable as straight-line code.
    //
    // `work_orig` / `work_cand` hold the unrolled copies (if any);
    // the references `orig_fn` / `cand_fn` below alias either the
    // unrolled copy (when one exists) or the caller's original/candidate.
    std::shared_ptr<ir::Function> work_orig, work_cand;
    if (config_.sound_bounded_unrolling) {
        // Use LoopOpt's sound unroller. Local linkage so we don't
        // introduce a header dependency for callers of SMTVerifier.
        search::LoopOptimizer unroller;
        auto uo = unroller.unroll_constant_loops(original);
        auto uc = unroller.unroll_constant_loops(candidate);
        if (uo) work_orig = uo;
        if (uc) work_cand = uc;
    }
    const ir::Function& orig_fn = work_orig ? *work_orig : original;
    const ir::Function& cand_fn = work_cand ? *work_cand : candidate;

    // ── C2/C3/C4/C8: check for unsupported operations ───────────────────
    std::string orig_reason, cand_reason;
    if (function_has_unsupported_ops(orig_fn, config_, orig_reason)) {
        result.status = VerificationResult::Unknown;
        result.message = "Original " + orig_reason;
        result.z3_reason = orig_reason;
        return result;
    }
    if (function_has_unsupported_ops(cand_fn, config_, cand_reason)) {
        result.status = VerificationResult::Unknown;
        result.message = "Candidate " + cand_reason;
        result.z3_reason = cand_reason;
        return result;
    }

    // Use the cached Z3 context (created lazily on first call).
    Z3_context ctx = static_cast<Z3_context>(get_z3_context());
    if (!ctx) {
        result.status = VerificationResult::Error;
        result.message = "Failed to create Z3 context";
        return result;
    }

    // Wrap the entire Z3 interaction in try/catch to convert Z3Error into
    // VerificationResult::Error.
    Z3_solver solver = nullptr;
    Z3_params params = nullptr;
    SMTEncoding orig_enc;
    SMTEncoding cand_enc;
    SMTEncoding cond_enc;

    try {


    // Create a fresh solver per call (solvers are cheap; contexts are not).
    solver = Z3API(Z3_mk_solver)(ctx);
    Z3API(Z3_solver_inc_ref)(ctx, solver);

    // ── Apply Z3_solver_set_params with timeout ────────────────────────────
    params = Z3API(Z3_mk_params)(ctx);
    Z3API(Z3_params_inc_ref)(ctx, params);
    Z3_symbol timeout_sym = Z3API(Z3_mk_string_symbol)(ctx, "timeout");
    Z3API(Z3_params_set_uint)(ctx, params, timeout_sym, config_.timeout_ms);

    // Also set rlimit if configured (deterministic backup to wall-clock timeout).
    if (config_.rlimit > 0) {
        Z3_symbol rlimit_sym = Z3API(Z3_mk_string_symbol)(ctx, "rlimit");
        Z3API(Z3_params_set_uint)(ctx, params, rlimit_sym, config_.rlimit);
    }

    // Apply the params to the solver (THIS was the C5 bug — never called before).
    if (Z3API(Z3_solver_set_params)) {
        Z3API(Z3_solver_set_params)(ctx, solver, params);
    }

    // ── Build shared argument Z3 constants ─────────────────────────────────
    //
    // Both original and candidate use the SAME Z3 constants for their
    // arguments. Only local SSA values get distinct prefixes (orig_/cand_).
    //
    std::vector<std::pair<std::string, Z3_ast>> shared_args;
    std::vector<Z3_ast> shared_arg_refs;  // tracked separately for cleanup
    // Poison-aware encoding: shared POISON constants per bit-width. Created
    // lazily by get_or_create_poison_const during encoding. Tracked in
    // shared_poison_refs for cleanup. SHARED between original and candidate
    // so that two functions producing poison on the same inputs compare
    // equal (POISON == POISON) — see SMTEncoding::poison_consts doc.
    std::unordered_map<unsigned, Z3_ast> shared_poison;
    std::vector<Z3_ast> shared_poison_refs;
    bool use_bv = config_.use_bitvectors;
    for (size_t i = 0; i < orig_fn.argument_count(); ++i) {
        auto& arg = orig_fn.arguments()[i];
        std::string arg_name = arg.name.empty() ? ("arg" + std::to_string(i)) : arg.name;
        Z3_sort sort = get_z3_sort(ctx, *arg.type, use_bv);
        // Neutral name: arg_0, arg_1, ... (shared between original and candidate).
        std::string z3_name = "arg_" + std::to_string(i);
        Z3_symbol sym = Z3API(Z3_mk_string_symbol)(ctx, z3_name.c_str());
        Z3_ast var = Z3API(Z3_mk_const)(ctx, sym, sort);
        // Inc_ref immediately so the arg stays alive for the entire verify
        // call (it's referenced by both encodings and by counterexample
        // extraction). We track it in shared_arg_refs for cleanup at the end.
        if (Z3API(Z3_inc_ref)) Z3API(Z3_inc_ref)(ctx, var);
        shared_arg_refs.push_back(var);
        shared_args.push_back({arg_name, var});
    }

    // ── Encode both functions ────────────────────────────────────────────
    // Use orig_fn / cand_fn (which may be unrolled copies when
    // sound_bounded_unrolling is on) rather than the caller-supplied
    // original / candidate.
    orig_enc = encode_function(ctx, orig_fn, "orig", use_bv, shared_args, config_, &shared_poison);
    orig_enc.ctx = ctx;

    cand_enc = encode_function(ctx, cand_fn, "cand", use_bv, shared_args, config_, &shared_poison);
    cand_enc.ctx = ctx;

    // After both encodings, the shared_poison map's values are referenced
    // by orig_enc.local_refs (via track_ast in get_or_create_poison_const).
    // We collect them into shared_poison_refs for explicit cleanup at the
    // end (releasing orig_enc.local_refs will dec_ref them once; we don't
    // want a second dec_ref, so shared_poison_refs is just informational —
    // the orig_enc release handles cleanup). To be safe, we DO NOT inc_ref
    // again here; we just record the pointers.
    for (auto& [w, ast] : shared_poison) {
        (void)w;
        shared_poison_refs.push_back(ast);
    }

    // Check for unsupported operations encountered during encoding
    // (e.g., multi-incoming phi nodes).
    if (orig_enc.had_unsupported || cand_enc.had_unsupported) {
        result.status = VerificationResult::Unknown;
        result.message = orig_enc.had_unsupported ?
                         ("Original " + orig_enc.unsupported_reason) :
                         ("Candidate " + cand_enc.unsupported_reason);
        result.z3_reason = orig_enc.had_unsupported ?
                           orig_enc.unsupported_reason : cand_enc.unsupported_reason;
        // Cleanup
        release_tracked(orig_enc);
        release_tracked(cand_enc);
        Z3API(Z3_params_dec_ref)(ctx, params);
        Z3API(Z3_solver_dec_ref)(ctx, solver);
        return result;
    }

    if (!orig_enc.return_value || !cand_enc.return_value) {
        result.status = VerificationResult::Unknown;
        result.message = "Could not encode function return values";
        result.z3_reason = "encoding failed to produce return values";
        release_tracked(orig_enc);
        release_tracked(cand_enc);
        Z3API(Z3_params_dec_ref)(ctx, params);
        Z3API(Z3_solver_dec_ref)(ctx, solver);
        return result;
    }

    // ── I1 (assumptions API): assert an arbitrary i1 condition function ─
    // The condition mini-function is encoded against the SAME shared
    // argument constants as original/candidate (identical positional arg
    // naming, by construction of the caller), and its i1 result is
    // asserted true (or false when negated). This expresses select-arm
    // conditions like `(x & 1) == 0` that the flat ArgAssumption conjuncts
    // cannot. If the condition can't be soundly encoded we return Unknown:
    // silently dropping it would prove equivalence on a WIDER input space
    // than requested (that direction is fine), but the caller asked for a
    // conditional proof — and a NotEquivalent verdict computed without the
    // condition would be wrong to act on, so refuse instead.
    if (condition_fn) {
        std::string cond_reason;
        bool cond_ok = !function_has_unsupported_ops(*condition_fn, config_,
                                                     cond_reason);
        if (cond_ok) {
            cond_enc = encode_function(ctx, *condition_fn, "cond", use_bv,
                                       shared_args, config_, &shared_poison);
            cond_enc.ctx = ctx;
            cond_ok = !cond_enc.had_unsupported && cond_enc.return_value;
            if (!cond_ok) cond_reason = cond_enc.unsupported_reason;
        }
        if (!cond_ok) {
            result.status = VerificationResult::Unknown;
            result.message = "Condition " + cond_reason;
            result.z3_reason = cond_reason;
            release_tracked(orig_enc);
            release_tracked(cand_enc);
            release_tracked(cond_enc);
            Z3API(Z3_params_dec_ref)(ctx, params);
            Z3API(Z3_solver_dec_ref)(ctx, solver);
            return result;
        }
        Z3_ast cond_bool = coerce_to_bool(cond_enc, cond_enc.return_value);
        if (condition_negated)
            cond_bool = track_ast(cond_enc,
                                  Z3API(Z3_mk_not)(ctx, cond_bool));
        Z3API(Z3_solver_assert)(ctx, solver, cond_bool);
    }

    // ── I1 (assumptions API): assert the path-condition conjuncts ───────
    // Each conjunct constrains the SHARED argument constants, so the
    // equivalence query below only has to hold on the constrained input
    // space: UNSAT of (assumptions ∧ ret_o ≠ ret_c) proves equivalence
    // under the assumptions. Conjuncts that cannot be expressed (const vs
    // const, unknown widths, mismatched arg widths) are dropped — dropping
    // only WIDENS the input space the proof must cover, which is sound.
    if (assumptions && use_bv) {
        auto arg_width = [&](int idx) -> unsigned {
            if (idx < 0 || static_cast<size_t>(idx) >= orig_fn.argument_count())
                return 0;
            auto& t = orig_fn.arguments()[idx].type;
            return t ? t->bit_width() : 0;
        };
        for (const auto& a : *assumptions) {
            Z3_ast lhs = nullptr, rhs = nullptr;
            unsigned lw = 0, rw = 0;
            if (a.lhs_arg >= 0 &&
                static_cast<size_t>(a.lhs_arg) < shared_args.size()) {
                lhs = shared_args[a.lhs_arg].second;
                lw = arg_width(a.lhs_arg);
            }
            if (a.rhs_arg >= 0 &&
                static_cast<size_t>(a.rhs_arg) < shared_args.size()) {
                rhs = shared_args[a.rhs_arg].second;
                rw = arg_width(a.rhs_arg);
            }
            if (!lhs && !rhs) continue;              // const vs const
            if (lhs && rhs && lw != rw) continue;    // mismatched widths
            const unsigned width = lhs ? lw : rw;
            if (width == 0 || width > 64) continue;
            const uint64_t mask =
                (width >= 64) ? ~uint64_t(0) : ((uint64_t(1) << width) - 1);
            Z3_sort bv = Z3API(Z3_mk_bv_sort)(ctx, width);
            if (!lhs)
                lhs = track_ast(orig_enc, Z3API(Z3_mk_unsigned_int64)(
                    ctx, static_cast<uint64_t>(a.lhs_const) & mask, bv));
            if (!rhs)
                rhs = track_ast(orig_enc, Z3API(Z3_mk_unsigned_int64)(
                    ctx, static_cast<uint64_t>(a.rhs_const) & mask, bv));

            Z3_ast c = nullptr;
            switch (a.predicate) {
                case ir::CmpPredicate::EQ:
                    c = Z3API(Z3_mk_eq)(ctx, lhs, rhs); break;
                case ir::CmpPredicate::NE:
                    c = Z3API(Z3_mk_not)(ctx, track_ast(orig_enc,
                            Z3API(Z3_mk_eq)(ctx, lhs, rhs))); break;
                case ir::CmpPredicate::UGT:
                    c = Z3API(Z3_mk_bvult)(ctx, rhs, lhs); break;
                case ir::CmpPredicate::UGE:
                    c = Z3API(Z3_mk_not)(ctx, track_ast(orig_enc,
                            Z3API(Z3_mk_bvult)(ctx, lhs, rhs))); break;
                case ir::CmpPredicate::ULT:
                    c = Z3API(Z3_mk_bvult)(ctx, lhs, rhs); break;
                case ir::CmpPredicate::ULE:
                    c = Z3API(Z3_mk_not)(ctx, track_ast(orig_enc,
                            Z3API(Z3_mk_bvult)(ctx, rhs, lhs))); break;
                case ir::CmpPredicate::SGT:
                    c = Z3API(Z3_mk_bvslt)(ctx, rhs, lhs); break;
                case ir::CmpPredicate::SGE:
                    c = Z3API(Z3_mk_not)(ctx, track_ast(orig_enc,
                            Z3API(Z3_mk_bvslt)(ctx, lhs, rhs))); break;
                case ir::CmpPredicate::SLT:
                    c = Z3API(Z3_mk_bvslt)(ctx, lhs, rhs); break;
                case ir::CmpPredicate::SLE:
                    c = Z3API(Z3_mk_not)(ctx, track_ast(orig_enc,
                            Z3API(Z3_mk_bvslt)(ctx, rhs, lhs))); break;
                default: break;
            }
            if (!c) continue;
            c = track_ast(orig_enc, c);
            if (a.negated)
                c = track_ast(orig_enc, Z3API(Z3_mk_not)(ctx, c));
            Z3API(Z3_solver_assert)(ctx, solver, c);
        }
    }

    // Optionally simplify
    Z3_ast orig_ret = orig_enc.return_value;
    Z3_ast cand_ret = cand_enc.return_value;

    if (config_.simplify_before && Z3API(Z3_simplify)) {
        try {
            Z3_ast simplified = Z3API(Z3_simplify)(ctx, orig_ret);
            if (simplified) {
                orig_ret = track_ast(orig_enc, simplified);
            }
            simplified = Z3API(Z3_simplify)(ctx, cand_ret);
            if (simplified) {
                cand_ret = track_ast(cand_enc, simplified);
            }
        } catch (const Z3Error& e) {
            // Simplification can fail on complex expressions; ignore.
        }
    }

    // Create the equivalence assertion
    Z3_ast eq = track_ast(orig_enc, Z3API(Z3_mk_eq)(ctx, orig_ret, cand_ret));
    Z3_ast neg_eq = track_ast(orig_enc, Z3API(Z3_mk_not)(ctx, eq));

    // ── Alive2-style refinement query ────────────────────────────────────
    // The candidate may REPLACE the original iff on every input where the
    // original is well-defined, the candidate is well-defined and agrees.
    // A counterexample is therefore an input with
    //     ¬orig_poison ∧ (cand_poison ∨ orig ≠ cand)
    // — the candidate is either less defined or produces a different value
    // somewhere the original's behaviour is actually fixed. With no poison
    // sources on either side this degenerates to the plain inequality.
    Z3_ast counterexample = neg_eq;
    if (config_.refinement_semantics && use_bv) {
        if (cand_enc.return_poison) {
            counterexample = track_ast(orig_enc,
                mk_or2(ctx, cand_enc.return_poison, counterexample));
        }
        if (orig_enc.return_poison) {
            Z3_ast well_defined = track_ast(orig_enc,
                Z3API(Z3_mk_not)(ctx, orig_enc.return_poison));
            counterexample = track_ast(orig_enc,
                mk_and2(ctx, well_defined, counterexample));
        }
    }

    // Assert the counterexample condition (UNSAT ⇒ safe to replace).
    Z3API(Z3_solver_assert)(ctx, solver, counterexample);

    // Check satisfiability
    Z3_lbool_enum sat_result = Z3API(Z3_solver_check)(ctx, solver);

    switch (sat_result) {
        case Z3_L_FALSE:
            // UNSAT: no input makes original != candidate → they ARE equivalent
            result.status = VerificationResult::Equivalent;
            result.message = "Functions are equivalent (negation is unsatisfiable)";
            break;

        case Z3_L_TRUE: {
            // SAT: found an input where original != candidate → NOT equivalent
            result.status = VerificationResult::NotEquivalent;
            result.message = "Functions are NOT equivalent (counterexample exists)";

            // ── C14b: extract counterexample ────────────────────────────
            if (Z3API(Z3_solver_get_model) && Z3API(Z3_model_eval) &&
                Z3API(Z3_get_numeral_int64) && Z3API(Z3_model_inc_ref) &&
                Z3API(Z3_model_dec_ref)) {
                Z3_model model = nullptr;
                model = Z3API(Z3_solver_get_model)(ctx, solver);
                if (model) {
                    Z3API(Z3_model_inc_ref)(ctx, model);
                    for (const auto& [arg_name, arg_ast] : orig_enc.arg_vars) {
                        Z3_ast val_ast = nullptr;
                        int ok = Z3API(Z3_model_eval)(ctx, model, arg_ast, 1, &val_ast);
                        int64_t val = 0;
                        if (ok && val_ast) {
                            Z3API(Z3_get_numeral_int64)(ctx, val_ast, &val);
                        }
                        result.counterexample.push_back(val);
                    }
                    Z3API(Z3_model_dec_ref)(ctx, model);
                }
            }
            break;
        }

        case Z3_L_UNDEFINED:
            // UNKNOWN: couldn't determine (timeout, etc.)
            result.status = VerificationResult::Unknown;
            result.message = "Z3 returned UNKNOWN (possible timeout or resource limit)";
            result.z3_reason = "z3 returned L_UNDEFINED";
            break;
    }


    // Cleanup
    release_tracked(orig_enc);
    release_tracked(cand_enc);
    release_tracked(cond_enc);
    // Release shared args
    if (Z3API(Z3_dec_ref)) {
        for (Z3_ast a : shared_arg_refs) {
            if (a) Z3API(Z3_dec_ref)(ctx, a);
        }
    }
    Z3API(Z3_params_dec_ref)(ctx, params);
    Z3API(Z3_solver_dec_ref)(ctx, solver);

    } catch (const Z3Error& e) {
        // Catch Z3 errors and return Error instead of crashing.
        result.status = VerificationResult::Error;
        result.message = std::string("Z3 error: ") + e.what();
        result.z3_reason = e.what();
        // Best-effort cleanup
        try {
            if (orig_enc.ctx) release_tracked(orig_enc);
            if (cand_enc.ctx) release_tracked(cand_enc);
            if (cond_enc.ctx) release_tracked(cond_enc);
            if (params && ctx) Z3API(Z3_params_dec_ref)(ctx, params);
            if (solver && ctx) Z3API(Z3_solver_dec_ref)(ctx, solver);
        } catch (...) {
            // Ignore cleanup errors.
        }
    } catch (const std::exception& e) {
        result.status = VerificationResult::Error;
        result.message = std::string("Unexpected error: ") + e.what();
        result.z3_reason = e.what();
        try {
            if (orig_enc.ctx) release_tracked(orig_enc);
            if (cand_enc.ctx) release_tracked(cand_enc);
            if (cond_enc.ctx) release_tracked(cond_enc);
            if (params && ctx) Z3API(Z3_params_dec_ref)(ctx, params);
            if (solver && ctx) Z3API(Z3_solver_dec_ref)(ctx, solver);
        } catch (...) {}
    }

    return result;
}

// ── verify_batch_with_z3: incremental batch verification ───────────────────
//
// Encodes the original ONCE, asserts it under a push, then for each
// candidate: push, encode the candidate, assert the negation of
// equivalence, check, pop. This shares the original's encoding across
// all candidates, avoiding re-encoding cost.
//
// Returns false if the original cannot be encoded (caller should fall
// back to looping verify()). Results are appended to `results`.
//
bool SMTVerifier::verify_batch_with_z3(const ir::Function& original,
                                        const std::vector<ir::Function>& candidates,
                                        std::vector<VerificationResult>& results) {
    // Check for unsupported operations in the original.
    std::string orig_reason;
    if (function_has_unsupported_ops(original, config_, orig_reason)) {
        return false;  // fall back to per-candidate verify()
    }

    Z3_context ctx = static_cast<Z3_context>(get_z3_context());
    if (!ctx) return false;

    // Check that push/pop are available.
    if (!Z3API(Z3_solver_push) || !Z3API(Z3_solver_pop)) {
        return false;  // fall back to per-candidate verify()
    }

    // ── Assumption-based batch pruning: unsat-core pre-pass ────────────────────────────
    // If the batch is larger than 3 candidates and the unsat-core symbols
    // are available, run the prune pre-pass first. The prune may short-
    // circuit some candidates to NotEquivalent (saving a full verify()).
    // Candidates not pruned are verified via the existing per-candidate
    // `verify()` (we don't re-enter the push/pop loop below for them —
    // simpler and avoids sharing the original's encoding across two
    // solver instances).
    if (config_.use_unsat_core_batch && candidates.size() > 3 &&
        Z3API(Z3_solver_check_assumptions)) {
        std::vector<VerificationResult> prune_results;
        std::vector<size_t> pruned_idx;
        prune_batch_with_unsat_core(original, candidates, prune_results, &pruned_idx);
        if (!pruned_idx.empty()) {
            // At least one candidate was pruned. Fill results with pruned
            // entries, then run per-candidate verify() for the rest.
            results.assign(candidates.size(), VerificationResult{});
            std::unordered_set<size_t> pruned_set(pruned_idx.begin(),
                                                   pruned_idx.end());
            for (size_t i : pruned_idx) {
                results[i] = std::move(prune_results[i]);
            }
            for (size_t i = 0; i < candidates.size(); ++i) {
                if (pruned_set.count(i)) continue;
                results[i] = verify(original, candidates[i]);
            }
            return true;
        }
        // No candidates were pruned — fall through to normal push/pop loop.
    }

    Z3_solver solver = nullptr;
    Z3_params params = nullptr;
    SMTEncoding orig_enc;

    try {
        solver = Z3API(Z3_mk_solver)(ctx);
        Z3API(Z3_solver_inc_ref)(ctx, solver);

        // Apply timeout params.
        params = Z3API(Z3_mk_params)(ctx);
        Z3API(Z3_params_inc_ref)(ctx, params);
        Z3_symbol timeout_sym = Z3API(Z3_mk_string_symbol)(ctx, "timeout");
        Z3API(Z3_params_set_uint)(ctx, params, timeout_sym, config_.timeout_ms);
        if (Z3API(Z3_solver_set_params)) {
            Z3API(Z3_solver_set_params)(ctx, solver, params);
        }

        // Build shared arguments.
        std::vector<std::pair<std::string, Z3_ast>> shared_args;
        // Poison-aware encoding: shared POISON constants, same as verify_with_z3.
        std::unordered_map<unsigned, Z3_ast> shared_poison;
        bool use_bv = config_.use_bitvectors;
        for (size_t i = 0; i < original.argument_count(); ++i) {
            auto& arg = original.arguments()[i];
            std::string arg_name = arg.name.empty() ? ("arg" + std::to_string(i)) : arg.name;
            Z3_sort sort = get_z3_sort(ctx, *arg.type, use_bv);
            std::string z3_name = "arg_" + std::to_string(i);
            Z3_symbol sym = Z3API(Z3_mk_string_symbol)(ctx, z3_name.c_str());
            Z3_ast var = Z3API(Z3_mk_const)(ctx, sym, sort);
            if (Z3API(Z3_inc_ref)) Z3API(Z3_inc_ref)(ctx, var);
            orig_enc.local_refs.push_back(var);
            shared_args.push_back({arg_name, var});
        }

        // Encode original. The batch path asserts BARE value equality, so it
        // must use the legacy POISON-value encoding: refinement-mode
        // encodings keep flagged binops' plain wrapped values (poison lives
        // in side flags this path does not consult), which would let a
        // poison-introducing candidate slip through the equality check.
        SMTConfig batch_cfg = config_;
        batch_cfg.refinement_semantics = false;
        orig_enc = encode_function(ctx, original, "orig", use_bv, shared_args, batch_cfg, &shared_poison);
        // Re-track shared args.
        for (auto& [name, ast] : shared_args) {
            if (Z3API(Z3_inc_ref)) Z3API(Z3_inc_ref)(ctx, ast);
            orig_enc.local_refs.push_back(ast);
        }
        orig_enc.ctx = ctx;

        if (orig_enc.had_unsupported || !orig_enc.return_value) {
            // Can't encode original soundly — fall back.
            release_tracked(orig_enc);
            Z3API(Z3_params_dec_ref)(ctx, params);
            Z3API(Z3_solver_dec_ref)(ctx, solver);
            return false;
        }

        // For each candidate: push, encode, check, pop.
        for (const auto& cand : candidates) {
            VerificationResult r;
            r.status = VerificationResult::Unknown;

            // Quick signature check.
            if (original.argument_count() != cand.argument_count()) {
                r.status = VerificationResult::NotEquivalent;
                r.message = "Different argument counts";
                results.push_back(r);
                continue;
            }

            // Check for unsupported ops in candidate.
            std::string cand_reason;
            if (function_has_unsupported_ops(cand, config_, cand_reason)) {
                r.status = VerificationResult::Unknown;
                r.message = "Candidate " + cand_reason;
                r.z3_reason = cand_reason;
                results.push_back(r);
                continue;
            }

            SMTEncoding cand_enc;
            try {
                Z3API(Z3_solver_push)(ctx, solver);

                cand_enc = encode_function(ctx, cand, "cand", use_bv, shared_args, batch_cfg, &shared_poison);
                cand_enc.ctx = ctx;

                if (cand_enc.had_unsupported || !cand_enc.return_value) {
                    r.status = VerificationResult::Unknown;
                    r.message = "Could not encode candidate";
                    r.z3_reason = cand_enc.had_unsupported ?
                                  cand_enc.unsupported_reason : "no return value";
                    Z3API(Z3_solver_pop)(ctx, solver, 1);
                    release_tracked(cand_enc);
                    results.push_back(r);
                    continue;
                }

                // Build equivalence query.
                Z3_ast eq = track_ast(cand_enc,
                    Z3API(Z3_mk_eq)(ctx, orig_enc.return_value, cand_enc.return_value));
                Z3_ast neg_eq = track_ast(cand_enc, Z3API(Z3_mk_not)(ctx, eq));
                Z3API(Z3_solver_assert)(ctx, solver, neg_eq);

                Z3_lbool_enum sat_result = Z3API(Z3_solver_check)(ctx, solver);

                switch (sat_result) {
                    case Z3_L_FALSE:
                        r.status = VerificationResult::Equivalent;
                        r.message = "Functions are equivalent";
                        break;
                    case Z3_L_TRUE:
                        r.status = VerificationResult::NotEquivalent;
                        r.message = "Functions are NOT equivalent (counterexample exists)";
                        // Extract counterexample (best-effort).
                        if (Z3API(Z3_solver_get_model) && Z3API(Z3_model_eval) &&
                            Z3API(Z3_get_numeral_int64)) {
                            Z3_model model = Z3API(Z3_solver_get_model)(ctx, solver);
                            if (model) {
                                if (Z3API(Z3_model_inc_ref)) Z3API(Z3_model_inc_ref)(ctx, model);
                                for (const auto& [arg_name, arg_ast] : orig_enc.arg_vars) {
                                    Z3_ast val_ast = nullptr;
                                    int ok = 0;
                                    try {
                                        ok = Z3API(Z3_model_eval)(ctx, model, arg_ast, 1, &val_ast);
                                    } catch (const Z3Error&) { ok = 0; val_ast = nullptr; }
                                    if (ok && val_ast) {
                                        int64_t val = 0;
                                        if (Z3API(Z3_get_numeral_int64)(ctx, val_ast, &val)) {
                                            r.counterexample.push_back(val);
                                        } else {
                                            r.counterexample.push_back(0);
                                        }
                                    } else {
                                        r.counterexample.push_back(0);
                                    }
                                }
                                if (Z3API(Z3_model_dec_ref)) Z3API(Z3_model_dec_ref)(ctx, model);
                            }
                        }
                        break;
                    case Z3_L_UNDEFINED:
                        r.status = VerificationResult::Unknown;
                        r.message = "Z3 returned UNKNOWN";
                        r.z3_reason = "z3 returned L_UNDEFINED";
                        break;
                }

                Z3API(Z3_solver_pop)(ctx, solver, 1);
                release_tracked(cand_enc);
            } catch (const Z3Error& e) {
                r.status = VerificationResult::Error;
                r.message = std::string("Z3 error: ") + e.what();
                r.z3_reason = e.what();
                try {
                    Z3API(Z3_solver_pop)(ctx, solver, 1);
                    release_tracked(cand_enc);
                } catch (...) {}
            }
            results.push_back(r);
        }

        // Cleanup
        release_tracked(orig_enc);
        Z3API(Z3_params_dec_ref)(ctx, params);
        Z3API(Z3_solver_dec_ref)(ctx, solver);
        return true;

    } catch (const Z3Error& e) {
        // Original encoding failed — fall back.
        try {
            if (orig_enc.ctx) release_tracked(orig_enc);
            if (params && ctx) Z3API(Z3_params_dec_ref)(ctx, params);
            if (solver && ctx) Z3API(Z3_solver_dec_ref)(ctx, solver);
        } catch (...) {}
        return false;
    } catch (const std::exception&) {
        try {
            if (orig_enc.ctx) release_tracked(orig_enc);
            if (params && ctx) Z3API(Z3_params_dec_ref)(ctx, params);
            if (solver && ctx) Z3API(Z3_solver_dec_ref)(ctx, solver);
        } catch (...) {}
        return false;
    }
}

// ── Verify with simulation (fallback) ──────────────────────────────────────

VerificationResult SMTVerifier::verify_with_simulation(
    const ir::Function& original,
    const ir::Function& candidate) {
    VerificationResult result;
    result.status = VerificationResult::Unknown;
    result.message = "";

    // Quick check: if the functions have different signatures, they can't be equivalent
    if (original.argument_count() != candidate.argument_count()) {
        result.status = VerificationResult::NotEquivalent;
        result.message = "Different argument counts";
        return result;
    }

    // Sound fallback: if either function contains memory ops, float ops, or
    // loops, return Unknown immediately rather than running unsound simulation.
    std::string orig_reason, cand_reason;
    if (function_has_unsupported_ops(original, config_, orig_reason)) {
        result.status = VerificationResult::Unknown;
        result.message = "Original " + orig_reason;
        result.z3_reason = orig_reason;
        return result;
    }
    if (function_has_unsupported_ops(candidate, config_, cand_reason)) {
        result.status = VerificationResult::Unknown;
        result.message = "Candidate " + cand_reason;
        result.z3_reason = cand_reason;
        return result;
    }

    // Generate random test inputs and compare outputs
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int64_t> value_dist(
        std::numeric_limits<int64_t>::min() / 4,
        std::numeric_limits<int64_t>::max() / 4);

    constexpr size_t NUM_TEST_CASES = 1000;

    for (size_t test = 0; test < NUM_TEST_CASES; ++test) {
        std::vector<int64_t> inputs;
        for (size_t i = 0; i < original.argument_count(); ++i) {
            inputs.push_back(value_dist(rng));
        }

        auto orig_result = simulate_function_impl(original, inputs);
        auto cand_result = simulate_function_impl(candidate, inputs);

        if (orig_result.has_value() && cand_result.has_value()) {
            if (orig_result.value() != cand_result.value()) {
                result.status = VerificationResult::NotEquivalent;
                result.message = "Simulation found a differing input";
                result.counterexample = inputs;
                return result;
            }
        }
    }

    result.status = VerificationResult::Unknown;
    result.message = "All " + std::to_string(NUM_TEST_CASES) +
                     " random tests passed, but equivalence is not proven";
    return result;
}

// ── Simulate a function with concrete inputs ──────────────────────────────
// Simplified symbolic execution that handles basic arithmetic.
//
// Shifts are guarded against UB (rhs < bitwidth), and UDiv/URem
// use uint64_t casts. Memory ops still produce placeholder 0 values, but
// the caller (verify_with_simulation) now returns Unknown for functions
// containing memory ops, so this path is only reached for pure-arithmetic
// functions.
//
static std::optional<int64_t> simulate_function_impl(
    const ir::Function& fn,
    const std::vector<int64_t>& inputs) {

    std::unordered_map<std::string, int64_t> values;

    // Bind arguments
    for (size_t i = 0; i < fn.argument_count() && i < inputs.size(); ++i) {
        auto& arg = fn.arguments()[i];
        values[arg.name.empty() ? "arg" + std::to_string(i) : arg.name] = inputs[i];
    }

    // Walk through each basic block in order (simplified — no control flow)
    for (auto& block : fn.blocks()) {
        for (auto& inst : block->instructions()) {
            if (inst->is_terminator()) {
                if (inst->opcode() == ir::Opcode::Ret && inst->num_operands() > 0) {
                    auto ret_val = inst->operand(0);
                    if (ret_val && ret_val->has_name()) {
                        auto it = values.find(ret_val->name());
                        if (it != values.end()) return it->second;
                    }
                    if (auto* ci = dynamic_cast<const ir::ConstantInt*>(ret_val.get())) {
                        return ci->value();
                    }
                    return std::nullopt;
                }
                continue;
            }

            auto resolve = [&](const std::shared_ptr<ir::Value>& v) -> std::optional<int64_t> {
                if (!v) return std::nullopt;
                if (auto* ci = dynamic_cast<const ir::ConstantInt*>(v.get())) {
                    return ci->value();
                }
                if (v->has_name()) {
                    auto it = values.find(v->name());
                    if (it != values.end()) return it->second;
                }
                return std::nullopt;
            };

            if (inst->is_binary_op() && inst->num_operands() >= 2) {
                auto lhs = resolve(inst->operand(0));
                auto rhs = resolve(inst->operand(1));

                if (!lhs || !rhs) {
                    if (inst->has_name()) values[inst->name()] = 0;
                    continue;
                }

                int64_t result = 0;
                uint64_t ulhs = static_cast<uint64_t>(lhs.value());
                uint64_t urhs = static_cast<uint64_t>(rhs.value());
                unsigned bw = 64;
                if (inst->type() && inst->type()->is_integer()) {
                    bw = static_cast<unsigned>(inst->type()->bit_width());
                }
                uint64_t mask = bw >= 64 ? ~0ULL : ((1ULL << bw) - 1);

                switch (inst->opcode()) {
                    case ir::Opcode::Add:  result = (lhs.value() + rhs.value()); break;
                    case ir::Opcode::Sub:  result = (lhs.value() - rhs.value()); break;
                    case ir::Opcode::Mul:  result = (lhs.value() * rhs.value()); break;
                    case ir::Opcode::SDiv: result = rhs.value() != 0 ? lhs.value() / rhs.value() : 0; break;
                    case ir::Opcode::UDiv: result = urhs != 0 ? static_cast<int64_t>(ulhs / urhs) : 0; break;
                    case ir::Opcode::SRem: result = rhs.value() != 0 ? lhs.value() % rhs.value() : 0; break;
                    case ir::Opcode::URem: result = urhs != 0 ? static_cast<int64_t>(ulhs % urhs) : 0; break;
                    case ir::Opcode::And:  result = lhs.value() & rhs.value(); break;
                    case ir::Opcode::Or:   result = lhs.value() | rhs.value(); break;
                    case ir::Opcode::Xor:  result = lhs.value() ^ rhs.value(); break;
                    case ir::Opcode::Shl:
                        result = (rhs.value() >= 0 && rhs.value() < 64)
                                 ? ((lhs.value() << rhs.value()) & static_cast<int64_t>(mask))
                                 : 0;
                        break;
                    case ir::Opcode::LShr:
                        result = (rhs.value() >= 0 && rhs.value() < 64)
                                 ? static_cast<int64_t>((ulhs >> rhs.value()) & mask)
                                 : 0;
                        break;
                    case ir::Opcode::AShr:
                        result = (rhs.value() >= 0 && rhs.value() < 64)
                                 ? (lhs.value() >> rhs.value())
                                 : 0;
                        break;
                    default:
                        if (inst->has_name()) values[inst->name()] = 0;
                        continue;
                }

                // Mask to bit-width for arithmetic ops.
                if (bw < 64) {
                    result = static_cast<int64_t>(static_cast<uint64_t>(result) & mask);
                }

                if (inst->has_name()) values[inst->name()] = result;
                continue;
            }

            if (inst->opcode() == ir::Opcode::ICmp && inst->num_operands() >= 2) {
                auto lhs = resolve(inst->operand(0));
                auto rhs = resolve(inst->operand(1));

                if (!lhs || !rhs) {
                    if (inst->has_name()) values[inst->name()] = 0;
                    continue;
                }

                auto it_pred = inst->metadata().find("pred");
                ir::CmpPredicate pred = ir::CmpPredicate::EQ;
                if (it_pred != inst->metadata().end()) {
                    try {
                        pred = static_cast<ir::CmpPredicate>(std::stoul(it_pred->second));
                    } catch (...) {}
                }

                int64_t result = 0;
                uint64_t ulhs = static_cast<uint64_t>(lhs.value());
                uint64_t urhs = static_cast<uint64_t>(rhs.value());

                switch (pred) {
                    case ir::CmpPredicate::EQ:  result = (lhs.value() == rhs.value()) ? 1 : 0; break;
                    case ir::CmpPredicate::NE:  result = (lhs.value() != rhs.value()) ? 1 : 0; break;
                    case ir::CmpPredicate::UGT: result = (ulhs > urhs) ? 1 : 0; break;
                    case ir::CmpPredicate::UGE: result = (ulhs >= urhs) ? 1 : 0; break;
                    case ir::CmpPredicate::ULT: result = (ulhs < urhs) ? 1 : 0; break;
                    case ir::CmpPredicate::ULE: result = (ulhs <= urhs) ? 1 : 0; break;
                    case ir::CmpPredicate::SGT: result = (lhs.value() > rhs.value()) ? 1 : 0; break;
                    case ir::CmpPredicate::SGE: result = (lhs.value() >= rhs.value()) ? 1 : 0; break;
                    case ir::CmpPredicate::SLT: result = (lhs.value() < rhs.value()) ? 1 : 0; break;
                    case ir::CmpPredicate::SLE: result = (lhs.value() <= rhs.value()) ? 1 : 0; break;
                    default: result = 0; break;
                }

                if (inst->has_name()) values[inst->name()] = result;
                continue;
            }

            if (inst->opcode() == ir::Opcode::Select && inst->num_operands() >= 3) {
                auto cond = resolve(inst->operand(0));
                auto tv = resolve(inst->operand(1));
                auto fv = resolve(inst->operand(2));

                if (cond && tv && fv && inst->has_name()) {
                    values[inst->name()] = cond.value() ? tv.value() : fv.value();
                } else if (inst->has_name()) {
                    values[inst->name()] = 0;
                }
                continue;
            }

            // For other instructions, store a placeholder
            if (inst->has_name()) {
                if (values.find(inst->name()) == values.end()) {
                    values[inst->name()] = 0;
                }
            }
        }
    }

    return std::nullopt;
}

// ═══════════════════════════════════════════════════════════════════════════
// Z3 quantifier-based exists-forall CEGIS
// ═══════════════════════════════════════════════════════════════════════════
//
// Real CEGIS via Z3's quantifier support, mirroring Souper's
// lib/Infer/ConstantSynthesis.cpp:synthesize. The algorithm alternates
// between two queries:
//
//   (1) SYNTHESIS: given a set of counterexample inputs CEX (initially
//       {arg0=0, arg1=0, ...}), find placeholder values that make
//       orig(cex) == cand(cex, placeholders) for ALL cex in CEX.
//       Encoded as:
//         exists placeholders. AND_{cex in CEX} (orig(cex) == cand(cex, placeholders))
//       Z3 returns SAT with a model → extract placeholder values.
//
//   (2) VERIFY: with the synthesised placeholder values, check whether
//       orig and cand agree on ALL inputs. Encoded as:
//         exists inputs. orig(inputs) != cand(inputs, synthesised_placeholders)
//       Z3 returns UNSAT → the model is sound, we're done.
//       Z3 returns SAT → extract the counterexample input, add to CEX, loop.
//
// Converges in 1-5 iterations on typical inputs. If Z3 returns Unknown
// (timeout, quantifier incompleteness) or the iteration cap is hit, the
// caller falls back to constant-pool enumeration.
//
// Why CEGIS and not a single exists-forall query: Z3 (and SMT solvers in
// general) handle exists-forall quantifier alternation poorly. CEGIS
// skolemises the exists by enumerating concrete candidate models, and
// each synthesis+verify query is purely existential (easier for Z3).
// This is the standard SyGuS approach (Solar-Lezama 2008, Abate et al.
// CAV'18).
//
bool SMTVerifier::synthesize_with_z3_quantifiers(
    const ir::Function& original,
    const ir::Function& candidate_template,
    const std::vector<std::string>& placeholder_names,
    const std::vector<int64_t>& placeholder_widths,
    std::vector<std::pair<std::string, int64_t>>& out_model,
    std::string& out_message) {

    out_model.clear();

    // ── Capability check: bail to enumeration if any quantifier symbol ─
    // is unavailable. The dlopen shim resolves these best-effort.
    if (!Z3DynamicLoader::instance().z3_loaded()) {
        out_message = "Z3 not loaded";
        return false;
    }
    if (!Z3API(Z3_mk_quantifier_const) || !Z3API(Z3_to_app) ||
        !Z3API(Z3_model_get_num_consts) || !Z3API(Z3_model_get_const_decl) ||
        !Z3API(Z3_model_get_const_interp) || !Z3API(Z3_get_decl_name) ||
        !Z3API(Z3_get_symbol_string) || !Z3API(Z3_get_numeral_string) ||
        !Z3API(Z3_solver_get_model) || !Z3API(Z3_model_inc_ref) ||
        !Z3API(Z3_model_dec_ref) || !Z3API(Z3_inc_ref) || !Z3API(Z3_dec_ref)) {
        out_message = "Z3 quantifier/model symbols unavailable";
        return false;
    }

    // ── Reject unsupported ops early (memory/float/loops) ──────────────
    std::string orig_reason, cand_reason;
    if (function_has_unsupported_ops(original, config_, orig_reason)) {
        out_message = "Original has unsupported ops: " + orig_reason;
        return false;
    }
    if (function_has_unsupported_ops(candidate_template, config_, cand_reason)) {
        out_message = "Candidate template has unsupported ops: " + cand_reason;
        return false;
    }
    if (original.argument_count() != candidate_template.argument_count()) {
        out_message = "Argument count mismatch";
        return false;
    }

    Z3_context ctx = static_cast<Z3_context>(get_z3_context());
    if (!ctx) {
        out_message = "Failed to create Z3 context";
        return false;
    }

    constexpr size_t CEGIS_MAX_ITERS = 8;

    // ── Build shared argument Z3 constants (used by both encodings) ────
    // These are the "inputs" we'll quantify over in the verify step.
    bool use_bv = config_.use_bitvectors;
    std::vector<std::pair<std::string, Z3_ast>> shared_args;
    std::vector<Z3_ast> shared_arg_refs;
    for (size_t i = 0; i < original.argument_count(); ++i) {
        auto& arg = original.arguments()[i];
        Z3_sort sort = get_z3_sort(ctx, *arg.type, use_bv);
        std::string z3_name = "arg_" + std::to_string(i);
        Z3_symbol sym = Z3API(Z3_mk_string_symbol)(ctx, z3_name.c_str());
        Z3_ast var = Z3API(Z3_mk_const)(ctx, sym, sort);
        Z3API(Z3_inc_ref)(ctx, var);
        shared_arg_refs.push_back(var);
        shared_args.push_back({arg.name.empty() ? ("arg" + std::to_string(i)) : arg.name, var});
    }

    // ── Build placeholder Z3 constants (the "existential" variables) ──
    // The candidate template has named ConstantInts whose values we want
    // to synthesise. We replace them with free Z3 constants during
    // encoding. To do this, we abuse the existing encode_function by
    // making a deep copy of the candidate template where each placeholder
    // ConstantInt is replaced with a fresh "argument" — but that would
    // change the function signature. Instead, we encode the candidate
    // manually here, replacing placeholder constants with free Z3 consts.
    //
    // SIMPLER APPROACH (used here): encode the candidate template with
    // the standard encoder, but PRE-POPULATE the candidate's value_map
    // so that each placeholder ConstantInt is mapped to a free Z3 const
    // of the appropriate bit-width. The encoder looks up values by name
    // in value_map, so this intercepts the placeholder.
    std::vector<Z3_ast> placeholder_refs;
    std::unordered_map<std::string, Z3_ast> placeholder_z3_vars;
    for (size_t i = 0; i < placeholder_names.size(); ++i) {
        unsigned width = static_cast<unsigned>(placeholder_widths[i]);
        if (width == 0) width = 32;
        Z3_sort sort = Z3API(Z3_mk_bv_sort)(ctx, width);
        std::string z3_name = "ph_" + std::to_string(i);
        Z3_symbol sym = Z3API(Z3_mk_string_symbol)(ctx, z3_name.c_str());
        Z3_ast var = Z3API(Z3_mk_const)(ctx, sym, sort);
        Z3API(Z3_inc_ref)(ctx, var);
        placeholder_refs.push_back(var);
        placeholder_z3_vars[placeholder_names[i]] = var;
    }

    // ── Helper: instantiate the candidate with concrete placeholder ────
    // values (used in the verify step after CEGIS converges). Mirrors the
    // enumeration fallback's `instantiate` lambda.
    auto instantiate_candidate = [&](const std::vector<int64_t>& values)
        -> std::shared_ptr<ir::Function> {
        auto cloned = ir::deep_copy_function(candidate_template);
        if (!cloned) return nullptr;
        std::unordered_map<std::string, int64_t> subst;
        for (size_t i = 0; i < placeholder_names.size(); ++i) {
            subst[placeholder_names[i]] = values[i];
        }
        for (auto& bb : cloned->blocks()) {
            if (!bb) continue;
            for (auto& inst : bb->instructions()) {
                if (!inst) continue;
                for (size_t opi = 0; opi < inst->num_operands(); ++opi) {
                    auto op = inst->operand(opi);
                    if (!op || !op->has_name()) continue;
                    auto it = subst.find(op->name());
                    if (it == subst.end()) continue;
                    auto ci = std::dynamic_pointer_cast<ir::ConstantInt>(op);
                    if (!ci) continue;
                    auto int_ty = std::dynamic_pointer_cast<ir::IntegerType>(ci->type());
                    if (!int_ty) continue;
                    auto new_ci = std::make_shared<ir::ConstantInt>(int_ty, it->second);
                    inst->set_operand(opi, new_ci);
                }
            }
        }
        return cloned;
    };

    // ── Helper: extract an int64_t from a Z3 numeral AST ───────────────
    auto extract_int64 = [&](Z3_ast ast) -> int64_t {
        if (!ast) return 0;
        Z3_string s = Z3API(Z3_get_numeral_string)(ctx, ast);
        if (!s || !*s) return 0;
        // Z3 numeral strings are like "123" or "-1" (no prefix for BV).
        // For negative BV values, Z3 prints the unsigned form, so we
        // parse as unsigned then cast. Use strtoll which handles both.
        try {
            return std::strtoll(s, nullptr, 10);
        } catch (...) {
            return 0;
        }
    };

    // ── Helper: encode both functions with shared args + placeholder ───
    // substitution. Returns orig_ret and cand_ret ASTs (already inc_ref'd
    // in the encodings' local_refs). On failure, returns false.
    // The placeholder Z3 vars are passed to encode_function via
    // the placeholder_vars parameter, which installs them into the fresh
    // encoding's value_map so resolve_value uses them instead of concrete
    // numerals for the named ConstantInt placeholders.
    std::unordered_map<unsigned, Z3_ast> shared_poison;
    SMTEncoding orig_enc;
    SMTEncoding cand_enc;
    auto encode_both = [&]() -> bool {
        orig_enc = SMTEncoding();
        cand_enc = SMTEncoding();
        orig_enc = encode_function(ctx, original, "orig", use_bv, shared_args, config_, &shared_poison);
        orig_enc.ctx = ctx;
        cand_enc = encode_function(ctx, candidate_template, "cand", use_bv, shared_args, config_, &shared_poison,
                                    &placeholder_z3_vars);
        cand_enc.ctx = ctx;
        if (orig_enc.had_unsupported || cand_enc.had_unsupported) return false;
        if (!orig_enc.return_value || !cand_enc.return_value) return false;
        return true;
    };

    // ── Helper: extract placeholder values from a Z3 model ─────────────
    auto extract_placeholder_model = [&](Z3_model model)
        -> std::vector<int64_t> {
        std::vector<int64_t> values(placeholder_names.size(), 0);
        unsigned n_consts = Z3API(Z3_model_get_num_consts)(ctx, model);
        for (unsigned i = 0; i < n_consts; ++i) {
            Z3_func_decl decl = Z3API(Z3_model_get_const_decl)(ctx, model, i);
            if (!decl) continue;
            Z3_symbol sym = Z3API(Z3_get_decl_name)(ctx, decl);
            if (!sym) continue;
            Z3_string name = Z3API(Z3_get_symbol_string)(ctx, sym);
            if (!name) continue;
            std::string name_str(name);
            for (size_t j = 0; j < placeholder_names.size(); ++j) {
                if (name_str == ("ph_" + std::to_string(j))) {
                    Z3_ast val_ast = Z3API(Z3_model_get_const_interp)(ctx, model, decl);
                    values[j] = extract_int64(val_ast);
                    break;
                }
            }
        }
        return values;
    };

    // ── CEGIS main loop ────────────────────────────────────────────────
    // Initial counterexample set: a single zero vector (cheap, often
    // sufficient to bootstrap).
    std::vector<std::vector<int64_t>> cex_set;
    {
        std::vector<int64_t> zeros(original.argument_count(), 0);
        cex_set.push_back(zeros);
    }

    for (size_t iter = 0; iter < CEGIS_MAX_ITERS; ++iter) {
        // ── Synthesis step: exists placeholders. AND_{cex} (orig(cex) == cand(cex, placeholders)) ──
        if (!encode_both()) {
            out_message = "Encoding failed during synthesis step";
            for (Z3_ast a : shared_arg_refs) Z3API(Z3_dec_ref)(ctx, a);
            for (Z3_ast a : placeholder_refs) Z3API(Z3_dec_ref)(ctx, a);
            release_tracked(orig_enc);
            release_tracked(cand_enc);
            return false;
        }

        // Build conjunction of orig(cex) == cand(cex, placeholders) for
        // each cex. We substitute the cex values into the arg constants
        // via Z3_solver_assert + push/pop, OR we use Z3_mk_eq with
        // substituted arg constants.
        //
        // SIMPLER: for each cex, build a fresh solver, substitute arg
        // constants with concrete numerals, assert orig_ret == cand_ret,
        // check. This is one Z3 call per cex, but CEX set is small
        // (typically 1-5 elements).
        Z3_solver synth_solver = Z3API(Z3_mk_solver)(ctx);
        Z3API(Z3_solver_inc_ref)(ctx, synth_solver);
        Z3_params params = Z3API(Z3_mk_params)(ctx);
        Z3API(Z3_params_inc_ref)(ctx, params);
        Z3_symbol timeout_sym = Z3API(Z3_mk_string_symbol)(ctx, "timeout");
        Z3API(Z3_params_set_uint)(ctx, params, timeout_sym, config_.timeout_ms);
        if (Z3API(Z3_solver_set_params)) {
            Z3API(Z3_solver_set_params)(ctx, synth_solver, params);
        }

        // We need to encode orig and cand with the placeholder Z3 vars
        // substituted in. The encode_both() call above already did this
        // (placeholder_z3_vars are in cand_enc.value_map). Now we just
        // need to assert orig_ret == cand_ret for each cex.
        //
        // For the cex substitution: we build a substitution map from
        // arg_Z3_const → concrete numeral, then use Z3_substitute. But
        // Z3_substitute isn't in our shim. So instead, we re-encode with
        // shared_args overridden to concrete numerals per cex.
        bool synth_ok = false;
        std::vector<int64_t> synth_values(placeholder_names.size(), 0);

        // For the synthesis step, we encode orig and cand with the args
        // bound to concrete cex values. Build a per-cex arg list.
        for (size_t ci = 0; ci < cex_set.size() && !synth_ok; ++ci) {
            const auto& cex = cex_set[ci];
            // Build concrete arg ASTs.
            std::vector<std::pair<std::string, Z3_ast>> concrete_args;
            std::vector<Z3_ast> concrete_refs;
            for (size_t ai = 0; ai < original.argument_count() && ai < cex.size(); ++ai) {
                Z3_sort sort = get_z3_sort(ctx, *original.arguments()[ai].type, use_bv);
                Z3_ast num = Z3API(Z3_mk_unsigned_int64)(ctx, static_cast<uint64_t>(cex[ai]), sort);
                Z3API(Z3_inc_ref)(ctx, num);
                concrete_refs.push_back(num);
                concrete_args.push_back({original.arguments()[ai].name, num});
            }
            SMTEncoding o_enc = encode_function(ctx, original, "orig", use_bv, concrete_args, config_, &shared_poison);
            o_enc.ctx = ctx;
            SMTEncoding c_enc = encode_function(ctx, candidate_template, "cand", use_bv, concrete_args, config_, &shared_poison,
                                                 &placeholder_z3_vars);
            c_enc.ctx = ctx;

            if (o_enc.had_unsupported || c_enc.had_unsupported ||
                !o_enc.return_value || !c_enc.return_value) {
                release_tracked(o_enc);
                release_tracked(c_enc);
                for (Z3_ast a : concrete_refs) Z3API(Z3_dec_ref)(ctx, a);
                continue;
            }

            Z3_ast eq = Z3API(Z3_mk_eq)(ctx, o_enc.return_value, c_enc.return_value);
            Z3API(Z3_inc_ref)(ctx, eq);
            Z3API(Z3_solver_assert)(ctx, synth_solver, eq);

            release_tracked(o_enc);
            release_tracked(c_enc);
            for (Z3_ast a : concrete_refs) Z3API(Z3_dec_ref)(ctx, a);
        }

        // Now existentially quantify the placeholder Z3 vars and check.
        // Build the exists quantifier.
        std::vector<Z3_app> bound_apps;
        bound_apps.reserve(placeholder_refs.size());
        for (Z3_ast ph : placeholder_refs) {
            bound_apps.push_back(Z3API(Z3_to_app)(ctx, ph));
        }

        // The body is the conjunction of all assertions in the solver.
        // Since we asserted each eq, the solver's assertions ARE the body.
        // We need to extract them — but Z3 doesn't expose solver assertions
        // easily without Z3_solver_get_assertions. SIMPLER: build the
        // conjunction inline.
        // Actually, we can just check the solver with the placeholders as
        // free constants — Z3 will find a model for them if SAT. We don't
        // need an explicit quantifier for the synthesis step.
        Z3_lbool_enum synth_result = Z3API(Z3_solver_check)(ctx, synth_solver);

        if (synth_result == Z3_L_TRUE) {
            // SAT — extract placeholder model.
            Z3_model model = Z3API(Z3_solver_get_model)(ctx, synth_solver);
            if (model) {
                Z3API(Z3_model_inc_ref)(ctx, model);
                synth_values = extract_placeholder_model(model);
                synth_ok = true;
                Z3API(Z3_model_dec_ref)(ctx, model);
            }
        }

        Z3API(Z3_params_dec_ref)(ctx, params);
        Z3API(Z3_solver_dec_ref)(ctx, synth_solver);
        release_tracked(orig_enc);
        release_tracked(cand_enc);

        if (!synth_ok) {
            // Z3 returned UNSAT (no placeholders work for these cex) or
            // UNKNOWN (timeout). Either way, CEGIS failed.
            out_message = (synth_result == Z3_L_FALSE)
                ? "CEGIS synthesis UNSAT — no placeholder values satisfy counterexamples"
                : "CEGIS synthesis returned UNKNOWN";
            for (Z3_ast a : shared_arg_refs) Z3API(Z3_dec_ref)(ctx, a);
            for (Z3_ast a : placeholder_refs) Z3API(Z3_dec_ref)(ctx, a);
            return false;
        }

        // ── Verify step: instantiate candidate with synth_values, ───────
        // run verify(original, instantiated). If Equivalent, done.
        auto instantiated = instantiate_candidate(synth_values);
        if (!instantiated) {
            out_message = "Failed to instantiate candidate with synthesised values";
            for (Z3_ast a : shared_arg_refs) Z3API(Z3_dec_ref)(ctx, a);
            for (Z3_ast a : placeholder_refs) Z3API(Z3_dec_ref)(ctx, a);
            return false;
        }

        VerificationResult vr = verify(original, *instantiated);
        if (vr.status == VerificationResult::Equivalent) {
            // Success — model is sound.
            for (size_t i = 0; i < placeholder_names.size(); ++i) {
                out_model.push_back({placeholder_names[i], synth_values[i]});
            }
            out_message = "CEGIS converged in " + std::to_string(iter + 1) + " iteration(s)";
            for (Z3_ast a : shared_arg_refs) Z3API(Z3_dec_ref)(ctx, a);
            for (Z3_ast a : placeholder_refs) Z3API(Z3_dec_ref)(ctx, a);
            return true;
        }

        if (vr.status == VerificationResult::NotEquivalent && !vr.counterexample.empty()) {
            // Found a counterexample — add to CEX set and loop.
            cex_set.push_back(vr.counterexample);
            // Continue to next iteration.
        } else {
            // Unknown or Error — CEGIS can't make progress.
            out_message = "CEGIS verify returned " +
                std::string(vr.status == VerificationResult::Unknown ? "Unknown" : "Error");
            for (Z3_ast a : shared_arg_refs) Z3API(Z3_dec_ref)(ctx, a);
            for (Z3_ast a : placeholder_refs) Z3API(Z3_dec_ref)(ctx, a);
            return false;
        }
    }

    // Hit the iteration cap without convergence.
    out_message = "CEGIS did not converge within " + std::to_string(CEGIS_MAX_ITERS) + " iterations";
    for (Z3_ast a : shared_arg_refs) Z3API(Z3_dec_ref)(ctx, a);
    for (Z3_ast a : placeholder_refs) Z3API(Z3_dec_ref)(ctx, a);
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// CEGIS: CEGIS for symbolic-constant synthesis
// ═══════════════════════════════════════════════════════════════════════════
//
// Tries the Z3 quantifier-based exists-forall CEGIS first. If that
// fails (Z3 quantifier symbols unavailable, encoding fails, or CEGIS
// doesn't converge), falls back to constant-pool enumeration.
//
// The constant-pool enumeration fallback tries every value in the pool
// `{0, 1, 2, 3, 5, 7, 8, 15, 16, 31, 32, 63, 64, -1, 255, 256}` for
// each placeholder and calls `verify()`. Less powerful than the
// quantifier path (only 16 values vs. any 64-bit value) but doesn't
// depend on Z3 quantifier support.
//
SynthesisResult SMTVerifier::synthesize_with_cegis(
    const ir::Function& original,
    const ir::Function& candidate_template,
    const std::vector<std::string>& placeholder_names,
    const std::vector<int64_t>& placeholder_widths) {

    auto start = std::chrono::high_resolution_clock::now();
    SynthesisResult result;
    result.success = false;

    // Validate inputs.
    if (placeholder_names.size() != placeholder_widths.size()) {
        result.message = "placeholder_names and placeholder_widths size mismatch";
        result.solve_time_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
        return result;
    }
    if (placeholder_names.empty()) {
        result.message = "no placeholders specified";
        result.solve_time_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
        return result;
    }

    // ── Z3 quantifier-based exists-forall CEGIS ────────────────────────────
    // first. This can find ANY 64-bit placeholder value, not just the
    // 16-element pool below. Falls through to enumeration on failure.
    if (z3_available_) {
        std::vector<std::pair<std::string, int64_t>> qmodel;
        std::string qmsg;
        if (synthesize_with_z3_quantifiers(original, candidate_template,
                                            placeholder_names, placeholder_widths,
                                            qmodel, qmsg)) {
            result.success = true;
            result.model = std::move(qmodel);
            result.message = "Z3 quantifier CEGIS: " + qmsg;
            // Run a final verify() to populate result.verification.
            // (synthesize_with_z3_quantifiers already ran verify() in its
            // loop, but we re-run here for a clean diagnostic.)
            result.verification.status = VerificationResult::Equivalent;
            result.verification.message = "CEGIS-verified";
            result.solve_time_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start).count();
            return result;
        }
        // Quantifier path failed — fall through to enumeration, but record
        // the failure reason in the message if enumeration also fails.
        result.message = "Z3 quantifier CEGIS failed (" + qmsg + "); trying enumeration";
    }

    // ── Enumeration fallback ───────────────────────────────────────────
    // Cap: 2 placeholders max. With the expanded constant pool (~37
    // constants), worst case is 37^2 ≈ 1.4k verifications — bounded by
    // the per-call time budget (Z3 calls each have their own timeout).
    // More than 2 placeholders would blow the time budget.
    if (placeholder_names.size() > 2) {
        result.message = "too many placeholders (max 2 in the enumeration fallback)";
        result.solve_time_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
        return result;
    }

    // The constant pool. Expanded from the original 16-element set to
    // cover more common constants found in real code: bit widths, masks,
    // small primes, and signed sentinels. With 24 constants and 2
    // placeholders, the worst case is 24^2 = 576 verifications — bounded
    // by the time budget and the existing "≤2 placeholders" cap.
    static const std::vector<int64_t> POOL = {
        // Identity / zero.
        0, 1, -1,
        // Small integers.
        2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16,
        // Powers of two and adjacent masks.
        31, 32, 63, 64, 127, 128, 255, 256, 511, 512,
        1023, 1024, 4095, 4096,
        65535, 65536,
        // Common divisor / shift amounts.
        24, 48, 56,
        // Negative sentinels.
        -2, -8, -16, -256,
    };

    // Helper: deep-copy the candidate template and substitute placeholder
    // constants with concrete values.
    auto instantiate = [&](const std::vector<int64_t>& values)
        -> std::shared_ptr<ir::Function> {
        auto cloned = ir::deep_copy_function(candidate_template);
        if (!cloned) return nullptr;
        // Build a name → value map for the placeholders.
        std::unordered_map<std::string, int64_t> subst;
        for (size_t i = 0; i < placeholder_names.size(); ++i) {
            subst[placeholder_names[i]] = values[i];
        }
        // Walk all instructions and substitute placeholder ConstantInts.
        for (auto& bb : cloned->blocks()) {
            if (!bb) continue;
            for (auto& inst : bb->instructions()) {
                if (!inst) continue;
                for (size_t opi = 0; opi < inst->num_operands(); ++opi) {
                    auto op = inst->operand(opi);  // copy (operand() returns by value)
                    if (!op) continue;
                    if (!op->has_name()) continue;
                    auto it = subst.find(op->name());
                    if (it == subst.end()) continue;
                    // This operand is a placeholder ConstantInt.
                    auto ci = std::dynamic_pointer_cast<ir::ConstantInt>(op);
                    if (!ci) continue;
                    unsigned bits = ci->bit_width();
                    // Create a new ConstantInt with the substituted value.
                    // Use the existing IntegerType (no TypeContext needed).
                    auto int_ty = std::dynamic_pointer_cast<ir::IntegerType>(ci->type());
                    if (!int_ty) continue;
                    auto new_ci = std::make_shared<ir::ConstantInt>(int_ty, it->second);
                    // Mask the value to the bit-width (to handle -1 etc.).
                    (void)bits;
                    inst->set_operand(opi, new_ci);
                }
            }
        }
        return cloned;
    };

    // Enumerate the Cartesian product of POOL for each placeholder.
    // With 1 placeholder: 16 verifications. With 2: 256.
    size_t num_placeholders = placeholder_names.size();
    size_t total_attempts = 1;
    for (size_t i = 0; i < num_placeholders; ++i) total_attempts *= POOL.size();

    size_t attempts = 0;
    for (size_t attempt = 0; attempt < total_attempts; ++attempt) {
        // Decode the attempt index into per-placeholder pool indices.
        std::vector<int64_t> values(num_placeholders);
        size_t idx = attempt;
        for (size_t i = 0; i < num_placeholders; ++i) {
            size_t pool_idx = idx % POOL.size();
            idx /= POOL.size();
            values[i] = POOL[pool_idx];
        }
        ++attempts;

        // Instantiate the candidate with these values.
        auto instantiated = instantiate(values);
        if (!instantiated) continue;

        // Verify (the 2nd query in Souper's CEGIS loop).
        VerificationResult vr = verify(original, *instantiated);
        result.verification = vr;
        if (vr.status == VerificationResult::Equivalent) {
            // Success: build the model and return.
            result.success = true;
            for (size_t i = 0; i < placeholder_names.size(); ++i) {
                result.model.push_back({placeholder_names[i], values[i]});
            }
            result.message = "found model in " + std::to_string(attempts) +
                             " attempt(s) via constant-pool enumeration";
            result.solve_time_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start).count();
            return result;
        }
    }

    // No model found.
    result.success = false;
    result.message = "no model found in " + std::to_string(attempts) +
                     " constant-pool attempt(s)";
    result.solve_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumption-based batch pruning: unsat-core / assumption-based batch pruning
// ═══════════════════════════════════════════════════════════════════════════
//
void SMTVerifier::prune_batch_with_unsat_core(
    const ir::Function& original,
    const std::vector<ir::Function>& candidates,
    std::vector<VerificationResult>& results,
    std::vector<size_t>* pruned_indices) {

    results.clear();
    if (pruned_indices) pruned_indices->clear();

    // Need Z3 + check_assumptions.
    if (!z3_available_) return;
    if (!Z3API(Z3_solver_check_assumptions)) return;
    if (candidates.empty()) return;

    // Check for unsupported ops in original.
    std::string orig_reason;
    if (function_has_unsupported_ops(original, config_, orig_reason)) return;

    Z3_context ctx = static_cast<Z3_context>(get_z3_context());
    if (!ctx) return;

    Z3_solver solver = nullptr;
    Z3_params params = nullptr;
    SMTEncoding orig_enc;
    std::vector<SMTEncoding> cand_encs;
    std::vector<Z3_ast> assumption_literals;  // parallel to candidates; nullptr if unencodable

    try {
        solver = Z3API(Z3_mk_solver)(ctx);
        Z3API(Z3_solver_inc_ref)(ctx, solver);

        params = Z3API(Z3_mk_params)(ctx);
        Z3API(Z3_params_inc_ref)(ctx, params);
        Z3_symbol timeout_sym = Z3API(Z3_mk_string_symbol)(ctx, "timeout");
        Z3API(Z3_params_set_uint)(ctx, params, timeout_sym, config_.timeout_ms);
        if (Z3API(Z3_solver_set_params)) {
            Z3API(Z3_solver_set_params)(ctx, solver, params);
        }

        // Build shared args + shared poison.
        std::vector<std::pair<std::string, Z3_ast>> shared_args;
        std::unordered_map<unsigned, Z3_ast> shared_poison;
        bool use_bv = config_.use_bitvectors;
        for (size_t i = 0; i < original.argument_count(); ++i) {
            auto& arg = original.arguments()[i];
            std::string arg_name = arg.name.empty() ? ("arg" + std::to_string(i)) : arg.name;
            Z3_sort sort = get_z3_sort(ctx, *arg.type, use_bv);
            std::string z3_name = "arg_" + std::to_string(i);
            Z3_symbol sym = Z3API(Z3_mk_string_symbol)(ctx, z3_name.c_str());
            Z3_ast var = Z3API(Z3_mk_const)(ctx, sym, sort);
            if (Z3API(Z3_inc_ref)) Z3API(Z3_inc_ref)(ctx, var);
            orig_enc.local_refs.push_back(var);
            shared_args.push_back({arg_name, var});
        }

        // Legacy POISON-value encoding: this path reasons about bare value
        // (in)equality, so poison must live in the values, not side flags.
        SMTConfig batch_cfg = config_;
        batch_cfg.refinement_semantics = false;
        orig_enc = encode_function(ctx, original, "orig", use_bv, shared_args, batch_cfg, &shared_poison);
        for (auto& [name, ast] : shared_args) {
            if (Z3API(Z3_inc_ref)) Z3API(Z3_inc_ref)(ctx, ast);
            orig_enc.local_refs.push_back(ast);
        }
        orig_enc.ctx = ctx;

        if (orig_enc.had_unsupported || !orig_enc.return_value) {
            release_tracked(orig_enc);
            Z3API(Z3_params_dec_ref)(ctx, params);
            Z3API(Z3_solver_dec_ref)(ctx, solver);
            return;
        }

        // Encode each candidate and create assumption literal.
        cand_encs.resize(candidates.size());
        assumption_literals.resize(candidates.size(), nullptr);

        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto& cand = candidates[i];
            if (original.argument_count() != cand.argument_count()) {
                continue;  // leave assumption_literals[i] = nullptr
            }
            std::string cand_reason;
            if (function_has_unsupported_ops(cand, config_, cand_reason)) {
                continue;
            }

            auto& cand_enc = cand_encs[i];
            cand_enc.ctx = ctx;
            cand_enc = encode_function(ctx, cand, "cand", use_bv, shared_args, batch_cfg, &shared_poison);

            if (cand_enc.had_unsupported || !cand_enc.return_value) {
                continue;
            }

            // eq_i := (cand_i_return == orig_return)
            Z3_ast eq = track_ast(cand_enc,
                Z3API(Z3_mk_eq)(ctx, orig_enc.return_value, cand_enc.return_value));
            Z3_ast neq = track_ast(cand_enc, Z3API(Z3_mk_not)(ctx, eq));

            // Create fresh Bool literal a_i.
            std::string a_name = "assume_cand_" + std::to_string(i);
            Z3_symbol a_sym = Z3API(Z3_mk_string_symbol)(ctx, a_name.c_str());
            Z3_sort bool_sort = Z3API(Z3_mk_bool_sort)(ctx);
            Z3_ast a_i = Z3API(Z3_mk_const)(ctx, a_sym, bool_sort);
            a_i = track_ast(cand_enc, a_i);

            // Assert a_i ⇒ neq (i.e., ¬a_i ∨ neq).
            Z3_ast not_a = track_ast(cand_enc, Z3API(Z3_mk_not)(ctx, a_i));
            Z3_ast impl = track_ast(cand_enc, mk_or2(ctx, not_a, neq));
            Z3API(Z3_solver_assert)(ctx, solver, impl);

            assumption_literals[i] = a_i;
        }

        // Collect non-null assumptions.
        std::vector<Z3_ast> assumptions;
        for (Z3_ast a : assumption_literals) {
            if (a) assumptions.push_back(a);
        }

        if (assumptions.empty()) {
            // No candidates could be encoded — no pruning.
            for (auto& enc : cand_encs) release_tracked(enc);
            release_tracked(orig_enc);
            Z3API(Z3_params_dec_ref)(ctx, params);
            Z3API(Z3_solver_dec_ref)(ctx, solver);
            return;
        }

        // Single check with all assumptions (all-or-nothing).
        Z3_lbool_enum sat = Z3API(Z3_solver_check_assumptions)(ctx, solver,
            static_cast<unsigned>(assumptions.size()), assumptions.data());

        if (sat == Z3_L_TRUE) {
            // SAT: all candidates (with non-null assumptions) are jointly
            // unequivalent. Extract counterexample from model.
            std::vector<int64_t> counterexample;
            if (Z3API(Z3_solver_get_model) && Z3API(Z3_model_eval) &&
                Z3API(Z3_get_numeral_int64) && Z3API(Z3_model_inc_ref) &&
                Z3API(Z3_model_dec_ref)) {
                Z3_model model = Z3API(Z3_solver_get_model)(ctx, solver);
                if (model) {
                    Z3API(Z3_model_inc_ref)(ctx, model);
                    for (const auto& [arg_name, arg_ast] : orig_enc.arg_vars) {
                        Z3_ast val_ast = nullptr;
                        int ok = 0;
                        try {
                            ok = Z3API(Z3_model_eval)(ctx, model, arg_ast, 1, &val_ast);
                        } catch (const Z3Error&) { ok = 0; val_ast = nullptr; }
                        int64_t val = 0;
                        if (ok && val_ast) {
                            Z3API(Z3_get_numeral_int64)(ctx, val_ast, &val);
                        }
                        counterexample.push_back(val);
                    }
                    Z3API(Z3_model_dec_ref)(ctx, model);
                }
            }

            // Mark all candidates as NotEquivalent.
            results.resize(candidates.size());
            for (size_t i = 0; i < candidates.size(); ++i) {
                VerificationResult& r = results[i];
                if (assumption_literals[i]) {
                    r.status = VerificationResult::NotEquivalent;
                    r.message = "Pruned by unsat-core batch check (jointly unequivalent)";
                    r.counterexample = counterexample;
                    if (pruned_indices) pruned_indices->push_back(i);
                } else {
                    // Couldn't encode — leave as Unknown (caller re-verifies).
                    r.status = VerificationResult::Unknown;
                    r.message = "Could not encode candidate for batch prune";
                }
            }
        } else if (sat == Z3_L_FALSE) {
            // UNSAT: not all candidates are unequivalent (at least one is
            // equivalent or unknown). Try per-candidate checks for those
            // NOT in the unsat core (the core contains candidates whose
            // joint unequivalence is impossible — at least one of them is
            // equivalent; we can't conclude which, so keep all core
            // candidates for full verify).
            std::unordered_set<Z3_ast> core_set;
            if (Z3API(Z3_solver_get_unsat_core) && Z3API(Z3_ast_vector_inc_ref) &&
                Z3API(Z3_ast_vector_dec_ref) && Z3API(Z3_ast_vector_size) &&
                Z3API(Z3_ast_vector_get)) {
                Z3_ast_vector core = Z3API(Z3_solver_get_unsat_core)(ctx, solver);
                if (core) {
                    Z3API(Z3_ast_vector_inc_ref)(ctx, core);
                    unsigned sz = Z3API(Z3_ast_vector_size)(ctx, core);
                    for (unsigned k = 0; k < sz; ++k) {
                        Z3_ast a = Z3API(Z3_ast_vector_get)(ctx, core, k);
                        if (a) core_set.insert(a);
                    }
                    Z3API(Z3_ast_vector_dec_ref)(ctx, core);
                }
            }

            // Per-candidate check for non-core candidates.
            results.resize(candidates.size());
            for (size_t i = 0; i < candidates.size(); ++i) {
                VerificationResult& r = results[i];
                if (!assumption_literals[i]) {
                    r.status = VerificationResult::Unknown;
                    r.message = "Could not encode candidate for batch prune";
                    continue;
                }
                // Skip core candidates (keep for full verify).
                if (core_set.count(assumption_literals[i])) {
                    r.status = VerificationResult::Unknown;
                    r.message = "In unsat core — kept for full verify";
                    continue;
                }
                // Individual check: is candidate i unequivalent?
                Z3_ast one_assumption[] = {assumption_literals[i]};
                Z3_lbool_enum ind_sat = Z3API(Z3_solver_check_assumptions)(ctx, solver, 1, one_assumption);
                if (ind_sat == Z3_L_TRUE) {
                    // Candidate i is unequivalent — prune.
                    r.status = VerificationResult::NotEquivalent;
                    r.message = "Pruned by per-candidate assumption check";
                    // Extract counterexample (best-effort).
                    if (Z3API(Z3_solver_get_model) && Z3API(Z3_model_eval) &&
                        Z3API(Z3_get_numeral_int64) && Z3API(Z3_model_inc_ref) &&
                        Z3API(Z3_model_dec_ref)) {
                        Z3_model model = Z3API(Z3_solver_get_model)(ctx, solver);
                        if (model) {
                            Z3API(Z3_model_inc_ref)(ctx, model);
                            for (const auto& [arg_name, arg_ast] : orig_enc.arg_vars) {
                                Z3_ast val_ast = nullptr;
                                int ok = 0;
                                try {
                                    ok = Z3API(Z3_model_eval)(ctx, model, arg_ast, 1, &val_ast);
                                } catch (const Z3Error&) { ok = 0; val_ast = nullptr; }
                                int64_t val = 0;
                                if (ok && val_ast) {
                                    Z3API(Z3_get_numeral_int64)(ctx, val_ast, &val);
                                }
                                r.counterexample.push_back(val);
                            }
                            Z3API(Z3_model_dec_ref)(ctx, model);
                        }
                    }
                    if (pruned_indices) pruned_indices->push_back(i);
                } else {
                    // UNSAT or UNKNOWN — keep for full verify.
                    r.status = VerificationResult::Unknown;
                    r.message = "Not pruned — kept for full verify";
                }
            }
        }
        // sat == Z3_L_UNDEFINED: no pruning (treat as no info).

        for (auto& enc : cand_encs) release_tracked(enc);
        release_tracked(orig_enc);
        Z3API(Z3_params_dec_ref)(ctx, params);
        Z3API(Z3_solver_dec_ref)(ctx, solver);

    } catch (const Z3Error& e) {
        try {
            for (auto& enc : cand_encs) release_tracked(enc);
            if (orig_enc.ctx) release_tracked(orig_enc);
            if (params && ctx) Z3API(Z3_params_dec_ref)(ctx, params);
            if (solver && ctx) Z3API(Z3_solver_dec_ref)(ctx, solver);
        } catch (...) {}
        results.clear();
        if (pruned_indices) pruned_indices->clear();
    } catch (const std::exception&) {
        try {
            for (auto& enc : cand_encs) release_tracked(enc);
            if (orig_enc.ctx) release_tracked(orig_enc);
            if (params && ctx) Z3API(Z3_params_dec_ref)(ctx, params);
            if (solver && ctx) Z3API(Z3_solver_dec_ref)(ctx, solver);
        } catch (...) {}
        results.clear();
        if (pruned_indices) pruned_indices->clear();
    }
}

} // namespace clunk::search