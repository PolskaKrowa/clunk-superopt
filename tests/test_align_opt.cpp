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
 * Clunk AlignOpt tests — provable vector-access alignment upgrades and
 * stack-realignment removal. The final test is a brute-force soundness
 * check that compares every raised `align` against concrete addresses.
 */
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"
#include "clunk/Parser/IRParser.h"
#include "clunk/Pipeline.h"
#include "clunk/Search/AlignOpt.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;
using namespace clunk::ir;
using namespace clunk::search;

// Parse `src`, run AlignOptimizer on function @f. Returns the rewritten
// function, or the ORIGINAL when nothing changed (so callers can always
// inspect the result); `changed` reports which happened.
struct Run {
    std::shared_ptr<Module> mod;
    std::shared_ptr<Function> fn;
    bool changed = false;
    AlignOptimizer::Stats stats;
};

static Run run(const std::string& src) {
    Run r;
    parser::IRParser p;
    r.mod = p.parse_string(src.c_str());
    auto orig = r.mod->function("f");
    AlignOptimizer opt;
    auto rewritten = opt.optimize(*orig, *r.mod);
    r.changed = rewritten != nullptr;
    r.fn = rewritten ? rewritten : orig;
    r.stats = opt.stats();
    return r;
}

static const Instruction* named(const Function& f, const std::string& n) {
    for (auto& bb : f.blocks())
        for (auto& i : bb->instructions())
            if (i && i->has_name() && i->name() == n) return i.get();
    return nullptr;
}

static const Instruction* nth(const Function& f, Opcode op, size_t k) {
    for (auto& bb : f.blocks())
        for (auto& i : bb->instructions())
            if (i && i->opcode() == op && k-- == 0) return i.get();
    return nullptr;
}

static unsigned al(const Instruction* i) { return i ? i->alignment().value_or(0) : 9999; }

// ── 1. Parameters ──────────────────────────────────────────────────────────
static void test_param_align() {
    auto r = run(R"(
define void @f(ptr align 32 %p, ptr %q) {
entry:
  %a = load <4 x double>, ptr %p, align 8
  %b = load <4 x double>, ptr %q, align 8
  store <4 x double> %a, ptr %p, align 8
  ret void
}
)");
    CHECK(r.changed, "aligned parameter enables a raise");
    CHECK(al(named(*r.fn, "a")) == 32, "load via align-32 param → align 32");
    CHECK(al(named(*r.fn, "b")) == 8,  "load via unannotated param is left alone");
    CHECK(al(nth(*r.fn, Opcode::Store, 0)) == 32, "store via align-32 param → align 32");
    CHECK(r.stats.accesses_raised == 2, "exactly two accesses raised");
}

static void test_param_align_capped_at_natural() {
    auto r = run(R"(
define void @f(ptr align 64 %p) {
entry:
  %a = load <2 x double>, ptr %p, align 1
  ret void
}
)");
    CHECK(al(named(*r.fn, "a")) == 16, "raise is capped at the vector's natural alignment");
}

static void test_never_lowers_existing_claim() {
    auto r = run(R"(
define void @f(ptr %p) {
entry:
  %a = load <4 x double>, ptr %p, align 32
  ret void
}
)");
    CHECK(!r.changed, "an unprovable but existing align 32 is never lowered");
    CHECK(al(named(*r.fn, "a")) == 32, "claim untouched");
}

static void test_scalars_untouched() {
    auto r = run(R"(
define void @f(ptr align 64 %p) {
entry:
  %a = load double, ptr %p, align 1
  ret void
}
)");
    CHECK(!r.changed, "scalar accesses are not vector moves: untouched");
}

// ── 2. GEP arithmetic ──────────────────────────────────────────────────────
static void test_gep_constant_offsets() {
    auto r = run(R"(
define void @f(ptr align 32 %p) {
entry:
  %g4 = getelementptr double, ptr %p, i64 4
  %g1 = getelementptr double, ptr %p, i64 1
  %g2 = getelementptr double, ptr %p, i64 2
  %a = load <4 x double>, ptr %g4, align 8
  %b = load <4 x double>, ptr %g1, align 1
  %c = load <4 x double>, ptr %g2, align 1
  ret void
}
)");
    CHECK(al(named(*r.fn, "a")) == 32, "base+32 keeps 32");
    CHECK(al(named(*r.fn, "b")) == 8,  "base+8 only proves 8 (1 → 8 is a legitimate raise)");
    CHECK(al(named(*r.fn, "c")) == 16, "base+16 proves 16");
}

static void test_gep_variable_index_known_bits() {
    auto r = run(R"(
define void @f(ptr align 32 %p, i64 %i) {
entry:
  %i4 = shl i64 %i, 2
  %g4 = getelementptr double, ptr %p, i64 %i4
  %g1 = getelementptr double, ptr %p, i64 %i
  %a = load <4 x double>, ptr %g4, align 1
  %b = load <4 x double>, ptr %g1, align 1
  ret void
}
)");
    CHECK(al(named(*r.fn, "a")) == 32, "(i<<2)*8 is a multiple of 32");
    CHECK(al(named(*r.fn, "b")) == 8,  "unknown i only proves the element size");
}

static void test_multi_index_array_gep() {
    auto r = run(R"(
define void @f(ptr align 32 %p, i64 %i) {
entry:
  %i2 = shl i64 %i, 1
  %g = getelementptr [8 x double], ptr %p, i64 0, i64 %i2
  %a = load <2 x double>, ptr %g, align 1
  ret void
}
)");
    CHECK(al(named(*r.fn, "a")) == 16, "[8 x double] step: (0*64) + (i<<1)*8 → 16");
}

static void test_struct_gep_not_trusted() {
    // size_bytes() ignores struct padding, so struct offsets must NOT be trusted.
    auto r = run(R"(
define void @f(ptr align 64 %p) {
entry:
  %g = getelementptr { i8, double }, ptr %p, i64 1
  %h = getelementptr { i8, double }, ptr %p, i64 0, i32 1
  %a = load <2 x double>, ptr %g, align 1
  %b = load <2 x double>, ptr %h, align 1
  ret void
}
)");
    CHECK(al(named(*r.fn, "a")) == 1, "struct stride is not trusted");
    CHECK(al(named(*r.fn, "b")) == 1, "struct field offset is not trusted");
}

static void test_zero_index_through_struct_is_fine() {
    auto r = run(R"(
define void @f(ptr align 32 %p) {
entry:
  %g = getelementptr { i8, double }, ptr %p, i64 0
  %a = load <4 x double>, ptr %g, align 1
  ret void
}
)");
    CHECK(al(named(*r.fn, "a")) == 32, "a constant-zero first index adds nothing, whatever the type");
}

// ── 3. Phis, selects, loops ────────────────────────────────────────────────
static std::string ptr_loop(int step_elems) {
    return std::string(R"(
define void @f(ptr align 32 %base, i64 %n) {
entry:
  br label %loop
loop:
  %p = phi ptr [ %base, %entry ], [ %next, %loop ]
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %v = load <4 x double>, ptr %p, align 1
  %next = getelementptr double, ptr %p, i64 )") + std::to_string(step_elems) + R"(
  %inc = add i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
)";
}

static void test_loop_carried_pointer_phi() {
    // Step of 4 doubles = 32 bytes: the entry value (32) and the step (32)
    // keep the phi 32-aligned on every iteration.
    auto r32 = run(ptr_loop(4));
    CHECK(al(named(*r32.fn, "v")) == 32, "phi advanced by 32 bytes stays 32-aligned");
    // Step of 2 doubles = 16 bytes: only 16 is invariant.
    auto r16 = run(ptr_loop(2));
    CHECK(al(named(*r16.fn, "v")) == 16, "phi advanced by 16 bytes is only 16-aligned");
    // Step of 1 double = 8 bytes.
    auto r8 = run(ptr_loop(1));
    CHECK(al(named(*r8.fn, "v")) == 8, "phi advanced by 8 bytes is only 8-aligned");
}

static void test_select_takes_the_weaker_pointer() {
    auto r = run(R"(
define void @f(ptr align 32 %a, ptr align 16 %b, i1 %c) {
entry:
  %s = select i1 %c, ptr %a, ptr %b
  %v = load <4 x double>, ptr %s, align 1
  ret void
}
)");
    CHECK(al(named(*r.fn, "v")) == 16, "select of a 32- and a 16-aligned pointer is 16-aligned");
}

// ── 3b. Induction variables (the vectoriser's loop shape) ──────────────────
static std::string index_loop(int start, int step, const char* idx_ty = "i64") {
    std::string ext = std::string(idx_ty) == "i64" ? "  %ix = add i64 %index, 0\n"
                                                   : "  %ix = zext i32 %index to i64\n";
    return "define void @f(ptr align 32 %x, i64 %n) {\nentry:\n  br label %body\nbody:\n"
           "  %index = phi " + std::string(idx_ty) + " [ " + std::to_string(start) + ", %entry ], [ %next, %body ]\n" + ext +
           "  %gep = getelementptr double, ptr %x, i64 %ix\n"
           "  %v = load <4 x double>, ptr %gep, align 8\n"
           "  %next = add nuw " + std::string(idx_ty) + " %index, " + std::to_string(step) + "\n"
           "  %done = icmp uge i64 %ix, %n\n  br i1 %done, label %exit, label %body\nexit:\n  ret void\n}\n";
}

static void test_induction_variable() {
    CHECK(al(named(*run(index_loop(0, 4)).fn, "v")) == 32, "i = 0; i += 4 over doubles → 32");
    CHECK(al(named(*run(index_loop(0, 8)).fn, "v")) == 32, "i += 8 → still capped at natural 32");
    CHECK(al(named(*run(index_loop(0, 2)).fn, "v")) == 16, "i += 2 → 16");
    CHECK(al(named(*run(index_loop(0, 1)).fn, "v")) == 8,  "i += 1 → 8");
    CHECK(al(named(*run(index_loop(1, 4)).fn, "v")) == 8,  "start offset 1 poisons the phi (i = 1, 5, 9...)");
    CHECK(al(named(*run(index_loop(2, 4)).fn, "v")) == 16, "start 2 step 4 → i ≡ 2 mod 4 → 16");
    CHECK(al(named(*run(index_loop(0, 4, "i32")).fn, "v")) == 32, "zext of an i32 induction variable");
}

static void test_and_mask_and_mul() {
    auto r = run(R"(
define void @f(ptr align 32 %x, i64 %n, i64 %m) {
entry:
  %masked = and i64 %n, -4
  %scaled = mul i64 %m, 4
  %g1 = getelementptr double, ptr %x, i64 %masked
  %g2 = getelementptr double, ptr %x, i64 %scaled
  %a = load <4 x double>, ptr %g1, align 1
  %b = load <4 x double>, ptr %g2, align 1
  ret void
}
)");
    CHECK(al(named(*r.fn, "a")) == 32, "n & -4 is a multiple of 4 → 32 bytes over doubles");
    CHECK(al(named(*r.fn, "b")) == 32, "m * 4 is a multiple of 4 → 32 bytes over doubles");
}

// ── 4. Globals ─────────────────────────────────────────────────────────────
static void test_global_alignment() {
    auto r = run(R"(
@g = global [8 x double] zeroinitializer, align 64
@h = global [8 x double] zeroinitializer

define void @f() {
entry:
  %p = getelementptr [8 x double], ptr @g, i64 0, i64 4
  %q = getelementptr [8 x double], ptr @h, i64 0, i64 4
  %a = load <4 x double>, ptr %p, align 1
  %b = load <4 x double>, ptr %q, align 1
  %c = load <4 x double>, ptr @g, align 1
  ret void
}
)");
    CHECK(al(named(*r.fn, "a")) == 32, "@g align 64 + 32 bytes → 32");
    CHECK(al(named(*r.fn, "b")) == 1,  "@h has no explicit alignment → nothing provable");
    CHECK(al(named(*r.fn, "c")) == 32, "direct @g access capped at natural 32");
}

// ── 5. Stack realignment ───────────────────────────────────────────────────
static void test_alloca_lowered_when_overaligned() {
    auto r = run(R"(
define double @f() {
entry:
  %buf = alloca [4 x double], align 32
  %g = getelementptr [4 x double], ptr %buf, i64 0, i64 1
  store double 1.0, ptr %g, align 8
  %x = load double, ptr %g, align 8
  ret double %x
}
)");
    CHECK(r.changed, "over-aligned alloca with modest users is relaxed");
    CHECK(al(named(*r.fn, "buf")) == 16, "relaxed to the stack alignment (16), not below");
    CHECK(r.stats.allocas_lowered == 1, "one alloca relaxed");
}

static void test_alloca_kept_when_access_needs_it() {
    auto r = run(R"(
define void @f(<4 x double> %v) {
entry:
  %buf = alloca <4 x double>, align 32
  store <4 x double> %v, ptr %buf, align 32
  ret void
}
)");
    CHECK(!r.changed, "an align-32 access keeps its align-32 alloca");
    CHECK(al(named(*r.fn, "buf")) == 32, "alloca untouched");
}

static void test_alloca_kept_when_escaping() {
    const char* variants[] = {
        // passed to a call
        "  call void @sink(ptr %buf)\n",
        // observed through ptrtoint
        "  %n = ptrtoint ptr %buf to i64\n  call void @use(i64 %n)\n",
        // stored as a value
        "  store ptr %buf, ptr %slot, align 8\n",
    };
    for (const char* v : variants) {
        std::string src =
            "declare void @sink(ptr)\ndeclare void @use(i64)\n"
            "define void @f(ptr %slot) {\nentry:\n"
            "  %buf = alloca [4 x double], align 32\n"
            "  %x = load double, ptr %buf, align 8\n" + std::string(v) + "  ret void\n}\n";
        auto r = run(src);
        CHECK(al(named(*r.fn, "buf")) == 32, std::string("escaping alloca keeps its alignment: ") + v);
    }
}

static void test_lifetime_markers_do_not_block() {
    auto r = run(R"(
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
define double @f() {
entry:
  %buf = alloca [4 x double], align 32
  call void @llvm.lifetime.start.p0(i64 32, ptr %buf)
  store double 2.0, ptr %buf, align 8
  %x = load double, ptr %buf, align 8
  call void @llvm.lifetime.end.p0(i64 32, ptr %buf)
  ret double %x
}
)");
    CHECK(al(named(*r.fn, "buf")) == 16, "llvm.lifetime.* markers are not escapes");
}

static void test_alloca_not_lowered_below_stack_align_and_untouched_if_small() {
    auto r = run(R"(
define double @f() {
entry:
  %a = alloca double, align 8
  %b = alloca <2 x double>, align 16
  store double 1.0, ptr %a, align 8
  %x = load double, ptr %a, align 8
  ret double %x
}
)");
    CHECK(!r.changed, "allocas at or below the stack alignment are never touched");
}

static void test_lowering_precedes_raising() {
    // The alloca is over-aligned (32) but every access says 8. Relaxing it
    // must win: the <4 x double> access must NOT be raised to 32 on the
    // strength of an alignment we are in the middle of removing.
    auto r = run(R"(
define <4 x double> @f() {
entry:
  %buf = alloca [8 x double], align 32
  %v = load <4 x double>, ptr %buf, align 8
  ret <4 x double> %v
}
)");
    CHECK(al(named(*r.fn, "buf")) == 16, "over-aligned alloca relaxed to 16");
    CHECK(al(named(*r.fn, "v")) == 16, "access raised only to what the relaxed alloca proves (16)");
}

static void test_alloca_gep_index_claim_reproved() {
    // Access at +8 claims align 16, which is NOT derivable (offset 8) even
    // before we touch anything. The re-proof must fail and leave the
    // alloca alone rather than silently dropping a claim's support.
    auto r = run(R"(
define void @f(<2 x double> %v) {
entry:
  %buf = alloca [8 x double], align 32
  %g = getelementptr [8 x double], ptr %buf, i64 0, i64 1
  store <2 x double> %v, ptr %g, align 16
  ret void
}
)");
    CHECK(al(named(*r.fn, "buf")) == 32, "alloca untouched when an access cannot be re-proven");
}

// ── 6. Purity & attribute preservation ─────────────────────────────────────
static void test_input_not_mutated_and_attrs_kept() {
    parser::IRParser p;
    auto m = p.parse_string(R"(
define void @f(ptr align 32 %p) #0 {
entry:
  %a = load <4 x double>, ptr %p, align 8
  ret void
}
attributes #0 = { nounwind }
)");
    auto orig = m->function("f");
    const auto attrs_before = orig->function_attributes();
    AlignOptimizer opt;
    auto out = opt.optimize(*orig, *m);
    CHECK(out != nullptr, "rewritten");
    if (!out) return;
    CHECK(al(named(*orig, "a")) == 8, "the input function is never mutated");
    CHECK(al(named(*out, "a")) == 32, "the copy carries the upgrade");
    CHECK(out->function_attributes() == attrs_before, "function attributes (#0, ...) survive the copy");
}

// ── 6b. Robustness: the search can emit operand-less instructions ──────────
static void test_malformed_instructions_do_not_throw() {
    parser::IRParser p;
    auto m = p.parse_string(R"(
define void @f(ptr align 32 %p, i32 %x) {
entry:
  %v = load <4 x double>, ptr %p, align 8
  ret void
}
)");
    auto fn = m->function("f");
    auto& entry = *fn->blocks().front();
    const auto i32 = fn->arguments()[1].type;
    const auto ptr = fn->arguments()[0].type;
    for (auto op : {Opcode::Shl, Opcode::Add, Opcode::Phi, Opcode::Select, Opcode::ZExt})
        entry.add_instruction(std::make_shared<Instruction>(op, i32, std::string("bad_") + std::to_string(int(op))));
    for (auto op : {Opcode::GetElementPtr, Opcode::BitCast, Opcode::Phi, Opcode::Select})
        entry.add_instruction(std::make_shared<Instruction>(op, ptr, std::string("badp_") + std::to_string(int(op))));
    bool threw = false;
    std::shared_ptr<Function> out;
    try { out = AlignOptimizer().optimize(*fn, *m); } catch (...) { threw = true; }
    CHECK(!threw, "operand-less instructions must not make the pass throw");
    CHECK(out && al(named(*out, "v")) == 32, "well-formed accesses are still upgraded alongside malformed ones");
}

// ── 7. End-to-end through the Pipeline ─────────────────────────────────────
static void test_pipeline_end_to_end() {
    const char* src = R"(
define void @f(ptr align 32 %p, ptr %q) {
entry:
  %a = load <4 x double>, ptr %p, align 8
  store <4 x double> %a, ptr %q, align 8
  ret void
}
)";
    parser::IRParser parser;
    auto m = parser.parse_string(src);
    for (bool enabled : {true, false}) {
        PipelineConfig cfg;
        cfg.enable_align_opt = enabled;
        cfg.opt_level = 1;
        cfg.time_budget = 5.0;
        Pipeline pl(cfg);
        auto res = pl.run(*m);
        const std::string out = res.optimised_module->to_string();
        const bool raised = out.find("load <4 x double>, ptr %p, align 32") != std::string::npos;
        CHECK(raised == enabled, std::string("pipeline align pass honours its flag (enabled=") + (enabled ? "1" : "0") + ")\n" + out);
        CHECK(out.find("ptr align 32 %p") != std::string::npos ||
              out.find("align 32 %p") != std::string::npos,
              "the parameter's `align 32` survives printing, so the raise is self-justifying\n" + out);
    }
    // The caller's module is never touched.
    CHECK(al(named(*m->function("f"), "a")) == 8, "pipeline leaves the input module intact");
}

// ── 8. Brute-force soundness ───────────────────────────────────────────────
// For a grid of (base alignment, element type, shift, constant offset) build
// a GEP, run the pass, and — for every concrete base address and index value
// — check that the alignment the pass wrote really divides the address. Real
// alloc sizes come from this table, NOT from the implementation.
static void test_bruteforce_soundness() {
    struct Ty { const char* ir; unsigned long size; };
    const Ty tys[] = {
        {"i8", 1}, {"i16", 2}, {"i32", 4}, {"i64", 8}, {"double", 8}, {"float", 4}, {"ptr", 8},
        {"<4 x float>", 16}, {"<3 x float>", 16 /* alloc size is rounded up */},
        {"<4 x double>", 32}, {"[3 x i32]", 12}, {"[4 x double]", 32}, {"[5 x i8]", 5},
        {"{ i8, i64 }", 16}, {"{ i32, i8 }", 8},
    };
    const unsigned bases[] = {1, 2, 4, 8, 16, 32, 64};
    size_t configs = 0, raised = 0;
    for (const Ty& t : tys)
        for (unsigned base : bases)
            for (int shift = 0; shift <= 3; ++shift)
                for (long c = 0; c <= 3; ++c) {
                    char buf[1024];
                    std::snprintf(buf, sizeof buf, R"(
define void @f(ptr align %u %%p, i64 %%n) {
entry:
  %%i = shl i64 %%n, %d
  %%j = add i64 %%i, %ld
  %%g = getelementptr %s, ptr %%p, i64 %%j
  %%v = load <2 x double>, ptr %%g, align 1
  ret void
}
)", base, shift, c, t.ir);
                    auto r = run(buf);
                    ++configs;
                    const unsigned got = al(named(*r.fn, "v"));
                    if (got > 1) ++raised;
                    CHECK(got == 1 || got == 2 || got == 4 || got == 8 || got == 16, "raised to a power of two ≤ natural");
                    // Every reachable address: base is any multiple of `base`,
                    // n is any value; offset = size*((n<<shift)+c).
                    bool sound = true;
                    for (unsigned long m = 0; m < 4 && sound; ++m)
                        for (unsigned long n = 0; n < 40 && sound; ++n) {
                            const unsigned long addr = m * base + t.size * ((n << shift) + c);
                            if (addr % got != 0) {
                                std::cerr << "UNSOUND: base=" << base << " type=" << t.ir << " shift=" << shift
                                          << " c=" << c << " → align " << got << " but addr=" << addr << "\n";
                                sound = false;
                            }
                        }
                    CHECK(sound, "raised alignment must divide every reachable address");
                }
    CHECK(configs > 1000, "grid was exercised");
    CHECK(raised > 100, "grid produced a meaningful number of raises (not vacuous)");
}

// Same idea for loops: simulate every iteration's concrete index.
static void test_bruteforce_induction() {
    size_t raised = 0;
    for (unsigned base : {1u, 4u, 8u, 16u, 32u, 64u})
        for (int start = 0; start <= 7; ++start)
            for (int step : {1, 2, 3, 4, 6, 8, 12, 16})
                for (unsigned long elem : {1ul, 4ul, 8ul}) {
                    const char* ety = elem == 1 ? "i8" : elem == 4 ? "i32" : "double";
                    char buf[1024];
                    std::snprintf(buf, sizeof buf, R"(
define void @f(ptr align %u %%x, i64 %%n) {
entry:
  br label %%body
body:
  %%index = phi i64 [ %d, %%entry ], [ %%next, %%body ]
  %%gep = getelementptr %s, ptr %%x, i64 %%index
  %%v = load <2 x double>, ptr %%gep, align 1
  %%next = add i64 %%index, %d
  %%done = icmp uge i64 %%next, %%n
  br i1 %%done, label %%exit, label %%body
exit:
  ret void
}
)", base, start, ety, step);
                    const unsigned got = al(named(*run(buf).fn, "v"));
                    if (got > 1) ++raised;
                    bool sound = true;
                    for (unsigned long m = 0; m < 4 && sound; ++m)
                        for (unsigned long t = 0; t < 64 && sound; ++t)
                            if ((m * base + elem * (start + step * t)) % got != 0) {
                                std::cerr << "UNSOUND(loop): base=" << base << " start=" << start << " step=" << step
                                          << " elem=" << elem << " → align " << got << "\n";
                                sound = false;
                            }
                    CHECK(sound, "induction-variable alignment divides every iteration's address");
                }
    CHECK(raised > 50, "induction grid is not vacuous");
}

int main() {
    std::cerr << "test_align_opt: provable vector alignment + stack realignment removal\n";
    test_param_align();
    test_param_align_capped_at_natural();
    test_never_lowers_existing_claim();
    test_scalars_untouched();
    test_gep_constant_offsets();
    test_gep_variable_index_known_bits();
    test_multi_index_array_gep();
    test_struct_gep_not_trusted();
    test_zero_index_through_struct_is_fine();
    test_loop_carried_pointer_phi();
    test_select_takes_the_weaker_pointer();
    test_induction_variable();
    test_and_mask_and_mul();
    test_global_alignment();
    test_alloca_lowered_when_overaligned();
    test_alloca_kept_when_access_needs_it();
    test_alloca_kept_when_escaping();
    test_lifetime_markers_do_not_block();
    test_alloca_not_lowered_below_stack_align_and_untouched_if_small();
    test_lowering_precedes_raising();
    test_alloca_gep_index_claim_reproved();
    test_input_not_mutated_and_attrs_kept();
    test_malformed_instructions_do_not_throw();
    test_pipeline_end_to_end();
    test_bruteforce_soundness();
    test_bruteforce_induction();
    std::cerr << "passed " << g_pass << ", failed " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
