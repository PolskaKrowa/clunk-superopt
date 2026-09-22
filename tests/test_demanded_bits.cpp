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
 * DemandedBits tests.
 *
 *  1. Hand-derived masks, one per rule.
 *  2. A SEMANTIC soundness property. For random functions, take a value,
 *     flip only the bits the analysis says nobody demands, and require the
 *     interpreter (itself validated against LLVM) to return exactly what it
 *     returned before. If the analysis ever under-reports demand, some
 *     input exposes it.
 */
#include <cstdint>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "clunk/Analysis/DemandedBits.h"
#include "clunk/Evaluator/DifferentialScreen.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Parser/IRParser.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;

static std::shared_ptr<ir::Function> fn(const std::string& src) {
    parser::IRParser p;
    return p.parse_string(src.c_str())->function("f");
}

static std::string hex(uint64_t v) { std::ostringstream o; o << "0x" << std::hex << v; return o.str(); }

// Check the demanded mask of `name` in a function body with signature `sig`.
#define EXPECT_DEMANDED(info, name, want) do { \
    const uint64_t got_ = (info).of(name); \
    CHECK(got_ == (uint64_t)(want), std::string("demanded(") + name + ") = " + hex(got_) + ", want " + hex((uint64_t)(want))); \
} while(0)

static analysis::DemandedBitsInfo analyse(const std::string& body, const char* sig = "i32 %x, i32 %y", const char* ret = "i32") {
    auto f = fn(std::string("define ") + ret + " @f(" + sig + ") {\nentry:\n" + body + "}\n");
    return analysis::compute_demanded_bits(*f);
}

// ── hand-derived masks ─────────────────────────────────────────────────────
static void test_roots_demand_everything() {
    auto d = analyse("  ret i32 %x\n");
    EXPECT_DEMANDED(d, "x", 0xFFFFFFFFu);
    EXPECT_DEMANDED(d, "y", 0);   // never used
    CHECK(d.is_dead("y"), "an unused argument is dead");
}

static void test_and_or_xor() {
    auto d = analyse("  %a = and i32 %x, 255\n  ret i32 %a\n");
    EXPECT_DEMANDED(d, "x", 0xFF);
    d = analyse("  %a = and i32 255, %x\n  ret i32 %a\n");
    EXPECT_DEMANDED(d, "x", 0xFF);                         // constant on the left too
    d = analyse("  %a = or i32 %x, 240\n  ret i32 %a\n");
    EXPECT_DEMANDED(d, "x", 0xFFFFFF0Fu);                  // bits the constant forces to 1 are irrelevant
    d = analyse("  %a = xor i32 %x, %y\n  %r = and i32 %a, 4080\n  ret i32 %r\n");
    EXPECT_DEMANDED(d, "a", 0xFF0);
    EXPECT_DEMANDED(d, "x", 0xFF0);
    EXPECT_DEMANDED(d, "y", 0xFF0);
    d = analyse("  %a = and i32 %x, %y\n  %r = and i32 %a, 15\n  ret i32 %r\n");
    EXPECT_DEMANDED(d, "x", 0xF);                          // variable and: D passes through to both
    EXPECT_DEMANDED(d, "y", 0xF);
}

static void test_carry_chains() {
    auto d = analyse("  %s = add i32 %x, %y\n  %r = and i32 %s, 255\n  ret i32 %r\n");
    EXPECT_DEMANDED(d, "s", 0xFF);
    EXPECT_DEMANDED(d, "x", 0xFF);   // carries only travel upwards: low 8 result bits need only the low 8 input bits
    EXPECT_DEMANDED(d, "y", 0xFF);
    d = analyse("  %s = add i32 %x, %y\n  %r = and i32 %s, 256\n  ret i32 %r\n");
    EXPECT_DEMANDED(d, "x", 0x1FF);  // bit 8 depends on every bit below it
    d = analyse("  %s = sub i32 %x, %y\n  %r = and i32 %s, 4096\n  ret i32 %r\n");
    EXPECT_DEMANDED(d, "x", 0x1FFF);
    d = analyse("  %s = mul i32 %x, %y\n  %r = and i32 %s, 15\n  ret i32 %r\n");
    EXPECT_DEMANDED(d, "x", 0xF);
    EXPECT_DEMANDED(d, "y", 0xF);
}

static void test_constant_shifts() {
    auto d = analyse("  %s = shl i32 %x, 8\n  ret i32 %s\n");
    EXPECT_DEMANDED(d, "x", 0x00FFFFFFu);      // the top 8 bits are shifted out
    d = analyse("  %s = lshr i32 %x, 8\n  ret i32 %s\n");
    EXPECT_DEMANDED(d, "x", 0xFFFFFF00u);      // the bottom 8 are shifted out
    d = analyse("  %s = ashr i32 %x, 8\n  ret i32 %s\n");
    EXPECT_DEMANDED(d, "x", 0xFFFFFF00u);      // (bit 31 is already demanded as the sign)
    d = analyse("  %s = ashr i32 %x, 8\n  %r = and i32 %s, 255\n  ret i32 %r\n");
    EXPECT_DEMANDED(d, "x", 0x0000FF00u);      // only the shifted-in copies of the sign are NOT needed here
    d = analyse("  %s = ashr i32 %x, 8\n  %r = and i32 %s, 0xFF000000\n  ret i32 %r\n");
    EXPECT_DEMANDED(d, "x", 0x80000000u);      // demanded bits 24-31 are all copies of the sign bit
}

static void test_variable_shifts() {
    auto d = analyse("  %s = lshr i32 %x, %y\n  %r = and i32 %s, 240\n  ret i32 %r\n");
    EXPECT_DEMANDED(d, "x", 0xFFFFFFF0u);      // any bit at or above the lowest demanded one may be shifted down
    EXPECT_DEMANDED(d, "y", 0xFFFFFFFFu);      // the whole amount matters (>= 32 is poison)
    d = analyse("  %s = shl i32 %x, %y\n  %r = and i32 %s, 240\n  ret i32 %r\n");
    EXPECT_DEMANDED(d, "x", 0xFF);             // any bit at or below the highest demanded one may be shifted up
}

static void test_casts() {
    auto d = analyse("  %t = trunc i32 %x to i8\n  ret i8 %t\n", "i32 %x", "i8");
    EXPECT_DEMANDED(d, "x", 0xFF);
    d = analyse("  %z = zext i8 %x to i32\n  ret i32 %z\n", "i8 %x");
    EXPECT_DEMANDED(d, "x", 0xFF);
    d = analyse("  %z = zext i8 %x to i32\n  %r = and i32 %z, 15\n  ret i32 %r\n", "i8 %x");
    EXPECT_DEMANDED(d, "x", 0xF);
    d = analyse("  %s = sext i8 %x to i32\n  ret i32 %s\n", "i8 %x");
    EXPECT_DEMANDED(d, "x", 0xFF);
    d = analyse("  %s = sext i8 %x to i32\n  %r = and i32 %s, 15\n  ret i32 %r\n", "i8 %x");
    EXPECT_DEMANDED(d, "x", 0xF);              // the extension bits are not demanded, so neither is the sign
    d = analyse("  %s = sext i8 %x to i32\n  %r = and i32 %s, 256\n  ret i32 %r\n", "i8 %x");
    EXPECT_DEMANDED(d, "x", 0x80);             // bit 8 is a copy of the sign bit (bit 7)
}

static void test_select_icmp_and_branches() {
    auto d = analyse("  %c = icmp eq i32 %x, 0\n  %s = select i1 %c, i32 %x, i32 %y\n  %r = and i32 %s, 15\n  ret i32 %r\n");
    EXPECT_DEMANDED(d, "x", 0xFFFFFFFFu);      // x is compared in full, so all of it is demanded
    EXPECT_DEMANDED(d, "y", 0xF);
    EXPECT_DEMANDED(d, "c", 1);
    d = analyse("  %c = icmp eq i32 %x, %y\n  ret i32 0\n");
    EXPECT_DEMANDED(d, "x", 0);                // an unused comparison demands nothing
    EXPECT_DEMANDED(d, "c", 0);
    d = analyse("  %c = icmp ult i32 %x, %y\n  br i1 %c, label %a, label %b\na:\n  ret i32 1\nb:\n  ret i32 2\n");
    EXPECT_DEMANDED(d, "c", 1);
    EXPECT_DEMANDED(d, "x", 0xFFFFFFFFu);
}

static void test_conservative_roots() {
    // A division can trap, so it demands every bit of both operands even if its result is unused.
    auto d = analyse("  %q = udiv i32 %x, %y\n  ret i32 0\n");
    EXPECT_DEMANDED(d, "x", 0xFFFFFFFFu);
    EXPECT_DEMANDED(d, "y", 0xFFFFFFFFu);
    d = analyse("  %p = alloca i32\n  store i32 %x, ptr %p\n  ret i32 0\n");
    EXPECT_DEMANDED(d, "x", 0xFFFFFFFFu);      // stored values are observable
    d = analyse("  %a = add i32 %x, 1\n  ret i32 0\n");
    CHECK(d.is_dead("a"), "an unused pure result is dead");
    EXPECT_DEMANDED(d, "x", 0);                // ...and so is everything only it reads
    CHECK(d.of("no_such_value") == ~uint64_t{0}, "an untracked name reports every bit");
}

static void test_phi_cycle() {
    auto f = fn(R"(
define i32 @f(i32 %n, i32 %a) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi i32 [ %a, %entry ], [ %acc2, %loop ]
  %inext = add i32 %i, 1
  %acc2 = add i32 %acc, %i
  %c = icmp ult i32 %inext, %n
  br i1 %c, label %loop, label %exit
exit:
  %r = and i32 %acc2, 255
  ret i32 %r
}
)");
    auto d = analysis::compute_demanded_bits(*f);
    EXPECT_DEMANDED(d, "acc2", 0xFF);
    EXPECT_DEMANDED(d, "acc", 0xFF);           // demand flows round the back-edge to a fixpoint
    EXPECT_DEMANDED(d, "a", 0xFF);
    EXPECT_DEMANDED(d, "i", 0xFFFFFFFFu);      // i feeds the loop test through inext: all bits matter
    EXPECT_DEMANDED(d, "inext", 0xFFFFFFFFu);
}

// ── semantic soundness property ────────────────────────────────────────────
struct Gen {
    std::vector<std::string> lines;                          // one instruction per line
    std::vector<std::pair<std::string, unsigned>> values;    // (name, width) of every value worth perturbing
    std::string header;
};

static Gen straight_line(std::mt19937_64& rng) {
    static const unsigned W[] = {8, 16, 32, 64};
    Gen g;
    const unsigned w0 = W[rng() % 4], w1 = W[rng() % 4];
    g.header = "define i" + std::to_string(w0) + " @f(i" + std::to_string(w0) + " %a, i" + std::to_string(w1) + " %b) {\nentry:";
    std::vector<std::pair<std::string, unsigned>> pool = {{"a", w0}, {"b", w1}};
    int n = 0;
    auto operand = [&](unsigned w) -> std::string {
        std::vector<std::string> ok;
        for (auto& [nm, ww] : pool) if (ww == w) ok.push_back("%" + nm);
        if (!ok.empty() && rng() % 4) return ok[rng() % ok.size()];
        static const long long C[] = {0, 1, 2, 3, 7, 8, 15, 31, 255, -1, -8, 100};
        return std::to_string(static_cast<long long>(C[rng() % 12]) & (w >= 64 ? -1LL : (1LL << (w - 1)) - 1));
    };
    for (int i = 0, len = 3 + rng() % 6; i < len; ++i) {
        const std::string nm = "t" + std::to_string(n++);
        const unsigned w = pool[rng() % pool.size()].second;
        const unsigned kind = rng() % 10;
        std::ostringstream l;
        if (kind < 6) {
            static const char* ops[] = {"add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr"};
            const char* op = ops[rng() % 9];
            std::string b = operand(w);
            if ((op[0] == 's' && op[1] == 'h') || op[0] == 'l' || (op[0] == 'a' && op[1] == 's'))
                b = rng() % 2 ? std::to_string(rng() % w) : "%" + pool[0].first;   // constant or variable amount
            l << "  %" << nm << " = " << op << " i" << w << " " << operand(w) << ", " << b;
        } else if (kind < 8) {
            const unsigned dst = W[rng() % 4];
            if (dst == w) { --n; continue; }
            const char* op = dst < w ? "trunc" : (rng() % 2 ? "zext" : "sext");
            std::string src;
            for (auto& [pn, pw] : pool) if (pw == w) { src = "%" + pn; break; }
            l << "  %" << nm << " = " << op << " i" << w << " " << src << " to i" << dst;
            pool.push_back({nm, dst}); g.values.push_back({nm, dst}); g.lines.push_back(l.str());
            continue;
        } else {
            const std::string c = "c" + std::to_string(n++);
            g.lines.push_back("  %" + c + " = icmp " + (rng() % 2 ? "ult" : "slt") + " i" + std::to_string(w) + " " + operand(w) + ", " + operand(w));
            l << "  %" << nm << " = select i1 %" << c << ", i" << w << " " << operand(w) << ", i" << w << " " << operand(w);
        }
        g.lines.push_back(l.str());
        pool.push_back({nm, w});
        g.values.push_back({nm, w});
    }
    // End by narrowing what is observed, so plenty of bits are undemanded.
    std::string last = "a"; 
    for (auto& [nm, ww] : pool) if (ww == w0) last = nm;
    static const unsigned long long M[] = {0xFF, 0xF0, 0xFFFF, 0x1, 0x80, 0xFFFFFF00ull, 0x7FFFFFFFull};
    const unsigned long long mask = M[rng() % 7] & (w0 >= 64 ? ~0ULL : (1ULL << w0) - 1);
    g.lines.push_back("  %res = and i" + std::to_string(w0) + " %" + last + ", " + std::to_string(static_cast<long long>(mask)));
    g.lines.push_back("  ret i" + std::to_string(w0) + " %res");
    return g;
}

static std::string assemble(const Gen& g, const std::vector<std::string>& lines) {
    std::string s = g.header + "\n";
    for (auto& l : lines) s += l + "\n";
    return s + "}\n";
}

// Insert `%name.n = xor iW %name, K` after the definition and route every other use through it.
static std::vector<std::string> inject_noise(const std::vector<std::string>& lines, const std::string& name,
                                             unsigned w, uint64_t noise) {
    const std::regex use("%" + name + "(?![A-Za-z0-9_.])");
    std::vector<std::string> out;
    const std::string def_prefix = "  %" + name + " = ";
    for (auto& l : lines) {
        if (l.rfind(def_prefix, 0) == 0) {
            out.push_back(l);
            out.push_back("  %" + name + ".n = xor i" + std::to_string(w) + " %" + name + ", " + std::to_string(static_cast<long long>(noise)));
        } else {
            out.push_back(std::regex_replace(l, use, "%" + name + ".n"));
        }
    }
    return out;
}

static uint64_t wmask(unsigned w) { return w >= 64 ? ~0ULL : (1ULL << w) - 1; }

static size_t g_experiments = 0, g_partial = 0;

static void check_noise_is_invisible(const std::string& original_src, const std::string& noisy_src,
                                     const std::string& what, const ir::Function& fo) {
    auto fnoisy = fn(noisy_src);
    if (!fnoisy) { CHECK(false, "noisy variant must parse: " + what + "\n" + noisy_src); return; }
    for (auto& in : evaluator::DifferentialScreen().make_inputs(fo)) {
        auto o = evaluator::Interpreter::run(fo, in);
        if (o.status != evaluator::ExecStatus::Value) continue;   // poison / UB in the original: no claim
        auto n = evaluator::Interpreter::run(*fnoisy, in);
        if (n.status != evaluator::ExecStatus::Value || n.value != o.value) {
            CHECK(false, "flipping only UNDEMANDED bits of " + what + " changed the result\n" + original_src + "--- noisy:\n" + noisy_src);
            return;
        }
    }
    ++g_experiments;
}

static void test_noise_in_undemanded_bits_is_invisible() {
    std::mt19937_64 rng(77);
    for (int t = 0; t < 200; ++t) {
        Gen g = straight_line(rng);
        const std::string src = assemble(g, g.lines);
        auto f = fn(src);
        if (!f) { CHECK(false, "generated IR must parse:\n" + src); continue; }
        auto d = analysis::compute_demanded_bits(*f);
        for (auto& [nm, w] : g.values) {
            if (!d.is_tracked(nm)) continue;
            const uint64_t und = ~d.of(nm) & wmask(w);
            if (!und) continue;                       // everything demanded: nothing to perturb
            ++g_partial;
            for (int k = 0; k < 2; ++k) {
                const uint64_t noise = (rng() & und) | (k == 0 ? und : 0);   // all undemanded bits once, random subset once
                check_noise_is_invisible(src, assemble(g, inject_noise(g.lines, nm, w, noise)), "%" + nm, *f);
            }
        }
    }
    CHECK(g_experiments > 500, "many perturbation experiments ran: " + std::to_string(g_experiments));
    CHECK(g_partial > 200, "with real partial demand (not vacuous): " + std::to_string(g_partial));
}

static void test_noise_through_a_loop() {
    const std::string src = R"(
define i32 @f(i32 %n, i32 %a) {
entry:
  %nb = and i32 %n, 7
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi i32 [ %a, %entry ], [ %acc2, %loop ]
  %inext = add i32 %i, 1
  %t = shl i32 %acc, 3
  %u = xor i32 %t, %i
  %acc2 = add i32 %u, %a
  %c = icmp ult i32 %inext, %nb
  br i1 %c, label %loop, label %exit
exit:
  %r = and i32 %acc2, 255
  ret i32 %r
}
)";
    auto f = fn(src);
    auto d = analysis::compute_demanded_bits(*f);
    CHECK(d.of("acc2") == 0xFF && d.of("a") == 0xFF, "the loop's accumulator only ever needs 8 bits");
    CHECK(d.of("acc") == 0x1F, "acc is demanded through the back-edge, and shl by 3 shifts three of its bits out");
    // Flip the undemanded bits of acc2 / acc / t / u / a and re-run through the loop.
    for (const char* nm : {"acc2", "acc", "u", "t"}) {
        const uint64_t und = ~d.of(nm) & 0xFFFFFFFFull;
        if (!und) continue;
        const std::regex use(std::string("%") + nm + "(?![A-Za-z0-9_.])");
        // route uses through a perturbed copy defined right after the definition
        std::istringstream in(src); std::string line;
        std::vector<std::string> lines;
        while (std::getline(in, line)) lines.push_back(line);
        std::vector<std::string> outl;
        const std::string def = std::string("  %") + nm + " = ";
        for (auto& l : lines) {
            if (l.rfind(def, 0) == 0) {
                outl.push_back(l);
                outl.push_back(std::string("  %") + nm + ".n = xor i32 %" + nm + ", " + std::to_string(static_cast<long long>(und & 0x5A5A5A5Aull) | 1LL << 20));
            } else outl.push_back(std::regex_replace(l, use, std::string("%") + nm + ".n"));
        }
        std::string noisy_src; for (auto& l : outl) noisy_src += l + "\n";
        check_noise_is_invisible(src, noisy_src, std::string("%") + nm + " (loop)", *f);
    }
}

int main() {
    std::cerr << "test_demanded_bits: which bits can anyone observe?\n";
    test_roots_demand_everything();
    test_and_or_xor();
    test_carry_chains();
    test_constant_shifts();
    test_variable_shifts();
    test_casts();
    test_select_icmp_and_branches();
    test_conservative_roots();
    test_phi_cycle();
    test_noise_in_undemanded_bits_is_invisible();
    test_noise_through_a_loop();
    std::cerr << "  (" << g_experiments << " noise experiments over " << g_partial << " partially-demanded values)\n";
    std::cerr << "passed " << g_pass << ", failed " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
