<div align=center>
<img width="512" height="512" alt="Clunk logo" src="https://github.com/user-attachments/assets/dc2363aa-d936-4fd7-a952-4b752861e908" />

<br>
<i>With a good set of tools, even anvils can move fast.</i>
<br>
<br>

[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Z3](https://img.shields.io/badge/SMT-Z3-orange.svg)](https://github.com/Z3Prover/z3)
[![Version](https://img.shields.io/badge/version-v0.1.0-lightgrey.svg)](#)

</div>

**Clunk** is an LLVM IR superoptimiser. It takes LLVM IR (`.ll` files) and
searches for instruction-level optimisations using stochastic search,
evolutionary algorithms, SMT verification (via Z3), e-graph rewriting, and
peephole mining. Written in C++17 and built with CMake.

>[!NOTE]
> This program is ***NOT*** intended to override LLVM's existing optimisation passes.
> Instead, it is to provide *additional* optimisations that clang/llvm-opt have missed
> for larger programs using atypical coding styles.

Version: v0.1.0

## Example

```llvm
; input.ll (derived from examples/cross_fn.ll)

define internal i32 @double(i32 %x) {
entry:
  %r = shl i32 %x, 1
  ret i32 %r
}

; Multi-block callee: double if x>=0 else 0
define internal i32 @double_if_pos(i32 %x) {
entry:
  %cmp = icmp sge i32 %x, 0
  br i1 %cmp, %then, %else
then:
  %r = call i32 @double(i32 %x)
  ret i32 %r
else:
  ret i32 0
}

; Both callers pass the constant 5 → IPCP should specialise.
define i32 @caller_a() {
entry:
  %c = call i32 @double_if_pos(i32 5)
  ret i32 %c
}

define i32 @caller_b() {
entry:
  %c = call i32 @double_if_pos(i32 5)
  ret i32 %c
}

; Dead function — never called.
define internal i32 @dead_helper(i32 %x) {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}

define i32 @main() {
entry:
  %a = call i32 @caller_a()
  %b = call i32 @caller_b()
  %s = add i32 %a, %b
  ret i32 %s
}
```

<div align=center>VVV</div>

```bash
$ clunk input.ll -o output.ll


=== Clunk Summary ===
Functions processed: 5
Functions optimised: 1
Average improvement:  5.53571x
Total time:           553.588 ms

$ █
```

<div align=center>VVV</div>

```llvm
; output.ll

; ModuleID = ''
define i32 @caller_a() {
entry.cont.inl0:
  ret i32 10
}

define i32 @caller_b() {
entry.cont.inl0:
  ret i32 10
}

define i32 @main() {
entry.cont.inl0.cont.inl0:
  ret i32 20
}
```

Clunk rewrote the whole program to directly return the processed arithmetic within the program, verified it against the
original with Z3 + Alive2, and rewrote the IR accordingly, All while evaluating each function with their own thread (the input program was small enough to allow for that, larger programs with many functions may assign multiple functions per thread, allowing clunk to quickly find cross-function optimisation opportunities, if any.)

One future addition to Clunk will be to add comments to the optimised IR which shows exactly what clunk did to optimise the code, making it easier for LLVM developers to write stronger optimisation passes.

## Features

- **Stochastic search** -- simulated annealing over instruction-level
  transformations.
- **Evolutionary search** -- tournament selection, crossover, and mutation
  over program variants.
- **SMT-based equivalence verification** -- Z3 integration (loaded at
  runtime via `dlopen`, no link-time dependency) to prove candidate
  transformations preserve program semantics.
- **E-graph rewriting** -- principled term rewriting with equality saturation
  for exploring equivalent program forms.
- **Peephole pattern mining** -- automatic discovery of optimisation
  patterns using CEGIS (counter-example guided inductive synthesis).
- **Hole-based progressive-deepening synthesis** -- replaces a function
  body with a "hole" and enumerates 1-instruction, then 2, then 3 (etc.)
  equivalents, SMT-verifying each. Finds the SHORTEST equivalent form
  (Massalin-style enumerative superoptimisation). Complements the
  stochastic / evolutionary phases as a deterministic,
  completeness-bounded search.
- **Block-level divide-and-conquer optimisation** -- when the whole-
  function search is stagnant, splits the function's single basic block
  into a binary tree of contiguous instruction ranges (halving at each
  level, minimum size 2 instructions) and SMT-proves cheaper equivalents
  for each range, preserving data flow between ranges. A range is
  declared "naturally optimal" if it cannot be improved, or "synthetically
  optimal" if it was rewritten. When both children of a block are
  synthetically optimal, the parent is re-tried (the new context may
  unlock further wins). When all children are naturally optimal, the
  parent is declared naturally optimal without re-trying. This finds
  wins the whole-function search misses (where the whole function has
  no short equivalent, but a sub-range does).
- **Series-expansion loop optimisation** -- detects single-block loops
  that compute arithmetic series and replaces the entire loop with its
  closed-form expression. Recognised patterns:
  - `for(i=0;i<n;i++) sum += i` → `n*(n-1)/2` (arithmetic series)
  - `for(i=0;i<n;i++) sum += c` → `c*n` (constant accumulation)
  - `for(i=0;i<n;i++) sum += i*c` → `c * n*(n-1)/2` (scaled arithmetic)
  - Handles nonzero start values and non-unit steps.
  Sound by construction (the algebra is exact, same trust tier as LICM
  and LoopOpt). Loops with `nsw`/`nuw` flags on the accumulator are
  refused (the closed form's intermediate multiplications may wrap
  differently than the original's sequential adds).
- **Vector synthesis** -- width-aware SIMD superoptimisation. Tries
  AVX-512 → AVX2 → AVX → scalar in cascade order, picking the widest
  tier that yields a verified cheaper rewrite. Performs lane
  decomposition (splitting wide vectors into narrower ones via
  `shufflevector`) and surrounding-code rewriting (promoting scalar
  `load` chains into vector `load <N x T>`).
- **Algorithmic preprocessor** -- module-level pre-pass that walks the
  call graph and detects functions (or compositions of functions) whose
  output is a predictable closed form (`f(x) ≡ C`, `f(x) ≡ c·x`,
  `f(x) ≡ c·x + b`). When a pattern is detected and SMT-proven, the
  function's body is rewritten to the minimal closed form, shrinking
  the work the per-function pipeline has to do.
- **Loop optimisation** -- loop-aware transformations and analysis.
- **Memory optimisation** -- memory access pattern improvements.
- **Cost model evaluation** -- TTI-based and MCA-based (llvm-mca) cost
  estimation to rank candidate programs.
- **GPU/PTX optimisation stubs** -- PTX emitter, occupancy model, divergence
  analysis, liveness analysis, and kernel launch optimisation.
- **Interpreter-based evaluation** -- fast program evaluation with caching
  for fitness assessment during search.
- **ncurses TUI** -- an optional live progress view (`--tui`) that shows
  every function being superoptimised in a two-panel layout: a scrollable
  function list on the left (with live stage + improvement ratio) and a
  detail panel on the right showing the current-best IR snapshot,
  round/score/verified status, and elapsed time. The user can navigate
  with ↑/↓, pin a function with Tab/p, and quit with q.

>[!WARNING]
> **E-graph rewriting is an experimental pass and is only intended for small-scale programs.**
>
> Instances where E-graphs return non-functional LLVM code should **NOT** be reported as a bug, as it's already a known issue.
> It is therefore disabled by default and can be re-enabled with the `--egraph` flag.

>[!WARNING]
> **Vector synthesis is not guaranteed to generate optimal vectorised code from scalar inputs**
>
> Code vectorisation is a very difficult task even for computers to perform efficiently and effectively for large operation spaces.
> The width-cascade pass (AVX-512 → AVX2 → AVX → scalar) covers the common cases — including lane decomposition for too-wide inputs and surrounding-code rewriting (scalar `load` → vector `load <N x T>`) — but it is conservative: it only fires when the cost model says the rewrite is strictly cheaper AND SMT proves equivalence.
> If at any point you notice clunk generating vectorised code that ends up being slower than the original, or if clunk misses an opportunity
> to create viable and performant vector code in comparison to a scalar equivalent, Please open an issue describing the exact function that was
> processed (or should've been processed) by clunk and what clunk has generated as its "ideal output".

## Building

### Requirements

- CMake 3.20 or later
- A C++17 compiler (GCC or Clang)
- Optional: Z3 + Alive2 shared libraries (loaded at runtime)
- Optional: llvm-mca (for MCA-based cost model)
- Optional: ncurses (libncurses-dev) — enables the `--tui` live progress view.
  When not installed, clunk builds without TUI support; `--tui` then prints a
  warning and runs without the TUI.

### Build instructions

```sh
mkdir build && cd build
cmake .. -DCLUNK_ENABLE_TESTS=ON -DCLUNK_ENABLE_BENCHMARKS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `CLUNK_ENABLE_Z3` | ON | Enable Z3 SMT verification (runtime dlopen) |
| `CLUNK_ENABLE_ALIVE2` | ON | Enable Alive2 code verification (runtime dlopen) |
| `CLUNK_ENABLE_TESTS` | ON | Build test suite |
| `CLUNK_ENABLE_BENCHMARKS` | ON | Build benchmarks |
| `CLUNK_ENABLE_GPU` | ON | Enable GPU/PTX optimisation stubs |
| `CLUNK_ENABLE_LTO` | ON | Link-time optimisation for Release builds |
| `CLUNK_ENABLE_NATIVE` | OFF | Compile with `-march=native` (non-portable) |
| `CLUNK_ENABLE_GC_SECTIONS` | ON | Compile with `-ffunction-sections -fdata-sections -Wl,--gc-sections` (more effective dead-code stripping from clunk) |
| `CLUNK_ENABLE_TUI` | auto | Build with ncurses TUI support (`--tui` flag). Auto-detected; forced OFF when ncurses is not found. |

## Usage

```
clunk [options] <input.ll>
```

### Key options

| Option | Description |
|--------|-------------|
| `--opt-level <N>` | Optimisation aggressiveness (default: max) |
| `--time-budget <sec>` | Wall-clock time limit for the search (default: 30s) |
| `--output <file>` | Write optimised IR to a file |
| `--verbose` | Increase output verbosity |
| `--mine` | Run peephole pattern mining |
| `--pattern-library <path>` | Path to a pattern library file |
| `--report-json` | Write a JSON report of the search to stdout |
| `--no-z3` | Disable SMT verification |
| `--no-alive2` | Disable Alive2 verification (not recommended, as I don't fully trust clunk's codegen fully yet) |
| `--no-gpu` | Disable GPU/PTX passes |
| `--no-miner` | Disable peephole miner |
| `--no-vector-synth` | Disable vector synthesis |
| `--no-hole-synth` | Disable hole-based progressive-deepening synthesis |
| `--no-block-opt` | Disable block-level divide-and-conquer optimisation (fallback when whole-function search is stagnant) |
| `--no-series-expand` | Disable series-expansion loop optimisation (arithmetic-series loops → closed forms) |
| `--no-algo-preprocessor` | Disable module-level algorithmic preprocessor |
| `--vector-width <tier>` | Widest vector tier to attempt: `avx512`, `avx2`, `avx`, or `auto` (default: auto) |
| `--tui` | Launch an ncurses TUI showing live superoptimiser progress (function list + current-best IR preview). Keys: ↑/↓ nav, Tab/p pin, r toggle raw IR, q quit. |
| `--mca` | Enable MCA-based cost model ranking |

## Project Structure

```
clunk/
  CMakeLists.txt          -- Top-level build configuration
  cmake/                  -- CMake modules (ClunkOpt, ClunkConfig)
  src/
    cli/                  -- Command-line interface + ncurses TUI (tui.cpp)
    IR/                   -- LLVM IR data structures and utilities
    Analysis/             -- Program analyses (known bits, dataflow)
    Search/               -- Search strategies, SMT verifier, e-graphs, mining,
                             hole-synth, block-opt, series-expand, algo-preprocessor
    Evaluator/            -- Cost models, interpreter, evaluation engine
    GPU/                  -- PTX emitter, occupancy, divergence, liveness
    Pattern/              -- Pattern library management
    Parser/               -- LLVM IR (.ll) parser
  include/clunk/          -- Public headers (mirrors src/ layout)
  tests/                  -- Unit tests
  benchmarks/             -- Benchmarks and corpus files
  scripts/                -- Helper scripts (diff testing, O3 comparison)
  vendor/                 -- Vendored dependencies
  examples/               -- Example .ll files
```

## License

Clunk is released under the [GNU General Public License v3 or later](LICENSE).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to
build from source, run the test suite, and submit a pull request. Please
open an issue or pull request on GitHub.