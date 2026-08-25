cmake_minimum_required(VERSION 3.20)
project(Clunk VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ──────────────────────────────────────────────────────────────────
option(CLUNK_ENABLE_Z3 "Enable Z3 SMT verification (runtime, via dlopen)" ON)
option(CLUNK_ENABLE_ALIVE2 "Enable Alive2 verification (runtime, via dlopen)" ON)
option(CLUNK_ENABLE_TESTS "Build test suite" ON)
option(CLUNK_ENABLE_BENCHMARKS "Build benchmarks" ON)
option(CLUNK_ENABLE_GPU "Enable GPU/PTX optimisation stubs" ON)

# ── Z3 ───────────────────────────────────────────────────────────────────────
# Z3 is now a purely runtime dependency loaded via dlopen/dlsym.
# There is NO link-time dependency on libz3 — we only need -ldl.
# The vendored z3.h is kept under vendor/ for reference only.
if(CLUNK_ENABLE_Z3)
    add_compile_definitions(CLUNK_HAS_Z3=1)
    message(STATUS "Z3 SMT verification: ENABLED (runtime dlopen, no link-time dependency)")
else()
    message(STATUS "Z3 SMT verification: DISABLED")
endif()

# ── Alive2 ─────────────────────────────────────────────────────────────────────
# Alive2 is a purely runtime dependency loaded via dlopen/dlsym.
# There is NO link-time dependency on libAlive2 — we only need -ldl.
# The vendored alive2_c.h is kept under vendor/alive2/ for reference.
# At runtime, AliveVerifier auto-detects:
#   1. libAlive2.so (in-process, preferred)
#   2. alive-tv binary on PATH (out-of-process fallback)
if(CLUNK_ENABLE_ALIVE2)
    add_compile_definitions(CLUNK_HAS_ALIVE2=1)
    message(STATUS "Alive2 verification: ENABLED (runtime dlopen, no link-time dependency)")
else()
    message(STATUS "Alive2 verification: DISABLED")
endif()

# ── Core library ─────────────────────────────────────────────────────────────
set(CLUNK_SOURCES
    src/IR/Type.cpp
    src/IR/Value.cpp
    src/IR/Instruction.cpp
    src/IR/BasicBlock.cpp
    src/IR/Function.cpp
    src/IR/Module.cpp
    src/IR/IRBuilder.cpp
    src/IR/LoopAnalysis.cpp
    src/IR/Scalarizer.cpp
    src/IR/DataflowPrune.cpp
    src/IR/StrengthReduce.cpp
    src/Analysis/KnownBits.cpp
    src/Analysis/PowersOfTwo.cpp
    src/Analysis/CallGraph.cpp
    src/Parser/IRParser.cpp
    src/Evaluator/Heuristics.cpp
    src/Evaluator/EvaluationEngine.cpp
    src/Evaluator/CostModel.cpp
    src/Evaluator/EvaluationCache.cpp
    src/Evaluator/Interpreter.cpp
    src/Evaluator/MCACostModel.cpp
    src/Search/StochasticSearch.cpp
    src/Search/EvolutionarySearch.cpp
    src/Search/SMTVerifier.cpp
    src/Search/AliveDynamicLoader.cpp
    src/Search/AliveVerifier.cpp
    src/Search/PeepholeMiner.cpp
    src/Search/VectorSynth.cpp
    src/Search/HoleSynth.cpp
    src/Search/BlockOptimiser.cpp
    src/Search/SeriesExpander.cpp
    src/Search/AlgoPreprocessor.cpp
    src/Search/LoopOpt.cpp
    src/Search/MemOpt.cpp
    src/Search/Inliner.cpp
    src/Search/CrossFunctionPasses.cpp
    src/Search/EgraphRewriter.cpp
    src/Search/RewriteCache.cpp
    src/Pattern/PatternLibrary.cpp
    src/Pipeline.cpp
)

if(CLUNK_ENABLE_GPU)
    list(APPEND CLUNK_SOURCES
        src/GPU/PTXOptimizer.cpp
        src/GPU/KernelLaunchOpt.cpp
        src/GPU/PTXEmitter.cpp
        src/GPU/ArchLimits.cpp
        src/GPU/CUDADriver.cpp
        src/GPU/LivenessAnalysis.cpp
        src/GPU/DivergenceAnalysis.cpp
        src/GPU/OccupancyModel.cpp
    )
    add_compile_definitions(CLUNK_HAS_GPU=1)
endif()

# ── ncurses (optional, for the --tui live progress view) ────────────────────
# Use CMake's built-in FindCurses module — it handles both pkg-config
# and manual search, and normalises the variables correctly (the
# previous hand-rolled detection used NCURSES_LIBRARY singular, which
# pkg_check_modules doesn't set — it sets NCURSES_LIBRARIES plural —
# so target_link_libraries got an empty string and the link failed).
#
# We set CURSES_NEED_WIDE=TRUE so FindCurses prefers the wide-character
# variant (ncursesw) when available — needed for proper UTF-8 support
# in the TUI chrome, and also avoids the non-wide libcurses.so linker
# script which references libncurses.so.6 (non-wide) that may not exist
# on systems that only ship the wide runtime.
#
# find_package(Curses) sets:
#   CURSES_FOUND          - TRUE if curses was found
#   CURSES_INCLUDE_DIR    - the curses include directory (singular, a path)
#   CURSES_LIBRARIES      - the curses libraries (a list, e.g. "ncursesw;tinfo")
#   CURSES_NCURSES_LIBRARY - the ncurses library path (if found)
set(CURSES_NEED_WIDE TRUE)
find_package(Curses QUIET)
if(CURSES_FOUND)
    option(CLUNK_ENABLE_TUI "Enable the ncurses TUI (--tui flag)" ON)
else()
    set(CLUNK_ENABLE_TUI OFF CACHE BOOL "Enable the ncurses TUI" FORCE)
endif()
if(CLUNK_ENABLE_TUI)
    add_compile_definitions(CLUNK_HAS_TUI=1)
    # Export the ncurses paths for tests/CMakeLists.txt to reuse.
    set(CLUNK_NCURSES_INCLUDE_DIR ${CURSES_INCLUDE_DIR} CACHE INTERNAL "ncurses include dir")
    set(CLUNK_NCURSES_LIBRARIES ${CURSES_LIBRARIES} CACHE INTERNAL "ncurses libraries")
    message(STATUS "ncurses TUI: ENABLED (include=${CURSES_INCLUDE_DIR}, libs=${CURSES_LIBRARIES})")
else()
    message(STATUS "ncurses TUI: DISABLED (ncurses not found)")
endif()

# ── Optional build optimisations ────────────────────────────────────────────
option(CLUNK_ENABLE_LTO "Enable link-time optimization (Release/RelWithDebInfo)" ON)
option(CLUNK_ENABLE_NATIVE "Compile with -march=native (non-portable, off by default)" OFF)
option(CLUNK_ENABLE_GC_SECTIONS "Dead-code stripping via -ffunction-sections -fdata-sections -Wl,--gc-sections" ON)

if(CLUNK_ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    message(STATUS "LTO: ENABLED for Release/RelWithDebInfo")
endif()

if(CLUNK_ENABLE_GC_SECTIONS)
    add_compile_options(-ffunction-sections -fdata-sections)
    add_link_options(-Wl,--gc-sections)
    message(STATUS "Dead-code stripping: ENABLED")
endif()

if(CLUNK_ENABLE_NATIVE)
    add_compile_options(-march=native -mtune=native)
    message(STATUS "Native arch (-march=native): ENABLED")
endif()

add_library(clunk_core STATIC ${CLUNK_SOURCES})
target_include_directories(clunk_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/vendor
)

if(CLUNK_ENABLE_Z3 OR CLUNK_ENABLE_ALIVE2)
    # Both Z3 and Alive2 are loaded at runtime via dlopen — we only need -ldl
    target_link_libraries(clunk_core PUBLIC dl)
endif()

target_compile_options(clunk_core PRIVATE -Wall -Wextra -Wpedantic)

# ── CLI executable ───────────────────────────────────────────────────────────
set(CLUNK_CLI_SOURCES src/cli/main.cpp)
if(CLUNK_ENABLE_TUI)
    list(APPEND CLUNK_CLI_SOURCES src/cli/tui.cpp)
endif()
add_executable(clunk ${CLUNK_CLI_SOURCES})
target_link_libraries(clunk PRIVATE clunk_core)
target_compile_options(clunk PRIVATE -Wall -Wextra -Wpedantic)
if(CLUNK_ENABLE_TUI)
    target_include_directories(clunk PRIVATE ${CLUNK_NCURSES_INCLUDE_DIR})
    target_link_libraries(clunk PRIVATE ${CLUNK_NCURSES_LIBRARIES})
endif()

# ── CMake integration package ───────────────────────────────────────────────
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/ClunkConfig.cmake.in
    ${CMAKE_BINARY_DIR}/ClunkConfig.cmake
    INSTALL_DESTINATION lib/cmake/Clunk
)
write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/ClunkConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# ── Tests ────────────────────────────────────────────────────────────────────
if(CLUNK_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Benchmarks ───────────────────────────────────────────────────────────────
if(CLUNK_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ── Install rules ────────────────────────────────────────────────────────────
install(TARGETS clunk_core clunk
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/clunk DESTINATION include)
install(DIRECTORY vendor/z3 DESTINATION include/clunk/vendor)
install(DIRECTORY vendor/alive2 DESTINATION include/clunk/vendor)
install(FILES
    ${CMAKE_BINARY_DIR}/ClunkConfig.cmake
    ${CMAKE_BINARY_DIR}/ClunkConfigVersion.cmake
    DESTINATION lib/cmake/Clunk
)
install(FILES cmake/ClunkOpt.cmake DESTINATION lib/cmake/Clunk)