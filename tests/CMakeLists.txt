# Each test file has its own main(), so build as separate executables

set(TEST_SOURCES
    test_ir
    test_parser
    test_evaluator
    test_search
    test_search_threadpool
    test_stoke_moves
    test_peephole_miner
    test_peephole_harvest
    test_peephole_cegis
    test_pattern
    test_egraph
    test_egraph_validity
    test_differential_screen
    test_liveout
    test_demanded_bits
    test_pipeline
    test_smt
    test_smt_soundness
    test_smt_cegis
    test_alive_verifier
    test_rewrite_cache
    test_cost_model_tti
    test_vector_synth
    test_loop_opt
    test_mem_opt
    test_align_opt
    test_fp_opt
    test_mca
    test_known_bits
    test_dataflow_prune
    test_powers_of_two
    test_strength_reduce
    test_cross_function
    test_smt_bounded_unrolling
    test_opt_level_max
    test_hole_synth
    test_block_opt
    test_series_expand
    test_algo_preprocessor
    test_tui
)

if(CLUNK_ENABLE_GPU)
    list(APPEND TEST_SOURCES
        test_gpu_emitter
        test_gpu_occupancy
        test_gpu_divergence
    )
endif()

foreach(test_name ${TEST_SOURCES})
    add_executable(clunk_test_${test_name} ${test_name}.cpp)
    target_link_libraries(clunk_test_${test_name} PRIVATE clunk_core)
    target_compile_options(clunk_test_${test_name} PRIVATE -Wall -Wextra)
    # The TUI test needs ncurses (linked via clunk_core's PUBLIC deps if
    # the TUI is enabled — but tui.cpp is in the clunk CLI exe, not the
    # core lib, so we link ncurses directly here when the TUI is on).
    if(CLUNK_ENABLE_TUI AND test_name STREQUAL "test_tui")
        target_sources(clunk_test_${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/cli/tui.cpp)
        target_include_directories(clunk_test_${test_name} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CLUNK_NCURSES_INCLUDE_DIR})
        target_link_libraries(clunk_test_${test_name} PRIVATE ${CLUNK_NCURSES_LIBRARIES})
    endif()
    add_test(NAME ${test_name} COMMAND clunk_test_${test_name})
endforeach()