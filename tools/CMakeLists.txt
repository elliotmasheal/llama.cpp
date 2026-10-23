# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(llama-bench)
    add_subdirectory(mtmd)
    # the quality gate, see bench/THREE-TIER.md F6
    add_subdirectory(perplexity)
endif()
