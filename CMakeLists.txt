cmake_minimum_required(VERSION 3.15)

project(paynt)

set(CMAKE_CXX_STANDARD 20)

# Configuration options
set(STORM_DIR_HINT "" CACHE STRING "A hint where the Storm library can be found.")
option(ALLOW_STORM_SYSTEM "Allow finding a storm version on the system" ON)
option(ALLOW_STORM_FETCH "Allow fetching storm" ON)
set(STORM_GIT_REPO "" CACHE STRING  "Git repo used for fetching storm")
set(STORM_GIT_TAG "" CACHE STRING "Git repo tag used for fetching storm")
option(PAYNT_INFO_PRETEND_FETCH "[INTERNAL] Use for overriding some flags in the info by cibuildwheel." OFF)
mark_as_advanced(PAYNT_INFO_PRETEND_FETCH)

if (NOT PAYNT_INFO_PRETEND_FETCH)
    # Query stormpy.info to determine Storm configuration
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import stormpy.info; print(stormpy.info.storm_from_system())"
        RESULT_VARIABLE _stormpy_result
        OUTPUT_VARIABLE STORMPY_FROM_SYSTEM
        ERROR_VARIABLE _stormpy_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT _stormpy_result EQUAL 0)
        message(FATAL_ERROR "PAYNT - Failed to query stormpy.info.storm_from_system(): ${_stormpy_error}")
    endif()

    # Configure Storm settings based on stormpy configuration
    if(STORMPY_FROM_SYSTEM STREQUAL "True")
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import stormpy.info; print(stormpy.info.storm_directory())"
            RESULT_VARIABLE _directory_result
            OUTPUT_VARIABLE STORM_DIRECTORY_INFO
            ERROR_VARIABLE _origin_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(NOT _directory_result EQUAL 0)
            message(FATAL_ERROR "PAYNT - Failed to query stormpy.info.storm_directory(): ${_origin_error}")
        endif()

        message(STATUS "PAYNT - StormPy uses system Storm from: ${STORM_DIRECTORY_INFO} - configuring PAYNT to use the same system Storm")

        set(STORM_DIR_HINT "${STORM_DIRECTORY_INFO}")
        set(ALLOW_STORM_SYSTEM ON)
        set(ALLOW_STORM_FETCH OFF)
    else()
        message(STATUS "PAYNT - StormPy uses fetched Storm - querying Storm origin information")

        # Get Storm origin information
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import stormpy.info; repo, tag, hash = stormpy.info.storm_origin_info(); print(f'{repo or \"\"};{tag or \"\"};{hash or \"\"}')"
            RESULT_VARIABLE _origin_result
            OUTPUT_VARIABLE STORM_ORIGIN_INFO
            ERROR_VARIABLE _origin_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(NOT _origin_result EQUAL 0)
            message(FATAL_ERROR "PAYNT - Failed to query stormpy.info.storm_origin_info(): ${_origin_error}")
        endif()
        
        # Parse repo and tag
        # We actually take the repo and commit hash information because we want to match the exact commit of Storm that was used
        # TODO maybe rename it to be less confusing, however be careful how this influences what information is stored in payntbind info
        list(GET STORM_ORIGIN_INFO 0 STORM_ORIGIN_REPO)
        list(GET STORM_ORIGIN_INFO 1 STORM_ORIGIN_TAG)
        
        message(STATUS "PAYNT - Storm origin repo: ${STORM_ORIGIN_REPO}")
        message(STATUS "PAYNT - Storm origin tag: ${STORM_ORIGIN_TAG}")
        
        # Configure to fetch the same Storm version as stormpy
        set(ALLOW_STORM_SYSTEM OFF)
        set(ALLOW_STORM_FETCH ON)
        
        if(NOT STORM_ORIGIN_REPO STREQUAL "")
            set(STORM_GIT_REPO "${STORM_ORIGIN_REPO}")
        else()
            message(FATAL_ERROR "PAYNT - Failed to determine Storm origin repository")
        endif()
        
        if(NOT STORM_ORIGIN_TAG STREQUAL "")
            set(STORM_GIT_TAG "${STORM_ORIGIN_TAG}")
        else()
            message(FATAL_ERROR "PAYNT - Failed to determine Storm origin tag")
        endif()
        
        message(STATUS "PAYNT - Configured to fetch Storm from: ${STORM_GIT_REPO} (tag: ${STORM_GIT_TAG})")
    endif()
endif()

# IMPORTANT: propagate the computed Storm configuration into the CMake cache.
#
# `payntbind/CMakeLists.txt` declares these knobs using `option(...)` and
# `set(... CACHE ...)` defaults. If we don't seed the cache here, those defaults
# can overwrite the values computed above (based on `stormpy.info`), causing
# `payntbind` to FetchContent Storm again during wheel builds.
set(STORM_DIR_HINT "${STORM_DIR_HINT}" CACHE STRING "A hint where the Storm library can be found." FORCE)
set(ALLOW_STORM_SYSTEM "${ALLOW_STORM_SYSTEM}" CACHE BOOL "Allow finding a storm version on the system" FORCE)
set(ALLOW_STORM_FETCH "${ALLOW_STORM_FETCH}" CACHE BOOL "Allow fetching storm" FORCE)
set(STORM_GIT_REPO "${STORM_GIT_REPO}" CACHE STRING "Git repo used for fetching storm" FORCE)
set(STORM_GIT_TAG "${STORM_GIT_TAG}" CACHE STRING "Git repo tag used for fetching storm" FORCE)
set(PAYNT_INFO_PRETEND_FETCH "${PAYNT_INFO_PRETEND_FETCH}" CACHE BOOL " [INTERNAL] Use for overriding some flags in the info by cibuildwheel." FORCE)


add_subdirectory(payntbind)