cmake_minimum_required(VERSION 3.16)

if(USE_GPU)
    project(FFW2D CXX C CUDA)
else()
    project(FFW2D CXX C)
endif()

# =====================================================================
#
# =====================================================================
option(USE_GPU "Build with GPU (CUDA) support using Kokkos" OFF)
option(USE_MPI "Build with MPI support on Linux" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")


if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:--extended-lambda>)
endif()


if(USE_GPU)
    message(STATUS "====== GPU (CUDA) ======")

    if(WIN32)
        set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -DKOKKOS_ENABLE_CUDA")

    else()
        if(NOT Kokkos_ROOT)
            set(Kokkos_ROOT "/usr/local/kokkos" CACHE PATH "Path to Kokkos")
        endif()
        set(CMAKE_CXX_COMPILER "${Kokkos_ROOT}/bin/nvcc_wrapper" CACHE STRING "CXX compiler" FORCE)

    endif()
else()
    message(STATUS "====== CPU  ======")

    if(USE_MPI AND CMAKE_SYSTEM_NAME MATCHES "Linux")
        message(STATUS "-->  use MPI ")
        set(CMAKE_CXX_COMPILER "mpicxx" CACHE STRING "CXX compiler" FORCE)
        set(CMAKE_C_COMPILER "mpicc" CACHE STRING "C compiler" FORCE)
    endif()

    # 强行注入你手写成功的 -fopenmp 参数，防止 Linux/Windows 下 CMake 犯傻
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -O3")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
endif()

# =====================================================================
#
# =====================================================================
find_package(Kokkos REQUIRED)

if(USE_MPI AND CMAKE_SYSTEM_NAME MATCHES "Linux")
    find_package(MPI REQUIRED)
endif()

# =====================================================================
#
# =====================================================================
add_executable(ffw2d main.cpp)
target_link_libraries(ffw2d PRIVATE Kokkos::kokkos)
if(NOT USE_GPU)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_libraries(ffw2d PRIVATE m dl)

        if(USE_MPI)
            target_link_libraries(ffw2d PRIVATE MPI::MPI_CXX)
        endif()

    elseif(WIN32)
        target_link_libraries(ffw2d PRIVATE m)
        target_link_options(ffw2d PRIVATE -static -static-libgcc -static-libstdc++)
        find_library(STATIC_GOMP_LIB NAMES libgomp.a PATHS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
        if(STATIC_GOMP_LIB)
            target_link_libraries(ffw2d PRIVATE ${STATIC_GOMP_LIB})
        else()
            target_link_options(ffw2d PRIVATE -Wl,-Bstatic -lgomp -Wl,-Bdynamic)
        endif()
        target_link_options(ffw2d PRIVATE -static -lpthread -Wl,--stack,134217728)
    endif()
endif()