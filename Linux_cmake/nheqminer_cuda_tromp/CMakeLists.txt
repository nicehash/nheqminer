project(nheqminer_cuda_tromp)
cmake_minimum_required(VERSION 2.8)

option(ENABLE_CUDA "Enable the cuda build" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# LINUX
if(CMAKE_COMPILER_IS_GNUCXX)
    # use native cpu features
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    # optimizations
    add_definitions(-O3)
endif()

# Common
include_directories(${nheqminer_SOURCE_DIR})

add_definitions(-DBOOST_ALL_NO_LIB -DBOOST_ALL_DYN_LINK -DBOOST_LOG_DYN_LINK)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-m64;--std=c++11;--disable-warnings;--ptxas-options=-v;-use_fast_math;-lineinfo)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--disable-warnings;--ptxas-options=-v;-use_fast_math;-lineinfo)

add_definitions(-DHIST)
#add_definitions(-DXINTREE)
#add_definitions(-DUNROLL)

list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)


FIND_PACKAGE(CUDA REQUIRED)
if(COMPUTE AND (COMPUTE GREATER 0))
        LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_${COMPUTE},code=sm_${COMPUTE})
else(COMPUTE AND (COMPUTE GREATER 0))
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};; -gencode arch=compute_20,code=sm_21; -gencode arch=compute_30,code=sm_30; -gencode arch=compute_35,code=sm_35; -gencode arch=compute_50,code=sm_50; -gencode arch=compute_52,code=sm_52; -gencode arch=compute_61,code=sm_61 )
endif(COMPUTE AND (COMPUTE GREATER 0))

include_directories(${CUDA_INCLUDE_DIRS})

find_package(Threads REQUIRED COMPONENTS)
find_package(Boost REQUIRED COMPONENTS system log_setup log date_time filesystem thread)

if(CUDA_FOUND)
message("CUDA FOUND")
else()
message("CUDA NOT FOUND")
endif()

if (Boost_FOUND)
  # From the offical documentation:
  # Add include directories to the build. [...] If the SYSTEM option is given,
  # the compiler will be told the directories are meant as system include
  # directories on some platforms (signalling this setting might achieve effects
  # such as the compiler skipping warnings [...])."
  include_directories (SYSTEM ${Boost_INCLUDE_DIR})

  # From the offical documentation:
  # "Specify directories in which the linker will look for libraries. [...] Note
  # that this command is rarely necessary. Library locations returned by
  # find_package() and find_library() are absolute paths. Pass these absolute
  # library file paths directly to the target_link_libraries() command. CMake
  # will ensure the linker finds them."
  link_directories (${Boost_LIBRARY_DIRS})
else()
    message("Boost_FOUND NOT FOUND")
endif ()

## Add solvers here
#add_definitions(-DUSE_CPU_XENONCAT)
#add_definitions(-DUSE_CPU_TROMP)
add_definitions(-DUSE_CUDA_TROMP)

#add_library ( xenoncat_avx1 SHARED IMPORTED GLOBAL )
#set_target_properties ( xenoncat_avx1 PROPERTIES IMPORTED_LOCATION "../../cpu_xenoncat/Linux/asm/equihash_avx1.o" )

#add_library ( xenoncat_avx2 SHARED IMPORTED GLOBAL )
#set_target_properties ( xenoncat_avx2 PROPERTIES IMPORTED_LOCATION "../../cpu_xenoncat/Linux/asm/equihash_avx2.o" )

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../../nheqminer/)

set(SOURCE_FILES
    # sources
    ../../nheqminer/amount.cpp
    ../../nheqminer/api.cpp
    ../../nheqminer/arith_uint256.cpp
    ../../nheqminer/crypto/sha256.cpp
    ../../nheqminer/json/json_spirit_reader.cpp
    ../../nheqminer/json/json_spirit_value.cpp
    ../../nheqminer/json/json_spirit_writer.cpp
    ../../nheqminer/libstratum/ZcashStratum.cpp
    ../../nheqminer/main.cpp
    ../../nheqminer/primitives/block.cpp
    ../../nheqminer/speed.cpp
    ../../nheqminer/uint256.cpp
    ../../nheqminer/utilstrencodings.cpp
    # headers
    ../../nheqminer/amount.h
    ../../nheqminer/api.hpp
    ../../nheqminer/arith_uint256.h
    ../../nheqminer/crypto/sha256.h
    ../../nheqminer/hash.h
    ../../nheqminer/json/json_spirit.h
    ../../nheqminer/json/json_spirit_error_position.h
    ../../nheqminer/json/json_spirit_reader.h
    ../../nheqminer/json/json_spirit_reader_template.h
    ../../nheqminer/json/json_spirit_stream_reader.h
    ../../nheqminer/json/json_spirit_utils.h
    ../../nheqminer/json/json_spirit_value.h
    ../../nheqminer/json/json_spirit_writer.h
    ../../nheqminer/json/json_spirit_writer_template.h
    ../../nheqminer/libstratum/StratumClient.cpp
    ../../nheqminer/libstratum/StratumClient.h
    ../../nheqminer/libstratum/ZcashStratum.cpp
    ../../nheqminer/libstratum/ZcashStratum.h
    ../../nheqminer/primitives/block.h
    ../../nheqminer/primitives/transaction.h
    ../../nheqminer/script/script.h
    ../../nheqminer/serialize.h
    ../../nheqminer/speed.hpp
    ../../nheqminer/streams.h
    ../../nheqminer/support/allocators/zeroafterfree.h
    ../../nheqminer/tinyformat.h
    ../../nheqminer/uint252.h
    ../../nheqminer/uint256.h
    ../../nheqminer/utilstrencodings.h
    ../../nheqminer/version.h
    ../../nheqminer/zcash/JoinSplit.hpp
    ../../nheqminer/zcash/NoteEncryption.hpp
    ../../nheqminer/zcash/Proof.hpp
    ../../nheqminer/zcash/Zcash.h
    ../../nheqminer/SolverStub.h # just a stub

#    # cpu tromp
#    ../../cpu_tromp/blake2/blake2bx.cpp
#    ../../cpu_tromp/cpu_tromp.cpp
#    ../../cpu_tromp/blake2/blake2-config.h
#    ../../cpu_tromp/blake2/blake2-impl.h
#    ../../cpu_tromp/blake2/blake2-round.h
#    ../../cpu_tromp/blake2/blake2.h
#    ../../cpu_tromp/blake2/blake2b-load-sse2.h
#    ../../cpu_tromp/blake2/blake2b-load-sse41.h
#    ../../cpu_tromp/blake2/blake2b-round.h
#    ../../cpu_tromp/cpu_tromp.hpp
#    ../../cpu_tromp/equi.h
#    ../../cpu_tromp/equi_miner.h

#    # cpu xenocat
#    ../../cpu_xenoncat/cpu_xenoncat.hpp
#    ../../cpu_xenoncat/xenoncat.cpp

    # cuda tromp
    ../../cuda_tromp/cuda_tromp.hpp
    ../../cuda_tromp/cuda_tromp.cpp
    ../../cuda_tromp/eqcuda.hpp
    ../../cuda_tromp/equi_miner.cu
    ../../cpu_tromp/blake2/blake2bx.cpp
    )

#add_executable(${PROJECT_NAME} ${SRC_LIST})
set(LIBS ${LIBS} ${Threads_LIBRARIES} ${Boost_LIBRARIES})

#message("-- CXXFLAGS: ${CMAKE_CXX_FLAGS}")
#message("-- LIBS: ${LIBS}")

#add_executable(${PROJECT_NAME} ${SOURCE_FILES})
#target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${LIBS} ${CUDA_LIBRARIES} )
