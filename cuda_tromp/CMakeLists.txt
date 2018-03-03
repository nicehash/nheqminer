set(EXECUTABLE cuda_tromp)

option(ENABLE_CUDA "Enable the cuda build" ON)

# depending on gcc version
# ;-std=c++11 => Ubuntu 14.04 check gcc versions
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)

file(GLOB SRC_LIST
    cuda_tromp.cpp
    equi_miner.cu )
file(GLOB HEADERS
    cuda_tromp.hpp
    eqcuda.hpp
    )


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


include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(..)
CUDA_ADD_LIBRARY(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
TARGET_LINK_LIBRARIES(${EXECUTABLE} ${CUDA_LIBRARIES})

message("-- CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
