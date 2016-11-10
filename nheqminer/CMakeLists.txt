#cmake global
cmake_minimum_required(VERSION 2.8)

option(STATIC_BUILD "Build with static libraries on Linux")
option(XENON "Build options with xenoncat")
option(MARCH "GCC options for architecture (default: native)")
option(MOPT "GCC options for optimization (default: -O3)")
option(NONINTEL "Set true if running on non-intel architectures")


project(nheqminer)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# LINUX
if(CMAKE_COMPILER_IS_GNUCXX)
	if (MARCH)
		set (GCCMARCH "${MARCH}")
	else()
		set (GCCMARCH "-march=native")
	endif()

    # use native cpu features
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCCMARCH}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCCMARCH}")

	# optimizations
	if (MOPT)
		add_definitions("${MOPT}")
	else()
		add_definitions("-O3")
	endif()
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
	if (XENON)
		if (XENON EQUAL 0)
			unset (XENONCAT)
		else()
			set (XENONCAT "${nheqminer_SOURCE_DIR}/external/equihash-xenon/Linux/asm/equihash_avx1.o" "${nheqminer_SOURCE_DIR}/external/equihash-xenon/Linux/asm/equihash_avx2.o")
			add_definitions(-DXENONCAT=1)
		endif()
	endif()
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
	include_directories(${nheqminer_SOURCE_DIR} ${nheqminer_SOURCE_DIR}/compat)
	if (XENON)
		if (XENON EQUAL 0)
			unset (XENONCAT)
		else()
			set (XENONCAT "${nheqminer_SOURCE_DIR}/external/equihash-xenon/Mac/equihash_avx1.o" "${nheqminer_SOURCE_DIR}/external/equihash-xenon/Mac/equihash_avx2.o")
			add_definitions(-DXENONCAT=1)
		endif()
	endif()

	# optimizations
	if (MOPT)
		add_definitions("${MOPT}")
	else()
		add_definitions("-O3")
	endif()
else()
	# Common
	include_directories(${nheqminer_SOURCE_DIR})
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
	# include_directories(${nheqminer_SOURCE_DIR} ${nheqminer_SOURCE_DIR}/compat /usr/x86_64-w64-mingw32/sys-root/mingw/include/)
	include_directories(${nheqminer_SOURCE_DIR} ${nheqminer_SOURCE_DIR}/compat)
	set (WINLIBS wsock32 ws2_32)
	if (XENON)
		if (XENON EQUAL 0)
			unset (XENONCAT)
		else()
			# Not a typo: when Mingw is included, we can use Linux version of object files
			set (XENONCAT "${nheqminer_SOURCE_DIR}/external/equihash-xenon/Linux/asm/equihash_avx1.o" "${nheqminer_SOURCE_DIR}/external/equihash-xenon/Linux/asm/equihash_avx2.o")
			add_definitions(-DXENONCAT=1)
			set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fpermissive")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
		endif()
	endif()
endif()

if (NONINTEL)
	add_definitions("-D__NONINTEL__")
	set (TROMPBLAKE "trompequihash/blake2-ref")
	set (TROMPBLAKESOURCES "trompequihash/blake2-ref/blake2b-ref.c")
	set (TROMPBLAKEHEADERS "${TROMPBLAKE}/blake2-impl.h" "${TROMPBLAKE}/blake2.h")
else()
	set (TROMPBLAKE "trompequihash/blake2")
	set (TROMPBLAKESOURCES "${TROMPBLAKE}/blake2bx.cpp")
	set (TROMPBLAKEHEADERS "${TROMPBLAKE}/blake2-config.h" "${TROMPBLAKE}/blake2-impl.h" "${TROMPBLAKE}/blake2-round.h" "${TROMPBLAKE}/blake2.h" "${TROMPBLAKE}/blake2b-load-sse2.h" "${TROMPBLAKE}/blake2b-load-sse41.h" "${TROMPBLAKE}/blake2b-round.h")
endif()

if (STATIC_BUILD)
	set(Boost_USE_STATIC_LIBS   ON)
else()
	add_definitions(-DBOOST_ALL_NO_LIB -DBOOST_ALL_DYN_LINK -DBOOST_LOG_DYN_LINK)
endif()

find_package(Threads REQUIRED COMPONENTS)
find_package(Boost REQUIRED COMPONENTS system log_setup log date_time filesystem thread)

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




set(SOURCE_FILES
# SOURCES
    main.cpp
    amount.cpp
    api.cpp
    arith_uint256.cpp
    # don't use this on Linux (Linux has it by default)
#    compat/strnlen.cpp
    crypto/ripemd160.cpp
    crypto/sha256.cpp
    json/json_spirit_reader.cpp
    json/json_spirit_value.cpp
    json/json_spirit_writer.cpp
    libstratum/StratumClient.cpp
    libstratum/ZcashStratum.cpp
    primitives/block.cpp
    speed.cpp
    ${TROMPBLAKESOURCES}
#    trompequihash/blake2/blake2bx.cpp
#    trompequihash/blake2-ref/blake2xb-ref.c
#    trompequihash/blake2-ref/blake2b-ref.c
    uint256.cpp
    utilstrencodings.cpp

# HEADERS
    amount.h
    api.hpp
    arith_uint256.h
    # don't use this on Linux (Linux has it by default)
#    compat/byteswap.h
#    compat/endian.h
##    compat/sanity.h
    crypto/common.h
    crypto/equihash.h
    crypto/ripemd160.h
    crypto/sha256.h
    hash.h
    json/json_spirit.h
    json/json_spirit_error_position.h
    json/json_spirit_reader.h
    json/json_spirit_reader_template.h
    json/json_spirit_stream_reader.h
    json/json_spirit_utils.h
    json/json_spirit_value.h
    json/json_spirit_writer.h
    json/json_spirit_writer_template.h
    libstratum/StratumClient.h
    libstratum/ZcashStratum.h
    primitives/block.h
    primitives/transaction.h
    script/script.h
    serialize.h
    speed.hpp
    streams.h
    support/allocators/zeroafterfree.h
    tinyformat.h
    ${TROMPBLAKEHEADERS}
    trompequihash/equi.h
    trompequihash/equi_miner.h
    trompequihash/equi_miner2.h
    trompequihash/pthreads/pthread.h
    uint252.h
    uint256.h
    utilstrencodings.h
    version.h
    zcash/JoinSplit.hpp
    zcash/NoteEncryption.hpp
    zcash/Proof.hpp
    zcash/Zcash.h
    )
#add_executable(${PROJECT_NAME} ${SRC_LIST})

if (STATIC_BUILD)
	SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
	SET(BUILD_SHARED_LIBRARIES OFF)
	if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
		message("Mac OS X does not support full static build, building only what we can")
	endif()
	if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
		set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static")
	endif()
	if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
		set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static")
	endif()
endif()


set(LIBS ${LIBS} ${Threads_LIBRARIES} ${Boost_LIBRARIES})

get_directory_property( DirDefs COMPILE_DEFINITIONS )

message("-- CXXFLAGS: ${CMAKE_CXX_FLAGS}")
message("-- LIBS: ${LIBS}")
message("-- XENON: ${XENON}")
message("-- XENONCAT: ${XENONCAT}")
message( "COMPILE_DEFINITIONS = ${DirDefs}" )

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
#target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} ${LIBS} ${CMAKE_THREAD_LIBS_INIT} ${nheqminer_SOURCE_DIR}/external/equihash-xenon/Linux/asm/equihash_avx2.o)
# target_link_libraries(${PROJECT_NAME} ${LIBS} ${CMAKE_THREAD_LIBS_INIT} "${XENONCAT}" "${WINLIBS}")
target_link_libraries(${PROJECT_NAME} ${LIBS} ${CMAKE_THREAD_LIBS_INIT} ${XENONCAT} ${WINLIBS})
if (STATIC_BUILD)
	# set_target_properties(${PROJECT_NAME} PROPERTIES LINK_SEARCH_START_STATIC 1)
	set_target_properties(${PROJECT_NAME} PROPERTIES LINK_SEARCH_END_STATIC 1)
endif()
