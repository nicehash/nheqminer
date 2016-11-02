SET(CMAKE_SYSTEM_NAME Windows)
SET(CMAKE_SYSTEM_PROCESSOR x86_64)

# specify the cross compiler
SET(CMAKE_C_COMPILER /usr/bin/x86_64-w64-mingw32-gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/x86_64-w64-mingw32-g++)

# where is the target environment
SET(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32/sys-root/mingw)

# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Make sure Qt can be detected by CMake
SET(QT_BINARY_DIR /usr/x86_64-w64-mingw32/bin /usr/bin)

# set the resource compiler (RHBZ #652435)
SET(CMAKE_RC_COMPILER /usr/bin/x86_64-w64-mingw32-windres)

# These are needed for compiling lapack (RHBZ #753906)
SET(CMAKE_Fortran_COMPILER /usr/bin/x86_64-w64-mingw32-gfortran)
SET(CMAKE_AR:FILEPATH /usr/bin/x86_64-w64-mingw32-ar)
SET(CMAKE_RANLIB:FILEPATH /usr/bin/x86_64-w64-mingw32-ranlib)

