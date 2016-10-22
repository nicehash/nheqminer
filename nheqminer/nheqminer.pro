TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

# remove possible other optimization flags
QMAKE_CFLAGS_RELEASE -= -O
QMAKE_CFLAGS_RELEASE -= -O1
QMAKE_CFLAGS_RELEASE -= -O2

QMAKE_CXXFLAGS_RELEASE -= -O
QMAKE_CXXFLAGS_RELEASE -= -O1
QMAKE_CXXFLAGS_RELEASE -= -O2


# add the desired -O3 if not present
QMAKE_CFLAGS_RELEASE *= -O3
QMAKE_CXXFLAGS_RELEASE *= -O3

#QMAKE_CXXFLAGS_RELEASE += -mnative

#QMAKE_CFLAGS_RELEASE += -mavx2
#QMAKE_CXXFLAGS_RELEASE += -mavx2

#QMAKE_CFLAGS_RELEASE += -mavx
#QMAKE_CXXFLAGS_RELEASE += -mavx

#QMAKE_CFLAGS_RELEASE += -msse2
#QMAKE_CXXFLAGS_RELEASE += -msse2

INCLUDEPATH += $$PWD/3rdparty/

# use this instead of CONFIG c++11 since some qmake versions use experimental flag
QMAKE_CXXFLAGS += -std=gnu++11

# Linux
DEFINES +=  HAVE_DECL_HTOBE16
DEFINES +=  HAVE_DECL_HTOLE16
DEFINES +=  HAVE_DECL_BE16TOH
DEFINES +=  HAVE_DECL_LE16TOH
DEFINES +=  HAVE_DECL_HTOBE32
DEFINES +=  HAVE_DECL_HTOLE32
DEFINES +=  HAVE_DECL_BE32TOH
DEFINES +=  HAVE_DECL_LE32TOH
DEFINES +=  HAVE_DECL_HTOBE64
DEFINES +=  HAVE_DECL_HTOLE64
DEFINES +=  HAVE_DECL_BE64TOH
DEFINES +=  HAVE_DECL_LE64TOH

DEFINES +=  HAVE_BYTESWAP_H
DEFINES +=  HAVE_DECL_BSWAP_16
DEFINES +=  HAVE_DECL_BSWAP_32
DEFINES +=  HAVE_DECL_BSWAP_64

SOURCES += main.cpp \
    compat/strnlen.cpp \
#    crypto/equihash.cpp \
    crypto/ripemd160.cpp \
    crypto/sha256.cpp \
    json/json_spirit_reader.cpp \
    json/json_spirit_value.cpp \
    json/json_spirit_writer.cpp \
    libstratum/StratumClient.cpp \
    libstratum/ZcashStratum.cpp \
    primitives/block.cpp \
    amount.cpp \
    api.cpp \
    arith_uint256.cpp \
    speed.cpp \
    uint256.cpp \
    utilstrencodings.cpp \
#    blake2/blake2b.c \
#    blake2/core.c \
    trompequihash/blake2/blake2bx.cpp

HEADERS += \
    blake2/blake2-impl.h \
    blake2/blake2.h \
    blake2/blamka-round-opt.h \
    blake2/blamka-round-ref.h \
    compat/byteswap.h \
    compat/endian.h \
    compat/sanity.h \
    crypto/common.h \
    crypto/equihash.h \
    crypto/ripemd160.h \
    crypto/sha256.h \
    json/json_spirit.h \
    json/json_spirit_error_position.h \
    json/json_spirit_reader.h \
    json/json_spirit_reader_template.h \
    json/json_spirit_stream_reader.h \
    json/json_spirit_utils.h \
    json/json_spirit_value.h \
    json/json_spirit_writer.h \
    json/json_spirit_writer_template.h \
    libstratum/StratumClient.h \
    libstratum/ZcashStratum.h \
    primitives/block.h \
    amount.h \
    api.hpp \
    arith_uint256.h \
    hash.h \
    serialize.h \
    speed.hpp \
    streams.h \
    tinyformat.h \
    uint252.h \
    uint256.h \
    utilstrencodings.h \
    version.h \
    trompequihash/blake2/blake2-config.h \
    trompequihash/blake2/blake2-impl.h \
    trompequihash/blake2/blake2-round.h \
    trompequihash/blake2/blake2.h \
    trompequihash/blake2/blake2b-load-sse2.h \
    trompequihash/blake2/blake2b-load-sse41.h \
    trompequihash/blake2/blake2b-round.h \
    trompequihash/equi.h \
    trompequihash/equi_miner.h \
    trompequihash/equi_miner2.h \
    trompequihash/pthreads/pthread.h \
    amount.h \
    api.hpp \
    arith_uint256.h \
    blake2/blake2.h \
    compat/byteswap.h \
    compat/endian.h \
    crypto/common.h \
    crypto/equihash.h \
    crypto/sha256.h \
    hash.h \
    json/json_spirit.h \
    json/json_spirit_error_position.h \
    json/json_spirit_reader.h \
    json/json_spirit_reader_template.h \
    json/json_spirit_stream_reader.h \
    json/json_spirit_utils.h \
    json/json_spirit_value.h \
    json/json_spirit_writer.h \
    json/json_spirit_writer_template.h \
    libstratum/StratumClient.h \
    libstratum/ZcashStratum.h \
    primitives/block.h \
    primitives/transaction.h \
    script/script.h \
    serialize.h \
    speed.hpp \
    streams.h \
    support/allocators/zeroafterfree.h \
    tinyformat.h \
    uint252.h \
    uint256.h \
    utilstrencodings.h \
    version.h \
    zcash/JoinSplit.hpp \
    zcash/NoteEncryption.hpp \
    zcash/Proof.hpp \
    zcash/Zcash.h \


LIBS += -pthread

LIBS += $$PWD/libs/linux_ubuntu/libboost_log.a
LIBS += $$PWD/libs/linux_ubuntu/libboost_system.a
LIBS += $$PWD/libs/linux_ubuntu/libboost_thread.a
