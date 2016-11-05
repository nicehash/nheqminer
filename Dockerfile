FROM alpine

RUN apk add --no-cache --virtual=build-dependencies git cmake make gcc g++ libc-dev boost-dev && \
    git clone --recursive https://github.com/kost/nheqminer.git && \
    cd /nheqminer/nheqminer && \
    mkdir build && \
    cd /nheqminer/nheqminer/build && \
    cmake -DSTATIC_BUILD=1 -DXENON=2 -DMARCH="-m64" .. && \
    make && \
    apk del --purge build-dependencies
    
ENTRYPOINT ["/nheqminer/nheqminer/build/nheqminer"]
