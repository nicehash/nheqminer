struct EH
hashtab rd 4096
bucket0ptr rw 4*256
bucket1ptr rw 4*256
workingpairs rd 16384
rd 4096
debugptr rq 1
debug rq 31

mids rq 32

basemap rd 256*11584
pairs rd 9*256*11584
buf rd 5*256*11584
ends

;debug, in qword
;0	rdtsc entry
;1	rdtsc done blake2b fill
;2	rdtsc done Stage 1
;3	number of pairs

;18	rdtsc done Stage 9
;19	number of pairs
;20	rdtsc done GetSolutions