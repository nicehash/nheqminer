format elf64
public Blake2PrepareMidstate2
public Blake2Run2

section '.text' executable align 64
include "proc_prepmidstate_avx1.asm"
align 16
include "proc_blake2_avx1.asm"

section '.data' writeable align 64
include "data_blake2b.asm"
