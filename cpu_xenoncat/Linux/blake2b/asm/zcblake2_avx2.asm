format elf64
public Blake2PrepareMidstate4
public Blake2Run4

section '.text' executable align 64
include "proc_prepmidstate_avx2.asm"
align 16
include "proc_blake2_avx2.asm"

section '.data' writeable align 64
include "data_blake2b.asm"
