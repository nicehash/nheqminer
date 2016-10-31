format elf64
public EhPrepare as 'EhPrepareAVX1'
public EhSolver as 'EhSolverAVX1'
public testinput as 'testinputAVX1'

include "struct.inc"
include "params.inc"
include "struct_eh.inc"
include "macro_eh.asm"

section '.text' executable align 64
include "proc_ehprepare_avx1.asm"
include "proc_ehsolver_avx1.asm"

section '.data' writeable align 64
include "data_blake2b.asm"
testinput file "t2.bin"
