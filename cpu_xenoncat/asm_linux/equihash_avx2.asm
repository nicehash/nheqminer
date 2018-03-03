format elf64
public EhPrepare as 'EhPrepareAVX2'
public EhSolver as 'EhSolverAVX2'

include "struct.inc"
include "params.inc"
include "struct_eh.inc"
include "macro_eh.asm"

section '.text' executable align 64
include "proc_ehprepare_avx2.asm"
include "proc_ehsolver_avx2.asm"

section '.data' writeable align 64
include "data_blake2b.asm"
