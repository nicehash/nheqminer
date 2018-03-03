include "macro_blake2b_avx1.asm"

macro RecordRdtsc
{
rdtsc
mov rcx, [rbp+EH.debugptr]
mov [rcx], eax
mov [rcx+4], edx
add rcx, 8
mov [rbp+EH.debugptr], rcx
}

macro Record reg1
{
mov rcx, [rbp+EH.debugptr]
mov [rcx], reg1
add rcx, 8
mov [rbp+EH.debugptr], rcx
}

EhSolver:
push r15
push r14
push r13
push r12
push rbp
push rbx
mov rax, rsp
sub rsp, 0x28
and rsp, -32
mov [rsp+0x20], rax

mov rbp, rdi
mov esi, esi
mov [rbp+EH.mids+0xd0], rsi
mov [rbp+EH.mids+0xd8], rsi
mov dword [rbp+EH.mids+0xdc], 1

lea rcx, [rbp+EH.debug]
rdtsc
mov [rcx], eax
mov [rcx+4], edx
add rcx, 8
mov [rbp+EH.debugptr], rcx

lea rdi, [rbp+EH.bucket1ptr]
mov r12, rdi
mov ecx, 256
xor eax, eax
rep stosq

lea rsi, [rbp+EH.pairs+STATE0_DEST]
xor ecx, ecx
lea rbx, [rbp+EH.basemap]

_LoopBlake2b:
Blake2beq2of2 rbp+EH.mids, rbp+EH.mids+0xc0
vmovdqa xmm7, xword [rbp+EH.mids+0xd0]
vpaddd xmm7, xmm7, xword [xctrinc]
vmovdqa xword [rbp+EH.mids+0xd0], xmm7
vmovdqa xmm7, xword [xshufb_bswap8]
vpshufb xmm0, xmm0, xmm7
vpshufb xmm1, xmm1, xmm7
vpshufb xmm2, xmm2, xmm7
vpshufb xmm3, xmm3, xmm7
vpshufb xmm4, xmm4, xmm7
vpshufb xmm5, xmm5, xmm7
vpshufb xmm6, xmm6, xmm7

vpsrlq xmm8, xmm0, 56
vpsllq xmm0, xmm0, 8
;vpsrlq xmm0, xmm0, 0
vpunpcklqdq xmm7, xmm2, xmm1
vpunpckhqdq xmm2, xmm2, xmm1
vpsrlq xmm1, xmm1, 56
vpor xmm9, xmm0, xmm1		;xmm9=bucket data 2,0
vpshufd xmm1, xmm3, 0x4e

vpextrw r8d, xmm8, 0
movzx eax, word [r12+r8*2]
cmp eax, 10240
jae _SkipA0
imul edx, eax, 24
imul r10d, eax, 1024
add eax, 1
mov word [r12+r8*2], ax
lea r11, [rbx+r8*4]
mov [r10+r11], ecx
imul r8d, r8d, PARTS*ITEMS*STATE0_BYTES
lea rdi, [rsi+r8]
add rdi, rdx
vpalignr xmm10, xmm7, xmm1, 15	;xmm10=xor data 0
vmovq rax, xmm10
movnti [rdi], rax
vpextrq rax, xmm10, 1
movnti [rdi+8], rax
vmovq rax, xmm9
movnti [rdi+16], rax
_SkipA0:

vpextrw r8d, xmm8, 4
movzx eax, word [r12+r8*2]
add ecx, 2
cmp eax, 10240
jae _SkipA2
imul edx, eax, 24
imul r10d, eax, 1024
add eax, 1
mov word [r12+r8*2], ax
lea r11, [rbx+r8*4]
mov [r10+r11], ecx
imul r8d, r8d, PARTS*ITEMS*STATE0_BYTES
lea rdi, [rsi+r8]
add rdi, rdx
vpalignr xmm11, xmm2, xmm3, 15	;xmm11=xor data 2
vmovq rax, xmm11
movnti [rdi], rax
vpextrq rax, xmm11, 1
movnti [rdi+8], rax
vpextrq rax, xmm9, 1
movnti [rdi+16], rax
_SkipA2:

vpsllq xmm8, xmm3, 8
vpsrlq xmm8, xmm8, 56
vpunpcklqdq xmm2, xmm4, xmm3
vpunpckhqdq xmm3, xmm4, xmm3
vpunpcklqdq xmm4, xmm6, xmm5
vpunpckhqdq xmm5, xmm6, xmm5
vpsrldq xmm0, xmm2, 6
vpsrldq xmm1, xmm3, 6
;vmovdqa xmm7, xword [xqmask64bit]
;vpand xmm0, xmm0, xmm7	;bucket data 1
;vpand xmm1, xmm1, xmm7	;bucket data 3

vpextrw r8d, xmm8, 0
movzx eax, word [r12+r8*2]
sub ecx, 1
cmp eax, 10240
jae _SkipA1
imul edx, eax, 24
imul r10d, eax, 1024
add eax, 1
mov word [r12+r8*2], ax
lea r11, [rbx+r8*4]
mov [r10+r11], ecx
imul r8d, r8d, PARTS*ITEMS*STATE0_BYTES
lea rdi, [rsi+r8]
add rdi, rdx
vpalignr xmm2, xmm2, xmm4, 6	;xor data 1
vmovq rax, xmm2
movnti [rdi], rax
vpextrq rax, xmm2, 1
movnti [rdi+8], rax
vmovq rax, xmm0
movnti [rdi+16], rax
_SkipA1:

vpextrw r8d, xmm8, 4
movzx eax, word [r12+r8*2]
add ecx, 2
cmp eax, 10240
jae _SkipA3
imul edx, eax, 24
imul r10d, eax, 1024
add eax, 1
mov word [r12+r8*2], ax
lea r11, [rbx+r8*4]
mov [r10+r11], ecx
imul r8d, r8d, PARTS*ITEMS*STATE0_BYTES
lea rdi, [rsi+r8]
add rdi, rdx
vpalignr xmm3, xmm3, xmm5, 6	;xor data 3
vmovq rax, xmm3
movnti [rdi], rax
vpextrq rax, xmm3, 1
movnti [rdi+8], rax
vmovq rax, xmm1
movnti [rdi+16], rax
_SkipA3:

add ecx, 1
test ecx, 0x0007ffff
jnz _LoopBlake2b
add r12, BUCKETS*2
add rsi, ITEMS*STATE0_BYTES
add rbx, ITEMS*BUCKETS*4
cmp ecx, 2097152
jb _LoopBlake2b

RecordRdtsc

;Stage1
	mov dword [rsp], 0
lea rdi, [rbp+EH.bucket0ptr]
mov r13, rdi
mov ecx, BUCKETS
xor eax, eax
rep stosq
xor r12d, r12d	;bucket multiplied by 2
lea r14, [rbp+EH.pairs+STATE1_DEST-STATE1_BYTES]
lea r15, [rbp+EH.pairs-BUCKETS*4]
_EhStage1:
lea rdi, [rbp+EH.hashtab]
mov ecx, HASHTAB_ENTRIES/2
xor eax, eax
rep stosq
xor r11d, r11d
xor esi, esi
_EhStage1inner:
mov r8d, r11d
shl r8d, 9	;log2 (BUCKET*2)
or r8d, r12d
movzx r8d, word [rbp+r8+EH.bucket1ptr]
mov r10d, STATE0_BYTES
imul edx, r11d, ITEMS*0x10000
lea eax, [r11+r12*2]	;PARTS/2
imul r9d, eax, ITEMS*STATE0_BYTES
lea r9, [rbp+r9+EH.pairs+STATE0_DEST+STATE0_OFFSET]
call _ProcEhMakeLinksShr4
add r11d, 1
cmp r11d, PARTS
jb _EhStage1inner
	add [rsp], esi
	imul r10d, r12d, PARTS*ITEMS*STATE0_BYTES/2
	lea r10, [rbp+r10+EH.pairs+STATE0_DEST]
	lea r8, [rbp+EH.workingpairs]
	EhXor1_3 1
add r12d, 2
test r12d, 0x7e
jnz _EhStage1
add r13, BUCKETS*2
add r14, ITEMS*STATE1_BYTES
add r15, ITEMS*BUCKETS*4
cmp r12d, BUCKETS*2
jb _EhStage1
RecordRdtsc
mov eax, [rsp]
Record rax

;Stage2
	mov dword [rsp], 0
lea rdi, [rbp+EH.bucket1ptr]
mov r13, rdi
mov ecx, BUCKETS
xor eax, eax
rep stosq
xor r12d, r12d	;bucket multiplied by 2
lea r14, [rbp+EH.pairs+STATE2_DEST-STATE2_BYTES]
lea r15, [rbp+EH.pairs+1*BLOCKUNIT-BUCKETS*4]
_EhStage2:
lea rdi, [rbp+EH.hashtab]
mov ecx, HASHTAB_ENTRIES/2
xor eax, eax
rep stosq
xor r11d, r11d
xor esi, esi
_EhStage2inner:
mov r8d, r11d
shl r8d, 9	;log2 (BUCKET*2)
or r8d, r12d
movzx r8d, word [rbp+r8+EH.bucket0ptr]
mov r10d, STATE1_BYTES
imul edx, r11d, ITEMS*0x10000
lea eax, [r11+r12*2]	;PARTS/2
imul r9d, eax, ITEMS*STATE1_BYTES
lea r9, [rbp+r9+EH.pairs+STATE1_DEST+STATE1_OFFSET]
call _ProcEhMakeLinks
add r11d, 1
cmp r11d, PARTS
jb _EhStage2inner
	add [rsp], esi
	imul r10d, r12d, PARTS*ITEMS*STATE1_BYTES/2
	lea r10, [rbp+r10+EH.pairs+STATE1_DEST]
	lea r8, [rbp+EH.workingpairs]
	EhXor1_3 2
add r12d, 2
test r12d, 0x7e
jnz _EhStage2
add r13, BUCKETS*2
add r14, ITEMS*STATE2_BYTES
add r15, ITEMS*BUCKETS*4
cmp r12d, BUCKETS*2
jb _EhStage2
RecordRdtsc
mov eax, [rsp]
Record rax

;Stage3
	mov dword [rsp], 0
lea rdi, [rbp+EH.bucket0ptr]
mov r13, rdi
mov ecx, BUCKETS
xor eax, eax
rep stosq
xor r12d, r12d	;bucket multiplied by 2
lea r14, [rbp+EH.pairs+STATE3_DEST-STATE3_BYTES]
lea r15, [rbp+EH.pairs+2*BLOCKUNIT-BUCKETS*4]
_EhStage3:
lea rdi, [rbp+EH.hashtab]
mov ecx, HASHTAB_ENTRIES/2
xor eax, eax
rep stosq
xor r11d, r11d
xor esi, esi
_EhStage3inner:
mov r8d, r11d
shl r8d, 9	;log2 (BUCKET*2)
or r8d, r12d
movzx r8d, word [rbp+r8+EH.bucket1ptr]
mov r10d, STATE2_BYTES
imul edx, r11d, ITEMS*0x10000
lea eax, [r11+r12*2]	;PARTS/2
imul r9d, eax, ITEMS*STATE2_BYTES
lea r9, [rbp+r9+EH.pairs+STATE2_DEST+STATE2_OFFSET]
call _ProcEhMakeLinksShr4
add r11d, 1
cmp r11d, PARTS
jb _EhStage3inner
	add [rsp], esi
	imul r10d, r12d, PARTS*ITEMS*STATE2_BYTES/2
	lea r10, [rbp+r10+EH.pairs+STATE2_DEST]
	lea r8, [rbp+EH.workingpairs]
	EhXor1_3 3
add r12d, 2
test r12d, 0x7e
jnz _EhStage3
add r13, BUCKETS*2
add r14, ITEMS*STATE3_BYTES
add r15, ITEMS*BUCKETS*4
cmp r12d, BUCKETS*2
jb _EhStage3
RecordRdtsc
mov eax, [rsp]
Record rax

;Stage4
	mov dword [rsp], 0
lea rdi, [rbp+EH.bucket1ptr]
mov r13, rdi
mov ecx, BUCKETS
xor eax, eax
rep stosq
xor r12d, r12d	;bucket multiplied by 2
lea r14, [rbp+EH.pairs+STATE4_DEST-STATE4_BYTES]
lea r15, [rbp+EH.pairs+3*BLOCKUNIT-BUCKETS*4]
_EhStage4:
lea rdi, [rbp+EH.hashtab]
mov ecx, HASHTAB_ENTRIES/2
xor eax, eax
rep stosq
xor r11d, r11d
xor esi, esi
_EhStage4inner:
mov r8d, r11d
shl r8d, 9	;log2 (BUCKET*2)
or r8d, r12d
movzx r8d, word [rbp+r8+EH.bucket0ptr]
mov r10d, STATE3_BYTES
imul edx, r11d, ITEMS*0x10000
lea eax, [r11+r12*2]	;PARTS/2
imul r9d, eax, ITEMS*STATE3_BYTES
lea r9, [rbp+r9+EH.pairs+STATE3_DEST+STATE3_OFFSET]
call _ProcEhMakeLinks
add r11d, 1
cmp r11d, PARTS
jb _EhStage4inner
	add [rsp], esi
	imul r10d, r12d, PARTS*ITEMS*STATE3_BYTES/2
	lea r10, [rbp+r10+EH.pairs+STATE3_DEST]
	lea r8, [rbp+EH.workingpairs]
	EhXor4
add r12d, 2
test r12d, 0x7e
jnz _EhStage4
add r13, BUCKETS*2
add r14, ITEMS*STATE4_BYTES
add r15, ITEMS*BUCKETS*4
cmp r12d, BUCKETS*2
jb _EhStage4
RecordRdtsc
mov eax, [rsp]
Record rax

;Stage5
	mov dword [rsp], 0
lea rdi, [rbp+EH.bucket0ptr]
mov r13, rdi
mov ecx, BUCKETS
xor eax, eax
rep stosq
xor r12d, r12d	;bucket multiplied by 2
lea r14, [rbp+EH.pairs+STATE5_DEST-STATE5_BYTES]
lea r15, [rbp+EH.pairs+4*BLOCKUNIT-BUCKETS*4]
_EhStage5:
lea rdi, [rbp+EH.hashtab]
mov ecx, HASHTAB_ENTRIES/2
xor eax, eax
rep stosq
xor r11d, r11d
xor esi, esi
_EhStage5inner:
mov r8d, r11d
shl r8d, 9	;log2 (BUCKET*2)
or r8d, r12d
movzx r8d, word [rbp+r8+EH.bucket1ptr]
mov r10d, STATE4_BYTES
imul edx, r11d, ITEMS*0x10000
lea eax, [r11+r12*2]	;PARTS/2
imul r9d, eax, ITEMS*STATE4_BYTES
lea r9, [rbp+r9+EH.pairs+STATE4_DEST+STATE4_OFFSET]
call _ProcEhMakeLinksShr4
add r11d, 1
cmp r11d, PARTS
jb _EhStage5inner
	add [rsp], esi
	imul r10d, r12d, PARTS*ITEMS*STATE4_BYTES/2
	lea r10, [rbp+r10+EH.pairs+STATE4_DEST]
	lea r8, [rbp+EH.workingpairs]
	EhXor5
add r12d, 2
test r12d, 0x7e
jnz _EhStage5
add r13, BUCKETS*2
add r14, ITEMS*STATE5_BYTES
add r15, ITEMS*BUCKETS*4
cmp r12d, BUCKETS*2
jb _EhStage5
RecordRdtsc
mov eax, [rsp]
Record rax

;Stage6
	mov dword [rsp], 0
lea rdi, [rbp+EH.bucket1ptr]
mov r13, rdi
mov ecx, BUCKETS
xor eax, eax
rep stosq
xor r12d, r12d	;bucket multiplied by 2
lea r14, [rbp+EH.pairs+STATE6_DEST-STATE6_BYTES]
lea r15, [rbp+EH.pairs+5*BLOCKUNIT-BUCKETS*4]
_EhStage6:
lea rdi, [rbp+EH.hashtab]
mov ecx, HASHTAB_ENTRIES/2
xor eax, eax
rep stosq
xor r11d, r11d
xor esi, esi
_EhStage6inner:
mov r8d, r11d
shl r8d, 9	;log2 (BUCKET*2)
or r8d, r12d
movzx r8d, word [rbp+r8+EH.bucket0ptr]
mov r10d, STATE5_BYTES
imul edx, r11d, ITEMS*0x10000
lea eax, [r11+r12*2]	;PARTS/2
imul r9d, eax, ITEMS*STATE5_BYTES
lea r9, [rbp+r9+EH.pairs+STATE5_DEST+STATE5_OFFSET]
call _ProcEhMakeLinks
add r11d, 1
cmp r11d, PARTS
jb _EhStage6inner
	add [rsp], esi
	imul r10d, r12d, PARTS*ITEMS*STATE5_BYTES/2
	lea r10, [rbp+r10+EH.pairs+STATE5_DEST]
	lea r8, [rbp+EH.workingpairs]
	EhXor6_7 6
add r12d, 2
test r12d, 0x7e
jnz _EhStage6
add r13, BUCKETS*2
add r14, ITEMS*STATE6_BYTES
add r15, ITEMS*BUCKETS*4
cmp r12d, BUCKETS*2
jb _EhStage6
RecordRdtsc
mov eax, [rsp]
Record rax

;Stage7
	mov dword [rsp], 0
lea rdi, [rbp+EH.bucket0ptr]
mov r13, rdi
mov ecx, BUCKETS
xor eax, eax
rep stosq
xor r12d, r12d	;bucket multiplied by 2
lea r14, [rbp+EH.pairs+STATE7_DEST-STATE7_BYTES]
lea r15, [rbp+EH.pairs+6*BLOCKUNIT-BUCKETS*4]
_EhStage7:
lea rdi, [rbp+EH.hashtab]
mov ecx, HASHTAB_ENTRIES/2
xor eax, eax
rep stosq
xor r11d, r11d
xor esi, esi
_EhStage7inner:
mov r8d, r11d
shl r8d, 9	;log2 (BUCKET*2)
or r8d, r12d
movzx r8d, word [rbp+r8+EH.bucket1ptr]
mov r10d, STATE6_BYTES
imul edx, r11d, ITEMS*0x10000
lea eax, [r11+r12*2]	;PARTS/2
imul r9d, eax, ITEMS*STATE6_BYTES
lea r9, [rbp+r9+EH.pairs+STATE6_DEST+STATE6_OFFSET]
call _ProcEhMakeLinksShr4
add r11d, 1
cmp r11d, PARTS
jb _EhStage7inner
	add [rsp], esi
	imul r10d, r12d, PARTS*ITEMS*STATE6_BYTES/2
	lea r10, [rbp+r10+EH.pairs+STATE6_DEST]
	lea r8, [rbp+EH.workingpairs]
	EhXor6_7 7
add r12d, 2
test r12d, 0x7e
jnz _EhStage7
add r13, BUCKETS*2
add r14, ITEMS*STATE7_BYTES
add r15, ITEMS*BUCKETS*4
cmp r12d, BUCKETS*2
jb _EhStage7
RecordRdtsc
mov eax, [rsp]
Record rax

;Stage8
	mov dword [rsp], 0
lea rdi, [rbp+EH.bucket1ptr]
mov r13, rdi
mov ecx, BUCKETS
xor eax, eax
rep stosq
xor r12d, r12d	;bucket multiplied by 2
lea r14, [rbp+EH.pairs+STATE8_DEST-STATE8_BYTES]
lea r15, [rbp+EH.pairs+7*BLOCKUNIT-BUCKETS*4]
_EhStage8:
lea rdi, [rbp+EH.hashtab]
mov ecx, HASHTAB_ENTRIES/2
xor eax, eax
rep stosq
xor r11d, r11d
xor esi, esi
_EhStage8inner:
mov r8d, r11d
shl r8d, 9	;log2 (BUCKET*2)
or r8d, r12d
movzx r8d, word [rbp+r8+EH.bucket0ptr]
mov r10d, STATE7_BYTES
imul edx, r11d, ITEMS*0x10000
lea eax, [r11+r12*2]	;PARTS/2
imul r9d, eax, ITEMS*STATE7_BYTES
lea r9, [rbp+r9+EH.pairs+STATE7_DEST+STATE7_OFFSET]
call _ProcEhMakeLinks
add r11d, 1
cmp r11d, PARTS
jb _EhStage8inner
	add [rsp], esi
	imul r10d, r12d, PARTS*ITEMS*STATE7_BYTES/2
	lea r10, [rbp+r10+EH.pairs+STATE7_DEST]
	lea r8, [rbp+EH.workingpairs]
	EhXor8
add r12d, 2
test r12d, 0x7e
jnz _EhStage8
add r13, BUCKETS*2
add r14, ITEMS*STATE8_BYTES
add r15, ITEMS*BUCKETS*4
cmp r12d, BUCKETS*2
jb _EhStage8
RecordRdtsc
mov eax, [rsp]
Record rax

;Stage9
	mov dword [rsp], 0
xor r12d, r12d	;bucket multiplied by 1
lea r15, [rbp+EH.pairs+8*BLOCKUNIT]
_EhStage9:
lea rdi, [rbp+EH.hashtab]
mov ecx, HASHTAB_ENTRIES/2
xor eax, eax
rep stosq
xor r11d, r11d
xor esi, esi
_EhStage9inner:
mov r8d, r11d
shl r8d, 8	;log2 (BUCKET)
or r8d, r12d
movzx r8d, word [rbp+r8*2+EH.bucket1ptr]
mov r10d, STATE8_BYTES
imul edx, r11d, ITEMS*0x10000
lea eax, [r11+r12*4]	;PARTS
imul r9d, eax, ITEMS*STATE8_BYTES
lea r9, [rbp+r9+EH.pairs+STATE8_DEST+STATE8_OFFSET]
call _ProcEhMakeLinksShr4
add r11d, 1
cmp r11d, PARTS
jb _EhStage9inner
	add [rsp], esi
	imul r10d, r12d, PARTS*ITEMS*STATE8_BYTES
	lea r10, [rbp+r10+EH.pairs+STATE8_DEST]
	lea r8, [rbp+EH.workingpairs]
	EhXor9
add r12d, 1
cmp r12d, BUCKETS
jb _EhStage9
mov eax, r15d
lea ecx, [rbp+EH.pairs+8*BLOCKUNIT]
sub eax, ecx
shr eax, 3
mov dword [rbp+EH.bucket0ptr], eax

RecordRdtsc
mov eax, [rsp]
Record rax

;mov ecx, fmtdn
;mov edx, dword [rbp+EH.bucket0ptr]
;call [printf]

;jmp _EhNoSolution
EhGetSolutions
mov ebx, eax
RecordRdtsc
mov eax, ebx

_EhSolverEpilog:
mov rsp, [rsp+0x20]
pop rbx
pop rbp
pop r12
pop r13
pop r14
pop r15
ret

_EhNoSolution:
xor eax, eax
jmp _EhSolverEpilog

_ProcEhMakeLinksShr4:
;r9: src bucket+offset
;r8d: src count
;r10d: stride
;esi: pairs count
;edx: input item id <<16
;Output:
;[workingpairs]
;esi: pairs count
;Destroy: rax, rcx, rdx, rbx, rsi, rdi, r8, r9
xor ebx, ebx
_LoopMakeLinksShr4:
movzx ecx, word [r9]
add r9, r10
shr ecx, 2
and ecx, HASHTAB_PATTERN_MASK shl 2
mov eax, [rbp+rcx+EH.hashtab]
cmp eax, HASHTAB_COUNT_2
jae _EhMultipairsShr4
mov edi, eax
and eax, HASHTAB_LOCATION_MASK
or eax, edx
mov [rbp+rsi*4+EH.workingpairs], eax
mov eax, edx
shr eax, 16
and edi, HASHTAB_COUNT_1
setnz bl
cmovnz eax, esi
add esi, ebx
add edi, HASHTAB_COUNT_1
or eax, edi
mov dword [rbp+rcx+EH.hashtab], eax
add edx, 0x10000
sub r8d, 1
ja _LoopMakeLinksShr4
ret

_EhMultipairsShr4:
mov edi, eax
add eax, HASHTAB_COUNT_1
and eax, HASHTAB_COUNT_MASK
jz _EhHashtabExShr4
_EhHashtabExShr4Back:
or eax, esi
mov dword [rbp+rcx+EH.hashtab], eax
mov ecx, edi
sub edi, HASHTAB_COUNT_2
and ecx, HASHTAB_LOCATION_MASK
lea rcx, [rbp+rcx*4+EH.workingpairs]
_LoopMultipairsShr4:
movzx eax, word [rcx]
add rcx, 4
or eax, edx
mov [rbp+rsi*4+EH.workingpairs], eax
add esi, 1
sub edi, HASHTAB_COUNT_1
jns _LoopMultipairsShr4
movzx eax, word [rcx-2]
or eax, edx
mov [rbp+rsi*4+EH.workingpairs], eax
add esi, 1
cmp esi, WORKINGPAIRS_LIMIT
jae _EhTruncateShr4
add edx, 0x10000
sub r8d, 1
ja _LoopMakeLinksShr4
ret

_EhHashtabExShr4:
mov eax, HASHTAB_COUNT_MASK
jmp _EhHashtabExShr4Back

_EhTruncateShr4:
mov esi, WORKINGPAIRS_LIMIT
ret


_ProcEhMakeLinks:
;r9: src bucket+offset
;r8d: src count
;r10d: stride
;esi: pairs count
;edx: input item id <<16
;Output:
;[workingpairs]
;esi: pairs count
;Destroy: rax, rcx, rdx, rbx, rsi, rdi, r8, r9
xor ebx, ebx
_LoopMakeLinks:
movzx ecx, word [r9]
add r9, r10
and ecx, HASHTAB_PATTERN_MASK
mov eax, dword [rbp+rcx*4+EH.hashtab]
cmp eax, HASHTAB_COUNT_2
jae _EhMultipairs
mov edi, eax
and eax, HASHTAB_LOCATION_MASK
or eax, edx
mov [rbp+rsi*4+EH.workingpairs], eax
mov eax, edx
shr eax, 16
and edi, HASHTAB_COUNT_1
setnz bl
cmovnz eax, esi
add esi, ebx
add edi, HASHTAB_COUNT_1
or eax, edi
mov dword [rbp+rcx*4+EH.hashtab], eax
add edx, 0x10000
sub r8d, 1
ja _LoopMakeLinks
ret

_EhMultipairs:
mov edi, eax
add eax, HASHTAB_COUNT_1
and eax, HASHTAB_COUNT_MASK
jz _EhHashtabEx
_EhHashtabExBack:
or eax, esi
mov dword [rbp+rcx*4+EH.hashtab], eax
mov ecx, edi
sub edi, HASHTAB_COUNT_2
and ecx, HASHTAB_LOCATION_MASK
lea rcx, [rbp+rcx*4+EH.workingpairs]
_LoopMultipairs:
movzx eax, word [rcx]
add rcx, 4
or eax, edx
mov [rbp+rsi*4+EH.workingpairs], eax
add esi, 1
sub edi, HASHTAB_COUNT_1
jns _LoopMultipairs
movzx eax, word [rcx-2]
or eax, edx
mov [rbp+rsi*4+EH.workingpairs], eax
add esi, 1
cmp esi, WORKINGPAIRS_LIMIT
jae _EhTruncate
add edx, 0x10000
sub r8d, 1
ja _LoopMakeLinks
ret

_EhHashtabEx:
mov eax, HASHTAB_COUNT_MASK
jmp _EhHashtabExBack

_EhTruncate:
mov esi, WORKINGPAIRS_LIMIT
ret
