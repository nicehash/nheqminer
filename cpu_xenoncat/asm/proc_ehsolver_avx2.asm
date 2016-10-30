include "macro_blake2b_avx2.asm"

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

_ProcEhSolver:
push r15
push r14
push r13
push r12
push rdi
push rsi
push rbp
push rbx
mov rax, rsp
sub rsp, 0xd8
and rsp, -32
vmovdqa [rsp+0x20], xmm6
vmovdqa [rsp+0x30], xmm7
vmovdqa [rsp+0x40], xmm8
vmovdqa [rsp+0x50], xmm9
vmovdqa [rsp+0x60], xmm10
vmovdqa [rsp+0x70], xmm11
vmovdqa [rsp+0x80], xmm12
vmovdqa [rsp+0x90], xmm13
vmovdqa [rsp+0xa0], xmm14
vmovdqa [rsp+0xb0], xmm15
mov [rsp+0xc0], rax

mov rbp, rcx
vmovd xmm0, edx
vpbroadcastq ymm0, xmm0
vpblendd ymm0, ymm0, yword [yctrinit], 0xaa
vmovdqa yword [rbp+EH.mids+0xe0], ymm0

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
vmovdqa ymm7, yword [rbp+EH.mids+0xe0]
vpaddd ymm7, ymm7, yword [yctrinc]
vmovdqa yword [rbp+EH.mids+0xe0], ymm7
vbroadcasti128 ymm7, xword [xshufb_bswap8]
vpshufb ymm0, ymm0, ymm7
vpshufb ymm1, ymm1, ymm7
vpshufb ymm2, ymm2, ymm7
vpshufb ymm3, ymm3, ymm7
vpshufb ymm4, ymm4, ymm7
vpshufb ymm5, ymm5, ymm7
vpshufb ymm6, ymm6, ymm7

vpsrlq ymm8, ymm0, 56
vpsllq ymm0, ymm0, 8
;vpsrlq ymm0, ymm0, 0
vpunpcklqdq ymm7, ymm2, ymm1
vpunpckhqdq ymm2, ymm2, ymm1
vpsrlq ymm1, ymm1, 56
vpor ymm9, ymm0, ymm1		;xmm9=bucket data 2,0
vpshufd ymm1, ymm3, 0x4e

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
vpalignr ymm10, ymm7, ymm1, 15	;xmm10=xor data 0
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
vpalignr ymm11, ymm2, ymm3, 15	;xmm11=xor data 2
vmovq rax, xmm11
movnti [rdi], rax
vpextrq rax, xmm11, 1
movnti [rdi+8], rax
vpextrq rax, xmm9, 1
movnti [rdi+16], rax
_SkipA2:

vextracti128 xmm8, ymm8, 1
vextracti128 xmm9, ymm9, 1
vextracti128 xmm10, ymm10, 1
vextracti128 xmm11, ymm11, 1

vpextrw r8d, xmm8, 0
movzx eax, word [r12+r8*2]
add ecx, 2
cmp eax, 10240
jae _SkipA4
imul edx, eax, 24
imul r10d, eax, 1024
add eax, 1
mov word [r12+r8*2], ax
lea r11, [rbx+r8*4]
mov [r10+r11], ecx
imul r8d, r8d, PARTS*ITEMS*STATE0_BYTES
lea rdi, [rsi+r8]
add rdi, rdx
vmovq rax, xmm10
movnti [rdi], rax
vpextrq rax, xmm10, 1
movnti [rdi+8], rax
vmovq rax, xmm9
movnti [rdi+16], rax
_SkipA4:

vpextrw r8d, xmm8, 4
movzx eax, word [r12+r8*2]
add ecx, 2
cmp eax, 10240
jae _SkipA6
imul edx, eax, 24
imul r10d, eax, 1024
add eax, 1
mov word [r12+r8*2], ax
lea r11, [rbx+r8*4]
mov [r10+r11], ecx
imul r8d, r8d, PARTS*ITEMS*STATE0_BYTES
lea rdi, [rsi+r8]
add rdi, rdx
vmovq rax, xmm11
movnti [rdi], rax
vpextrq rax, xmm11, 1
movnti [rdi+8], rax
vpextrq rax, xmm9, 1
movnti [rdi+16], rax
_SkipA6:

vpsllq ymm8, ymm3, 8
vpsrlq ymm8, ymm8, 56

vpunpcklqdq ymm2, ymm4, ymm3
vpunpckhqdq ymm3, ymm4, ymm3
vpunpcklqdq ymm4, ymm6, ymm5
vpunpckhqdq ymm5, ymm6, ymm5
vpsrldq ymm0, ymm2, 6
vpsrldq ymm1, ymm3, 6
;vbroadcasti128 ymm7, xword [xqmask64bit]
;vpand ymm0, ymm0, ymm7	;bucket data 1
;vpand ymm1, ymm1, ymm7	;bucket data 3

vpextrw r8d, xmm8, 0
movzx eax, word [r12+r8*2]
sub ecx, 5
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
vpalignr ymm2, ymm2, ymm4, 6	;xor data 1
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
vpalignr ymm3, ymm3, ymm5, 6	;xor data 3
vmovq rax, xmm3
movnti [rdi], rax
vpextrq rax, xmm3, 1
movnti [rdi+8], rax
vmovq rax, xmm1
movnti [rdi+16], rax
_SkipA3:

vextracti128 xmm8, ymm8, 1
vextracti128 xmm0, ymm0, 1
vextracti128 xmm1, ymm1, 1
vextracti128 xmm2, ymm2, 1
vextracti128 xmm3, ymm3, 1

vpextrw r8d, xmm8, 0
movzx eax, word [r12+r8*2]
add ecx, 2
cmp eax, 10240
jae _SkipA5
imul edx, eax, 24
imul r10d, eax, 1024
add eax, 1
mov word [r12+r8*2], ax
lea r11, [rbx+r8*4]
mov [r10+r11], ecx
imul r8d, r8d, PARTS*ITEMS*STATE0_BYTES
lea rdi, [rsi+r8]
add rdi, rdx
vmovq rax, xmm2
movnti [rdi], rax
vpextrq rax, xmm2, 1
movnti [rdi+8], rax
vmovq rax, xmm0
movnti [rdi+16], rax
_SkipA5:

vpextrw r8d, xmm8, 4
movzx eax, word [r12+r8*2]
add ecx, 2
cmp eax, 10240
jae _SkipA7
imul edx, eax, 24
imul r10d, eax, 1024
add eax, 1
mov word [r12+r8*2], ax
lea r11, [rbx+r8*4]
mov [r10+r11], ecx
imul r8d, r8d, PARTS*ITEMS*STATE0_BYTES
lea rdi, [rsi+r8]
add rdi, rdx
vmovq rax, xmm3
movnti [rdi], rax
vpextrq rax, xmm3, 1
movnti [rdi+8], rax
vmovq rax, xmm1
movnti [rdi+16], rax
_SkipA7:

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
vmovdqa xmm6, [rsp+0x20]
vmovdqa xmm7, [rsp+0x30]
vmovdqa xmm8, [rsp+0x40]
vmovdqa xmm9, [rsp+0x50]
vmovdqa xmm10, [rsp+0x60]
vmovdqa xmm11, [rsp+0x70]
vmovdqa xmm12, [rsp+0x80]
vmovdqa xmm13, [rsp+0x90]
vmovdqa xmm14, [rsp+0xa0]
vmovdqa xmm15, [rsp+0xb0]
mov rsp, [rsp+0xc0]
pop rbx
pop rbp
pop rsi
pop rdi
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
