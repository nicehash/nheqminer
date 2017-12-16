;For the following EhXor*,
;Inputs:
;r12d bucket id (0-255) multiplied by 2, (0-510)
;r13 dstbucketctr
;r14 dst, with compensation -STATE*_BYTES
;r15 dstpairs, with compensation -4
;esi pairs count
;r10 src bucket
;r8 ptr to workingpairs

macro EhXor1_3 stage
;Destroy: rax, rcx, rdx, rbx, rsi, rdi, r8
{
local _Loop1, _ZeroLoop, _Skip1
test esi, esi
jz _ZeroLoop
shl r12, 25	;6 bits out of 8 bits visible
_Loop1:
mov edi, [r8]
add r8, 4

if stage=1
;multiply 24, performed as *3*8
lea eax, [rdi+rdi*2]	;SIMD within a register
movzx edx, ax
shr eax, 16
lea rdx, [r10+rdx*8]
lea rax, [r10+rax*8]

else if stage=2
;multiply 24, performed as *3*8
lea eax, [rdi+rdi*2]	;SWAR
movzx edx, ax
shr eax, 16
lea rdx, [r10+rdx*8]
lea rax, [r10+rax*8]

else
;multiply 20, performed as *5*4
lea eax, [rdi+rdi*4]	;SWAR, caution: near limit 11584*5=57920
movzx edx, ax
shr eax, 16
lea rdx, [r10+rdx*4]
lea rax, [r10+rax*4]
end if

vmovdqu xmm0, [rdx]
vpxor xmm0, xmm0, [rax]

if stage=1
mov rdx, [rdx+16]
xor rdx, [rax+16]
mov rax, rdx
shr rdx, 44
else if stage=2
mov rdx, [rdx+16]
xor rdx, [rax+16]
mov rax, rdx
shr rdx, 24
else
mov edx, [rdx+16]
xor edx, [rax+16]
mov eax, edx
shr edx, 4
end if

movzx ecx, word [r13+rdx*2]
cmp ecx, ITEMS
jae _Skip1
add ecx, 1
mov word [r13+rdx*2], cx
mov ebx, ecx
shl ebx, 10	;log2 (BUCKETS*4)
add rbx, r15
lea rbx, [rbx+rdx*4]

if stage=1
imul edx, edx, PARTS*ITEMS*STATE1_BYTES
imul ecx, ecx, STATE1_BYTES
else if stage=2
imul edx, edx, PARTS*ITEMS*STATE2_BYTES
imul ecx, ecx, STATE2_BYTES
else
imul edx, edx, PARTS*ITEMS*STATE3_BYTES
imul ecx, ecx, STATE3_BYTES
end if

add rdx, r14
add rdx, rcx
vmovdqu [rdx], xmm0

if stage=1
mov [rdx+16], rax
else if stage=2
mov [rdx+16], eax
else
mov [rdx+16], eax
end if

movzx ecx, di
shr edi, 16
lea eax, [rdi-1]
imul eax, edi
shr eax, 1
add eax, ecx
or eax, r12d
mov [rbx], eax
_Skip1:
sub esi, 1
jnz _Loop1
shr r12, 25
_ZeroLoop:
}

macro EhXor4
;Destroy: rax, rcx, rdx, rbx, rsi, rdi, r8
{
local _Loop1, _ZeroLoop, _Skip1

test esi, esi
jz _ZeroLoop
shl r12, 25
_Loop1:
mov edi, [r8]
add r8, 4

;multiply 20, performed as *5*4
lea eax, [rdi+rdi*4]	;SWAR, caution: near limit 11584*5=57920
movzx edx, ax
shr eax, 16
lea rdx, [r10+rdx*4]
lea rax, [r10+rax*4]

vmovdqu xmm0, [rdx]
vpxor xmm0, xmm0, [rax]
vptest xmm0, xmm0
jz _Skip1

mov edx, [rdx+12]
xor edx, [rax+12]
shr edx, 16

movzx ecx, word [r13+rdx*2]
cmp ecx, ITEMS
jae _Skip1
add ecx, 1
mov word [r13+rdx*2], cx
mov ebx, ecx
shl ebx, 10	;log2 (BUCKETS*4)
add rbx, r15
lea rbx, [rbx+rdx*4]

imul edx, edx, PARTS*ITEMS*STATE4_BYTES
shl ecx, 4	;log2 STATE4_BYTES

add rdx, r14
add rdx, rcx
vmovdqu [rdx], xmm0

movzx ecx, di
shr edi, 16
lea eax, [rdi-1]
imul eax, edi
shr eax, 1
add eax, ecx
or eax, r12d
mov [rbx], eax
_Skip1:
sub esi, 1
jnz _Loop1
shr r12, 25
_ZeroLoop:
}

macro EhXor5
;Destroy: rax, rcx, rdx, rbx, rsi, rdi, r8
{
local _Loop1, _ZeroLoop, _Skip1

test esi, esi
jz _ZeroLoop
shl r12, 25
_Loop1:
mov edi, [r8]
add r8, 4

;multiply 16, performed as *2*8
lea eax, [rdi+rdi]	;SWAR
movzx edx, ax
shr eax, 16

vmovdqu xmm0, [r10+rdx*8]
vpxor xmm0, xmm0, [r10+rax*8]
vptest xmm0, xmm0
jz _Skip1

vpextrq rdx, xmm0, 1
mov rax, rdx
shr rdx, 28

movzx ecx, word [r13+rdx*2]
cmp ecx, ITEMS
jae _Skip1
add ecx, 1
mov word [r13+rdx*2], cx
mov ebx, ecx
shl ebx, 10	;log2 (BUCKETS*4)
add rbx, r15
lea rbx, [rbx+rdx*4]

imul edx, edx, PARTS*ITEMS*STATE5_BYTES
imul ecx, ecx, STATE5_BYTES

add rdx, r14
add rdx, rcx
vmovq [rdx], xmm0
mov [rdx+8], eax

movzx ecx, di
shr edi, 16
lea eax, [rdi-1]
imul eax, edi
shr eax, 1
add eax, ecx
or eax, r12d
mov [rbx], eax
_Skip1:
sub esi, 1
jnz _Loop1
shr r12, 25
_ZeroLoop:
}

macro EhXor6_7 stage
;Destroy: rax, rcx, rdx, rbx, rsi, rdi, r8
{
local _Loop1, _ZeroLoop, _Skip1

test esi, esi
jz _ZeroLoop
shl r12, 25
_Loop1:
mov edi, [r8]
add r8, 4

;multiply 12, performed as *3*4
lea eax, [rdi+rdi*2]	;SWAR
movzx edx, ax
shr eax, 16
lea rdx, [r10+rdx*4]
lea rax, [r10+rax*4]

vmovq xmm0, [rdx]
vpinsrd xmm0, xmm0, [rdx+8], 2
vmovq xmm1, [rax]
vpinsrd xmm1, xmm1, [rax+8], 2
vpxor xmm0, xmm0, xmm1
vptest xmm0, xmm0
jz _Skip1

if stage=6
vpextrd edx, xmm0, 2
mov eax, edx
shr edx, 8
else
vpextrd edx, xmm0, 1
mov eax, edx
shr edx, 20
end if

movzx ecx, word [r13+rdx*2]
cmp ecx, ITEMS
jae _Skip1
add ecx, 1
mov word [r13+rdx*2], cx
mov ebx, ecx
shl ebx, 10	;log2 (BUCKETS*4)
add rbx, r15
lea rbx, [rbx+rdx*4]

if stage=6
imul edx, edx, PARTS*ITEMS*STATE6_BYTES
imul ecx, ecx, STATE6_BYTES
else
imul edx, edx, PARTS*ITEMS*STATE7_BYTES
shl ecx, 3	;log2 STATE7_BYTES
end if

add rdx, r14
add rdx, rcx
vmovq [rdx], xmm0

if stage=6
mov [rdx+8], eax
end if

movzx ecx, di
shr edi, 16
lea eax, [rdi-1]
imul eax, edi
shr eax, 1
add eax, ecx
or eax, r12d
mov [rbx], eax
_Skip1:
sub esi, 1
jnz _Loop1
shr r12, 25
_ZeroLoop:
}

macro EhXor8
;Destroy: rax, rcx, rdx, rbx, rsi, rdi, r8
{
local _Loop1, _ZeroLoop, _Skip1

test esi, esi
jz _ZeroLoop
shl r12, 25
_Loop1:
mov edi, [r8]
add r8, 4

mov eax, edi
movzx edx, di
shr eax, 16

mov rdx, [r10+rdx*8]
xor rdx, [r10+rax*8]
jz _Skip1

mov eax, edx
shr rdx, 32

movzx ecx, word [r13+rdx*2]
cmp ecx, ITEMS
jae _Skip1
add ecx, 1
mov word [r13+rdx*2], cx
mov ebx, ecx
shl ebx, 10	;log2 (BUCKETS*4)
add rbx, r15
lea rbx, [rbx+rdx*4]

imul edx, edx, PARTS*ITEMS*STATE8_BYTES
shl ecx, 2	;log2 STATE8_BYTES

add rdx, r14
add rdx, rcx
mov [rdx], eax

movzx ecx, di
shr edi, 16
lea eax, [rdi-1]
imul eax, edi
shr eax, 1
add eax, ecx
or eax, r12d
mov [rbx], eax
_Skip1:
sub esi, 1
jnz _Loop1
shr r12, 25
_ZeroLoop:
}

macro EhXor9
;r12d=bucket id, not multiplied by 2
;Output r15
;Destroy: rax, rcx, rdx, rbx, rsi, rdi, r8
{
local _Loop1, _ZeroLoop, _Skip1

test esi, esi
jz _ZeroLoop
mov eax, r12d
shl rax, 32
or r12, rax
_Loop1:
mov edi, [r8]
add r8, 4

lea ecx, [rdi*4]
movzx eax, cx
shr ecx, 16

mov eax, [r10+rax]
xor eax, [r10+rcx]
jnz _Skip1

movzx eax, di
and edi, 0xffff0000
shl rdi, 16
or rax, rdi
shl rax, 8
or rax, r12
mov [r15], rax
add r15, 8

_Skip1:
sub esi, 1
jnz _Loop1
_ZeroLoop:
}

macro ExpandTree
;ib
;Input:
;rdx: the pair
{
local _LoopOuter, _LoopInner, _LoopCheckDup, _LoopCheckDupInner, _EnterCheckDup, _LoopBasemap, \
	_LoopSort1, _LoopSort2, _LoopSort3, _LoopSort4, _LoopSort4inner, _LoopSort4swap, _Sort4noswap, _Invalid, _EndExpandTree, _Skip1
;lea rdi, [rbp+EH.pairs+9*BLOCKUNIT]
;mov ecx, 512
;mov eax, -1
;rep stosd
lea rdi, [rbp+EH.pairs+9*BLOCKUNIT]
mov r8, rdi
lea rsi, [rbp+EH.pairs+7*BLOCKUNIT]
mov eax, edx
shr rdx, 32
vmovd xmm0, dword [rsi+rax*4]
vpinsrd xmm0, xmm0, dword [rsi+rdx*4], 2
vpand xmm1, xmm0, xmm7		;0x03ffffff
vpsrld xmm0, xmm0, 26
vmovd xmm2, eax
vpinsrd xmm2, xmm2, edx, 2
vpmuludq xmm2, xmm2, xmm5	;divide (2896*256)
vpsrlq xmm2, xmm2, 19+32
vpsllq xmm2, xmm2, 6
vpor xmm0, xmm0, xmm2

vpshufd xmm2, xmm1, 0x88
vpaddd xmm2, xmm2, xmm2
vpaddd xmm2, xmm2, xmm6
vcvtdq2pd xmm2, xmm2
vsqrtpd xmm2, xmm2
vcvtpd2dq xmm2, xmm2
vpshufd xmm2, xmm2, 0xd8	;high
vpsubd xmm3, xmm2, xmm6
vpmuludq xmm3, xmm2, xmm3
vpsrld xmm3, xmm3, 1
vpsubd xmm1, xmm1, xmm3		;low
vpshufd xmm0, xmm0, 0xa0
vpsllq xmm2, xmm2, 32
vpor xmm1, xmm1, xmm2
vpslld xmm1, xmm1, 8
vpor xmm0, xmm0, xmm1		;3210
vpshufd xmm1, xmm0, 0x4a	;3210^1022
vpcmpeqd xmm1, xmm0, xmm1
vptest xmm1, xmm1
jnz _Invalid


lea rsi, [rbp+EH.pairs+6*BLOCKUNIT]
vpslld xmm4, xmm0, 2

vmovd eax, xmm4
vpextrd edx, xmm4, 1
vmovd xmm0, dword [rsi+rax]
vpinsrd xmm0, xmm0, dword [rsi+rdx], 2
vpand xmm1, xmm0, xmm7		;0x03ffffff
vpsrld xmm0, xmm0, 26

vmovd xmm2, eax
vpinsrd xmm2, xmm2, edx, 2
vpmuludq xmm2, xmm2, xmm5	;divide (2896*256*4)
vpsrlq xmm2, xmm2, 21+32
vpsllq xmm2, xmm2, 6
vpor xmm0, xmm0, xmm2

;mov ebx, edx
;mov ecx, ITEMS*BUCKETS*4
;xor edx, edx
;div ecx
;vmovd xmm2, eax
;mov eax, ebx
;xor edx, edx
;div ecx
;vpinsrd xmm2, xmm2, eax, 2
;vpslld xmm2, xmm2, 6
;vpor xmm0, xmm0, xmm2

vpextrd eax, xmm4, 2
vpextrd edx, xmm4, 3
vpshufd xmm2, xmm1, 0x88
vpaddd xmm2, xmm2, xmm2
vpaddd xmm2, xmm2, xmm6
vcvtdq2pd xmm2, xmm2
vsqrtpd xmm2, xmm2
vcvtpd2dq xmm2, xmm2
vpshufd xmm2, xmm2, 0xd8	;high
vpsubd xmm3, xmm2, xmm6
vpmuludq xmm3, xmm2, xmm3
vpsrld xmm3, xmm3, 1
vpsubd xmm1, xmm1, xmm3		;low
vpshufd xmm0, xmm0, 0xa0
vpsllq xmm2, xmm2, 32
vpor xmm1, xmm1, xmm2
vpslld xmm1, xmm1, 8
vpor xmm4, xmm0, xmm1		;3210
vpshufd xmm1, xmm4, 0x4e
vpcmpeqd xmm1, xmm4, xmm1
vptest xmm1, xmm1
jnz _Invalid
vpslld xmm4, xmm4, 2
vmovdqa [rdi], xmm4

vmovd xmm0, dword [rsi+rax]
vpinsrd xmm0, xmm0, dword [rsi+rdx], 2
vpand xmm1, xmm0, xmm7		;0x03ffffff
vpsrld xmm0, xmm0, 26

vmovd xmm2, eax
vpinsrd xmm2, xmm2, edx, 2
vpmuludq xmm2, xmm2, xmm5	;divide (2896*256*4)
vpsrlq xmm2, xmm2, 21+32
vpsllq xmm2, xmm2, 6
vpor xmm0, xmm0, xmm2

vpshufd xmm2, xmm1, 0x88
vpaddd xmm2, xmm2, xmm2
vpaddd xmm2, xmm2, xmm6
vcvtdq2pd xmm2, xmm2
vsqrtpd xmm2, xmm2
vcvtpd2dq xmm2, xmm2
vpshufd xmm2, xmm2, 0xd8	;high
vpsubd xmm3, xmm2, xmm6
vpmuludq xmm3, xmm2, xmm3
vpsrld xmm3, xmm3, 1
vpsubd xmm1, xmm1, xmm3		;low
vpshufd xmm0, xmm0, 0xa0
vpsllq xmm2, xmm2, 32
vpor xmm1, xmm1, xmm2
vpslld xmm1, xmm1, 8
vpor xmm0, xmm0, xmm1		;7654
vpslld xmm0, xmm0, 2
vpcmpeqd xmm1, xmm0, xmm4
vptest xmm1, xmm1
jnz _Invalid
vpshufd xmm2, xmm0, 0x4e
vpcmpeqd xmm1, xmm2, xmm4
vptest xmm1, xmm1
jnz _Invalid
vpcmpeqd xmm1, xmm0, xmm2
vptest xmm1, xmm1
jnz _Invalid
vmovdqa [rdi+1024], xmm0


mov ebx, 512
_LoopOuter:
sub rsi, BLOCKUNIT
mov rdi, r8
lea r9, [rdi+2048]
_LoopInner:
mov eax, [rdi]
mov edx, [rdi+4]
vmovd xmm0, dword [rsi+rax]
vpinsrd xmm0, xmm0, dword [rsi+rdx], 2
vpand xmm1, xmm0, xmm7		;0x03ffffff
vpsrld xmm0, xmm0, 26

vmovd xmm2, eax
vpinsrd xmm2, xmm2, edx, 2
vpmuludq xmm2, xmm2, xmm5	;divide (2896*256*4)
vpsrlq xmm2, xmm2, 21+32
vpsllq xmm2, xmm2, 6
vpor xmm0, xmm0, xmm2

mov eax, [rdi+8]
mov edx, [rdi+12]
vpshufd xmm2, xmm1, 0x88
vpaddd xmm2, xmm2, xmm2
vpaddd xmm2, xmm2, xmm6
vcvtdq2pd xmm2, xmm2
vsqrtpd xmm2, xmm2
vcvtpd2dq xmm2, xmm2
vpshufd xmm2, xmm2, 0xd8	;high
vpsubd xmm3, xmm2, xmm6
vpmuludq xmm3, xmm2, xmm3
vpsrld xmm3, xmm3, 1
vpsubd xmm1, xmm1, xmm3		;low
vpshufd xmm0, xmm0, 0xa0
vpsllq xmm1, xmm1, 32
vpor xmm1, xmm1, xmm2
vpslld xmm1, xmm1, 8
vpor xmm0, xmm0, xmm1		;3210
vpslld xmm0, xmm0, 2
vmovdqa [rdi], xmm0

vmovd xmm0, dword [rsi+rax]
vpinsrd xmm0, xmm0, dword [rsi+rdx], 2
vpand xmm1, xmm0, xmm7		;0x03ffffff
vpsrld xmm0, xmm0, 26

vmovd xmm2, eax
vpinsrd xmm2, xmm2, edx, 2
vpmuludq xmm2, xmm2, xmm5	;divide (2896*256*4)
vpsrlq xmm2, xmm2, 21+32
vpsllq xmm2, xmm2, 6
vpor xmm0, xmm0, xmm2

vpshufd xmm2, xmm1, 0x88
vpaddd xmm2, xmm2, xmm2
vpaddd xmm2, xmm2, xmm6
vcvtdq2pd xmm2, xmm2
vsqrtpd xmm2, xmm2
vcvtpd2dq xmm2, xmm2
vpshufd xmm2, xmm2, 0xd8	;high
vpsubd xmm3, xmm2, xmm6
vpmuludq xmm3, xmm2, xmm3
vpsrld xmm3, xmm3, 1
vpsubd xmm1, xmm1, xmm3		;low
vpshufd xmm0, xmm0, 0xa0
vpsllq xmm1, xmm1, 32
vpor xmm1, xmm1, xmm2
vpslld xmm1, xmm1, 8
vpor xmm0, xmm0, xmm1		;7654
vpslld xmm0, xmm0, 2
vmovdqa [rdi+rbx], xmm0

lea rdi, [rdi+rbx*2]
cmp rdi, r9
jb _LoopInner
shr ebx, 1
cmp ebx, 8
ja _LoopOuter

if 1
mov rdi, r8
mov esi, 128
jmp _EnterCheckDup
_LoopCheckDup:
mov ecx, esi

_LoopCheckDupInner:
vmovdqa xmm4, [rax]
vpcmpeqd xmm8, xmm0, xmm4
vpcmpeqd xmm9, xmm1, xmm4
vpor xmm8, xmm8, xmm9
vpcmpeqd xmm10, xmm2, xmm4
vpcmpeqd xmm4, xmm3, xmm4
vpor xmm4, xmm10, xmm4
vpor xmm4, xmm8, xmm4
vptest xmm4, xmm4
jnz _Invalid
add rax, 16
sub ecx, 1
jnz _LoopCheckDupInner

_EnterCheckDup:
vmovdqa xmm0, [rdi]
vpshufd xmm2, xmm0, 0x50
vpshufd xmm3, xmm0, 0xee
vpcmpeqd xmm2, xmm3, xmm2	;02,03,12,13
vptest xmm2, xmm2
jnz _Invalid
vpshufd xmm1, xmm0, 0x00
vpshufd xmm2, xmm0, 0xff
vpshufd xmm3, xmm0, 0xaa
vpshufd xmm0, xmm0, 0x55
add rdi, 16
mov rax, rdi
sub esi, 1
jnz _LoopCheckDup
end if

lea rsi, [rbp+EH.basemap]
mov rdi, r8
mov ecx, 512
_LoopBasemap:
mov eax, [rdi]
mov eax, [rsi+rax]
mov [rdi], eax
add rdi, 4
sub ecx, 1
jnz _LoopBasemap

mov rdi, r8
_LoopSort1:
mov eax, [rdi]
mov ecx, [rdi+4]
mov edx, eax
cmp eax, ecx
cmova eax, ecx
cmova ecx, edx
mov [rdi], eax
mov [rdi+4], ecx
add rdi, 8
cmp rdi, r9
jb _LoopSort1

mov rdi, r8
_LoopSort2:
mov rax, [rdi]
mov rcx, [rdi+8]
mov rdx, rax
cmp eax, ecx
cmova rax, rcx
cmova rcx, rdx
mov [rdi], rax
mov [rdi+8], rcx
add rdi, 16
cmp rdi, r9
jb _LoopSort2

mov rdi, r8
_LoopSort3:
vmovdqa xmm0, [rdi]
vmovdqa xmm1, [rdi+16]
mov rdx, rdi
lea rbx, [rdi+16]
mov rcx, rdx
mov eax, [rdi]
cmp eax, [rdi+16]
cmova rdx, rbx
cmova rbx, rcx
vmovdqa [rdx], xmm0
vmovdqa [rbx], xmm1
add rdi, 32
cmp rdi, r9
jb _LoopSort3

mov esi, 32
_LoopSort4:
mov rdi, r8

_LoopSort4inner:
mov eax, [rdi]
cmp eax, [rdi+rsi]
jb _Sort4noswap
mov rax, rdi
lea rcx, [rdi+rsi]
_LoopSort4swap:
vmovdqa ymm0, [rax]
vmovdqa ymm1, [rax+rsi]
vmovdqa [rax+rsi], ymm0
vmovdqa [rax], ymm1
add rax, 32
cmp rax, rcx
jb _LoopSort4swap
_Sort4noswap:
lea rdi, [rdi+rsi*2]
cmp rdi, r9
jb _LoopSort4inner

add esi, esi
cmp esi, 512*4
jb _LoopSort4

_Skip1:
mov eax, 1
jmp _EndExpandTree
_Invalid:
xor eax, eax
_EndExpandTree:
}

macro EhCompact dst, src
;32bits->21bits
;32bytes->21bytes
;Caution: Overwrite extra 3 bytes at the end
;Input: [src]
;Output: [dst]
;Destroy: rax, rcx, rdx, r10, dst, src
{
local _Loop1
mov r10d, 512/8
_Loop1:
mov eax, [src]
mov edx, [src+4]
shl rax, 21
or rax, rdx
mov edx, [src+8]
shl rax, 21
or rax, rdx
mov edx, [src+12]
mov ecx, edx	;save 20lsb
shr edx, 20
shl rax, 1
or rax, rdx
bswap rax
mov [dst], rax
mov edx, [src+16]
shl rcx, 21
or rcx, rdx
mov edx, [src+20]
shl rcx, 21
or rcx, rdx
mov edx, [src+24]
mov eax, edx	;save 19lsb
shr edx, 19
shl rcx, 2
or rcx, rdx
bswap rcx
mov [dst+8], rcx
mov edx, [src+28]
shl rax, 21
or rax, rdx
shl rax, 24
bswap rax
mov [dst+16], rax	;extra 3 bytes
add src, 32
add dst, 21
sub r10d, 1
jnz _Loop1
}

macro EhGetSolutions
{
local _LoopFindValid, _Skip1, _ZeroLoop

xor r13d, r13d
mov r14d, dword [rbp+EH.bucket0ptr]
lea r12, [rbp+EH.hashtab]
test r14d, r14d
jz _ZeroLoop
lea r15, [rbp+EH.pairs+8*BLOCKUNIT]
mov eax, 0x03ffffff
vmovd xmm7, eax
mov eax, 1
vmovd xmm6, eax
mov eax, 0xb509e68b	;recip of 2896*256
vmovd xmm5, eax
vpshufd xmm7, xmm7, 0
vpshufd xmm6, xmm6, 0
vpshufd xmm5, xmm5, 0
_LoopFindValid:
cmp r13d, 64
jae _ZeroLoop
mov rdx, [r15]
ExpandTree
add r13d, eax
test eax, eax
jz _Skip1
lea rsi, [rbp+EH.pairs+9*BLOCKUNIT]
	EhCompact r12, rsi
;mov rdi, r12
;mov ecx, 2048/8
;rep movsq
;add r12, 2048
_Skip1:
add r15, 8
sub r14d, 1
jnz _LoopFindValid
_ZeroLoop:
mov eax, r13d
}
