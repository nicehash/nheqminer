_ProcEhPrepare:
sub rsp, 0x1f8
vmovdqa [rsp+0x180], xmm6
vmovdqa [rsp+0x190], xmm7
vmovdqa [rsp+0x1a0], xmm8
vmovdqa [rsp+0x1b0], xmm9
vmovdqa [rsp+0x1c0], xmm10
vmovdqa [rsp+0x1d0], xmm11

vmovdqa xmm10, xword [xshufb_ror24]
vmovdqa xmm11, xword [xshufb_ror16]

vmovdqa xmm0, xword [s0]
vmovdqa xmm1, xword [s2]
vmovdqa xmm2, xword [s4]
vmovdqa xmm3, xword [s6]
vmovdqa xmm4, xword [iv]
vmovdqa xmm5, xword [iv+0x10]
vmovdqa xmm6, xword [iv4xor128]
vmovdqa xmm7, xword [iv4xor128+0x10]

mov r8, rsp
; mov r9d, blake2sigma
lea r9, [blake2sigma]
lea r11, [blake2sigma+160]
call _ProcBlakeMsgSched
call _ProcBlakeRound
add r8, 0x80
add r9, 16
call _ProcBlakeMsgSched
call _ProcBlakeRound
add r8, 0x80
add r9, 16
_LoopEhPrepare1:
call _ProcBlakeMsgSched
call _ProcBlakeRound
add r9, 16
; cmp r9, blake2sigma+160
cmp r9, r11
jb _LoopEhPrepare1
mov r8, rsp
call _ProcBlakeRound
add r8, 0x80
call _ProcBlakeRound

vpxor xmm0, xmm0, xmm4
vpxor xmm1, xmm1, xmm5
vpxor xmm2, xmm2, xmm6
vpxor xmm3, xmm3, xmm7
vpxor xmm0, xmm0, xword [s0]
vpxor xmm1, xmm1, xword [s2]
vpxor xmm2, xmm2, xword [s4]
vpxor xmm3, xmm3, xword [s6]
vmovdqa xword [rcx+EH.mids+0x80], xmm0
vmovdqa xword [rcx+EH.mids+0x90], xmm1
vmovdqa xword [rcx+EH.mids+0xa0], xmm2
vmovdqa xword [rcx+EH.mids+0xb0], xmm3
vmovq xmm8, [rdx+0x80]
vpshufd xmm4, xmm8, 0x44
vmovdqa xword [rcx+EH.mids+0xc0], xmm4

;Begin second message block
vmovdqa xmm4, xword [iv]
vmovdqa xmm5, xword [iv+0x10]
vmovdqa xmm6, xword [iv4xor144]
vmovdqa xmm7, xword [iv6inverted]
vpaddq xmm0, xmm0, xmm2
vpaddq xmm1, xmm1, xmm3
vpaddq xmm0, xmm0, xmm8		;xmm8[63:0]=message
vpxor xmm6, xmm6, xmm0
vpxor xmm7, xmm7, xmm1
vpshufd xmm6, xmm6, 0xb1
	vmovq [rcx+EH.mids+0x08], xmm6	;v12
vpshufd xmm7, xmm7, 0xb1
vpaddq xmm4, xmm4, xmm6
	vmovq [rcx+EH.mids+0x10], xmm4	;v8
vpaddq xmm5, xmm5, xmm7
vpxor xmm2, xmm2, xmm4
vpxor xmm3, xmm3, xmm5
vpshufb xmm2, xmm2, xmm10
	vmovq [rcx+EH.mids+0x18], xmm2	;v4
vpshufb xmm3, xmm3, xmm10

vpaddq xmm0, xmm0, xmm2
	vmovq [rcx+EH.mids], xmm0	;v0
vpaddq xmm1, xmm1, xmm3
	vpextrq [rcx+EH.mids+0x60], xmm1, 1	;v3
;add message (nonce, index) to xmm0 here, but we don't have
vpxor xmm6, xmm6, xmm0
vpxor xmm7, xmm7, xmm1
vpshufb xmm6, xmm6, xmm11
vpshufb xmm7, xmm7, xmm11
	vmovdqa xword [rcx+EH.mids+0x40], xmm7	;v14,15
vpaddq xmm4, xmm4, xmm6
	vpextrq [rcx+EH.mids+0x70], xmm4, 1	;v9
vpaddq xmm5, xmm5, xmm7
	vmovdqa xword [rcx+EH.mids+0x50], xmm5	;v10,11
vpxor xmm2, xmm2, xmm4
vpxor xmm3, xmm3, xmm5
vpaddq xmm8, xmm2, xmm2
vpsrlq xmm2, xmm2, 63
vpor xmm8, xmm2, xmm8		;xmm8 takes xmm2
vpaddq xmm2, xmm3, xmm3		;xmm2 is temp
vpsrlq xmm3, xmm3, 63
vpor xmm3, xmm3, xmm2

vpalignr xmm2, xmm3, xmm8, 8	;xmm2 resume
	vmovdqa xword [rcx+EH.mids+0x20], xmm2	;v5,6
vpsrldq xmm3, xmm3, 8
	vmovq [rcx+EH.mids+0x68], xmm3		;v7
vpsrldq xmm7, xmm6, 8
vpaddq xmm0, xmm0, xmm2
	vpextrq [rcx+EH.mids+0x30], xmm0, 1	;v1
vpaddq xmm1, xmm1, xmm3
	vmovq [rcx+EH.mids+0x78], xmm1		;v2
vpxor xmm7, xmm7, xmm1
vpshufd xmm7, xmm7, 0xb1
	vmovq [rcx+EH.mids+0x38], xmm7		;v13

vmovdqa xmm6, [rsp+0x180]
vmovdqa xmm7, [rsp+0x190]
vmovdqa xmm8, [rsp+0x1a0]
vmovdqa xmm9, [rsp+0x1b0]
vmovdqa xmm10, [rsp+0x1c0]
vmovdqa xmm11, [rsp+0x1d0]
add rsp, 0x1f8
ret

align 16
_ProcBlakeMsgSched:
;rdx=src
;r8=dst
;r9=sigma table
xor r10d, r10d
_LoopBlakeMsgSched:
movzx eax, byte [r9+r10]
mov rax, [rdx+rax*8]
mov [r8+r10*8], rax
add r10d, 1
cmp r10d, 16
jb _LoopBlakeMsgSched
ret

align 16
_ProcBlakeRound:
vpaddq xmm0, xmm0, xmm2
vpaddq xmm1, xmm1, xmm3
vpaddq xmm0, xmm0, [r8]
vpaddq xmm1, xmm1, [r8+0x10]
vpxor xmm6, xmm6, xmm0
vpxor xmm7, xmm7, xmm1
vpshufd xmm6, xmm6, 0xb1
vpshufd xmm7, xmm7, 0xb1
vpaddq xmm4, xmm4, xmm6
vpaddq xmm5, xmm5, xmm7
vpxor xmm2, xmm2, xmm4
vpxor xmm3, xmm3, xmm5
vpshufb xmm2, xmm2, xmm10
vpshufb xmm3, xmm3, xmm10
vpaddq xmm0, xmm0, xmm2
vpaddq xmm1, xmm1, xmm3
vpaddq xmm0, xmm0, [r8+0x20]
vpaddq xmm1, xmm1, [r8+0x30]
vpxor xmm6, xmm6, xmm0
vpxor xmm7, xmm7, xmm1
vpshufb xmm9, xmm6, xmm11	;xmm9 takes xmm6
vpshufb xmm7, xmm7, xmm11
vpaddq xmm4, xmm4, xmm9
vpaddq xmm5, xmm5, xmm7
vpxor xmm2, xmm2, xmm4
vpxor xmm3, xmm3, xmm5
vpaddq xmm8, xmm2, xmm2
vpsrlq xmm2, xmm2, 63
vpor xmm8, xmm2, xmm8		;xmm8 takes xmm2
vpaddq xmm2, xmm3, xmm3		;xmm2 is temp
vpsrlq xmm3, xmm3, 63
vpor xmm3, xmm3, xmm2

vpalignr xmm2, xmm3, xmm8, 8	;xmm2 resume
vpalignr xmm3, xmm8, xmm3, 8
vpalignr xmm6, xmm9, xmm7, 8	;xmm6 resume
vpalignr xmm7, xmm7, xmm9, 8
vpaddq xmm0, xmm0, xmm2
vpaddq xmm1, xmm1, xmm3
vpaddq xmm0, xmm0, [r8+0x40]
vpaddq xmm1, xmm1, [r8+0x50]
vpxor xmm6, xmm6, xmm0
vpxor xmm7, xmm7, xmm1
vpshufd xmm6, xmm6, 0xb1
vpshufd xmm7, xmm7, 0xb1
vpaddq xmm5, xmm5, xmm6
vpaddq xmm4, xmm4, xmm7
vpxor xmm2, xmm2, xmm5
vpxor xmm3, xmm3, xmm4
vpshufb xmm2, xmm2, xmm10
vpshufb xmm3, xmm3, xmm10
vpaddq xmm0, xmm0, xmm2
vpaddq xmm1, xmm1, xmm3
vpaddq xmm0, xmm0, [r8+0x60]
vpaddq xmm1, xmm1, [r8+0x70]
vpxor xmm6, xmm6, xmm0
vpxor xmm7, xmm7, xmm1
vpshufb xmm9, xmm6, xmm11	;xmm9 takes xmm6
vpshufb xmm7, xmm7, xmm11
vpaddq xmm5, xmm5, xmm9
vpaddq xmm4, xmm4, xmm7
vpxor xmm2, xmm2, xmm5
vpxor xmm3, xmm3, xmm4
vpaddq xmm8, xmm2, xmm2
vpsrlq xmm2, xmm2, 63
vpor xmm8, xmm2, xmm8		;xmm8 takes xmm2
vpaddq xmm2, xmm3, xmm3		;xmm2 is temp
vpsrlq xmm3, xmm3, 63
vpor xmm3, xmm3, xmm2
vpalignr xmm2, xmm8, xmm3, 8	;xmm2 resume
vpalignr xmm3, xmm3, xmm8, 8
vpalignr xmm6, xmm7, xmm9, 8	;xmm6 resume
vpalignr xmm7, xmm9, xmm7, 8
ret
