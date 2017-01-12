EhPrepare:
sub rsp, 0x188
vbroadcasti128 ymm6, xword [xshufb_ror24]
vbroadcasti128 ymm7, xword [xshufb_ror16]

vmovdqa ymm0, yword [s0]
vmovdqa ymm1, yword [s4]
vmovdqa ymm2, yword [iv]
vmovdqa ymm3, yword [iv4xor128]

mov r8, rsp
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
cmp r9, r11
jb _LoopEhPrepare1
mov r8, rsp
call _ProcBlakeRound
add r8, 0x80
call _ProcBlakeRound

vpxor ymm0, ymm0, ymm2
vpxor ymm1, ymm1, ymm3
vpxor ymm0, ymm0, yword [s0]
vpxor ymm1, ymm1, yword [s4]
vmovdqa yword [rdi+EH.mids+0x80], ymm0
vmovdqa yword [rdi+EH.mids+0xa0], ymm1
vmovq xmm5, [rsi+0x80]
vpbroadcastq ymm4, xmm5
vmovdqa yword [rdi+EH.mids+0xc0], ymm4

;Begin second message block
vmovdqa ymm2, yword [iv]
vmovdqa ymm3, yword [iv4xor144]	;also loads iv6inverted
vpaddq ymm0, ymm0, ymm1
vpaddq ymm0, ymm0, ymm5		;ymm5[63:0]=message
vpxor ymm3, ymm3, ymm0
vpshufd ymm3, ymm3, 0xb1
	vmovq [rdi+EH.mids+0x08], xmm3	;v12
vpaddq ymm2, ymm2, ymm3
	vmovq [rdi+EH.mids+0x10], xmm2	;v8
vpxor ymm1, ymm1, ymm2
vpshufb ymm1, ymm1, ymm6
	vmovq [rdi+EH.mids+0x18], xmm1	;v4

vpaddq ymm0, ymm0, ymm1
	vmovq [rdi+EH.mids], xmm0	;v0, v3 ready
;add message (nonce, index) to xmm0 here, but we don't have
vpxor ymm3, ymm3, ymm0
vpshufb ymm3, ymm3, ymm7
vextracti128 xmm4, ymm3, 1
	vmovdqa xword [rdi+EH.mids+0x40], xmm4	;v14,15
vpaddq ymm2, ymm2, ymm3
	vpextrq [rdi+EH.mids+0x70], xmm2, 1	;v9
vextracti128 xmm5, ymm2, 1
	vmovdqa xword [rdi+EH.mids+0x50], xmm5	;v10,11
vpxor ymm1, ymm1, ymm2
vpaddq ymm4, ymm1, ymm1
vpsrlq ymm1, ymm1, 63
vpor ymm1, ymm1, ymm4
;Valid:
;    v1  v2  v3
;    v5  v6  v7
;    v9  v10 v11
;    v13 v14 v15
;
;v1 v2 <- v6 v7
;v13 <- v2

vpermq ymm1, ymm1, 0x39
	vmovdqa xword [rdi+EH.mids+0x20], xmm1	;v5,6

vextracti128 xmm4, ymm0, 1
vextracti128 xmm5, ymm1, 1
	vpextrq [rdi+EH.mids+0x60], xmm4, 1	;v3
	vmovq [rdi+EH.mids+0x68], xmm5		;v7

vpsrldq xmm3, xmm3, 8
vpaddq xmm0, xmm0, xmm1
	vpextrq [rdi+EH.mids+0x30], xmm0, 1	;v1
vpaddq xmm4, xmm4, xmm5
	vmovq [rdi+EH.mids+0x78], xmm4		;v2
vpxor xmm3, xmm3, xmm4
vpshufd xmm3, xmm3, 0xb1
	vmovq [rdi+EH.mids+0x38], xmm3		;v13

add rsp, 0x188
ret

align 16
_ProcBlakeMsgSched:
;rsi=src
;r8=dst
;r9=sigma table
xor r10d, r10d
_LoopBlakeMsgSched:
movzx eax, byte [r9+r10]
mov rax, [rsi+rax*8]
mov [r8+r10*8], rax
add r10d, 1
cmp r10d, 16
jb _LoopBlakeMsgSched
ret

align 16
_ProcBlakeRound:
vpaddq ymm0, ymm0, ymm1
vpaddq ymm0, ymm0, [r8]
vpxor ymm3, ymm3, ymm0
vpshufd ymm3, ymm3, 0xb1
vpaddq ymm2, ymm2, ymm3
vpxor ymm1, ymm1, ymm2
vpshufb ymm1, ymm1, ymm6	;ror24
vpaddq ymm0, ymm0, ymm1
vpaddq ymm0, ymm0, [r8+0x20]
vpxor ymm3, ymm3, ymm0
vpshufb ymm3, ymm3, ymm7	;ror16
vpaddq ymm2, ymm2, ymm3
vpxor ymm1, ymm1, ymm2
vpaddq ymm4, ymm1, ymm1
vpsrlq ymm1, ymm1, 63
vpor ymm1, ymm1, ymm4

vpermq ymm1, ymm1, 0x39
vpermq ymm2, ymm2, 0x4e
vpermq ymm3, ymm3, 0x93

vpaddq ymm0, ymm0, ymm1
vpaddq ymm0, ymm0, [r8+0x40]
vpxor ymm3, ymm3, ymm0
vpshufd ymm3, ymm3, 0xb1
vpaddq ymm2, ymm2, ymm3
vpxor ymm1, ymm1, ymm2
vpshufb ymm1, ymm1, ymm6	;ror24
vpaddq ymm0, ymm0, ymm1
vpaddq ymm0, ymm0, [r8+0x60]
vpxor ymm3, ymm3, ymm0
vpshufb ymm3, ymm3, ymm7	;ror16
vpaddq ymm2, ymm2, ymm3
vpxor ymm1, ymm1, ymm2
vpaddq ymm4, ymm1, ymm1
vpsrlq ymm1, ymm1, 63
vpor ymm1, ymm1, ymm4

vpermq ymm1, ymm1, 0x93
vpermq ymm2, ymm2, 0x4e
vpermq ymm3, ymm3, 0x39
ret
