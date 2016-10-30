;void Blake2Run4(unsigned char *hashout, void *midstate, uint32_t indexctr);
;hashout: hash output buffer: 4*64 bytes
;midstate: 256 bytes from Blake2PrepareMidstate4
;indexctr: For n=200, k=9: {0, 4, 8, ..., 1048572}

include "macro_blake2b_avx2.asm"

Blake2Run4:
mov rax, rsp
sub rsp, 0x28
and rsp, -32
mov [rsp+0x20], rax

vmovd xmm0, edx		;indexctr
vpbroadcastd ymm0, xmm0
vpaddd ymm0, ymm0, yword [yctrinit]
vpblendd ymm0, ymm0, yword [rsi+0xe0], 0x55
vmovdqa yword [rsi+0xe0], ymm0

Blake2beq2of2 rsi, rsi+0xc0

vpunpcklqdq ymm8, ymm0, ymm1
vpunpckhqdq ymm9, ymm0, ymm1
vpunpcklqdq ymm10, ymm2, ymm3
vpunpckhqdq ymm11, ymm2, ymm3
vpunpcklqdq ymm12, ymm4, ymm5
vpunpckhqdq ymm13, ymm4, ymm5
vpunpcklqdq ymm14, ymm6, ymm7
vpunpckhqdq ymm15, ymm6, ymm7
vperm2i128 ymm0, ymm8, ymm10, 0x20
vperm2i128 ymm1, ymm12, ymm14, 0x20
vperm2i128 ymm2, ymm9, ymm11, 0x20
vperm2i128 ymm3, ymm13, ymm15, 0x20
vperm2i128 ymm4, ymm8, ymm10, 0x31
vperm2i128 ymm5, ymm12, ymm14, 0x31
vperm2i128 ymm6, ymm9, ymm11, 0x31
vperm2i128 ymm7, ymm13, ymm15, 0x31

vmovdqa [rdi], ymm0
vmovdqa [rdi+0x20], ymm1
vmovdqa [rdi+0x40], ymm2
vmovdqa [rdi+0x60], ymm3
vmovdqa [rdi+0x80], ymm4
vmovdqa [rdi+0xa0], ymm5
vmovdqa [rdi+0xc0], ymm6
vmovdqa [rdi+0xe0], ymm7

mov rsp, [rsp+0x20]
ret
