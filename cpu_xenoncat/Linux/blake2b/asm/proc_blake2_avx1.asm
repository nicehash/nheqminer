;void Blake2Run2(unsigned char *hashout, void *midstate, uint32_t indexctr);
;hashout: hash output buffer: 2*64 bytes
;midstate: 256 bytes from Blake2PrepareMidstate2
;indexctr: For n=200, k=9: {0, 2, 4, ..., 1048574}

include "macro_blake2b_avx1.asm"

Blake2Run2:
mov rax, rsp
sub rsp, 0x28
and rsp, -32
mov [rsp+0x20], rax

mov [rsi+0xd4], edx
add edx, 1
mov [rsi+0xdc], edx

Blake2beq2of2 rsi, rsi+0xc0

vpunpcklqdq xmm8, xmm0, xmm1
vpunpckhqdq xmm1, xmm0, xmm1
vpunpcklqdq xmm10, xmm2, xmm3
vpunpckhqdq xmm3, xmm2, xmm3
vpunpcklqdq xmm12, xmm4, xmm5
vpunpckhqdq xmm5, xmm4, xmm5
vpunpcklqdq xmm14, xmm6, xmm7
vpunpckhqdq xmm7, xmm6, xmm7

vmovdqa [rdi], xmm8
vmovdqa [rdi+0x10], xmm10
vmovdqa [rdi+0x20], xmm12
vmovdqa [rdi+0x30], xmm14
vmovdqa [rdi+0x40], xmm1
vmovdqa [rdi+0x50], xmm3
vmovdqa [rdi+0x60], xmm5
vmovdqa [rdi+0x70], xmm7

mov rsp, [rsp+0x20]
ret
