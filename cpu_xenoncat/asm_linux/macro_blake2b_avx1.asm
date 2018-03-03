macro hR0 m0,m1,m2,m3,m4,m5,m6,m7,lim,src
{
vpaddq xmm0,xmm0,xmm4
vpaddq xmm1,xmm1,xmm5
vpaddq xmm2,xmm2,xmm6
vpaddq xmm3,xmm3,xmm7
if m0<lim
vpaddq xmm0,xmm0, xword [src+m0*16]
end if
if m1<lim
vpaddq xmm1,xmm1, xword [src+m1*16]
end if
if m2<lim
vpaddq xmm2,xmm2, xword [src+m2*16]
end if
if m3<lim
vpaddq xmm3,xmm3, xword [src+m3*16]
end if
vpxor xmm12,xmm12,xmm0
vpxor xmm13,xmm13,xmm1
vpxor xmm14,xmm14,xmm2
vpxor xmm15,xmm15,xmm3
vpshufd xmm12,xmm12,0xB1
vpshufd xmm13,xmm13,0xB1
vpshufd xmm14,xmm14,0xB1
vpshufd xmm15,xmm15,0xB1
vpaddq xmm8,xmm8,xmm12
vpaddq xmm9,xmm9,xmm13
vpaddq xmm10,xmm10,xmm14
vpaddq xmm11,xmm11,xmm15
vpxor xmm4,xmm4,xmm8
vpxor xmm5,xmm5,xmm9
vpxor xmm6,xmm6,xmm10
vpxor xmm7,xmm7,xmm11
vmovdqa [rsp], xmm8
vmovdqa xmm8, xword [xshufb_ror24]
vpshufb xmm4,xmm4,xmm8
vpshufb xmm5,xmm5,xmm8
vpshufb xmm6,xmm6,xmm8
vpshufb xmm7,xmm7,xmm8
vmovdqa xmm8, [rsp]

vpaddq xmm0,xmm0,xmm4
vpaddq xmm1,xmm1,xmm5
vpaddq xmm2,xmm2,xmm6
vpaddq xmm3,xmm3,xmm7
if m4<lim
vpaddq xmm0,xmm0, xword [src+m4*16]
end if
if m5<lim
vpaddq xmm1,xmm1, xword [src+m5*16]
end if
if m6<lim
vpaddq xmm2,xmm2, xword [src+m6*16]
end if
if m7<lim
vpaddq xmm3,xmm3, xword [src+m7*16]
end if
vpxor xmm12,xmm12,xmm0
vpxor xmm13,xmm13,xmm1
vpxor xmm14,xmm14,xmm2
vpxor xmm15,xmm15,xmm3
vmovdqa [rsp], xmm0
vmovdqa xmm0, xword [xshufb_ror16]
vpshufb xmm12,xmm12,xmm0
vpshufb xmm13,xmm13,xmm0
vpshufb xmm14,xmm14,xmm0
vpshufb xmm15,xmm15,xmm0
vpaddq xmm8,xmm8,xmm12
vpaddq xmm9,xmm9,xmm13
vpaddq xmm10,xmm10,xmm14
vpaddq xmm11,xmm11,xmm15
vpxor xmm4,xmm4,xmm8
vpxor xmm5,xmm5,xmm9
vpxor xmm6,xmm6,xmm10
vpxor xmm7,xmm7,xmm11

vpaddq xmm0,xmm4,xmm4
vpsrlq xmm4,xmm4,63
vpor xmm4,xmm4,xmm0
vpaddq xmm0,xmm5,xmm5
vpsrlq xmm5,xmm5,63
vpor xmm5,xmm5,xmm0
vpaddq xmm0,xmm6,xmm6
vpsrlq xmm6,xmm6,63
vpor xmm6,xmm6,xmm0
vpaddq xmm0,xmm7,xmm7
vpsrlq xmm7,xmm7,63
vpor xmm7,xmm7,xmm0

vmovdqa xmm0, [rsp]
}

macro hR1 m0,m1,m2,m3,m4,m5,m6,m7,lim,src
{
vpaddq xmm0,xmm0,xmm5
vpaddq xmm1,xmm1,xmm6
vpaddq xmm2,xmm2,xmm7
vpaddq xmm3,xmm3,xmm4
if m0<lim
vpaddq xmm0,xmm0, xword [src+m0*16]
end if
if m1<lim
vpaddq xmm1,xmm1, xword [src+m1*16]
end if
if m2<lim
vpaddq xmm2,xmm2, xword [src+m2*16]
end if
if m3<lim
vpaddq xmm3,xmm3, xword [src+m3*16]
end if
vpxor xmm15,xmm15,xmm0
vpxor xmm12,xmm12,xmm1
vpxor xmm13,xmm13,xmm2
vpxor xmm14,xmm14,xmm3
vpshufd xmm15,xmm15,0xB1
vpshufd xmm12,xmm12,0xB1
vpshufd xmm13,xmm13,0xB1
vpshufd xmm14,xmm14,0xB1
vpaddq xmm10,xmm10,xmm15
vpaddq xmm11,xmm11,xmm12
vpaddq xmm8,xmm8,xmm13
vpaddq xmm9,xmm9,xmm14
vpxor xmm5,xmm5,xmm10
vpxor xmm6,xmm6,xmm11
vpxor xmm7,xmm7,xmm8
vpxor xmm4,xmm4,xmm9
vmovdqa [rsp], xmm10
vmovdqa xmm10, xword [xshufb_ror24]
vpshufb xmm5,xmm5,xmm10
vpshufb xmm6,xmm6,xmm10
vpshufb xmm7,xmm7,xmm10
vpshufb xmm4,xmm4,xmm10
vmovdqa xmm10, [rsp]

vpaddq xmm0,xmm0,xmm5
vpaddq xmm1,xmm1,xmm6
vpaddq xmm2,xmm2,xmm7
vpaddq xmm3,xmm3,xmm4
if m4<lim
vpaddq xmm0,xmm0, xword [src+m4*16]
end if
if m5<lim
vpaddq xmm1,xmm1, xword [src+m5*16]
end if
if m6<lim
vpaddq xmm2,xmm2, xword [src+m6*16]
end if
if m7<lim
vpaddq xmm3,xmm3, xword [src+m7*16]
end if
vpxor xmm15,xmm15,xmm0
vpxor xmm12,xmm12,xmm1
vpxor xmm13,xmm13,xmm2
vpxor xmm14,xmm14,xmm3
vmovdqa [rsp], xmm0
vmovdqa xmm0, xword [xshufb_ror16]
vpshufb xmm15,xmm15,xmm0
vpshufb xmm12,xmm12,xmm0
vpshufb xmm13,xmm13,xmm0
vpshufb xmm14,xmm14,xmm0
vpaddq xmm10,xmm10,xmm15
vpaddq xmm11,xmm11,xmm12
vpaddq xmm8,xmm8,xmm13
vpaddq xmm9,xmm9,xmm14
vpxor xmm5,xmm5,xmm10
vpxor xmm6,xmm6,xmm11
vpxor xmm7,xmm7,xmm8
vpxor xmm4,xmm4,xmm9

vpaddq xmm0,xmm5,xmm5
vpsrlq xmm5,xmm5,63
vpor xmm5,xmm5,xmm0
vpaddq xmm0,xmm6,xmm6
vpsrlq xmm6,xmm6,63
vpor xmm6,xmm6,xmm0
vpaddq xmm0,xmm7,xmm7
vpsrlq xmm7,xmm7,63
vpor xmm7,xmm7,xmm0
vpaddq xmm0,xmm4,xmm4
vpsrlq xmm4,xmm4,63
vpor xmm4,xmm4,xmm0

vmovdqa xmm0, [rsp]
}

macro Blake2bRounds2 lim,src
{
;ROUND 0
;hR0 0,2,4,6,1,3,5,7,lim,src
;hR1 8,10,12,14,9,11,13,15,lim,src

;ROUND 1
hR0 14,4,9,13,10,8,15,6,lim,src
hR1 1,0,11,5,12,2,7,3,lim,src

;ROUND 2
hR0 11,12,5,15,8,0,2,13,lim,src
hR1 10,3,7,9,14,6,1,4,lim,src

;ROUND 3
hR0 7,3,13,11,9,1,12,14,lim,src
hR1 2,5,4,15,6,10,0,8,lim,src

;ROUND 4
hR0 9,5,2,10,0,7,4,15,lim,src
hR1 14,11,6,3,1,12,8,13,lim,src

;ROUND 5
hR0 2,6,0,8,12,10,11,3,lim,src
hR1 4,7,15,1,13,5,14,9,lim,src

;ROUND 6
hR0 12,1,14,4,5,15,13,10,lim,src
hR1 0,6,9,8,7,3,2,11,lim,src

;ROUND 7
hR0 13,7,12,3,11,14,1,9,lim,src
hR1 5,15,8,2,0,4,6,10,lim,src

;ROUND 8
hR0 6,14,11,0,15,9,3,8,lim,src
hR1 12,13,1,10,2,7,4,5,lim,src

;ROUND 9
hR0 10,8,7,1,2,4,6,5,lim,src
hR1 15,9,3,13,11,14,12,0,lim,src

;ROUND 10
hR0 0,2,4,6,1,3,5,7,lim,src
hR1 8,10,12,14,9,11,13,15,lim,src

;ROUND 11
hR0 14,4,9,13,10,8,15,6,lim,src
hR1 1,0,11,5,12,2,7,3,lim,src
}

macro Blake2beq2of2 mids, src
{
vmovddup xmm0, qword [mids]
vpaddq xmm0,xmm0, xword [src+1*16]
vmovddup xmm12, qword [mids+0x08]
vpxor xmm12,xmm12,xmm0
vpshufb xmm12,xmm12, xword [xshufb_ror16]
vmovddup xmm8, qword [mids+0x10]
vpaddq xmm8,xmm8,xmm12
vmovddup xmm4, qword [mids+0x18]
vpxor xmm4,xmm4,xmm8
vpaddq xmm2,xmm4,xmm4	;xmm2 is temp
vpsrlq xmm4,xmm4,63
vpor xmm4,xmm4,xmm2

vmovddup xmm5, qword [mids+0x20]
vpaddq xmm0,xmm0,xmm5
vmovddup xmm1, qword [mids+0x30]
vpxor xmm12,xmm12,xmm1
vpshufd xmm12,xmm12,0xB1
vmovddup xmm13, qword [mids+0x38]
vpaddq xmm8,xmm8,xmm13
vmovddup xmm3, qword [mids+0x60]
vpaddq xmm3,xmm3,xmm4
vmovddup xmm15, qword [mids+0x48]
vpxor xmm15,xmm15,xmm0
vpshufd xmm15,xmm15,0xB1
vmovddup xmm11, qword [mids+0x58]
vpaddq xmm11,xmm11,xmm12
vmovddup xmm7, qword [mids+0x68]
vpxor xmm7,xmm7,xmm8
vmovddup xmm14, qword [mids+0x40]
vpxor xmm14,xmm14,xmm3
vpshufd xmm14,xmm14,0xB1
vmovddup xmm10, qword [mids+0x50]
vpaddq xmm10,xmm10,xmm15
vmovddup xmm6, qword [mids+0x28]
vpxor xmm6,xmm6,xmm11
vmovddup xmm9, qword [mids+0x70]
vpaddq xmm9,xmm9,xmm14
vpxor xmm5,xmm5,xmm10
vpxor xmm4,xmm4,xmm9
vmovdqa xmm2, xword [xshufb_ror24]	;xmm2 is temp
vpshufb xmm5,xmm5,xmm2
vpshufb xmm6,xmm6,xmm2
vpshufb xmm7,xmm7,xmm2
vpshufb xmm4,xmm4,xmm2
vmovddup xmm2, qword [mids+0x78]

vpaddq xmm0,xmm0,xmm5
vpaddq xmm1,xmm1,xmm6
vpaddq xmm2,xmm2,xmm7
vpaddq xmm3,xmm3,xmm4
vpxor xmm15,xmm15,xmm0
vpxor xmm12,xmm12,xmm1
vpxor xmm13,xmm13,xmm2
vpxor xmm14,xmm14,xmm3
vmovdqa [rsp], xmm0
vmovdqa xmm0, xword [xshufb_ror16]
vpshufb xmm15,xmm15,xmm0
vpshufb xmm12,xmm12,xmm0
vpshufb xmm13,xmm13,xmm0
vpshufb xmm14,xmm14,xmm0
vpaddq xmm10,xmm10,xmm15
vpaddq xmm11,xmm11,xmm12
vpaddq xmm8,xmm8,xmm13
vpaddq xmm9,xmm9,xmm14
vpxor xmm5,xmm5,xmm10
vpxor xmm6,xmm6,xmm11
vpxor xmm7,xmm7,xmm8
vpxor xmm4,xmm4,xmm9
vpaddq xmm0,xmm5,xmm5
vpsrlq xmm5,xmm5,63
vpor xmm5,xmm5,xmm0
vpaddq xmm0,xmm6,xmm6
vpsrlq xmm6,xmm6,63
vpor xmm6,xmm6,xmm0
vpaddq xmm0,xmm7,xmm7
vpsrlq xmm7,xmm7,63
vpor xmm7,xmm7,xmm0
vpaddq xmm0,xmm4,xmm4
vpsrlq xmm4,xmm4,63
vpor xmm4,xmm4,xmm0
vmovdqa xmm0, [rsp]

Blake2bRounds2 2,src

vpxor xmm0, xmm0, xmm8
vpxor xmm1, xmm1, xmm9
vpxor xmm2, xmm2, xmm10
vpxor xmm3, xmm3, xmm11
vpxor xmm4, xmm4, xmm12
vpxor xmm5, xmm5, xmm13
vpxor xmm6, xmm6, xmm14
;vpxor xmm7, xmm7, xmm15
vmovddup xmm8, qword [mids+0x80]
vmovddup xmm9, qword [mids+0x88]
vmovddup xmm10, qword [mids+0x90]
vmovddup xmm11, qword [mids+0x98]
vmovddup xmm12, qword [mids+0xa0]
vmovddup xmm13, qword [mids+0xa8]
vmovddup xmm14, qword [mids+0xb0]
;vmovddup xmm15, qword [mids+0xb8]
vpxor xmm0, xmm0, xmm8
vpxor xmm1, xmm1, xmm9
vpxor xmm2, xmm2, xmm10
vpxor xmm3, xmm3, xmm11
vpxor xmm4, xmm4, xmm12
vpxor xmm5, xmm5, xmm13
vpxor xmm6, xmm6, xmm14
;vpxor xmm7, xmm7, xmm15
}
