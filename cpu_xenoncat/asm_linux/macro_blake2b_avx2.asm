macro hR0 m0,m1,m2,m3,m4,m5,m6,m7,lim,src
{
vpaddq ymm0,ymm0,ymm4
vpaddq ymm1,ymm1,ymm5
vpaddq ymm2,ymm2,ymm6
vpaddq ymm3,ymm3,ymm7
if m0<lim
vpaddq ymm0,ymm0, yword [src+m0*32]
end if
if m1<lim
vpaddq ymm1,ymm1, yword [src+m1*32]
end if
if m2<lim
vpaddq ymm2,ymm2, yword [src+m2*32]
end if
if m3<lim
vpaddq ymm3,ymm3, yword [src+m3*32]
end if
vpxor ymm12,ymm12,ymm0
vpxor ymm13,ymm13,ymm1
vpxor ymm14,ymm14,ymm2
vpxor ymm15,ymm15,ymm3
vpshufd ymm12,ymm12,0xB1
vpshufd ymm13,ymm13,0xB1
vpshufd ymm14,ymm14,0xB1
vpshufd ymm15,ymm15,0xB1
vpaddq ymm8,ymm8,ymm12
vpaddq ymm9,ymm9,ymm13
vpaddq ymm10,ymm10,ymm14
vpaddq ymm11,ymm11,ymm15
vpxor ymm4,ymm4,ymm8
vpxor ymm5,ymm5,ymm9
vpxor ymm6,ymm6,ymm10
vpxor ymm7,ymm7,ymm11
vmovdqa [rsp], ymm8
vbroadcasti128 ymm8, xword [xshufb_ror24]
vpshufb ymm4,ymm4,ymm8
vpshufb ymm5,ymm5,ymm8
vpshufb ymm6,ymm6,ymm8
vpshufb ymm7,ymm7,ymm8
vmovdqa ymm8, [rsp]

vpaddq ymm0,ymm0,ymm4
vpaddq ymm1,ymm1,ymm5
vpaddq ymm2,ymm2,ymm6
vpaddq ymm3,ymm3,ymm7
if m4<lim
vpaddq ymm0,ymm0, yword [src+m4*32]
end if
if m5<lim
vpaddq ymm1,ymm1, yword [src+m5*32]
end if
if m6<lim
vpaddq ymm2,ymm2, yword [src+m6*32]
end if
if m7<lim
vpaddq ymm3,ymm3, yword [src+m7*32]
end if
vpxor ymm12,ymm12,ymm0
vpxor ymm13,ymm13,ymm1
vpxor ymm14,ymm14,ymm2
vpxor ymm15,ymm15,ymm3
vmovdqa [rsp], ymm0
vbroadcasti128 ymm0, xword [xshufb_ror16]
vpshufb ymm12,ymm12,ymm0
vpshufb ymm13,ymm13,ymm0
vpshufb ymm14,ymm14,ymm0
vpshufb ymm15,ymm15,ymm0
vpaddq ymm8,ymm8,ymm12
vpaddq ymm9,ymm9,ymm13
vpaddq ymm10,ymm10,ymm14
vpaddq ymm11,ymm11,ymm15
vpxor ymm4,ymm4,ymm8
vpxor ymm5,ymm5,ymm9
vpxor ymm6,ymm6,ymm10
vpxor ymm7,ymm7,ymm11

vpaddq ymm0,ymm4,ymm4
vpsrlq ymm4,ymm4,63
vpor ymm4,ymm4,ymm0
vpaddq ymm0,ymm5,ymm5
vpsrlq ymm5,ymm5,63
vpor ymm5,ymm5,ymm0
vpaddq ymm0,ymm6,ymm6
vpsrlq ymm6,ymm6,63
vpor ymm6,ymm6,ymm0
vpaddq ymm0,ymm7,ymm7
vpsrlq ymm7,ymm7,63
vpor ymm7,ymm7,ymm0

vmovdqa ymm0, [rsp]
}

macro hR1 m0,m1,m2,m3,m4,m5,m6,m7,lim,src
{
vpaddq ymm0,ymm0,ymm5
vpaddq ymm1,ymm1,ymm6
vpaddq ymm2,ymm2,ymm7
vpaddq ymm3,ymm3,ymm4
if m0<lim
vpaddq ymm0,ymm0, yword [src+m0*32]
end if
if m1<lim
vpaddq ymm1,ymm1, yword [src+m1*32]
end if
if m2<lim
vpaddq ymm2,ymm2, yword [src+m2*32]
end if
if m3<lim
vpaddq ymm3,ymm3, yword [src+m3*32]
end if
vpxor ymm15,ymm15,ymm0
vpxor ymm12,ymm12,ymm1
vpxor ymm13,ymm13,ymm2
vpxor ymm14,ymm14,ymm3
vpshufd ymm15,ymm15,0xB1
vpshufd ymm12,ymm12,0xB1
vpshufd ymm13,ymm13,0xB1
vpshufd ymm14,ymm14,0xB1
vpaddq ymm10,ymm10,ymm15
vpaddq ymm11,ymm11,ymm12
vpaddq ymm8,ymm8,ymm13
vpaddq ymm9,ymm9,ymm14
vpxor ymm5,ymm5,ymm10
vpxor ymm6,ymm6,ymm11
vpxor ymm7,ymm7,ymm8
vpxor ymm4,ymm4,ymm9
vmovdqa [rsp], ymm10
vbroadcasti128 ymm10, xword [xshufb_ror24]
vpshufb ymm5,ymm5,ymm10
vpshufb ymm6,ymm6,ymm10
vpshufb ymm7,ymm7,ymm10
vpshufb ymm4,ymm4,ymm10
vmovdqa ymm10, [rsp]

vpaddq ymm0,ymm0,ymm5
vpaddq ymm1,ymm1,ymm6
vpaddq ymm2,ymm2,ymm7
vpaddq ymm3,ymm3,ymm4
if m4<lim
vpaddq ymm0,ymm0, yword [src+m4*32]
end if
if m5<lim
vpaddq ymm1,ymm1, yword [src+m5*32]
end if
if m6<lim
vpaddq ymm2,ymm2, yword [src+m6*32]
end if
if m7<lim
vpaddq ymm3,ymm3, yword [src+m7*32]
end if
vpxor ymm15,ymm15,ymm0
vpxor ymm12,ymm12,ymm1
vpxor ymm13,ymm13,ymm2
vpxor ymm14,ymm14,ymm3
vmovdqa [rsp], ymm0
vbroadcasti128 ymm0, xword [xshufb_ror16]
vpshufb ymm15,ymm15,ymm0
vpshufb ymm12,ymm12,ymm0
vpshufb ymm13,ymm13,ymm0
vpshufb ymm14,ymm14,ymm0
vpaddq ymm10,ymm10,ymm15
vpaddq ymm11,ymm11,ymm12
vpaddq ymm8,ymm8,ymm13
vpaddq ymm9,ymm9,ymm14
vpxor ymm5,ymm5,ymm10
vpxor ymm6,ymm6,ymm11
vpxor ymm7,ymm7,ymm8
vpxor ymm4,ymm4,ymm9

vpaddq ymm0,ymm5,ymm5
vpsrlq ymm5,ymm5,63
vpor ymm5,ymm5,ymm0
vpaddq ymm0,ymm6,ymm6
vpsrlq ymm6,ymm6,63
vpor ymm6,ymm6,ymm0
vpaddq ymm0,ymm7,ymm7
vpsrlq ymm7,ymm7,63
vpor ymm7,ymm7,ymm0
vpaddq ymm0,ymm4,ymm4
vpsrlq ymm4,ymm4,63
vpor ymm4,ymm4,ymm0

vmovdqa ymm0, [rsp]
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
vpbroadcastq ymm0, qword [mids]
vpaddq ymm0,ymm0, yword [src+1*32]
vpbroadcastq ymm12, qword [mids+0x08]
vpxor ymm12,ymm12,ymm0
vbroadcasti128 ymm2, xword [xshufb_ror16]	;ymm2 is temp
vpshufb ymm12,ymm12,ymm2
vpbroadcastq ymm8, qword [mids+0x10]
vpaddq ymm8,ymm8,ymm12
vpbroadcastq ymm4, qword [mids+0x18]
vpxor ymm4,ymm4,ymm8
vpaddq ymm2,ymm4,ymm4	;ymm2 is temp
vpsrlq ymm4,ymm4,63
vpor ymm4,ymm4,ymm2

vpbroadcastq ymm5, qword [mids+0x20]
vpaddq ymm0,ymm0,ymm5
vpbroadcastq ymm1, qword [mids+0x30]
vpxor ymm12,ymm12,ymm1
vpshufd ymm12,ymm12,0xB1
vpbroadcastq ymm13, qword [mids+0x38]
vpaddq ymm8,ymm8,ymm13
vpbroadcastq ymm3, qword [mids+0x60]
vpaddq ymm3,ymm3,ymm4
vpbroadcastq ymm15, qword [mids+0x48]
vpxor ymm15,ymm15,ymm0
vpshufd ymm15,ymm15,0xB1
vpbroadcastq ymm11, qword [mids+0x58]
vpaddq ymm11,ymm11,ymm12
vpbroadcastq ymm7, qword [mids+0x68]
vpxor ymm7,ymm7,ymm8
vpbroadcastq ymm14, qword [mids+0x40]
vpxor ymm14,ymm14,ymm3
vpshufd ymm14,ymm14,0xB1
vpbroadcastq ymm10, qword [mids+0x50]
vpaddq ymm10,ymm10,ymm15
vpbroadcastq ymm6, qword [mids+0x28]
vpxor ymm6,ymm6,ymm11
vpbroadcastq ymm9, qword [mids+0x70]
vpaddq ymm9,ymm9,ymm14
vpxor ymm5,ymm5,ymm10
vpxor ymm4,ymm4,ymm9
vbroadcasti128 ymm2, xword [xshufb_ror24]	;ymm2 is temp
vpshufb ymm5,ymm5,ymm2
vpshufb ymm6,ymm6,ymm2
vpshufb ymm7,ymm7,ymm2
vpshufb ymm4,ymm4,ymm2
vpbroadcastq ymm2, qword [mids+0x78]

vpaddq ymm0,ymm0,ymm5
vpaddq ymm1,ymm1,ymm6
vpaddq ymm2,ymm2,ymm7
vpaddq ymm3,ymm3,ymm4
vpxor ymm15,ymm15,ymm0
vpxor ymm12,ymm12,ymm1
vpxor ymm13,ymm13,ymm2
vpxor ymm14,ymm14,ymm3
vmovdqa [rsp], ymm0
vbroadcasti128 ymm0, xword [xshufb_ror16]
vpshufb ymm15,ymm15,ymm0
vpshufb ymm12,ymm12,ymm0
vpshufb ymm13,ymm13,ymm0
vpshufb ymm14,ymm14,ymm0
vpaddq ymm10,ymm10,ymm15
vpaddq ymm11,ymm11,ymm12
vpaddq ymm8,ymm8,ymm13
vpaddq ymm9,ymm9,ymm14
vpxor ymm5,ymm5,ymm10
vpxor ymm6,ymm6,ymm11
vpxor ymm7,ymm7,ymm8
vpxor ymm4,ymm4,ymm9
vpaddq ymm0,ymm5,ymm5
vpsrlq ymm5,ymm5,63
vpor ymm5,ymm5,ymm0
vpaddq ymm0,ymm6,ymm6
vpsrlq ymm6,ymm6,63
vpor ymm6,ymm6,ymm0
vpaddq ymm0,ymm7,ymm7
vpsrlq ymm7,ymm7,63
vpor ymm7,ymm7,ymm0
vpaddq ymm0,ymm4,ymm4
vpsrlq ymm4,ymm4,63
vpor ymm4,ymm4,ymm0
vmovdqa ymm0, [rsp]

Blake2bRounds2 2,src

vpxor ymm0, ymm0, ymm8
vpxor ymm1, ymm1, ymm9
vpxor ymm2, ymm2, ymm10
vpxor ymm3, ymm3, ymm11
vpxor ymm4, ymm4, ymm12
vpxor ymm5, ymm5, ymm13
vpxor ymm6, ymm6, ymm14
;vpxor ymm7, ymm7, ymm15
vpbroadcastq ymm8, qword [mids+0x80]
vpbroadcastq ymm9, qword [mids+0x88]
vpbroadcastq ymm10, qword [mids+0x90]
vpbroadcastq ymm11, qword [mids+0x98]
vpbroadcastq ymm12, qword [mids+0xa0]
vpbroadcastq ymm13, qword [mids+0xa8]
vpbroadcastq ymm14, qword [mids+0xb0]
;vpbroadcastq ymm15, qword [mids+0xb8]
vpxor ymm0, ymm0, ymm8
vpxor ymm1, ymm1, ymm9
vpxor ymm2, ymm2, ymm10
vpxor ymm3, ymm3, ymm11
vpxor ymm4, ymm4, ymm12
vpxor ymm5, ymm5, ymm13
vpxor ymm6, ymm6, ymm14
;vpxor ymm7, ymm7, ymm15
}
