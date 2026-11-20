; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-alignment-versioning=0 < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-align-version-byte-loads=0 < %s | FileCheck %s --check-prefix=NOBL
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-alignment-versioning-elements=0 < %s | FileCheck %s --check-prefix=ONE
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-signed-bytes-half-lanes=0 < %s | FileCheck %s --check-prefix=NOLANE
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-signed-byte-pairs=0 < %s | FileCheck %s --check-prefix=NOPAIR

; A loop over a half row known aligned to 2 bytes only: a copy for a 4-byte
; aligned base (a test of the kernel argument, uniform) whose loads at
; offsets that are multiples of 4 merge into 4-byte (and wider) loads, and
; the loop as it was otherwise (L15-E236). Bytes are left alone.

; CHECK-LABEL: rowsum:
; CHECK: and {{r[0-9]+}}, u0, 3
; CHECK: Loop Header
; CHECK: device_load {{[0-9]}}, i16, x,
; CHECK: Loop Header
; CHECK: device_load {{[0-9]}}, i32, xy,
; CHECK: stop
; OFF-LABEL: rowsum:
; OFF-NOT: and {{r[0-9]+}}, u0, 3
; OFF-NOT: i32, xy
; OFF: stop
define agx_kernel void @rowsum(ptr addrspace(1) %p, ptr addrspace(1) %o) {
entry:
  br label %l
l:
  %i = phi i32 [ 0, %entry ], [ %i1, %l ]
  %s = phi float [ 0.0, %entry ], [ %s4, %l ]
  %b = shl nuw nsw i32 %i, 2
  %z0 = zext i32 %b to i64
  %g0 = getelementptr inbounds half, ptr addrspace(1) %p, i64 %z0
  %v0 = load half, ptr addrspace(1) %g0, align 2
  %g1 = getelementptr inbounds half, ptr addrspace(1) %g0, i64 1
  %v1 = load half, ptr addrspace(1) %g1, align 2
  %g2 = getelementptr inbounds half, ptr addrspace(1) %g0, i64 2
  %v2 = load half, ptr addrspace(1) %g2, align 2
  %g3 = getelementptr inbounds half, ptr addrspace(1) %g0, i64 3
  %v3 = load half, ptr addrspace(1) %g3, align 2
  %g4 = getelementptr inbounds half, ptr addrspace(1) %g0, i64 4
  %v4 = load half, ptr addrspace(1) %g4, align 2
  %g6 = getelementptr inbounds half, ptr addrspace(1) %g0, i64 6
  %v6 = load half, ptr addrspace(1) %g6, align 2
  %e0 = fpext half %v0 to float
  %e1 = fpext half %v1 to float
  %e2 = fpext half %v2 to float
  %e3 = fpext half %v3 to float
  %e4 = fpext half %v4 to float
  %e6 = fpext half %v6 to float
  %s0 = fadd float %s, %e0
  %s1 = fadd float %s0, %e1
  %s2 = fadd float %s1, %e2
  %s3 = fadd float %s2, %e3
  %s5 = fadd float %s3, %e4
  %s4 = fadd float %s5, %e6
  %i1 = add nuw nsw i32 %i, 2
  %c = icmp ult i32 %i1, 256
  br i1 %c, label %l, label %exit
exit:
  store float %s4, ptr addrspace(1) %o, align 4
  ret void
}

; CHECK-LABEL: bytes:
; CHECK-NOT: and {{r[0-9]+}}, u0, 3
; CHECK: stop
define agx_kernel void @bytes(ptr addrspace(1) %p, ptr addrspace(1) %o) {
entry:
  br label %l
l:
  %i = phi i32 [ 0, %entry ], [ %i1, %l ]
  %s = phi i32 [ 0, %entry ], [ %t3, %l ]
  %z0 = zext i32 %i to i64
  %g0 = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %z0
  %v0 = load i8, ptr addrspace(1) %g0, align 1
  %g1 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 1
  %v1 = load i8, ptr addrspace(1) %g1, align 1
  %g2 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 2
  %v2 = load i8, ptr addrspace(1) %g2, align 1
  %g3 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 3
  %v3 = load i8, ptr addrspace(1) %g3, align 1
  %e0 = zext i8 %v0 to i32
  %e1 = zext i8 %v1 to i32
  %e2 = zext i8 %v2 to i32
  %e3 = zext i8 %v3 to i32
  %t0 = add i32 %s, %e0
  %t1 = add i32 %t0, %e1
  %t2 = add i32 %t1, %e2
  %t3 = add i32 %t2, %e3
  %i1 = add nuw nsw i32 %i, 4
  %c = icmp ult i32 %i1, 1024
  br i1 %c, label %l, label %exit
exit:
  store i32 %t3, ptr addrspace(1) %o, align 4
  ret void
}

; Bytes that only go, extended, into integer products and sums (an int8 dot
; product) are versioned too (L15-E560): aligned, four load as one access.
; CHECK-LABEL: bytes_dot:
; CHECK: and {{r[0-9]+}}, {{[ru][0-9]+}}, 3
; CHECK: device_load {{[0-9]}}, i8, xyzw,
; NOBL-LABEL: bytes_dot:
; NOBL-NOT: and {{r[0-9]+}}, {{[ru][0-9]+}}, 3
; NOBL-NOT: i8, xyzw
; NOBL: stop
define agx_kernel void @bytes_dot(ptr addrspace(1) %p, ptr addrspace(1) %q, ptr addrspace(1) %o) {
entry:
  br label %l
l:
  %i = phi i32 [ 0, %entry ], [ %i1, %l ]
  %s = phi i32 [ 0, %entry ], [ %t7, %l ]
  %z0 = zext i32 %i to i64
  %g0 = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %z0
  %h0 = getelementptr inbounds i8, ptr addrspace(1) %q, i64 %z0
  %v0 = load i8, ptr addrspace(1) %g0, align 1
  %w0 = load i8, ptr addrspace(1) %h0, align 1
  %e0 = sext i8 %v0 to i32
  %f0 = sext i8 %w0 to i32
  %m0 = mul nsw i32 %e0, %f0
  %t0 = add i32 %s, %m0
  %g1 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 1
  %h1 = getelementptr inbounds i8, ptr addrspace(1) %h0, i64 1
  %v1 = load i8, ptr addrspace(1) %g1, align 1
  %w1 = load i8, ptr addrspace(1) %h1, align 1
  %e1 = sext i8 %v1 to i32
  %f1 = sext i8 %w1 to i32
  %m1 = mul nsw i32 %e1, %f1
  %t1 = add i32 %t0, %m1
  %g2 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 2
  %h2 = getelementptr inbounds i8, ptr addrspace(1) %h0, i64 2
  %v2 = load i8, ptr addrspace(1) %g2, align 1
  %w2 = load i8, ptr addrspace(1) %h2, align 1
  %e2 = sext i8 %v2 to i32
  %f2 = sext i8 %w2 to i32
  %m2 = mul nsw i32 %e2, %f2
  %t2 = add i32 %t1, %m2
  %g3 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 3
  %h3 = getelementptr inbounds i8, ptr addrspace(1) %h0, i64 3
  %v3 = load i8, ptr addrspace(1) %g3, align 1
  %w3 = load i8, ptr addrspace(1) %h3, align 1
  %e3 = sext i8 %v3 to i32
  %f3 = sext i8 %w3 to i32
  %m3 = mul nsw i32 %e3, %f3
  %t3 = add i32 %t2, %m3
  %g4 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 4
  %h4 = getelementptr inbounds i8, ptr addrspace(1) %h0, i64 4
  %v4 = load i8, ptr addrspace(1) %g4, align 1
  %w4 = load i8, ptr addrspace(1) %h4, align 1
  %e4 = sext i8 %v4 to i32
  %f4 = sext i8 %w4 to i32
  %m4 = mul nsw i32 %e4, %f4
  %t4 = add i32 %t3, %m4
  %g5 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 5
  %h5 = getelementptr inbounds i8, ptr addrspace(1) %h0, i64 5
  %v5 = load i8, ptr addrspace(1) %g5, align 1
  %w5 = load i8, ptr addrspace(1) %h5, align 1
  %e5 = sext i8 %v5 to i32
  %f5 = sext i8 %w5 to i32
  %m5 = mul nsw i32 %e5, %f5
  %t5 = add i32 %t4, %m5
  %g6 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 6
  %h6 = getelementptr inbounds i8, ptr addrspace(1) %h0, i64 6
  %v6 = load i8, ptr addrspace(1) %g6, align 1
  %w6 = load i8, ptr addrspace(1) %h6, align 1
  %e6 = sext i8 %v6 to i32
  %f6 = sext i8 %w6 to i32
  %m6 = mul nsw i32 %e6, %f6
  %t6 = add i32 %t5, %m6
  %g7 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 7
  %h7 = getelementptr inbounds i8, ptr addrspace(1) %h0, i64 7
  %v7 = load i8, ptr addrspace(1) %g7, align 1
  %w7 = load i8, ptr addrspace(1) %h7, align 1
  %e7 = sext i8 %v7 to i32
  %f7 = sext i8 %w7 to i32
  %m7 = mul nsw i32 %e7, %f7
  %t7 = add i32 %t6, %m7
  %i1 = add nuw nsw i32 %i, 8
  %c = icmp ult i32 %i1, 1024
  br i1 %c, label %l, label %exit
exit:
  store i32 %t7, ptr addrspace(1) %o, align 4
  ret void
}

; The PTX front end's addresses: an i64 kernel argument, cast to a pointer
; and back, plus the offsets, through inttoptr. The test is of the cast
; argument.
; CHECK-LABEL: ptx:
; CHECK: and {{r[0-9]+}}, u0, 3
; CHECK: device_load {{[0-9]}}, i32, xy,
; CHECK: stop
define agx_kernel void @ptx(i64 %a, ptr addrspace(1) %o) {
entry:
  %p = inttoptr i64 %a to ptr
  %d = addrspacecast ptr %p to ptr addrspace(1)
  %base = ptrtoint ptr addrspace(1) %d to i64
  br label %l
l:
  %off = phi i64 [ 0, %entry ], [ %off1, %l ]
  %s = phi float [ 0.0, %entry ], [ %s7, %l ]
  %ia = add i64 %base, %off
  %g0 = inttoptr i64 %ia to ptr addrspace(1)
  %v0 = load half, ptr addrspace(1) %g0, align 2
  %g1 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 2
  %v1 = load half, ptr addrspace(1) %g1, align 2
  %g2 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 4
  %v2 = load half, ptr addrspace(1) %g2, align 2
  %g3 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 6
  %v3 = load half, ptr addrspace(1) %g3, align 2
  %g4 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 8
  %v4 = load half, ptr addrspace(1) %g4, align 2
  %g5 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 10
  %v5 = load half, ptr addrspace(1) %g5, align 2
  %g6 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 12
  %v6 = load half, ptr addrspace(1) %g6, align 2
  %g7 = getelementptr inbounds i8, ptr addrspace(1) %g0, i64 14
  %v7 = load half, ptr addrspace(1) %g7, align 2
  %e0 = fpext half %v0 to float
  %e1 = fpext half %v1 to float
  %e2 = fpext half %v2 to float
  %e3 = fpext half %v3 to float
  %e4 = fpext half %v4 to float
  %e5 = fpext half %v5 to float
  %e6 = fpext half %v6 to float
  %e7 = fpext half %v7 to float
  %s0 = fadd float %s, %e0
  %s1 = fadd float %s0, %e1
  %s2 = fadd float %s1, %e2
  %s3 = fadd float %s2, %e3
  %s4 = fadd float %s3, %e4
  %s5 = fadd float %s4, %e5
  %s6 = fadd float %s5, %e6
  %s7 = fadd float %s6, %e7
  %off1 = add nuw nsw i64 %off, 16
  %c = icmp ult i64 %off1, 1024
  br i1 %c, label %l, label %exit
exit:
  store float %s7, ptr addrspace(1) %o, align 4
  ret void
}

; One <4 x half> load a step (MLIR's vector<4xf16> at align 2): four 16-bit
; loads, which count as four accesses towards the minimum (L15-E478); the
; aligned copy loads the vector at once.
; CHECK-LABEL: vec4_sumsq:
; CHECK: and {{r[0-9]+}}, u0, 3
; CHECK: device_load 0, i32, xy,
; ONE-LABEL: vec4_sumsq:
; ONE-NOT: and {{r[0-9]+}}, u0, 3
; ONE-NOT: i32, xy
; ONE: stop
define agx_kernel void @vec4_sumsq(ptr addrspace(1) %x, ptr addrspace(1) %y) {
entry:
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %l4 = shl nuw nsw i32 %lane, 2
  %l64 = zext i32 %l4 to i64
  br label %loop
loop:
  %j = phi i64 [ %l64, %entry ], [ %jn, %loop ]
  %a = phi <4 x float> [ zeroinitializer, %entry ], [ %an, %loop ]
  %p = getelementptr half, ptr addrspace(1) %x, i64 %j
  %h = load <4 x half>, ptr addrspace(1) %p, align 2
  %v = fpext <4 x half> %h to <4 x float>
  %an = call <4 x float> @llvm.fma.v4f32(<4 x float> %v, <4 x float> %v, <4 x float> %a)
  %jn = add nuw nsw i64 %j, 128
  %c = icmp ult i64 %jn, 1024
  br i1 %c, label %loop, label %done
done:
  %e = extractelement <4 x float> %an, i32 0
  %q = getelementptr float, ptr addrspace(1) %y, i32 %lane
  store float %e, ptr addrspace(1) %q
  ret void
}
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)

; A vector of bytes counts (L15-E487): its lanes are sign-extended in their
; 16-bit halves, the products' 16-bit sources, without a copy to 32 bits.
; (Scalar bytes, the case above, still not.)
; CHECK-LABEL: dot_i8:
; CHECK: and {{r[0-9]+}}, {{r[0-9]+}}, 3
; CHECK: %loop
; CHECK: device_load {{[0-9]}}, i8, x,
; CHECK: mov_imm [[BIAS:r[0-9]+]], 8388736
; CHECK: %loop.aligned
; CHECK: device_load {{[0-9]}}, i8, xyzw,
; CHECK: xor [[W:r[0-9]+]], {{r[0-9]+}}, [[BIAS]]
; CHECK: isub [[W]]l, [[W]]l, 128
; CHECK: imadd {{r[0-9]+}}, {{r[0-9]+[lh]}}.sx, {{r[0-9]+[lh]}}.sx,
; CHECK: stop
; Each pair of lanes (a 32-bit register's halves) is biased by one xor with
; 0x00800080 (8388736) (L15-E488); with the option off, one xor per lane.
; NOPAIR-LABEL: dot_i8:
; NOPAIR: %loop.aligned
; NOPAIR: xor {{r[0-9]+[lh]}}, {{r[0-9]+[lh]}}, 128
; NOPAIR: stop
; NOLANE-LABEL: dot_i8:
; NOLANE: %loop.aligned
; NOLANE: bitop 0101, {{r[0-9]+}}, {{r[0-9]+[lh]}}, 0
; NOLANE-NOT: .sx
; NOLANE: stop
; OFF-LABEL: dot_i8:
; OFF-NOT: i8, xyzw
; OFF: stop
define agx_kernel void @dot_i8(ptr addrspace(1) %a, ptr addrspace(1) %x, ptr addrspace(1) %y) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %t64 = zext i32 %t to i64
  %row = shl nuw nsw i64 %t64, 10
  %base = getelementptr i8, ptr addrspace(1) %a, i64 %row
  br label %loop

loop:
  %j = phi i64 [ 0, %entry ], [ %j.next, %loop ]
  %acc = phi <4 x i32> [ zeroinitializer, %entry ], [ %acc.next, %loop ]
  %pa = getelementptr i8, ptr addrspace(1) %base, i64 %j
  %va = load <4 x i8>, ptr addrspace(1) %pa, align 1
  %px = getelementptr i8, ptr addrspace(1) %x, i64 %j
  %vx = load <4 x i8>, ptr addrspace(1) %px, align 1
  %ea = sext <4 x i8> %va to <4 x i32>
  %ex = sext <4 x i8> %vx to <4 x i32>
  %p = mul nsw <4 x i32> %ex, %ea
  %acc.next = add <4 x i32> %p, %acc
  %j.next = add nuw nsw i64 %j, 4
  %c = icmp ult i64 %j.next, 1024
  br i1 %c, label %loop, label %exit

exit:
  %s0 = extractelement <4 x i32> %acc.next, i64 0
  %s1 = extractelement <4 x i32> %acc.next, i64 1
  %s2 = extractelement <4 x i32> %acc.next, i64 2
  %s3 = extractelement <4 x i32> %acc.next, i64 3
  %s01 = add i32 %s0, %s1
  %s23 = add i32 %s2, %s3
  %s = add i32 %s01, %s23
  %py = getelementptr i32, ptr addrspace(1) %y, i64 %t64
  store i32 %s, ptr addrspace(1) %py, align 4
  ret void
}
