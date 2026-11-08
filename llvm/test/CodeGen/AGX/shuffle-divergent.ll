; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s

; simd_shuffle ORs the low two index bits of the threads of each quad, and
; reads the index of inactive threads too. Divergent indices are split into
; the four quad positions, and register indices are passed in r0h, which is
; zero whenever no shuffle is executing.

declare i32 @llvm.agx.thread.index.in.simdgroup()
declare float @llvm.agx.simd.shuffle.f32(float, i32)
declare float @llvm.agx.simd.shuffle.down.f32(float, i32)
declare float @llvm.agx.quad.shuffle.f32(float, i32)

; Within a block the execution mask does not change between the four
; shuffles, so only the last one resets r0h.
; CHECK-LABEL: divergent:
; CHECK:         mov_imm r0h, 0
; CHECK-NOT:     mov_imm r0h
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NOT:     mov_imm r0h
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NOT:     mov_imm r0h
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NOT:     mov_imm r0h
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NEXT:    mov_imm r0h, 0
; CHECK-NOT:     simd_shuffle
; CHECK:         stop
define agx_kernel void @divergent(ptr addrspace(1) %out, float %x) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %idx = mul i32 %lane, 7
  %r = call float @llvm.agx.simd.shuffle.f32(float %x, i32 %idx)
  %p = getelementptr float, ptr addrspace(1) %out, i32 %lane
  store float %r, ptr addrspace(1) %p
  ret void
}

; A uniform index is used directly.
; CHECK-LABEL: uniform:
; CHECK-NOT:     r0h
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, u3l
; CHECK-NOT:     simd_shuffle
; CHECK:         stop
define agx_kernel void @uniform(ptr addrspace(1) %out, float %x, i32 %i) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %v = uitofp i32 %lane to float
  %r = call float @llvm.agx.simd.shuffle.f32(float %v, i32 %i)
  %p = getelementptr float, ptr addrspace(1) %out, i32 %lane
  store float %r, ptr addrspace(1) %p
  ret void
}

; A divergent delta reads the computed thread; quad shuffles select between
; the four immediate positions.
; CHECK-LABEL: relative:
; CHECK-COUNT-4: simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NOT:     simd_shuffle_down
; CHECK:         quad_shuffle {{r[0-9]+}}, {{r[0-9]+}}, 0
; CHECK:         quad_shuffle {{r[0-9]+}}, {{r[0-9]+}}, 1
; CHECK:         quad_shuffle {{r[0-9]+}}, {{r[0-9]+}}, 2
; CHECK:         quad_shuffle {{r[0-9]+}}, {{r[0-9]+}}, 3
; CHECK:         stop
define agx_kernel void @relative(ptr addrspace(1) %out, float %x) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %d = and i32 %lane, 3
  %v = uitofp i32 %lane to float
  %a = call float @llvm.agx.simd.shuffle.down.f32(float %v, i32 %d)
  ; (not lane & 3 itself: a thread reading its own position is no shuffle)
  %e = sub i32 3, %d
  %b = call float @llvm.agx.quad.shuffle.f32(float %a, i32 %e)
  %p = getelementptr float, ptr addrspace(1) %out, i32 %lane
  store float %b, ptr addrspace(1) %p
  ret void
}

; Without the merge, divergent indices are used directly.
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=-quad-shuffle-index-merge < %s | FileCheck %s --check-prefix=NOMERGE
; NOMERGE-LABEL: divergent:
; NOMERGE-COUNT-1: simd_shuffle
; NOMERGE-NOT:   simd_shuffle
; NOMERGE:       stop

; Positions that the known bits of the index exclude are left out: with bit 0
; set, only positions 1 and 3 remain; with both low bits known, one shuffle
; reads the thread.
; CHECK-LABEL: known_bits:
; CHECK-COUNT-3: simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NOT:     simd_shuffle
; CHECK:         stop
define agx_kernel void @known_bits(ptr addrspace(1) %out, float %x) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %t = shl i32 %lane, 1
  %idx = or i32 %t, 1
  %r = call float @llvm.agx.simd.shuffle.f32(float %x, i32 %idx)
  %idx2 = shl i32 %lane, 2
  %r2 = call float @llvm.agx.simd.shuffle.f32(float %r, i32 %idx2)
  %p = getelementptr float, ptr addrspace(1) %out, i32 %lane
  store float %r2, ptr addrspace(1) %p
  ret void
}

; A reset before a change of the execution mask stays: the threads inactive
; in the if must read zero at the shuffle after it.
; CHECK-LABEL: across_if:
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NEXT:    mov_imm r0h, 0
; CHECK:         pop_exec
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NEXT:    mov_imm r0h, 0
; CHECK:         stop
define agx_kernel void @across_if(ptr addrspace(1) %out, float %x) {
entry:
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %idx = shl i32 %lane, 2
  %c = icmp ult i32 %lane, 5
  br i1 %c, label %then, label %join
then:
  %r = call float @llvm.agx.simd.shuffle.f32(float %x, i32 %idx)
  br label %join
join:
  %v = phi float [ %r, %then ], [ %x, %entry ]
  %r2 = call float @llvm.agx.simd.shuffle.f32(float %v, i32 %idx)
  %p = getelementptr float, ptr addrspace(1) %out, i32 %lane
  store float %r2, ptr addrspace(1) %p
  ret void
}

; simd_shuffle reads from each source quad the thread at the OR of the low
; index bits of that quad's own threads (measured on the M2 Max: the reader's
; low bits do not matter). An index of the thread index alone whose source
; quads all hold the position they are read at reads the thread it names with
; one shuffle, though it differs across the SIMD-group: here thread 4q + w
; reads 4 * (5q mod 8) + q mod 4, and quad 5q mod 8 holds 5q mod 4 = q mod 4.
; CHECK-LABEL: quad_positions:
; CHECK-COUNT-1: simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NOT:     simd_shuffle
; CHECK:         stop
define agx_kernel void @quad_positions(ptr addrspace(1) %out, float %x) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %q = lshr i32 %lane, 2
  %q5 = mul i32 %q, 5
  %qs = and i32 %q5, 7
  %hi = shl i32 %qs, 2
  %lo = and i32 %q, 3
  %idx = or i32 %hi, %lo
  %r = call float @llvm.agx.simd.shuffle.f32(float %x, i32 %idx)
  %p = getelementptr float, ptr addrspace(1) %out, i32 %lane
  store float %r, ptr addrspace(1) %p
  ret void
}

; Low bits the same in each reading quad are not enough: thread 4q + w reads
; quad q ^ 2 at position (q >> 1) & 3, but that quad's threads hold
; ((q ^ 2) >> 1) & 3. (Wrong on the hardware without the split.)
; CHECK-LABEL: reader_positions:
; CHECK-COUNT-4: simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NOT:     simd_shuffle
; CHECK:         stop
define agx_kernel void @reader_positions(ptr addrspace(1) %out, float %x) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %l8 = xor i32 %lane, 8
  %hi = and i32 %l8, 28
  %sh = lshr i32 %lane, 3
  %lo = and i32 %sh, 3
  %idx = or i32 %hi, %lo
  %r = call float @llvm.agx.simd.shuffle.f32(float %x, i32 %idx)
  %p = getelementptr float, ptr addrspace(1) %out, i32 %lane
  store float %r, ptr addrspace(1) %p
  ret void
}
