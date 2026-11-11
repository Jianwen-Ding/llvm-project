; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-relative-counters=0 < %s | FileCheck %s --check-prefix=OFF

; x's pointer, stepped by 128 bytes beside a counter j from a runtime start
; (the lane) by 32 whose range is not known (the int8 GEMV of agx-bench):
; j less its start (32 k after k iterations, below 2^31 by the trip count's
; bound: j is i32 and the step 32) is the index register, one 32-bit isub
; instead of the 64-bit add (L15-E203).

; CHECK-LABEL: int8_dp4a_gemv:
; CHECK: Loop Header
; CHECK: isub [[I:r[0-9]+]], r{{[0-9]+}}, r{{[0-9]+}}
; CHECK-NOT: iadd r{{[0-9]+}}_r{{[0-9]+}}
; CHECK: device_load {{.*}}, [[I]], unsigned
; CHECK: jmp_exec_any
; OFF-LABEL: int8_dp4a_gemv:
; OFF: Loop Header
; OFF: iadd r{{[0-9]+}}_r{{[0-9]+}}, r{{[0-9]+}}_r{{[0-9]+}}, 128

define agx_kernel void @int8_dp4a_gemv(ptr addrspace(1) noalias readonly %a, ptr addrspace(1) noalias readonly %x, ptr addrspace(1) noalias readonly %sa, ptr addrspace(1) noalias %y, float %sx, i32 %k) {
entry:
  %b = call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %sg = call i32 @llvm.agx.simdgroup.index.in.threadgroup()
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %b4 = shl i32 %b, 2
  %row = add i32 %b4, %sg
  %words = lshr i32 %k, 2
  %rowoff = mul i32 %row, %words
  %any = icmp ult i32 %lane, %words
  br i1 %any, label %loop, label %done
loop:
  %j = phi i32 [ %lane, %entry ], [ %jn, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %accn, %loop ]
  %ia = add i32 %rowoff, %j
  %pa = getelementptr i32, ptr addrspace(1) %a, i32 %ia
  %px = getelementptr i32, ptr addrspace(1) %x, i32 %j
  %wa = load i32, ptr addrspace(1) %pa, align 4
  %wx = load i32, ptr addrspace(1) %px, align 4
  %va = bitcast i32 %wa to <4 x i8>
  %vx = bitcast i32 %wx to <4 x i8>
  %ea = sext <4 x i8> %va to <4 x i32>
  %ex = sext <4 x i8> %vx to <4 x i32>
  %m = mul nsw <4 x i32> %ea, %ex
  %d = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %m)
  %accn = add i32 %acc, %d
  %jn = add i32 %j, 32
  %more = icmp ult i32 %jn, %words
  br i1 %more, label %loop, label %done
done:
  %part = phi i32 [ 0, %entry ], [ %accn, %loop ]
  %sum = call i32 @llvm.agx.simd.reduce.i32(i32 %part, i32 16)
  %first = icmp eq i32 %lane, 0
  br i1 %first, label %store, label %end
store:
  %ps = getelementptr float, ptr addrspace(1) %sa, i32 %row
  %s = load float, ptr addrspace(1) %ps, align 4
  %sf = sitofp i32 %sum to float
  %s2 = fmul float %s, %sx
  %r = fmul float %sf, %s2
  %py = getelementptr float, ptr addrspace(1) %y, i32 %row
  store float %r, ptr addrspace(1) %py, align 4
  br label %end
end:
  ret void
}
