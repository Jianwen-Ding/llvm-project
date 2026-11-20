; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-masked-tail-unmasked-body=0 < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -O3 -agx-masked-tail-unmasked-body=0 -agx-upper-bound-unknown-ranges=0 < %s | FileCheck %s --check-prefix=OFF
;
; A sliding window's loop after L15-E480 / E481 (its bound lowered, its
; start raised past the masked head): the start's and the bound's
; saturating arithmetic leaves the counter's bound to SCEV's ranges, which
; AGXNarrowInductions' upper bounds now take where their own structural
; bound has none (1024 - lane): the loop's index arithmetic and masks are
; 32-bit (L15-E482). (L15-E518 / E519 take the masks out of the loop: off
; here.)

; CHECK-LABEL: window:
; CHECK: Loop Header
; CHECK-NOT: iadd {{r[0-9]+}}_{{r[0-9]+}}
; CHECK: jmp_exec_any
; OFF-LABEL: window:
; OFF: Loop Header
; OFF: iadd {{r[0-9]+}}_{{r[0-9]+}}
; OFF: jmp_exec_any

define agx_kernel void @window(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 range(i32 0, 1024) %row32, i32 range(i32 0, 32) %lane32, i32 range(i32 0, 1025) %n32) {
entry:
  %row = zext i32 %row32 to i64
  %lane = zext i32 %lane32 to i64
  %n = zext i32 %n32 to i64
  %g = icmp ult i64 0, %n
  br i1 %g, label %loop, label %done
loop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %loop ]
  %r = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  %a = add nuw i64 %j, %lane
  %u1 = icmp ule i64 %a, %row
  %aw = add nuw i64 %a, 128
  %u2 = icmp uge i64 %aw, %row
  %ok = and i1 %u1, %u2
  %q = getelementptr float, ptr addrspace(1) %p, i64 %j
  %x = load float, ptr addrspace(1) %q
  %s = select i1 %ok, float %x, float 0.0
  %r.n = fadd float %r, %s
  %j.n = add nuw i64 %j, 32
  %c = icmp ult i64 %j.n, %n
  br i1 %c, label %loop, label %done
done:
  %f = phi float [ 0.0, %entry ], [ %r.n, %loop ]
  store float %f, ptr addrspace(1) %o
  ret void
}
