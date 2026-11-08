; CUDA's __mulhi is the high word of the signed product and __umulhi of the
; unsigned one (alive2: __mulhi(-1610612736, 7) is -3, not the 4 of the
; unsigned product).
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s

; CHECK-LABEL: define void @k(
; CHECK: %[[A:.*]] = sext i32 %{{.*}} to i64
; CHECK: %[[B:.*]] = sext i32 %{{.*}} to i64
; CHECK: %[[P:.*]] = mul i64 %[[A]], %[[B]]
; CHECK: %[[H:.*]] = lshr i64 %[[P]], 32
; CHECK: trunc i64 %[[H]] to i32
; CHECK: %[[C:.*]] = zext i32 %{{.*}} to i64
; CHECK: %[[D:.*]] = zext i32 %{{.*}} to i64
; CHECK: %[[Q:.*]] = mul i64 %[[C]], %[[D]]
; CHECK: lshr i64 %[[Q]], 32
; CHECK-NOT: mulhi

declare i32 @__mulhi(i32, i32)
declare i32 @__umulhi(i32, i32)

define void @k(ptr addrspace(1) %out, i32 %a, i32 %b) "air-kernel" {
  %h = call i32 @__mulhi(i32 %a, i32 %b)
  %u = call i32 @__umulhi(i32 %a, i32 %b)
  %s = add i32 %h, %u
  store i32 %s, ptr addrspace(1) %out
  ret void
}
