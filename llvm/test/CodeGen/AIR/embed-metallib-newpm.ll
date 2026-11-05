; (A module the device would reject: this test is about another part of the target.)
; Drive AIREmbedderPass via the new pass manager. The embedder serializes
; the module into a .metallib byte blob held in a private global with section
; ".metallib".
; RUN: opt -air-skip-device-checks -mtriple=air -passes=air-embed -S %s | FileCheck %s

; CHECK: @metal.metallib = private constant [{{[0-9]+}} x i8]
; CHECK-SAME: section ".metallib"
; CHECK: @llvm.compiler.used = appending global

define void @kernel() {
  ret void
}
