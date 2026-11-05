; ?: a bare !air.kernel entry
target triple = "air"
define void @k(ptr addrspace(1) %o) {
  store i32 7, ptr addrspace(1) %o
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k}
