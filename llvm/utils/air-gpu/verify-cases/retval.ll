; ?: a kernel that returns a value
target triple = "air"
define i32 @k(ptr addrspace(1) %o) {
  store i32 7, ptr addrspace(1) %o
  ret i32 1
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"o"}
