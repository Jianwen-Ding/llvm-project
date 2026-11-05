; ?: a SIMD-group function AIR does not have
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air"

declare i32 @air.simd_nonsense.i32(i32)
define void @k(ptr addrspace(1) "air-buffer-no-alias" %o, <3 x i32> %tidtg) #0 {
  %tidtg_x = extractelement <3 x i32> %tidtg, i32 0
  %guard_is_t0 = icmp eq i32 %tidtg_x, 0
  br i1 %guard_is_t0, label %body, label %exit

body:                                             ; preds = %0
  %1 = bitcast ptr addrspace(1) %o to ptr addrspace(1)
  store i32 7, ptr addrspace(1) %1, align 4, !alias.scope !18
  %mz = call i32 @air.simd_nonsense.i32(i32 1)
  ret void

exit:                                             ; preds = %0
  ret void
}

attributes #0 = { "air-kernel" }

!air.kernel = !{!4}
!air.version = !{!5}
!air.language_version = !{!6}
!air.compile_options = !{!7, !8, !9}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15}

!0 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"o"}
!1 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tidtg"}
!2 = !{}
!3 = !{!0, !1}
!4 = !{ptr @k, !2, !3}
!5 = !{i32 2, i32 8, i32 0}
!6 = !{!"Metal", i32 4, i32 0, i32 0}
!7 = !{!"air.compile.denorms_disable"}
!8 = !{!"air.compile.fast_math_enable"}
!9 = !{!"air.compile.framebuffer_fetch_enable"}
!10 = !{i32 7, !"air.max_device_buffers", i32 31}
!11 = !{i32 7, !"air.max_constant_buffers", i32 31}
!12 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!13 = !{i32 7, !"air.max_textures", i32 128}
!14 = !{i32 7, !"air.max_read_write_textures", i32 8}
!15 = !{i32 7, !"air.max_samplers", i32 16}
!16 = distinct !{!16, i1 false, !"air-alias-scopes(k)"}
!17 = distinct !{!17, !16, !"air-alias-scope-arg(0)"}
!18 = !{!17}
