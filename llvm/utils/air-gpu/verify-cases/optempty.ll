; ?: an empty compile option
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air"

@tg = internal addrspace(3) global [4 x i32] undef, align 16
@c = internal addrspace(2) constant [2 x i32] [i32 3, i32 5], align 4

declare void @air.wg.barrier(i32, i32)

declare i32 @air.atomic.global.add.s.i32(ptr addrspace(1), i32, i32, i32, i1)

declare [3 x i32] @air.thread_position_in_threadgroup()

define void @k(ptr addrspace(1) "air-buffer-no-alias" %o, ptr addrspace(1) "air-buffer-no-alias" %_scalar_buf) #0 {
  %__base_tg = getelementptr inbounds [4 x i32], ptr addrspace(3) @tg, i64 0, i64 0
  %p_gep = getelementptr inbounds float, ptr addrspace(1) %_scalar_buf, i64 0
  %p_raw = load float, ptr addrspace(1) %p_gep, align 4, !alias.scope !19, !noalias !20
  %p_i32 = bitcast float %p_raw to i32
  %p_lo64 = zext i32 %p_i32 to i64
  %p_gep_hi = getelementptr inbounds float, ptr addrspace(1) %_scalar_buf, i64 1
  %p_hi_raw = load float, ptr addrspace(1) %p_gep_hi, align 4, !alias.scope !19, !noalias !20
  %p_hi32 = bitcast float %p_hi_raw to i32
  %p_hi64 = zext i32 %p_hi32 to i64
  %p_shift = shl i64 %p_hi64, 32
  %p = or i64 %p_shift, %p_lo64
  %guard_tid = call [3 x i32] @air.thread_position_in_threadgroup()
  %guard_tid_x = extractvalue [3 x i32] %guard_tid, 0
  %guard_is_t0 = icmp eq i32 %guard_tid_x, 0
  call void @air.wg.barrier(i32 2, i32 1)
  br label %.tgbr

.tgbr:                                            ; preds = %0
  br i1 %guard_is_t0, label %body, label %exit

body:                                             ; preds = %.tgbr
  store i32 3, ptr addrspace(3) %__base_tg, align 4
  call void @air.wg.barrier(i32 2, i32 1)
  %w = load i32, ptr addrspace(3) %__base_tg, align 4
  %a = call i32 @air.atomic.global.add.s.i32(ptr addrspace(1) %o, i32 %w, i32 0, i32 2, i1 false)
  %1 = bitcast ptr addrspace(1) %o to ptr addrspace(1)
  store i32 %a, ptr addrspace(1) %1, align 4, !alias.scope !20, !noalias !19
  ret void

exit:                                             ; preds = %.tgbr
  ret void

; uselistorder directives
  uselistorder ptr addrspace(1) %o, { 1, 0 }
}

; uselistorder directives
uselistorder ptr @air.wg.barrier, { 1, 0 }

attributes #0 = { "air-kernel" }

!air.kernel = !{!4}
!air.version = !{!5}
!air.language_version = !{!6}
!air.compile_options = !{!7, !8, !9}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15}

!0 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"o"}
!1 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"_scalar_buf"}
!2 = !{}
!3 = !{!0, !1}
!4 = !{ptr @k, !2, !3}
!5 = !{i32 2, i32 8, i32 0}
!6 = !{!"Metal", i32 4, i32 0, i32 0}
!7 = !{}
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
!18 = distinct !{!18, !16, !"air-alias-scope-arg(1)"}
!19 = !{!18}
!20 = !{!17}
