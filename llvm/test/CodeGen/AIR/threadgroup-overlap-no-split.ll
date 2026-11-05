; A byte array whose accesses span each other's offsets is one variable read another way, not separate
; buffers packed in an arena: it stays one global. Split, a read inside a stored vector reads another
; variable and gets zero (PTX st.shared.v4.u32 then ld.shared.u8 with an mbarrier, on an M2 Max).
; RUN: llc -mtriple=air %s -o - | FileCheck %s
; CHECK-NOT: __off
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air"

@k.shared = internal addrspace(3) global [72 x i8] undef, align 16

; Function Attrs: denormal_fpenv(preservesign)
define void @k(ptr addrspace(2) byref(i64) %0, ptr addrspace(2) byref(i64) %1) local_unnamed_addr #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = load i64, ptr addrspace(2) %0, align 8
  store i32 524288, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64), align 4
  %8 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64), i64 4
  store i32 1, ptr addrspace(3) %8, align 4
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr addrspace(3) @k.shared, align 16
  %9 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 16
  store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr addrspace(3) %9, align 16
  call void @air.atomic.fence(i32 2, i32 3, i32 1)
  %10 = load atomic i32, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64) syncscope("workgroup") monotonic, align 4
  br label %11

11:                                               ; preds = %11, %2
  %12 = phi i32 [ %10, %2 ], [ %29, %11 ]
  %13 = add i32 %12, 2146959362
  %14 = and i32 %13, 2147483647
  %15 = and i32 %12, -2147483648
  %16 = or disjoint i32 %15, %14
  %17 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64), i64 4
  %18 = load atomic i32, ptr addrspace(3) %17 syncscope("workgroup") monotonic, align 4
  %19 = shl i32 %18, 19
  %20 = and i32 %19, 2146959360
  %21 = or disjoint i32 %15, %20
  %22 = xor i32 %21, -2147483648
  %23 = icmp eq i32 %14, 0
  %24 = select i1 %23, i32 %22, i32 %16
  call void @air.atomic.fence(i32 2, i32 5, i32 1)
  store i32 %12, ptr %6, align 4
  %25 = call i32 @air.atomic.local.cmpxchg.weak.i32(ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64), ptr %6, i32 %24, i32 0, i32 0, i32 1, i1 false), !agx.atomic.ordering !19
  call void @air.atomic.fence(i32 2, i32 5, i32 1)
  %26 = icmp eq i32 %25, %12
  %27 = insertvalue { i32, i1 } poison, i32 %25, 0
  %28 = insertvalue { i32, i1 } %27, i1 %26, 1
  %29 = extractvalue { i32, i1 } %28, 0
  %30 = extractvalue { i32, i1 } %28, 1
  br i1 %30, label %31, label %11

31:                                               ; preds = %11
  br i1 %23, label %32, label %61

32:                                               ; preds = %31
  %33 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64), i64 4
  %34 = load atomic i32, ptr addrspace(3) %33 syncscope("workgroup") monotonic, align 4
  %35 = lshr exact i32 %22, 3
  %36 = and i32 %35, 268435456
  br label %37

37:                                               ; preds = %37, %32
  %38 = phi i32 [ %34, %32 ], [ %59, %37 ]
  %39 = lshr i32 %38, 25
  %40 = and i32 %39, 1
  %41 = and i32 %38, -503316481
  %42 = or disjoint i32 %41, %36
  %43 = shl nuw nsw i32 %40, 26
  %44 = shl nuw nsw i32 %40, 27
  %45 = and i32 %38, 134217728
  %46 = xor i32 %44, %45
  %47 = or disjoint i32 %43, %46
  %48 = or disjoint i32 %47, %42
  %49 = xor i32 %48, 134217728
  %50 = or disjoint i32 %42, %45
  %51 = and i32 %38, 16777216
  %52 = icmp eq i32 %51, 0
  %53 = select i1 %52, i32 %50, i32 %49
  %54 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64), i64 4
  call void @air.atomic.fence(i32 2, i32 5, i32 1)
  store i32 %38, ptr %5, align 4
  %55 = call i32 @air.atomic.local.cmpxchg.weak.i32(ptr addrspace(3) %54, ptr %5, i32 %53, i32 0, i32 0, i32 1, i1 false), !agx.atomic.ordering !19
  call void @air.atomic.fence(i32 2, i32 5, i32 1)
  %56 = icmp eq i32 %55, %38
  %57 = insertvalue { i32, i1 } poison, i32 %55, 0
  %58 = insertvalue { i32, i1 } %57, i1 %56, 1
  %59 = extractvalue { i32, i1 } %58, 0
  %60 = extractvalue { i32, i1 } %58, 1
  br i1 %60, label %61, label %37

61:                                               ; preds = %37, %31
  call void @air.atomic.fence(i32 2, i32 2, i32 1)
  call void @air.atomic.fence(i32 2, i32 3, i32 1)
  %62 = load atomic i32, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64) syncscope("workgroup") monotonic, align 4
  br label %63

63:                                               ; preds = %63, %61
  %64 = phi i32 [ %62, %61 ], [ %81, %63 ]
  %65 = add i32 %64, 2147483646
  %66 = and i32 %65, 2147483647
  %67 = and i32 %64, -2147483648
  %68 = or disjoint i32 %67, %66
  %69 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64), i64 4
  %70 = load atomic i32, ptr addrspace(3) %69 syncscope("workgroup") monotonic, align 4
  %71 = shl i32 %70, 19
  %72 = and i32 %71, 2146959360
  %73 = or disjoint i32 %67, %72
  %74 = xor i32 %73, -2147483648
  %75 = icmp eq i32 %66, 0
  %76 = select i1 %75, i32 %74, i32 %68
  call void @air.atomic.fence(i32 2, i32 5, i32 1)
  store i32 %64, ptr %4, align 4
  %77 = call i32 @air.atomic.local.cmpxchg.weak.i32(ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64), ptr %4, i32 %76, i32 0, i32 0, i32 1, i1 false), !agx.atomic.ordering !19
  call void @air.atomic.fence(i32 2, i32 5, i32 1)
  %78 = icmp eq i32 %77, %64
  %79 = insertvalue { i32, i1 } poison, i32 %77, 0
  %80 = insertvalue { i32, i1 } %79, i1 %78, 1
  %81 = extractvalue { i32, i1 } %80, 0
  %82 = extractvalue { i32, i1 } %80, 1
  br i1 %82, label %83, label %63

83:                                               ; preds = %63
  br i1 %75, label %84, label %113

84:                                               ; preds = %83
  %85 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64), i64 4
  %86 = load atomic i32, ptr addrspace(3) %85 syncscope("workgroup") monotonic, align 4
  %87 = lshr exact i32 %74, 3
  %88 = and i32 %87, 268435456
  br label %89

89:                                               ; preds = %89, %84
  %90 = phi i32 [ %86, %84 ], [ %111, %89 ]
  %91 = lshr i32 %90, 25
  %92 = and i32 %91, 1
  %93 = and i32 %90, -503316481
  %94 = or disjoint i32 %93, %88
  %95 = shl nuw nsw i32 %92, 26
  %96 = shl nuw nsw i32 %92, 27
  %97 = and i32 %90, 134217728
  %98 = xor i32 %96, %97
  %99 = or disjoint i32 %95, %98
  %100 = or disjoint i32 %99, %94
  %101 = xor i32 %100, 134217728
  %102 = or disjoint i32 %94, %97
  %103 = and i32 %90, 16777216
  %104 = icmp eq i32 %103, 0
  %105 = select i1 %104, i32 %102, i32 %101
  %106 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @k.shared, i64 64), i64 4
  call void @air.atomic.fence(i32 2, i32 5, i32 1)
  store i32 %90, ptr %3, align 4
  %107 = call i32 @air.atomic.local.cmpxchg.weak.i32(ptr addrspace(3) %106, ptr %3, i32 %105, i32 0, i32 0, i32 1, i1 false), !agx.atomic.ordering !19
  call void @air.atomic.fence(i32 2, i32 5, i32 1)
  %108 = icmp eq i32 %107, %90
  %109 = insertvalue { i32, i1 } poison, i32 %107, 0
  %110 = insertvalue { i32, i1 } %109, i1 %108, 1
  %111 = extractvalue { i32, i1 } %110, 0
  %112 = extractvalue { i32, i1 } %110, 1
  br i1 %112, label %113, label %89

113:                                              ; preds = %89, %83
  call void @air.atomic.fence(i32 2, i32 2, i32 1)
  %114 = load i8, ptr addrspace(3) @k.shared, align 1
  %115 = inttoptr i64 %7 to ptr addrspace(1)
  store i8 %114, ptr addrspace(1) %115, align 1
  %116 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 1
  %117 = load i8, ptr addrspace(3) %116, align 1
  %118 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 1
  store i8 %117, ptr addrspace(1) %118, align 1
  %119 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 2
  %120 = load i8, ptr addrspace(3) %119, align 1
  %121 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 2
  store i8 %120, ptr addrspace(1) %121, align 1
  %122 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 3
  %123 = load i8, ptr addrspace(3) %122, align 1
  %124 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 3
  store i8 %123, ptr addrspace(1) %124, align 1
  %125 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 4
  %126 = load i8, ptr addrspace(3) %125, align 1
  %127 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 4
  store i8 %126, ptr addrspace(1) %127, align 1
  %128 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 5
  %129 = load i8, ptr addrspace(3) %128, align 1
  %130 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 5
  store i8 %129, ptr addrspace(1) %130, align 1
  %131 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 6
  %132 = load i8, ptr addrspace(3) %131, align 1
  %133 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 6
  store i8 %132, ptr addrspace(1) %133, align 1
  %134 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 7
  %135 = load i8, ptr addrspace(3) %134, align 1
  %136 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 7
  store i8 %135, ptr addrspace(1) %136, align 1
  %137 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 8
  %138 = load i8, ptr addrspace(3) %137, align 1
  %139 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 8
  store i8 %138, ptr addrspace(1) %139, align 1
  %140 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 9
  %141 = load i8, ptr addrspace(3) %140, align 1
  %142 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 9
  store i8 %141, ptr addrspace(1) %142, align 1
  %143 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 10
  %144 = load i8, ptr addrspace(3) %143, align 1
  %145 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 10
  store i8 %144, ptr addrspace(1) %145, align 1
  %146 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 11
  %147 = load i8, ptr addrspace(3) %146, align 1
  %148 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 11
  store i8 %147, ptr addrspace(1) %148, align 1
  %149 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 12
  %150 = load i8, ptr addrspace(3) %149, align 1
  %151 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 12
  store i8 %150, ptr addrspace(1) %151, align 1
  %152 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 13
  %153 = load i8, ptr addrspace(3) %152, align 1
  %154 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 13
  store i8 %153, ptr addrspace(1) %154, align 1
  %155 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 14
  %156 = load i8, ptr addrspace(3) %155, align 1
  %157 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 14
  store i8 %156, ptr addrspace(1) %157, align 1
  %158 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 15
  %159 = load i8, ptr addrspace(3) %158, align 1
  %160 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 15
  store i8 %159, ptr addrspace(1) %160, align 1
  %161 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 16
  %162 = load i8, ptr addrspace(3) %161, align 1
  %163 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 16
  store i8 %162, ptr addrspace(1) %163, align 1
  %164 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 17
  %165 = load i8, ptr addrspace(3) %164, align 1
  %166 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 17
  store i8 %165, ptr addrspace(1) %166, align 1
  %167 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 18
  %168 = load i8, ptr addrspace(3) %167, align 1
  %169 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 18
  store i8 %168, ptr addrspace(1) %169, align 1
  %170 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 19
  %171 = load i8, ptr addrspace(3) %170, align 1
  %172 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 19
  store i8 %171, ptr addrspace(1) %172, align 1
  %173 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 20
  %174 = load i8, ptr addrspace(3) %173, align 1
  %175 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 20
  store i8 %174, ptr addrspace(1) %175, align 1
  %176 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 21
  %177 = load i8, ptr addrspace(3) %176, align 1
  %178 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 21
  store i8 %177, ptr addrspace(1) %178, align 1
  %179 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 22
  %180 = load i8, ptr addrspace(3) %179, align 1
  %181 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 22
  store i8 %180, ptr addrspace(1) %181, align 1
  %182 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 23
  %183 = load i8, ptr addrspace(3) %182, align 1
  %184 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 23
  store i8 %183, ptr addrspace(1) %184, align 1
  %185 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 24
  %186 = load i8, ptr addrspace(3) %185, align 1
  %187 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 24
  store i8 %186, ptr addrspace(1) %187, align 1
  %188 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 25
  %189 = load i8, ptr addrspace(3) %188, align 1
  %190 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 25
  store i8 %189, ptr addrspace(1) %190, align 1
  %191 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 26
  %192 = load i8, ptr addrspace(3) %191, align 1
  %193 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 26
  store i8 %192, ptr addrspace(1) %193, align 1
  %194 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 27
  %195 = load i8, ptr addrspace(3) %194, align 1
  %196 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 27
  store i8 %195, ptr addrspace(1) %196, align 1
  %197 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 28
  %198 = load i8, ptr addrspace(3) %197, align 1
  %199 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 28
  store i8 %198, ptr addrspace(1) %199, align 1
  %200 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 29
  %201 = load i8, ptr addrspace(3) %200, align 1
  %202 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 29
  store i8 %201, ptr addrspace(1) %202, align 1
  %203 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 30
  %204 = load i8, ptr addrspace(3) %203, align 1
  %205 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 30
  store i8 %204, ptr addrspace(1) %205, align 1
  %206 = getelementptr inbounds nuw i8, ptr addrspace(3) @k.shared, i64 31
  %207 = load i8, ptr addrspace(3) %206, align 1
  %208 = getelementptr inbounds nuw i8, ptr addrspace(1) %115, i64 31
  store i8 %207, ptr addrspace(1) %208, align 1
  ret void
}

declare void @air.atomic.fence(i32, i32, i32)

declare i32 @air.atomic.local.cmpxchg.weak.i32(ptr addrspace(3), ptr, i32, i32, i32, i32, i1)

attributes #0 = { denormal_fpenv(preservesign) "air-kernel" "air-shared-block"="k.shared" "no-trapping-math"="true" "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.version = !{!14}
!air.language_version = !{!15}
!air.compile_options = !{!16, !17, !18}

!0 = !{i32 2, !"agx-stack-size", i32 1024}
!1 = !{i32 4, !"air.precise-math", i32 1}
!2 = !{i32 4, !"air.ieee-fp32", i32 1}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @k, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"long", !"air.arg_name", !"0"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"1"}
!14 = !{i32 2, i32 8, i32 0}
!15 = !{!"Metal", i32 4, i32 0, i32 0}
!16 = !{!"air.compile.denorms_enable"}
!17 = !{!"air.compile.fast_math_disable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{i32 4, i32 0}
