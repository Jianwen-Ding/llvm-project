; Bitonic sort of 32 unsigned keys held one per lane of a SIMD-group (a warp sort, as in CUB's WarpMergeSort and
; top-k kernels): the 15 compare-exchange steps of the network, each a shuffle xor, a min / max and a select on the
; lane bits; then the sorted keys are stored back. No memory but the load and the store.
; agx-cost: --wg 128
target triple = "agx-apple-macosx"
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.simd.shuffle.xor.i32(i32, i32)
declare i32 @llvm.umin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
define agx_kernel void @bitonic_simd32(ptr addrspace(1) %keys) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %p = getelementptr i32, ptr addrspace(1) %keys, i32 %i
  %v0 = load i32, ptr addrspace(1) %p, align 4
  %os1 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v0, i32 1)
  %los1 = call i32 @llvm.umin.i32(i32 %v0, i32 %os1)
  %his1 = call i32 @llvm.umax.i32(i32 %v0, i32 %os1)
  %bjs1 = and i32 %lane, 1
  %lows1 = icmp eq i32 %bjs1, 0
  %bks1 = and i32 %lane, 2
  %ups1 = icmp eq i32 %bks1, 0
  %mins1 = icmp eq i1 %lows1, %ups1
  %v1 = select i1 %mins1, i32 %los1, i32 %his1
  %os2 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v1, i32 2)
  %los2 = call i32 @llvm.umin.i32(i32 %v1, i32 %os2)
  %his2 = call i32 @llvm.umax.i32(i32 %v1, i32 %os2)
  %bjs2 = and i32 %lane, 2
  %lows2 = icmp eq i32 %bjs2, 0
  %bks2 = and i32 %lane, 4
  %ups2 = icmp eq i32 %bks2, 0
  %mins2 = icmp eq i1 %lows2, %ups2
  %v2 = select i1 %mins2, i32 %los2, i32 %his2
  %os3 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v2, i32 1)
  %los3 = call i32 @llvm.umin.i32(i32 %v2, i32 %os3)
  %his3 = call i32 @llvm.umax.i32(i32 %v2, i32 %os3)
  %bjs3 = and i32 %lane, 1
  %lows3 = icmp eq i32 %bjs3, 0
  %bks3 = and i32 %lane, 4
  %ups3 = icmp eq i32 %bks3, 0
  %mins3 = icmp eq i1 %lows3, %ups3
  %v3 = select i1 %mins3, i32 %los3, i32 %his3
  %os4 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v3, i32 4)
  %los4 = call i32 @llvm.umin.i32(i32 %v3, i32 %os4)
  %his4 = call i32 @llvm.umax.i32(i32 %v3, i32 %os4)
  %bjs4 = and i32 %lane, 4
  %lows4 = icmp eq i32 %bjs4, 0
  %bks4 = and i32 %lane, 8
  %ups4 = icmp eq i32 %bks4, 0
  %mins4 = icmp eq i1 %lows4, %ups4
  %v4 = select i1 %mins4, i32 %los4, i32 %his4
  %os5 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v4, i32 2)
  %los5 = call i32 @llvm.umin.i32(i32 %v4, i32 %os5)
  %his5 = call i32 @llvm.umax.i32(i32 %v4, i32 %os5)
  %bjs5 = and i32 %lane, 2
  %lows5 = icmp eq i32 %bjs5, 0
  %bks5 = and i32 %lane, 8
  %ups5 = icmp eq i32 %bks5, 0
  %mins5 = icmp eq i1 %lows5, %ups5
  %v5 = select i1 %mins5, i32 %los5, i32 %his5
  %os6 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v5, i32 1)
  %los6 = call i32 @llvm.umin.i32(i32 %v5, i32 %os6)
  %his6 = call i32 @llvm.umax.i32(i32 %v5, i32 %os6)
  %bjs6 = and i32 %lane, 1
  %lows6 = icmp eq i32 %bjs6, 0
  %bks6 = and i32 %lane, 8
  %ups6 = icmp eq i32 %bks6, 0
  %mins6 = icmp eq i1 %lows6, %ups6
  %v6 = select i1 %mins6, i32 %los6, i32 %his6
  %os7 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v6, i32 8)
  %los7 = call i32 @llvm.umin.i32(i32 %v6, i32 %os7)
  %his7 = call i32 @llvm.umax.i32(i32 %v6, i32 %os7)
  %bjs7 = and i32 %lane, 8
  %lows7 = icmp eq i32 %bjs7, 0
  %bks7 = and i32 %lane, 16
  %ups7 = icmp eq i32 %bks7, 0
  %mins7 = icmp eq i1 %lows7, %ups7
  %v7 = select i1 %mins7, i32 %los7, i32 %his7
  %os8 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v7, i32 4)
  %los8 = call i32 @llvm.umin.i32(i32 %v7, i32 %os8)
  %his8 = call i32 @llvm.umax.i32(i32 %v7, i32 %os8)
  %bjs8 = and i32 %lane, 4
  %lows8 = icmp eq i32 %bjs8, 0
  %bks8 = and i32 %lane, 16
  %ups8 = icmp eq i32 %bks8, 0
  %mins8 = icmp eq i1 %lows8, %ups8
  %v8 = select i1 %mins8, i32 %los8, i32 %his8
  %os9 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v8, i32 2)
  %los9 = call i32 @llvm.umin.i32(i32 %v8, i32 %os9)
  %his9 = call i32 @llvm.umax.i32(i32 %v8, i32 %os9)
  %bjs9 = and i32 %lane, 2
  %lows9 = icmp eq i32 %bjs9, 0
  %bks9 = and i32 %lane, 16
  %ups9 = icmp eq i32 %bks9, 0
  %mins9 = icmp eq i1 %lows9, %ups9
  %v9 = select i1 %mins9, i32 %los9, i32 %his9
  %os10 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v9, i32 1)
  %los10 = call i32 @llvm.umin.i32(i32 %v9, i32 %os10)
  %his10 = call i32 @llvm.umax.i32(i32 %v9, i32 %os10)
  %bjs10 = and i32 %lane, 1
  %lows10 = icmp eq i32 %bjs10, 0
  %bks10 = and i32 %lane, 16
  %ups10 = icmp eq i32 %bks10, 0
  %mins10 = icmp eq i1 %lows10, %ups10
  %v10 = select i1 %mins10, i32 %los10, i32 %his10
  %os11 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v10, i32 16)
  %los11 = call i32 @llvm.umin.i32(i32 %v10, i32 %os11)
  %his11 = call i32 @llvm.umax.i32(i32 %v10, i32 %os11)
  %bjs11 = and i32 %lane, 16
  %lows11 = icmp eq i32 %bjs11, 0
  %v11 = select i1 %lows11, i32 %los11, i32 %his11
  %os12 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v11, i32 8)
  %los12 = call i32 @llvm.umin.i32(i32 %v11, i32 %os12)
  %his12 = call i32 @llvm.umax.i32(i32 %v11, i32 %os12)
  %bjs12 = and i32 %lane, 8
  %lows12 = icmp eq i32 %bjs12, 0
  %v12 = select i1 %lows12, i32 %los12, i32 %his12
  %os13 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v12, i32 4)
  %los13 = call i32 @llvm.umin.i32(i32 %v12, i32 %os13)
  %his13 = call i32 @llvm.umax.i32(i32 %v12, i32 %os13)
  %bjs13 = and i32 %lane, 4
  %lows13 = icmp eq i32 %bjs13, 0
  %v13 = select i1 %lows13, i32 %los13, i32 %his13
  %os14 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v13, i32 2)
  %los14 = call i32 @llvm.umin.i32(i32 %v13, i32 %os14)
  %his14 = call i32 @llvm.umax.i32(i32 %v13, i32 %os14)
  %bjs14 = and i32 %lane, 2
  %lows14 = icmp eq i32 %bjs14, 0
  %v14 = select i1 %lows14, i32 %los14, i32 %his14
  %os15 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v14, i32 1)
  %los15 = call i32 @llvm.umin.i32(i32 %v14, i32 %os15)
  %his15 = call i32 @llvm.umax.i32(i32 %v14, i32 %os15)
  %bjs15 = and i32 %lane, 1
  %lows15 = icmp eq i32 %bjs15, 0
  %v15 = select i1 %lows15, i32 %los15, i32 %his15
  store i32 %v15, ptr addrspace(1) %p, align 4
  ret void
}
