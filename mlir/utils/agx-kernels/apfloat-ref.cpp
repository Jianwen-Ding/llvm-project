// apfloat-ref TYPE MODE < hex inputs: LLVM APFloat's conversions and arithmetic of the
// small float types, the reference of agx-small-float-check.py. MODE: ext (a code to f32
// bits), trunc32 / trunc64 (f32 / f64 bits to a code), add / mul / div / mad (two codes,
// the second W bits up; mad is (x * y) + x rounded twice). Build:
//   clang++ -std=c++17 -I llvm/include -I BUILD/include apfloat-ref.cpp \
//     BUILD/lib/libLLVMSupport.a BUILD/lib/libLLVMDemangle.a -lpthread -o apfloat-ref
#include "llvm/ADT/APFloat.h"
#include <cstdio>
#include <cstring>
#include <string>
using namespace llvm;
static const fltSemantics *sem(const char *n) {
#define T(N, F) if (!strcmp(n, N)) return &APFloat::F();
  T("f8E4M3FN", Float8E4M3FN) T("f8E5M2", Float8E5M2) T("f8E4M3FNUZ", Float8E4M3FNUZ)
  T("f8E5M2FNUZ", Float8E5M2FNUZ) T("f8E4M3B11FNUZ", Float8E4M3B11FNUZ) T("f8E3M4", Float8E3M4)
  T("f8E4M3", Float8E4M3) T("f6E2M3FN", Float6E2M3FN) T("f6E3M2FN", Float6E3M2FN) T("f4E2M1FN", Float4E2M1FN) T("f8E8M0FNU", Float8E8M0FNU)
  return nullptr;
}
int main(int argc, char **argv) {
  const fltSemantics *S = sem(argv[1]);
  std::string mode = argv[2];
  unsigned W = APFloat::semanticsSizeInBits(*S);
  unsigned long long v;
  while (scanf("%llx", &v) == 1) {
    bool lost;
    if (mode == "mad") {
      unsigned long long u = v & ((1ull << W) - 1), w2 = v >> W;
      APFloat f(*S, APInt(W, u)), g(*S, APInt(W, w2)), x0 = f;
      f.multiply(g, APFloat::rmNearestTiesToEven);
      f.add(x0, APFloat::rmNearestTiesToEven);
      printf("%llx %d\n", (unsigned long long)f.bitcastToAPInt().getZExtValue(), (int)f.isNaN());
      continue;
    }
    if (mode == "add" || mode == "mul" || mode == "div") {
      unsigned long long u = v & ((1ull << W) - 1), w2 = v >> W;
      APFloat f(*S, APInt(W, u)), g(*S, APInt(W, w2));
      if (mode == "add") f.add(g, APFloat::rmNearestTiesToEven);
      else if (mode == "mul") f.multiply(g, APFloat::rmNearestTiesToEven);
      else f.divide(g, APFloat::rmNearestTiesToEven);
      printf("%llx %d\n", (unsigned long long)f.bitcastToAPInt().getZExtValue(), (int)f.isNaN());
      continue;
    }
    if (mode == "ext") {
      APFloat f(*S, APInt(W, v));
      f.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &lost);
      printf("%llx\n", (unsigned long long)f.bitcastToAPInt().getZExtValue());
    } else {
      bool d = mode == "trunc64";
      APFloat f(d ? APFloat::IEEEdouble() : APFloat::IEEEsingle(), APInt(d ? 64 : 32, v));
      if ((f.isNaN() || f.isInfinity()) && APFloat::semanticsHasNaN(*S) == false && !APFloat::semanticsHasInf(*S)) { printf("skip\n"); continue; }
      if (f.isInfinity() && !APFloat::semanticsHasInf(*S) && APFloat::semanticsHasNaN(*S)) { printf("%llx\n", (unsigned long long)APFloat::getNaN(*S, f.isNegative()).bitcastToAPInt().getZExtValue()); continue; }
      f.convert(*S, APFloat::rmNearestTiesToEven, &lost);
      printf("%llx\n", (unsigned long long)f.bitcastToAPInt().getZExtValue());
    }
  }
}
