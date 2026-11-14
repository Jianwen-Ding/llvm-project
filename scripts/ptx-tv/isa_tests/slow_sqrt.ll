define float @sqrt_rz_f32(float %a) {
  %d = call float asm "sqrt.rz.f32 $0, $1;", "=f,f"(float %a)
  ret float %d
}

define double @sqrt_rp_f64(double %a) {
  %d = call double asm "sqrt.rp.f64 $0, $1;", "=d,d"(double %a)
  ret double %d
}
