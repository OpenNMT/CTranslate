#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <vector>

#ifdef SIMD_SSE
# define SIMD_TYPE __m128i
# define SIMD_VSIZE 8
#elif SIMD_AVX2
# define SIMD_TYPE __m256i
# define SIMD_VSIZE 16
#else
# error "no simd type defined"
#endif

namespace onmt
{
  namespace simd
  {

    // We quantize with 10 bits of precision. This works well "universally".
    // See the top of SSE_MatricMult.cc for more info on why.
    const float quant_mult = 1000.0;

    // If we quantize to n bits and then multiple the values together, the result will be quantized to n^2 bits.
    // So we must divide by 1.0/(n^2) to get back the original value.
    const float unquant_mult = 1.0 / (quant_mult * quant_mult);

    void Quantize(const float * input,
                  SIMD_TYPE * output,
                  int num_rows,
                  int width);

    void MatrixMult(const SIMD_TYPE * A,
                    const SIMD_TYPE * B,
                    float * C,
                    int num_A_rows,
                    int num_B_rows,
                    int width,
                    const std::vector<size_t> & subdict);

  }
}
