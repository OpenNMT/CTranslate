#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

// We quantize with 10 bits of precision. This works well "universally".
// See the top of this file for more info on why.
//double quant_mult = pow(2.0, 10.0);
const float quant_mult = 1000.0;

// If we quantize to n bits and then multiple the values together, the result will be quantized to n^2 bits.
// So we must divide by 1.0/(n^2) to get back the original value.
const float unquant_mult = 1.0 / (quant_mult * quant_mult);


void Quantize(const float * input,
              __m128i * output,
              int num_rows,
              int width);

void SSE_MatrixMult(const __m128i * A,
                    const __m128i * B,
                    float * C,
                    int num_A_rows,
                    int num_B_rows,
                    int width);
