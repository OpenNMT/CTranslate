#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

void Quantize(const float * input, __m128i * output, float quant_mult, int num_rows, int width);
void SSE_MatrixMult(const __m128i * A, const __m128i * B, float * C,
                    float unquant_mult, int num_A_rows, int num_B_rows, int width);
