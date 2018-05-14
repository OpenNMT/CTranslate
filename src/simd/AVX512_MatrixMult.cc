// Copyright (c) 2017 Microsoft Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "onmt/simd/MatrixMult.h"

#include <cassert>

// This implementation is heavily inspired by reference SSE2 implementation provided
// as complementary material of paper "Sharp Models on Dull Hardware: Fast and Accurate
// Neural Machine Translation Decoding on the CPU". The implementation has been extended
// to support AVX2 instructions set - and to remove constraints on A numbers of row.

namespace onmt
{
  namespace simd
  {

    void Quantize(const float * input,
                  __m512i * output,
                  int num_rows,
                  int width)
    {
      assert(width % 32 == 0);

      int num_input_chunks = width / 32;

      // Fill an AVX float with 8 copies of the quant mult
      __m512 sse_quant_mult = _mm512_set_ps(quant_mult, quant_mult, quant_mult,
                                            quant_mult, quant_mult, quant_mult,
                                            quant_mult, quant_mult,
                                            quant_mult, quant_mult, quant_mult,
                                            quant_mult, quant_mult, quant_mult,
                                            quant_mult, quant_mult);

      for (int i = 0; i < num_rows; i++)
      {
        const float * input_row = input + i * width;
        __m512i * output_row = output + i * num_input_chunks;
        for (int j = 0; j < num_input_chunks; j++)
        {
          const float * x = input_row + j * 32;
          // Process 8 floats at once, since each __m512i can contain 16 16-bit integers.

          // Load floats into AVX registers.
          __m512 f_0 = _mm512_loadu_ps(x);
          __m512 f_1 = _mm512_loadu_ps(x + 16);

          // Multiply by quantization factor (e.g., if quant_mult = 1000.0, 0.34221 --> 342.21)
          __m512 m_0 = _mm512_mul_ps(f_0, sse_quant_mult);
          __m512 m_1 = _mm512_mul_ps(f_1, sse_quant_mult);

          // Cast float to 32-bit int (e.g., 342.21 --> 342)
          __m512i i_0 = _mm512_cvtps_epi32(m_0);
          __m512i i_1 = _mm512_cvtps_epi32(m_1);

          // Cast 32-bit int to 16-bit int
          *(output_row + j) = _mm512_packs_epi32(i_0, i_1);
        }
      }
    }

    void MatrixMult(const __m512i * A, const __m512i * B, float * C, int num_A_rows, int num_B_rows, int width)
    {
        assert(width % 32 == 0);

        int avx_width = width / 32;

        // loop over A rows - 4 at a time
        int i;
        for (i = 0; i < num_A_rows - 3; i += 4) {
            const __m512i * A1_row = A + (i+0) * avx_width;
            const __m512i * A2_row = A + (i+1) * avx_width;
            const __m512i * A3_row = A + (i+2) * avx_width;
            const __m512i * A4_row = A + (i+3) * avx_width;

            for (int j = 0; j < num_B_rows; j++) {
                const __m512i * B_row = B + j * avx_width;

                __m512i sum1 = _mm512_setzero_si512();
                __m512i sum2 = _mm512_setzero_si512();
                __m512i sum3 = _mm512_setzero_si512();
                __m512i sum4 = _mm512_setzero_si512();

                // This is just a simple dot product, unrolled four ways.
                for (int k = 0; k < avx_width; k++) {
                    __m512i b = *(B_row + k);
                    
                    __m512i a1 = *(A1_row + k);
                    __m512i a2 = *(A2_row + k);
                    __m512i a3 = *(A3_row + k);
                    __m512i a4 = *(A4_row + k);

                    // multiply and add
                    sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
                    sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(b, a2));
                    sum3 = _mm512_add_epi32(sum3, _mm512_madd_epi16(b, a3));
                    sum4 = _mm512_add_epi32(sum4, _mm512_madd_epi16(b, a4));
                }

                float * C1 = C + (i+0)*num_B_rows + j;
                (*C1) = _mm512_reduce_add_epi32(sum1)*un_quant_mult;

                float * C2 = C + (i+1)*num_B_rows + j;
                (*C2) = _mm512_reduce_add_epi32(sum2)*un_quant_mult;

                float * C3 = C + (i+2)*num_B_rows + j;
                (*C3) = _mm512_reduce_add_epi32(sum3)*un_quant_mult;

                float * C4 = C + (i+3)*num_B_rows + j;
                (*C4) = _mm512_reduce_add_epi32(sum4)*un_quant_mult;

            }
        }
      // finalize the last rows
      switch (num_A_rows - i)
      {
      case 3:
      {
        const __m512i * A1_row = A + (i+0) * avx_width;
        const __m512i * A2_row = A + (i+1) * avx_width;
        const __m512i * A3_row = A + (i+2) * avx_width;

        for (int j = 0; j < num_B_rows; j++) {
          const __m512i * B_row = B + j * avx_width;

          __m512i sum1 = _mm512_setzero_si512();
          __m512i sum2 = _mm512_setzero_si512();
          __m512i sum3 = _mm512_setzero_si512();

          // This is just a simple dot product, unrolled four ways.
          for (int k = 0; k < avx_width; k++) {
              __m512i b = *(B_row + k);
              
              __m512i a1 = *(A1_row + k);
              __m512i a2 = *(A2_row + k);
              __m512i a3 = *(A3_row + k);

              // multiply and add
              sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
              sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(b, a2));
              sum3 = _mm512_add_epi32(sum3, _mm512_madd_epi16(b, a3));
          }

          float * C1 = C + (i+0)*num_B_rows + j;
          (*C1) = _mm512_reduce_add_epi32(sum1)*un_quant_mult;

          float * C2 = C + (i+1)*num_B_rows + j;
          (*C2) = _mm512_reduce_add_epi32(sum2)*un_quant_mult;

          float * C3 = C + (i+2)*num_B_rows + j;
          (*C3) = _mm512_reduce_add_epi32(sum3)*un_quant_mult;
        }
      }
      break;
      case 2:
      {
        const __m512i * A1_row = A + (i+0) * avx_width;
        const __m512i * A2_row = A + (i+1) * avx_width;

        for (int j = 0; j < num_B_rows; j++) {
          const __m512i * B_row = B + j * avx_width;

          __m512i sum1 = _mm512_setzero_si512();
          __m512i sum2 = _mm512_setzero_si512();

          // This is just a simple dot product, unrolled four ways.
          for (int k = 0; k < avx_width; k++) {
              __m512i b = *(B_row + k);
              
              __m512i a1 = *(A1_row + k);
              __m512i a2 = *(A2_row + k);

              // multiply and add
              sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
              sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(b, a2));
          }

          float * C1 = C + (i+0)*num_B_rows + j;
          (*C1) = _mm512_reduce_add_epi32(sum1)*un_quant_mult;

          float * C2 = C + (i+1)*num_B_rows + j;
          (*C2) = _mm512_reduce_add_epi32(sum2)*un_quant_mult;
        }
      }
      break;
      case 1:
      {
        const __m512i * A1_row = A + (i+0) * avx_width;

        for (int j = 0; j < num_B_rows; j++) {
          const __m512i * B_row = B + j * avx_width;

          __m512i sum1 = _mm512_setzero_si512();

          // This is just a simple dot product, unrolled four ways.
          for (int k = 0; k < avx_width; k++) {
              __m512i b = *(B_row + k);
              
              __m512i a1 = *(A1_row + k);

              // multiply and add
              sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
          }

          float * C1 = C + (i+0)*num_B_rows + j;
          (*C1) = _mm512_reduce_add_epi32(sum1)*un_quant_mult;
        }
      }
      break;
    }
  }
}


