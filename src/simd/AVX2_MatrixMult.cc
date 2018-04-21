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

#include <cassert>
#include "onmt/simd/MatrixMult.h"

// This implementation is heavily inspired by reference SSE2 implementation provided
// as complementary material of paper "Sharp Models on Dull Hardware: Fast and Accurate
// Neural Machine Translation Decoding on the CPU". The implementation has been extended
// to support AVX2 instructions set - and to remove constraints on A numbers of row.

void Quantize(const float * input,
              __m256i * output,
              int num_rows,
              int width)
{
    assert(width % 16 == 0);
    
    int num_input_chunks = width / 16;
    __m256i const perm_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    // Fill an AVX float with 8 copies of the quant mult
    __m256 sse_quant_mult = _mm256_set_ps(quant_mult, quant_mult, quant_mult,
                                          quant_mult, quant_mult, quant_mult,
                                          quant_mult, quant_mult);
    
    for (int i = 0; i < num_rows; i++) {
        const float * input_row = input + i * width;
        __m256i * output_row = output + i * num_input_chunks;
        for (int j = 0; j < num_input_chunks; j++) {
            const float * x = input_row + j * 16;
            // Process 8 floats at once, since each __m256i can contain 16 16-bit integers.
            
            // Load floats into AVX registers.
            __m256 f_0 = _mm256_loadu_ps(x);
            __m256 f_1 = _mm256_loadu_ps(x + 8);
            
            // Multiply by quantization factor (e.g., if quant_mult = 1000.0, 0.34221 --> 342.21)
            __m256 m_0 = _mm256_mul_ps(f_0, sse_quant_mult);
            __m256 m_1 = _mm256_mul_ps(f_1, sse_quant_mult);
            
            // Cast float to 32-bit int (e.g., 342.21 --> 342)
            __m256i i_0 = _mm256_cvtps_epi32(m_0);
            __m256i i_1 = _mm256_cvtps_epi32(m_1);
            
            // Cast 32-bit int to 16-bit int and permute the blocks in order
            __m256i pack = _mm256_packs_epi32(i_0, i_1);
            // Need to perm - the 2 128 bits lanes are interleaved
            *(output_row + j) = _mm256_permutevar8x32_epi32(pack, perm_mask);
        }
    }
}

/* horizontal i32 add on __m256 - returns m128i register */ 
static inline __m128i _mm256i_sum8 (__m256i a)
{
    // add 2*8
    a  = _mm256_hadd_epi32(a, a);
    // add again - low and high are summed now
    a  = _mm256_hadd_epi32(a, a);
    // add low and high part
    return _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
}

void SIMD_MatrixMult(const __m256i * A,
                     const __m256i * B,
                     float * C,
                     int num_A_rows,
                     int num_B_rows,
                     int width,
                     const std::vector<size_t> &subdict)
{
    assert(width % 16 == 0);

    int avx_width = width / 16;

    // loop over A rows - 4 at a time
    int i;
    for (i = 0; i < num_A_rows - 3; i += 4)
    {
        const __m256i * A1_row = A + (i+0) * avx_width;
        const __m256i * A2_row = A + (i+1) * avx_width;
        const __m256i * A3_row = A + (i+2) * avx_width;
        const __m256i * A4_row = A + (i+3) * avx_width;

        for (int j = 0; j < num_B_rows; j++)
        {
            int B_row_idx = subdict.size() ? subdict[j] : j;
            const __m256i * B_row = B + B_row_idx * avx_width;

            __m256i sum1 = _mm256_setzero_si256();
            __m256i sum2 = _mm256_setzero_si256();
            __m256i sum3 = _mm256_setzero_si256();
            __m256i sum4 = _mm256_setzero_si256();

            // This is just a simple dot product, unrolled four ways.
            for (int k = 0; k < avx_width; k++)
            {
                __m256i b = *(B_row + k);
                
                __m256i a1 = *(A1_row + k);
                __m256i a2 = *(A2_row + k);
                __m256i a3 = *(A3_row + k);
                __m256i a4 = *(A4_row + k);

                // multiply and add
                sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(b, a1));
                sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(b, a2));
                sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(b, a3));
                sum4 = _mm256_add_epi32(sum4, _mm256_madd_epi16(b, a4));
            }
            
            // horizontal add and convert to float
            __m128i sum1_128 = _mm256i_sum8(sum1);
            __m128i sum2_128 = _mm256i_sum8(sum2);
            __m128i sum3_128 = _mm256i_sum8(sum3);
            __m128i sum4_128 = _mm256i_sum8(sum4);
            
            float * C1 = C + (i+0)*num_B_rows + j;
            float * C2 = C + (i+1)*num_B_rows + j;
            float * C3 = C + (i+2)*num_B_rows + j;
            float * C4 = C + (i+3)*num_B_rows + j;

             _mm_store_ss(C1, _mm_cvtepi32_ps(sum1_128));
            *(C1) *= unquant_mult;            
            _mm_store_ss(C2, _mm_cvtepi32_ps(sum2_128));
            *(C2) *= unquant_mult;
            _mm_store_ss(C3, _mm_cvtepi32_ps(sum3_128));
            *(C3) *= unquant_mult;
            _mm_store_ss(C4, _mm_cvtepi32_ps(sum4_128));
            *(C4) *= unquant_mult;
        }
    }
    // finalize the last rows
    switch (num_A_rows - i)
    {
        case 3:
            {
                const __m256i * A1_row = A + (i+0) * avx_width;
                const __m256i * A2_row = A + (i+1) * avx_width;
                const __m256i * A3_row = A + (i+2) * avx_width;
                for (int j = 0; j < num_B_rows; j++)
                {
                    int B_row_idx = subdict.size() ? subdict[j] : j;
                    const __m256i * B_row = B + B_row_idx * avx_width;
                    __m256i sum1 = _mm256_setzero_si256();
                    __m256i sum2 = _mm256_setzero_si256();
                    __m256i sum3 = _mm256_setzero_si256();
                    for (int k = 0; k < avx_width; k++)
                    {
                        __m256i b = *(B_row + k);
                        __m256i a1 = *(A1_row + k);
                        __m256i a2 = *(A2_row + k);
                        __m256i a3 = *(A3_row + k);
                        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(b, a1));
                        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(b, a2));
                        sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(b, a3));
                    }
                    __m128i sum1_128 = _mm256i_sum8(sum1);
                    __m128i sum2_128 = _mm256i_sum8(sum2);
                    __m128i sum3_128 = _mm256i_sum8(sum3);
                    float * C1 = C + (i+0)*num_B_rows + j;
                    float * C2 = C + (i+1)*num_B_rows + j;
                    float * C3 = C + (i+2)*num_B_rows + j;
                     _mm_store_ss(C1, _mm_cvtepi32_ps(sum1_128));
                    *(C1) *= unquant_mult;            
                    _mm_store_ss(C2, _mm_cvtepi32_ps(sum2_128));
                    *(C2) *= unquant_mult;
                    _mm_store_ss(C3, _mm_cvtepi32_ps(sum3_128));
                    *(C3) *= unquant_mult;
                }
            }
            break;
        case 2:
            {
                const __m256i * A1_row = A + (i+0) * avx_width;
                const __m256i * A2_row = A + (i+1) * avx_width;
                for (int j = 0; j < num_B_rows; j++)
                {
                    int B_row_idx = subdict.size() ? subdict[j] : j;
                    const __m256i * B_row = B + B_row_idx * avx_width;
                    __m256i sum1 = _mm256_setzero_si256();
                    __m256i sum2 = _mm256_setzero_si256();
                    for (int k = 0; k < avx_width; k++)
                    {
                        __m256i b = *(B_row + k);
                        __m256i a1 = *(A1_row + k);
                        __m256i a2 = *(A2_row + k);
                        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(b, a1));
                        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(b, a2));
                    }
                    __m128i sum1_128 = _mm256i_sum8(sum1);
                    __m128i sum2_128 = _mm256i_sum8(sum2);
                    float * C1 = C + (i+0)*num_B_rows + j;
                    float * C2 = C + (i+1)*num_B_rows + j;
                     _mm_store_ss(C1, _mm_cvtepi32_ps(sum1_128));
                    *(C1) *= unquant_mult;            
                    _mm_store_ss(C2, _mm_cvtepi32_ps(sum2_128));
                    *(C2) *= unquant_mult;
                }
            }
            break;
        case 1:
            {
                const __m256i * A1_row = A + (i+0) * avx_width;
                for (int j = 0; j < num_B_rows; j++)
                {
                    int B_row_idx = subdict.size() ? subdict[j] : j;
                    const __m256i * B_row = B + B_row_idx * avx_width;
                    __m256i sum1 = _mm256_setzero_si256();
                    for (int k = 0; k < avx_width; k++)
                    {
                        __m256i b = *(B_row + k);                    
                        __m256i a1 = *(A1_row + k);
                        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(b, a1));
                    }
                    __m128i sum1_128 = _mm256i_sum8(sum1);
                    float * C1 = C + (i+0)*num_B_rows + j;
                    _mm_store_ss(C1, _mm_cvtepi32_ps(sum1_128));
                    *(C1) *= unquant_mult;            
                }
            }
            break;
        default:
            break;
    }
}

