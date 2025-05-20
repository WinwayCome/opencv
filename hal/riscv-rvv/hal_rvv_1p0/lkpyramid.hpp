#ifndef OPENCV_HAL_RVV_LKPYRAMID_HPP_INCLUDED
#define OPENCV_HAL_RVV_LKPYRAMID_HPP_INCLUDED

#include <riscv_vector.h>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cassert>
namespace cv
{
    namespace cv_hal_rvv
    {

        namespace lk_pyramid
        {
#undef cv_hal_ScharrDeriv
#define cv_hal_ScharrDeriv cv::cv_hal_rvv::lk_pyramid::scharrDeriv
#undef cv_hal_LKOpticalFlowLevel
#define cv_hal_LKOpticalFlowLevel cv::cv_hal_rvv::lk_pyramid::lkOpticalFlowLevel
            // #ifndef CV_HAL_ERROR_OK
            // #define CV_HAL_ERROR_OK 0
            // #endif
            // #ifndef CV_HAL_ERROR_NOT_IMPLEMENTED
            // #define CV_HAL_ERROR_NOT_IMPLEMENTED 1
            // #endif
            // #ifndef CV_HAL_ERROR_UNKNOWN
            // #define CV_HAL_ERROR_UNKNOWN -1
            // #endif
            // #ifndef FLT_EPSILON
            // #define FLT_EPSILON 1.19209290e-7f
            // #endif
            // #ifndef uchar
            // #define uchar unsigned char
            // #endif
            // #ifndef size_t
            // #define size_t int64_t
            // #endif

            typedef short deriv_type;

            // scharrDeriv-----------------------------------------------
            inline size_t myAlignSize(size_t sz, int n = 16)
            {
                assert((n & (n - 1)) == 0);
                return (sz + n - 1) & ~(size_t)(n - 1);
            }

            template <typename T>
            inline T *myAlignPtr(T *ptr, int n = (int)sizeof(T))
            {
                assert((n & (n - 1)) == 0);
                return reinterpret_cast<T *>(
                    (reinterpret_cast<size_t>(ptr) + n - 1) & ~(size_t)(n - 1));
            }
            inline int scharrDeriv(const uchar *src_data, size_t src_step,
                                   short *dst_data, size_t dst_step,
                                   int width, int height, int cn)
            {
                const int colsn = width * cn;
                const int delta = static_cast<int>(
                    myAlignSize(static_cast<std::size_t>((width + 2) * cn), 16));

                std::vector<deriv_type> buf(delta * 2 + 32);
                deriv_type *trow0 = myAlignPtr(buf.data() + cn, 16);
                deriv_type *trow1 = myAlignPtr(trow0 + delta, 16);

                for (int y = 0; y < height; ++y)
                {
                    const uchar *srow0 = src_data + ((y > 0)        ? (y - 1) * src_step
                                                     : (height > 1) ? 1 * src_step
                                                                    : 0);
                    const uchar *srow1 = src_data + y * src_step;
                    const uchar *srow2 = src_data + ((y < height - 1) ? (y + 1) * src_step
                                                     : (height > 1)   ? (height - 2) * src_step
                                                                      : 0);

                    deriv_type *drow = reinterpret_cast<deriv_type *>(
                        reinterpret_cast<uchar *>(dst_data) + y * dst_step);

                    size_t vl = __riscv_vsetvl_e8m4(colsn);

                    for (size_t x = 0; x < colsn; x += vl)
                    {
                        vl = std::min(vl, colsn - x);
                        auto s1_u8 = __riscv_vle8_v_u8m4(srow1 + x, vl);
                        auto s0_u8 = __riscv_vle8_v_u8m4(srow0 + x, vl);
                        auto s2_u8 = __riscv_vle8_v_u8m4(srow2 + x, vl);

                        auto s1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vzext_vf2_u16m8(s1_u8, vl));
                        auto s0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vzext_vf2_u16m8(s0_u8, vl));
                        auto s2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vzext_vf2_u16m8(s2_u8, vl));
                        // 12

                        auto partB = __riscv_vmul_vx_i16m8(s1, 10, vl);
                        auto sum02 = __riscv_vadd_vv_i16m8(s0, s2, vl);
                        auto t1 = __riscv_vsub_vv_i16m8(s2, s0, vl);
                        auto partA = __riscv_vmul_vx_i16m8(sum02, 3, vl);

                        auto t0 = __riscv_vadd_vv_i16m8(partA, partB, vl);

                        __riscv_vse16_v_i16m8(trow0 + x, t0, vl);
                        __riscv_vse16_v_i16m8(trow1 + x, t1, vl);
                    }

                    int x0 = (width > 1 ? 1 : 0) * cn;
                    int x1 = (width > 1 ? width - 2 : 0) * cn;
                    for (int k = 0; k < cn; ++k)
                    {
                        trow0[-cn + k] = trow0[x0 + k];
                        trow0[colsn + k] = trow0[x1 + k];
                        trow1[-cn + k] = trow1[x0 + k];
                        trow1[colsn + k] = trow1[x1 + k];
                    }

                    vl = __riscv_vsetvl_e16m4(colsn);
                    for (size_t x = 0; x < colsn; x += vl)
                    {
                        vl = std::min(vl, colsn - x);
                        auto t0_prev = __riscv_vle16_v_i16m4(trow0 + x - cn, vl);
                        auto t0_next = __riscv_vle16_v_i16m4(trow0 + x + cn, vl);

                        auto t1_prev = __riscv_vle16_v_i16m4(trow1 + x - cn, vl);
                        auto t1_curr = __riscv_vle16_v_i16m4(trow1 + x, vl);
                        auto t1_next = __riscv_vle16_v_i16m4(trow1 + x + cn, vl);

                        auto dx = __riscv_vsub_vv_i16m4(t0_next, t0_prev, vl);

                        auto sum_prev_next = __riscv_vadd_vv_i16m4(t1_prev, t1_next, vl);

                        auto part3 = __riscv_vmul_vx_i16m4(sum_prev_next, 3, vl);
                        auto part10 = __riscv_vmul_vx_i16m4(t1_curr, 10, vl);
                        auto dy = __riscv_vadd_vv_i16m4(part3, part10, vl);

                        vint16m4x2_t pack;
                        pack = __riscv_vset_v_i16m4_i16m4x2(pack, 0, dx);
                        pack = __riscv_vset_v_i16m4_i16m4x2(pack, 1, dy);
                        __riscv_vsseg2e16_v_i16m4x2(drow + 2 * x, pack, vl);
                    }
                }
                return CV_HAL_ERROR_OK;
            }

            inline int lkOpticalFlowLevel(
                const uchar *prev_data, size_t prev_data_step,
                const short *prev_deriv_data, size_t prev_deriv_step,
                const uchar *next_data, size_t next_step,
                int width, int height, int cn,
                const float *prev_points, float *next_points, size_t point_count,
                uchar *status, float *err,
                int win_width, int win_height,
                int termination_count, double termination_epsilon,
                bool get_min_eigen_vals,
                float min_eigen_vals_threshold)
            {
                const int W_BITS = 14;
                const int W_BITS1 = 14;
                const float FLT_SCALE = 1.f / (1 << 20);

                // halfWin.x and halfWin.y
                const float halfWinX = (win_width - 1) * 0.5f;
                const float halfWinY = (win_height - 1) * 0.5f;

                int cn2 = cn * 2;
                int dstep = int(prev_deriv_step / sizeof(int16_t));
                int stepI = int(prev_data_step);
                int stepJ = int(next_step);

                std::size_t buf_elems = std::size_t(win_width) * win_height * (cn + cn2);
                std::vector<int16_t> buffer(buf_elems);
                int16_t *IWinBuf = buffer.data();
                int16_t *derivIWinBuf = IWinBuf + win_width * win_height * cn;

                for (std::size_t idx = 0; idx < point_count; ++idx)
                {
                    float px = prev_points[2 * idx] - halfWinX;
                    float py = prev_points[2 * idx + 1] - halfWinY;
                    int ipx = int(std::floor(px));
                    int ipy = int(std::floor(py));

                    if (ipx < -win_width || ipx >= width || ipy < -win_height || ipy >= height)
                    {
                        if (status)
                            status[idx] = 0;
                        if (err)
                            err[idx] = 0.f;
                        continue;
                    }

                    float a = px - ipx;
                    float b = py - ipy;
                    int iw00 = int(std::round((1.f - a) * (1.f - b) * (1 << W_BITS)));
                    int iw01 = int(std::round(a * (1.f - b) * (1 << W_BITS)));
                    int iw10 = int(std::round((1.f - a) * b * (1 << W_BITS)));
                    int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

                    int64_t iA11 = 0, iA12 = 0, iA22 = 0;
                    for (int y = 0; y < win_height; ++y)
                    {
                        const uint8_t *srcRow = prev_data + (ipy + y) * prev_data_step + ipx * cn;
                        const int16_t *dsrcRow = prev_deriv_data + (ipy + y) * dstep + ipx * cn2;
                        int16_t *Iptr = IWinBuf + y * win_width * cn;
                        int16_t *dIptr = derivIWinBuf + y * win_width * cn2;

                        size_t width = win_width * cn;
                        size_t vl = __riscv_vsetvl_e8m2(width);
                        for (int x = 0; x < width; x += vl)
                        {
                            vl = std::min(vl, width - x);
                            auto link00 = __riscv_vle8_v_u8m2(srcRow + x, vl);
                            auto link01 = __riscv_vle8_v_u8m2(srcRow + x + cn, vl);
                            auto link10 = __riscv_vle8_v_u8m2(srcRow + x + stepI, vl);
                            auto link11 = __riscv_vle8_v_u8m2(srcRow + x + stepI + cn, vl);
                            

                            auto lv00 = __riscv_vzext_vf2_u16m4(link00, vl);
                            auto lv01 = __riscv_vzext_vf2_u16m4(link01, vl);
                            auto lv10 = __riscv_vzext_vf2_u16m4(link10, vl);
                            auto lv11 = __riscv_vzext_vf2_u16m4(link11, vl);
                            

                            auto v00 = __riscv_vreinterpret_v_u16m4_i16m4(lv00);
                            auto v01 = __riscv_vreinterpret_v_u16m4_i16m4(lv01);
                            auto v10 = __riscv_vreinterpret_v_u16m4_i16m4(lv10);
                            auto v11 = __riscv_vreinterpret_v_u16m4_i16m4(lv11);
                            

                            auto vsum = __riscv_vmv_v_x_i32m8(1 << (W_BITS1 - 5 - 1), vl); // 8 + 4 = 12
                            vsum = __riscv_vwmacc_vx_i32m8(vsum, (int16_t)iw00, v00, vl);
                            vsum = __riscv_vwmacc_vx_i32m8(vsum, (int16_t)iw01, v01, vl);
                            vsum = __riscv_vwmacc_vx_i32m8(vsum, (int16_t)iw10, v10, vl);
                            vsum = __riscv_vwmacc_vx_i32m8(vsum, (int16_t)iw11, v11, vl);
                            

                            auto v_result = __riscv_vnclip_wx_i16m4(vsum, W_BITS1 - 5, 0, vl);
                            

                            __riscv_vse16_v_i16m4(Iptr + x, v_result, vl);
                        }
                        width = win_width * cn2;
                        vl = __riscv_vsetvl_e16m4(width);
                        for (int x = 0; x < width; x += vl)
                        {
                            vl = std::min(vl, width - x);

                            auto g00 = __riscv_vle16_v_i16m4(dsrcRow + x, vl);
                            auto g01 = __riscv_vle16_v_i16m4(dsrcRow + x + cn2, vl);
                            auto g10 = __riscv_vle16_v_i16m4(dsrcRow + x + dstep, vl);
                            auto g11 = __riscv_vle16_v_i16m4(dsrcRow + x + dstep + cn2, vl);

                            auto mix32 = __riscv_vmv_v_x_i32m8(1 << (W_BITS1 - 1), vl);
                            mix32 = __riscv_vwmacc_vx_i32m8(mix32, (int16_t)iw00, g00, vl);
                            mix32 = __riscv_vwmacc_vx_i32m8(mix32, (int16_t)iw01, g01, vl);
                            mix32 = __riscv_vwmacc_vx_i32m8(mix32, (int16_t)iw10, g10, vl);
                            mix32 = __riscv_vwmacc_vx_i32m8(mix32, (int16_t)iw11, g11, vl);

                            auto mix16 = __riscv_vnclip_wx_i16m4(mix32, W_BITS1, 0, vl);
                            __riscv_vse16_v_i16m4(dIptr + x, mix16, vl);
                        }
                        vl = __riscv_vsetvl_e16m8(width);
                        size_t vl_half = vl >> 1;
                        for (int x = 0; x < width; x += vl)
                        {
                            if (width - x < vl)
                            {
                                vl_half = (width - x) >> 1;
                                if (vl_half == 0)
                                    continue;
                            }
                            auto pair = __riscv_vlseg2e16_v_i16m4x2(dIptr + x, vl_half);
                            auto Ix16 = __riscv_vget_v_i16m4x2_i16m4(pair, 0);
                            auto Iy16 = __riscv_vget_v_i16m4x2_i16m4(pair, 1);

                            auto fx_i32 = __riscv_vwsub_vx_i32m8(Ix16, 0, vl_half);
                            auto fy_i32 = __riscv_vwsub_vx_i32m8(Iy16, 0, vl_half);

                            auto fx_f32 = __riscv_vfcvt_f_x_v_f32m8(fx_i32, vl_half);
                            auto fy_f32 = __riscv_vfcvt_f_x_v_f32m8(fy_i32, vl_half);

                            auto zero_f = __riscv_vfmv_v_f_f32m1(0.f, vl_half);

                            auto acc11 = __riscv_vfmv_v_f_f32m8(0.f, vl_half);
                            acc11 = __riscv_vfmacc_vv_f32m8(acc11, fx_f32, fx_f32, vl_half);
                            auto sum11f = __riscv_vfredusum_vs_f32m8_f32m1(acc11, zero_f, vl_half);
                            iA11 += __riscv_vmv_x_s_i32m1_i32(__riscv_vfcvt_x_f_v_i32m1(sum11f, 1));

                            auto acc12 = __riscv_vfmv_v_f_f32m8(0.f, vl_half);
                            acc12 = __riscv_vfmacc_vv_f32m8(acc12, fx_f32, fy_f32, vl_half);
                            auto sum12f = __riscv_vfredusum_vs_f32m8_f32m1(acc12, zero_f, vl_half);
                            iA12 += __riscv_vmv_x_s_i32m1_i32(__riscv_vfcvt_x_f_v_i32m1(sum12f, 1));

                            auto acc22 = __riscv_vfmv_v_f_f32m8(0.f, vl_half);
                            acc22 = __riscv_vfmacc_vv_f32m8(acc22, fy_f32, fy_f32, vl_half);
                            auto sum22f = __riscv_vfredusum_vs_f32m8_f32m1(acc22, zero_f, vl_half);
                            iA22 += __riscv_vmv_x_s_i32m1_i32(__riscv_vfcvt_x_f_v_i32m1(sum22f, 1));
                        }
                    }

                    float A11 = iA11 * FLT_SCALE;
                    float A12 = iA12 * FLT_SCALE;
                    float A22 = iA22 * FLT_SCALE;
                    float D = A11 * A22 - A12 * A12;
                    float minEig = (A22 + A11 - std::sqrt((A11 - A22) * (A11 - A22) + 4.f * A12 * A12)) / (2 * win_width * win_height);

                    if (err && get_min_eigen_vals)
                        err[idx] = minEig;
                    if (minEig < min_eigen_vals_threshold || D < FLT_EPSILON)
                    {
                        if (status)
                            status[idx] = 0;
                        continue;
                    }
                    D = 1.f / D;

                    float nx = next_points[2 * idx] - halfWinX;
                    float ny = next_points[2 * idx + 1] - halfWinY;

                    float prev_dx = 0, prev_dy = 0;
                    for (int it = 0; it < termination_count; ++it)
                    {
                        int inx = int(std::floor(nx));
                        int iny = int(std::floor(ny));
                        if (inx < -win_width || inx >= width || iny < -win_height || iny >= height)
                        {
                            if (status)
                                status[idx] = 0;
                            break;
                        }

                        float aa = nx - inx;
                        float bb = ny - iny;
                        int jw00 = int(std::round((1.f - aa) * (1.f - bb) * (1 << W_BITS)));
                        int jw01 = int(std::round(aa * (1.f - bb) * (1 << W_BITS)));
                        int jw10 = int(std::round((1.f - aa) * bb * (1 << W_BITS)));
                        int jw11 = (1 << W_BITS) - jw00 - jw01 - jw10;

                        int64_t ib1 = 0, ib2 = 0;

                        for (int y = 0; y < win_height; ++y)
                        {
                            const uint8_t *JptrRow = next_data + (iny + y) * next_step + inx * cn;
                            const int16_t *IptrRow = IWinBuf + y * win_width * cn;
                            const int16_t *dIptrRow = derivIWinBuf + y * win_width * cn2;

                            int vl = 0;

                            for (int x = 0; x < win_width * cn; x += vl, dIptrRow += 2 * vl)
                            {
                                vl = __riscv_vsetvl_e8m2(win_width * cn - x);

                                auto Jlink00 = __riscv_vle8_v_u8m2(JptrRow + x, vl);
                                auto Jlink01 = __riscv_vle8_v_u8m2(JptrRow + x + cn, vl);
                                auto Jlink10 = __riscv_vle8_v_u8m2(JptrRow + x + stepJ, vl);
                                auto Jlink11 = __riscv_vle8_v_u8m2(JptrRow + x + stepJ + cn, vl);

                                auto I_val = __riscv_vle16_v_i16m4(IptrRow + x, vl);

                                auto Jlv00 = __riscv_vzext_vf2_u16m4(Jlink00, vl);
                                auto Jlv01 = __riscv_vzext_vf2_u16m4(Jlink01, vl);
                                auto Jlv10 = __riscv_vzext_vf2_u16m4(Jlink10, vl);
                                auto Jlv11 = __riscv_vzext_vf2_u16m4(Jlink11, vl);

                                auto Jv00 = __riscv_vreinterpret_v_u16m4_i16m4(Jlv00);
                                auto Jv01 = __riscv_vreinterpret_v_u16m4_i16m4(Jlv01);
                                auto Jv10 = __riscv_vreinterpret_v_u16m4_i16m4(Jlv10);
                                auto Jv11 = __riscv_vreinterpret_v_u16m4_i16m4(Jlv11);

                                auto v_sum = __riscv_vmv_v_x_i32m8(1 << (W_BITS1 - 5 - 1), vl);
                                v_sum = __riscv_vwmacc_vx_i32m8(v_sum, (int16_t)jw00, Jv00, vl);
                                v_sum = __riscv_vwmacc_vx_i32m8(v_sum, (int16_t)jw01, Jv01, vl);
                                v_sum = __riscv_vwmacc_vx_i32m8(v_sum, (int16_t)jw10, Jv10, vl);
                                v_sum = __riscv_vwmacc_vx_i32m8(v_sum, (int16_t)jw11, Jv11, vl);
                                // 8

                                auto It_val = __riscv_vnclip_wx_i16m4(v_sum, W_BITS1 - 5, 0, vl);
                                auto It_diff = __riscv_vsub_vv_i16m4(It_val, I_val, vl);

                                auto grad = __riscv_vlseg2e16_v_i16m4x2(dIptrRow, vl);
                                auto Ix = __riscv_vget_v_i16m4x2_i16m4(grad, 0);
                                auto Iy = __riscv_vget_v_i16m4x2_i16m4(grad, 1); 

                                auto fx = __riscv_vwmul_vv_i32m8(It_diff, Ix, vl);
                                auto fy = __riscv_vwmul_vv_i32m8(It_diff, Iy, vl);

                                auto fx_f = __riscv_vfcvt_f_x_v_f32m8(fx, vl);
                                auto fy_f = __riscv_vfcvt_f_x_v_f32m8(fy, vl);

                                auto zero_f = __riscv_vfmv_v_f_f32m1(0.0f, vl); // 17
                                auto sum_b1_f = __riscv_vfredusum_vs_f32m8_f32m1(fx_f, zero_f, vl);
                                auto sum_b2_f = __riscv_vfredusum_vs_f32m8_f32m1(fy_f, zero_f, vl);
                                auto sum_b1_i = __riscv_vfcvt_x_f_v_i32m1(sum_b1_f, 1);
                                auto sum_b2_i = __riscv_vfcvt_x_f_v_i32m1(sum_b2_f, 1);
                                ib1 += __riscv_vmv_x_s_i32m1_i32(sum_b1_i);
                                ib2 += __riscv_vmv_x_s_i32m1_i32(sum_b2_i);
                            }
                        }
                        float b1 = ib1 * FLT_SCALE;
                        float b2 = ib2 * FLT_SCALE;

                        float dx = (A12 * b2 - A22 * b1) * D;
                        float dy = (A12 * b1 - A11 * b2) * D;
                        nx += dx;
                        ny += dy;

                        next_points[2 * idx] = nx + halfWinX;
                        next_points[2 * idx + 1] = ny + halfWinY;

                        if (dx * dx + dy * dy <= termination_epsilon)
                            break;
                        if (it > 0 && std::fabs(dx + prev_dx) < 0.01f && std::fabs(dy + prev_dy) < 0.01f)
                        {
                            next_points[2 * idx] -= dx * 0.5f;
                            next_points[2 * idx + 1] -= dy * 0.5f;
                            break;
                        }
                        prev_dx = dx;
                        prev_dy = dy;
                    }

                    if (status && err && !get_min_eigen_vals)
                    {
                        int inx = int(std::floor(nx));
                        int iny = int(std::floor(ny));
                        if (inx < -win_width || inx >= width || iny < -win_height || iny >= height)
                        {
                            status[idx] = 0;
                            continue;
                        }
                        float aa = nx - inx;
                        float bb = ny - iny;
                        int jw00 = int(std::round((1.f - aa) * (1.f - bb) * (1 << W_BITS)));
                        int jw01 = int(std::round(aa * (1.f - bb) * (1 << W_BITS)));
                        int jw10 = int(std::round((1.f - aa) * bb * (1 << W_BITS)));
                        int jw11 = (1 << W_BITS) - jw00 - jw01 - jw10;

                        float errval = 0.f;
                        for (int y = 0; y < win_height; ++y)
                        {
                            const uint8_t *JptrRow = next_data + (iny + y) * next_step + inx * cn;
                            const int16_t *IptrRow = IWinBuf + y * win_width * cn;
                            size_t vl = 0;
                            for (int x = 0; x < win_width * cn; x += vl)
                            {
                                vl = __riscv_vsetvl_e8m2(win_width * cn - x);

                                auto j00_u8 = __riscv_vle8_v_u8m2(JptrRow + x, vl);
                                auto j01_u8 = __riscv_vle8_v_u8m2(JptrRow + x + cn, vl);
                                auto j10_u8 = __riscv_vle8_v_u8m2(JptrRow + x + stepJ, vl);
                                auto j11_u8 = __riscv_vle8_v_u8m2(JptrRow + x + stepJ + cn, vl);

                                auto j00_u16 = __riscv_vzext_vf2_u16m4(j00_u8, vl);
                                auto j01_u16 = __riscv_vzext_vf2_u16m4(j01_u8, vl);
                                auto j10_u16 = __riscv_vzext_vf2_u16m4(j10_u8, vl);
                                auto j11_u16 = __riscv_vzext_vf2_u16m4(j11_u8, vl);

                                auto j00 = __riscv_vreinterpret_v_u16m4_i16m4(j00_u16);
                                auto j01 = __riscv_vreinterpret_v_u16m4_i16m4(j01_u16);
                                auto j10 = __riscv_vreinterpret_v_u16m4_i16m4(j10_u16);
                                auto j11 = __riscv_vreinterpret_v_u16m4_i16m4(j11_u16);

                                auto acc = __riscv_vmv_v_x_i32m8(1 << (W_BITS1 - 5 - 1), vl);
                                acc = __riscv_vwmacc_vx_i32m8(acc, jw00, j00, vl);
                                acc = __riscv_vwmacc_vx_i32m8(acc, jw01, j01, vl);
                                acc = __riscv_vwmacc_vx_i32m8(acc, jw10, j10, vl);
                                acc = __riscv_vwmacc_vx_i32m8(acc, jw11, j11, vl);

                                auto interp = __riscv_vsra_vx_i32m8(acc, W_BITS1 - 5, vl);

                                auto I16 = __riscv_vle16_v_i16m4(IptrRow + x, vl);
                                auto I32 = __riscv_vwsub_vx_i32m8(I16, 0, vl);

                                auto diff = __riscv_vsub_vv_i32m8(interp, I32, vl);

                                auto diff_f = __riscv_vfcvt_f_x_v_f32m8(diff, vl);
                                auto abs_f = __riscv_vfabs_v_f32m8(diff_f, vl);

                                auto sum_f = __riscv_vfredusum_vs_f32m8_f32m1(abs_f, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
                                errval += __riscv_vfmv_f_s_f32m1_f32(sum_f);
                            }
                        }
                        err[idx] = errval / (32.f * win_width * cn * win_height);
                    }
                }

                return CV_HAL_ERROR_OK;
            }

        }
    }

}

#endif // OPENCV_HAL_RVV_LKPYRAMID_HPP_INCLUDED