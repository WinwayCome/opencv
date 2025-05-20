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
                if (!src_data || !dst_data || width <= 0 || height <= 0 || cn <= 0)
                    return -1;

                const int colsn = width * cn;
                const int delta = static_cast<int>(
                    myAlignSize(static_cast<size_t>((width + 2) * cn), 16));

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

                    for (int x = 0; x < colsn; ++x)
                    {
                        int t0 = (srow0[x] + srow2[x]) * 3 + srow1[x] * 10;
                        int t1 = static_cast<int>(srow2[x]) - static_cast<int>(srow0[x]);
                        trow0[x] = static_cast<deriv_type>(t0);
                        trow1[x] = static_cast<deriv_type>(t1);
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

                    for (int x = 0; x < colsn; ++x)
                    {
                        deriv_type dx = static_cast<deriv_type>(trow0[x + cn] - trow0[x - cn]);
                        deriv_type dy = static_cast<deriv_type>(
                            (trow1[x + cn] + trow1[x - cn]) * 3 + trow1[x] * 10);

                        drow[2 * x] = dx;
                        drow[2 * x + 1] = dy;
                    }
                }
                return CV_HAL_ERROR_OK;
            }

            inline int lkOpticalFlowLevel(
                const uchar* prev_data, size_t prev_data_step,
                const short *prev_deriv_data, size_t prev_deriv_step,
                const uchar* next_data, size_t next_step,
                int width, int height, int cn,
                const float *prev_points, float *next_points, size_t point_count,
                uchar* status, float *err,
                int win_width, int win_height,
                int termination_count, double termination_epsilon,
                bool get_min_eigen_vals,
                float min_eigen_vals_threshold)
            {
                bool printed_debug = false;
                const int W_BITS = 14;
                const int W_BITS1 = 14;
                const float FLT_SCALE = 1.f / (1 << 20);

                // halfWin.x and halfWin.y
                const float halfWinX = (win_width - 1) * 0.5f;
                const float halfWinY = (win_height - 1) * 0.5f;

                int cn2 = cn * 2;
                int dstep = int(prev_deriv_step / sizeof(short));
                int stepI = int(prev_data_step); // bytes per row in prev_data
                int stepJ = int(next_step);      // bytes per row in next_data

                // allocate buffer for IWinBuf and derivIWinBuf
                size_t buf_elems = size_t(win_width) * win_height * (cn + cn2);
                std::vector<short> buffer(buf_elems);
                short *IWinBuf = buffer.data();
                short *derivIWinBuf = IWinBuf + win_width * win_height * cn;

                // iterate each point
                for (size_t idx = 0; idx < point_count; ++idx)
                {
                    // read and shift previous point
                    float px = prev_points[2 * idx] - halfWinX;
                    float py = prev_points[2 * idx + 1] - halfWinY;
                    int ipx = int(std::floor(px));
                    int ipy = int(std::floor(py));

                    // boundary check
                    if (ipx < -win_width || ipx >= width || ipy < -win_height || ipy >= height)
                    {
                        if (status)
                            status[idx] = 0;
                        if (err)
                            err[idx] = 0.f;
                        continue;
                    }

                    // compute interpolation weights
                    float a = px - ipx;
                    float b = py - ipy;
                    int iw00 = int(std::round((1.f - a) * (1.f - b) * (1 << W_BITS)));
                    int iw01 = int(std::round(a * (1.f - b) * (1 << W_BITS)));
                    int iw10 = int(std::round((1.f - a) * b * (1 << W_BITS)));
                    int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

                    // compute structure tensor accumulators
                    int64_t iA11 = 0, iA12 = 0, iA22 = 0;
                    for (int y = 0; y < win_height; ++y)
                    {
                        const uchar* srcRow = prev_data + (ipy + y) * prev_data_step + ipx * cn;
                        const short *dsrcRow = prev_deriv_data + (ipy + y) * dstep + ipx * cn2;
                        short *Iptr = IWinBuf + y * win_width * cn;
                        short *dIptr = derivIWinBuf + y * win_width * cn2;

                        for (int x = 0; x < win_width * cn; ++x, dsrcRow += 2, dIptr += 2)
                        {
                            int ival = ((srcRow[x] * iw00 + srcRow[x + cn] * iw01 + srcRow[x + stepI] * iw10 + srcRow[x + stepI + cn] * iw11) >> (W_BITS1 - 5));
                            int ixval = ((dsrcRow[0] * iw00 + dsrcRow[cn2] * iw01 + dsrcRow[dstep] * iw10 + dsrcRow[dstep + cn2] * iw11) >> W_BITS1);
                            int iyval = ((dsrcRow[1] * iw00 + dsrcRow[cn2 + 1] * iw01 + dsrcRow[dstep + 1] * iw10 + dsrcRow[dstep + cn2 + 1] * iw11) >> W_BITS1);

                            Iptr[x] = short(ival);
                            dIptr[0] = short(ixval);
                            dIptr[1] = short(iyval);

                            iA11 += int64_t(ixval) * ixval;
                            iA12 += int64_t(ixval) * iyval;
                            iA22 += int64_t(iyval) * iyval;
                        }
                    }

                    // form the 2×2 symmetric matrix
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

                    // prepare next point, shifted by half window
                    float nx = next_points[2 * idx] - halfWinX;
                    float ny = next_points[2 * idx + 1] - halfWinY;

                    // iterative refinement
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
                            const uchar* JptrRow = next_data + (iny + y) * next_step + inx * cn;
                            const short *IptrRow = IWinBuf + y * win_width * cn;
                            const short *dIptrRow = derivIWinBuf + y * win_width * cn2;

                            for (int x = 0; x < win_width * cn; ++x, dIptrRow += 2)
                            {
                                int diff = (((JptrRow[x] * jw00 + JptrRow[x + cn] * jw01 + JptrRow[x + stepJ] * jw10 + JptrRow[x + stepJ + cn] * jw11) >> (W_BITS1 - 5)) - IptrRow[x]);
                                ib1 += int64_t(diff) * dIptrRow[0];
                                ib2 += int64_t(diff) * dIptrRow[1];
                            }
                        }
                        float b1 = ib1 * FLT_SCALE;
                        float b2 = ib2 * FLT_SCALE;

                        // solve for delta
                        float dx = (A12 * b2 - A22 * b1) * D;
                        float dy = (A12 * b1 - A11 * b2) * D;
                        nx += dx;
                        ny += dy;

                        // write back next_points
                        next_points[2 * idx] = nx + halfWinX;
                        next_points[2 * idx + 1] = ny + halfWinY;

                        // stopping criteria
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

                    // final error estimation when not using min eigvals
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
                            const uchar* JptrRow = next_data + (iny + y) * next_step + inx * cn;
                            const short *IptrRow = IWinBuf + y * win_width * cn;
                            for (int x = 0; x < win_width * cn; ++x)
                            {
                                int diff = (((JptrRow[x] * jw00 + JptrRow[x + cn] * jw01 + JptrRow[x + stepJ] * jw10 + JptrRow[x + stepJ + cn] * jw11) >> (W_BITS1 - 5)) - IptrRow[x]);
                                errval += std::fabs((float)diff);
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