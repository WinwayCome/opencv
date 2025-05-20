#ifndef OPENCV_HAL_RVV_LKPYRAMID_HPP_INCLUDED
#define OPENCV_HAL_RVV_LKPYRAMID_HPP_INCLUDED

#include <riscv_vector.h>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
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
#ifndef CV_HAL_ERROR_OK
#define CV_HAL_ERROR_OK 0
#endif
#ifndef CV_HAL_ERROR_NOT_IMPLEMENTED
#define CV_HAL_ERROR_NOT_IMPLEMENTED 1
#endif
#ifndef CV_HAL_ERROR_UNKNOWN
#define CV_HAL_ERROR_UNKNOWN -1
#endif
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290e-7f
#endif
            typedef short deriv_type;







            
            inline int scharrDeriv(const uchar *src_data, size_t src_step,
                                   short *dst_data, size_t dst_step,
                                   int width, int height, int cn)
            {
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
            }

            inline int lkOpticalFlowLevel(
                const uint8_t *prev_data, std::size_t prev_data_step,
                const int16_t *prev_deriv_data, std::size_t prev_deriv_step,
                const uint8_t *next_data, std::size_t next_step,
                int width, int height, int cn,
                const float *prev_points, float *next_points, std::size_t point_count,
                uint8_t *status, float *err,
                int win_width, int win_height,
                int termination_count, double termination_epsilon,
                bool get_min_eigen_vals,
                float min_eigen_vals_threshold)
            {
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
            }

        }
    }

}

#endif // OPENCV_HAL_RVV_LKPYRAMID_HPP_INCLUDED