/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file OpImageHistogram.h
 *
 * @brief Defines types and functions to handle the Image Histogram operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_IMAGEHISTOGRAM ImageHistogram
 * @{
 */

#ifndef PVA_SOLUTIONS_OPIMAGEHISTOGRAM_H
#define PVA_SOLUTIONS_OPIMAGEHISTOGRAM_H

#include <PvaOperator.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Parameters for image histogram operator.
typedef struct
{
    /// starting intensity value to be counted
    float start;
    /// ending intensity value to be counted
    float end;
    /// number of bins
    int32_t numBins;
} PvaImageHistogramParams;

/**
 * Constructs and an instance of the ImageHistogram operator.
 *
 * \b Limitations:
 *
 *      The ranges of the image histogram parameters are listed as follows:
 *      - 0 <= start < end <= 256 for input tensor data type NVCV_DATA_TYPE_U8
 *      - 0 <= start < end <= 65536 for input tensor data type NVCV_DATA_TYPE_U16
 *      - 0 < numBins <= 256 for input tensor data type NVCV_DATA_TYPE_U8
 *      - 0 < numBins <= 16384 for input tensor data type NVCV_DATA_TYPE_U16
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inTensorRequirements Pointer to the NVCVTensorRequirements structure which contains input Tensor layout rank, shape and data type information.
 *
 *  \b Input:
 *      Data Layout:    [HWC, NHWC]
 *      Batches:        [1]
 *      Channels:       [1]
 *
 *      Rank:           3 for [HWC] or 4 for [NHWC].
 *
 *    Data Type      | Allowed
 *    -------------- | -------------
 *    8bit  Unsigned | Yes
 *    8bit  Signed   | No
 *    16bit Unsigned | Yes
 *    16bit Signed   | No
 *    32bit Unsigned | No
 *    32bit Signed   | No
 *    32bit Float    | No
 *    64bit Float    | No
 *
 *
 * @param [in] outTensorRequirements Pointer to the NVCVTensorRequirements structure which contains output Tensor layout rank, shape and data type information.
 *
 *   \b Output:
 *      Data Layout:    [HWC, NHWC]
 *      Batches:        [1]
 *      Height:         [1]
 *      Width:          >= numBins
 *      Channels:       [1]
 *
 *      Rank:           3 for [HWC] or 4 for [NHWC].
 *
 *
 *    Data Type      | Allowed
 *    -------------- | -------------
 *    8bit  Unsigned | No
 *    8bit  Signed   | No
 *    16bit Unsigned | No
 *    16bit Signed   | No
 *    32bit Unsigned | Yes
 *    32bit Signed   | Yes
 *    32bit Float    | No
 *    64bit Float    | No
 *
 *
 *   \b Input/Output \b Dependency:
 *
 *     Property      |  Input == Output
 *    -------------- | -------------
 *     Data Layout   | N/A
 *     Data Type     | N/A
 *     Number        | N/A
 *     Channels      | N/A
 *     Width         | N/A
 *     Height        | N/A
 *
 * @param [in] imageHistogramParams Pointer to the ImageHistogram parameters.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaImageHistogramCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                                   NVCVTensorRequirements const *const outTensorRequirements,
                                   PvaImageHistogramParams const *const imageHistogramParams);

#ifdef __cplusplus
}

/**
 * Submits the ImageHistogram operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input tensor handle.
 *
 * @param [out] out Output histogram handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaImageHistogramSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const in,
                                   NVCVTensorHandle out);

/**
 * Submits the ImageHistogram operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor handle.
 * @param [out] out Output histogram handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaImageHistogramSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const in,
                                   NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPIMAGEHISTOGRAM_H */