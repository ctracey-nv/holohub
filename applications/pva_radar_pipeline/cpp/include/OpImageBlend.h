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
 * @file OpImageBlend.h
 *
 * @brief Defines types and functions to handle the ImageBlend operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_IMAGEBLEND ImageBlend
 * @{
 */

#ifndef PVA_SOLUTIONS_OPIMAGEBLEND_H
#define PVA_SOLUTIONS_OPIMAGEBLEND_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructs an instance of the ImageBlend operator.
 *
 * Limitations:
 *
 *   Input/Output:
 *      Image format | Allowed
 *      ------------ | -------
 *      U8           | Yes
 *      YUYV         | Yes
 *      UYVY         | Yes
 *      VYUY         | Yes
 *      YUV8p        | Yes
 *      BGR8         | Yes
 *      RGB8         | Yes
 *      BGRA8        | Yes
 *      RGBA8        | Yes  
 *      BGR8p        | Yes
 *      RGB8p        | Yes
 *
 *   Input/Output Dependency:
 *
 *     Property      |  Input == Output
 *    -------------- | -------------
 *     Width         | Yes
 *     Height        | Yes
 *     Image format  | Yes
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] imageRequirements Pointer to the NVCVImageRequirements structure which contains image width, height and format information.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaImageBlendCreate(NVCVOperatorHandle *handle, NVCVImageRequirements const *const imageRequirements);

#ifdef __cplusplus
}

/**
 * Submits the ImageBlend operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in0 Input image 0 handle.
 *
 * @param [in] in1 Input image 1 handle.
 *
 * @param [in] alpha The blend factor Alpha in range [0.0, 1.0].
 *
 * @param [out] out Output image handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaImageBlendSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle const in0,
                               NVCVImageHandle const in1, const float alpha, NVCVImageHandle out);

/**
 * Submits the ImageBlend operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in0 Input image 0 handle.
 * @param [in] in1 Input image 1 handle.
 * @param [in] alpha The blend factor Alpha in range [0.0, 1.0].
 * @param [out] out Output image handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaImageBlendSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle const in0,
                               NVCVImageHandle const in1, const float alpha, NVCVImageHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPIMAGEBLEND_H */
