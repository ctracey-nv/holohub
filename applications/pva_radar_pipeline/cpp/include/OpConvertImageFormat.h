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
 * @file OpConvertImageFormat.h
 *
 * @brief Defines types and functions to handle the ConvertImageFormat operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_CONVERTIMAGEFORMAT ConvertImageFormat
 * @{
 */

#ifndef PVA_SOLUTIONS_OPCONVERTIMAGEFORMAT_H
#define PVA_SOLUTIONS_OPCONVERTIMAGEFORMAT_H

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
 * Constructs an instance of the ConvertImageFormat operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inImageRequirements Pointer to the NVCVImageRequirements structure which contains input image width, height and format information.
 *
 *   \b Input:
 *    This structure must be filled by using nvcvImageCalcRequirementsPva().
 *
 * @param [in] outImageRequirements Pointer to the NVCVImageRequirements structure which contains output image width, height and format information.
 *
 *   \b Output:
 *    This structure must be filled by using nvcvImageCalcRequirementsPva().
 *
 *   \b Input/Output \b Dependency:
 *
 *     Property      |  Input == Output
 *    -------------- | -------------
 *     Width         | Yes
 *     Height        | Yes
 *     Image format  | No
 *
 *    The supported combinations of input and output image formats for conversion are shown in the following table.
 *    Rows represent the input image format and columns represent the output image format.
 *
 *     in/out  | NV12   | YUYV   | UYVY   | VYUY   | YUV8p  | BGR8   | RGB8   | BGRA8  | BGR8p  | RGB8p
 *    ---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------
 *      NV12   | -      | -      | -      | -      | -      | yes    | yes    | -      | yes    | yes
 *      YUYV   | yes    | -      | -      | -      | -      | yes    | yes    | -      | yes    | yes
 *      UYVY   | yes    | -      | -      | -      | -      | yes    | yes    | -      | yes    | yes
 *      VYUY   | yes    | -      | -      | -      | -      | yes    | yes    | -      | yes    | yes
 *      YUV8p  | yes    | yes    | yes    | yes    | -      | -      | -      | -      | -      | -
 *      BGR8   | yes    | yes    | yes    | yes    | -      | -      | -      | -      | -      | -
 *      RGB8   | yes    | yes    | yes    | yes    | -      | -      | -      | yes    | -      | -
 *      BGRA8  | -      | -      | -      | -      | -      | -      | -      | -      | -      | -
 *      BGR8p  | yes    | yes    | yes    | yes    | -      | -      | -      | -      | -      | -
 *      RGB8p  | yes    | yes    | yes    | yes    | -      | -      | -      | -      | -      | -
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaConvertImageFormatCreate(NVCVOperatorHandle *handle,
                                       NVCVImageRequirements const *const inImageRequirements,
                                       NVCVImageRequirements const *const outImageRequirements);

#ifdef __cplusplus
}

/**
 * Submits the ConvertImageFormat operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input image handle.
 *
 * @param [out] out Output image handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaConvertImageFormatSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle const in,
                                       NVCVImageHandle out);

/**
 * Submits the ConvertImageFormat operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [out] out Output image handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaConvertImageFormatSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle const in,
                                       NVCVImageHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPCONVERTIMAGEFORMAT_H */
