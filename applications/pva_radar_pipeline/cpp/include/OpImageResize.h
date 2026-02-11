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
 * @file OpImageResize.h
 *
 * @brief Defines types and functions to handle the image resize operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_IMAGERESIZE ImageResize
 * @{
 */

#ifndef PVA_SOLUTIONS_OPIMAGERESIZE_H
#define PVA_SOLUTIONS_OPIMAGERESIZE_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the ImageResize operator.
 * 
 * \b Limitations:
 *   - The scale factor for width: outputWidth / inputWidth >= 1/3.0f.
 *   - The scale factor for height: outputHeight / inputHeight >= 1/3.0f.
 *   - Input and output image sizes must be larger than 64x64.
 * 
 * \b Input:
 *      Image sizes and formats are specified in inImageRequirements.
 * 
 *      Image Format   | Allowed
 *      -------------- | -------------
 *      RGB            | Yes
 *      RGBA           | Yes
 *      NV12           | Yes
 *      U8             | Yes
 *      U16            | Yes
 *
 * \b Output:
 *      Image sizes and formats are specified in outImageRequirements.
 * 
 * \b Input/Output \b Dependency:
 *      Property    | Input == Output
 *     -------------| -------------
 *      Format      | Yes
 * 
 * \b Interpolation \b Type:
 *    
 *      Method Type        | Allowed
 *      ------------------ | -------------
 *      Nearest Neighbor   | Yes
 *      Bilinear           | Yes
 *      Catmull-Rom        | No
 * 
 * \b Parameters
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inImageRequirements Pointer to the NVCVImageRequirements structure which contains image shape,
 *                                 strides, and format information.
 *
 * @param [in] outImageRequirements Pointer to the output image NVCVImageRequirements structure.
 *
 * @param [in] resizeType Interpolation method to perform image resize. \ref PVAInterpolationType.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
*/
NVCVStatus pvaImageResizeCreate(NVCVOperatorHandle *handle, NVCVImageRequirements *inImageRequirements,
                                NVCVImageRequirements *outImageRequirements, PVAInterpolationType resizeType);

#ifdef __cplusplus
}

/**
 * Submits the ImageResize operator to a cuPVA stream.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input image handle.
 *
 * @param [out] out Output image handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaImageResizeSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVImageHandle in,
                                NVCVImageHandle out);

/**
 * Submits the ImageResize operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image handle.
 * @param [out] out Output image handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaImageResizeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageHandle in,
                                NVCVImageHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPIMAGERESIZE_H */