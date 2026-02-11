/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpRangeFFT.h
 *
 * @brief Defines types and functions to handle the RangeFFT operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_RANGEFFT RadarRangeFFT
 * @{
 */

#ifndef PVA_SOLUTIONS_OPRANGEFFT_H
#define PVA_SOLUTIONS_OPRANGEFFT_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructs an instance of the RangeFFT operator.
 *
 *   \b Limitations:
 *
 *      The supported number of samples must be in the range [4, 1024] and must be even and factorizable using only factors of 2, 3, 4, and 5.
 *      The number of range bins can't be greater than (the number of samples / 2 + 1).
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inTensorRequirements Pointer to the NVCVTensorRequirements structure which contains input Tensor layout rank, shape and data type information.
 *
 *  \b Input:
 *      Data Layout:    [HCW, NHCW]
 *          W: Number of samples.
 *          C: Number of RX channels.
 *          H: Number of chirps.
 *
 *      Batches:        [1]
 *
 *      Rank:           3 for [HCW] or 4 for [NHCW].
 *
 *    Data Type        | Allowed
 *    ---------------- | -------------
 *    8bit  Unsigned   | No
 *    8bit  Signed     | No
 *    16bit Unsigned   | No
 *    16bit Signed     | No
 *    32bit Unsigned   | No
 *    32bit Signed     | Yes
 *    32bit Float      | No
 *    64bit Float      | No
 *    2x16bit Unsigned | No
 *    2x16bit Signed   | No
 *    2x32bit Unsigned | No
 *    2x32bit Signed   | No
 *
 * @param [in] winTensorRequirements Pointer to the NVCVTensorRequirements structure which contains window Tensor layout rank, shape and data type information.
 *
 *  \b Input:
 *      Data Layout:    [W, NW]
 *          W: Number of coefficients. Must be the same as the number of samples in the input tensor.
 *
 *      Batches:        [1]
 *
 *      Rank:           1 for [W] or 2 for [NW].
 *
 *    Data Type        | Allowed
 *    ---------------- | -------------
 *    8bit  Unsigned   | No
 *    8bit  Signed     | No
 *    16bit Unsigned   | No
 *    16bit Signed     | No
 *    32bit Unsigned   | No
 *    32bit Signed     | Yes
 *    32bit Float      | No
 *    64bit Float      | No
 *    2x16bit Unsigned | No
 *    2x16bit Signed   | No
 *    2x32bit Unsigned | No
 *    2x32bit Signed   | No
 *
 * @param [in] outTensorRequirements Pointer to the NVCVTensorRequirements structure which contains output Tensor layout rank, shape and data type information.
 *
 *   \b Output:
 *      Data Layout:    [HCW, NHCW]
 *          W: Number of range bins.
 *          C: Number of RX channels.
 *          H: Number of chirps.
 *
 *      Batches:        [1]
 *
 *      Rank:           3 for [HCW] or 4 for [NHCW].
 *
 *    Data Type        | Allowed
 *    ---------------- | -------------
 *    8bit  Unsigned   | No
 *    8bit  Signed     | No
 *    16bit Unsigned   | No
 *    16bit Signed     | No
 *    32bit Unsigned   | No
 *    32bit Signed     | No
 *    32bit Float      | No
 *    64bit Float      | No
 *    2x16bit Unsigned | No
 *    2x16bit Signed   | No
 *    2x32bit Unsigned | No
 *    2x32bit Signed   | Yes
 *
 *   \b Input/Output \b Dependency:
 *
 *     Property      |  Input == Output
 *    -------------- | -------------
 *     Data Layout   | Yes
 *     Data Type     | No
 *     Number        | Yes
 *     Channels      | Yes
 *     Width         | N/A
 *     Height        | Yes
 *
 * @param [in] params Pointer to the RangeFFT parameters. \ref PVARangeFFTParams.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaRangeFFTCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const inTensorRequirements,
                             NVCVTensorRequirements const *const winTensorRequirements,
                             NVCVTensorRequirements const *const outTensorRequirements,
                             PVARangeFFTParams const *const params);

#ifdef __cplusplus
}

/**
 * Submits the RangeFFT operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input tensor handle.
 *
 * @param [in] win Window tensor handle.
 *
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaRangeFFTSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const in,
                             NVCVTensorHandle const win, NVCVTensorHandle out);

/**
 * Submits the RangeFFT operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor handle.
 * @param [in] win Window tensor handle.
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaRangeFFTSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const in,
                             NVCVTensorHandle const win, NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPRANGEFFT_H */