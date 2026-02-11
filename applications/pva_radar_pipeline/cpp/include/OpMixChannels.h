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
 * @file OpMixChannels.h
 *
 * @brief Defines types and functions to handle the mix channels operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_MIX_CHANNELS MixChannels
 * @{
 */

#ifndef PVA_SOLUTIONS_OPMIXCHANNELS_H
#define PVA_SOLUTIONS_OPMIXCHANNELS_H
#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the MixChannels operator.
 *
 * Limitations: height >= 64, width >= 64
 *
 * \b Input: for each case, all input tensors must have the same height and width.
 * 
 *    PVAMixChannelsCode  | inTensorCount | dtype              | rank | layout | shape
 *    ------------------- | ------------- | ------------------ | ---- | ------ | -----------
 *    PVA_SPLIT_RGBA8_TO_U8   | 1             | NVCV_DATA_TYPE_U8  | 3    | HWC    | [h, w, 4]
 *    PVA_MERGE_U8_TO_RGBA8   | 4             | NVCV_DATA_TYPE_U8  | 3    | HWC    | [h, w, 1]
 * 
 * \b Output: for each case, all output tensors must have the same height and width.
 * 
 *    PVAMixChannelsCode  | outTensorCount | dtype              | rank | layout | shape
 *    ------------------- | -------------- | ------------------ | ---- | ------ | -----------
 *    PVA_SPLIT_RGBA8_TO_U8   | 4              | NVCV_DATA_TYPE_U8  | 3    | HWC    | [h, w, 1]
 *    PVA_MERGE_U8_TO_RGBA8   | 1              | NVCV_DATA_TYPE_U8  | 3    | HWC    | [h, w, 4]
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] inTensorReqPtrs Pointer to array of input NVCVTensorRequirements pointers. 
 * 
 * @param [in] outTensorReqPtrs Pointer to array of output NVCVTensorRequirements pointers.
 * 
 * @param [in] inTensorCount Number of input tensors.
 * 
 * @param [in] outTensorCount Number of output tensors.
 * 
 * @param [in] code Mix channels case code.
 * 
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMixChannelsCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements **inTensorReqPtrs,
                                NVCVTensorRequirements **outTensorReqPtrs, int inTensorCount, int outTensorCount,
                                PVAMixChannelsCode code);

#ifdef __cplusplus
}

/**
 * Submits the MixChannels operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] inHandles Pointer to array of input NVCVTensorHandle.
 * 
 * @param [out] outHandles Pointer to array of output NVCVTensorHandle.
 * 
 * @param [in] inTensorCount Number of input tensors.
 * 
 * @param [in] outTensorCount Number of output tensors.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMixChannelsSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle *inHandles,
                                NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount);

/**
 * Submits the MixChannels operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] inHandles Pointer to array of input NVCVTensorHandle.
 * @param [out] outHandles Pointer to array of output NVCVTensorHandle.
 * @param [in] inTensorCount Number of input tensors.
 * @param [in] outTensorCount Number of output tensors.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaMixChannelsSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle *inHandles,
                                NVCVTensorHandle *outHandles, int inTensorCount, int outTensorCount);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPMIXCHANNELS_H */