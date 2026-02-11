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
 * @file OpDLInference.h
 *
 * @brief Defines types and functions to handle the DL Inference operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_DL_INFERENCE DLInference
 * @{
 */

#ifndef PVA_SOLUTIONS_OPDLINFERENCE_H
#define PVA_SOLUTIONS_OPDLINFERENCE_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum number of input tensors supported by the DL inference operator */
#define PVA_DL_INFERENCE_MAX_NUM_INPUTS 8
/** Maximum number of output tensors supported by the DL inference operator */
#define PVA_DL_INFERENCE_MAX_NUM_OUTPUTS 8

/**
 * Parameters for creating an DL inference operator.
 */
typedef struct PvaDLInferenceCreateParamRec
{
    NVCVTensorRequirements *inReqs[PVA_DL_INFERENCE_MAX_NUM_INPUTS];   /**< Array of input tensor requirements */
    NVCVTensorRequirements *outReqs[PVA_DL_INFERENCE_MAX_NUM_OUTPUTS]; /**< Array of output tensor requirements */
    int32_t numInReqs;                                                 /**< Number of input tensor requirements */
    int32_t numOutReqs;                                                /**< Number of output tensor requirements */
    char *networkName;                                                 /**< Name of the network */
} PvaDLInferenceCreateParams;

/**
 * Parameters for executing a DL inference operation.
 */
typedef struct PvaDLInferenceSubmitParamRec
{
    NVCVTensorHandle inTensors[PVA_DL_INFERENCE_MAX_NUM_INPUTS];   /**< Array of input tensor handles */
    NVCVTensorHandle outTensors[PVA_DL_INFERENCE_MAX_NUM_OUTPUTS]; /**< Array of output tensor handles */
    int32_t numInTensors;                                          /**< Number of input tensors */
    int32_t numOutTensors;                                         /**< Number of output tensors */
} PvaDLInferenceSubmitParams;

/**
 * Creates an instance of the DL inference operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                    Must not be NULL.
 * @param [in] params Pointer to the PvaDLInferenceCreateParams structure containing
 *                   tensor requirements for inputs and outputs.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle or params is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY   Not enough memory to create the operator.
 * @retval NVCV_SUCCESS               Operation executed successfully.
 */
NVCVStatus pvaDLInferenceCreate(NVCVOperatorHandle *handle, PvaDLInferenceCreateParams *params);

#ifdef __cplusplus
}

/**
 * Submits the DLInference operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator. Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 * @param [in] params Pointer to the PvaDLInferenceSubmitParams structure containing
 *                   input and output tensor handles.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle or params is null.
 * @retval NVCV_ERROR_INTERNAL        Internal error in the operator.
 * @retval NVCV_SUCCESS              Operation executed successfully.
 */
NVCVStatus pvaDLInferenceSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, PvaDLInferenceSubmitParams *params);

/**
 * Submits the DLInference operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] params Pointer to the PvaDLInferenceSubmitParams structure containing input and output tensor handles.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle or params is null.
 * @retval NVCV_ERROR_INTERNAL        Internal error in the operator.
 * @retval NVCV_SUCCESS              Operation executed successfully.
 */
NVCVStatus pvaDLInferenceSubmit(NVCVOperatorHandle handle, cudaStream_t stream, PvaDLInferenceSubmitParams *params);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPDLINFERENCE_H */