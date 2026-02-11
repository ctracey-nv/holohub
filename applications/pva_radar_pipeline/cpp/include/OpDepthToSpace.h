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
 * @file OpDepthToSpace.h
 *
 * @brief Defines types and functions to handle the Depth-to-Space operation.
 * @defgroup PVA_OPERATOR_ALGORITHM_DEPTHTOSPACE DepthToSpace
 * @{
 */

#ifndef PVA_SOLUTIONS_OPDEPTHTOSPACE_H
#define PVA_SOLUTIONS_OPDEPTHTOSPACE_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Constructs an instance of the DepthToSpace operator.
 *
 * DepthToSpace squeezes the depth dimension and moves the squeezed elements into both the width and height dimension.
 * Depth here is the synonym of the channel dimension and these two are used interchangeably. After performing the
 * DepthToSpace operation, the total number elements keep the same. The output width and height dimension are both
 * enlarged by a ratio, which is called block size parameter of the DepthToSpace operation. The reverse operation is
 * called SpaceToDepth. More info can also be found in https://onnx.ai/onnx/operators/onnx__DepthToSpace.html
 *
 *
 * \b Input:
 *      Data Layout:    [CHW]
 *      Note that this operator only supports tightly packed layout for both input and output, which means that strides[1] must
 *      equal shape[2] in the tensorRequirements struct.
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *
 * \b Output:
 *      Data Layout:    [CHW]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *
 * \b Input/Output \b Dependency:
 *      Property      |  Input == Output
 *     -------------- | -------------
 *      Data Layout   | Yes
 *      Data Type     | Yes
 *      Number        | Yes
 *      Channels      | No, Output Channels = (Input Channels) / (blkSz^2). Note that input channels should be multiple of (blkSz^2).
 *      Width         | No, Output Width = (Input Width) * (blkSz)
 *      Height        | No, Output Height = (Output Height) * (blkSz)
 *
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] tensorRequirements Pointer to the NVCVTensorRequirements structure which contains Tensor rank, shape, layout and data type information.
 *
 * @param [in] blkSz Block size, only supports 2 or 3 or 4. The implementation will use this parameter as well as the tensor size within tensorRequirements to do the tiling and mapping table
 *            generation, which is not light weight. Therefore, the blkSz parameter is specified during the create phase instead of the submission phase.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       The possible cases might be the following:
 *                                            1) Either tensorRequirements is NULL or handle and tensorRequirements point to an invalid address.
 *                                            2) Block size is not either 2 or 3 or 4.
 *                                            3) The channel number is not multiple of block size.
 *                                            4) The tensor is not in NCHW layout, or data type is not either 8bit signed or unsigned.
 *                                            5) Auto tiling algorithm fails to find a reasonable tile size.
 * @retval NVCV_ERROR_OUT_OF_MEMORY          The possible cases might be the following:
 *                                            1) Failed to allocate memory for the operator.
 *                                            2) Failed to allocate memory for holding the mapping table, which the host initializes then send it for the device's use.
 * @retval NVCV_SUCCESS                      Operation executed successfully.
 */
NVCVStatus pvaDepthToSpaceCreate(NVCVOperatorHandle *handle, NVCVTensorRequirements const *const tensorRequirements,
                                 int32_t blkSz);

#ifdef __cplusplus
}

/**
 * Submits the DepthToSpace operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] in Input tensor handle.
 *
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       The possible cases might be the following:
 *                                            1) The handle or stream or in or out is either NULL or points to an invalid address.
 *                                            2) The input or output tensor does not meet the requirements used to create the operator handle.
 * @retval NVCV_SUCCESS                      Operation executed successfully.
 */
NVCVStatus pvaDepthToSpaceSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle in,
                                 NVCVTensorHandle out);

/**
 * Submits the DepthToSpace operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor handle.
 * @param [out] out Output tensor handle.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT       The possible cases might be invalid parameters.
 * @retval NVCV_SUCCESS                      Operation executed successfully.
 */
NVCVStatus pvaDepthToSpaceSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                 NVCVTensorHandle out);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPDEPTHTOSPACE_H */