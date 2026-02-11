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
 * @file OpDoaAngleFFT.h
 *
 * @brief Defines types and functions to handle the DOA Angle FFT + Target Processing operation.
 * 
 * This operator combines Direction of Arrival (DoA) angle estimation with target processing
 * to produce complete target information (velocity, range, angles, and Cartesian coordinates)
 * in a single operator call.
 * 
 * ** API inputs and outputs:**
 * - Inputs: detectionCount, snapshots, detectionList, nci (optional), ddmDopplerOffsets  
 * - Outputs: targetCount, targetList (velocity, range, azimuth, elevation, X, Y, Z, power(optional))
 * 
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_DOA_ANGLE_FFT RadarDoaAngleFFT
 * @{
 */

#ifndef PVA_SOLUTIONS_OPDOAANGLEFFT_H
#define PVA_SOLUTIONS_OPDOAANGLEFFT_H

#include <PvaOperator.h>
#include <PvaOperatorTypes.h>
#include <RadarAdvancedOperatorTypes.h>
#include <RadarAdvancedSubmitDefinitions.h>
#include <cuda_runtime.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructs an instance of the DOA Angle FFT + Target Processing operator.
 *
 * This operator performs 2D Angle FFT for Direction of Arrival (DOA) estimation
 * followed by target processing to compute velocity, range, angles, Cartesian coordinates and optional power.
 * It processes virtual antenna snapshots and combines them with detection information to produce
 * complete target information.
 *
 * \b Limitations:
 *
 *      The maximum number of detections and targets is 8192.
 *      The local peak detection is not supported yet. Parameter is ignored.
 *      The product of the number of virtual azimuth elements and the number of virtual elevation elements must equal
 *      the number of Tx antennas multiplied by the number of Rx antennas:
 *          numVirtualAzimuthElements * numVirtualElevationElements = numTxAntennas * numRxAntennas
 *
 *      Operator currently supports the following parameter values:
 *       - numTxAntennas = 8
 *       - numRxAntennas = 8
 *       - numVirtualAzimuthElements = 16
 *       - numVirtualElevationElements = 4
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] snapshotsTensorRequirements Snapshots tensor requirements
 *          Data Layout:    [CHW]
 *              C: Maximum number of targets
 *              H: Number of Tx antennas.
 *              W: Number of Rx antennas.
 *
 *          Rank:           3
 *          Data Type:      NVCV_DATA_TYPE_2S32 (Complex SQ11.20)
 *
 * @param [in] detectionListTensorRequirements Range-Doppler Detection List tensor requirements
 *          Data Layout:    [HW]
 *              H: Number of detections (numDetections)
 *              W: 2 (rangeIdx, dopplerIdx)
 *
 *          Rank:           2
 *          Data Type:      NVCV_DATA_TYPE_S32
 *
 * @param [in] nciTensorRequirements NCI final output tensor requirements (optional, can be NULL)
 *          Data Layout:    [HW]
 *              H: Number of Range bins (numRangeBins)
 *              W: Number of Folded Doppler bins (numFoldedDopplerBins)
 *
 *          Rank:           2
 *          Data Type:      NVCV_DATA_TYPE_U32
 *
 *          Description: NCI power values used for quadratic interpolation to achieve sub-bin
 *                      accuracy in range and Doppler. If NULL, quadratic interpolation is disabled.
 *                      Required when params->targetProcessingParams.enableQuadraticInterpolation is true.
 *
 * @param [in] ddmDopplerOffsetsTensorRequirements DDM Doppler Offsets tensor requirements
 *          Data Layout:    [W]
 *              W: Number of Doppler folds (numDopplerFolds)
 *
 *          Rank:           1
 *          Data Type:      NVCV_DATA_TYPE_F32
 *
 * @param [in] targetListTensorRequirements Target list tensor requirements
 *          Data Layout:    [HW]
 *              H= 7 (without peak power) or 8 (with peak power)
 *              W: Number of targets (numTargets)
 *
 *          Rank:           2
 *          Data Type:      NVCV_DATA_TYPE_F32
 *
 *          Description: Complete target information including:
 *                      Row 0: Velocity (m/s)
 *                      Row 1: Range (meters)
 *                      Row 2: Azimuth angles (degrees)
 *                      Row 3: Elevation angles (degrees)
 *                      Row 4: X coordinate (meters)
 *                      Row 5: Y coordinate (meters)
 *                      Row 6: Z coordinate (meters)
 *                      Row 7: optional: Peak power (dB)
 *
 * @param [in] params Pointer to the DoaAngleFFT + Target Processing parameters. \ref PVADoaAngleFFTParams.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaDoaAngleFFTCreate(NVCVOperatorHandle *handle,
                                NVCVTensorRequirements const *const snapshotsTensorRequirements,
                                NVCVTensorRequirements const *const detectionListTensorRequirements,
                                NVCVTensorRequirements const *const nciTensorRequirements,
                                NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                NVCVTensorRequirements const *const targetListTensorRequirements,
                                PVADoaAngleFFTParams const *const params);

#ifdef __cplusplus
}

/**
 * Submits the DoA Angle FFT + Target Processing operator to a cuPVA stream.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 *
 * @param [in] stream Handle to a valid cuPVA stream.
 *
 * @param [in] detectionCount Input tensor: Detection Count
 *                            Data Layout: [W] where W=1
 *                            Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] snapshots Input tensor: Snapshots
 *                       Data Layout: [CHW] where C=Maximum number of targets, H=Number of Tx antennas, W=Number of Rx antennas
 *                       Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20)
 *
 * @param [in] detectionList Input tensor: Range-Doppler Detection List
 *                           Data Layout: [HW] where H=numDetections, W=2
 *                           Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] nci Input tensor: NCI (Non-Coherent Integration) power values (optional, can be NULL)
 *                 Data Layout: [HW] where H=numDopplerBins, W=numRangeBins
 *                 Data Type: NVCV_DATA_TYPE_U32
 *                 Used for quadratic interpolation of range/Doppler indices.
 *                 If NULL, quadratic interpolation is disabled.
 *
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets
 *                               Data Layout: [W] where W=numDopplerFolds
 *                               Data Type: NVCV_DATA_TYPE_F32
 *
 * @param [out] targetCount Output tensor: Target Count
 *                          Data Layout: [W] where W=1
 *                          Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [out] targetList Output tensor: Target List (complete target information)
 *                         Data Layout: [HW] where H= 7 (without peak power) or 8 (with peak power), W=Maximum number of targets
 *                         Data Type: NVCV_DATA_TYPE_F32
 *                         Row 0: Velocity (m/s)
 *                         Row 1: Range (meters)
 *                         Row 2: Azimuth angles (degrees)
 *                         Row 3: Elevation angles (degrees)
 *                         Row 4: X coordinate (meters)
 *                         Row 5: Y coordinate (meters)
 *                         Row 6: Z coordinate (meters)
 *                         Row 7: Optional:Peak power (dB)
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDoaAngleFFTSubmit(NVCVOperatorHandle handle, cupvaStream_t stream, NVCVTensorHandle const detectionCount,
                                NVCVTensorHandle const snapshots, NVCVTensorHandle const detectionList,
                                NVCVTensorHandle const nci, NVCVTensorHandle const ddmDopplerOffsets,
                                NVCVTensorHandle targetCount, NVCVTensorHandle targetList);

/**
 * Submits the DoA Angle FFT + Target Processing operator to a CUDA stream.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] detectionCount Input tensor: Detection Count
 * @param [in] snapshots Input tensor: Snapshots
 * @param [in] detectionList Input tensor: Detection List (Range-Doppler detection indices)
 * @param [in] nci Input tensor: NCI (optional, can be NULL for no quadratic interpolation)
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets
 * @param [out] targetCount Output tensor: Target Count
 * @param [out] targetList Output tensor: Target List (complete target information: velocity, range, azimuth, elevation, x, y, z, power (optional))
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDoaAngleFFTSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle const detectionCount,
                                NVCVTensorHandle const snapshots, NVCVTensorHandle const detectionList,
                                NVCVTensorHandle const nci, NVCVTensorHandle const ddmDopplerOffsets,
                                NVCVTensorHandle targetCount, NVCVTensorHandle targetList);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPDOAANGLEFFT_H */
