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
 * @file OpDoaBartlettBeamforming.h
 *
 * @brief Defines types and functions to handle the Bartlett Beamforming DOA + Target Processing operation.
 *
 * This operator combines Direction of Arrival (DoA) angle estimation using Bartlett beamforming
 * with target processing to produce complete target information (velocity, range, angles,
 * and Cartesian coordinates) in a single operator call.
 *
 * ** API inputs and outputs:**
 * - Inputs: detectionCount, snapshots, steeringVectors, azimuthBins, elevationBins, detectionList, nci (optional), ddmDopplerOffsets
 * - Outputs: targetCount, targetList (velocity, range, azimuth, elevation, X, Y, Z, power(optional))
 *
 * @defgroup PVA_OPERATOR_ALGORITHM_RADAR_DOA_BARTLETT_BEAMFORMING RadarDoaBartlettBeamforming
 * @{
 */

#ifndef PVA_SOLUTIONS_OPDOABARTLETTBEAMFORMING_H
#define PVA_SOLUTIONS_OPDOABARTLETTBEAMFORMING_H

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
 * Constructs an instance of the Bartlett Beamforming DOA + Target Processing operator.
 *
 * This operator performs Bartlett beamforming (conventional beamforming) for
 * Direction of Arrival (DOA) estimation using pre-computed steering vectors,
 * followed by target processing to compute velocity, range, angles, Cartesian
 * coordinates and optional power.
 *
 * \b Limitations:
 *
 *      The maximum number of detections and targets is 8192.
 *      The number of virtual antennas must match numTxAntennas × numRxAntennas from the antenna config.
 *      Maximum virtual channels: numTxAntennas × numRxAntennas ≤ 64.
 *      numTxAntennas × numRxAntennas must be divisible by 8.
 *      numAzimuthBins × numElevationBins ≤ 2048.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] snapshotsTensorRequirements Snapshots tensor requirements
 *          Data Layout:    [CHW]
 *              C: Number of detections (numDetections)
 *              H: Number of TX antennas (numTxAntennas)
 *              W: Number of RX antennas (numRxAntennas)
 *
 *          Rank:           3
 *          Data Type:      NVCV_DATA_TYPE_2S32 (Complex SQ11.20 fixed-point)
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
 *              W: Number of Doppler folds (numDopplerFolds), typically 1
 *              Note: Currently only the first element (TX1 offset) is used
 *
 *          Rank:           1
 *          Data Type:      NVCV_DATA_TYPE_F32
 *
 * @param [in] targetListTensorRequirements Target list tensor requirements
 *          Data Layout:    [HW]
 *              H= 7 (without peak power) or 8 (with peak power)
 *              W= Maximum number of targets (capacity)
 *
 *          Rank:           2
 *          Data Type:      NVCV_DATA_TYPE_F32
 *          Note: Actual valid target count is returned in targetCount tensor
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
 * @param [in] params Pointer to the Bartlett Beamforming + Target Processing parameters. \ref PVADoaBartlettBeamformingParams.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Handle is null or some parameter is outside valid range.
 * @retval NVCV_ERROR_OUT_OF_MEMORY        Not enough memory to create the operator.
 * @retval NVCV_ERROR_INVALID_IMAGE_FORMAT Image format is invalid.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaDoaBartlettBeamformingCreate(NVCVOperatorHandle *handle,
                                           NVCVTensorRequirements const *const snapshotsTensorRequirements,
                                           NVCVTensorRequirements const *const detectionListTensorRequirements,
                                           NVCVTensorRequirements const *const nciTensorRequirements,
                                           NVCVTensorRequirements const *const ddmDopplerOffsetsTensorRequirements,
                                           NVCVTensorRequirements const *const targetListTensorRequirements,
                                           PVADoaBartlettBeamformingParams const *const params);

/**
 * Computes steering vectors for Bartlett beamforming DOA estimation.
 *
 * This utility function computes steering vectors based on virtual antenna array locations
 * and angle bins, and fills the provided tensor. The steering vectors can be reused
 * across multiple operator submissions as long as the antenna configuration and
 * angle bins remain unchanged.
 *
 * Steering vectors are computed as: e^(-j·2π·phase) where
 * phase = vcPosition.x · cos(elevation) · sin(azimuth) + vcPosition.y · sin(elevation)
 * Virtual antenna array locations are in wavelengths.
 *
 * @param [in] virtualArrayLocations Virtual antenna array locations tensor (in wavelengths)
 *                         Shape: [numTxAntennas, numRxAntennas]
 *                         Data Type: NVCV_DATA_TYPE_2F32 (2-channel float32, representing [x, y] coordinates)
 *                         Memory Layout: For each TX-RX antenna pair, the 2 channels store:
 *                                        Channel 0: x position
 *                                        Channel 1: y position
 *                         + Must not be NULL.
 *                         + numTxAntennas × numRxAntennas ≤ 64 (maximum 64 virtual channels).
 *                         + numTxAntennas × numRxAntennas must be divisible by 8.
 *
 * @param [in] azimuthBins Azimuth angle bins tensor (degrees)
 *                         Data Layout: [W] where W = numAzimuthBins
 *                         Data Type: NVCV_DATA_TYPE_F32
 *                         + Must not be NULL.
 *
 * @param [in] elevationBins Elevation angle bins tensor (degrees)
 *                           Data Layout: [W] where W = numElevationBins
 *                           Data Type: NVCV_DATA_TYPE_F32
 *                           + Must not be NULL.
 *
 * @param [out] steeringVectors Output tensor: Steering vectors
 *                              Data Layout: [numAzimuthBins][numElevationBins][numTxAntennas][numRxAntennas]
 *                              Rank: 4
 *                              Data Type: NVCV_DATA_TYPE_2S16 (Complex Q15 fixed-point)
 *                              + Must not be NULL.
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT     Some parameter is NULL or outside valid range, or tensor doesn't match requirements.
 * @retval NVCV_ERROR_INTERNAL             Internal error during computation.
 * @retval NVCV_SUCCESS                    Operation executed successfully.
 */
NVCVStatus pvaDoaBartlettBeamformingComputeSteeringVectors(NVCVTensorHandle virtualArrayLocations,
                                                           NVCVTensorHandle azimuthBins, NVCVTensorHandle elevationBins,
                                                           NVCVTensorHandle steeringVectors);

#ifdef __cplusplus
}

/**
 * Submits the Bartlett Beamforming + Target Processing operator to a cuPVA stream.
 *
 * Performs beamforming on the provided snapshots using pre-computed steering vectors
 * to estimate azimuth and elevation angles for each detection, then performs target
 * processing to compute complete target information.
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
 *                       Data Layout: [CHW] where C=numDetections, H=numTxAntennas, W=numRxAntennas
 *                       Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20 fixed-point)
 *                       Description: Antenna snapshots extracted from range-Doppler map for each detection.
 *
 * @param [in] steeringVectors Input tensor: Pre-computed steering vectors
 *                             Data Layout: [numAzimuthBins][numElevationBins][numTxAntennas][numRxAntennas]
 *                             Rank: 4
 *                             Data Type: NVCV_DATA_TYPE_2S16 (Complex Q15 fixed-point)
 *                             + Must not be NULL.
 *                             + Must be computed using pvaDoaBartlettBeamformingComputeSteeringVectors.
 *                             + Must match the antenna configuration and angle bins.
 *
 * @param [in] azimuthBins Input tensor: Azimuth angle bins for peak index to angle mapping
 *                         Data Layout: [W] where W = numAzimuthBins
 *                         Data Type: NVCV_DATA_TYPE_F32
 *                         + Must not be NULL.
 *                         + Must match the azimuth bins used to compute steering vectors.
 *
 * @param [in] elevationBins Input tensor: Elevation angle bins for peak index to angle mapping
 *                           Data Layout: [W] where W = numElevationBins
 *                           Data Type: NVCV_DATA_TYPE_F32
 *                           + Must not be NULL.
 *                           + Must match the elevation bins used to compute steering vectors.
 *
 * @param [in] detectionList Input tensor: Range-Doppler Detection List
 *                           Data Layout: [HW] where H=numDetections, W=2
 *                           Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [in] nci Input tensor: NCI (Non-Coherent Integration) final output power values (optional, can be NULL)
 *                 Data Layout: [HW] where H=numDopplerBins, W=numRangeBins
 *                 Data Type: NVCV_DATA_TYPE_U32
 *                 Used for quadratic interpolation based refinement of range/Doppler detection indices.
 *                 If NULL, quadratic interpolation is disabled.
 *
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets
 *                               Data Layout: [W] where W=numDopplerFolds (typically 1)
 *                               Data Type: NVCV_DATA_TYPE_F32
 *                               Note: Currently only the first element (TX1 offset) is used
 *
 * @param [out] targetCount Output tensor: Target Count
 *                          Data Layout: [W] where W=1
 *                          Data Type: NVCV_DATA_TYPE_S32
 *
 * @param [out] targetList Output tensor: Target List (complete target information)
 *                         Data Layout: [HW] where H= 7 (without peak power) or 8 (with peak power), W=Maximum number of targets (capacity)
 *                         Data Type: NVCV_DATA_TYPE_F32
 *                         Row 0: Velocity (m/s)
 *                         Row 1: Range (meters)
 *                         Row 2: Azimuth angles (degrees)
 *                         Row 3: Elevation angles (degrees)
 *                         Row 4: X coordinate (meters)
 *                         Row 5: Y coordinate (meters)
 *                         Row 6: Z coordinate (meters)
 *                         Row 7: Optional: Peak power (dB)
 *                         Note: Actual valid target count is returned in targetCount tensor
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null or steering vectors don't match operator configuration.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDoaBartlettBeamformingSubmit(NVCVOperatorHandle handle, cupvaStream_t stream,
                                           NVCVTensorHandle const detectionCount, NVCVTensorHandle const snapshots,
                                           NVCVTensorHandle const steeringVectors, NVCVTensorHandle const azimuthBins,
                                           NVCVTensorHandle const elevationBins, NVCVTensorHandle const detectionList,
                                           NVCVTensorHandle const nci, NVCVTensorHandle const ddmDopplerOffsets,
                                           NVCVTensorHandle targetCount, NVCVTensorHandle targetList);

/**
 * Submits the Bartlett Beamforming + Target Processing operator to a CUDA stream.
 *
 * Performs beamforming on the provided snapshots using pre-computed steering vectors
 * to estimate azimuth and elevation angles for each detection, then performs target
 * processing to compute complete target information.
 *
 * @copydoc PVA_CUDA_STREAM_REQUIREMENTS
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] detectionCount Input tensor: Detection Count
 *                            Data Layout: [W] where W=1
 *                            Data Type: NVCV_DATA_TYPE_S32
 * @param [in] snapshots Input tensor: Snapshots
 *                       Data Layout: [CHW] where C=numDetections, H=numTxAntennas, W=numRxAntennas
 *                       Data Type: NVCV_DATA_TYPE_2S32 (Complex SQ11.20 fixed-point)
 *                       Description: Antenna snapshots extracted from range-Doppler map for each detection.
 * @param [in] steeringVectors Input tensor: Pre-computed steering vectors
 *                             Data Layout: [numAzimuthBins][numElevationBins][numTxAntennas][numRxAntennas]
 *                             Rank: 4
 *                             Data Type: NVCV_DATA_TYPE_2S16 (Complex Q15 fixed-point)
 *                             + Must not be NULL.
 *                             + Must be computed using pvaDoaBartlettBeamformingComputeSteeringVectors.
 *                             + Must match the antenna configuration and angle bins.
 * @param [in] azimuthBins Input tensor: Azimuth angle bins for peak index to angle mapping
 *                         Data Layout: [W] where W = numAzimuthBins
 *                         Data Type: NVCV_DATA_TYPE_F32
 *                         + Must not be NULL.
 *                         + Must match the azimuth bins used to compute steering vectors.
 * @param [in] elevationBins Input tensor: Elevation angle bins for peak index to angle mapping
 *                           Data Layout: [W] where W = numElevationBins
 *                           Data Type: NVCV_DATA_TYPE_F32
 *                           + Must not be NULL.
 *                           + Must match the elevation bins used to compute steering vectors.
 * @param [in] detectionList Input tensor: Range-Doppler Detection List
 *                           Data Layout: [HW] where H=numDetections, W=2
 *                           Data Type: NVCV_DATA_TYPE_S32
 * @param [in] nci Input tensor: NCI (optional, can be NULL for no quadratic interpolation)
 * @param [in] ddmDopplerOffsets Input tensor: DDM Doppler Offsets
 *                               Data Layout: [W] where W=numDopplerFolds (typically 1)
 *                               Data Type: NVCV_DATA_TYPE_F32
 *                               Note: Currently only the first element (TX1 offset) is used
 * @param [out] targetCount Output tensor: Target Count
 *                          Data Layout: [W] where W=1
 *                          Data Type: NVCV_DATA_TYPE_S32
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
 *                         Row 7: Optional: Peak power (dB)
 *
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null or steering vectors don't match operator configuration.
 * @retval NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval NVCV_SUCCESS                Operation executed successfully.
 */
NVCVStatus pvaDoaBartlettBeamformingSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                           NVCVTensorHandle const detectionCount, NVCVTensorHandle const snapshots,
                                           NVCVTensorHandle const steeringVectors, NVCVTensorHandle const azimuthBins,
                                           NVCVTensorHandle const elevationBins, NVCVTensorHandle const detectionList,
                                           NVCVTensorHandle const nci, NVCVTensorHandle const ddmDopplerOffsets,
                                           NVCVTensorHandle targetCount, NVCVTensorHandle targetList);

#endif // __cplusplus

/** @} */
#endif /* PVA_SOLUTIONS_OPDOABARTLETTBEAMFORMING_H */
