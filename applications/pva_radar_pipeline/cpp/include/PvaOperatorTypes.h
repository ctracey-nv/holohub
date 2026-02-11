/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file PvaOperatorTypes.h
 *
 * @brief Defines types to be used by operators.
 * @defgroup PVA_OPERATOR_CORE_PVA_OPERATOR_TYPES PvaOperatorTypes
 * @{
 */

#ifndef PVA_OPERATOR_TYPES_H
#define PVA_OPERATOR_TYPES_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Morphology Operation Codes
 */
typedef enum
{
    /// Erode operation
    PVA_ERODE = 0,
    /// Dilate operation
    PVA_DILATE = 1,
} PVAMorphologyType;

/**
 * @brief MixChannels Operation Codes
 *
 * Mix channels is a copy operation from a set of input channels to a set of output channels.
 *
 * The set of inputs and outputs may be given by any number of input and output images,
 * where each image may have one or more channels.
 *
 * The channel mapping from input to output is given by two arrays, one with the indices of
 * the input channels, and the another with the corresponding channel index in the output.
 *
 * The channel indices are enumerated starting from zero and increasing monotonically across
 * all channels of all images provided in the array. For example, given an array of 3 RGB
 * images, index 5 corresponds to the B channel of the second image.
 */
typedef enum
{
    /// split 1 rgba8 image into 4 u8 images, [0, 1, 2, 3] -> [0, 1, 2, 3]
    PVA_SPLIT_RGBA8_TO_U8 = 0,

    /// merge 4 u8 images into 1 rgba8 image, [0, 1, 2, 3] -> [0, 1, 2, 3]
    PVA_MERGE_U8_TO_RGBA8 = 1
} PVAMixChannelsCode;

/**
 * @enum PVADistanceType
 * @brief Defines the types of distance metrics for the DistanceTransform operator.
 *
 * Specifies the different distance calculation methods available.
 */
typedef enum
{
    /// Manhattan distance (L1 norm).
    PVA_DIST_L1 = 0,
    /// Euclidean distance (L2 norm).
    PVA_DIST_L2,
    /// Hamming distance.
    PVA_DIST_HAMMING
} PVADistanceType;

/**
 * @brief Interpolation method types for image operations.
 */
typedef enum
{
    /// Nearest neighbor
    PVA_INTERPOLATION_NN = 0,
    /// Bilinear interpolation
    PVA_INTERPOLATION_LINEAR
} PVAInterpolationType;

/**
 * @brief Warp perspective operator's transformation types
 */
typedef enum
{
    /// Perspective transformation
    PVA_WARP_TRANSFORMATION_PERSPECTIVE = 0,
    /// Affine transformation
    PVA_WARP_TRANSFORMATION_AFFINE = 1,
} PVAWarpTransformationType;

/**
 * @brief Brute Force Matcher operator's maximum matches per descriptor
 */
#define PVA_MAX_BF_MATCHES_PER_DESCRIPTOR (3)

/**
 * @brief Brute Force Matcher operator's maximum number of descriptors for either queries or references
 */
#define PVA_MAX_BF_DESCRIPTOR_COUNT (1024)

/**
 * @brief Brute Force Matcher operator's length of Brief Descriptor Array
 */
#define PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH (32)

/**
 * @brief Brute Force Matcher operator's structure containing the BRIEF Descriptor
 */
typedef struct
{
    /** Description vector of a BRIEF descriptor */
    uint8_t data[PVA_BRIEF_DESCRIPTOR_ARRAY_LENGTH];
} PVABriefDescriptor;

/**
 * @brief Structure containing match information for a single query descriptor
 */
typedef struct PVABFMatchesTypeRec
{
    /// Reference descriptor indices (-1 indicates no match).
    int32_t refIndex[PVA_MAX_BF_MATCHES_PER_DESCRIPTOR];

    /// Hamming distances to matched reference descriptors.
    float distance[PVA_MAX_BF_MATCHES_PER_DESCRIPTOR];
} PVABFMatchesType;

/**
 * @brief Image Statistics operation flags
 */
typedef enum
{
    /// Calculate pixel count
    PVA_IMAGE_STAT_FLAG_PIXEL_COUNT = (1u << 0),
    /// Calculate per-channel sum
    PVA_IMAGE_STAT_FLAG_SUM = (1u << 1),
    /// Calculate per-channel mean, triggers PVA_IMAGE_STAT_FLAG_SUM and PVA_IMAGE_STAT_FLAG_PIXEL_COUNT
    PVA_IMAGE_STAT_FLAG_MEAN = ((1u << 2) | PVA_IMAGE_STAT_FLAG_SUM | PVA_IMAGE_STAT_FLAG_PIXEL_COUNT),
    /// Calculate per-channel variance, triggers PVA_IMAGE_STAT_FLAG_MEAN
    PVA_IMAGE_STAT_FLAG_VARIANCE = ((1u << 3) | PVA_IMAGE_STAT_FLAG_MEAN),
    /// Calculate full covariance matrix, triggers PVA_IMAGE_STAT_FLAG_VARIANCE
    PVA_IMAGE_STAT_FLAG_COVARIANCE = ((1u << 4) | PVA_IMAGE_STAT_FLAG_VARIANCE)
} PVAImageStatFlag;

/**
 * @brief Image Statistics output structure
 */
typedef struct
{
    /** @brief Number of pixels processed. */
    int pixelCount;
    /** @brief Per-channel sum of pixel values. Index corresponds to channel number. */
    float sum[4];
    /** @brief Per-channel mean of pixel values. Index corresponds to channel number. */
    float mean[4];
    /** @brief Covariance matrix.
     *         covariance[c][c] is the variance of channel c.
     *         covariance[c][c'] (for c != c') is the covariance between channel c and channel c'.
     */
    float covariance[4][4];
} PVAImageStatOutput;

/**
 * @brief Data types for the Gaussian Mixture Model (GMM) used in background subtraction.
 *
 * Specifies the available formats for storing GMM parameters such as weight, variance, and mean.
 */
typedef enum
{
    PVA_GMM_DATA_TYPE_FP32,
    PVA_GMM_DATA_TYPE_FP16
} PVAGMMDataType;

/**
 * @brief Batch FFT window types
 */
typedef enum
{
    /// Hanning window
    PVA_BATCH_FFT_WINDOW_HANNING,
    /// User defined window
    PVA_BATCH_FFT_WINDOW_USER_DEFINED,
} PVABatchFFTWindowType;

/**
 * @brief Range FFT parameters
 */
typedef struct PVARangeFFTParamsRec
{
    /// Window type
    PVABatchFFTWindowType windowType;
} PVARangeFFTParams;

/**
 * @enum PVADopplerFFTOutputLayout
 * @brief Output layout options for Doppler FFT operation
 */
typedef enum
{
    /// Output layout: (Doppler bins x Rx channels x range bins)
    PVA_DOPPLER_FFT_OUTPUT_LAYOUT_DOPPLER_RX_RANGE = 0,
    /// Output layout: (range bins x Rx channels x Doppler bins) - transposed
    PVA_DOPPLER_FFT_OUTPUT_LAYOUT_RANGE_RX_DOPPLER = 1,
    /// Output layout: (range bins x Doppler bins x Rx channels) - transposed with L2 buffering
    PVA_DOPPLER_FFT_OUTPUT_LAYOUT_RANGE_DOPPLER_RX = 2
} PVADopplerFFTOutputLayout;

/**
 * @brief Doppler FFT parameters
 */
typedef struct PVADopplerFFTParamsRec
{
    /// Window type
    PVABatchFFTWindowType windowType;
    /// Output layout
    PVADopplerFFTOutputLayout outputLayout;
} PVADopplerFFTParams;

/**
 * @enum PVAFlipDirection
 * @brief Flip direction options for image flip operation
 */
typedef enum
{
    /// Flip image horizontally (left-right)
    PVA_FLIP_HORIZONTAL = 0,
    /// Flip image vertically (up-down)
    PVA_FLIP_VERTICAL = 1,
    /// Flip image both horizontally and vertically (180° rotation)
    PVA_FLIP_BOTH = 2
} PVAFlipDirection;

/**
 * @brief Radar CFAR algorithm types for target detection.
 *
 * This enumeration defines the available Constant False Alarm Rate (CFAR) algorithms
 * that can be used for radar target detection. Each algorithm provides different
 * approaches to estimate the noise floor and set adaptive thresholds for target detection.
 */
typedef enum
{
    /** @brief Cell Averaging (CA) algorithm - Thresholds are set by averaging the signal values from both leading and trailing training cells */
    PVA_CFAR_CA = 0,
    /** @brief Order Statistic (OS) algorithm - Thresholds are set using the median value of the ranked signal amplitudes from both leading and trailing training cells */
    PVA_CFAR_CA_OS = 1,
    /** @brief Greatest Of (GO) algorithm - Thresholds are set using the greater of the averages from the leading and trailing training cells */
    PVA_CFAR_CA_GO = 2,
    /** @brief Smallest Of (SO) algorithm - Thresholds are set using the smaller of the averages from the leading and trailing training cells */
    PVA_CFAR_CA_SO = 3
} PVARadarCFARType;

/** @brief Configuration parameters for radar CFAR target detection operator.
 *
 * This structure contains all the necessary parameters to configure the Constant False Alarm Rate (CFAR)
 * algorithm for radar target detection. The parameters control the detection sensitivity, algorithm type,
 * and processing window dimensions for optimal target detection performance.
 *
 * The CFAR algorithm uses training cells to estimate the noise floor and guard cells to protect
 * the cell under test (CUT) from target energy leakage, ensuring accurate threshold calculation.
 *
 * CFAR Cell Arrangement:
 * N = numTrain
 * K = numGuard
 *
 * @verbatim
 * | T1 | T2 | T3 |......| TN | G1 | ... | GK | CUT | G1 | ... | GK | T1 | T2 | T3 |......| TN |
 * |<-------- Trailing ------>|<--- Guard --->|     |<--- Guard --->|<-------- Leading ------->|
 * @endverbatim
 */
typedef struct PvaRadarCFARParamsRec
{
    /** Number of horizontal training cells for CFAR algorithm (range: 0 < numHorGuard < numHorTrain <= (Width-1)/2)
     * Constraint: 0 <= numHorGuard < numHorTrain < 256
     * Constraint: 1 <= numHorTrain + numHorGuard < 256
     * Training cells are used to estimate the noise floor in the horizontal direction
     */
    int32_t numHorTrain;
    /** Number of horizontal guard cells for CFAR algorithm (range: 0 <= numHorGuard < numHorTrain)
     * Constraint: 0 <= numHorGuard < numHorTrain < 256
     * Guard cells protect the cell under test from target energy leakage in horizontal direction
     */
    int32_t numHorGuard;
    /** Number of vertical training cells for CFAR algorithm (range: 0 < numVerGuard < numVerTrain <= (Height-1)/2)
     * Constraint: 0 <= numVerGuard < numVerTrain < 256
     * Constraint: 1 <= numVerTrain + numVerGuard < 256
     * Training cells are used to estimate the noise floor in the vertical direction
     */
    int32_t numVerTrain;
    /** Number of vertical guard cells for CFAR algorithm (range: 0 <= numVerGuard < numVerTrain)
     * Constraint: 0 <= numVerGuard < numVerTrain < 256
     * Guard cells protect the cell under test from target energy leakage in vertical direction
     */
    int32_t numVerGuard;
    /** CFAR horizontal threshold multiplier
     * Controls detection sensitivity - higher values increase detection threshold
     * noiseEstimate is the noise estimate in the horizontal direction
     * detectionThreshold = noiseEstimate * thresholdFactor
     */
    float horizontalThresholdFactor;
    /** CFAR vertical threshold multiplier
     * Controls detection sensitivity - higher values increase detection threshold
     * detectionThreshold = noiseEstimate * thresholdFactor
     * noiseEstimate is the noise estimate in the vertical direction
     */
    float verticalThresholdFactor;
    /** Controls padding mode for horizontal dimension during CFAR processing.
     *
     * When enabled (true):
     *   - Uses cyclic padding, Uses the training cells from the opposite edge for boundary training cells.
     *   - Recommended when horizontal data exhibits periodic/cyclic characteristics or when
     *     background noise and clutter properties are assumed to be consistent across edges.
     *
     * When disabled (false):
     *   - Uses available neighbouring pixels for noise estimation at boundaries.
     *   - Training cells beyond image boundaries are not included in noise estimation.
     *   - Recommended when the signal represents an event that is known to start and end within the collected
     *     data window, with a genuinely zero-level background outside that window.
     */
    bool isHorizontalCyclicPadding;
    /** Controls padding mode for vertical dimension during CFAR processing.
     *
     * When enabled (true):
     *   - Uses cyclic padding, Uses the training cells from the opposite edge for boundary training cells.
     *   - Recommended when vertical data exhibits periodic/cyclic characteristics or when
     *     background noise and clutter properties are assumed to be consistent across edges.
     *
     * When disabled (false):
     *   - Uses available neighbouring pixels for noise estimation at boundaries.
     *   - Training cells beyond image boundaries are not included in noise estimation.
     *   - Recommended when the signal represents an event that is known to start and end within the collected
     *     data window, with a genuinely zero-level background outside that window.
     */
    bool isVerticalCyclicPadding;
    /** Enables peak grouping algorithm for target detection refinement
     * When enabled, the algorithm filters detections to retain only local maxima by comparing
     * each detection against its immediate horizontal and vertical neighbors.
     */
    bool enablePeakGrouping;
} PvaRadarCFARParams;

/**
 * @brief CCL Connectivity Types
 */
typedef enum
{
    /// 4-connectivity
    PVA_4_CONNECTIVITY = 0,
    /// 8-connectivity
    PVA_8_CONNECTIVITY = 1,
} PVAConnectivityType;

#ifdef __cplusplus
}
#endif

/** @} */
#endif /* PVA_OPERATOR_TYPES_H */
