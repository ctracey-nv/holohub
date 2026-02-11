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
 * @file RadarAdvancedOperatorTypes.h
 *
 * @brief Type definitions for radar advanced operators
 */

#ifndef PVA_SOLUTIONS_RADAR_ADVANCED_OPERATOR_TYPES_H
#define PVA_SOLUTIONS_RADAR_ADVANCED_OPERATOR_TYPES_H

#include <OpDopplerFFT.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Angle estimation precision levels for DOA Angle FFT operator.
 *
 * This enum controls the zero-padding applied to the virtual antenna array
 * snapshot before performing the spatial FFT for angle estimation. The precision
 * level determines the FFT scaling factor (zero-padding factor), which affects
 * the density of the frequency grid in the FFT output.
 *
 * Zero-padding produces a denser frequency grid, which is equivalent to sinc
 * interpolation in the angle domain. This enables more precise localization of
 * spectral peaks without increasing the fundamental angular resolution (which
 * is determined solely by the physical antenna aperture).
 *
 * The FFT lengths are computed as:
 *   - Azimuth FFT length:   numVirtualAzimuthElements * scalingFactor
 *   - Elevation FFT length: numVirtualElevationElements * scalingFactor
 */
typedef enum
{
    PVA_ANGLE_ESTIMATION_PRECISION_LOW     = 0, ///< Minimal zero-padding, fastest
    PVA_ANGLE_ESTIMATION_PRECISION_DEFAULT = 1, ///< Balanced precision and performance
    PVA_ANGLE_ESTIMATION_PRECISION_HIGH    = 2  ///< Maximum precision, slowest
} PVADoaAngleFFTPrecision;

/**
 * @brief Parameters for the Target Processing operator.
 *
 * This structure contains parameters for target processing, including the standard
 * radar parameters and optional RV (Range-Velocity) decoupling parameters.
 *
 * RV Decoupling:
 * When enableRvDecoupling is true, the operator uses FMCW coupling formulas to
 * correct for the range-velocity coupling in FMCW radar systems. This provides
 * more accurate range and velocity estimates compared to the standard method.
 *
 * The RV decoupling algorithm uses a 5-hypothesis velocity disambiguation loop.
 */
typedef struct
{
    /// RF wavelength (meters)
    float wavelength;
    /// Chirp sweep time (seconds)
    float sweepTime;
    /// Range bin resolution (meters)
    float rangeResolution;
    /// Number of chirps
    int32_t numChirps;
    /// Flag to specify if Doppler FFT is shifted to make zero velocity Doppler bin the center of the FFT bins
    bool fftShiftedDoppler;
    /// Enable range-velocity decoupling
    bool enableRvDecoupling;
    /// Enable quadratic interpolation for sub-bin accuracy (requires NCI data)
    bool enableQuadraticInterpolation;

    // =========================================================================
    // RV Decoupling Parameters (only used when enableRvDecoupling is true)
    // Base FMCW radar parameters - derived values are computed internally
    // by the operator (following the DOA operator pattern).
    // =========================================================================

    /// Sampling frequency (Hz) - used to compute f_fast_delta = fs / numSamples
    float fs;
    /// Number of samples per chirp - used with fs for f_fast_delta
    int32_t nofSamples;
    /// Pulse Repetition Interval (seconds) - used for f_slow_delta and tempVCoeff
    float PRI;
    /// Velocity resolution (km/h) - used for velocityAmbHigh/Low computation (converted to m/s internally)
    float deltaV;
    /// Range coupling constant: contR = c / (2 * slope) (meters per Hz)
    float contR;
    /// Velocity fast constant for FMCW coupling (m/s per Hz)
    float contVFast;
    /// Velocity slow constant for FMCW coupling (m/s per Hz)
    float contVSlow;

    // =========================================================================
    // Coordinate System Options (for spherical-to-Cartesian conversion)
    // =========================================================================

    /// Longitudinal axis convention for Cartesian coordinates
    /// 0 = X_FORWARD: X is forward (standard spherical), Y is lateral
    /// 1 = Y_FORWARD: Y is forward (automotive radar convention), X is lateral
    int32_t longitudinalAxis;
    /// Ground projection mode
    /// 0 = SLANT_RANGE: X/Y use slant range directly (no cos(elevation) factor)
    /// 1 = GROUND_PLANE: X/Y scaled by cos(elevation) for ground-plane position
    int32_t groundProjection;
} PVATargetProcessingParams;

/**
 * @brief Parameters for the DOA Angle FFT and Target Processing operator.
 *
 * This structure combines parameters for DOA (Direction of Arrival) angle estimation
 * and target processing. The operator performs 2D Angle FFT to estimate azimuth/elevation
 * angles, then converts the radar measurements into physical coordinates and velocities.
 */
typedef struct
{
    // DoA Angle FFT parameters
    /// Number of virtual horizontal (azimuth) antenna elements (assumed half-wavelength spacing)
    int32_t numVirtualAzimuthElements;
    /// Number of virtual vertical (elevation) antenna elements (assumed half-wavelength spacing)
    int32_t numVirtualElevationElements;
    /// Angle estimation precision level.
    ///
    /// Controls the zero-padding applied to the virtual antenna array snapshot
    /// before performing the spatial FFT for angle estimation. See @ref PVADoaAngleFFTPrecision
    /// for details on precision levels and their corresponding scaling factors.
    ///
    /// The scaling factor determines the FFT lengths:
    ///   - Azimuth FFT length:   numVirtualAzimuthElements * scalingFactor
    ///   - Elevation FFT length: numVirtualElevationElements * scalingFactor
    ///
    /// Higher precision provides finer angle bin spacing, improving peak detection
    /// accuracy at the cost of increased computation.
    PVADoaAngleFFTPrecision angleEstimationPrecision;
    /// Enable local peak detection to find multiple targets per detection
    bool enableLocalPeakDetection;

    PVATargetProcessingParams targetProcessingParams;
} PVADoaAngleFFTParams;

/**
 * @brief Parameters for the Snapshot Extraction operator.
 */
typedef struct
{
    /// Number of Doppler folds for DDM disambiguation
    int32_t numDopplerFolds;
    /// Range-Doppler map memory layout format
    /// Use PVA_DOPPLER_FFT_OUTPUT_LAYOUT_RANGE_RX_DOPPLER or PVA_DOPPLER_FFT_OUTPUT_LAYOUT_RANGE_DOPPLER_RX
    PVADopplerFFTOutputLayout rangeDopplerFormat;
} PVASnapshotExtractionParams;

/**
 * @brief Parameters for the Bartlett Beamforming DOA + Target Processing operator.
 *
 * This structure combines parameters for DOA (Direction of Arrival) angle estimation
 * using Bartlett beamforming and target processing. The operator performs beamforming
 * to estimate azimuth/elevation angles, then converts the radar measurements into
 * physical coordinates and velocities.
 *
 * \b Important:
 *
 *      Quadratic interpolation (enableInterpolation = true) is strongly recommended
 *      for coarse bin spacing to achieve sub-degree accuracy.
 *      Without interpolation, angle accuracy degrades significantly
 */
typedef struct
{
    // Bartlett Beamforming parameters
    /// Enable quadratic interpolation for sub-bin accuracy
    bool enableInterpolation;
    /// Enable peak power (dB) output in target list
    bool enablePowerOutput;

    // Target Processing parameters
    PVATargetProcessingParams targetProcessingParams;
} PVADoaBartlettBeamformingParams;

#ifdef __cplusplus
}

#endif

#endif /* PVA_SOLUTIONS_RADAR_ADVANCED_OPERATOR_TYPES_H */
