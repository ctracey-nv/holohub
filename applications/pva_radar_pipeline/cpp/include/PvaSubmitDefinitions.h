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
 * @file PvaSubmitDefinitions.h
 *
 * @brief Declarations and C dispatch macros for PVA operator submit functions.
 *        This file contains the actual function declarations and C _Generic dispatchers.
 */

#ifndef PVA_SOLUTIONS_PVASUBMITDEFINITIONS_H
#define PVA_SOLUTIONS_PVASUBMITDEFINITIONS_H

#include "PvaSubmitMacros.h"

#include <PvaOperatorTypes.h>
#include <cupva_host_scheduling.h>
#include <nvcv/Image.h>

// Forward declarations for types used in submit function declarations
typedef struct PvaDLInferenceSubmitParamRec PvaDLInferenceSubmitParams;
typedef float WarpMatrixType[3][3];
typedef struct PvaRadarCFARParamsRec PvaRadarCFARParams;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Internal implementation: Executes morphology operation with cuPVA/CUDA stream.
 * Users should call pvaMorphologySubmit() instead.
 */
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaMorphologySubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaImageHistogramSubmit, (NVCVTensorHandle const, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaCornerSubPixSubmit, (NVCVTensorHandle, inCorners), (NVCVTensorHandle, outCorners),
                             (NVCVTensorHandle, image), (int32_t const, maxIters), (float const, eps),
                             (int32_t const, numCorners));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDepthToSpaceSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaMinMaxLocSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, minVal),
                             (NVCVTensorHandle, minLoc), (NVCVTensorHandle, numMin), (NVCVTensorHandle, maxVal),
                             (NVCVTensorHandle, maxLoc), (NVCVTensorHandle, numMax));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaTemplateMatchingSubmit, (NVCVTensorHandle, inImage), (NVCVTensorHandle, inTemplate),
                             (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaMixChannelsSubmit, (NVCVTensorHandle *, inHandles), (NVCVTensorHandle *, outHandles),
                             (int, inTensorCount), (int, outTensorCount));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBlToPlSubmit, (NVCVImageHandle, in), (NVCVImageHandle, out));

DECLARE_PVA_SUBMIT_FUNCTIONS(pvaConv2dSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaGaussianFilterSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaGaussianPyramidSubmit, (NVCVImageHandle, inImageHandle),
                             (NVCVImageHandle *, outImageHandles));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBoxFilterSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaConvertImageFormatSubmit, (NVCVImageHandle const, in), (NVCVImageHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaFastCornerDetectorSubmit, (NVCVTensorHandle, in), (int32_t, intensityThreshold),
                             (NVCVTensorHandle, loc), (NVCVTensorHandle, numLoc));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaImageBlendSubmit, (NVCVImageHandle const, in0), (NVCVImageHandle const, in1),
                             (float const, alpha), (NVCVImageHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBatchSVDSubmit, (NVCVTensorHandle const, src), (NVCVTensorHandle const, U),
                             (NVCVTensorHandle const, S), (NVCVTensorHandle const, V), (int32_t const, maxIters),
                             (float const, tol));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaImageResizeSubmit, (NVCVImageHandle, in), (NVCVImageHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBlurFilterROISubmit, (NVCVImageHandle const, in), (NVCVImageHandle, out),
                             (NVCVTensorHandle const, rect), (size_t const, numRects));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDistanceTransformSubmit, (NVCVTensorHandle const, inImage), (uint16_t, maxDistance),
                             (NVCVTensorHandle, outDistance), (NVCVTensorHandle, outLabel));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBilateralFilterSubmit, (NVCVTensorHandle, in), (float const, sigmaRange),
                             (float const, sigmaSpace), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBruteForceMatcherSubmit, (PVABriefDescriptor *, query), (int32_t, queryCount),
                             (PVABriefDescriptor *, reference), (int32_t, referenceCount),
                             (int32_t, maxMatchesPerQuery), (PVABFMatchesType *, matches), (uint8_t, enableCrossCheck),
                             (uint8_t, enableDistanceRatioTest), (int32_t, lowesTestThresholdNumerator),
                             (int32_t, lowesTestThresholdDenominator));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaImageStatsSubmit, (NVCVImageHandle, in), (NVCVImageHandle, mask),
                             (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaCannyEdgeDetectorSubmit, (cupvaCmdStatus_t *, cmdStatus), (NVCVTensorHandle, image),
                             (NVCVTensorHandle, edgeMap), (int32_t const, thresholdStrong),
                             (int32_t const, thresholdWeak));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaMedianFilterSubmit, (NVCVImageHandle, in), (NVCVTensorHandle, kernel),
                             (NVCVImageHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaHistogramEqualizationSubmit, (NVCVTensorHandle, in), (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDLInferenceSubmit, (PvaDLInferenceSubmitParams *, params));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaWarpPerspectiveSubmit, (NVCVImageHandle, in), (WarpMatrixType, invWarpMatrix),
                             (NVCVImageHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaBackgroundSubtractorSubmit, (NVCVImageHandle, in), (NVCVImageHandle, outMask),
                             (NVCVImageHandle, outBackgroundImage), (float, learningRate), (float, varThreshold),
                             (bool, enableShadowDetection), (uint8_t, shadowPixelValue));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaRangeFFTSubmit, (NVCVTensorHandle const, in), (NVCVTensorHandle const, win),
                             (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaDopplerFFTSubmit, (NVCVTensorHandle const, in), (NVCVTensorHandle const, win),
                             (NVCVTensorHandle, out));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaNciSubmit, (NVCVTensorHandle const *, inHandles), (NVCVTensorHandle *, outHandles));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaRemapSubmit, (NVCVImageHandle, in), (NVCVImageHandle, out),
                             (NVCVTensorHandle, warpMap));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaORBDescExtractorSubmit, (NVCVImageHandle, in),
                             (NVCVTensorHandle, inCornersTensorHandle), (NVCVTensorHandle, outDescriptorsTensorHandle));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaImageFlipSubmit, (NVCVImageHandle, in), (NVCVImageHandle, out),
                             (PVAFlipDirection, flipDirection));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaRadarCFARSubmit, (NVCVTensorHandle const, inputTensor),
                             (NVCVTensorHandle, outDetectionListTensor), (NVCVTensorHandle, outDetectionCountTensor),
                             (PvaRadarCFARParams const *, radarCFARParams));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaFloydWarshallSubmit, (NVCVTensorHandle, lenTensor), (NVCVTensorHandle, nextTensor));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaCCLSubmit, (NVCVTensorHandle, image), (NVCVTensorHandle, labels),
                             (NVCVTensorHandle, numLabels));
DECLARE_PVA_SUBMIT_FUNCTIONS(pvaHOGSubmit, (NVCVImageHandle const, in), (NVCVTensorHandle const, out));
#ifdef __cplusplus
}
#endif

// C-only: _Generic dispatch for type-based function selection
#ifndef __cplusplus
#    define pvaMorphologySubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaMorphologySubmit, handle, stream, in, out)
#    define pvaImageHistogramSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaImageHistogramSubmit, handle, stream, in, out)
#    define pvaCornerSubPixSubmit(handle, stream, inCorners, outCorners, image, maxIters, eps, numCorners)             \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaCornerSubPixSubmit, handle, stream, inCorners, outCorners, image, maxIters, eps, \
                                   numCorners)
#    define pvaDepthToSpaceSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDepthToSpaceSubmit, handle, stream, in, out)
#    define pvaMinMaxLocSubmit(handle, stream, in, minVal, minLoc, numMin, maxVal, maxLoc, numMax)                 \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaMinMaxLocSubmit, handle, stream, in, minVal, minLoc, numMin, maxVal, maxLoc, \
                                   numMax)
#    define pvaTemplateMatchingSubmit(handle, stream, inImage, inTemplate, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaTemplateMatchingSubmit, handle, stream, inImage, inTemplate, out)
#    define pvaMixChannelsSubmit(handle, stream, inHandles, outHandles, inTensorCount, outTensorCount)         \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaMixChannelsSubmit, handle, stream, inHandles, outHandles, inTensorCount, \
                                   outTensorCount)
#    define pvaBlToPlSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBlToPlSubmit, handle, stream, in, out)
#    define pvaConv2dSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaConv2dSubmit, handle, stream, in, out)
#    define pvaGaussianFilterSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaGaussianFilterSubmit, handle, stream, in, out)
#    define pvaGaussianPyramidSubmit(handle, stream, inImageHandle, outImageHandles) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaGaussianPyramidSubmit, handle, stream, inImageHandle, outImageHandles)
#    define pvaBoxFilterSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBoxFilterSubmit, handle, stream, in, out)
#    define pvaConvertImageFormatSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaConvertImageFormatSubmit, handle, stream, in, out)
#    define pvaFastCornerDetectorSubmit(handle, stream, in, intensityThreshold, loc, numLoc) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaFastCornerDetectorSubmit, handle, stream, in, intensityThreshold, loc, numLoc)
#    define pvaImageBlendSubmit(handle, stream, in0, in1, alpha, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaImageBlendSubmit, handle, stream, in0, in1, alpha, out)
#    define pvaBatchSVDSubmit(handle, stream, src, U, S, V, maxIters, tol) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBatchSVDSubmit, handle, stream, src, U, S, V, maxIters, tol)
#    define pvaImageResizeSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaImageResizeSubmit, handle, stream, in, out)
#    define pvaBlurFilterROISubmit(handle, stream, in, out, rect, numRects) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBlurFilterROISubmit, handle, stream, in, out, rect, numRects)
#    define pvaDistanceTransformSubmit(handle, stream, inImage, maxDistance, outDistance, outLabel)               \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDistanceTransformSubmit, handle, stream, inImage, maxDistance, outDistance, \
                                   outLabel)
#    define pvaBilateralFilterSubmit(handle, stream, in, sigmaRange, sigmaSpace, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBilateralFilterSubmit, handle, stream, in, sigmaRange, sigmaSpace, out)
#    define pvaBruteForceMatcherSubmit(handle, stream, query, queryCount, reference, referenceCount,           \
                                       maxMatchesPerQuery, matches, enableCrossCheck, enableDistanceRatioTest, \
                                       lowesTestThresholdNumerator, lowesTestThresholdDenominator)             \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBruteForceMatcherSubmit, handle, stream, query, queryCount, reference,   \
                                   referenceCount, maxMatchesPerQuery, matches, enableCrossCheck,              \
                                   enableDistanceRatioTest, lowesTestThresholdNumerator,                       \
                                   lowesTestThresholdDenominator)
#    define pvaImageStatsSubmit(handle, stream, in, mask, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaImageStatsSubmit, handle, stream, in, mask, out)
#    define pvaCannyEdgeDetectorSubmit(handle, stream, cmdStatus, image, edgeMap, thresholdStrong, thresholdWeak) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaCannyEdgeDetectorSubmit, handle, stream, cmdStatus, image, edgeMap,         \
                                   thresholdStrong, thresholdWeak)
#    define pvaMedianFilterSubmit(handle, stream, in, kernel, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaMedianFilterSubmit, handle, stream, in, kernel, out)
#    define pvaHistogramEqualizationSubmit(handle, stream, in, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaHistogramEqualizationSubmit, handle, stream, in, out)
#    define pvaDLInferenceSubmit(handle, stream, params) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDLInferenceSubmit, handle, stream, params)
#    define pvaWarpPerspectiveSubmit(handle, stream, in, invWarpMatrix, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaWarpPerspectiveSubmit, handle, stream, in, invWarpMatrix, out)
#    define pvaBackgroundSubtractorSubmit(handle, stream, in, outMask, outBackgroundImage, learningRate, varThreshold, \
                                          enableShadowDetection, shadowPixelValue)                                     \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaBackgroundSubtractorSubmit, handle, stream, in, outMask, outBackgroundImage,     \
                                   learningRate, varThreshold, enableShadowDetection, shadowPixelValue)
#    define pvaRangeFFTSubmit(handle, stream, in, win, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaRangeFFTSubmit, handle, stream, in, win, out)
#    define pvaDopplerFFTSubmit(handle, stream, in, win, out) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaDopplerFFTSubmit, handle, stream, in, win, out)
#    define pvaNciSubmit(handle, stream, inHandles, outHandles) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaNciSubmit, handle, stream, inHandles, outHandles)
#    define pvaRemapSubmit(handle, stream, in, out, warpMap) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaRemapSubmit, handle, stream, in, out, warpMap)
#    define pvaORBDescExtractorSubmit(handle, stream, in, inCornersTensorHandle, outDescriptorsTensorHandle) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaORBDescExtractorSubmit, handle, stream, in, inCornersTensorHandle,     \
                                   outDescriptorsTensorHandle)
#    define pvaImageFlipSubmit(handle, stream, in, out, flipDirection) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaImageFlipSubmit, handle, stream, in, out, flipDirection)
#    define pvaRadarCFARSubmit(handle, stream, inputTensor, outDetectionListTensor, outDetectionCountTensor, \
                               radarCFARParams)                                                              \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaRadarCFARSubmit, handle, stream, inputTensor, outDetectionListTensor,  \
                                   outDetectionCountTensor, radarCFARParams)
#    define pvaFloydWarshallSubmit(handle, stream, lenTensor, nextTensor) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaFloydWarshallSubmit, handle, stream, lenTensor, nextTensor)
#    define pvaCCLSubmit(handle, stream, image, labels, numLabels) \
        DEFINE_PVA_SUBMIT_DISPATCH(pvaCCLSubmit, handle, stream, image, labels, numLabels)
#    define pvaHOGSubmit(handle, stream, in, out) DEFINE_PVA_SUBMIT_DISPATCH(pvaHOGSubmit, handle, stream, in, out)
#endif

#endif /* PVA_SOLUTIONS_PVASUBMITDEFINITIONS_H */
