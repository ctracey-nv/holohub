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
 * @file PvaOperator.h
 *
 * @brief Defines types and functions to handle operators
 * @defgroup PVA_OPERATOR_CORE_PVA_OPERATOR PvaOperator
 * @{
 */

#ifndef PVA_OPERATOR_H
#define PVA_OPERATOR_H

// Include submit function helper declarations for C _Generic dispatch
#include <PvaSubmitDefinitions.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @cond SKIP_THIS */
typedef struct NVCVOperator *NVCVOperatorHandle;
/** @endcond */

/**
 * Destroys the operator instance.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 */
void nvcvOperatorDestroy(NVCVOperatorHandle handle);

#ifdef __cplusplus
}
#endif

/**
 * @page PVA_CUDA_STREAM_REQUIREMENTS CUDA Stream Support Requirements
 *
 * @note CUDA stream support requirements:
 *       - PVA SDK 2.7.0 or later
 *       - Jetpack 7 or later
 *       - DriveOS 7 or later
 *       - x86 Emulator is not supported
 */

/** @} */
#endif /* PVA_OPERATOR_H */
