/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gxf/std/tensor.hpp"
#include "holoscan/holoscan.hpp"

#include <ErrorCheckMacros.h>
#include <OpConvertImageFormat.h>
#include <OpGaussianFilter.h>
#include <PvaAllocator.h>
#include <cupva_host.h>
#include <cstdint>
#include <holoscan/core/system/gpu_resource_monitor.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_recorder/video_stream_recorder.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <iostream>
#include <string>

class PvaStuff {
 public:
  PvaStuff() = default;
  void init(int32_t width, int32_t height, int32_t inputLinePitch, int32_t outputLinePitch) {
    m_width = width;
    m_height = height;
    m_inputLinePitch = inputLinePitch;
    m_outputLinePitch = outputLinePitch;
    int32_t error = 0;
    NVCVAllocatorHandle alloc;
    NVCVImageRequirements reqRGB = {0};
    NVCVImageRequirements reqYUV = {0};
    NVCVTensorRequirements reqY = {0};
    NVCVTensorLayout layout;
    int64_t shape[3] = {height, width, 1};

    /// TODO: querry CUDA interop support
    m_useCUDAInterop = false;

    /// create a cupva stream if cuda stream cannot be used
    if (!m_useCUDAInterop) {
      CUPVA_CHECK_ERROR_GOTO(CupvaStreamCreate(&m_stream, CUPVA_PVA0, CUPVA_VPU_ANY), error, exit);
      CUPVA_CHECK_ERROR_GOTO(
          CupvaSyncObjCreate(&m_syncObj, false, CUPVA_SIGNALER_WAITER, CUPVA_SYNC_YIELD),
          error,
          exit);
    }

    /// create a PVA allocator
    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorConstructPva(&alloc), error, exit);

    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(width, height, NVCV_IMAGE_FORMAT_RGB8, 0, 0, &reqRGB),
        error,
        exit);
    assert(reqRGB.planeRowStride[0] == inputLinePitch);
    assert(reqRGB.planeRowStride[0] == outputLinePitch);

    if (!m_useCUDAInterop) {
      NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&reqRGB, alloc, &m_rgbImageIn), error, exit);
      NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&reqRGB, alloc, &m_rgbImageOut), error, exit);
    }

    NVCV_CHECK_ERROR_GOTO(
        nvcvImageCalcRequirementsPva(width, height, NVCV_IMAGE_FORMAT_NV12, 0, 0, &reqYUV),
        error,
        exit);

    NVCV_CHECK_ERROR_GOTO(nvcvTensorLayoutMake("HWC", &layout), error, exit);

    NVCV_CHECK_ERROR_GOTO(
        nvcvTensorCalcRequirementsPva(3, shape, NVCV_DATA_TYPE_U8, layout, 0, 0, &reqY),
        error,
        exit);

    if (m_useCUDAInterop) {
      /// TODO: share one slab of cuda memory for each YUV/Y pair
    } else {
      NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&reqYUV, alloc, &m_yuvImageIn), error, exit);
      NVCV_CHECK_ERROR_GOTO(nvcvImageConstruct(&reqYUV, alloc, &m_yuvImageOut), error, exit);
      NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reqY, alloc, &m_yTensorIn), error, exit);
      NVCV_CHECK_ERROR_GOTO(nvcvTensorConstruct(&reqY, alloc, &m_yTensorOut), error, exit);
    }

    NVCV_CHECK_ERROR_GOTO(
        pvaConvertImageFormatCreate(&m_convertRGBToYUVHandle, &reqRGB, &reqYUV), error, exit);
    NVCV_CHECK_ERROR_GOTO(
        pvaConvertImageFormatCreate(&m_convertYUVToRGBHandle, &reqYUV, &reqRGB), error, exit);

    NVCV_CHECK_ERROR_GOTO(
        pvaGaussianFilterCreate(
            &m_gaussianFilterHandle, &reqY, 1.0f, 1.0f, 5, NVCV_BORDER_REPLICATE, 0),
        error,
        exit);

    NVCV_CHECK_ERROR_GOTO(nvcvAllocatorDecRef(alloc, nullptr), error, exit);

    m_initialized = true;
    return;
  exit:
    throw std::runtime_error("Fatal nvcv or cupva error in init()");
  }

  void process(uint8_t* src, uint8_t* dst, cudaStream_t stream) {
    int32_t error = 0;

    if (m_useCUDAInterop) {
      /// TODO: wrap input cuda buffer into an NVCVImage (zero-copy)
      // NVCVImageData inData = {0};
      // inData.format = NVCV_IMAGE_FORMAT_RGB8;
      // inData.bufferType = NVCV_IMAGE_BUFFER_STRIDED_CUDA;
      // inData.buffer.strided.numPlanes = 1;
      // inData.buffer.strided.planes[0].width = m_width;
      // inData.buffer.strided.planes[0].height = m_height;
      // inData.buffer.strided.planes[0].rowStride = m_inputLinePitch;
      // inData.buffer.strided.planes[0].basePtr = src;
      // nvcvImageWrapDataConstruct(&inData, NULL, nullptr, &m_rgbImageIn);
    } else {
      /// explicit copy input cuda buffer to host memory mapped to PVA
      NVCVImageData inData;
      NVCV_CHECK_ERROR_GOTO(nvcvImageExportData(m_rgbImageIn, &inData), error, exit);
      auto inDev = inData.buffer.strided.planes[0].basePtr;
      uint8_t* inHost;
      CUPVA_CHECK_ERROR_GOTO(CupvaMemGetHostPointer((void**)&inHost, inDev), error, exit);
      assert(inData.buffer.strided.planes[0].rowStride == m_inputLinePitch);
      cudaMemcpyAsync(inHost, src, m_inputLinePitch * m_height, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
    }
    /// TODO: use cuda stream instead of cupva stream
    NVCV_CHECK_ERROR_GOTO(
        pvaConvertImageFormatSubmit(m_convertRGBToYUVHandle, m_stream, m_rgbImageIn, m_yuvImageIn),
        error,
        exit);

    if (!m_useCUDAInterop) {
      // wait on cupva stream
      cupvaFence_t fence;
      CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, m_syncObj), error, exit);
      cupvaCmd_t rf;
      CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), error, exit);
      cupvaCmd_t const* cmds[1] = {&rf};
      cupvaCmdStatus_t status[1] = {NULL};
      CUPVA_CHECK_ERROR_GOTO(
          CupvaStreamSubmit(m_stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), error, exit);
      bool waitResult;
      CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitResult), error, exit);
      assert(waitResult);

      /// Copy Y plane from YUV image to Y tensor
      NVCVImageData yuvData;
      NVCV_CHECK_ERROR_GOTO(nvcvImageExportData(m_yuvImageIn, &yuvData), error, exit);
      auto yPlane = yuvData.buffer.strided.planes[0].basePtr;
      uint8_t* yPlaneHostPtr;
      CUPVA_CHECK_ERROR_GOTO(CupvaMemGetHostPointer((void**)&yPlaneHostPtr, yPlane), error, exit);
      assert(yuvData.buffer.strided.planes[0].rowStride == m_inputLinePitch);

      NVCVTensorData yData;
      NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(m_yTensorIn, &yData), error, exit);
      auto yDevPtr = yData.buffer.strided.basePtr;
      uint8_t* yHostPtr;
      CUPVA_CHECK_ERROR_GOTO(CupvaMemGetHostPointer((void**)&yHostPtr, yDevPtr), error, exit);
      assert(yData.buffer.strided.strides[0] == m_inputLinePitch);

      memcpy(yHostPtr, yPlaneHostPtr, m_inputLinePitch * m_height);
    }
    NVCV_CHECK_ERROR_GOTO(
        pvaGaussianFilterSubmit(m_gaussianFilterHandle, m_stream, m_yTensorIn, m_yTensorOut),
        error,
        exit);

    if (!m_useCUDAInterop) {
      // wait on cupva stream
      cupvaFence_t fence;
      CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, m_syncObj), error, exit);
      cupvaCmd_t rf;
      CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), error, exit);
      cupvaCmd_t const* cmds[1] = {&rf};
      cupvaCmdStatus_t status[1] = {NULL};
      CUPVA_CHECK_ERROR_GOTO(
          CupvaStreamSubmit(m_stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), error, exit);
      bool waitResult;
      CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitResult), error, exit);
      assert(waitResult);

      /// Copy blurred Y plane from Y tensor to output YUV image
      NVCVTensorData yData;
      NVCV_CHECK_ERROR_GOTO(nvcvTensorExportData(m_yTensorOut, &yData), error, exit);
      auto yDevPtr = yData.buffer.strided.basePtr;
      uint8_t* yHostPtr;
      CUPVA_CHECK_ERROR_GOTO(CupvaMemGetHostPointer((void**)&yHostPtr, yDevPtr), error, exit);
      assert(yData.buffer.strided.strides[1] == m_inputLinePitch);

      NVCVImageData yuvData;
      NVCV_CHECK_ERROR_GOTO(nvcvImageExportData(m_yuvImageOut, &yuvData), error, exit);
      auto yPlane = yuvData.buffer.strided.planes[0].basePtr;
      uint8_t* yPlaneHostPtr;
      CUPVA_CHECK_ERROR_GOTO(CupvaMemGetHostPointer((void**)&yPlaneHostPtr, yPlane), error, exit);
      assert(yuvData.buffer.strided.planes[0].rowStride == m_outputLinePitch);

      memcpy(yPlaneHostPtr, yHostPtr, m_outputLinePitch * m_height);

      auto uvOutPlane = yuvData.buffer.strided.planes[1].basePtr;
      uint8_t* uvOutPlaneHostPtr;
      CUPVA_CHECK_ERROR_GOTO(
          CupvaMemGetHostPointer((void**)&uvOutPlaneHostPtr, uvOutPlane), error, exit);
      assert(yuvData.buffer.strided.planes[1].rowStride == m_outputLinePitch);

      NVCV_CHECK_ERROR_GOTO(nvcvImageExportData(m_yuvImageIn, &yuvData), error, exit);
      auto uvInPlane = yuvData.buffer.strided.planes[1].basePtr;
      uint8_t* uvInPlaneHostPtr;
      CUPVA_CHECK_ERROR_GOTO(
          CupvaMemGetHostPointer((void**)&uvInPlaneHostPtr, uvInPlane), error, exit);
      assert(yuvData.buffer.strided.planes[1].rowStride == m_outputLinePitch);

      memcpy(uvOutPlaneHostPtr, uvInPlaneHostPtr, m_outputLinePitch * m_height / 2);
    }
    NVCV_CHECK_ERROR_GOTO(pvaConvertImageFormatSubmit(
                              m_convertYUVToRGBHandle, m_stream, m_yuvImageOut, m_rgbImageOut),
                          error,
                          exit);

    if (m_useCUDAInterop) {
      /// TODO: wrap output into NVCVImage (zero-copy)
    } else {
      // wait on cupva stream
      cupvaFence_t fence;
      CUPVA_CHECK_ERROR_GOTO(CupvaFenceInit(&fence, m_syncObj), error, exit);
      cupvaCmd_t rf;
      CUPVA_CHECK_ERROR_GOTO(CupvaCmdRequestFencesInit(&rf, &fence, 1), error, exit);
      cupvaCmd_t const* cmds[1] = {&rf};
      cupvaCmdStatus_t status[1] = {NULL};
      CUPVA_CHECK_ERROR_GOTO(
          CupvaStreamSubmit(m_stream, cmds, status, 1, CUPVA_IN_ORDER, -1, -1), error, exit);
      bool waitResult;
      CUPVA_CHECK_ERROR_GOTO(CupvaFenceWait(&fence, -1, &waitResult), error, exit);
      assert(waitResult);

      /// Copy RGB output image to output cuda buffer
      NVCVImageData rgbData;
      NVCV_CHECK_ERROR_GOTO(nvcvImageExportData(m_rgbImageOut, &rgbData), error, exit);
      auto outDevPtr = rgbData.buffer.strided.planes[0].basePtr;
      uint8_t* outHostPtr;
      CUPVA_CHECK_ERROR_GOTO(CupvaMemGetHostPointer((void**)&outHostPtr, outDevPtr), error, exit);
      assert(rgbData.buffer.strided.planes[0].rowStride == m_outputLinePitch);
      cudaMemcpyAsync(
          dst, outHostPtr, m_outputLinePitch * m_height, cudaMemcpyHostToDevice, stream);
    }
    return;

  exit:
    throw std::runtime_error("Fatal nvcv or cupva error in process()");
  };
  bool isInitialized() const { return m_initialized; }
  void deinit() {
    // cleanup stuff
    if (!m_useCUDAInterop) {
      nvcvImageDecRef(m_rgbImageIn, nullptr);
      nvcvImageDecRef(m_yuvImageIn, nullptr);
      nvcvTensorDecRef(m_yTensorIn, nullptr);
      nvcvTensorDecRef(m_yTensorOut, nullptr);
      nvcvImageDecRef(m_rgbImageOut, nullptr);
      nvcvImageDecRef(m_yuvImageOut, nullptr);
    }

    nvcvOperatorDestroy(m_convertRGBToYUVHandle);
    nvcvOperatorDestroy(m_gaussianFilterHandle);
    nvcvOperatorDestroy(m_convertYUVToRGBHandle);
    if (!m_useCUDAInterop) {
      CupvaStreamDestroy(m_stream);
      CupvaSyncObjDestroy(m_syncObj);
    }
  };

  ~PvaStuff() {
    if (m_initialized) {
      deinit();
    }
  };

 private:
  int32_t m_width;
  int32_t m_height;
  int32_t m_inputLinePitch;
  int32_t m_outputLinePitch;
  bool m_initialized = false;
  bool m_useCUDAInterop = false;

  cupvaStream_t m_stream;
  cupvaSyncObj_t m_syncObj;
  NVCVOperatorHandle m_convertRGBToYUVHandle;
  NVCVImageHandle m_rgbImageIn;
  NVCVImageHandle m_yuvImageIn;  // unfiltered

  NVCVOperatorHandle m_gaussianFilterHandle;
  NVCVTensorHandle m_yTensorIn;   // aliased with m_yuvImageIn if cuda interop enabled
  NVCVTensorHandle m_yTensorOut;  // aliased with m_yuvImageOut if cuda interop enabled

  NVCVOperatorHandle m_convertYUVToRGBHandle;
  NVCVImageHandle m_yuvImageOut;  // filtered
  NVCVImageHandle m_rgbImageOut;
};

namespace holoscan::ops {
class PVAVideoFilterExecutor : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PVAVideoFilterExecutor);
  PVAVideoFilterExecutor() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(allocator_, "allocator", "Allocator", "Allocator to allocate output tensor.");
    spec.input<gxf::Entity>("input");
    spec.output<gxf::Entity>("output");
  }
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto maybe_input_message = op_input.receive<gxf::Entity>("input");
    if (!maybe_input_message.has_value()) {
      HOLOSCAN_LOG_ERROR("Failed to receive input message gxf::Entity");
      return;
    }
    auto input_tensor = maybe_input_message.value().get<holoscan::Tensor>();
    if (!input_tensor) {
      HOLOSCAN_LOG_ERROR("Failed to receive holoscan::Tensor from input message gxf::Entity");
      return;
    }

    // get the CUDA stream with included wait on input
    cudaStream_t stream = op_input.receive_cuda_stream("input");

    // get handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), allocator_->gxf_cid());

    // cast Holoscan::Tensor to nvidia::gxf::Tensor to use its APIs directly
    nvidia::gxf::Tensor input_tensor_gxf{input_tensor->dl_ctx()};
    auto strides = nvidia::gxf::ComputeTrivialStrides(
        input_tensor_gxf.shape(),
        nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8));
    auto out_message = CreateTensorMap(context.context(),
                                       allocator.value(),
                                       {{"output",
                                         nvidia::gxf::MemoryStorageType::kDevice,
                                         input_tensor_gxf.shape(),
                                         nvidia::gxf::PrimitiveType::kUnsigned8,
                                         0,
                                         strides}},
                                       false);

    if (!out_message) {
      std::runtime_error("failed to create out_message");
    }
    const auto output_tensor = out_message.value().get<nvidia::gxf::Tensor>();
    if (!output_tensor) {
      std::runtime_error("failed to create out_tensor");
    }

    uint8_t* input_tensor_data = static_cast<uint8_t*>(input_tensor->data());
    uint8_t* output_tensor_data = static_cast<uint8_t*>(output_tensor.value()->pointer());
    if (output_tensor_data == nullptr) {
      throw std::runtime_error("Failed to allocate memory for the output image");
    }

    const int32_t imageWidth{static_cast<int32_t>(input_tensor->shape()[1])};
    const int32_t imageHeight{static_cast<int32_t>(input_tensor->shape()[0])};
    const int32_t inputLinePitch{static_cast<int32_t>(strides[0])};
    const int32_t outputLinePitch{static_cast<int32_t>(strides[0])};

    if (!pvaOperatorTask_.isInitialized()) {
      pvaOperatorTask_.init(imageWidth, imageHeight, inputLinePitch, outputLinePitch);
    }
    pvaOperatorTask_.process(input_tensor_data, output_tensor_data, stream);
    auto result = gxf::Entity(std::move(out_message.value()));

    op_output.emit(result, "output");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  PvaStuff pvaOperatorTask_;
};
}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    uint32_t max_width{1920};
    uint32_t max_height{1080};
    int64_t source_block_size = max_width * max_height * 3;

    std::shared_ptr<BlockMemoryPool> pva_allocator =
        make_resource<BlockMemoryPool>("allocator", 1, source_block_size, 1);

    auto pva_video_filter = make_operator<ops::PVAVideoFilterExecutor>(
        "pva_video_filter", Arg("allocator") = pva_allocator);

    auto source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"));

    auto recorder = make_operator<ops::VideoStreamRecorderOp>("recorder", from_config("recorder"));
    auto visualizer1 = make_operator<ops::HolovizOp>(
        "holoviz1", from_config("holoviz"), Arg("window_title") = std::string("Original Stream"));
    auto visualizer2 =
        make_operator<ops::HolovizOp>("holoviz2",
                                      from_config("holoviz"),
                                      Arg("window_title") = std::string("Image Sharpened Stream"));

    add_flow(source, pva_video_filter);
    add_flow(source, visualizer1, {{"output", "receivers"}});
    // add_flow(pva_video_filter, recorder);
    add_flow(pva_video_filter, visualizer2, {{"output", "receivers"}});
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/main.yaml";
  app->config(config_path);

  app->run();

  return 0;
}
