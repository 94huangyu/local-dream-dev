#ifndef QNNMODEL_HPP
#define QNNMODEL_HPP

#include <HTP/QnnHtpContext.h>
#include <HTP/QnnHtpDevice.h>
#include <dlfcn.h>
#include <inttypes.h>

#include <Config.hpp>
#include <QnnSampleApp.hpp>
#include <QnnTypeMacros.hpp>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "DataUtil.hpp"
#include "Logger.hpp"
#include "SDUtils.hpp"

using namespace qnn::tools::sample_app;

class QnnModel : public QnnSampleApp {
 public:
  // Generic named-IO contract used by new QNN formats. The caller owns input
  // memory for the duration of executeNamedGraph; outputs are copied out so a
  // later graph invocation cannot overwrite them. This deliberately avoids
  // assuming Anima's sample/context/hidden tensor names.
  struct NamedTensorView {
    std::string name;
    const void *data = nullptr;
    size_t byte_count = 0;
  };
  struct NamedTensorValue {
    std::string name;
    std::vector<uint8_t> bytes;
  };
  // H5（P2 生图速度）：把一次段执行的墙钟拆成「拷入 / graphExecute / 拷出」。
  // 目的是回答"122.7 s 步循环里有多少不是算力"——D1 只能从加速器周期反推出
  // 「part1a 41.2% 非算力」，那是②推导；这里是①实测。默认 nullptr ⇒ 其它
  // pipeline（Anima/SDXL/SD15）的三参数调用完全不受影响。
  struct ExecSplit {
    long long in_us = 0, exec_us = 0, out_us = 0;
  };

  Qnn_Tensor_t *inputs = nullptr;
  Qnn_Tensor_t *outputs = nullptr;
  void *m_modelHandle = nullptr;
  bool anima_io_logged_ = false;
  QnnModel(QnnFunctionPointers qnnFunctionPointers, std::string inputListPaths,
           std::string opPackagePaths, void *backendHandle,
           std::string outputPath = s_defaultOutputPath, bool debug = false,
           qnn::tools::iotensor::OutputDataType outputDataType =
               qnn::tools::iotensor::OutputDataType::FLOAT_ONLY,
           qnn::tools::iotensor::InputDataType inputDataType =
               qnn::tools::iotensor::InputDataType::FLOAT,
           ProfilingLevel profilingLevel = ProfilingLevel::OFF,
           bool dumpOutputs = false, std::string cachedBinaryPath = "",
           std::string saveBinaryName = "")
      : QnnSampleApp(qnnFunctionPointers, inputListPaths, opPackagePaths,
                     backendHandle, outputPath, debug, outputDataType,
                     inputDataType, profilingLevel, dumpOutputs,
                     cachedBinaryPath, saveBinaryName) {}

  // Executes graph 0 only when every declared input is supplied by name and
  // has the exact client-buffer byte size. This turns a stale ONNX/QNN
  // manifest into an immediate actionable error instead of an unsafe memcpy.
  StatusCode executeNamedGraph(const char *tag,
                               const std::vector<NamedTensorView> &named_inputs,
                               std::vector<NamedTensorValue> &named_outputs,
                               ExecSplit *split = nullptr) {
    const auto t_us = [] {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
          .count();
    };
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    const auto graphInfo = (*m_graphsInfo)[m_activeGraphIdx];

    if (named_inputs.size() != graphInfo.numInputTensors) {
      QNN_ERROR("%s: graph expects %u inputs, received %zu", tag,
                graphInfo.numInputTensors, named_inputs.size());
      return StatusCode::FAILURE;
    }

    const long long t_in0 = split ? t_us() : 0;
    for (uint32_t i = 0; i < graphInfo.numInputTensors; ++i) {
      Qnn_Tensor_t &tensor = inputs[i];
      const char *tensor_name = QNN_TENSOR_GET_NAME(tensor);
      const NamedTensorView *provided = nullptr;
      for (const auto &candidate : named_inputs) {
        if (tensor_name && candidate.name == tensor_name) {
          provided = &candidate;
          break;
        }
      }
      if (!provided || !provided->data) {
        QNN_ERROR("%s: missing input tensor '%s'", tag,
                  tensor_name ? tensor_name : "<unnamed>");
        return StatusCode::FAILURE;
      }
      const size_t expected = QNN_TENSOR_GET_CLIENT_BUF(tensor).dataSize;
      if (provided->byte_count != expected) {
        QNN_ERROR("%s: input '%s' has %zu bytes, expected %zu", tag,
                  tensor_name, provided->byte_count, expected);
        return StatusCode::FAILURE;
      }
      memcpy(QNN_TENSOR_GET_CLIENT_BUF(tensor).data, provided->data, expected);
    }

    const long long t_exec0 = split ? t_us() : 0;
    if (!runGraph(graphInfo, tag)) return StatusCode::FAILURE;
    const long long t_out0 = split ? t_us() : 0;

    named_outputs.clear();
    named_outputs.reserve(graphInfo.numOutputTensors);
    for (uint32_t i = 0; i < graphInfo.numOutputTensors; ++i) {
      const Qnn_Tensor_t &tensor = outputs[i];
      const size_t bytes = QNN_TENSOR_GET_CLIENT_BUF(tensor).dataSize;
      NamedTensorValue value;
      value.name = QNN_TENSOR_GET_NAME(tensor);
      value.bytes.resize(bytes);
      memcpy(value.bytes.data(), QNN_TENSOR_GET_CLIENT_BUF(tensor).data, bytes);
      named_outputs.push_back(std::move(value));
    }
    if (split) {
      split->in_us += t_exec0 - t_in0;
      split->exec_us += t_out0 - t_exec0;
      split->out_us += t_us() - t_out0;
    }
    return StatusCode::SUCCESS;
  }

  // --- HTP spill-fill buffer sharing across contexts (group registration) ---
  // When enabled, this context joins a group that shares a single HTP
  // spill-fill scratch buffer instead of each context allocating its own.
  // The first model in the group passes firstHandle == nullptr (it becomes the
  // group head); every subsequent model passes the head's context handle
  // (getContextHandle()). `maxBytes` must be >= the largest spill-fill
  // requirement among all models in the group; only the head's value is
  // honored by the backend. Must be called BEFORE initialize()/createFromBinary
  // so the config is in place when the context is created.
  // NOTE: old (pre-2.35) context binaries report spillFillBufferSize == 0 in
  // their metadata, so the size cannot be read back from them — it must be
  // supplied here explicitly.
  void setSpillFillGroup(uint64_t maxBytes, Qnn_ContextHandle_t firstHandle) {
    if (maxBytes == 0) return;  // disabled: leave m_contextConfig untouched
    m_sfHtpConfig.option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    m_sfHtpConfig.groupRegistration.firstGroupHandle = firstHandle;
    m_sfHtpConfig.groupRegistration.maxSpillFillBuffer = maxBytes;
    m_sfCtxConfig.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    m_sfCtxConfig.customConfig = &m_sfHtpConfig;
    m_sfConfigActive = true;
    m_sfCtxConfigPtrs[0] = &m_sfCtxConfig;
    m_sfCtxConfigPtrs[1] = nullptr;
    // Consumed by QnnSampleApp::createFromBinary / QnnModel::createFromBuffer.
    // (2026-09-02) assembled together with ENABLE_GRAPHS, see rebuildContextConfig().
    rebuildContextConfig();
  }

  // --- Multi-graph context: enable & select exactly one graph (ledger #86) ---
  // A context binary may hold several graphs that share weights (e.g. one per
  // aspect ratio). Two facts, both measured on device 2026-09-02:
  //   1. Without QNN_CONTEXT_CONFIG_ENABLE_GRAPHS the whole binary is
  //      deserialized and createFromBinary fails with 0x3ea (unsigned-PD
  //      capacity, ledger #57) -- so this is REQUIRED, not an optimization.
  //   2. graphRetrieve() returns QNN_SUCCESS even for graphs that were NOT
  //      enabled, so its return code must never be used to decide usability.
  //      The graph is instead selected BY NAME below.
  // Must be called BEFORE initialize()/createFromBinary().
  // When never called, m_activeGraphIdx stays 0 and behaviour is unchanged.
  void setEnabledGraph(const char *graphName) {
    if (graphName == nullptr || *graphName == '\0') return;
    m_enabledGraphName = graphName;
    m_egNames[0] = m_enabledGraphName.c_str();
    m_egNames[1] = nullptr;
    m_egCtxConfig.option = QNN_CONTEXT_CONFIG_ENABLE_GRAPHS;
    m_egCtxConfig.enableGraphs = m_egNames;
    m_egConfigActive = true;
    rebuildContextConfig();
  }

  // Assembles m_contextConfig from whichever optional configs are active.
  //
  // 🔴🔴 2026-09-04 事故根因：这里原本用 `option` 的值判断「这个配置设过了没有」——
  //   `if (m_sfCtxConfig.option == QNN_CONTEXT_CONFIG_OPTION_CUSTOM)`。
  //   而 QnnContext.h:99 里 **`QNN_CONTEXT_CONFIG_OPTION_CUSTOM == 0`**，
  //   `QnnContext_Config_t m_sfCtxConfig{}` 又把 option 零初始化 ⇒ **该判断恒为真**。
  //   于是只要 setEnabledGraph() 被调用过，就会把一个 `customConfig == nullptr` 的
  //   spill-fill 配置一起交给 contextCreateFromBinary ⇒ HTP 解引用空指针 ⇒
  //   设备上 SIGSEGV（tombstone: SEGV_MAPERR, fault addr 0x0, #00~#03 在 libQnnHtp.so）。
  //   现网 1:1 也受影响，因为交付契约里四个 transformer 段本来就带 qnn_graph_name。
  // ⇒ **「设过没有」必须用独立的布尔位表示，不得借用带 0 值的枚举当哨兵。**
  //   （约束 8：判据里涉及具体数值编码的，定稿前必须先验证该编码本身成立。）
  void rebuildContextConfig() {
    size_t k = 0;
    if (m_sfConfigActive) m_ctxCfgPtrs[k++] = &m_sfCtxConfig;
    if (m_egConfigActive) m_ctxCfgPtrs[k++] = &m_egCtxConfig;
    m_ctxCfgPtrs[k] = nullptr;
    m_contextConfig = (k > 0) ? m_ctxCfgPtrs : nullptr;
  }

  // Valid only after a successful initialize()/createFromBinary(): the QNN
  // context handle, used as the group head reference for the other models.
  Qnn_ContextHandle_t getContextHandle() const { return m_context; }

  // Queries the HTP backend for this context's real spill-fill scratch
  // requirement (bytes). Valid only after the context is created. Returns 0 if
  // the query is unavailable/fails. Use this to discover the right value for
  // setSpillFillGroup() instead of guessing — note the spillFillBufferSize in
  // the binary metadata is 0 for pre-2.35 binaries, but this runtime query
  // returns the true requirement.
  uint64_t querySpillFillSize() const {
    if (m_context == nullptr ||
        m_qnnFunctionPointers.qnnInterface.contextGetProperty == nullptr)
      return 0;
    QnnHtpContext_CustomProperty_t custom{};
    custom.option = QNN_HTP_CONTEXT_GET_PROP_MAX_SPILLFILL_BUFFER_SIZE;
    QnnContext_Property_t prop{};
    prop.option = QNN_CONTEXT_PROPERTY_OPTION_CUSTOM;
    prop.customProperty = &custom;
    QnnContext_Property_t *props[2] = {&prop, nullptr};
    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnInterface.contextGetProperty(m_context, props))
      return 0;
    return custom.spillfillBufferSize;
  }

  ~QnnModel() {
    // Tear down per-graph input/output tensors first (allocated lazily by
    // setupInputAndOutputTensors). Must run before freeContext() since it
    // relies on graphsInfo for tensor counts.
    if ((inputs != nullptr || outputs != nullptr) && m_graphsInfo != nullptr &&
        m_graphsCount > 0) {
      m_ioTensor.tearDownInputAndOutputTensors(
          inputs, outputs, (*m_graphsInfo)[m_activeGraphIdx].numInputTensors,
          (*m_graphsInfo)[m_activeGraphIdx].numOutputTensors);
    }
    inputs = nullptr;
    outputs = nullptr;

    // freeContext() is not idempotent — only call when graphs are still alive.
    if (m_graphsInfo != nullptr) {
      freeContext();
    }
    freeDevice();
    terminateBackend();
    if (m_modelHandle != nullptr) {
      dlclose(m_modelHandle);
      m_modelHandle = nullptr;
    }
  }

  StatusCode enablePerformaceMode() {
    uint32_t powerConfigId;
    uint32_t deviceId = 0;
    uint32_t coreId = 0;
    auto qnnInterface = m_qnnFunctionPointers.qnnInterface;

    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr =
        qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
      QNN_ERROR("device error");
      return StatusCode::FAILURE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra =
        static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;
    Qnn_ErrorHandle_t perfInfraErr =
        perfInfra.createPowerConfigId(deviceId, coreId, &powerConfigId);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("createPowerConfigId failed");
      return StatusCode::FAILURE;
    }
    QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency;
    memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
    rpcControlLatency.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcControlLatency.rpcControlLatencyConfig = 100;
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs1[] = {
        &rpcControlLatency, NULL};
    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs1);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("setPowerConfig failed");
      return StatusCode::FAILURE;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime;
    memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
    rpcPollingTime.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPollingTime.rpcPollingTimeConfig = 9999;
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs2[] = {
        &rpcPollingTime, NULL};
    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs2);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("setPowerConfig failed");
      return StatusCode::FAILURE;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
    memset(&powerConfig, 0, sizeof(powerConfig));
    powerConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    powerConfig.dcvsV3Config.dcvsEnable = 0;
    powerConfig.dcvsV3Config.setDcvsEnable = 1;
    powerConfig.dcvsV3Config.contextId = powerConfigId;
    powerConfig.dcvsV3Config.powerMode =
        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    powerConfig.dcvsV3Config.setSleepLatency = 1;
    powerConfig.dcvsV3Config.setBusParams = 1;
    powerConfig.dcvsV3Config.setCoreParams = 1;
    powerConfig.dcvsV3Config.sleepDisable = 1;
    powerConfig.dcvsV3Config.setSleepDisable = 1;
    powerConfig.dcvsV3Config.sleepLatency = 40;
    powerConfig.dcvsV3Config.busVoltageCornerMin =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerTarget =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerMax =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerMin =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerTarget =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerMax =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs3[] = {
        &powerConfig, NULL};
    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs3);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("setPowerConfig failed");
      return StatusCode::FAILURE;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t adaptivePollingTime;
    memset(&adaptivePollingTime, 0, sizeof(adaptivePollingTime));
    adaptivePollingTime.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_ADAPTIVE_POLLING_TIME;
    adaptivePollingTime.adaptivePollingTimeConfig = 1000;
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs4[] = {
        &adaptivePollingTime, NULL};
    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs4);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("setPowerConfig failed");
      return StatusCode::FAILURE;
    }

    return StatusCode::SUCCESS;
  }

  StatusCode executeUnetGraphs(float *latents, int timestep,
                               float *text_embedding, float *latents_pred) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting unet execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 3) {
      QNN_ERROR("Expecting 3 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // latents
    {
      uint16_t *latents_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      int elementCount = 1 * 4 * sample_width * sample_height;
      qnn::tools::datautil::floatToTfN(
          latents_uint16, latents,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
    }

    // position/timestep
    {
      int32_t *positionData =
          static_cast<int32_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[1]).data);
      positionData[0] = timestep;
    }

    // text_embedding
    {
      uint16_t *text_embedding_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[2]).data);
      int elementCount = 1 * 77 * text_embedding_size;
      qnn::tools::datautil::floatToTfN(
          text_embedding_uint16, text_embedding,
          inputs[2].v1.quantizeParams.scaleOffsetEncoding.offset,
          inputs[2].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
    }

    // execute graph
    QNN_DEBUG("Executing unet graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("unet graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("unet graph execution failed!");
    }

    // get output
    if (StatusCode::SUCCESS == returnStatus) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloatInto(latents_pred, &outputs[0])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }

    return returnStatus;
  }

  StatusCode executeVaeEncoderGraphs(float *pixel_values, float *mean,
                                     float *std) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting vae encoder execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // pixel_values
    {
      uint16_t *pixel_values_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      int elementCount = 1 * 3 * output_width * output_height;
      qnn::tools::datautil::floatToTfN(
          pixel_values_uint16, pixel_values,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
    }

    // execute graph
    QNN_DEBUG("Executing vae encoder graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("vae encoder graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("vae encoder graph execution failed!");
    }

    // get output
    if (StatusCode::SUCCESS == returnStatus) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloatInto(mean, &outputs[0])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloatInto(std, &outputs[1])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    return returnStatus;
  }

  StatusCode executeVaeDecoderGraphs(float *latents, float *pixel_values) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting vae decoder execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // latents
    {
      uint16_t *latents_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      int elementCount = 1 * 4 * sample_width * sample_height;
      qnn::tools::datautil::floatToTfN(
          latents_uint16, latents,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
    }

    // execute graph
    QNN_DEBUG("Executing vae decoder graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("vae decoder graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("vae decoder graph execution failed!");
    }

    // get output
    if (StatusCode::SUCCESS == returnStatus) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloatInto(pixel_values, &outputs[0])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    return returnStatus;
  }

  StatusCode executeUnetGraphsSDXL(float *sample, int timestep,
                                   float *encoder_hidden_states,
                                   float *text_embeds, float *time_ids,
                                   float *out_sample) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting sdxl unet execution for graphIdx: %d", graphIdx);

    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 5) {
      QNN_ERROR("Expecting 5 input tensors for sdxl unet, got %d",
                graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // sample (fp32, 1x4xHxW)
    {
      int elementCount = 1 * 4 * sample_width * sample_height;
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
             sample, elementCount * sizeof(float));
    }

    // timestep (int32, 1)
    {
      int32_t *ts =
          static_cast<int32_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[2]).data);
      ts[0] = timestep;
    }

    // encoder_hidden_states (fp32, 1x77x2048)
    {
      int elementCount = 1 * 77 * (text_embedding_size + text_embedding_size_2);
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[1]).data),
             encoder_hidden_states, elementCount * sizeof(float));
    }

    // text_embeds (fp32, 1x1280)
    {
      int elementCount = 1 * text_embedding_size_2;
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[4]).data),
             text_embeds, elementCount * sizeof(float));
    }

    // time_ids (fp32, 1x6)
    {
      int elementCount = 1 * 6;
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[3]).data),
             time_ids, elementCount * sizeof(float));
    }

    QNN_DEBUG("Executing sdxl unet graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("sdxl unet graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("sdxl unet graph execution failed!");
      return returnStatus;
    }

    // out_sample (fp32, 1x4xHxW)
    int elementCount = 1 * 4 * sample_width * sample_height;
    memcpy(out_sample,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           elementCount * sizeof(float));

    return returnStatus;
  }

  StatusCode executeVaeEncoderGraphsSDXL(float *pixel_values, float *mean,
                                         float *std) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting sdxl vae encoder execution for graphIdx: %d", graphIdx);

    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensor for sdxl vae encoder, got %d",
                graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // pixel_values (fp32, 1x3xHxW)
    {
      int elementCount = 1 * 3 * output_width * output_height;
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
             pixel_values, elementCount * sizeof(float));
    }

    QNN_DEBUG("Executing sdxl vae encoder graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("sdxl vae encoder graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("sdxl vae encoder graph execution failed!");
      return returnStatus;
    }

    int elementCount = 1 * 4 * sample_width * sample_height;
    memcpy(mean,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           elementCount * sizeof(float));
    memcpy(std,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[1]).data),
           elementCount * sizeof(float));

    return returnStatus;
  }

  StatusCode executeVaeDecoderGraphsSDXL(float *latents, float *pixel_values) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting sdxl vae decoder execution for graphIdx: %d", graphIdx);

    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensor for sdxl vae decoder, got %d",
                graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // latents (fp32, 1x4xHxW)
    {
      int elementCount = 1 * 4 * sample_width * sample_height;
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
             latents, elementCount * sizeof(float));
    }

    QNN_DEBUG("Executing sdxl vae decoder graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("sdxl vae decoder graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("sdxl vae decoder graph execution failed!");
      return returnStatus;
    }

    int elementCount = 1 * 3 * output_width * output_height;
    memcpy(pixel_values,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           elementCount * sizeof(float));

    return returnStatus;
  }

  // ---- shared helpers for the Anima execute paths --------------------------
  bool ensureIoTensors() {
    if (inputs != nullptr && outputs != nullptr) return true;
    if (qnn::tools::iotensor::StatusCode::SUCCESS !=
        m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                              (*m_graphsInfo)[m_activeGraphIdx])) {
      QNN_ERROR("Error setting up Input/Output tensors");
      return false;
    }
    return true;
  }

  static size_t tensorElems(const Qnn_Tensor_t &t) {
    uint32_t rank = QNN_TENSOR_GET_RANK(t);
    uint32_t *dims = QNN_TENSOR_GET_DIMENSIONS(t);
    size_t n = 1;
    for (uint32_t i = 0; i < rank; ++i) n *= (dims ? dims[i] : 1);
    return n;
  }

  static Qnn_Tensor_t *findTensor(Qnn_Tensor_t *arr, uint32_t count,
                                  const char *name) {
    for (uint32_t i = 0; i < count; ++i) {
      const char *tn = QNN_TENSOR_GET_NAME(arr[i]);
      if (tn && std::strcmp(tn, name) == 0) return &arr[i];
    }
    return nullptr;
  }

  bool writeNamedFloat(const qnn_wrapper_api::GraphInfo_t &graphInfo,
                       const char *name, const float *src, size_t elems) {
    Qnn_Tensor_t *t = findTensor(inputs, graphInfo.numInputTensors, name);
    if (!t) {
      QNN_ERROR("anima: missing input tensor '%s'", name);
      return false;
    }
    memcpy(QNN_TENSOR_GET_CLIENT_BUF(*t).data, src, elems * sizeof(float));
    return true;
  }

  bool runGraph(const qnn_wrapper_api::GraphInfo_t &graphInfo,
                const char *tag) {
    auto start = std::chrono::high_resolution_clock::now();
    auto st = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);
    auto end = std::chrono::high_resolution_clock::now();
    QNN_INFO(
        "%s graph execution time: %d ms", tag,
        (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count());
    if (QNN_GRAPH_NO_ERROR != st) {
      QNN_ERROR("%s graph execution failed!", tag);
      return false;
    }
    return true;
  }

  // One-time dump of a graph's IO names+sizes, so the on-device tensor naming
  // can be verified against export_onnx_anima.py's input/output_names.
  void logAnimaIoOnce(const char *tag,
                      const qnn_wrapper_api::GraphInfo_t &graphInfo) {
    if (anima_io_logged_) return;
    anima_io_logged_ = true;
    for (uint32_t i = 0; i < graphInfo.numInputTensors; ++i)
      QNN_INFO("[anima %s] in[%u] name=%s elems=%zu", tag, i,
               QNN_TENSOR_GET_NAME(inputs[i]), tensorElems(inputs[i]));
    for (uint32_t i = 0; i < graphInfo.numOutputTensors; ++i)
      QNN_INFO("[anima %s] out[%u] name=%s elems=%zu", tag, i,
               QNN_TENSOR_GET_NAME(outputs[i]), tensorElems(outputs[i]));
  }

  // ---- Anima (split DiT + 16-ch Wan VAE) -----------------------------------
  // Anima graphs are converted with --preserve_io_datatype, so every IO tensor
  // is fp32 (direct memcpy, like the SDXL path). The split DiT crosses only TWO
  // named intermediates between part-1 and part-2 (hidden, emb); part-2
  // recomputes adaln + rope internally from `timestamp` (and the static grid),
  // and the app re-supplies `context`. Passing adaln/rope as flat graph inputs
  // used to force part-2's residual stream into a slow layout (~+200M
  // cycles/block); computing them inside removes it. We bind every tensor by
  // its (ONNX) name. Matches export_onnx_anima_scaled.py PART1_OUT_NAMES.
  static constexpr const char *kAnimaPart1OutNames[2] = {"hidden", "emb"};

  // part-1: (sample, timestamp, encoder_hidden_states) -> (hidden, emb).
  StatusCode executeAnimaUnetPart1(float *sample, float timestep,
                                   float *context,
                                   std::vector<std::vector<float>> &out2) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logAnimaIoOnce("unet_part1", graphInfo);

    const int latent_elems =
        1 * anima_latent_channels * sample_width * sample_height;
    const int ctx_elems = 1 * anima_text_seq_len * anima_text_embedding_size;
    if (!writeNamedFloat(graphInfo, "sample", sample, latent_elems) ||
        !writeNamedFloat(graphInfo, "timestamp", &timestep, 1) ||
        !writeNamedFloat(graphInfo, "encoder_hidden_states", context,
                         ctx_elems))
      return StatusCode::FAILURE;

    if (!runGraph(graphInfo, "anima unet part1")) return StatusCode::FAILURE;

    out2.resize(2);
    for (int k = 0; k < 2; ++k) {
      Qnn_Tensor_t *t = findTensor(outputs, graphInfo.numOutputTensors,
                                   kAnimaPart1OutNames[k]);
      if (!t) {
        QNN_ERROR("anima part1: missing output '%s'", kAnimaPart1OutNames[k]);
        return StatusCode::FAILURE;
      }
      size_t n = tensorElems(*t);
      out2[k].resize(n);
      memcpy(out2[k].data(), QNN_TENSOR_GET_CLIENT_BUF(*t).data,
             n * sizeof(float));
    }
    return StatusCode::SUCCESS;
  }

  // part-2: (hidden, emb) + timestamp + context -> out_sample (1x16xHxW).
  // adaln + rope are recomputed inside the graph from `timestamp`/grid.
  StatusCode executeAnimaUnetPart2(const std::vector<std::vector<float>> &in2,
                                   float timestep, float *context,
                                   float *out_sample) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logAnimaIoOnce("unet_part2", graphInfo);

    const int ctx_elems = 1 * anima_text_seq_len * anima_text_embedding_size;
    if (!writeNamedFloat(graphInfo, "hidden", in2[0].data(),
                         (int)in2[0].size()) ||
        !writeNamedFloat(graphInfo, "emb", in2[1].data(), (int)in2[1].size()) ||
        !writeNamedFloat(graphInfo, "timestamp", &timestep, 1) ||
        !writeNamedFloat(graphInfo, "context", context, ctx_elems))
      return StatusCode::FAILURE;

    if (!runGraph(graphInfo, "anima unet part2")) return StatusCode::FAILURE;

    const int latent_elems =
        1 * anima_latent_channels * sample_width * sample_height;
    memcpy(out_sample,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           latent_elems * sizeof(float));
    return StatusCode::SUCCESS;
  }

  // 16-ch latent -> 3-ch pixels. Same shape contract as the SDXL decoder but
  // with 16 input channels.
  StatusCode executeAnimaVaeDecoder(float *latents, float *pixel_values) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logAnimaIoOnce("vae_decoder", graphInfo);

    const int latent_elems =
        1 * anima_latent_channels * sample_width * sample_height;
    memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
           latents, latent_elems * sizeof(float));

    if (!runGraph(graphInfo, "anima vae decoder")) return StatusCode::FAILURE;

    const int pixel_elems = 1 * 3 * output_width * output_height;
    memcpy(pixel_values,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           pixel_elems * sizeof(float));
    return StatusCode::SUCCESS;
  }

  // 3-ch pixels (1x3xHxW, -1..1) -> 16-ch latent distribution (mean, std).
  // Same single-input / two-output contract as the SDXL VAE encoder, but with
  // 16 latent channels. Used for Anima img2img / inpaint / aspect-pad.
  StatusCode executeAnimaVaeEncoder(float *pixel_values, float *mean,
                                    float *std_dev) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logAnimaIoOnce("vae_encoder", graphInfo);

    const int pixel_elems = 1 * 3 * output_width * output_height;
    memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
           pixel_values, pixel_elems * sizeof(float));

    if (!runGraph(graphInfo, "anima vae encoder")) return StatusCode::FAILURE;

    const int latent_elems =
        1 * anima_latent_channels * sample_width * sample_height;
    memcpy(mean,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           latent_elems * sizeof(float));
    memcpy(std_dev,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[1]).data),
           latent_elems * sizeof(float));
    return StatusCode::SUCCESS;
  }

  // Merged Qwen3 + LLM-adapter text encoder. Inputs: input_embedding
  // [1,QWEN_SEQ,1024] fp32, t5_ids [1,T5_SEQ] int32, t5_mask [1,T5_SEQ] fp32,
  // qwen_mask [1,QWEN_SEQ] fp32. Output: context [1,T5_SEQ,1024] fp32. (t5_ids
  // is declared int32 in the graph thanks to --preserve_io_datatype, and the
  // C++ QNN API takes a real int32 buffer — unlike qnn-net-run which wants
  // float32 raw.)
  StatusCode executeAnimaClip(const float *input_embedding,
                              const int32_t *t5_ids, const float *t5_mask,
                              const float *qwen_mask, float *out_context) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logAnimaIoOnce("clip", graphInfo);

    const int QS = anima_qwen_seq_len;
    const int TS = anima_text_seq_len;
    const int D = anima_text_embedding_size;
    if (!writeNamedFloat(graphInfo, "input_embedding", input_embedding,
                         (size_t)QS * D) ||
        !writeNamedFloat(graphInfo, "t5_mask", t5_mask, TS) ||
        !writeNamedFloat(graphInfo, "qwen_mask", qwen_mask, QS))
      return StatusCode::FAILURE;
    // t5_ids is int32 (not float) — bind it directly.
    Qnn_Tensor_t *ti = findTensor(inputs, graphInfo.numInputTensors, "t5_ids");
    if (!ti) {
      QNN_ERROR("anima clip: missing input 't5_ids'");
      return StatusCode::FAILURE;
    }
    memcpy(QNN_TENSOR_GET_CLIENT_BUF(*ti).data, t5_ids,
           (size_t)TS * sizeof(int32_t));

    if (!runGraph(graphInfo, "anima clip")) return StatusCode::FAILURE;

    Qnn_Tensor_t *to =
        findTensor(outputs, graphInfo.numOutputTensors, "context");
    if (!to) {
      QNN_ERROR("anima clip: missing output 'context'");
      return StatusCode::FAILURE;
    }
    memcpy(out_context, QNN_TENSOR_GET_CLIENT_BUF(*to).data,
           (size_t)TS * D * sizeof(float));
    return StatusCode::SUCCESS;
  }

  StatusCode executeUpscalerGraphs(float *input_image, float *output_image) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting upscaler execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // input_image (quantized to uint8, 1x3x192x192)
    {
      // uint8_t *input_uint8 =
      //     static_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      // int elementCount = 1 * 3 * 192 * 192;
      // qnn::tools::datautil::floatToTfN(
      //     input_uint8, input_image,
      //     inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
      //     inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale,
      //     elementCount);
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
             input_image, 1 * 3 * 192 * 192 * sizeof(float));
    }

    // execute graph
    QNN_DEBUG("Executing upscaler graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("upscaler graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("upscaler graph execution failed!");
    }

    // get output
    // if (StatusCode::SUCCESS == returnStatus) {
    //   float *tmp = nullptr;
    //   int elementCount = 1 * 3 * 768 * 768;
    //   if (qnn::tools::iotensor::StatusCode::SUCCESS !=
    //       m_ioTensor.convertToFloat(&tmp, &outputs[0])) {
    //     returnStatus = StatusCode::FAILURE;
    //     return returnStatus;
    //   }
    //   memcpy(output_image, tmp, elementCount * sizeof(float));
    //   free(tmp);
    // }
    memcpy(output_image,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           1 * 3 * 768 * 768 * sizeof(float));
    return returnStatus;
  }

  StatusCode createFromBuffer(const uint8_t *buffer, uint64_t bufferSize) {
    if (nullptr == buffer || 0 == bufferSize) {
      QNN_ERROR("Invalid buffer provided. Buffer is null or size is 0.");
      return StatusCode::FAILURE;
    }

    if (nullptr ==
            m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface
                       .systemContextGetBinaryInfo ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
      QNN_ERROR("QNN System function pointers are not populated.");
      return StatusCode::FAILURE;
    }

    auto returnStatus = StatusCode::SUCCESS;
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};

    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(
            &sysCtxHandle)) {
      QNN_ERROR("Could not create system handle.");
      returnStatus = StatusCode::FAILURE;
    }

    const QnnSystemContext_BinaryInfo_t *binaryInfo{nullptr};
    Qnn_ContextBinarySize_t binaryInfoSize{0};

    void *nonConstBuffer =
        const_cast<void *>(static_cast<const void *>(buffer));

    if (StatusCode::SUCCESS == returnStatus &&
        QNN_SUCCESS !=
            m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                sysCtxHandle, nonConstBuffer, bufferSize, &binaryInfo,
                &binaryInfoSize)) {
      QNN_ERROR("Failed to get context binary info");
      returnStatus = StatusCode::FAILURE;
    }

    if (StatusCode::SUCCESS == returnStatus &&
        !copyMetadataToGraphsInfo(binaryInfo, m_graphsInfo, m_graphsCount)) {
      QNN_ERROR("Failed to copy metadata.");
      returnStatus = StatusCode::FAILURE;
    }

    m_qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    // Resolve the enabled graph name to an index. Doing it by name is the only
    // safe way: graphRetrieve() succeeds even for graphs that were not
    // deserialized (measured on device, ledger #86), so an index-based guess
    // would silently execute a graph that is not resident.
    if (StatusCode::SUCCESS == returnStatus && !m_enabledGraphName.empty()) {
      bool found = false;
      for (size_t i = 0; i < m_graphsCount; ++i) {
        const char *gn = (*m_graphsInfo)[i].graphName;
        if (gn != nullptr && m_enabledGraphName == gn) {
          m_activeGraphIdx = i;
          found = true;
          break;
        }
      }
      if (!found) {
        QNN_ERROR("Enabled graph '%s' is not present in this context binary",
                  m_enabledGraphName.c_str());
        returnStatus = StatusCode::FAILURE;
      }
    }

    if (StatusCode::SUCCESS == returnStatus &&
        nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
      QNN_ERROR("contextCreateFromBinaryFnHandle is nullptr.");
      returnStatus = StatusCode::FAILURE;
    }

    if (StatusCode::SUCCESS == returnStatus &&
        m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
            m_backendHandle, m_deviceHandle,
            (const QnnContext_Config_t **)m_contextConfig, nonConstBuffer,
            bufferSize, &m_context, m_profileBackendHandle)) {
      QNN_ERROR("Could not create context from binary.");
      returnStatus = StatusCode::FAILURE;
    }

    if (ProfilingLevel::OFF != m_profilingLevel) {
      extractBackendProfilingInfo(m_profileBackendHandle, nullptr);
    }

    m_isContextCreated = true;

    if (StatusCode::SUCCESS == returnStatus) {
      for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
        if (nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
          QNN_ERROR("graphRetrieveFnHandle is nullptr.");
          returnStatus = StatusCode::FAILURE;
          break;
        }
        if (QNN_SUCCESS != m_qnnFunctionPointers.qnnInterface.graphRetrieve(
                               m_context, (*m_graphsInfo)[graphIdx].graphName,
                               &((*m_graphsInfo)[graphIdx].graph))) {
          QNN_ERROR("Unable to retrieve graph handle for graph Idx: %d",
                    graphIdx);
          returnStatus = StatusCode::FAILURE;
        }
      }
    }

    if (StatusCode::SUCCESS != returnStatus) {
      QNN_DEBUG("Cleaning up graph Info structures.");
      qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    }

    return returnStatus;
  }

 private:
  // Backing storage for the spill-fill group-registration context config.
  // These must outlive the contextCreateFromBinary call, so they live as
  // members rather than locals in setSpillFillGroup().
  QnnHtpContext_CustomConfig_t m_sfHtpConfig{};
  QnnContext_Config_t m_sfCtxConfig{};
  QnnContext_Config_t *m_sfCtxConfigPtrs[2] = {nullptr, nullptr};
  // --- multi-graph selection (ledger #86). Must outlive createFromBinary. ---
  std::string m_enabledGraphName;
  const char *m_egNames[2] = {nullptr, nullptr};
  QnnContext_Config_t m_egCtxConfig{};
  QnnContext_Config_t *m_ctxCfgPtrs[3] = {nullptr, nullptr, nullptr};
  // 「该配置是否已设置」只能由这两个布尔位回答，见 rebuildContextConfig() 的说明。
  bool m_sfConfigActive = false;
  bool m_egConfigActive = false;
  // Index into m_graphsInfo of the graph this model executes. 0 unless
  // setEnabledGraph() named a different one.
  size_t m_activeGraphIdx = 0;
};

#endif  // QNNMODEL_HPP
