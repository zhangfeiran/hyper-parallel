// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef PANGU_MX_PLANNED_WORKER_H_
#define PANGU_MX_PLANNED_WORKER_H_
namespace MulticoreRuntime {
class MxPlannedWorker : public MxTaskWorker<VectorWorkerPolicy::ALL> {
 public:
  __aicore__ inline void ProcessPlanned(bool gradients = false) {
    if (gradients != (this->replica_gradient_config_ != nullptr)) AscendC::Trap();
    ReadyHandshakeMeta meta;
    const bool has_ready = this->LoadReadyHandshakeMeta(&meta);
    if (meta.completion_event != 0 || this->core_num != 32) AscendC::Trap();
    if (has_ready) this->WaitForDeviceReady(meta);
#ifndef __DAV_CUBE__
    if (this->replica_gradient_config_ != nullptr && this->worker_id_ % 2 == 0 && this->worker_id_ < 8) {
      this->core_num = 4;
      this->ProcessReplicaGradients();
      return;
    }
#endif
    const auto mode = getDiagnosticMode(this->runtimeConfigPtr);
    if (mode == DiagnosticMode::PROFILE)
      Run<true>();
    else if (mode == DiagnosticMode::NORMAL)
      Run<false>();
    else
      AscendC::Trap();
  }

 private:
  template <bool Profiled>
  __aicore__ inline void Run() {
    uint32_t begin, end, stride;
#ifdef __DAV_CUBE__
    begin = this->worker_id_;
    end = this->cube_task_num;
    stride = 32;
#else
    auto offsets =
      reinterpret_cast<__gm__ int32_t *>(this->runtimeConfigPtr + getMixTaskIndexsOffset(this->runtimeConfigPtr));
    begin = offsets[this->worker_id_];
    end = offsets[this->worker_id_ + 1];
    stride = 1;
    if (begin > end || end > this->vector_task_num) AscendC::Trap();
#endif
    CycleTraceRecorder recorder;
    if constexpr (Profiled)
      recorder.Init(this->worker_id_, this->input_list[this->DIAGNOSTIC_IDX],
                    getAicProfileRecordCapacity(this->runtimeConfigPtr),
                    getAivProfileRecordCapacity(this->runtimeConfigPtr), getCubeWorkerCapacity(this->runtimeConfigPtr));
    for (uint32_t pos = begin; pos < end; pos += stride) {
#ifdef __DAV_CUBE__
      const TaskId id = this->cube_task_indexs.GetValue(pos);
#else
      const TaskId id = this->vector_task_indexs.GetValue(pos);
#endif
      if constexpr (Profiled)
        this->ExecuteTaskProfiled(id, recorder);
      else
        this->ExecuteTaskFast(id);
    }
  }
};
}  // namespace MulticoreRuntime
#endif
