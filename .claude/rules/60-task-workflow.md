# 阶段化任务工作流约束

## 执行顺序

严格按 `TASKS.md` 推进：

1. Phase 0: baseline。
2. Phase 1: instrumentation。
3. Phase 2: dynamic ExpertCache。
4. Phase 3: async transfer/compute。
5. Phase 4: transition/prefetch predictor。
6. Phase 5: `MUL_MAT_ID` prefill/decode。
7. Phase 6: RX 6700S autotune。
8. Phase 7: expert FFN fusion 和 shared residency。
9. Phase 8: MTP integration and comparison。
10. Phase 9: iGPU experiments。
11. Phase 10: BAR/SAM/advanced memory experiments。

## 每项任务

每项任务都要在 `TASKS.md` 记录状态、依赖、涉及路径、设计决策、验收命令、指标、回退方案和原始结果位置。先完成前一阶段的 baseline/telemetry，再进入下一阶段。

跨多个 backend、新增 runtime subsystem、改变 tensor layout 或引入新的同步模型前，先在 `TASKS.md` 写风险和待确认决策，等待用户确认后再编码。预测、cache 和实验开关默认关闭，除非有明确的 correctness 和性能证据。
