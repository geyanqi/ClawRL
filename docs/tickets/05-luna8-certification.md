# 05 — Luna@8 认证或明确耗尽 TRY_8

**What to build:** 为一个 Trace 实现 Luna `items_per_turn=8` 的最多三次 prompt attempt，独立演示首次认证成功和 `TRY_8_EXHAUSTED` 两种结果。

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] BaseJudgePrompt 与 TraceJudgePrompt 分别版本化；每个被审计 candidate 使用新的、不相交的 32-item holdout。
- [ ] AlignmentPolicy 在 holdout 生成前冻结；Auditor 只向 Optimizer 暴露 pass/fail 和预注册 aggregate diagnostics。
- [ ] AlignmentAuditor 比较时，Sol 与 Luna 必须评分完全相同的 holdout item set，且 aggregation=`calibrated_scalar` 进入 certification identity。
- [ ] 成功产物绑定 fit/holdout、prompt、teacher/student config、RewardSchema/Scalarizer、items、algorithm contract、policy 和 attempt identity。
- [ ] 三个 candidate 都失败后只产生 `TRY_8_EXHAUSTED`；不得继续隐式搜索、改门槛或复用已看过的 holdout。
- [ ] 缺任一数值 policy、model 或 reward contract 时 production 在外部 holdout/scorer 调用前 BLOCKED。
