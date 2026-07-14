# 14 — 单 Trace/128-rollout ScoringSession Router

**What to build:** 将一个 UID 的 128 条 rollout 路由到唯一 `(run_id, global_step, uid)` ScoringSession，并覆盖所有 terminal JudgePack variants。

**Blocked by:** 06, 07, 10, 13

**Status:** ready-for-agent

- [ ] Router 只在 ExpectedTrajectorySet 完整后启动，按 pack-selected 4/8/16/32 或 Sol tier 分波次评分，最多 5 个常驻 subthreads。
- [ ] Router 重启使用新的 Codex sessions，并只从 request、Trajectory、JudgePack 和 committed events 重建；不复活隐藏会话记忆。
- [ ] 每条结果通过 fenced resolver 产生维度向量、finite calibrated scalar、confidence、failure tags、evidence、turn-local tie-groups、JudgePack/Scalarizer versions 和 session/thread/turn trace refs；invalid payload 不可 resolve。
- [ ] Global RouterCapacity 只排队 excess work；不自动扩容、创建账号、绕过限额或根据预测延迟做 admission rejection。
- [ ] 训练期不调用 Sol shadow、不现场优化 prompt，也不修改已冻结 scalarizer/aggregation/algorithm contract。
