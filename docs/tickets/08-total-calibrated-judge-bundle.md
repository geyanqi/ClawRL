# 08 — 发布 total calibrated-scalar 100-Trace JudgeBundle

**What to build:** 对完整 DatasetVersion 编排既有 per-Trace 认证状态机，发布 scalar 可比较、无重复、全覆盖的 JudgeBundle。

**Blocked by:** 06, 07

**Status:** ready-for-agent

- [ ] DatasetVersion 恰有 100 Trace，每个 pack 可追溯到 32 条唯一 fit Trajectory/teacher labels；fixture 覆盖 Luna、Sol fallback 和 Golden terminal mix。
- [ ] Trace keyset 精确一致；任一缺失、重复或 `uncertifiable` Trace 阻止 bundle 和 `JUDGE_CERTIFY` readiness。
- [ ] JudgeBundle 单向引用 data-only DatasetVersion；RewardSchema/Scalarizer mismatch、非有限值或跨 pack 不可比都会阻止发布。
- [ ] v1 只接受 `calibrated_scalar`；hierarchical/microgroup 在任何 scorer、merge 或 normalization 调用前返回 `UNSUPPORTED_REWARD_STRATEGY`。
- [ ] Local tie-groups 只作诊断；不得隐式插入 Spark/verifier，group-relative all-low signal 不能授权 calibrated training。
- [ ] 编译重启时只处理未完成 Trace；相同已 commit pack 被复用，不同 hash 的 pack 被隔离并阻止发布。
- [ ] 一个 coverage manifest 原子引用全部 pack、Golden、RewardSchema/Scalarizer、aggregation、algorithm 和 total hash；ExperimentSpec 拒绝 dataset ID/hash 或 Trace keyset 不匹配的 DatasetVersion/JudgeBundle 组合。
