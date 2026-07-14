# 11 — 扩展 classic verl 的 per-Trajectory identity

**What to build:** 作为显式 wide-prefactor exception，在任何异步 dispatch 前写入稳定 identity，并产出可检查的 classic trajectory dump/contract report。

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] 每条 row 保留 `run_id/global_step/trace_id/uid/rollout_index/expected_rollout_count/judge_pack_id`，同 UID index 精确为 `0..n-1`。
- [ ] Identity 穿过 generation、repeat、union、balance、chunk/padding、RewardLoop worker、dump 和声明支持的 classic resume 路径。
- [ ] Batch `global_steps` 与 row `global_step` 始终一致；不一致、缺失或重复 index fail closed。
- [ ] 未验证的 classic 变体被 TRAIN readiness 拒绝；既有 RewardManager 行为与 CI 不被破坏。本票不宣称完成 CFS reward 集成。
