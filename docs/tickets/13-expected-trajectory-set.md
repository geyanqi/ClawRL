# 13 — ExpectedTrajectorySet 冻结完整 rollout 基数

**What to build:** 在任何 scorer request 前提交一个 step 的精确 UID×rollout-index 集合、Trace/JudgePack mapping 和 immutable Trajectory slots。

**Blocked by:** 10, 11

**Status:** ready-for-agent

- [ ] Expected set 来自冻结 spec，包含每个 UID 的 `0..n-1` 与每个 slot 的 Trace/JudgePack；fixture 数量可配置而非隐藏写死 128。
- [ ] 每个 slot 使用已验证的 publish-if-absent protocol 提交 TrajectoryManifest，stable key 不依赖到达顺序。
- [ ] 缺失、重复、串 step、mapping 改变或 trajectory hash 改变在任何 scorer 调用前停止。
- [ ] Classic contract 通过；v1 集成在其专属 RewardManager ticket 中使用相同验证器，不能另造 identity 语义。
