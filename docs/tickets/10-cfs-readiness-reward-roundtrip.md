# 10 — CFS readiness probe 与 fenced reward roundtrip

**What to build:** 通过 Harness 验证一个 CFS backend 的跨 client publish-if-absent 能力；合格 backend 随后让一个 immutable synthetic Trajectory slot 完成 request、leased attempt、attempt-result 和唯一 ResolvedReward。

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] Probe 只访问专用 namespace，并验证 no-replace：恰好一个 winner、loser 行为确定；普通覆盖 rename 明确不合格。
- [ ] Consumer 看不到 partial payload，能在期限内校验 hash/size/visibility；结果不依赖 directory listing/counting，cleanup failure 被记录。
- [ ] 不合格 backend 产出 machine-readable BLOCKED evidence 且不创建 reward request；probe evidence 本身不自动使任一 TRAIN phase green。
- [ ] 合格 backend 先 publish-if-absent 唯一 TrajectoryManifest；stable key 的不同 trajectory/request hash 是 corruption，不是 retry。
- [ ] RewardRequest 回显 Trajectory hash；attempt ordinal/lease 冻结，前一 attempt 未 terminal 或 lease-expired 时不能开启下一 ordinal。
- [ ] 外部 FenceAuthority 只允许 current resolver epoch 按最小 eligible valid ordinal publish ResolvedReward。
- [ ] Resolve 前拒绝 RewardSchema/Scalarizer 版本不匹配、缺字段、非有限 scalar、越界 confidence 和 malformed turn tie-groups。
- [ ] Same-hash replay 幂等，conflicting hash corruption，late result 隔离；重启只从 committed artifacts 恢复且绝不填 neutral reward。
