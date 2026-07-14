# 27 — 独立 122B ExperimentSpec gate 首次短跑

**What to build:** 使用新 122B JudgeBundle 和单独编写/批准的 ExperimentSpec 完成一条受监控 fixture 短跑。

**Blocked by:** 26

**Status:** ready-for-agent

- [ ] 旧/部分 bundle、缺少批准的 122B spec 或由系统自动复制 35B spec 都在任何 optimizer update 前被拒绝。
- [ ] 新 spec 有独立 hash，显式包含 optimizer、LR、parallelism、resource、retry 和 monitoring；数值恰好相同仍可接受，只要不是自动复制。
- [ ] Bundle、spec、ClusterAdapter 与 `TRAIN_122B` readiness 全部 green 后才允许第一个 update。
- [ ] Fixture 路径完成 reward、checkpoint/StepApplied、monitor 和 RunRecord；缺真实资源时 production BLOCKED，不冒充训练完成。
