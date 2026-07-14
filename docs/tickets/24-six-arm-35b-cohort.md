# 24 — Budgeted 六路 35B cohort 选择唯一候选

**What to build:** 在预注册最终协议、预算和受监控运行能力下，执行 `1 control + 4 exploration + 1 control replication` 并按 PromotionPolicy 选择唯一开发候选。

**Blocked by:** 08, 09, 20, 23

**Status:** ready-for-agent

- [ ] 四个 exploration 配置唯一；replication 除 run ID/独立 seed 外与 control 一致，不复制本轮 winner。
- [ ] 六个逻辑角色始终存在；并发不足时排队，总资源≤96 GPU，所有 child 在提交前 reserve budget。
- [ ] 每个 ExperimentSpec 明确冻结 DatasetVersion、JudgeBundle、prompt、RewardSchema/Scalarizer、calibrated aggregation、algorithm、generation 和 resources；Cohort/Promotion/Monitoring policy 与 protocol hash 同样在提交前冻结。
- [ ] 六个 arm 全部达到 success、deterministic failure 或 policy stop 后才选择一个候选；Future EvaluationDataset 不可见。
- [ ] 短跑淘汰释放的资源只能按冻结 cohort/budget 复用给排队 arm 或后续获批 action，不能静默改变六个逻辑角色。
- [ ] 受控实验与黑盒实验的结论边界被持久化，黑盒结果不得声称单因素因果。
