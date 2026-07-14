# 25 — Governor 执行一个有界迭代并产出 transfer decision

**What to build:** 让一个 Governor tick 从已批准 cohort 证据产生最多一个有限 ActionPlan，最终选择新迭代或冻结唯一 `TransferCandidate`。

**Blocked by:** 24

**Status:** ready-for-agent

- [ ] Fixture 分支可提出新 DatasetVersion、需再认证的新 ExperimentSpec，或 terminal `stop_and_transfer(candidate_id)`；成功主线产出一个 immutable TransferCandidate。
- [ ] DecisionProposal 显式包含 hypothesis、evidence、config diff、worst-case budget 和 expected result。
- [ ] ActionPlan 是有限 child DAG，每个 child 有独立 idempotency、budget reservation 和完成条件；禁止递归/无界 spawn。
- [ ] 所有读写/提交继续经过 purpose、phase readiness、BudgetLedger 和 Harness；Governor 不能自增预算。
- [ ] 新 DatasetVersion 分支必须重新经过 DataSourceSkill、SQL、安全用途、sanitizer 和 lineage path；超出 Router capacity 的工作被 deny/defer，不触发自动扩容。
- [ ] 改 data/prompt/scalarizer/algorithm 会产生新 artifact/认证，不能修改在途 run；reserved aggregation 被拒绝。
- [ ] Governor 不可读取 Future EvaluationDataset；ExperimentSummary 引用持久证据，黑盒结果不写成单因素因果。
- [ ] Governor 重启从 immutable child DAG 与 committed outcomes 恢复，已完成 child action 不重复执行。
