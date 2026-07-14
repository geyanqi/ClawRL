# 32 — 完整 fixture closed-loop acceptance

**What to build:** 组合既有 slices，从 ingest、100×32 Judge 认证、16×128 reward step、六路实验、Governor transfer、122B gate 到 sealed Future-100，验证完整 lineage 和 fail-closed campaign。

**Blocked by:** 15, 31

**Status:** ready-for-agent

- [ ] Success fixture 只通过顶层 `DatasetVersion + JudgeBundle + ExperimentSpec` seam 驱动，并产出完整 RunRecord、DecisionRecord、reward、TransferCandidate 和 FinalEval outcome lineage。
- [ ] Fixture 类型结构上不能构造生产 adapter 或产生真实副作用；本票只做 composition，不重复实现下层领域行为。
- [ ] 另一个 fail-closed campaign 在选定 phase 的首个缺失/冲突 artifact 处终止，且无后续 scorer、optimizer、cluster 或 final-eval 副作用。
- [ ] Phase matrix 证明 `DATA_INGEST → JUDGE_CERTIFY → TRAIN_35B → TRAIN_122B → FINAL_EVAL` 不会因前一 gate green 而自动放行后一 gate。
- [ ] 所有不可协商负向约束在组合路径中仍成立：无 GT 训练、无 shadow Sol、无 run 内换目标、无 reserved aggregation、无 Router 自动扩容、Future data 不回流训练。
