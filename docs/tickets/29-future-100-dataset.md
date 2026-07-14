# 29 — 提交 unseen eval_only Future-100 Dataset

**What to build:** 在 T0 后按冻结 predicate、identity normalizer、窗口和 seed 收集并提交恰好 100 条 Future Eval Prompt。

**Blocked by:** 28

**Status:** ready-for-agent

- [ ] Event time 与 ingestion time 标准化为 UTC 且都严格 `>T0`；Prompt identity 不存在于候选使用的任何训练/开发 DatasetVersion。
- [ ] 第一冻结窗口按 seed 采样；不足 100 只允许一次预声明的同长度扩展并对 union 重算，仍不足则 campaign INVALID。
- [ ] 不根据 base/trained answer 放松 predicate、替换样本或临时增加语义去重；未由 normalizer 定义的 near-duplicate 检测不实施。
- [ ] EvaluationDataset 是 immutable `eval_only`，不能进入 RL loader、Governor 输入或触发新候选。
- [ ] 缺真实 source/window/governance 时 `FINAL_EVAL` 在查询前 BLOCKED。
