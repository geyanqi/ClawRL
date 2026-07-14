# 06 — TRY_4 收敛到 Luna、Sol fallback 或 uncertifiable

**What to build:** 从 `TRY_8_EXHAUSTED` 开始执行最多三次 Luna@4 attempts，并得到唯一 terminal certification outcome。

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] 每个 @4 candidate 使用新的 32-item holdout；任一通过即生成 terminal Luna@4 pack，且不再尝试 8/16/32。
- [ ] 三次 @4 全失败时生成 Sol fallback pack、GoldenHardEntry、失败 lineage 和 nonblocking human escalation。
- [ ] 只有 Sol 在 calibrated-scalar contract 下也无法产生有效 label/variance 时才输出 `uncertifiable`，它会阻止 bundle 发布。
- [ ] Sol 只有 group-relative order 的情况仅写诊断，不能授权 v1 训练或调松 rubric 制造 scalar。
- [ ] Teacher 有 variance/student collapse、低 scalar 但仍有相对 order、Teacher 无有效 calibrated variance 三类情况产生不同 evidence/tags，不能被混为一个 fallback 原因。
