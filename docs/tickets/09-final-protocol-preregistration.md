# 09 — 在候选选择前冻结 FinalEvaluationProtocol

**What to build:** 提交一个完整 canonical FinalEvaluationProtocol 并返回 preregistration receipt，使没有该 receipt 的 promotion/freeze 请求被拒绝。

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Protocol 在任何 candidate result 前固定 predicate/query schema、双时间条件、identity normalizer、一次 window extension、100/60 和原始 Initial Eval Rubric。
- [ ] Semantic EvaluationEnvironment、sample seed、50/50 A/B seed、retry/idempotency、verdict 和 invalid 规则全部进入 hash。
- [ ] 原对象不可修改；变更只能产生新的不兼容 protocol，不能替代已绑定 campaign 的版本。
- [ ] 缺 trusted clock、真实 source、sealed storage 或 provider contract 时 receipt 可在 fixture 产生，但 production `FINAL_EVAL` 明确 BLOCKED。
