# 02 — 通过 Harness 执行一个幂等 typed action

**What to build:** 打通 `DecisionProposal → ActionPlan → HarnessDecision → ActionObservation → DecisionOutcome`，证明只有获批 typed action 能产生真实副作用。

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] 获批 action 只访问 allowlist fixture resource，并记录 proposal、policy、调用者、输入/输出 hash 和 AuditEvent。
- [ ] 普通文本、代码块和 Prompt Injection 保持惰性；越权凭据、网络、mount 或命令 proposal 被拒且零副作用。
- [ ] 每个 child action 使用 proposal-derived idempotency key；崩溃恢复先查询外部状态，不重复执行已完成 action。
- [ ] Allow、deny、provider error 和 recovery 各自产生不可变、可关联的 outcome，而不是原地修改同一记录。
