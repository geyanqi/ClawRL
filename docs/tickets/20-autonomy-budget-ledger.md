# 20 — Fenced BudgetLedger 授权有限 ActionPlan

**What to build:** 用一个真实 fixture ActionPlan 演示多维 worst-case reserve、Harness allow、actual reconcile 和 over-budget deny。

**Blocked by:** 10

**Status:** ready-for-agent

- [ ] Ledger 原子限制 GPU/96-GPU cap、GPU-hours、job/cohort、query/row、scorer spend 和 wall-clock；缺失/负数等于零权限。
- [ ] 每个有限 child action 在 Harness allow 前 reserve，完成后 reconcile；拆分 children 不能绕过总上限。
- [ ] 只有当前 fencing epoch 的单 writer/CAS 可 reserve/reconcile；Governor 不能提高或替换自身 budget。
- [ ] Budget exhaustion 阻止新工作但不自动杀死在途 job；仅减少资源/终止风险的 emergency action 不得因实验预算耗尽被拒。
