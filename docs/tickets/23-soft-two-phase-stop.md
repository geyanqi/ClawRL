# 23 — 软异常执行 checkpoint→grace→cancel

**What to build:** 让 Governor 依据冻结 MonitoringPolicy 和证据执行软异常两阶段停止，并与 durable checkpoint protocol 对齐。

**Blocked by:** 19, 22

**Status:** ready-for-agent

- [ ] Reward/KL/entropy/length/Judge failure fixture 产生引用证据的有限 ActionPlan，按 checkpoint→configured grace→cancel 执行。
- [ ] Checkpoint success、timeout 和 failure 都有确定 terminal outcome；每个 action 幂等且可恢复。
- [ ] 单独 reward 上升不产生 stop proposal，run 继续。
- [ ] 不得通过修改 prompt、scalarizer、aggregation、algorithm 或 generation config 进行“恢复”。
- [ ] 实验预算耗尽不阻止 checkpoint/cancel，但不能借 emergency path 扩大训练工作。
