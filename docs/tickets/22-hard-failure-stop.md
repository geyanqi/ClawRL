# 22 — 硬故障确定性关闭并取消 run

**What to build:** 把硬故障 Observation 转换为经 Harness 执行的一次 cancel 和不可变失败终态，不经过 Governor 软判断。

**Blocked by:** 18, 21

**Status:** ready-for-agent

- [ ] NaN/Inf、OOM、重复崩溃、heartbeat/scheduler 停滞、CFS corruption 和 permanent reward failure 都有确定分类。
- [ ] 当前 fenced controller 执行一次 idempotent cancel/close；stale controller、重复日志和 late observation 不能重复 action 或改变终态。
- [ ] Reward terminal stop 与 ClusterAdapter cancel 闭环，失败后没有 trainer reward 或 optimizer update。
- [ ] 实验预算耗尽不能阻止安全 stop/cancel；emergency exemption 不能被用于提交新训练、查询或 scorer 工作。
