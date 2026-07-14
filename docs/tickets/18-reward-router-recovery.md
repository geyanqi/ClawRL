# 18 — Reward/Router 恢复到唯一 step-ready 结果

**What to build:** 对 `16 UID × 128 rollout = 2048 slots` 的 fixture 注入乱序、重试、竞态和重启，得到完整 step-ready 或 terminal stop，而不执行 optimizer update。

**Blocked by:** 16, 17

**Status:** ready-for-agent

- [ ] 每个声明支持的 trainer path 都冻结 2048 slots；所有 resolved 后只出现一个 step-ready，缺失/重复/串 step 不可通过。
- [ ] 覆盖 duplicate/conflicting result、out-of-order attempt、lease expiry、resolver/Router/controller restart 和 late-result quarantine。
- [ ] 相同 run/spec resume 复用 committed slot/reward；未知 spec 或 changed hash 被拒，attempt exhaustion 产生 typed stop request。
- [ ] 任何失败路径都不返回 partial/neutral reward；Router capacity 只形成可恢复队列，不触发自动扩容或账号创建。
- [ ] Production 永久 trace 治理未 green 时 TRAIN readiness BLOCKED。
