# 12 — 迁移 identity 到 verl v1 与 TransferQueue

**What to build:** 把 classic identity contract 迁移到 v1 colocated/asynchronous reward 与 TransferQueue checkpoint reissue，并产出同构的可检查 dump。

**Blocked by:** 11

**Status:** ready-for-agent

- [ ] v1 reward reconstruction 不只携带 raw prompt；UID、step、rollout index、expected count、Trace 和 JudgePack identity 全部保留。
- [ ] TransferQueue 持久化原 slot ordinal；dispatch、返回、checkpoint/reissue 后不得从队列位置、响应 hash 或到达顺序猜 index。
- [ ] Resume 后相同 slot identity 与 content hash 不变，重复/冲突立即 fail closed。
- [ ] 无法证明 identity 的 v1 mode 被 TRAIN readiness 拒绝；classic 语义和旧 RewardManager 兼容性保持不变。
