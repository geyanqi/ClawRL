# 17 — verl v1 返回 grouped calibrated rewards

**What to build:** 在 v1/TransferQueue trainer seam 完成与 classic 等价的 grouped reward outcome，并验证 async resume 后 identity 写回。

**Blocked by:** 12, 14

**Status:** ready-for-agent

- [ ] v1 先发布完整 UID group 再等待 barrier，colocated worker 不因 singleton await 或事件循环阻塞而 deadlock。
- [ ] 128 个 scalar、extra info 与 trace refs 经过 TQ dispatch/reissue、chunking/reorder/dump 后精确匹配原 slot。
- [ ] Trainer-visible extra info 与 classic 相同，显式保留 dimensions、confidence、failure tags、evidence、tie-groups、JudgePack/Scalarizer versions 和 session/thread/turn refs。
- [ ] 不从 TQ key、返回顺序或 response hash 推断 rollout index；缺任一 slot 时 trainer 收不到 reward tensor。
- [ ] Reserved aggregation 和 run 内 reward contract mutation 在调用 Router 前被拒；classic/v1 输出 schema 一致。
