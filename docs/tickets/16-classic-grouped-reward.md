# 16 — Classic RewardLoop 返回 grouped calibrated rewards

**What to build:** 通过 classic importlib RewardManager 完整发布同 UID 的异步 requests、等待 trace-ready，并向 trainer 返回 128 个有序 scalar 与 extra info。

**Blocked by:** 14

**Status:** ready-for-agent

- [ ] 所有 sample coroutine 先非阻塞发布再等待 group barrier；CFS polling 不阻塞 actor event loop，避免 singleton `run_single` 自锁。
- [ ] Worker chunking、reorder 和 padding 后，128 个 finite scalar 仍按原 rollout identity 回到 trainer，extra info/trace refs 完整。
- [ ] Trainer-visible extra info 保留 dimensions、confidence、failure tags、evidence、turn-local tie-groups、JudgePack/Scalarizer versions 和 session/thread/turn refs。
- [ ] 任一 rollout 缺失、NaN、版本不匹配时不产生 reward tensor；local tie-groups 不被伪装成全局 rank。
- [ ] Reserved aggregation ExperimentSpec 在任何 reward work 前被拒；run 内 prompt、scalarizer、aggregation 和 algorithm mutation 被拒。
