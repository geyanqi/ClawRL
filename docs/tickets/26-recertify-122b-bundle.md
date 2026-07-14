# 26 — 用 100×32 fresh 122B rollout 再认证 JudgeBundle

**What to build:** 只接受 Governor terminal `stop_and_transfer` 的唯一 35B TransferCandidate，为全部 Trace 生成 fresh 122B distribution 并发布新 total bundle。

**Blocked by:** 25

**Status:** ready-for-agent

- [ ] Governor 仍在迭代、预算耗尽但未 transfer、非 terminal 或无唯一 TransferCandidate 时，不产生任何 122B rollout。
- [ ] 100 个 Trace 各生成恰好 32 条 fresh 122B Trajectory；不得复用 35B fit、holdout 或 certification identity。
- [ ] 失配 Trace 新建 attempt、重编 prompt 或使用 Sol fallback；任一 uncertifiable/缺失 Trace 阻止 bundle。
- [ ] 所有 Trace——包括旧 prompt 仍通过的 Trace——都产生绑定 fresh 32-rollout evidence 的新 122B CertificationReport；只有新 bundle hash 不算再认证证据。
- [ ] 新 JudgeBundle 无缺口且有独立 hash；旧、部分或仅重新标记的 35B bundle 被拒。
- [ ] 缺 122B inference model/config 或真实 scorer readiness 时 `TRAIN_122B` 在 rollout 前 BLOCKED。
