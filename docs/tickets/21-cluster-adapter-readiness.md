# 21 — ClusterAdapter fixture 短跑与 phase readiness

**What to build:** 用 fixture adapter 完成 submit、status、logs、artifact、checkpoint、stop/cancel contract，并为真实 provider 建立副作用前 gate。

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] Fixture job 正常关闭并生成引用冻结 spec、日志、checkpoint 和 artifact hash 的 RunRecord。
- [ ] Submit/status/log/checkpoint/graceful-stop/force-cancel 是可区分的 typed operations，均携带 action idempotency key；崩溃恢复先查询 provider 状态再决定动作。
- [ ] Typed success/error 覆盖所有 adapter operation，fixture success 不被标记为 production smoke。
- [ ] Provider error 会关闭为包含证据的 failed RunRecord，不只返回瞬时 typed error。
- [ ] 缺 jobbuilder、image、CFS/NAS mount、queue、resource 或 credential 时 `TRAIN_35B/TRAIN_122B` 在 submit 前 BLOCKED。
