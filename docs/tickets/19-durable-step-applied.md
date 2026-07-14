# 19 — Durable checkpoint manifest gate `StepApplied`

**What to build:** 对一个已 step-ready 的 reward set 执行 optimizer update，并以 checkpoint-first protocol 协调 durable state 与唯一 `StepApplied`。

**Blocked by:** 18

**Status:** ready-for-agent

- [ ] Update 后同步提交包含 model/optimizer state、run/spec/step 和 reward-set hash 的 durable checkpoint manifest，再发布引用其 hash 的 StepApplied。
- [ ] 两者都存在时不得 replay；checkpoint 已 commit、marker 缺失时只能验证并 backfill marker，不能重做 update。
- [ ] 两者都不存在时从上一 checkpoint replay 一次；marker 存在但 checkpoint 缺失/不匹配时判 corruption 并停止。
- [ ] 恢复只有加载 StepApplied 引用的 checkpoint 后才能推进 next step。
- [ ] Async save、不可确认 durability、不可稳定 hash 或不可恢复 optimizer state 的 backend 在 TRAIN readiness 中 BLOCKED。
