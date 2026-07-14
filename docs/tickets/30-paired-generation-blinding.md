# 30 — 提交 paired generations 与 sealed 50/50 mapping

**What to build:** 让 base/trained 在同一 EvaluationEnvironment 下各提交 100 个 generation，并生成 scorer 无权读取的平衡 A/B mapping。

**Blocked by:** 29

**Status:** ready-for-agent

- [ ] 每个 Prompt/model role 恰有一个 committed evaluator-visible Trajectory，共 200 个；环境仅 checkpoint identity 不同。
- [ ] 只有 provider 能确认无结果或支持相同 idempotency key 时才重试；非幂等 unknown outcome 或 malformed committed generation 使 campaign INVALID。
- [ ] 预提交 seed 产生严格平衡 mapping：trained 为 A 50 条、为 B 50 条；mapping 存在独立 sealed storage，scorer credential 无读取权限。
- [ ] 缺 trusted provider result lookup/idempotency、sealing ACL/KMS 或独立 evaluator credential 时在 generation 前 BLOCKED。
- [ ] 缺失 item 不替换、不 resample；INVALID 不自动创建第二 campaign。
