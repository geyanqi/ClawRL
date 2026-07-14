# 07 — 将 @8-certified pack 扩展至 16/32

**What to build:** 对成功的 Luna@8 prompt 使用独立 attempts 和新 holdout 依次尝试 16、32 items-per-turn，并精确保留最后一个有效 pack。

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] 16 失败时保留完全相同的 @8 prompt/pack 且不尝试 32；16 通过、32 失败时保留完全相同的 @16 pack。
- [ ] 每一级都有新的、不相交的 32-item holdout，并由 Sol 以 4 items-per-turn 标注。
- [ ] 每个候选 batch size 上，Sol 与 student 必须评估完全相同的 holdout item set。
- [ ] 失败级 prompt version 不得泄漏到被保留 pack；看过 holdout 后改 policy 必须创建新 attempt/holdout。
- [ ] Luna@4 terminal pack 不进入扩展路径；不得用训练一步生成 holdout，也不得发起训练期 Sol shadow audit。
