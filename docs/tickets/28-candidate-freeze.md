# 28 — 冻结唯一 CandidateFreeze 与可信 T0

**What to build:** 在已预注册 FinalEvaluationProtocol 下，将唯一完成的 122B candidate、base 和全部开发 artifact 绑定为不可变 CandidateFreeze。

**Blocked by:** 27

**Status:** ready-for-agent

- [ ] Freeze 绑定 trained/base checkpoint、DatasetVersion、122B JudgeBundle、ExperimentSpec、protocol hash 和可信 controller UTC `T0`。
- [ ] Base/trained 必须引用相同 EvaluationEnvironment hash，只有 checkpoint/model identity 不同；semantic generation、tool/Harness、wrapper 和 visible schema 一致。
- [ ] 多候选、未完成 run、环境 mismatch、缺 artifact 或 protocol hash 变化都会阻止 freeze。
- [ ] Freeze 后不能替换 checkpoint、environment、protocol 或 T0；Promotion/Future data 不能再创建新 candidate。
