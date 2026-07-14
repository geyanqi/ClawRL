# 04 — 单 Trace 生成 32 条 fit Trajectory 与 Sol labels

**What to build:** 用冻结 GeneratorPlan 为一个 TrainingTrace 生成 32 条唯一 fit Trajectory，并由 Sol 使用 Initial Eval Rubric 产生 TeacherLabelSet。

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] 恰好 32 个唯一 TrajectoryManifest 被提交；不足、重复或复制 response 补数会使 Trace incomplete。
- [ ] 原线上 response 只有在 GeneratorPlan 显式列出时才计入；generator identity 只进 lineage，不进入 Judge payload。
- [ ] Sol 固定 4 items-per-turn、最多 5 个常驻 subthreads，通过多波次完成 32 条并保存 turn lineage。
- [ ] TeacherLabelSet 引用 Initial Eval Rubric hash、Sol model/inference config、全部输入 content hashes 和完整 session/thread/turn trace。
- [ ] TeacherScorer、PromptOptimizer、AlignmentAuditor 的 session/input 隔离；无效或缺失 Sol label fail closed。
- [ ] 缺 model/rubric/generator 配置或永久 trace 治理批准时，production `JUDGE_CERTIFY` 在模型调用前 BLOCKED。
