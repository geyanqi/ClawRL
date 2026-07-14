# 31 — Commit Sol verdicts、解封并判定 59/60

**What to build:** 使用原始 Initial Eval Rubric 对 100 个 sealed A/B pair 各提交一个 Sol verdict，完整后一次解封并形成不可变终态。

**Blocked by:** 30

**Status:** ready-for-agent

- [ ] Sol payload 只有 rubric 与 A/B evaluator-visible trajectories，不包含 model identity、student prompt 或 sealed mapping。
- [ ] 每项只接受一个 committed `A|B|tie`；malformed、missing、unknown-outcome judgment 使 campaign INVALID，不替换、不重评。
- [ ] 100 verdict 全部 commit 前不能解封；解封后只有明确 trained 选择计胜，base/tie 均不计。
- [ ] 60 trained wins 为 PASSED，59 为 FAILED；PASSED/FAILED/INVALID terminal 不可修改且不启动第二 campaign、重训或 arm selection。
- [ ] FinalEvaluationProtocol/schema drift 在 campaign 中使其 INVALID，不能改绑新 protocol 后继续。
- [ ] Immutable FinalEvalRun manifest 引用 protocol、CandidateFreeze、EvaluationDataset、200 generations、sealed mapping commitment、100 verdicts、trained-win count 和 terminal outcome hash。
- [ ] 最终报告只声明 Future Prompt 上的 Sol pairwise preference，不声明业务质量、人类偏好或事实正确率。
