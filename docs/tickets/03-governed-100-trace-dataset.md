# 03 — Governed DataSourceSkill 生成 100-Trace DatasetVersion

**What to build:** 用一个具体 fixture 数据源完成只读查询、立即脱敏、校验、去重、Badcase 选择和 data-only DatasetVersion 发布。

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] SQL AST 只允许 SELECT、获批表列和 join/filter；`*`、DDL、DML、UDF 和越权对象在查询前被拒绝。
- [ ] Raw OnlineTrace 不离开 adapter；sanitizer sentinel 不得出现在 CFS、日志、模型输入或 DatasetVersion，泄漏即整次 ingest 失败。
- [ ] DataSourceSkill 明确 Prompt/response/tool/model 字段映射、event/ingestion time、主键/去重键、数据配比和重复上报语义。
- [ ] 查询结果通过 row/byte/time、字段完整性和唯一性不变量；去重后产出恰好 100 个唯一 `training_allowed` TrainingTrace。
- [ ] 相同获批窗口与脱敏后内容确定地产生相同 DatasetVersion/hash；query、sanitizer、selection 或内容变化产生新 identity，并记录完整 lineage。
- [ ] `judge_only/eval_only` 同时被 loader 与 ExperimentSpec validator 拒绝；DatasetVersion 不引用或回写 Judge artifact。
- [ ] Run-once 可由外部 scheduler 幂等重试但不递归创建定时任务；缺真实 schema、凭据或治理批准时 `DATA_INGEST` 在查询前 BLOCKED。
