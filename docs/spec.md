# Codex 自治 RL 系统最终规格

> 状态：Final / implementation source of truth
>
> 背景：[human_motivation.md](./human_motivation.md)
>
> 优先级：本文件是实现、验收与后续决策的唯一规格 source of truth

## Problem Statement

当前 RL 流程中的数据准备、Judge 对齐、reward 生产、训练任务编排、日志监控、实验选择和经验沉淀彼此割裂，需要人工在 SQL、共享存储、模型评测、集群任务和训练日志之间反复协调。更严重的是，如果数据、Judge、reward 语义、训练配置和自治 action 缺少不可变身份与强制边界，系统即使能够跑起来，也无法证明一次 optimizer update 使用了哪批 Trajectory、哪个 JudgePack 和哪种 reward contract。

本项目的唯一优化目标是：在训练阶段已知且可反复查询 `Sol + Initial Eval Rubric` 的条件下，使训练后的冻结模型在冻结之后出现的、未来未见的困难 Prompt 上，更容易获得 Sol 的 pairwise preference。

这个目标不等于真实业务质量。系统允许模型学习 Sol evaluator 的稳定偏好或盲点，只要这种行为能够泛化到 Future Eval Prompt。项目不得把成功表述为事实正确率、人类偏好、真实用户价值、任务成功率或线上业务 KPI 的提升。

系统必须同时满足四类约束：Codex 可以在预算内自治决策；真实 action 必须经过 Harness；跨进程状态必须来自不可变持久化产物而不是会话记忆；任何身份、reward、权限或最终评测的不确定性都必须 fail closed。

## Solution

构建一个以 CFS 为共享交换面、以 verl RewardLoop 为训练集成 seam、以 Codex Governor 为自治决策核心、以 Harness 为唯一真实 action 边界的 RL 控制系统。

数据侧通过 `DataSourceSkill` 读取、校验、去重并立即脱敏 Online Trace，形成只包含训练数据和 lineage 的不可变 `DatasetVersion`。Judge 侧用 Sol 对每条 Trace 的 fit/holdout Trajectory 产生 teacher labels，通过严格状态机认证 Luna；失败 Trace 使用 Sol fallback 并进入 Golden Hard Pool。认证结果形成单向引用 DatasetVersion 的完整 `JudgeBundle`，`ExperimentSpec` 独立绑定二者。

训练侧在进入异步 RewardLoop 前为每条 Trajectory 固化稳定 identity 和预期基数。自定义 RewardManager 将请求写入 CFS；Router 按单 Trace、单 step 的 ScoringSession 路由到最多 5 个常驻 Judge subthreads。CFS 采用内容寻址 payload、原子 publish-if-absent manifest、fencing epoch 和唯一 `ResolvedReward`。缺失、冲突或无效 reward 会阻止 optimizer update 并触发停训，不能填中性分。

实验侧在 96 GPU 上形成 `1 control + 4 exploration + 1 replication` 的 35B cohort。Governor 只能在冻结的策略、资源白名单和 `AutonomyBudget` 内提出有限 ActionPlan；硬故障确定性终止，软异常执行 checkpoint、grace、cancel 两阶段停止。确定 35B 候选后，122B 必须使用全新 rollout 重新认证完整 JudgeBundle，并使用单独审批的 122B ExperimentSpec。

最终验收在候选选择前冻结协议，在唯一候选冻结后的新时间窗收集恰好 100 条合格 Prompt，执行 base/trained 同环境生成、身份隔离、平衡 A/B 盲评和一次 Sol 判断。只有 trained 被明确选中才计胜，至少 60 胜通过。

系统提供 `fixture` 与 `production` 两种执行 profile。Fixture 使用确定性 fake adapters 完成全部高层行为验收，且结构上不能触达真实资源；Production 按阶段生成 `ReadinessReport`，缺少模型、prompt、策略、凭据、CFS 或集群配置时，在任何副作用之前拒绝执行。

## User Stories

1. 作为 RL 项目负责人，我希望成功只由 Future Eval Prompt 上的 Sol pairwise preference 定义，从而避免未经验证的业务质量声明。
2. 作为 RL 项目负责人，我希望最终评测协议先于候选选择冻结，并且只评测一个候选，从而避免把未来数据变成开发集。
3. 作为数据提供者，我希望每个数据源声明字段、时间、去重、用途、SQL 和脱敏约束，从而让自治拉数不越界。
4. 作为数据审计者，我希望原始数据只在 adapter 内短暂存在，任何 CFS、日志或模型输入都只包含已脱敏数据。
5. 作为数据审计者，我希望 `training_allowed`、`judge_only` 和 `eval_only` 用途在运行时强制执行，从而防止 GT 和未来评测泄漏到 RL dataloader。
6. 作为 Governor，我希望在获批约束内选择恰好 100 条唯一 Badcase Trace，并生成可复现的 DatasetVersion。
7. 作为 Judge 开发者，我希望每条 Trace 有恰好 32 条唯一 fit Trajectory 和 Sol teacher labels，从而建立稳定的认证输入。
8. 作为 TeacherScorer，我希望 Sol 固定每 turn 处理 4 items，单 Session 最多 5 个常驻 subthreads，并通过多波次完成任务。
9. 作为 PromptOptimizer，我希望共享 BaseJudgePrompt 与 TraceJudgePrompt 分别版本化，并且只能使用 fit 数据和审计器暴露的汇总诊断。
10. 作为 AlignmentAuditor，我希望 AlignmentPolicy 在 holdout 生成前冻结，每次候选审计使用全新且不相交的 32 条 holdout。
11. 作为 Judge 开发者，我希望 Luna 按明确的 `8 → 4` 降级和 `8 → 16 → 32` 扩展状态机认证，从而避免实现者自行解释搜索流程。
12. 作为 Judge 开发者，我希望 Luna@4 仍失败时自动产生 Sol fallback、GoldenHardEntry 和 human escalation 记录，而不是阻塞整个 DatasetVersion。
13. 作为 Reward 设计者，我希望 v1 只接受 `calibrated_scalar`，并明确拒绝尚未定义的 hierarchical 和 microgroup semantics。
14. 作为训练开发者，我希望每条 reward 都能追溯到稳定的 run、step、UID、rollout index、Trajectory 和 JudgePack。
15. 作为训练开发者，我希望 step 的预期 Trajectory 集合在评分前冻结，从而在异步重排、重试和恢复后仍能发现缺失、重复或串 step。
16. 作为 Router，我希望每个 ScoringSession 只服务一个 `(run_id, global_step, uid)`，并能仅凭持久化产物在新 Codex session 中重建上下文。
17. 作为训练运营者，我希望重复请求、并发结果和 Router 重启只能解析出一个确定 reward；冲突内容必须停止 step。
18. 作为训练运营者，我希望 reward 超时可按冻结策略重试，耗尽后停止 run，绝不静默注入中性 reward。
19. 作为安全平台负责人，我希望模型文本和工具请求都只是 ActionProposal，只有 Harness 授权后的 typed action 才能产生真实副作用。
20. 作为 Judge Agent，我希望可以通过 Harness 使用获批的 Python、JavaScript、正则和临时 scratch，而工具使用本身不使 reward 失效。
21. 作为审计者，我希望 Run、Decision、Action、reward 和 scoring trace 都是不可变、内容寻址、可验证 hash 链的一部分。
22. 作为训练运营者，我希望 ClusterAdapter 统一 submit、status、logs、checkpoint、graceful stop 和 cancel，且真实配置缺失时明确报告未就绪。
23. 作为训练监控者，我希望 NaN、Inf、OOM、重复崩溃、heartbeat 停滞和永久 reward 故障走确定性终止。
24. 作为训练监控者，我希望软异常由 Governor 给出证据并执行 checkpoint、grace、cancel，而 reward 上升本身不能触发停止。
25. 作为实验负责人，我希望一轮包含 1 个 control、4 个唯一 exploration 和 1 个 control replication，并冻结每个 arm 的完整配置。
26. 作为实验负责人，我希望 replication 只改变 run identity 和独立 seed，从而估计 control 方差；winner replication 留给后续 cohort。
27. 作为平台负责人，我希望 Governor 的数据、scorer、GPU、job、时间和费用权限都由不能自我扩大的预算账本强制执行。
28. 作为模型负责人，我希望所有 100 条 Trace 都用 32 条全新 122B rollout 再认证，未完成时不能执行任何 122B optimizer step。
29. 作为模型负责人，我希望 122B 使用独立编写和批准的 ExperimentSpec，而不是由系统机械复制 35B 超参。
30. 作为最终评测者，我希望 future 数据同时满足 event time 与 ingestion time 严格晚于冻结时刻，并排除训练/开发 Prompt identity。
31. 作为最终评测者，我希望 base 和 trained 只允许 checkpoint identity 不同，其他有效生成与工具环境完全一致。
32. 作为最终评测者，我希望 A/B 映射在 100 个 judgment 全部提交前对 scorer 保密，且严格平衡位置。
33. 作为最终评测者，我希望任何未知结果、无效输出或缺失 item 都让本次 campaign 变为 INVALID，而不是替换样本或重评。
34. 作为审计者，我希望 60 个 trained wins 明确通过、59 个明确失败，tie 和 base 胜永远不计入 trained wins。
35. 作为实现者，我希望 fixture profile 能端到端演示所有阶段，同时不需要伪造生产模型、数据库、CFS 或集群凭据。
36. 作为生产运营者，我希望每个阶段在副作用前单独通过 readiness gate，而不是被一个笼统的全局开关误放行。

## Implementation Decisions

### 1. 优先级与不可协商约束

- 本文是当前实现规格；任何后续决策必须先更新本文，再同步受影响的 tickets。
- RL 训练数据只来自线上回流 Badcase。GT 业务评测数据只能是 `judge_only` 或 `eval_only`，不得进入 RL dataloader。
- Judge 使用工具必须经过 Harness；获批工具使用不自动使 reward 作废。
- 正式训练期不持续运行 Sol shadow audit，不在 run 内动态更换 prompt、scalarizer、aggregation 或 RL algorithm contract。
- reward 上升本身不是停训条件。
- Router 不自动扩容、不创建账号、不绕过 provider 限额，也不根据预测延迟自动拒绝训练 job；并发由外部配置和预算控制。
- Future EvaluationDataset 不能用于选择 arm、产生新候选、触发重训或修改最终协议。
- 最终 Sol 使用原始 Initial Eval Rubric，而不是 student Base/Trace Judge prompt。
- 对外成功声明仅限 Sol pairwise preference，不扩展为业务质量。

### 2. 执行 profile 与阶段就绪

- `execution_profile` 只能是 `fixture` 或 `production`。
- Fixture 使用 fake DataSource、scorer、CFS、Harness 和 ClusterAdapter；其配置类型不能持有生产凭据或实例化生产 adapter。
- Production 在每个阶段生成机器可读 `ReadinessReport`，包含 `DATA_INGEST`、`JUDGE_CERTIFY`、`TRAIN_35B`、`TRAIN_122B`、`FINAL_EVAL` 五个 gate。
- 一个 phase 只有在该 phase 所需的外部 artifact hash、凭据引用、治理批准、策略数值和 adapter contract 全部通过后才能执行。较早 phase 通过不代表后续 phase 就绪。
- Fixture prompt/model ID、placeholder、缺失或未知 major schema 在 production 中无效。真实 smoke 只能在对应 gate green 后显式运行，不能被 fixture test 冒充。

### 3. 不可变产物与领域关系

- 所有 JSON artifact 使用 RFC 8785 canonical JSON 的 UTF-8 bytes，并用 SHA-256 计算内容 hash；对象自身 hash 字段不参与 digest。大 payload 单独以 `hash + size` 内容寻址引用。
- 每个 schema 都有 `schema_version`；未知 major version fail closed。同一语义输入必须跨进程产生相同 bytes 和 hash。
- `OnlineTrace` 是 adapter 内的原始记录；`TrainingTrace` 是立即脱敏后的不可变训练任务；`Trajectory` 是 policy 对 TrainingTrace 的新 rollout。
- `DatasetVersion` 只拥有脱敏 TrainingTrace refs、用途、来源/选择 lineage、配比和内容 hash。它不引用 JudgePack，也不会被后续认证回写。
- `JudgePack` 绑定一个 `(dataset_version_id, trace_id)`；`JudgeBundle` 对一个 DatasetVersion 提供无重复、全覆盖的 Trace→JudgePack 映射和 coverage manifest。
- `ExperimentSpec` 独立引用 DatasetVersion 与 JudgeBundle，并校验 dataset/hash 相同且 Trace keyset 完全一致。
- 关键不可变对象包括 `GeneratorPlan`、`TrajectoryManifest`、`TeacherLabelSet`、`HoldoutSet`、`AlignmentPolicy`、`AlignmentAttempt`、`CertificationReport`、`GoldenHardEntry`、`RewardSchema`、`Scalarizer`、`JudgePack`、`JudgeBundle`、`ExperimentSpec`、`CohortSpec`、`TransferCandidate`、`RunRecord` 和 `DecisionRecord`。

### 4. 生命周期、决策与 action 记录

- 每个 run 只有一个逻辑 `RunController` writer。Adapter、scorer 和 monitor 只写不可变 `Observation`；controller 按单调 sequence 和 previous-event hash 写 `RunEvent`。
- `RunClosed` 是不可变终态 manifest。关闭后到达的 Observation 被隔离，不能修改结果。
- 自治动作拆为 `DecisionProposal`、`ActionPlan`、`HarnessDecision`、`ActionObservation`、`DecisionOutcome`，不能原地补写同一记录。
- 每个外部 child action 使用 proposal ID 派生的 idempotency key。崩溃恢复时先查询外部状态，再决定是否重试 submit、stop 或 cancel。
- RunController、Router resolver 和 BudgetLedger writer 都带由外部强制的单调 fencing epoch；旧 epoch 不能关闭 run、解析 reward 或消费预算。

### 5. DataSourceSkill 与 DatasetVersion

- DataSourceSkill 声明 purpose classification、表/字段业务含义、event/ingestion time、主键/去重键、重复上报规则、Prompt/response/tool/model 映射、join/filter allowlist、sanitizer、行/字节/时间限制和查询后不变量。
- SQL 必须通过 AST 校验：只允许 SELECT、获批表/字段和 join/filter；禁止 `*`、DDL、DML、UDF 和未批准对象。
- 原始 row 只能存在于 adapter 内。Sanitizer 在写 CFS、日志或发送模型之前执行；sentinel 泄漏测试失败时整个 ingest fail closed。
- 首版 DatasetVersion 在配置去重后必须包含恰好 100 个唯一 TrainingTrace ID。选择 SQL、时间窗、sanitizer、dedupe、purpose 和内容 hash 全部进入 lineage。
- `training_allowed` 才能进入 RL dataloader。`judge_only` 与 `eval_only` 由 loader 和 ExperimentSpec validator 双重拒绝。
- 数据拉取由外部 scheduler/daemon 调用幂等 run-once entrypoint；系统不依赖桌面 Scheduled Task，也不在 action 内递归创建无界定时任务。

### 6. 认证输入与角色隔离

- 每个 TrainingTrace 由冻结的 `GeneratorPlan` 产生恰好 32 条唯一、不可变 fit Trajectory。原线上 response 只有在 GeneratorPlan 显式列出时才可计入，不能通过复制 response 补足 32。
- Generator identity 只写 lineage，不能进入任何 TeacherScorer、student Judge 或 AlignmentAuditor payload。
- TeacherScorer、PromptOptimizer、AlignmentAuditor 使用独立 session 和权限。TeacherScorer 只写 Sol labels；PromptOptimizer 不拥有 audit raw data；AlignmentAuditor 不能修改 prompt 或 policy。
- 共享 `BaseJudgePrompt` 与每 Trace 的 `TraceJudgePrompt` 独立版本化。
- Sol 对所有 fit/holdout 固定 4 items-per-turn。单 ScoringSession 最多 5 个常驻 subthreads，32 items 通过 `5×4 + 3×4` 等多波次完成。
- 每个候选 attempt 使用 32 fit items；每个进入 audit 的候选都生成全新、32 条、与 fit 和所有历史 holdout 不相交的 HoldoutSet。Holdout 由获批外部 generator pool 产生，再由 Sol 标注。
- Auditor 对 PromptOptimizer 只公开 pass/fail 和预注册的 aggregate diagnostics；item-level response/label 仅供审计。下一 attempt 必须使用新的 holdout。

### 7. Luna 认证状态机

- `AlignmentPolicy` 在对应 holdout 生成前冻结；缺少数值、看过 holdout 后改门槛或复用旧 holdout都不能认证。
- JudgePack identity 至少包含 DatasetVersion/Trace、fit/holdout IDs、Base/Trace prompt、teacher/student inference config、items-per-turn、RewardSchema/Scalarizer、aggregation、RL algorithm contract、AlignmentPolicy 和 attempt ID。
- 搜索从 `TRY_8` 开始，最多 3 个 prompt candidate attempts。某个 attempt 通过 holdout 后，以该 prompt 依次进入 `TRY_16`、`TRY_32`；每一级使用独立 attempt 和新 holdout。首次扩展失败即保留上一已认证 pack，失败级 prompt 不得替换它。
- 如果 3 个 Luna@8 attempts 全失败，进入 `TRY_4`，最多 3 个 attempts。Luna@4 一旦认证即为 terminal student pack，不再回升 8/16/32。
- 如果 Luna@4 全失败，产生 `scorer_tier=sol`、`certification_mode=teacher_fallback` 的 terminal JudgePack，同时追加 GoldenHardEntry、失败 attempts 和 human escalation。Sol fallback 不需要 student holdout。
- 每条 Trace 最终只能是：certified student pack、Sol fallback pack，或 `uncertifiable`。只有 Sol 本身无法在 calibrated-scalar contract 下给出有效 label/variance 时才是 uncertifiable；它会阻止 JudgeBundle totality 和训练。
- Sol scalar 无方差但仍有 group-relative order 只记录诊断，v1 不得借此启动 calibrated-scalar 训练。
- scorer model、prompt、items-per-turn、RewardSchema/Scalarizer、aggregation、algorithm contract 或跨模型 response distribution 改变都要求新认证。run 内 checkpoint 正常演进是冻结 ExperimentSpec 允许的 on-policy drift；35B→122B 不豁免。

### 8. Reward contract 与 aggregation

- Judge result 包含维度向量、overall scalar、confidence、failure tags、evidence、当前 turn 的有序 tie-groups、JudgePack/Scalarizer 版本和 session/thread/turn trace 引用。
- RewardSchema 固定字段、范围和缺失策略；Scalarizer 固定有限输出范围与单调 mapping；confidence 必须在 `[0,1]`。无效、NaN、缺字段或版本不匹配会阻止 reward resolve。
- Bundle certification 必须审计所有 pack 的 scalar 可比较性。控制组使用校准到同一 Sol 标尺的 scalar。
- v1 唯一可接受的 `RewardAggregationStrategy` 是 `calibrated_scalar`。local rank/tie-groups 只用于诊断，不拼接为 128-rollout 全局排序，也不影响 optimizer reward。
- `hierarchical_rank` 和 `local_microgroup_rank` 是保留名称；v1 validator 返回 `UNSUPPORTED_REWARD_STRATEGY`。Governor 不能选择它们，系统不能调度 merge/tournament 或 microgroup normalization。启用前必须有新 ADR、算法 contract、认证与测试。

### 9. verl reward identity 与 step 基数

- 现有 verl surface 不被假定天然提供可靠的 per-trajectory `global_step` 和 `rollout_index`。在任何异步 dispatch/reorder 前，metadata prefactor 必须为每条 sample 写入 `run_id`、单数 `global_step`、`trace_id`、`uid`、`rollout_index`、`expected_rollout_count` 和 `judge_pack_id`。
- 为 verl 兼容保留 batch-level `global_steps`，并在进入 reward path 时断言它与每条 sample 的 `global_step` 一致。
- `rollout_index` 在同一 UID 内确定为 `0..n-1`，必须穿过 generation、repeat、union、balance、chunk/padding、RewardLoop worker、dump 和 checkpoint resume。每个明确支持的 classic/v1 路径都需 contract test；未验证路径由 ReadinessReport 拒绝。
- 每个 step 在评分前提交 `ExpectedTrajectorySet`，包含所有 UID 与其 `0..n-1` 的精确笛卡尔 tuple，以及 trace/JudgePack mapping。缺失、重复、串 step、mapping 变化或 Trajectory hash 变化会在评分前停止 step。
- 稳定 reward key 为 `(run_id, global_step, uid, rollout_index, judge_pack_id)`。对应 rollout slot 首先以 publish-if-absent 提交唯一 TrajectoryManifest；首个有效提交获胜，晚到的不同内容被隔离并视为 corruption。
- RewardRequest 必须引用并回显 TrajectoryManifest hash，因此同一稳定 key 下的不同 request/trajectory hash 不是幂等重试，而是完整性错误。

### 10. CFS 状态机、发布与恢复

- 逻辑协议始终先写 immutable payload，再发布包含 hash/size 的 commit manifest。消费者只轮询由预期 keyset 推导出的确定 manifest path，不通过目录计数或 listing 推断 ready。
- Production 必须提供跨 client 原子的 publish-if-absent，例如 `O_EXCL`、no-replace rename、conditional put 或等价能力；普通会覆盖目标的 rename 不合格。
- `CfsCapabilityProbe` 从不同 mount client 并发竞争，验证恰好一个 winner、无 partial read、checksum 正确、在配置可见性期限内可读且 loser 行为确定。能力不足时对应 production phase 不就绪。
- 单 reward 状态为 request published → attempt claimed/leased → attempt-result committed → resolved。Attempt 使用唯一 ID、冻结 deadline/lease/retry policy；每个 attempt 写独立 result manifest。
- 只有持有当前 fencing epoch 的 Router resolver 可以 publish 唯一 `ResolvedReward`。首个有效 resolved manifest 获胜；相同 hash 重放幂等，不同 hash 是 corruption；晚到 result 被隔离。
- `trace-ready` 只依赖该 UID 的全部 resolved manifests；`step-ready` 只依赖 ExpectedTrajectorySet 的全部 UID ready。任何缺失或重试耗尽产生 typed terminal error 和 run-stop-request，不产生 optimizer update。
- Durable exactly-once 以 checkpoint 为准，而不是以孤立 marker 为准。Optimizer update 后必须先完成一个同步 durable checkpoint manifest，其中包含 model/optimizer state、run/spec/step 和 reward-set hash；随后以 publish-if-absent 写 `StepApplied`，且它必须引用该 checkpoint hash。
- 如果 checkpoint 已 commit 而 `StepApplied` 缺失，新 fenced controller 只能为同一 checkpoint 补发 marker，不能重做 update；如果 `StepApplied` 存在，恢复必须加载它引用的 checkpoint 后才能推进；只有两者都不存在时才允许从上一 checkpoint 重放该 step。
- Async save、无法确认 durable completion、无法稳定 hash 或无法恢复 optimizer state 的 backend 不具备该保证，必须被对应 TRAIN readiness gate 拒绝。
- Resume 只能复用相同 run ID、相同 ExperimentSpec hash 和持久化的 next global_step；已提交 slot/reward 被复用，未解析 slot 可继续 attempt。
- 永久保存的是已脱敏、压缩、内容寻址并带 hash 的 scoring trace。Production 在启用永久保留前必须具备外部批准的 ACL、加密、容量、删除权和 incident policy；系统不得自行发明这些治理配置。

### 11. ScoringSession 与 Harness

- ScoringSession identity 为 `(run_id, global_step, uid)`，并记录 trace_id 以查找 JudgePack。不同 step/UID 不能共享隐藏会话状态。
- 每次 routing attempt 可以使用新 Codex session/thread ID；恢复必须只依赖 request、Trajectory、JudgePack 和已提交事件，不依赖复活旧会话。
- 单 Session 最多 5 个常驻 subthreads，并同时受全局 `RouterCapacityConfig` 限制。items-per-turn 完全来自已认证 JudgePack。
- Judge 训练期只调用 JudgePack 绑定 tier，不偷偷运行 Sol shadow audit，也不动态优化 prompt。
- Policy、Judge 或 Governor 的脚本、命令和工具调用都只是 typed `ActionProposal`。纯文本永远惰性。
- Harness 根据 action type、resource allowlist、调用者、phase、预算和 policy hash 返回 allow/deny/error。Deny 无真实副作用并产生 AuditEvent。
- Judge 工具默认只能读取当前已脱敏 batch、使用受限 ephemeral scratch，不能访问网络、凭据或未授权 mount，并有 CPU、内存、时间和输出限制。Production whitelist 缺失时工具调用 fail closed。

### 12. Cluster、monitoring 与运行记录

- `ClusterAdapter` 统一 submit、query status、read logs、collect artifacts、checkpoint、graceful stop 和 cancel；每个调用携带 action idempotency key。
- 真实 jobbuilder 脚本、image、CFS/NAS mount、资源、队列和凭据均为外部配置。缺失时 fixture contract 可通过，但 `TRAIN_35B` 或 `TRAIN_122B` readiness 必须 BLOCKED。
- 硬故障包括 NaN/Inf、OOM、重复崩溃、heartbeat/scheduler 停滞、CFS corruption、永久 reward failure。确定性 monitor 生成 Observation，RunController 经 Harness 进入失败终态并 cancel。
- 软信号包括 reward/KL/entropy/length/Judge failure rate 的异常。Governor 必须引用证据和冻结 MonitoringPolicy，先请求 checkpoint，等待冻结 grace，再 cancel；checkpoint 失败也有确定终态。
- reward 上升不单独触发软停训。run 内不得更换目标来挽救异常。

### 13. 六路实验与 Governor 自治

- Campaign 假设硬上限为 12×8=96 GPU；`max_concurrent_experiments` 可让六个逻辑 arm 排队，但不能删除角色。
- `CohortSpec` 固定 1 control、4 个配置唯一的 exploration，以及 `replication_of=control_arm_id` 的 1 个 replication。Replication 除 run ID 和独立 seed 外与 control 一致，用于估计 control 方差；winner replication 属于后续 cohort。
- 所有六个 arm 到达 success、deterministic failure 或 policy stop 之一后，PromotionPolicy 才能选择唯一开发期候选。
- CohortPolicy、PromotionPolicy、MonitoringPolicy 和 AutonomyBudget 在提交前冻结。指标和阈值未提供时 production 不得自行猜测。
- Campaign-scoped `AutonomyBudget` 至少限制 96 GPU、GPU-hours、cohort/job 数、data rows/queries、scorer calls 或 spend、wall-clock deadline。缺失或负数表示零权限，不表示无限。
- `BudgetLedger` 在 Harness 授权前按 worst case 原子 reserve，完成后 reconcile actual usage。Governor 不能提高或替换自身预算，也不能通过拆 action 绕过限额。
- 一个 Governor tick 最多产生一个有界、不可变 ActionPlan；它可以包含有限 DAG，例如六个 job submission，但每个 child 都有独立 idempotency key、预算 reservation 和完成条件，不能递归无界 spawn。
- 受控消融可以得出单轴结论；黑盒多变量搜索只能记录“完整配置更优”。ExperimentSummary 和 DecisionOutcome 必须引用证据。
- Governor 可基于已批准的开发 RunRecord 提议新的 DatasetVersion、数据配比、Judge attempt 或 ExperimentSpec，但所有读写/提交仍经过用途、预算、readiness 和 Harness。
- 一个 Governor tick 最终可以选择继续有限迭代，或以 terminal `stop_and_transfer(candidate_id)` 产出唯一、不可变 `TransferCandidate`。Governor 非 terminal、选择继续迭代、预算耗尽但未明确 transfer，或存在多个候选时，122B 阶段不得开始生成 rollout。

### 14. 122B transfer gate

- 35B 阶段只能提供数据、prompt 和 scorer tier 的起点，不能把 35B JudgeBundle 标记为适用于 122B。122B 再认证只接受 Governor terminal `stop_and_transfer` 产生的唯一 TransferCandidate。
- 对 DatasetVersion 的全部 100 Trace，每条生成恰好 32 条全新 122B inference Trajectory，并重新执行认证。失败 Trace 新建 attempt、重编 prompt 或升级 Sol tier。
- 只有一个覆盖全部 Trace 的新 122B JudgeBundle 可通过 gate；部分覆盖、旧 bundle 或沿用旧 holdout 都无效。
- 122B ExperimentSpec 必须单独编写、审批和 content-hash，绑定新 bundle，并有自己的 optimizer、LR、parallelism、resource 和 monitoring 配置。数值可以恰好相同，但系统不得从 35B spec 自动复制生成。
- 新 bundle 与新 ExperimentSpec 都就绪之前，不允许任何 122B optimizer step。

### 15. Future-100 最终评测

- `FinalEvaluationProtocol` 在开发期候选选择前冻结，包含 DataSourceSkill/PromptIdentityNormalizer、困难 predicate/query schema、window 与一次扩展规则、sample=100、threshold=60、Initial Eval Rubric/wrapper、base/trained semantic generation config、sample seed、balanced A/B seed、retry 和 verdict schema。
- PromotionPolicy 只能读取 CandidateFreeze 前列出的开发产物；Future EvaluationDataset 的 ID/hash 对 Governor 不可见，也不能触发新 candidate 或 122B run。
- `CandidateFreeze` 在唯一候选选定后绑定 trained checkpoint、base checkpoint、DatasetVersion、JudgeBundle、ExperimentSpec、`EvaluationEnvironment` hash 和可信 controller clock 产生的 UTC `T0`。
- EvaluationEnvironment 覆盖 semantic decoding、tool/Harness policy、prompt wrapper 和 evaluator-visible trajectory schema。Base/trained 必须引用同一 hash，只有 checkpoint/model identity 可不同；backend 差异只有显式标为 non-semantic 才允许。
- 状态机为 `PROTOCOL_FROZEN → CANDIDATE_FROZEN → COLLECTING → EVAL_DATASET_COMMITTED → GENERATING → BLINDED_SCORING → UNSEALING → PASSED|FAILED|INVALID`，terminal 不可改变。
- 合格 row 必须满足标准化后的 event_time 与 ingestion_time 都严格大于 T0，且由冻结 PromptIdentityNormalizer 得到的 identity hash 不存在于候选用过的任何训练/开发 DatasetVersion。
- v1 只做精确的版本化 identity 排除；未在 normalizer 中定义的语义近重复检测属于范围外，系统不能看过答案后临时增加过滤。
- 第一个冻结时间窗关闭后按预提交 seed 在合格唯一集合中采样。少于 100 时只允许一次预声明的同长度扩展并在 union 上重新应用同一规则；仍少于 100 则 INVALID，不能放松 predicate。
- EvaluationDataset 是 immutable `eval_only` artifact，绝不能进入 DatasetVersion。
- 对每个 Prompt，base/trained 各提交恰好一个已 commit generation。Exactly-once 指恰好一个 committed result，不保证只有一次 provider invocation。只有 provider 可确认无结果或支持同一 idempotency key 时才重试；非幂等 provider 的 unknown outcome 使 campaign INVALID。
- 使用预提交 seed 确定严格平衡的位置分配，使 trained 在 50 条为 A、50 条为 B。给 Sol 的 payload 只包含 rubric 与 A/B evaluator-visible trajectories；A/B→model 映射保存在 scorer 无权读取的 sealed manifest。
- Sol 对每对只提交一个 committed verdict，规范化为 `A|B|tie`。任何 malformed committed result、unknown outcome 或 100 项中的缺失都使 campaign INVALID；不替换 Prompt、不重评。
- 所有 100 个 judgment commit 后才能解封 mapping。只有明确选择 trained 的 verdict 计胜；trained wins≥60 为 PASSED，≤59 为 FAILED，base 与 tie 均不计胜。
- FinalEvaluationProtocol 不能因 schema drift 或结果不理想而修改；不兼容变更使该 campaign INVALID。v1 不自动开启第二次 campaign。

## Testing Decisions

- 主要高层 seam 是：输入 `DatasetVersion + JudgeBundle + ExperimentSpec`，使用 fixture DataSource、Codex scorer、CFS、Harness 和 ClusterAdapter，观察完整 `RunRecord/DecisionRecord`、reward artifacts 和 terminal outcome。测试外部行为与持久化产物，不绑定内部类调用。
- Fixture profile 必须覆盖成功路径和每个 fail-closed 边界；production adapter 通过 contract fixture 验证。真实 smoke 是 readiness green 后的独立受控 gate，缺失外部配置时报告 BLOCKED，不能静默 skip 后算通过。
- Artifact contract tests 验证 RFC 8785 bytes、SHA-256、schema major rejection、跨进程稳定 hash、数据/Judge 单向 DAG、append-only event hash chain、terminal quarantine 和 stale fencing epoch 拒绝。
- Data tests 覆盖 SELECT allowlist、禁止 `*`/DDL/DML/UDF、去重、100 条唯一 Trace、用途隔离、sanitizer sentinel、lineage、eval_only/GT loader 拒绝。
- Judge tests 覆盖 100×32 基数、Sol 4 items/turn、最多 5 threads、角色输入隔离、三次@8、三次@4、16/32 新 holdout、失败保留上一 pack、Sol fallback、Golden Hard、uncertifiable 和 bundle totality。
- Reward tests 覆盖 RewardSchema/Scalarizer、跨 turn 标尺、local tie-groups 仅诊断，以及 hierarchical/microgroup 返回 `UNSUPPORTED_REWARD_STRATEGY` 且不产生任何相关调用。
- 复用 verl RewardLoop/RewardManager CPU 与 E2E 模式，验证 metadata 在所有声明支持的 repeat、async、balance、chunk/padding、dump 和 resume 路径中保持；ExpectedTrajectorySet 对缺失、重复和串 step fail closed。
- CFS contract tests 使用多 client race 验证 publish-if-absent、无 partial read、可见性期限、checksum、fencing、attempt lease、唯一 ResolvedReward、乱序/重复/重启、trace-ready/step-ready 和 corruption stop。
- Resume tests 验证相同 run/spec 复用 committed slot、未知 spec 拒绝、checkpoint-first/`StepApplied` reconciliation 的全部 crash window、异步或不可验证 save backend 被 readiness 拒绝，以及 reward retry 耗尽生成 stop request 而非中性分。
- Harness tests 验证文本惰性、allow/deny/error、资源越权、prompt injection、工具资源限制、ActionProposal idempotency、外部状态查询后恢复和完整 AuditEvent。
- Monitoring tests 覆盖硬故障立即终止、软异常 checkpoint→grace→cancel、checkpoint 失败终态，以及 reward 上升单独出现时继续运行。
- Cohort tests 覆盖六个逻辑角色、96-GPU hard cap、排队、control replication 配置等价、预算 reserve/reconcile、action splitting 防绕过、终态 barrier 和唯一候选 promotion。
- 122B tests 验证非 terminal/继续迭代/无唯一 TransferCandidate 时零 rollout，随后验证 100×32 fresh rollout、新 total bundle、单独 ExperimentSpec、旧/部分 bundle 拒绝，以及 gate 之前零 optimizer update。
- Final evaluation tests 验证 phase state machine、可信 UTC T0、双时间严格大于、identity 排除、一次 window extension、确定采样、相同环境 hash、50/50 blinding、sealed mapping、59/60 边界、tie 语义、unknown provider outcome 和缺失 item 导致 INVALID。
- 每条“不可协商约束”至少有一个负向 fixture test，证明系统不会因实现缺省而恢复已被否决的旧方案。

## Out of Scope

- 证明真实业务价值、事实正确率、人类偏好、真实任务成功率或线上 KPI。
- 人工盲审、线上 A/B、独立 reality anchor 和生产发布流量切换。
- 使用 GT 业务评测数据训练 policy，或使用 Future EvaluationDataset 继续调参。
- 训练期持续 Sol shadow audit、run 内动态 reward 目标或多次最终投票。
- v1 实现 hierarchical rank、local microgroup rank、merge tournament 或对应 RL semantics。
- Router 自动扩容、账号创建、provider 限额绕过和基于预计延迟的 admission rejection。
- 自动创建第二次最终评测 campaign，或在不足 100 条时临时放松困难 predicate。
- 未由 PromptIdentityNormalizer 明确定义的语义近重复检测。
- 修复 CFS 产品本身；系统只探测并要求 publish-if-absent、可见性和容量能力。
- 自动制定永久 trace 的法规、ACL、加密、删除权、容量或 incident policy。
- 从黑盒多变量实验推断单因素因果关系。

## Further Notes

- 以下 production 输入仍是显式 blocker，不能由实现者自行猜测：DataSourceSkill 与真实表/字段；Sol/Luna/外部 generator model slug 和 inference config；Initial Eval Rubric、BaseJudgePrompt 与 promotion wrapper；AlignmentPolicy、RewardSchema 和 Scalarizer 数值；CFS root、跨 client publish-if-absent 与可见性参数；RouterCapacityConfig、deadline/lease/retry；Harness whitelist 与审计 schema；jobbuilder/image/mount/queue；Monitoring/Cohort/Promotion policy；AutonomyBudget；35B/122B ExperimentSpec；永久 trace 治理批准；Final Evaluation 数据窗与 provider idempotency 能力。
- 缺少这些输入不阻塞 fixture 实现和 ticket 验收，但对应 production phase 的 ReadinessReport 必须明确 BLOCKED。
- 首版规模固定为 100 Trace、每 Trace 32 fit Trajectory；训练示例仍可使用每 step 16 Prompt×128 rollout，但具体 batch 只能由冻结 ExperimentSpec 决定，不能成为隐藏常量。
- 建议将目标/数据用途、artifact identity、Judge certification、reward semantics、CFS publication、Harness、autonomy budget、monitoring、122B gate 与 final evaluation 分别沉淀为 ADR；ADR 不得覆盖本文已冻结的验收行为，除非产生新规格版本。
- 本文仅定义与测试系统，不授权真实拉数、真实 scorer 调用、集群提交、停训、生产发布或 issue tracker 发布。
