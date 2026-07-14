# 01 — 关闭一条 profiled synthetic Trace

**What to build:** 用一条 synthetic、pre-certified Trace 打通最窄 fixture walking skeleton，产生 canonical artifacts、一次 reward、不可变事件链和 `RunClosed`；同一输入切到 production 时在首个副作用前报告未就绪。

**Blocked by:** None

**Status:** ready-for-agent

- [ ] Fixture 输入只经过 fake adapters 到达成功终态，并产出 evidence-linked reward、RunRecord 和 DecisionOutcome。
- [ ] 在任一已提交事件后重启都从持久化状态继续，已完成 action 不重复；fake scorer/cluster error 会关闭为可审计失败终态，而不是只抛进程内异常。
- [ ] Artifact 使用 canonical UTF-8 JSON 与 SHA-256，跨进程相同输入得到相同 bytes/hash；未知 major schema 和损坏 payload fail closed。
- [ ] 只有当前 fencing epoch 的 controller 能按 sequence/previous-hash 追加事件和关闭 run；stale writer 被拒，terminal 后 Observation 被隔离。
- [ ] Fixture 配置不能携带生产凭据或构造 production adapter；placeholder production 配置返回 phase ReadinessReport 且零真实副作用。
- [ ] 本票不建立全量 schema registry、五阶段规则或通用 workflow engine；后续对象随对应 slice 增量加入。
