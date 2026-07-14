# Codex 自治 RL 最终交付

本目录是后续实现与验收的唯一入口。

- [spec.md](./spec.md)：最终规格与决策 source of truth。
- [tickets](./tickets)：按 blocker DAG 排列的 32 个 `ready-for-agent` vertical-slice tickets。
- [human_motivation.md](./human_motivation.md)：项目动机、业务背景与人类核心洞察。

## 使用规则

- 实施只从 `tickets/` 中 blocker 已完成的 frontier ticket 开始。
- 规格与 ticket 冲突时以本目录的 `spec.md` 为准。
- 新决策必须先更新最终规格，再同步受影响 tickets；不要直接修改历史版本来改变当前行为。
- Fixture 验收不代表 production readiness。真实数据、scorer、CFS、集群或最终评测只在对应 phase gate green 后执行。
