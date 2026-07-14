# 15 — Judge 工具经真实 grouped route 执行

**What to build:** 在 Trace-128 ScoringSession 中同时演示获批本地 helper 和被注入的越权 action，保留合规 scoring trace。

**Blocked by:** 14

**Status:** ready-for-agent

- [ ] 获批 Python/JavaScript/regex helper 只能读取当前 UID/step 的已脱敏 batch 和 ephemeral scratch，并受 CPU、内存、时间、输出限制。
- [ ] Helper 结果可参与有效 reward；工具使用本身不使 Judge result 作废。
- [ ] 凭据、网络、未授权 mount、其他 UID/step 或 raw payload 请求被拒且零副作用，plain text 仍保持惰性。
- [ ] Tool proposal、HarnessDecision、输出、reward 和 Codex trace 压缩、内容寻址并带 hash。
- [ ] 缺 sandbox/whitelist 或永久 trace 的 ACL、加密、容量、删除权、incident approval 时 production readiness BLOCKED。
