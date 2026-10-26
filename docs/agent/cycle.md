# 持续推进与证据闭环

会话围绕用户目标；一轮是一次可证伪实验；交付点是可运行、可验收的增量。一次会话可连续多轮，不能把单变量实验误解成只许改一处就交接。

## 0. preflight

1. `python bench/tools/state_check.py` 回 0；非 0 先诊断并在授权内修复，不带红灯执行。
2. 按根入口选择用户指定任务、本人在途任务或队首；队列任务读 `--brief <id>`。领取和续领见 `docs/agent/resources.md`。
3. 看 `git status --short`，区分本任务、他人改动和已有半成品；保留他人工作，不为清洁工作区删除/覆盖它。
4. 重执行前核对资源：所需资源空闲或由本人当前任务持有；计时另满足平衡电源 + AC、RAM 预算、无其它推理/构建。读代码、写文档不等 GPU 空闲。

## 1. 连续闭环

1. **定出口**：把 `done_when` 对应到命令、输入、预期输出、门限与证据。分清局部实现完成和功能验收；缺失测试先补，不能把未知写成 n-a。
2. **读链路**：读设计相关节、完整函数及调用链、实际 CLI/模型入口、独立参考和关联失败 evidence。必读清单是最低要求，不是研究边界。
3. **立假设**：首次排错就记录现象、候选根因、区分性实验与预期；优先找最早偏差。契约缺失先落实维序、数值语义、状态/资源所有权；细项见 `skills/code-review/SKILL.md`。
4. **小实验**：一次隔离一个解释变量，先证明反例失败，再作最小修改。短输入与阶段探针可定位，必须标诊断，不替代固定验收档位；不存在的检查器在本任务内补齐。
5. **跑真实链**：构建本次源码；依次跑受影响局部检查、消费者串联、真实入口和边界回归。失败就返回定位，不把只有编译/格式/局部 PASS 的代码交付。先正确性，后性能。
6. **判证据**：按 `docs/agent/gates.md` 比实际值与权威阈值；不满足或未运行都不能 pass。正式性能用统一脚本、预热、r >= 3；诊断/确定性回归报真实次数、输入与退出码，不从单次结果推性能。
7. **存检查点**：新增读数进 `RESULTS.md`，判决/排除项进 `DECISIONS.md`，必要时同步状态；无新数字不造空报告。相关多轮可汇成一个证据节，原始产物落 `bench/raw/`，不另建进度文档。
8. **验收并继续**：逐项核对出口，全部满足才 `done`。未满足保持 `doing`；确需暂停则 `partial/blocked` 并留证据、下一条命令与判据。同步队列与 STATE，校验回 0；有授权才提交。宽泛推进请求继续相关已就绪任务，不能以"下一步建议"收尾。

转交验收项必须有承接任务、依赖关系和消费者阻断条件；不能以拆任务抹掉未验收项，也不能为绕过依赖造假 `done`。

## 2. 检查点报告

只在有结果/结论时记录，不为每次阅读写仪式报告；对用户的更新保持简短。

```
task    : <id or explicit request> | <commit or uncommitted>
change  : <hypothesis, change, expected discriminator>
reads   : <actual documents and reference implementation>
scope   : <input/config; diagnostics or acceptance>
evidence: <command, exit code, raw artifact, repeat count>
result  : <measured value vs threshold; coverage; pass/fail/unverified>
decision: <adopt/reject/partial and why>
next    : <next command and criterion; or all acceptance items met>
blocker : <none or evidence, affected work, resume condition>
```

正式跑数另附 `bench/README.md` 的完整口径与必报列、中位/极差及门禁判决。`reads`、证据真值与验收覆盖由 agent 核实，校验器不验证实际阅读或代码正确性。

## 3. 引用

新证据节标题下写 `<!-- slug: task-discriminator -->`，引用 `RESULTS.md#<slug>` 或 `DECISIONS.md#<slug>`。节号仅显示；不改历史标题/数字。lint 对重复 slug 报错、悬空引用告警；新增引用的告警须处理。
