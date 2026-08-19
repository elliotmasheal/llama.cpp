# 仓库安全与 Harness 边界

## 既有改动

- 开始任务前运行 `git status --short --untracked-files=all`。
- 只修改当前任务声明的文件。当前工作树已有大量用户删除改动，包括上游跟踪的 `AGENTS.md`、`.github/`、`.gemini/` 和 `.devops/`；不得用 `git restore`、checkout 或清理命令擅自恢复、删除或重排它们。
- `.claude/worktrees/` 下的 agent worktree 不属于本任务目标，不要移动、删除或修改。

## 生成物

- 构建目录使用 `build-*` 或 `tmp/` 等已有忽略目录。
- 模型、日志、SPIR-V、benchmark 原始结果和大型二进制不放入源码目录的版本控制范围。
- 修改 shader 或构建逻辑时，检查生成物是否位于 build 目录，不要把生成 header 当作手工源码编辑。

## AI 和外部操作

- 遵守 `CONTRIBUTING.md`，以及工作树中存在时的 `AGENTS.md`。
- 不自动 commit、push、创建 PR、发 issue、发评论或代写 PR 描述、commit message、reviewer 回复。
- AI 生成的实质性贡献按项目要求披露；贡献者必须能解释和维护每一行改动。

## 安全运行

- 未知 GGUF、输入格式和服务端测试按 `SECURITY.md` 要求在隔离环境中运行。
- 不把下载模型、网络服务、RPC 或多租户运行当作默认安全环境。
