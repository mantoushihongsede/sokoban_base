# Sokoban Agent Prototype

一个面向 GitHub 展示的精简版项目包，用来展示我在 Agent / LLM 方向上的三类能力：

- 可验证环境建模：把 Sokoban 做成结构化、可回放、可自动打分的推理环境
- 工具调用设计：把可达性、合法推箱、死锁分析等能力封装成可供模型调用的工具
- 自动评测链路：把任务生成、模型调用、结果解析、评分与汇总串成一条批量实验流程

这不是一个完整产品，也不是成熟 benchmark。它更准确的定位是：

> 一个基于 Sokoban 的 LLM / Agent 评测与 tool-calling 原型。

## 这个包里有什么

- `core/`
  Sokoban 环境、状态、动作与序列化
- `analyzers/`
  可达性分析、合法推箱枚举、死锁检测、状态解释、tool schema / dispatch
- `tasks/`
  离线评测任务、任务生成与结果汇总
- `evaluation/`
  批量运行器
- `llm_protocol/`
  多 provider client 适配与 prompt / parser
- `scripts/run_all_tasks_batch.py`
  批量离线评测入口
- `scripts/main.py`
  多轮 function-calling demo 原型入口
- `datasets/`
  便于快速演示的小规模样例数据

## 适合如何理解这个项目

如果你是第一次看这个仓库，可以把它理解成一个“给 Agent 做可验证实验”的小型工作台：

1. 用 Sokoban 提供明确规则和可验证状态
2. 把环境能力封装成工具，供模型按需调用
3. 用结构化任务和自动评分来评估模型推理行为
4. 用多轮工具调用原型验证“观察 -> 调工具 -> 选动作 -> 推进状态”的闭环

## 当前最值得看的能力

### 1. 可验证环境与分析工具

- 动作合法性判断
- 状态转移预测
- 静态死格识别
- 严格死锁检测
- 单箱状态解释


### 2. 自动评测链路

项目支持把 Sokoban 状态组织成统一任务，再调用模型输出 JSON，最后做自动评分与汇总。

当前保留的核心任务包括：

- `action_legality`
- `state_transition`
- `deadlock_detection`
- `static_dead_squares`
- `box_status_explanation`
- `box_priority_ranking`
- `box_target_assignment`

### 3. Tool Calling 原型

`scripts/main.py` 展示了多轮 function-calling 的实验方向：

- 模型读取当前状态
- 调用分析工具
- 选择一个推箱动作
- 推进到下一个状态

这部分目前仍然是原型。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或：

```bash
pip install -e .
```

### 2. 配置环境变量

复制 `env/` 里的样例文件并改成真实配置：

- `env/.env.deepseek.example` -> `env/.env.deepseek`
- `env/.env.openai.example` -> `env/.env.openai`
- `env/.env.gemini.example` -> `env/.env.gemini`
- `env/.env.claude.example` -> `env/.env.claude`

当前建议：

- 正式批量评测优先用 DeepSeek 官方接口
- 非 DeepSeek 的 provider 可以通过 OpenRouter 配置

## 运行方式

### 1. 运行离线评测

```bash
python scripts/run_all_tasks_batch.py
```

这个脚本会：

1. 从 `datasets/` 读取样例
2. 生成评测任务
3. 调用模型
4. 把 summary 和详细结果写入 `outputs/`

当前脚本默认使用较小样本，主要用于验证链路能否跑通。

### 2. 运行 tool-calling demo 原型

```bash
python scripts/main.py
```

这个入口使用一个内置的小关卡做多轮动作选择实验，用作展示项目方向。
