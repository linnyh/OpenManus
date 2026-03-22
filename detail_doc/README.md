# OpenManus 项目详细技术文档

> **项目概述**：OpenManus 是一个开源的通用 AI Agent 框架，基于 MetaGPT 团队开发，能够实现类似 Manus 的通用任务处理能力，无需邀请码即可使用。

## 目录结构

```
detail_doc/
├── README.md                           # 项目总览文档
├── 01_项目概述与架构.md                # 项目简介、核心特性、架构设计
├── 02_Agent系统详解.md                  # Agent核心架构、类层次结构、执行流程
├── 03_工具系统详解.md                  # 工具基类、工具集合、工具实现
├── 04_LLM系统详解.md                   # LLM客户端、多后端支持、Token管理
├── 05_MCP协议详解.md                   # MCP客户端、服务端实现
├── 06_A2A协议详解.md                   # Agent间通信协议
├── 07_Flow工作流系统.md                # PlanningFlow、多Agent协作
├── 08_沙箱系统详解.md                  # Docker沙箱、安全执行环境
├── 09_配置系统详解.md                  # 配置管理、环境变量
├── 10_数据模型详解.md                  # 核心数据结构、Schema定义
├── 11_浏览器自动化详解.md              # BrowserUseTool、多引擎搜索
├── 12_入口点与使用.md                  # 主程序入口、使用示例
└── 13_依赖与技术栈.md                  # 技术栈、依赖管理
```

## 核心模块概览

| 模块 | 路径 | 功能描述 |
|------|------|---------|
| Agent系统 | `app/agent/` | Agent核心实现、状态管理、执行循环 |
| 工具系统 | `app/tool/` | 工具基类、工具集合、浏览器/代码执行等工具 |
| LLM系统 | `app/llm.py` | 大语言模型客户端封装 |
| Flow系统 | `app/flow/` | 多Agent协作工作流 |
| MCP协议 | `app/mcp/` | Model Context Protocol实现 |
| A2A协议 | `protocol/a2a/` | Agent间通信协议 |
| 沙箱系统 | `app/sandbox/` | Docker隔离执行环境 |
| 配置系统 | `app/config.py` | 全局配置管理 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 复制并配置配置文件
cp config/config.example.toml config/config.toml

# 运行 Manus Agent
python main.py

# 运行 Flow 工作流
python run_flow.py

# 启动 MCP Server
python run_mcp_server.py
```

## 核心设计原则

1. **模块化设计**：Agent、Tool、Flow 完全解耦
2. **异步优先**：全程 async/await 异步编程
3. **协议支持**：完整的 MCP 和 A2A 协议实现
4. **安全隔离**：Docker 沙箱支持代码安全执行
5. **灵活配置**：支持多种 LLM 后端和部署方式

---

*文档生成时间：2026-03-22*
