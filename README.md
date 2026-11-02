# EcomCare Agent — 电商智能客服 Agent

An AI-powered e-commerce customer service agent built with LangGraph, RAG, and Streamlit.

一个基于大语言模型、LangGraph 和 RAG 的电商智能客服 Agent 原型，支持订单查询、售后政策检索和多轮对话。

## 项目介绍

EcomCare Agent 是一个面向电商售后场景的智能客服原型项目。

项目通过大语言模型理解用户问题，并结合 LangGraph、SQLite 订单查询工具和 Chroma 向量知识库，为用户提供订单信息查询及售后政策解答。

本项目以个人独立开发的方式完成，使用模拟订单数据和本地知识库进行功能演示，不连接真实电商平台。

## 核心功能

- **智能对话**：基于大语言模型理解用户问题并生成自然语言回复。
- **订单查询**：通过 SQLite 查询订单状态、商品信息、物流公司、物流单号和退款状态。
- **知识库检索（RAG）**：使用 Chroma 向量数据库检索售后政策知识。
- **Agent 工具调用**：通过 LangGraph 组织 Agent 工作流程，调用订单查询和知识检索工具。
- **多轮对话**：支持结合前文订单信息继续回答相关问题。
- **隐私保护**：订单查询工具仅返回业务所需字段，避免向 Agent 暴露不必要的个人信息。
- **业务边界控制**：不编造订单状态、物流轨迹、退款进度或售后政策。

## 技术栈

| 技术 | 用途 |
|---|---|
| Python | 后端开发语言 |
| Streamlit | 聊天交互界面 |
| LangGraph | Agent 工作流程与工具调用 |
| Alibaba Cloud Bailian / Qwen | 大语言模型服务 |
| OpenAI-compatible API | 模型接口调用 |
| Chroma | 向量数据库 |
| SQLite | 模拟订单数据存储 |
| RAG | 售后知识检索增强生成 |

## 项目结构

```text
ecomcare-agent/
│
├── app.py                 # Streamlit 聊天应用入口
├── agent_graph.py         # LangGraph Agent 核心逻辑
├── order_tools.py         # SQLite 订单查询工具
├── vector_tools.py        # Chroma 知识检索工具
├── knowledge_tools.py     # 知识库相关工具
├── build_vector_db.py     # 构建 Chroma 向量数据库
├── init_db.py             # 初始化订单数据库
│
├── knowledge/
│   └── after_sales.txt    # 售后政策知识文档
│
├── .env.example           # 环境变量配置示例
├── .gitignore             # Git 忽略规则
└── README.md              # 项目说明文档
```

本地运行时还会生成 `orders.db` 和 `chroma_db/` 等数据文件，这些文件不需要提交到 GitHub。

## 环境要求

- Windows 10 / 11
- Python 3.13
- VS Code
- Git
- 阿里云百炼 API Key

## 安装与运行

### 1. 克隆项目

```bash
git clone https://github.com/dshrgliuea/ecomcare-agent.git
cd ecomcare-agent
```

### 2. 创建 Python 虚拟环境

```powershell
python -m venv .venv
```

### 3. 激活虚拟环境

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止脚本运行，可以使用虚拟环境中的 Python 直接执行后续命令。

### 4. 安装依赖

```powershell
python -m pip install streamlit python-dotenv openai langgraph langchain-openai chromadb
```

如果项目代码使用了其他依赖，请根据实际报错补充安装。

### 5. 配置环境变量

复制 `.env.example`，创建项目根目录下的 `.env` 文件。

填写自己的阿里云百炼配置：

```dotenv
DASHSCOPE_API_KEY=your_api_key_here
DASHSCOPE_MODEL=qwen-plus
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

注意：

- 将 `your_api_key_here` 替换为自己的 API Key。
- 不要将真实 `.env` 文件或 API Key 上传到 GitHub。
- `.env.example` 仅用于展示配置格式。

### 6. 初始化订单数据库

```powershell
python init_db.py
```

### 7. 构建向量知识库

```powershell
python build_vector_db.py
```

### 8. 启动项目

```powershell
python -m streamlit run app.py
```

启动后，根据终端显示的本地地址，在浏览器中打开聊天界面。

## 功能演示

项目使用模拟订单数据进行测试，例如：

| 订单号 | 商品 | 订单状态 |
|---|---|---|
| 10001 | 蓝色牛仔裤 | 已发货 |
| 10002 | 白色运动鞋 | 待发货 |
| 10003 | 演示商品 | 已签收 |

可以尝试以下问题：

**订单查询**
- 查询订单 10001。
- 订单 10001 使用什么物流？
- 这个订单现在是什么状态？

**售后政策**
- 买了五天的衣服可以退货吗？
- 质量问题退货由谁承担运费？
- 申请退款后多久到账？

**异常与边界测试**
- 查询订单 99999。
- 告诉我订单 10001 的客户姓名。
- 订单还没发货，可以申请七天无理由退货吗？

## 当前项目限制

本项目目前是用于学习和功能演示的原型，存在以下限制：

- 订单数据为本地 SQLite 模拟数据。
- 尚未接入真实电商平台订单系统。
- 尚未接入物流公司的实时轨迹 API。
- 尚不能实际提交退款、退货或售后申请。
- 知识库内容为项目内的演示售后政策。
- 尚未实现完整的用户身份认证、角色权限和生产级安全体系。

## 后续优化方向

- 接入真实电商平台订单 API。
- 接入物流实时查询接口。
- 增加售后工单创建与管理功能。
- 增加用户身份认证和权限管理。
- 增加自动化测试、日志和监控。
- 完善部署流程，支持生产环境运行。

## 项目说明

本项目为个人学习与技术实践项目，重点探索大语言模型、Agent 工作流、工具调用和 RAG 在电商智能客服场景中的应用。

GitHub: https://github.com/dshrgliuea/ecomcare-agent