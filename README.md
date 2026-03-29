# Qwen2.5 RAG 与 Agent 智能体系统
基于 Qwen2.5 开源大语言模型构建的本地部署项目，实现 RAG 检索问答、Agent 工具调用、vLLM 加速推理与 Web 可视化交互。

## 功能特性
- 本地 Qwen2.5 模型加载与推理
- vLLM 高速推理加速
- RAG 文档检索增强问答
- Agent 智能体（天气查询工具）
- FastAPI + Streamlit 前后端交互界面
- 支持流式输出与对话历史

## 环境配置
```bash
# 创建虚拟环境
conda create -n rag python=3.11
conda activate rag

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
快速启动
基础推理
bash
运行
python basic_infer.py
vLLM 加速服务
bash
运行
python -m vllm.entrypoints.openai.api_server --model 你的模型路径 --port 8000
Web 聊天机器人
bash
运行
python fastapi.py
streamlit run streamlit.py
RAG 文档问答
bash
运行
python rag_qa.py
Agent 天气查询
bash
运行
python agent_weather.py
项目结构
plaintext
├── model/            # 模型存放目录
├── basic_infer.py    # 基础推理
├── vllm_infer.py     # 加速推理
├── fastapi.py        # 后端服务
├── streamlit.py      # 前端界面
├── rag_qa.py         # RAG 问答
├── agent_weather.py  # Agent 智能体
├── requirements.txt  # 依赖文件
└── README.md
核心技术
RAG：检索增强生成，提升回答准确性
Agent：自主决策 + 外部工具调用
vLLM：高效推理引擎
Embedding：文本向量化与向量检索
