# RAG LangChain Japanese

基于 LangChain + RAG 的个人日语面试练习老师。

第一阶段目标：先做一个本地 CLI 原型，让 AI 能读取自己的 Markdown 日语笔记，并基于笔记回答单词、语法、会话、面试、主题讨论相关问题。

## 项目目录

```text
src/rag_japanese_teacher/   # Python 源码
  core/                     # 配置、模型 provider
  phase0/                   # 原始输入整理
  knowledge/                # notes 文档加载
  rag/                      # RAG 索引和问答
  cli/                      # 命令行入口
raw_inputs/                 # Phase 0 原始输入资料
notes/                      # RAG 正式知识库
docs/                       # 阶段说明文档
```

`raw_inputs/` 和 `notes/` 的区别：

- `raw_inputs/`: 放还没整理的上课笔记、面试总结。
- `notes/`: 放已经整理好的 Markdown 知识点，Phase 1 会索引这里。

代码结构说明见：

[docs/code-architecture.md](docs/code-architecture.md)

上传到 GitHub 的步骤见：

[docs/github-upload.md](docs/github-upload.md)

## 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

之后每次打开新终端，都需要先进入项目目录并激活虚拟环境：

```bash
cd /Users/luyang/Documents/project/RAG_Langchain_Japanese
source .venv/bin/activate
```

激活成功后，终端前面应该能看到 `(.venv)`。这时可以直接使用：

```bash
jp-teacher --help
```

如果不想激活虚拟环境，也可以直接使用完整路径：

```bash
.venv/bin/jp-teacher --help
```

## 2. 配置环境变量

```bash
cp .env.example .env
```

默认使用最省钱的本地 Ollama 方案，不需要 OpenAI API key。

`.env` 默认配置：

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:14b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
```

如果你后续想切回 OpenAI，再把 `LLM_PROVIDER` 改成 `openai` 并填写 `OPENAI_API_KEY`。

### 本地模型选择

当前机器是 M1 Pro / 32GB 内存，推荐默认：

```bash
OLLAMA_MODEL=qwen3:14b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
```

选择理由：

- Qwen3 对中文、日文、英文都比较友好，适合日语学习和面试表达。
- `qwen3:14b` 比 4B/8B 更稳，回答自然度和纠错能力更好。
- `qwen3-embedding:4b` 比轻量 embedding 更适合中文、日文、英文混合检索。

如果速度太慢，降级为：

```bash
OLLAMA_MODEL=qwen3:8b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
```

如果想追求更高质量，并且可以接受更慢速度，可以尝试：

```bash
OLLAMA_MODEL=qwen3:30b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:8b
```

## 3. 安装和准备 Ollama

先安装 Ollama，并确认 Ollama 服务正在运行。

如果命令行提示没有连接到 Ollama，可以先运行：

```bash
ollama serve
```

然后另开一个终端拉取本地模型：

```bash
ollama pull qwen3:14b
ollama pull qwen3-embedding:4b
```

推荐先用小模型，省钱、轻量、容易跑起来。质量不够时再升级到更大的模型。

## 4. 建立向量索引

如果原始资料还没有整理到 `notes/`，先单独运行 Phase 0：

```bash
source .venv/bin/activate
jp-teacher phase0 scan
jp-teacher phase0 build
```

如果你当前终端还在 `(base)` 环境，也可以直接运行：

```bash
.venv/bin/jp-teacher phase0 scan
.venv/bin/jp-teacher phase0 build
```

`phase0 build` 会调用本地模型处理原始资料。运行时会显示当前正在处理的文件、生成的草稿数量和写入路径；如果文档较长，本地模型可能需要等待一段时间。

Phase 0 会先把长文档拆成多个 chunk 再交给模型处理。可以用下面命令查看每个原始文件会被拆成几块：

```bash
jp-teacher phase0 scan
```

如果之前已经生成过少量 `notes/` 文件，重新 build 时默认不会覆盖同名文件。想重新生成同名文件时使用：

```bash
jp-teacher phase0 build --overwrite
```

Phase 0 会读取：

```text
raw_inputs/class_notes/
raw_inputs/interview_summaries/
```

并生成正式知识库文件到：

```text
notes/
```

然后再建立向量索引：

```bash
jp-teacher ingest
```

`ingest` 会显示当前进度：

```text
Loading Markdown notes from notes
Loaded 114 Markdown note(s)
Creating embedding model
Embedding notes and writing Chroma index to .chroma
Indexed 114 Markdown note(s)
```

如果停在 `Embedding notes...`，通常是在调用本地 Ollama embedding 模型生成向量。笔记越多，等待越久。

或者：

```bash
python -m rag_japanese_teacher.cli.main ingest
```

## 5. 提问

```bash
jp-teacher ask "「浸透」を使って面接で使える表現を作って" --mode vocabulary
```

也可以进入交互模式：

```bash
jp-teacher ask --mode interview
```

## 6. 学习模式

支持的模式：

- `vocabulary`: 单词复习
- `grammar`: 语法分析
- `conversation`: 高频会话
- `interview`: 面试练习
- `theme`: 主题讨论
- `general`: 综合问答
