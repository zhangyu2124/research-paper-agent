# 面试讲解稿

## 项目定位

这是一个面向个人科研场景的本地论文库 RAG/Agent 系统。它不是普通聊天机器人，而是围绕“论文检索、证据召回、相关工作分析”设计的 Agent 应用。

## 可以强调的核心价值

- 解决本地论文太多、人工查找困难的问题。
- 支持中文问题检索英文论文正文。
- 回答不是凭空生成，而是带有 `chunk_id`、页码和相似度。
- 使用 Agent 工具编排，而不是把所有逻辑写死在一个检索函数里。
- 模型权重和向量索引可以本地化，适合个人知识库场景。

## 系统流程

```text
PDF 论文
-> 元数据抽取
-> 正文切片
-> GTE multilingual embedding
-> 本地向量索引
-> LangGraph Agent 工具调用
-> 带证据回答
```

## 面试官可能问的问题

### 1. 为什么用 Agent，而不是普通 RAG？

普通 RAG 往往是固定流程：用户问题 -> 检索 -> 拼 prompt -> 回答。这个项目使用 Agent 的原因是论文问题类型不固定：有时需要先找相关论文，有时需要直接查正文证据，有时需要读取某个 chunk 的前后文。Agent 可以根据问题选择不同工具，比如先用 `search_papers` 找候选论文，再用 `search_paper_vectors` 找正文证据。

### 2. 为什么同时保留关键词检索和向量检索？

向量检索适合语义匹配，尤其适合中文问题检索英文论文。但关键词检索对精确术语、方法名、论文名更稳定，比如 RAPTOR、LongAgent、GraphReader 这类专有名词。所以系统保留关键词检索作为 fallback，形成 hybrid retrieval 的雏形。

### 3. 为什么选择 GTE multilingual？

论文正文主要是英文，但用户提问可能是中文。GTE multilingual 支持多语言语义对齐，适合“中文 query -> 英文 chunk”的检索场景。另外它可以本地运行，不需要每次调用在线 embedding API。

### 4. chunk 是怎么设计的？

系统先按 PDF 页抽取文本，再在页内按固定长度切 chunk，并保留页码、chunk 序号、论文 ID、标题和来源文件。这样每个检索结果都可以追溯到具体论文和页码，方便验证回答依据。

### 5. 如何减少幻觉？

主要从三方面控制：第一，Prompt 要求研究问题必须先调用工具；第二，回答中必须引用工具结果里的 paper_id、chunk_id、页码和相似度；第三，不允许模型推断工具结果中没有的作者、venue、实验指标或页码。

### 6. 当前系统有什么不足？

当前向量索引用 NumPy 做相似度计算，适合小规模个人论文库；大规模场景应该换 FAISS、Chroma 或 Milvus。系统暂时没有 reranker，复杂问题的证据排序还有优化空间。前端还没有独立的引用卡片，证据主要以文本形式展示。

## Demo 问题

```text
请使用语义向量检索，说明 RAPTOR 的 tree traversal retrieval 是怎么工作的，并给出 chunk_id、页码和相似度。
```

```text
对比 RAPTOR、LongAgent、GraphReader 在长文档处理上的区别，并引用 PDF 正文证据。
```

```text
检索和 long document summarization agent 相关的论文，列出 paper_id、标题和相关原因。
```

```text
论文库里有哪些和 multi-agent 或 agent workflow 相关的论文？
```

## 你可以主动补充的一句话

这个项目目前更像一个个人论文库 RAG/Agent 原型，我重点做了数据入库、向量召回和 Agent 工具编排。下一步我会加 reranker 和检索评测集，把它从能用进一步做成可评估、可优化的检索系统。
