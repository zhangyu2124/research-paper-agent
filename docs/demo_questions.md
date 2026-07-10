# Demo 问题清单

下面这些问题适合用于本地演示、README 截图和面试讲解。

## 1. 单篇论文机制解释

```text
请使用语义向量检索，说明 RAPTOR 的 tree traversal retrieval 是怎么工作的，并给出 chunk_id、页码和相似度。
```

期望效果：

- 能召回 RAPTOR 论文。
- 能说明 tree traversal 从根节点逐层向下检索。
- 能给出 chunk_id、页码和 similarity score。

## 2. 多篇论文对比

```text
对比 RAPTOR、LongAgent、GraphReader 在长文档处理上的区别，并引用 PDF 正文证据。
```

期望效果：

- 能区分 tree-organized retrieval、multi-agent long context、graph-based reading。
- 每篇论文至少引用一个证据来源。

## 3. 研究方向检索

```text
检索和 long document summarization agent 相关的论文，列出 paper_id、标题和相关原因。
```

期望效果：

- 先返回相关论文列表。
- 说明每篇论文与 long document summarization / agent 的关系。

## 4. Agent workflow 方向

```text
论文库里有哪些和 multi-agent 或 agent workflow 相关的论文？
```

期望效果：

- 能找到 AFlow、A2Flow、AgentSwift、LongAgent 等相关论文。
- 能按研究主题做简单归类。

## 5. 论文阅读建议

```text
如果我要做一个论文检索 Agent 项目，当前论文库中哪些论文最值得优先读？请按阅读顺序推荐。
```

期望效果：

- 不只是列论文，而是给出阅读顺序。
- 能说明每篇论文对项目设计的参考价值。

## 6. 精确证据追问

```text
请找出 RAPTOR 论文中描述 collapsed tree retrieval 的正文片段，给出 chunk_id、页码和原文摘要。
```

期望效果：

- 使用向量检索或关键词检索定位 collapsed tree retrieval。
- 返回可追溯证据。

## 7. 方法局限分析

```text
结合论文库中的证据，分析长文档 RAG 或长文档 Agent 方法可能有哪些局限。
```

期望效果：

- 能综合多篇论文。
- 能从检索证据中抽象出局限，例如上下文长度、检索噪声、层级摘要损失、工具调用成本等。
