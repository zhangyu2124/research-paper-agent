# 项目描述备选稿

> 本文件用于保存项目事实和候选表述，不直接修改正式简历。

## 项目名称

Research Paper Agent：本地论文库混合检索 RAG Agent

## 技术栈

Python、LangGraph、LangChain、RAG、GTE Embedding、BM25、RRF、CrossEncoder、DeepSeek API、NumPy、JSONL

## 简历候选描述

- 面向研究生阅读英文论文时检索效率低、回答缺少出处的问题，基于 Chat LangChain 二次开发本地论文库 RAG Agent，支持论文发现、内容问答、方法对比和相关工作分析。
- 设计 PDF 论文入库流程，完成元数据抽取、按页切片和证据溯源字段设计，构建包含 20 份论文、1569 个 chunk 的本地知识库，每条证据保留 `paper_id`、`chunk_id`、页码和来源文件。
- 使用 `gte-multilingual-base` 进行 Dense Retrieval，使用 BM25 进行 Sparse Retrieval，并通过 RRF 融合两路排名；加入重复 chunk 去重与可选 `bge-reranker-v2-m3` Cross-Encoder 重排。
- 将召回、融合、去重和重排封装为统一证据检索工具，由 LangGraph Agent 通过 LangChain Tool Calling 调用；回答返回检索模式、阶段分数和原文证据，降低模型自行组合检索策略造成的不稳定性。
- 构建 20 条中英文人工标注查询，按论文级结果对 Dense、BM25、Hybrid 和 Hybrid + Reranker 进行消融评测；Hybrid 模式达到 Recall@5 1.00、MRR@10 1.00，平均检索延迟约 62.7 ms，并根据 CPU 延迟测试将 Reranker 设计为可选模式。

## 一句话版本

基于 LangGraph / LangChain 构建本地论文库 RAG Agent，通过 GTE + BM25 + RRF 实现中英文混合检索，并提供可选 Cross-Encoder 重排、证据溯源和论文级检索评测。

## 面试 30 秒讲法

这个项目解决的是本地英文论文难检索、模型回答缺少出处的问题。PDF 入库后会按页切成带论文 ID 和页码的 chunk，查询时并行执行 GTE 向量检索和 BM25 关键词检索，再用 RRF 融合排名并去重。LangGraph Agent 通过 LangChain Tool Calling 调用统一证据检索工具，最后基于原文生成回答。我还加入了可选 Cross-Encoder 重排，并用 20 条人工标注查询做了四种模式对比，最终根据精度和延迟选择 Hybrid 作为默认方案。

## 面试 2 分钟讲法

项目数据侧包含三步。第一步从本地 PDF 抽取论文元数据；第二步按页解析正文并切分 chunk，同时保留 `paper_id`、`chunk_id`、页码和来源文件；第三步使用 `gte-multilingual-base` 生成 768 维向量并保存本地 NumPy 索引。目前语料有 20 份论文和 1569 个 chunk。

检索侧采用两路召回。Dense Retrieval 负责语义匹配和中文问题检索英文正文，BM25 负责论文名、方法名和缩写等精确术语匹配。因为两种分数不在同一量纲，我没有直接做加权求和，而是使用 RRF 根据名次融合。融合后再按标题和正文内容去重，避免重复 PDF 占满结果。系统还可以加载 `bge-reranker-v2-m3`，对查询和候选证据做 Cross-Encoder 打分。

Agent 侧仍是单 Agent 多工具结构。LangGraph 提供 Agent 运行和状态流程，LangChain 提供模型接入、Tool Calling 和工具封装。Agent 不再分别调用向量工具与关键词工具，而是调用统一的 `search_paper_evidence`，由工具内部固定执行 Dense、BM25、RRF 和去重，减少模型选择检索策略带来的不稳定性。

评测侧准备了 20 条中英文人工标注查询，并按论文级结果计算 Recall@5、MRR@10、nDCG@10 和平均延迟。Hybrid 的 Recall@5 和 MRR@10 都是 1.00，平均延迟约 62.7 ms；Reranker 在当前小语料上没有继续提升准确率，CPU 延迟约 18.1 秒，因此默认关闭，只保留为可选高成本模式。这个结果也说明我不仅实现功能，还根据评测结果做了线上策略选择。

## 面试时必须诚实说明

- 这是单 Agent 多工具系统，不是多智能体系统。
- 当前论文库和评测集规模较小，指标只代表当前同域语料，不代表通用基准。
- Reranker 已完成本地接入和实测，但因 CPU 延迟默认关闭。
- NumPy 全量相似度检索适合个人论文库，更大规模可迁移到 FAISS、Qdrant 或 Milvus。
