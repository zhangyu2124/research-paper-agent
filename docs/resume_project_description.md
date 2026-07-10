# 简历项目描述

## 项目名称

基于 LangGraph 的本地论文库 RAG/Agent 检索系统

## 技术栈

Python, LangGraph, LangChain, Next.js, TypeScript, DeepSeek API, GTE multilingual, sentence-transformers, pypdf, NumPy

## 简历版描述

基于开源 Chat LangChain 项目进行二次开发，构建面向本地科研论文库的 Research Paper Agent。系统支持 PDF 自动入库、论文元数据抽取、正文切片、GTE multilingual 本地向量检索、LangGraph Agent 工具调用和带证据溯源的论文问答，可通过 Next.js 前端进行交互式检索与分析。

## 简历要点

- 改造 Chat LangChain 的 LangGraph Agent 架构，将原文档问答场景迁移为本地论文库 RAG/Agent 检索场景。
- 实现 PDF 入库流水线，使用 pypdf 抽取论文正文并按页切分 chunk，保留 `paper_id`、`chunk_id`、页码和来源文件等证据信息。
- 接入本地 `Alibaba-NLP/gte-multilingual-base` embedding 模型，构建 768 维向量索引，支持中文问题对英文论文内容进行语义检索。
- 设计多工具 Agent，包括论文元信息检索、向量 chunk 检索、关键词 fallback、chunk 上下文读取和论文列表查询。
- 优化 Agent Prompt，使回答基于工具结果生成，并在涉及具体论文内容时返回 chunk_id、页码、相似度分数和证据片段。
- 改造 Next.js 前端界面，将原 Chat LangChain 文案与工具状态替换为论文检索助手交互体验。

## 一句话版本

基于 LangGraph / LangChain 构建本地论文库 RAG Agent，实现 PDF 入库、正文切片、本地 GTE 向量检索和带证据溯源的论文问答系统。

## 面试 30 秒讲法

这个项目是我基于 Chat LangChain 改造的本地论文库 RAG Agent。它的核心流程是把本地 PDF 论文解析成元数据和正文 chunk，再用 GTE multilingual 在本地构建向量索引。用户在前端用中文提问时，LangGraph Agent 会根据问题调用论文检索、向量检索或 chunk 上下文工具，最后由大模型基于检索证据生成回答，并返回 chunk_id、页码和相似度，方便做论文阅读和相关工作分析。

## 面试 2 分钟讲法

我做这个项目的动机是把自己平时阅读的大模型、RAG、Agent 方向论文整理成本地可检索的知识库。原始项目 Chat LangChain 是面向 LangChain 文档问答的，我保留了它的 LangGraph Agent 和 Next.js 前端框架，但把后端工具、Prompt 和数据流程重构成论文检索场景。

数据侧，我写了三个脚本：第一个从 PDF 里抽取论文级元数据，第二个按页抽取正文并切成 chunk，第三个使用 GTE multilingual embedding 模型把所有 chunk 编码成 768 维向量，保存成本地向量索引。这样做的好处是模型和索引都在本地，适合个人论文库，不依赖在线 embedding API。

Agent 侧，我设计了多个工具：`search_papers` 用于论文级粗召回，`search_paper_vectors` 用于语义向量检索，`search_paper_chunks` 用于关键词 fallback，`get_paper_chunk_context` 用于读取 chunk 前后文。Prompt 里要求 Agent 对研究类问题先调用工具，再基于工具结果回答，并在回答中给出 chunk_id、页码和相似度。

目前这个项目可以支持中文问题检索英文论文，比如询问 RAPTOR 的 tree traversal retrieval，系统可以召回对应 PDF 页面的英文片段，再由大模型总结机制并给出证据来源。后续我计划加 reranker、检索评测集和前端引用来源卡片，进一步提升可解释性和检索质量。
