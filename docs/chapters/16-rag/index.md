# 第 16 章 · 检索增强生成（RAG） {#retrieval-augmented-generation-rag}

<p>检索增强生成（Retrieval-Augmented Generation, RAG） <a class="book-citation" id="cite-lewis2020retrieval-1" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a> 已成为在生产环境中部署大型语言模型时最具实际影响力的技术之一。RAG 不再仅依赖训练时编码进模型权重中的知识，而是为 LLM 配备一个动态、可更新的外部记忆——使其能够在广泛的知识密集型任务中给出准确、有据可查且可验证的响应。</p>

## 本章内容

<ul class="chapter-outline">
<li id="motivation-and-problem-statement"><a href="section-01.md">16.1　动机与问题陈述</a></li>
<li id="core-rag-architecture"><a href="section-02.md">16.2　核心 RAG 架构</a></li>
<li id="retrieval-methods"><a href="section-03.md">16.3　检索方法</a></li>
<li id="sec:chunking"><a href="section-04.md">16.4　分块策略</a></li>
<li id="advanced-rag-patterns"><a href="section-05.md">16.5　高级 RAG 模式</a></li>
<li id="efficient-rag-decoding-refrag"><a href="section-06.md">16.6　高效 RAG 解码：REFRAG</a></li>
<li id="sec:agentic_rag"><a href="section-07.md">16.7　智能体 RAG</a></li>
<li id="evaluation-dup3"><a href="section-08.md">16.8　评测</a></li>
<li id="production-considerations"><a href="section-09.md">16.9　生产考量</a></li>
<li id="rag-fine-tuning-synergy"><a href="section-10.md">16.10　RAG 与微调的协同</a></li>
<li id="comprehensive-rag-approach-comparison"><a href="section-11.md">16.11　各种 RAG 方法的综合对比</a></li>
</ul>
