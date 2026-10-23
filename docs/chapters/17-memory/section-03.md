# 17.3　记忆架构 {#sec:memory-architectures}

## 17.3.1　基于 RAG 的记忆 {#sec:rag-memory}

<p>检索增强生成（Retrieval-Augmented Generation, RAG） <a class="book-citation" id="cite-lewis2020retrieval-1" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a> 是 LLM Agent 外部记忆的主流范式。记忆存储 <span class="arithmatex">\(\mathcal{M}\)</span> 是文档集合 <span class="arithmatex">\(\{d_i\}_{i=1}^N\)</span>；检索把查询 <span class="arithmatex">\(q\)</span> 映射到一个有序子集。</p>

#### Embedding 存储与向量数据库。 {#embedding-stores-and-vector-databases.}

<p>每个文档 <span class="arithmatex">\(d_i\)</span> 由 Embedding 模型 <span class="arithmatex">\(\phi\)</span> 编码：<span class="arithmatex">\(\mathbf{v}_i = \phi(d_i) \in \mathbb{R}^{D}\)</span>。查询同样被编码：<span class="arithmatex">\(\mathbf{q} = \phi(q)\)</span>。检索按相似度返回 top-<span class="arithmatex">\(k\)</span> 文档：</p>



<div class="arithmatex">\[\text{Retrieve}(q, \mathcal{M}, k) = \underset{S \subseteq [N],\, |S|=k}{\arg\max} \sum_{i \in S} \text{sim}(\mathbf{q}, \mathbf{v}_i),\]</div>

<p>其中 <span class="arithmatex">\(\text{sim}(\cdot,\cdot)\)</span> 通常为余弦相似度。近似最近邻（Approximate Nearest-Neighbor, ANN）索引（FAISS <a class="book-citation" id="cite-johnson2019billion-1" href="../../references.md#ref-johnson2019billion" aria-label="参考文献 171">[171]</a>、HNSW <a class="book-citation" id="cite-malkov2018efficient-1" href="../../references.md#ref-malkov2018efficient" aria-label="参考文献 248">[248]</a>、ScaNN <a class="book-citation" id="cite-guo2020scann-1" href="../../references.md#ref-guo2020scann" aria-label="参考文献 126">[126]</a>）使得 <span class="arithmatex">\(N \sim 10^7\)</span> 量级下仍然可行。</p>

#### 检索策略。 {#retrieval-strategies.}

<ul>
<li><p><strong>稠密检索（Dense retrieval）。</strong> 查询和文档都由神经编码器编码（例如 DPR <a class="book-citation" id="cite-karpukhin2020dense-1" href="../../references.md#ref-karpukhin2020dense" aria-label="参考文献 180">[180]</a>、<code>text-embedding-3-large</code>）。能捕捉语义相似度，但需要 GPU 推理。</p></li>
<li><p><strong>稀疏检索（Sparse retrieval）。</strong> 基于 Token 重叠的 BM25 或 TF-IDF。快速、可解释、在精确关键词匹配上表现强劲。</p></li>
<li><p><strong>混合检索（Hybrid retrieval）。</strong> 通过倒数排名融合（Reciprocal Rank Fusion, RRF）合并稠密与稀疏分数：</p>



<div class="arithmatex">\[\text{RRF}(d, k) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)},\]</div>

<p>其中 <span class="arithmatex">\(k=60\)</span> 是平滑常数。混合检索一致地优于单独使用任一种 <a class="book-citation" id="cite-chen2022hybrid-1" href="../../references.md#ref-chen2022hybrid" aria-label="参考文献 44">[44]</a>。</p></li>
</ul>

#### 重排序（Re-ranking）。 {#re-ranking.}

<p>交叉编码器（cross-encoder）重排序器 <span class="arithmatex">\(f_\psi(q, d) \in [0,1]\)</span> 与查询联合打分每个被检索文档，以 <span class="arithmatex">\(O(k)\)</span> 次前向传播的代价换取更高准确率。整体流水线为：用 ANN 检索 <span class="arithmatex">\(k&#x27; \gg k\)</span> 个候选，再用交叉编码器重排序，返回 top <span class="arithmatex">\(k\)</span>。</p>

<div class="admonition warning"><p class="admonition-title">检索引发的幻觉风险</p>
<p>RAG 并不能消除幻觉——反而可能<em>引入</em>幻觉。如果被检索的文档过时、错误或只是表面相关，模型可能自信地把错误信息纳入回答。务必包含来源元数据（来源、时间戳、置信度），并考虑加入忠实性（faithfulness）验证步骤。</p>
</div>

## 17.3.2　基于摘要的记忆 {#summarization-based-memory}

<p>当原样存储过于昂贵或过于嘈杂时，<em>摘要化</em>在存储之前先压缩信息。</p>

#### 渐进式摘要（Progressive Summarization）。 {#progressive-summarization.}

<p>在每一步 <span class="arithmatex">\(t\)</span>，Agent 维护一个运行中的摘要 <span class="arithmatex">\(S_t\)</span>。当新信息 <span class="arithmatex">\(e_t\)</span> 到达时：</p>



<div class="arithmatex">\[S_{t+1} = \text{LLM}\!\left(\texttt{``Summarize: [}S_t\texttt{] + [}e_t\texttt{]&#x27;&#x27;}\right).\]</div>

<p>这样把记忆规模保持在 <span class="arithmatex">\(O(1)\)</span>，但有丢失细节的风险。</p>

#### 分层压缩（Hierarchical Compression）。 {#hierarchical-compression.}

<p>将记忆组织为层级 <span class="arithmatex">\(L_0 \supset L_1 \supset \cdots \supset L_K\)</span>，其中 <span class="arithmatex">\(L_0\)</span> 是原文，每个 <span class="arithmatex">\(L_{i+1}\)</span> 都是 <span class="arithmatex">\(L_i\)</span> 的摘要。检索首先访问 <span class="arithmatex">\(L_K\)</span>（压缩程度最高、最快），按需逐层深入。这呼应了 Forte <a class="book-citation" id="cite-forte2022building-1" href="../../references.md#ref-forte2022building" aria-label="参考文献 96">[96]</a> 的<em>渐进式摘要</em>技术。</p>

#### 何时摘要、何时原样存储。 {#when-to-summarize-vs.-store-verbatim.}

<ul>
<li><p>原样存储：精确事实、代码片段、数值结果、用户原话。</p></li>
<li><p>摘要存储：叙事性上下文、推理链、冗余观察。</p></li>
<li><p>直接丢弃：噪声、无信息量的失败工具调用。</p></li>
</ul>

## 17.3.3　基于图的记忆 {#graph-based-memory}

#### 知识图谱。 {#knowledge-graphs.}

<p>知识图谱 <span class="arithmatex">\(\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})\)</span> 把事实存储为三元组 <span class="arithmatex">\((h, r, t)\)</span>，其中 <span class="arithmatex">\(h, t \in \mathcal{V}\)</span> 是实体，<span class="arithmatex">\(r \in \mathcal{R}\)</span> 是关系。Agent 可通过 SPARQL <a class="book-citation" id="cite-harris2013sparql-1" href="../../references.md#ref-harris2013sparql" aria-label="参考文献 134">[134]</a>、Cypher <a class="book-citation" id="cite-francis2018cypher-1" href="../../references.md#ref-francis2018cypher" aria-label="参考文献 97">[97]</a> 或自然语言到图的翻译进行查询。</p>

#### 实体-关系抽取。 {#entity-relation-extraction.}

<p>新观察由抽取模型 <span class="arithmatex">\(\text{IE}: \text{text} \to \{(h_i, r_i, t_i)\}\)</span> 解析并合并到 <span class="arithmatex">\(\mathcal{G}\)</span> 中。共指消解（coreference resolution）与实体链接（entity linking）保证一致性。</p>

#### GraphRAG。 {#graphrag.}

<p>GraphRAG <a class="book-citation" id="cite-edge2024local-1" href="../../references.md#ref-edge2024local" aria-label="参考文献 82">[82]</a> 在 RAG 之上叠加图遍历：给定一个查询，先检索种子实体，然后通过 <span class="arithmatex">\(k\)</span> 跳邻域遍历来挖掘 Embedding 相似度未直接匹配到的相关事实。这对多跳推理尤其强大：</p>



<div class="arithmatex">\[\text{GraphRetrieve}(q, \mathcal{G}, k) = \bigcup_{v \in \text{seeds}(q)} \mathcal{N}_k(v, \mathcal{G}),\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{N}_k(v, \mathcal{G})\)</span> 是 <span class="arithmatex">\(v\)</span> 的 <span class="arithmatex">\(k\)</span> 跳邻域。</p>

#### 时序知识图谱。 {#temporal-knowledge-graphs.}

<p>事实具有有效期：<span class="arithmatex">\((h, r, t, [t_\text{start}, t_\text{end}])\)</span>。时序知识图谱 <a class="book-citation" id="cite-lacroix2020tensor-1" href="../../references.md#ref-lacroix2020tensor" aria-label="参考文献 197">[197]</a> 支持类似 “2023 年 OpenAI 的 CEO 是谁？” 的查询，而不会把过去与现在的状态混淆在一起。</p>

## 17.3.4　键值记忆网络（Key-Value Memory Networks） {#key-value-memory-networks}

<p>可微分记忆网络 <a class="book-citation" id="cite-weston2014memory-1" href="../../references.md#ref-weston2014memory" aria-label="参考文献 377">[377]</a><a class="book-citation" id="cite-sukhbaatar2015end-1" href="../../references.md#ref-sukhbaatar2015end" aria-label="参考文献 342">[342]</a> 把记忆表示为一组键值对 <span class="arithmatex">\(\{(\mathbf{k}_i, \mathbf{v}_i)\}_{i=1}^M\)</span>，并通过基于软 Attention 的检索访问：</p>



<div class="arithmatex">\[\alpha_i = \text{softmax}\!\left(\frac{\mathbf{q}^\top \mathbf{k}_i}{\sqrt{D}}\right), \qquad
  \mathbf{c} = \sum_{i=1}^M \alpha_i \mathbf{v}_i.\]</div>

<p>检索得到的上下文 <span class="arithmatex">\(\mathbf{c}\)</span> 是查询的可微函数，从而支持端到端训练。现代 Transformer 的 Attention 就是这种机制的特例。在智能体场景中，记忆槽可以通过梯度下降更新，也可以通过显式的写入操作更新。</p>

## 17.3.5　MemGPT 与虚拟上下文管理 {#sec:memgpt}

<p>MemGPT <a class="book-citation" id="cite-packer2023memgpt-1" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a> 引入了类似操作系统虚拟内存的<em>虚拟上下文</em>抽象。记忆被组织为多层：</p>

#### 页入/页出策略（Page-In / Page-Out）。 {#page-in-page-out-strategies.}

<p>Agent 基于以下因素决定<em>哪些</em>记忆要提升到热上下文（page-in），<em>哪些</em>要逐出（page-out）：</p>

<ul>
<li><p><strong>近期性（Recency）：</strong> 最近访问过的条目更有可能再次被需要。</p></li>
<li><p><strong>相关性（Relevance）：</strong> 与当前查询相似度高的条目。</p></li>
<li><p><strong>重要性（Importance）：</strong> 写入时被标记为高重要性的条目。</p></li>
</ul>

#### 自驱动的记忆管理。 {#self-directed-memory-management.}

<p>在 MemGPT 中，LLM 自身把记忆管理的函数调用（<code>memory_search</code>、<code>memory_insert</code>、<code>memory_delete</code>）作为其动作空间的一部分发出。这使记忆管理成为<em>可学习的行为</em>而非硬编码的 Policy——天然适合作为 RL 训练的目标（见 <a href="section-07.md#sec:rl-memory">17.7</a> 节）。</p>
