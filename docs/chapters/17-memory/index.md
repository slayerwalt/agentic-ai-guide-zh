# 第 17 章 · 智能体记忆系统 {#sec:agentic-memory}

## 17.1　动机：为什么 Agent 需要记忆 {#sec:memory-motivation}

<p>大语言模型本质上是无状态的函数近似器：给定 Prompt <span class="arithmatex">\(x\)</span>，它们产生续写分布 <span class="arithmatex">\(p_\theta(y \mid x)\)</span>。每一次推理调用都从零开始。<em>上下文窗口</em>（context window）——模型可以关注的有限 Token 序列——是生成时唯一可用的信息。对短小、自包含的任务而言这已足够；但对长周期的智能体任务而言，这是一个根本性瓶颈。</p>

<div class="admonition note"><p class="admonition-title">上下文窗口瓶颈</p>
<p>设 <span class="arithmatex">\(L\)</span> 为最大上下文长度（例如 GPT-4o 的 <span class="arithmatex">\(L = 128{,}000\)</span> 个 Token）。单个 Token 大约编码 4 个字符；一本典型书籍约含 <span class="arithmatex">\(\sim\!500{,}000\)</span> 词 <span class="arithmatex">\(\approx 670{,}000\)</span> 个 Token。即便不计成本，一个运行数日的自主 Agent 所累积的观察、工具输出和推理轨迹<em>不可能</em>塞进任何固定窗口。记忆系统正是对这一物理约束的工程回应。</p>
</div>

<p>当 Agent 缺乏持久化记忆时，会出现三种截然不同的失败模式：</p>

<ol>
<li><p><strong>上下文的灾难性遗忘（Catastrophic Forgetting）。</strong> 一旦事件滚出上下文窗口便不可挽回地丢失。Agent 无法回溯一万个 Token 前所做的决策。</p></li>
<li><p><strong>无法从经验中学习。</strong> 没有情景式（episodic）存储，每个 Episode 对 Agent 都像是第一次。成功的策略无法复用，错误则反复发生。</p></li>
<li><p><strong>缺乏个性化。</strong> 用户偏好、领域事实、关系历史必须在每次会话中重新建立，导致用户体验与效率双双下降。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">记忆作为认知架构</p>
<p>认知科学在生物智能体中区分出多种记忆系统 <a class="book-citation" id="cite-tulving1985memory-1" href="../../references.md#ref-tulving1985memory" aria-label="参考文献 356">[356]</a><a class="book-citation" id="cite-squire1992declarative-1" href="../../references.md#ref-squire1992declarative" aria-label="参考文献 337">[337]</a>：<em>工作记忆</em>（对信息的主动操控）、<em>情景记忆</em>（自传式事件）、<em>语义记忆</em>（世界知识）以及<em>程序记忆</em>（技能与习惯）。有效的智能体 AI 系统从类似的划分中获益——并非因为我们要模拟神经科学，而是因为这些类别真实地反映了截然不同的<em>访问模式</em>、<em>更新频率</em>和<em>检索机制</em>。</p>
</div>

<p>形式化地，我们把 Agent 建模为一个元组 <span class="arithmatex">\(\mathcal{A} = (\pi_\theta, \mathcal{M}, \mathcal{R}, \mathcal{W})\)</span>，其中 <span class="arithmatex">\(\pi_\theta\)</span> 是 Policy（LLM），<span class="arithmatex">\(\mathcal{M}\)</span> 是记忆存储，<span class="arithmatex">\(\mathcal{R}: \mathcal{Q} \times \mathcal{M} \to \mathcal{D}\)</span> 是把查询映射到检索结果的检索函数，<span class="arithmatex">\(\mathcal{W}: \mathcal{M} \times \mathcal{E} \to \mathcal{M}\)</span> 是用新经验 <span class="arithmatex">\(\mathcal{E}\)</span> 更新记忆的写入函数。在每一步 <span class="arithmatex">\(t\)</span>，Agent 观察到 <span class="arithmatex">\(o_t\)</span>，检索相关上下文 <span class="arithmatex">\(c_t = \mathcal{R}(o_t, \mathcal{M})\)</span>，并采取行动：</p>



<div class="arithmatex">\[a_t \sim \pi_\theta\!\left(\cdot \;\middle|\; [s_t;\, c_t;\, h_t]\right),\]</div>

<p>其中 <span class="arithmatex">\(s_t\)</span> 是当前的系统 Prompt，<span class="arithmatex">\(c_t\)</span> 是检索到的记忆，<span class="arithmatex">\(h_t\)</span> 是近期的上下文历史。行动之后，Agent 可写入新信息：<span class="arithmatex">\(\mathcal{M} \leftarrow \mathcal{W}(\mathcal{M},\, (o_t, a_t, r_t))\)</span>。</p>

## 17.2　记忆类型分类 {#sec:memory-taxonomy}

<figure id="fig:memory-taxonomy"><a href="../../assets/figures/fig_052_memory-taxonomy.png" target="_blank" rel="noopener" aria-label="查看图 17.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_052_memory-taxonomy.png" alt="智能体记忆系统的四分类法，与认知科学的划分相对应。每种记忆类型具有截然不同的访问模式、更新频率和检索机制。" loading="lazy"></a><figcaption><span class="figure-number">图 17.1　</span>智能体记忆系统的四分类法，与认知科学的划分相对应。每种记忆类型具有截然不同的访问模式、更新频率和检索机制。</figcaption></figure>

### 17.2.1　工作记忆（短期） {#working-memory-short-term}

<p>工作记忆是 Agent 的<em>活动工作空间</em>：当前正在被操控的信息。在 LLM Agent 中它对应于：</p>

<ul>
<li><p><strong>草稿板（Scratchpads）。</strong> 在产出最终答案之前写入专用缓冲区的中间推理步骤（例如思维链 Chain-of-Thought <a class="book-citation" id="cite-wei2022chain-1" href="../../references.md#ref-wei2022chain" aria-label="参考文献 374">[374]</a>、scratchpad <a class="book-citation" id="cite-nye2021show-1" href="../../references.md#ref-nye2021show" aria-label="参考文献 272">[272]</a>）。</p></li>
<li><p><strong>思维链缓冲区。</strong> 在答案 Token <span class="arithmatex">\(a\)</span> 之前生成的推理 Token 序列 <span class="arithmatex">\(z_1, z_2, \ldots, z_k\)</span>，建模为 <span class="arithmatex">\(p(a \mid x) = \sum_z p(a \mid x, z)\,p(z \mid x)\)</span>。</p></li>
<li><p><strong>对话上下文。</strong> 保留在上下文窗口中的近期轮次历史 <span class="arithmatex">\([(u_1, a_1), \ldots, (u_t, a_t)]\)</span>。</p></li>
</ul>

<p>工作记忆是<em>快速的</em>（零检索延迟——它已经在上下文中）、<em>易失的</em>（上下文清空时即丢失），且<em>容量受限</em>（受 <span class="arithmatex">\(L\)</span> 约束）。</p>

### 17.2.2　情景记忆（基于经验） {#episodic-memory-experience-based}

<p>情景记忆存储<em>具体的过往事件</em>，并按上下文与时间索引。对 Agent 而言：</p>

<ul>
<li><p><strong>过往交互。</strong> 先前对话、任务尝试及其结果的完整或摘要记录。</p></li>
<li><p><strong>成功轨迹。</strong> 可作为少样本范例被检索、以服务于未来相似任务的高 Reward 动作序列。</p></li>
<li><p><strong>失败案例。</strong> 带有根因标注的错误记录，使 Agent 能够避免重复犯错。</p></li>
<li><p><strong>检索增强的情景回忆。</strong> 给定新任务 <span class="arithmatex">\(q\)</span>，检索最相似的 <span class="arithmatex">\(k\)</span> 个过去 Episode <span class="arithmatex">\(\{e_i\}_{i=1}^k\)</span> 并加入上下文。</p></li>
</ul>

<p>情景记忆通常实现为基于 Episode 摘要 Embedding 的向量存储（见 <a href="#sec:rag-memory">17.3.1</a> 节）。</p>

### 17.2.3　语义记忆（世界知识） {#semantic-memory-world-knowledge}

<p>语义记忆编码与具体 Episode 解耦的<em>一般性事实与概念</em>：</p>

<ul>
<li><p><strong>事实性知识。</strong> 实体、属性与关系（例如 “Paris 是 France 的首都”）。</p></li>
<li><p><strong>领域概念。</strong> 与 Agent 任务领域相关的定义、分类法与本体。</p></li>
<li><p><strong>知识图谱。</strong> 结构化表示 <span class="arithmatex">\(\mathcal{G} = (\mathcal{V}, \mathcal{E})\)</span>，节点 <span class="arithmatex">\(v \in \mathcal{V}\)</span> 为实体，边 <span class="arithmatex">\(e \in \mathcal{E}\)</span> 为带类型的关系。</p></li>
</ul>

<p>与情景记忆不同，语义记忆是<em>上下文无关的</em>：水在 <span class="arithmatex">\(100^\circ\)</span>C 沸腾这一事实，无论何时何地学到都成立。</p>

### 17.2.4　程序记忆（技能） {#procedural-memory-skills}

<p>程序记忆编码<em>如何做事</em>——已被自动化的技能与动作模式：</p>

<ul>
<li><p><strong>习得的工具使用模式。</strong> 对哪种任务调用哪个 API、如何格式化输入、如何处理错误。</p></li>
<li><p><strong>动作序列。</strong> 多步过程（例如 “部署代码：跑测试 <span class="arithmatex">\(\to\)</span> 构建镜像 <span class="arithmatex">\(\to\)</span> 推送 <span class="arithmatex">\(\to\)</span> 更新 manifest”）。</p></li>
<li><p><strong>作为记忆的 Policy。</strong> 模型权重 <span class="arithmatex">\(\theta\)</span> 本身即编码了程序性知识；在成功轨迹上微调是程序记忆巩固的一种形式。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">记忆类型的分类示例</p>
<p>一个协助软件开发的 Agent 会用到：</p>

<ul>
<li><p><strong>工作记忆</strong>：当前正在编辑的文件、刚收到的错误信息。</p></li>
<li><p><strong>情景记忆</strong>：“上周我在模块 X 第 42 行加了一个空值检查，修复了一个类似的 <code>NullPointerException</code>。”</p></li>
<li><p><strong>语义记忆</strong>：“Python 的 <code>asyncio.gather</code> 并发运行协程；除非设置 <code>return_exceptions=True</code>，否则异常会向上传播。”</p></li>
<li><p><strong>程序记忆</strong>：标准调试工作流——复现 <span class="arithmatex">\(\to\)</span> 隔离 <span class="arithmatex">\(\to\)</span> 提假设 <span class="arithmatex">\(\to\)</span> 验证 <span class="arithmatex">\(\to\)</span> 修复。</p></li>
</ul>
</div>

## 17.3　记忆架构 {#sec:memory-architectures}

### 17.3.1　基于 RAG 的记忆 {#sec:rag-memory}

<p>检索增强生成（Retrieval-Augmented Generation, RAG） <a class="book-citation" id="cite-lewis2020retrieval-1" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a> 是 LLM Agent 外部记忆的主流范式。记忆存储 <span class="arithmatex">\(\mathcal{M}\)</span> 是文档集合 <span class="arithmatex">\(\{d_i\}_{i=1}^N\)</span>；检索把查询 <span class="arithmatex">\(q\)</span> 映射到一个有序子集。</p>

##### Embedding 存储与向量数据库。 {#embedding-stores-and-vector-databases.}

<p>每个文档 <span class="arithmatex">\(d_i\)</span> 由 Embedding 模型 <span class="arithmatex">\(\phi\)</span> 编码：<span class="arithmatex">\(\mathbf{v}_i = \phi(d_i) \in \mathbb{R}^{D}\)</span>。查询同样被编码：<span class="arithmatex">\(\mathbf{q} = \phi(q)\)</span>。检索按相似度返回 top-<span class="arithmatex">\(k\)</span> 文档：</p>



<div class="arithmatex">\[\text{Retrieve}(q, \mathcal{M}, k) = \underset{S \subseteq [N],\, |S|=k}{\arg\max} \sum_{i \in S} \text{sim}(\mathbf{q}, \mathbf{v}_i),\]</div>

<p>其中 <span class="arithmatex">\(\text{sim}(\cdot,\cdot)\)</span> 通常为余弦相似度。近似最近邻（Approximate Nearest-Neighbor, ANN）索引（FAISS <a class="book-citation" id="cite-johnson2019billion-1" href="../../references.md#ref-johnson2019billion" aria-label="参考文献 171">[171]</a>、HNSW <a class="book-citation" id="cite-malkov2018efficient-1" href="../../references.md#ref-malkov2018efficient" aria-label="参考文献 248">[248]</a>、ScaNN <a class="book-citation" id="cite-guo2020scann-1" href="../../references.md#ref-guo2020scann" aria-label="参考文献 126">[126]</a>）使得 <span class="arithmatex">\(N \sim 10^7\)</span> 量级下仍然可行。</p>

##### 检索策略。 {#retrieval-strategies.}

<ul>
<li><p><strong>稠密检索（Dense retrieval）。</strong> 查询和文档都由神经编码器编码（例如 DPR <a class="book-citation" id="cite-karpukhin2020dense-1" href="../../references.md#ref-karpukhin2020dense" aria-label="参考文献 180">[180]</a>、<code>text-embedding-3-large</code>）。能捕捉语义相似度，但需要 GPU 推理。</p></li>
<li><p><strong>稀疏检索（Sparse retrieval）。</strong> 基于 Token 重叠的 BM25 或 TF-IDF。快速、可解释、在精确关键词匹配上表现强劲。</p></li>
<li><p><strong>混合检索（Hybrid retrieval）。</strong> 通过倒数排名融合（Reciprocal Rank Fusion, RRF）合并稠密与稀疏分数：</p>



<div class="arithmatex">\[\text{RRF}(d, k) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)},\]</div>

<p>其中 <span class="arithmatex">\(k=60\)</span> 是平滑常数。混合检索一致地优于单独使用任一种 <a class="book-citation" id="cite-chen2022hybrid-1" href="../../references.md#ref-chen2022hybrid" aria-label="参考文献 44">[44]</a>。</p></li>
</ul>

##### 重排序（Re-ranking）。 {#re-ranking.}

<p>交叉编码器（cross-encoder）重排序器 <span class="arithmatex">\(f_\psi(q, d) \in [0,1]\)</span> 与查询联合打分每个被检索文档，以 <span class="arithmatex">\(O(k)\)</span> 次前向传播的代价换取更高准确率。整体流水线为：用 ANN 检索 <span class="arithmatex">\(k&#x27; \gg k\)</span> 个候选，再用交叉编码器重排序，返回 top <span class="arithmatex">\(k\)</span>。</p>

<div class="admonition warning"><p class="admonition-title">检索引发的幻觉风险</p>
<p>RAG 并不能消除幻觉——反而可能<em>引入</em>幻觉。如果被检索的文档过时、错误或只是表面相关，模型可能自信地把错误信息纳入回答。务必包含来源元数据（来源、时间戳、置信度），并考虑加入忠实性（faithfulness）验证步骤。</p>
</div>

### 17.3.2　基于摘要的记忆 {#summarization-based-memory}

<p>当原样存储过于昂贵或过于嘈杂时，<em>摘要化</em>在存储之前先压缩信息。</p>

##### 渐进式摘要（Progressive Summarization）。 {#progressive-summarization.}

<p>在每一步 <span class="arithmatex">\(t\)</span>，Agent 维护一个运行中的摘要 <span class="arithmatex">\(S_t\)</span>。当新信息 <span class="arithmatex">\(e_t\)</span> 到达时：</p>



<div class="arithmatex">\[S_{t+1} = \text{LLM}\!\left(\texttt{``Summarize: [}S_t\texttt{] + [}e_t\texttt{]&#x27;&#x27;}\right).\]</div>

<p>这样把记忆规模保持在 <span class="arithmatex">\(O(1)\)</span>，但有丢失细节的风险。</p>

##### 分层压缩（Hierarchical Compression）。 {#hierarchical-compression.}

<p>将记忆组织为层级 <span class="arithmatex">\(L_0 \supset L_1 \supset \cdots \supset L_K\)</span>，其中 <span class="arithmatex">\(L_0\)</span> 是原文，每个 <span class="arithmatex">\(L_{i+1}\)</span> 都是 <span class="arithmatex">\(L_i\)</span> 的摘要。检索首先访问 <span class="arithmatex">\(L_K\)</span>（压缩程度最高、最快），按需逐层深入。这呼应了 Forte <a class="book-citation" id="cite-forte2022building-1" href="../../references.md#ref-forte2022building" aria-label="参考文献 96">[96]</a> 的<em>渐进式摘要</em>技术。</p>

##### 何时摘要、何时原样存储。 {#when-to-summarize-vs.-store-verbatim.}

<ul>
<li><p>原样存储：精确事实、代码片段、数值结果、用户原话。</p></li>
<li><p>摘要存储：叙事性上下文、推理链、冗余观察。</p></li>
<li><p>直接丢弃：噪声、无信息量的失败工具调用。</p></li>
</ul>

### 17.3.3　基于图的记忆 {#graph-based-memory}

##### 知识图谱。 {#knowledge-graphs.}

<p>知识图谱 <span class="arithmatex">\(\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})\)</span> 把事实存储为三元组 <span class="arithmatex">\((h, r, t)\)</span>，其中 <span class="arithmatex">\(h, t \in \mathcal{V}\)</span> 是实体，<span class="arithmatex">\(r \in \mathcal{R}\)</span> 是关系。Agent 可通过 SPARQL <a class="book-citation" id="cite-harris2013sparql-1" href="../../references.md#ref-harris2013sparql" aria-label="参考文献 134">[134]</a>、Cypher <a class="book-citation" id="cite-francis2018cypher-1" href="../../references.md#ref-francis2018cypher" aria-label="参考文献 97">[97]</a> 或自然语言到图的翻译进行查询。</p>

##### 实体-关系抽取。 {#entity-relation-extraction.}

<p>新观察由抽取模型 <span class="arithmatex">\(\text{IE}: \text{text} \to \{(h_i, r_i, t_i)\}\)</span> 解析并合并到 <span class="arithmatex">\(\mathcal{G}\)</span> 中。共指消解（coreference resolution）与实体链接（entity linking）保证一致性。</p>

##### GraphRAG。 {#graphrag.}

<p>GraphRAG <a class="book-citation" id="cite-edge2024local-1" href="../../references.md#ref-edge2024local" aria-label="参考文献 82">[82]</a> 在 RAG 之上叠加图遍历：给定一个查询，先检索种子实体，然后通过 <span class="arithmatex">\(k\)</span> 跳邻域遍历来挖掘 Embedding 相似度未直接匹配到的相关事实。这对多跳推理尤其强大：</p>



<div class="arithmatex">\[\text{GraphRetrieve}(q, \mathcal{G}, k) = \bigcup_{v \in \text{seeds}(q)} \mathcal{N}_k(v, \mathcal{G}),\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{N}_k(v, \mathcal{G})\)</span> 是 <span class="arithmatex">\(v\)</span> 的 <span class="arithmatex">\(k\)</span> 跳邻域。</p>

##### 时序知识图谱。 {#temporal-knowledge-graphs.}

<p>事实具有有效期：<span class="arithmatex">\((h, r, t, [t_\text{start}, t_\text{end}])\)</span>。时序知识图谱 <a class="book-citation" id="cite-lacroix2020tensor-1" href="../../references.md#ref-lacroix2020tensor" aria-label="参考文献 197">[197]</a> 支持类似 “2023 年 OpenAI 的 CEO 是谁？” 的查询，而不会把过去与现在的状态混淆在一起。</p>

### 17.3.4　键值记忆网络（Key-Value Memory Networks） {#key-value-memory-networks}

<p>可微分记忆网络 <a class="book-citation" id="cite-weston2014memory-1" href="../../references.md#ref-weston2014memory" aria-label="参考文献 377">[377]</a><a class="book-citation" id="cite-sukhbaatar2015end-1" href="../../references.md#ref-sukhbaatar2015end" aria-label="参考文献 342">[342]</a> 把记忆表示为一组键值对 <span class="arithmatex">\(\{(\mathbf{k}_i, \mathbf{v}_i)\}_{i=1}^M\)</span>，并通过基于软 Attention 的检索访问：</p>



<div class="arithmatex">\[\alpha_i = \text{softmax}\!\left(\frac{\mathbf{q}^\top \mathbf{k}_i}{\sqrt{D}}\right), \qquad
  \mathbf{c} = \sum_{i=1}^M \alpha_i \mathbf{v}_i.\]</div>

<p>检索得到的上下文 <span class="arithmatex">\(\mathbf{c}\)</span> 是查询的可微函数，从而支持端到端训练。现代 Transformer 的 Attention 就是这种机制的特例。在智能体场景中，记忆槽可以通过梯度下降更新，也可以通过显式的写入操作更新。</p>

### 17.3.5　MemGPT 与虚拟上下文管理 {#sec:memgpt}

<p>MemGPT <a class="book-citation" id="cite-packer2023memgpt-1" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a> 引入了类似操作系统虚拟内存的<em>虚拟上下文</em>抽象。记忆被组织为多层：</p>

##### 页入/页出策略（Page-In / Page-Out）。 {#page-in-page-out-strategies.}

<p>Agent 基于以下因素决定<em>哪些</em>记忆要提升到热上下文（page-in），<em>哪些</em>要逐出（page-out）：</p>

<ul>
<li><p><strong>近期性（Recency）：</strong> 最近访问过的条目更有可能再次被需要。</p></li>
<li><p><strong>相关性（Relevance）：</strong> 与当前查询相似度高的条目。</p></li>
<li><p><strong>重要性（Importance）：</strong> 写入时被标记为高重要性的条目。</p></li>
</ul>

##### 自驱动的记忆管理。 {#self-directed-memory-management.}

<p>在 MemGPT 中，LLM 自身把记忆管理的函数调用（<code>memory_search</code>、<code>memory_insert</code>、<code>memory_delete</code>）作为其动作空间的一部分发出。这使记忆管理成为<em>可学习的行为</em>而非硬编码的 Policy——天然适合作为 RL 训练的目标（见 <a href="#sec:rl-memory">17.7</a> 节）。</p>

## 17.4　记忆操作 {#sec:memory-operations}

### 17.4.1　写入：把信息提交到记忆 {#write-committing-to-memory}

<p>并非每条观察都应被存储。写入决策本质上是一个过滤问题：</p>



<div class="arithmatex">\[\text{Write}(e) = \mathbf{1}\!\left[\text{importance}(e) &gt; \tau\right],\]</div>

<p>其中 <span class="arithmatex">\(\tau\)</span> 为阈值，<span class="arithmatex">\(\text{importance}(e)\)</span> 可以是：</p>

<ul>
<li><p><strong>惊讶度（Surprise）：</strong> <span class="arithmatex">\(-\log p_\theta(e \mid \text{context})\)</span>——出乎意料的事件信息量更大。</p></li>
<li><p><strong>Reward 信号：</strong> 与高 <span class="arithmatex">\(|r_t|\)</span>（正或负）相关联的事件值得记住。</p></li>
<li><p><strong>LLM 自评：</strong> 提示模型在 1–10 的尺度上为重要性打分。</p></li>
</ul>

##### 冲突检测。 {#contradiction-detection.}

<p>在写入新事实 <span class="arithmatex">\(f_\text{new}\)</span> 之前，应检查它与已有记忆是否冲突：</p>



<div class="arithmatex">\[\text{Conflict}(f_\text{new}, \mathcal{M}) = \exists\, f \in \mathcal{M} : \text{Contradicts}(f_\text{new}, f).\]</div>

<p>冲突检测可以通过自然语言推理（NLI）模型实现，也可以通过提示 LLM 来完成。一旦发生冲突，Agent 必须决定：覆盖、带时间戳并保留两者，或者标记为需要人工审查。</p>

##### 记忆格式与粒度。 {#memory-format-and-granularity.}

<p>除了<em>存什么</em>，<em>怎么存</em>也至关重要。记忆条目从原子事实到冗长对话记录不等，各有不同的取舍：</p>

<p class="table-caption" id="table-17-1"><span class="table-number">表 17.1　</span>不同记忆粒度的取舍。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>格式</strong></th><th scope="col"><strong>优点</strong></th><th scope="col"><strong>缺点</strong></th></tr></thead>
<tbody><tr><td><strong>原子事实</strong></td><td></td><td></td></tr>
<tr><td>“用户偏好 Python。”</td><td>检索精确；可组合；去重与冲突检测容易</td><td>失去上下文；存在抽取错误；对细微信息脆弱</td></tr>
<tr><td><strong>结构化笔记</strong></td><td></td><td></td></tr>
<tr><td>(A-MEM <a class="book-citation" id="cite-xu2025amem-1" href="../../references.md#ref-xu2025amem" aria-label="参考文献 400">[400]</a>)</td><td>元数据丰富（标签、链接）；支持图遍历；在精确性与上下文之间取得平衡</td><td>写入成本更高；需要预先设计 schema</td></tr>
<tr><td><strong>摘要式 Episode</strong></td><td></td><td></td></tr>
<tr><td>(MemGPT <a class="book-citation" id="cite-packer2023memgpt-1--section-04" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a>)</td><td>保留叙事连贯性；紧凑；适合多轮（Multi-Turn）推理</td><td>摘要有损；难以局部更新</td></tr>
<tr><td><strong>原始对话记录</strong></td><td>无损；无抽取错误；支持精确引用</td><td>存储量大；检索嘈杂；扫描代价高</td></tr></tbody>
</table>
</div>

<p>实际生产系统往往会组合多种粒度 <a class="book-citation" id="cite-chhikara2025mem0-1" href="../../references.md#ref-chhikara2025mem0" aria-label="参考文献 53">[53]</a>：抽取原子事实以支持精确召回，维护摘要式 Episode 以保留叙事性上下文，并把原始对话记录存入冷存储以满足可审计性需求。Generative Agents 架构 <a class="book-citation" id="cite-park2023generative-1" href="../../references.md#ref-park2023generative" aria-label="参考文献 285">[285]</a> 将观察存储为原子的“记忆对象”，附带自然语言描述、重要性分数与时间戳——同时支持精确检索与时序推理。</p>

##### 设计准则。 {#design-guidelines.}

<ul>
<li><p><strong>粒度匹配查询类型。</strong> 如果用户问的是事实性问题（“我的 API key 是什么？”），原子事实更佳；如果问的是上下文性问题（“我们当时为什么决定用 Redis？”），就需要 Episode 摘要。</p></li>
<li><p><strong>在可承受的前提下用尽可能细的粒度存储</strong>，然后在其上构建粗粒度视图。把原子事实摘要化很容易；但从有损摘要中恢复原子是不可能的。</p></li>
<li><p><strong>记录来源。</strong> 每条记忆都应回链至来源（对话轮次、文档、工具输出），以便 Agent 可校验、用户可审计。</p></li>
</ul>

### 17.4.2　读取 / 检索 {#read-retrieve}

##### 查询的构造。 {#query-formulation.}

<p>检索查询 <span class="arithmatex">\(q\)</span> 未必要直接使用原始观察。更好的策略包括：</p>

<ul>
<li><p><strong>HyDE（假设性文档 Embedding，Hypothetical Document Embeddings） <a class="book-citation" id="cite-gao2022precise-1" href="../../references.md#ref-gao2022precise" aria-label="参考文献 106">[106]</a>：</strong> 先生成一个假设性的答案，对其做 Embedding，然后以该 Embedding 作为查询。</p></li>
<li><p><strong>查询扩展（Query expansion）：</strong> 生成查询的多个改写，并取检索结果的并集。</p></li>
<li><p><strong>回退式 Prompt（Step-back prompting）：</strong> 在检索之前先把具体查询抽象为一个更一般的问题。</p></li>
</ul>

##### 时间衰减与近期性偏置。 {#temporal-decay-and-recency-bias.}

<p>更久远的记忆可能相关性更低。一个带时间权重的评分是：</p>



<div class="arithmatex">\[\text{score}(d, q, t) = \lambda \cdot \text{sim}(\mathbf{q}, \mathbf{v}_d) + (1-\lambda) \cdot \exp\!\left(-\frac{t - t_d}{\tau_\text{decay}}\right),\]</div>

<p>其中 <span class="arithmatex">\(t_d\)</span> 是记忆的创建时间，<span class="arithmatex">\(\tau_\text{decay}\)</span> 控制衰减速率。Generative Agents 论文 <a class="book-citation" id="cite-park2023generative-2" href="../../references.md#ref-park2023generative" aria-label="参考文献 285">[285]</a> 采用了类似的近期性加权检索。</p>

### 17.4.3　更新：冲突解决与巩固 {#update-conflict-resolution-and-consolidation}

<p>记忆巩固（consolidation）会合并相关记忆，以减少冗余并浮现出更高层的模式：</p>



<div class="arithmatex">\[\mathcal{M}&#x27; = \text{Consolidate}(\mathcal{M}) = \text{Cluster}(\mathcal{M}) \cup \text{Summarize}(\text{Cluster}(\mathcal{M})).\]</div>

##### 遗忘机制。 {#forgetting-mechanisms.}

<p>生物记忆会遗忘；人工记忆也应如此。可用策略包括：</p>

<ul>
<li><p><strong>LRU 驱逐：</strong> 容量超限时移除最近最少使用的条目。</p></li>
<li><p><strong>重要性加权的遗忘：</strong> <span class="arithmatex">\(p(\text{forget}\,|\,d) \propto \exp(-\text{importance}(d))\)</span>。</p></li>
<li><p><strong>间隔重复（Spaced repetition）：</strong> 反复被访问的记忆保留得更久，遵循指数遗忘曲线 <a class="book-citation" id="cite-ebbinghaus1885memory-1" href="../../references.md#ref-ebbinghaus1885memory" aria-label="参考文献 81">[81]</a>。</p></li>
</ul>

### 17.4.4　反思：元认知操作 {#reflect-meta-cognitive-operations}

<p>反思（reflection） <a class="book-citation" id="cite-park2023generative-3" href="../../references.md#ref-park2023generative" aria-label="参考文献 285">[285]</a><a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a> 是一种高阶记忆操作：Agent 读取自己的记忆并生成<em>洞见</em>：</p>



<div class="arithmatex">\[\text{Reflect}(\mathcal{M}) \to \{i_1, i_2, \ldots\} \subset \mathcal{M}_\text{semantic},\]</div>

<p>其中每条洞见 <span class="arithmatex">\(i_j\)</span> 都是从多条情景记忆中归纳出的更高层抽象。</p>

<div class="admonition example"><p class="admonition-title">反思的实践（Reflexion）</p>
<p>在三次解决某个编程题失败之后，Agent 进行反思：</p>

<ol>
<li><p>从情景记忆中检索这三次失败的 Episode。</p></li>
<li><p>生成一条洞见：“我总是忘记处理输入列表为空的边界情况。”</p></li>
<li><p>把这条洞见存入语义记忆。</p></li>
<li><p>在下一次尝试中检索这条洞见，并显式地检查空输入。</p></li>
</ol>

<p>这就是 Reflexion <a class="book-citation" id="cite-shinn2023reflexion-2" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a> 的核心机制：通过自我反思的“言语化强化学习”。</p>
</div>

##### 反思结果存放在哪里？ {#where-do-reflections-live}

<p>反思从情景记忆<em>读取</em>，但向语义记忆<em>写入</em>。所产生的洞见是上下文无关的概括（“始终检查空输入”），并非具体 Episode 的记录——因此属于语义记忆 <span class="arithmatex">\(\mathcal{M}_\text{semantic}\)</span>。然而在反思过程本身中，中间推理（检索到的 Episode + 综合 Prompt + 生成的洞见）占用的是<em>工作记忆</em>（上下文窗口）。一言以蔽之：</p>

<ul>
<li><p><strong>输入</strong>：情景记忆（具体的过往事件）</p></li>
<li><p><strong>计算</strong>：工作记忆（上下文中的主动推理）</p></li>
<li><p><strong>输出</strong>：语义记忆（持久化的、归纳后的洞见）</p></li>
</ul>

<p>这与生物记忆巩固相对应——情景经验在睡眠与反思中逐渐被转化为语义知识。</p>

## 17.5　多轮对话中的记忆 {#sec:memory-multiturn}

### 17.5.1　用户建模与偏好追踪 {#user-modeling-and-preference-tracking}

<p>持久化的用户模型 <span class="arithmatex">\(\mathcal{U}\)</span> 存储：</p>

<ul>
<li><p><strong>显式偏好：</strong> 用户明确表达的喜好/厌恶、沟通风格偏好。</p></li>
<li><p><strong>隐式偏好：</strong> 从行为中推断（例如用户总是要求 Python 代码、偏好简洁回答）。</p></li>
<li><p><strong>专业水平：</strong> 从词汇与问题复杂度中推断出的领域知识水平。</p></li>
<li><p><strong>目标与上下文：</strong> 进行中的项目、当前任务、组织角色。</p></li>
</ul>

<p>用户模型在每次交互后更新：</p>



<div class="arithmatex">\[\mathcal{U}_{t+1} = \text{Update}(\mathcal{U}_t,\, (u_t, a_t, \text{feedback}_t)).\]</div>

### 17.5.2　会话连续性 {#session-continuity}

<p>没有记忆，每次对话都从“冷启动”开始。有了会话记忆后：</p>

<ol>
<li><p>在会话开始时，检索用户模型 <span class="arithmatex">\(\mathcal{U}\)</span> 与近期会话摘要。</p></li>
<li><p>注入个性化的系统 Prompt：“你正在帮助 Alice，一位从事分布式训练项目的高级 ML 工程师。上次会话你帮她调试了一个 Gradient 同步问题。”</p></li>
<li><p>在会话结束时，对会话进行摘要并更新 <span class="arithmatex">\(\mathcal{U}\)</span>。</p></li>
</ol>

### 17.5.3　通过记忆实现个性化 {#personalization-through-memory}

<p>个性化同时提升<em>效率</em>（更少的澄清问题）与<em>质量</em>（针对用户专业度校准的回答）。核心技术包括：</p>

<ul>
<li><p><strong>自适应详尽度：</strong> 根据用户的历史互动调整回答长度。</p></li>
<li><p><strong>领域引导（Domain priming）：</strong> 从语义记忆中预置相关领域上下文。</p></li>
<li><p><strong>主动回忆：</strong> 无需用户提示即主动浮现相关的历史交互（“你上个月问过这个话题，当时我们的结论是……”）。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">隐私与记忆</p>
<p>持久化的用户记忆带来了显著的隐私问题。Agent 必须：（1）在存储个人信息前获取明确同意；（2）提供检视与删除已存储记忆的机制；（3）在多用户部署中强制执行访问控制；（4）遵守数据留存法规（GDPR、CCPA）。记忆系统应秉持“默认隐私（privacy-by-default）”原则进行设计。</p>
</div>

## 17.6　多智能体系统中的记忆 {#sec:memory-multiagent}

<p>当多个 Agent 协作完成一个共享任务时，记忆变成了一种<em>协调机制</em>——而不仅是个人的知识库。负责任务分解的规划 Agent 必须把子目标传达给执行 Agent；批评者（critic）Agent 必须访问到与被评估 Agent 相同的上下文；研究型 Agent 团队必须避免重复工作。如果没有共享记忆，Agent 之间就只能通过直接消息传递一切，造成带宽瓶颈，并在对话滚出上下文时丢失信息。共享记忆通过提供所有 Agent 都可读写的、持久化且可查询的底座来解决这一问题——把隐式协调（“我希望对方记得”）转换为显式状态（“答案就在 blackboard 上”）。</p>

### 17.6.1　共享记忆池 {#shared-memory-pools}

<p>在多智能体系统中，Agent 可以在共享记忆存储 <span class="arithmatex">\(\mathcal{M}_\text{shared}\)</span> 之外，再各自保留私有存储 <span class="arithmatex">\(\mathcal{M}_i\)</span>：</p>



<div class="arithmatex">\[\text{context}_i(t) = \mathcal{R}(\mathcal{M}_i, q_i) \cup \mathcal{R}(\mathcal{M}_\text{shared}, q_i).\]</div>

<p>共享记忆使<em>隐式协调</em>成为可能：Agent <span class="arithmatex">\(A\)</span> 写入一个发现；Agent <span class="arithmatex">\(B\)</span> 无需显式通信即可检索到。</p>

### 17.6.2　Blackboard 架构 {#blackboard-architecture}

<p><em>blackboard</em>（黑板）模式 <a class="book-citation" id="cite-hayes1985blackboard-1" href="../../references.md#ref-hayes1985blackboard" aria-label="参考文献 135">[135]</a> 是一种经典的多智能体协调机制：</p>

<p>每个 Agent 都从 blackboard 读取并向其写入。一个<em>控制器</em>监控 blackboard，并在某个 Agent 的前置条件满足时将其激活。这把 Agent 彼此解耦：它们通过共享状态而非直接消息进行通信。</p>

### 17.6.3　共享知识中的共识与冲突 {#consensus-and-conflict-in-shared-knowledge}

<p>当多个 Agent 同时写入共享记忆时，冲突难以避免。常用的解决策略包括：</p>

<ul>
<li><p><strong>Last-write-wins（后写者胜出）：</strong> 简单但会丢失信息。</p></li>
<li><p><strong>版本化记忆：</strong> 维护所有写入的历史；Agent 可以查询任一版本。</p></li>
<li><p><strong>投票 / 共识：</strong> 在某个事实被提交之前，要求 <span class="arithmatex">\(n\)</span> 个 Agent 中有 <span class="arithmatex">\(k\)</span> 个达成一致。</p></li>
<li><p><strong>置信度加权合并：</strong> <span class="arithmatex">\(f_\text{merged} = \sum_i w_i f_i\)</span>，其中 <span class="arithmatex">\(w_i\)</span> 是 Agent <span class="arithmatex">\(i\)</span> 的置信度。</p></li>
<li><p><strong>指定权威：</strong> 把不同记忆区域的所有权分配给特定的 Agent。</p></li>
</ul>

<div class="admonition question"><p class="admonition-title">开放问题：分布式记忆一致性</p>
<p>在并发写入、网络分区与对抗性 Agent 的存在下，多智能体系统应当如何维持记忆一致性？经典的分布式系统方案（Paxos、Raft）可以套用但代价高昂。带有界过时（bounded staleness）的近似一致性对许多智能体任务可能已经足够——但合适的取舍仍是一个开放的研究问题。</p>
</div>

## 17.7　用强化学习训练记忆系统 {#sec:rl-memory}

### 17.7.1　记忆操作的奖励信号 {#reward-signals-for-memory-operations}

<p>记忆操作（读、写、更新、反思）可以被视为 RL 框架中的动作。挑战在于设计能激励<em>有用</em>记忆行为的 Reward 信号：</p>

<ul>
<li><p><strong>任务 Reward 回传。</strong> 如果某次记忆检索导致了正确答案，则把功劳归于该检索动作。稀疏但毫不含糊。</p></li>
<li><p><strong>检索精度 Reward。</strong> <span class="arithmatex">\(r_\text{retrieve} = \text{Relevance}(d_\text{retrieved}, \text{task})\)</span>，由一个学习得到的相关性模型估计。</p></li>
<li><p><strong>记忆效率 Reward。</strong> 对不必要的写入施加惩罚：<span class="arithmatex">\(r_\text{write} = -\lambda \cdot \mathbf{1}[\text{write}]\)</span>，从而鼓励选择性存储。</p></li>
<li><p><strong>一致性 Reward。</strong> 奖励内部一致（无矛盾）的记忆状态。</p></li>
</ul>

<p>在第 <span class="arithmatex">\(t\)</span> 步对一次记忆操作 <span class="arithmatex">\(m_t\)</span> 的组合 Reward：</p>



<div class="arithmatex">\[r_t^{\text{mem}} = r_t^{\text{task}} + \alpha \cdot r_t^{\text{retrieve}} + \beta \cdot r_t^{\text{write}} + \gamma \cdot r_t^{\text{consistency}}.\]</div>

### 17.7.2　学习“记什么” {#learning-what-to-remember}

<p>“记什么”是一个元学习挑战：Agent 必须学到一个能最大化未来任务表现的写入 Policy <span class="arithmatex">\(\pi_\text{write}(e)\)</span>。其难点在于：</p>

<ol>
<li><p>一条记忆的价值只有在未来才会显现（延迟 Reward）。</p></li>
<li><p>写入时未来可能查询的空间是未知的。</p></li>
<li><p>记忆之间相互影响：存储 <span class="arithmatex">\(e\)</span> 的价值依赖于 <span class="arithmatex">\(\mathcal{M}\)</span> 中已有的其他内容。</p></li>
</ol>

<p>常见的方法有：</p>

<ul>
<li><p><strong>事后重标记（Hindsight relabeling）</strong> <a class="book-citation" id="cite-andrychowicz2017hindsight-1" href="../../references.md#ref-andrychowicz2017hindsight" aria-label="参考文献 8">[8]</a>。一个 Episode 成功后，回溯性地把被检索到的记忆标记为“重要”，并训练写入 Policy 去存储类似条目。</p></li>
<li><p><strong>元强化学习（Meta-RL）</strong> <a class="book-citation" id="cite-duan2016rl2-1" href="../../references.md#ref-duan2016rl2" aria-label="参考文献 79">[79]</a>。在一组任务分布上训练写入 Policy；Policy 学到存储能在任务间泛化的信息。</p></li>
<li><p><strong>好奇心驱动的存储（Curiosity-driven storage）</strong> <a class="book-citation" id="cite-pathak2017curiosity-1" href="../../references.md#ref-pathak2017curiosity" aria-label="参考文献 286">[286]</a>。存储令人意外（预测误差高）的观察，因为它们更可能蕴含信息。</p></li>
</ul>

### 17.7.3　记忆增强的 Policy 优化 {#memory-augmented-policy-optimization}

<p>联合优化 Policy 与其记忆系统的思想可追溯到可微分记忆网络 <a class="book-citation" id="cite-graves2016hybrid-1" href="../../references.md#ref-graves2016hybrid" aria-label="参考文献 119">[119]</a>，并由 REALM <a class="book-citation" id="cite-guu2020realm-1" href="../../references.md#ref-guu2020realm" aria-label="参考文献 128">[128]</a> 推广到检索增强的 LLM。一个记忆增强 Agent 的完整 Policy Gradient 目标为：</p>



<div class="arithmatex">\[\mathcal{L}(\theta, \phi) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^T \gamma^t r_t\right] - \lambda \cdot \mathcal{L}_\text{mem}(\phi),\]</div>

<p>其中 <span class="arithmatex">\(\theta\)</span> 是 LLM 参数，<span class="arithmatex">\(\phi\)</span> 是记忆系统参数（例如检索模型权重），<span class="arithmatex">\(\mathcal{L}_\text{mem}\)</span> 是对记忆复杂度的正则项。</p>

<div class="admonition note"><p class="admonition-title">关键洞见：把记忆作为可学习的归纳偏置</p>
<p>用 RL 训练记忆操作，让 Agent 得以发展出面向具体任务的记忆策略。一个编码 Agent 学会存储 API 签名；一个研究型 Agent 学会存储引文链；一个客服 Agent 学会存储用户投诉模式。记忆系统由此成为一种为 Agent 领域量身打造的<em>可学习归纳偏置</em>。</p>
</div>

## 17.8　记忆方案的对比 {#sec:memory-comparison}

<p class="table-caption" id="table-17-2"><span class="table-number">表 17.2　</span>在关键维度上对智能体记忆架构的对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>架构</strong></th><th scope="col"><strong>容量</strong></th><th scope="col"><strong>检索</strong></th><th scope="col"><strong>更新成本</strong></th><th scope="col"><strong>可训练</strong></th><th scope="col"><strong>最适合</strong></th></tr></thead>
<tbody><tr><td>上下文内（工作记忆）</td><td><span class="arithmatex">\(O(L)\)</span> Token</td><td>0 ms</td><td>免费</td><td>通过微调</td><td>短任务、主动推理</td></tr>
<tr><td>稠密 RAG <a class="book-citation" id="cite-lewis2020retrieval-1--section-08" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a></td><td><span class="arithmatex">\(O(10^7)\)</span> 文档</td><td>10–50 ms</td><td><span class="arithmatex">\(O(1)\)</span> Embedding</td><td>仅编码器</td><td>语义检索、问答</td></tr>
<tr><td>稀疏（BM25） <a class="book-citation" id="cite-robertson2009probabilistic-1" href="../../references.md#ref-robertson2009probabilistic" aria-label="参考文献 309">[309]</a></td><td><span class="arithmatex">\(O(10^8)\)</span> 文档</td><td>1–5 ms</td><td><span class="arithmatex">\(O(|d|)\)</span> 索引</td><td>否</td><td>关键词检索、法律/医疗</td></tr>
<tr><td>混合 RAG <a class="book-citation" id="cite-chen2022hybrid-1--section-08" href="../../references.md#ref-chen2022hybrid" aria-label="参考文献 44">[44]</a></td><td><span class="arithmatex">\(O(10^7)\)</span> 文档</td><td>15–60 ms</td><td><span class="arithmatex">\(O(1)\)</span> Embedding</td><td>仅编码器</td><td>通用检索</td></tr>
<tr><td>摘要</td><td>无上限</td><td>0 ms（在上下文内）</td><td><span class="arithmatex">\(O(|e|)\)</span> 次 LLM 调用</td><td>通过微调</td><td>长对话、叙事</td></tr>
<tr><td>知识图谱 <a class="book-citation" id="cite-lacroix2020tensor-1--section-08" href="../../references.md#ref-lacroix2020tensor" aria-label="参考文献 197">[197]</a></td><td><span class="arithmatex">\(O(10^9)\)</span> 三元组</td><td>5–100 ms</td><td><span class="arithmatex">\(O(1)\)</span> 插入</td><td>Embedding 层</td><td>结构化事实、多跳</td></tr>
<tr><td>键值记忆网络 <a class="book-citation" id="cite-sukhbaatar2015end-1--section-08" href="../../references.md#ref-sukhbaatar2015end" aria-label="参考文献 342">[342]</a></td><td><span class="arithmatex">\(O(M)\)</span> 槽位</td><td><span class="arithmatex">\(O(M)\)</span> Attention</td><td>一次 Gradient 更新</td><td>完全可训</td><td>端到端可微任务</td></tr>
<tr><td>MemGPT 分层 <a class="book-citation" id="cite-packer2023memgpt-1--section-08" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a></td><td>无上限</td><td>0–100 ms</td><td>混合</td><td>通过 RL</td><td>长周期 Agent、助手</td></tr>
<tr><td>Graph RAG <a class="book-citation" id="cite-edge2024local-1--section-08" href="../../references.md#ref-edge2024local" aria-label="参考文献 82">[82]</a></td><td><span class="arithmatex">\(O(10^7)\)</span> 节点</td><td>20–200 ms</td><td><span class="arithmatex">\(O(1)\)</span> 插入</td><td>仅编码器</td><td>复杂推理、社区结构</td></tr></tbody>
</table>
</div>

## 17.9　评估记忆系统 {#sec:memory-evaluation}

<p>评估智能体记忆颇具挑战，因为记忆操作的质量只能<em>间接</em>显现——通过长时间跨度的下游任务表现。一个对所存事实有完美召回率的记忆系统，若检索出无关上下文或撑爆 LLM 的上下文窗口，仍可能整体失败。</p>

### 17.9.1　评估维度 {#evaluation-dimensions}

<p>LongMemEval <a class="book-citation" id="cite-wu2024longmemeval-1" href="../../references.md#ref-wu2024longmemeval" aria-label="参考文献 384">[384]</a> 提出长期记忆系统必须展现的五项核心能力：</p>

<ol>
<li><p><strong>信息抽取。</strong> 系统能否从对话轮次中识别并存储显著事实？通过事实召回率衡量：有多少真值事实可从记忆中恢复？</p></li>
<li><p><strong>跨会话推理。</strong> 系统能否综合分散在多次过往会话中的信息？例如：“根据我们上周和昨天的对话，项目范围发生了哪些变化？”</p></li>
<li><p><strong>时序推理。</strong> 系统能否正确回答与时间相关的查询？例如“在组织调整<em>之前</em>我说我的优先事项是什么？”这就要求区分不同时间的状态。</p></li>
<li><p><strong>知识更新。</strong> 当事实发生变化时（用户搬到新城市、偏好改变），记忆能否反映最新状态，同时保留历史？</p></li>
<li><p><strong>拒答（Abstention）。</strong> 当系统并无相关记忆时，它能否正确地说“我不知道”，而不是幻觉出一段看似合理却完全是捏造的回忆？</p></li>
</ol>

### 17.9.2　基准（Benchmarks） {#benchmarks}

<p class="table-caption" id="table-17-3"><span class="table-number">表 17.3　</span>评估智能体记忆系统的基准。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>基准</strong></th><th scope="col"><strong>发表会议/期刊</strong></th><th scope="col"><strong>规模</strong></th><th scope="col"><strong>关注点</strong></th></tr></thead>
<tbody><tr><td>LongMemEval <a class="book-citation" id="cite-wu2024longmemeval-2" href="../../references.md#ref-wu2024longmemeval" aria-label="参考文献 384">[384]</a></td><td>ICLR 2025</td><td>500 个问题，可扩展的历史</td><td>五项记忆能力；多会话对话</td></tr>
<tr><td>LOCOMO <a class="book-citation" id="cite-maharana2024locomo-1" href="../../references.md#ref-maharana2024locomo" aria-label="参考文献 247">[247]</a></td><td>EMNLP 2024</td><td>多会话对话</td><td>对话之上的单跳、时序、多跳与开放域问答</td></tr>
<tr><td>InfiniteBench <a class="book-citation" id="cite-zhang2024infinitebench-1" href="../../references.md#ref-zhang2024infinitebench" aria-label="参考文献 428">[428]</a></td><td>ACL 2024</td><td>100K+ Token 上下文</td><td>长上下文召回，并非专门为记忆设计，但能测试上限</td></tr></tbody>
</table>
</div>

### 17.9.3　度量指标 {#metrics}

##### 记忆层面的指标。 {#memory-level-metrics.}

<ul>
<li><p><strong>记忆召回率（Memory Recall）</strong>：<span class="arithmatex">\(\frac{\text{可从记忆中检索到的真值事实数}}{\text{真值事实总数}}\)</span>。衡量存储的完备程度。</p></li>
<li><p><strong>记忆精确率（Memory Precision）</strong>：<span class="arithmatex">\(\frac{\text{top-}k\text{ 检索中相关条目数}}{k}\)</span>。衡量检索的噪声。</p></li>
<li><p><strong>延迟（Latency）</strong>：从查询到取回上下文的耗时（p50 与 p95）。</p></li>
<li><p><strong>Token 效率</strong>：每次查询注入上下文的 Token 总数。越低越好——多余的上下文既降低 LLM 准确率又增加成本。</p></li>
</ul>

##### 下游指标。 {#downstream-metrics.}

<ul>
<li><p><strong>回答准确率</strong>：在记忆条件下最终回答的正确性（EM、F1 或 LLM-as-judge）。</p></li>
<li><p><strong>忠实性（Faithfulness）</strong>：回答是否准确反映记忆中的内容，而没有捏造？</p></li>
<li><p><strong>个性化质量</strong>：用户满意度，通过偏好评分或在记忆增强与无记忆系统之间做 A/B 测试来衡量。</p></li>
<li><p><strong>自相矛盾率</strong>：系统给出与此前所述事实不一致回答的频率。</p></li>
</ul>

##### 运维层面的指标。 {#operational-metrics.}

<ul>
<li><p><strong>写入选择性</strong>：触发记忆写入的轮次比例。过高 <span class="arithmatex">\(\to\)</span> 噪声；过低 <span class="arithmatex">\(\to\)</span> 信息缺口。</p></li>
<li><p><strong>过时率（Staleness）</strong>：尽管已存在更新，仍检索到过时事实的频率。</p></li>
<li><p><strong>存储增长率</strong>：每小时交互所存储的 Token 数。无界增长是不可持续的。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">评估鸿沟</p>
<p>大多数记忆论文都在短基准（10–50 个会话）上做评估。真实的生产 Agent 要运行数月、跨越数千个会话。长周期评估——记忆漂移、矛盾累积与存储膨胀在那里成为主要失败模式——仍是一个开放挑战。从业者应在基准分数之外，对运维指标进行纵向监控。</p>
</div>

## 17.10　实现模式 {#sec:memory-implementation}

### 17.10.1　基于 Embedding 的向量存储记忆 {#vector-store-memory-with-embeddings}

<p>最常见的记忆模式把条目存储为 Embedding 向量，并附带元数据（时间戳、重要性分数、标签）。检索把余弦相似度与时间衰减结合起来，使近期且重要的记忆优先浮现；去重与 LRU 驱逐则把存储规模约束在上限以内。</p>

<p class="code-caption"><span class="code-number">代码 17.1　</span>基于 Embedding、带重要性打分与混合检索的向量存储记忆。</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">dataclasses</span><span class="w"> </span><span class="kn">import</span> <span class="n">dataclass</span><span class="p">,</span> <span class="n">field</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">datetime</span><span class="w"> </span><span class="kn">import</span> <span class="n">datetime</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Optional</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">MemoryEntry</span><span class="p">:</span>
    <span class="c1"># """A single memory entry with metadata."""</span>
<span class="w">    </span><span class="sd">"""带元数据的单条记忆条目。"""</span>
    <span class="n">content</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">embedding</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">ndarray</span>
    <span class="n">timestamp</span><span class="p">:</span> <span class="n">datetime</span> <span class="o">=</span> <span class="n">field</span><span class="p">(</span><span class="n">default_factory</span><span class="o">=</span><span class="n">datetime</span><span class="o">.</span><span class="n">now</span><span class="p">)</span>
    <span class="n">importance</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.5</span>
    <span class="n">access_count</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">last_accessed</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="n">datetime</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">tags</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="n">field</span><span class="p">(</span><span class="n">default_factory</span><span class="o">=</span><span class="nb">list</span><span class="p">)</span>
    <span class="n">source</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">"agent"</span>

<span class="k">class</span><span class="w"> </span><span class="nc">VectorMemoryStore</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    Hybrid dense+sparse memory store with temporal decay.</span>
<span class="sd">    Supports importance-weighted retrieval and LRU eviction.</span>
<span class="sd">    带时间衰减的稠密+稀疏混合记忆存储。</span>
<span class="sd">    支持基于重要性的加权检索与 LRU 驱逐。</span>
<span class="sd">    """</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span>
        <span class="n">embed_fn</span><span class="p">,</span>           <span class="c1"># callable: str -&gt; np.ndarray  # 可调用对象：str -&gt; np.ndarray</span>
        <span class="n">max_entries</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">10_000</span><span class="p">,</span>
        <span class="n">decay_rate</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.01</span><span class="p">,</span>   <span class="c1"># per hour  # 每小时</span>
        <span class="n">recency_weight</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.3</span><span class="p">,</span>
    <span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">embed_fn</span> <span class="o">=</span> <span class="n">embed_fn</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">max_entries</span> <span class="o">=</span> <span class="n">max_entries</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">decay_rate</span> <span class="o">=</span> <span class="n">decay_rate</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">recency_weight</span> <span class="o">=</span> <span class="n">recency_weight</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">MemoryEntry</span><span class="p">]</span> <span class="o">=</span> <span class="p">[]</span>

    <span class="c1"># -- Write --------------------------------------------------------------</span>
    <span class="c1"># -- 写入 --------------------------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">write</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span>
        <span class="n">content</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
        <span class="n">importance</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">,</span>
        <span class="n">tags</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
        <span class="n">check_duplicates</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
    <span class="p">)</span> <span class="o">-&gt;</span> <span class="n">MemoryEntry</span><span class="p">:</span>
        <span class="c1"># """Commit a new memory, evicting if at capacity."""</span>
<span class="w">        </span><span class="sd">"""提交一条新记忆，若已达容量上限则驱逐。"""</span>
        <span class="k">if</span> <span class="n">check_duplicates</span> <span class="ow">and</span> <span class="bp">self</span><span class="o">.</span><span class="n">_is_duplicate</span><span class="p">(</span><span class="n">content</span><span class="p">):</span>
            <span class="c1"># Skip near-duplicate entries</span>
            <span class="k">return</span> <span class="kc">None</span>  <span class="c1"># 跳过近似重复条目</span>

        <span class="n">embedding</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">embed_fn</span><span class="p">(</span><span class="n">content</span><span class="p">)</span>
        <span class="n">entry</span> <span class="o">=</span> <span class="n">MemoryEntry</span><span class="p">(</span>
            <span class="n">content</span><span class="o">=</span><span class="n">content</span><span class="p">,</span>
            <span class="n">embedding</span><span class="o">=</span><span class="n">embedding</span><span class="p">,</span>
            <span class="n">importance</span><span class="o">=</span><span class="n">importance</span><span class="p">,</span>
            <span class="n">tags</span><span class="o">=</span><span class="n">tags</span> <span class="ow">or</span> <span class="p">[],</span>
        <span class="p">)</span>

        <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">)</span> <span class="o">&gt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">max_entries</span><span class="p">:</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">_evict</span><span class="p">()</span>

        <span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">entry</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">entry</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_is_duplicate</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">content</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">threshold</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.95</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">bool</span><span class="p">:</span>
        <span class="c1"># """Check if a near-duplicate already exists."""</span>
<span class="w">        </span><span class="sd">"""检查是否已存在近似重复项。"""</span>
        <span class="k">if</span> <span class="ow">not</span> <span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">:</span>
            <span class="k">return</span> <span class="kc">False</span>
        <span class="n">emb</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">embed_fn</span><span class="p">(</span><span class="n">content</span><span class="p">)</span>
        <span class="n">sims</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_cosine_similarities</span><span class="p">(</span><span class="n">emb</span><span class="p">)</span>
        <span class="k">return</span> <span class="nb">float</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">max</span><span class="p">(</span><span class="n">sims</span><span class="p">))</span> <span class="o">&gt;</span> <span class="n">threshold</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_evict</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="c1"># """Remove the least important + least recent entry."""</span>
<span class="w">        </span><span class="sd">"""移除最不重要且最久未访问的条目。"""</span>
        <span class="n">now</span> <span class="o">=</span> <span class="n">datetime</span><span class="o">.</span><span class="n">now</span><span class="p">()</span>
        <span class="n">scores</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">e</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">:</span>
            <span class="n">age_hours</span> <span class="o">=</span> <span class="p">(</span><span class="n">now</span> <span class="o">-</span> <span class="n">e</span><span class="o">.</span><span class="n">timestamp</span><span class="p">)</span><span class="o">.</span><span class="n">total_seconds</span><span class="p">()</span> <span class="o">/</span> <span class="mi">3600</span>
            <span class="n">recency</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="bp">self</span><span class="o">.</span><span class="n">decay_rate</span> <span class="o">*</span> <span class="n">age_hours</span><span class="p">)</span>
            <span class="n">score</span> <span class="o">=</span> <span class="n">e</span><span class="o">.</span><span class="n">importance</span> <span class="o">*</span> <span class="p">(</span><span class="mi">1</span> <span class="o">-</span> <span class="bp">self</span><span class="o">.</span><span class="n">recency_weight</span><span class="p">)</span> \
                  <span class="o">+</span> <span class="n">recency</span> <span class="o">*</span> <span class="bp">self</span><span class="o">.</span><span class="n">recency_weight</span>
            <span class="n">scores</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">score</span><span class="p">)</span>
        <span class="n">worst_idx</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">argmin</span><span class="p">(</span><span class="n">scores</span><span class="p">))</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="o">.</span><span class="n">pop</span><span class="p">(</span><span class="n">worst_idx</span><span class="p">)</span>

    <span class="c1"># -- Retrieve -----------------------------------------------------------</span>
    <span class="c1"># -- 检索 -----------------------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">retrieve</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span>
        <span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
        <span class="n">k</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
        <span class="n">recency_boost</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
    <span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="n">MemoryEntry</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""</span>
<span class="sd">        Hybrid retrieval: dense similarity + temporal recency.</span>
<span class="sd">        Returns top-k entries sorted by combined score.</span>
<span class="sd">        混合检索：稠密相似度 + 时间近期性。</span>
<span class="sd">        返回按组合分数排序的前 k 条。</span>
<span class="sd">        """</span>
        <span class="k">if</span> <span class="ow">not</span> <span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">:</span>
            <span class="k">return</span> <span class="p">[]</span>

        <span class="n">q_emb</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">embed_fn</span><span class="p">(</span><span class="n">query</span><span class="p">)</span>
        <span class="n">dense_scores</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_cosine_similarities</span><span class="p">(</span><span class="n">q_emb</span><span class="p">)</span>

        <span class="n">now</span> <span class="o">=</span> <span class="n">datetime</span><span class="o">.</span><span class="n">now</span><span class="p">()</span>
        <span class="n">combined</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">entry</span><span class="p">,</span> <span class="n">d_score</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span>
            <span class="nb">zip</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">,</span> <span class="n">dense_scores</span><span class="p">)</span>
        <span class="p">):</span>
            <span class="k">if</span> <span class="n">recency_boost</span><span class="p">:</span>
                <span class="n">age_h</span> <span class="o">=</span> <span class="p">(</span><span class="n">now</span> <span class="o">-</span> <span class="n">entry</span><span class="o">.</span><span class="n">timestamp</span><span class="p">)</span><span class="o">.</span><span class="n">total_seconds</span><span class="p">()</span> <span class="o">/</span> <span class="mi">3600</span>
                <span class="n">recency</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="bp">self</span><span class="o">.</span><span class="n">decay_rate</span> <span class="o">*</span> <span class="n">age_h</span><span class="p">)</span>
                <span class="n">score</span> <span class="o">=</span> <span class="p">(</span><span class="mi">1</span> <span class="o">-</span> <span class="bp">self</span><span class="o">.</span><span class="n">recency_weight</span><span class="p">)</span> <span class="o">*</span> <span class="n">d_score</span> \
                      <span class="o">+</span> <span class="bp">self</span><span class="o">.</span><span class="n">recency_weight</span> <span class="o">*</span> <span class="n">recency</span>
            <span class="k">else</span><span class="p">:</span>
                <span class="n">score</span> <span class="o">=</span> <span class="n">d_score</span>
            <span class="n">combined</span><span class="o">.</span><span class="n">append</span><span class="p">((</span><span class="n">score</span><span class="p">,</span> <span class="n">i</span><span class="p">))</span>

        <span class="n">combined</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">reverse</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="n">top_k</span> <span class="o">=</span> <span class="p">[</span><span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="k">for</span> <span class="n">_</span><span class="p">,</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">combined</span><span class="p">[:</span><span class="n">k</span><span class="p">]]</span>

        <span class="c1"># Update access metadata</span>
        <span class="c1"># 更新访问元数据</span>
        <span class="k">for</span> <span class="n">entry</span> <span class="ow">in</span> <span class="n">top_k</span><span class="p">:</span>
            <span class="n">entry</span><span class="o">.</span><span class="n">access_count</span> <span class="o">+=</span> <span class="mi">1</span>
            <span class="n">entry</span><span class="o">.</span><span class="n">last_accessed</span> <span class="o">=</span> <span class="n">now</span>

        <span class="k">return</span> <span class="n">top_k</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_cosine_similarities</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">query_emb</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">ndarray</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">np</span><span class="o">.</span><span class="n">ndarray</span><span class="p">:</span>
        <span class="c1"># """Vectorized cosine similarity against all stored embeddings."""</span>
<span class="w">        </span><span class="sd">"""对所有已存 Embedding 进行向量化余弦相似度计算。"""</span>
        <span class="n">matrix</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">e</span><span class="o">.</span><span class="n">embedding</span> <span class="k">for</span> <span class="n">e</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">])</span>
        <span class="n">norms</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="n">matrix</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">keepdims</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="n">matrix_norm</span> <span class="o">=</span> <span class="n">matrix</span> <span class="o">/</span> <span class="p">(</span><span class="n">norms</span> <span class="o">+</span> <span class="mf">1e-8</span><span class="p">)</span>
        <span class="n">q_norm</span> <span class="o">=</span> <span class="n">query_emb</span> <span class="o">/</span> <span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="n">query_emb</span><span class="p">)</span> <span class="o">+</span> <span class="mf">1e-8</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">matrix_norm</span> <span class="o">@</span> <span class="n">q_norm</span>

    <span class="c1"># -- Reflect ------------------------------------------------------------</span>
    <span class="c1"># -- 反思 ------------------------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">reflect</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">llm_fn</span><span class="p">,</span> <span class="n">k</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">10</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""</span>
<span class="sd">        Meta-cognitive reflection: retrieve recent memories,</span>
<span class="sd">        synthesize higher-level insights, and store them back.</span>
<span class="sd">        元认知反思：检索近期记忆，综合出更高层的洞见，并写回存储。</span>
<span class="sd">        """</span>
        <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">)</span> <span class="o">&lt;</span> <span class="mi">3</span><span class="p">:</span>
            <span class="k">return</span> <span class="p">[]</span>

        <span class="c1"># Retrieve recent high-importance memories</span>
        <span class="c1"># 检索近期高重要性的记忆</span>
        <span class="n">recent</span> <span class="o">=</span> <span class="nb">sorted</span><span class="p">(</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">,</span> <span class="n">key</span><span class="o">=</span><span class="k">lambda</span> <span class="n">e</span><span class="p">:</span> <span class="n">e</span><span class="o">.</span><span class="n">timestamp</span><span class="p">,</span> <span class="n">reverse</span><span class="o">=</span><span class="kc">True</span>
        <span class="p">)[:</span><span class="n">k</span><span class="p">]</span>
        <span class="n">context</span> <span class="o">=</span> <span class="s2">"</span><span class="se">\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="sa">f</span><span class="s2">"- </span><span class="si">{</span><span class="n">e</span><span class="o">.</span><span class="n">content</span><span class="si">}</span><span class="s2">"</span> <span class="k">for</span> <span class="n">e</span> <span class="ow">in</span> <span class="n">recent</span><span class="p">)</span>

        <span class="c1"># Ask LLM to generate insights</span>
        <span class="c1"># 请 LLM 生成洞见</span>
        <span class="n">prompt</span> <span class="o">=</span> <span class="p">(</span>
            <span class="s2">"Given these recent memories, extract 2-3 high-level "</span>
            <span class="s2">"insights or patterns:</span><span class="se">\n</span><span class="s2">"</span> <span class="o">+</span> <span class="n">context</span>
        <span class="p">)</span>
        <span class="n">raw_insights</span> <span class="o">=</span> <span class="n">llm_fn</span><span class="p">(</span><span class="n">prompt</span><span class="p">)</span>

        <span class="c1"># Store each insight as a high-importance memory</span>
        <span class="c1"># 把每条洞见作为高重要性记忆存储</span>
        <span class="n">insights</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">line</span> <span class="ow">in</span> <span class="n">raw_insights</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">"</span><span class="se">\n</span><span class="s2">"</span><span class="p">):</span>
            <span class="n">line</span> <span class="o">=</span> <span class="n">line</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span><span class="o">.</span><span class="n">lstrip</span><span class="p">(</span><span class="s2">"-*"</span><span class="p">)</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span>
            <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">line</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mi">20</span><span class="p">:</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">write</span><span class="p">(</span>
                    <span class="sa">f</span><span class="s2">"[INSIGHT] </span><span class="si">{</span><span class="n">line</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span>
                    <span class="n">importance</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span>
                    <span class="n">check_duplicates</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                <span class="p">)</span>
                <span class="n">insights</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">line</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">insights</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">get_stats</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
        <span class="c1"># """Return memory statistics for monitoring."""</span>
<span class="w">        </span><span class="sd">"""返回记忆统计信息，便于监控。"""</span>
        <span class="k">return</span> <span class="p">{</span>
            <span class="s2">"total_entries"</span><span class="p">:</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">),</span>
            <span class="s2">"avg_importance"</span><span class="p">:</span> <span class="nb">float</span><span class="p">(</span>
                <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">([</span><span class="n">e</span><span class="o">.</span><span class="n">importance</span> <span class="k">for</span> <span class="n">e</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">])</span>
            <span class="p">)</span> <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">entries</span> <span class="k">else</span> <span class="mf">0.0</span><span class="p">,</span>
            <span class="s2">"oldest_entry"</span><span class="p">:</span> <span class="nb">min</span><span class="p">(</span>
                <span class="p">(</span><span class="n">e</span><span class="o">.</span><span class="n">timestamp</span> <span class="k">for</span> <span class="n">e</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">entries</span><span class="p">),</span> <span class="n">default</span><span class="o">=</span><span class="kc">None</span>
            <span class="p">),</span>
        <span class="p">}</span>
</code></pre></div>

### 17.10.2　分层记忆管理器 {#hierarchical-memory-manager}

<p>受 MemGPT <a class="book-citation" id="cite-packer2023memgpt-1--section-10" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a> 启发，这一模式把记忆组织为三层：<em>热（hot）</em>（在上下文内，立即可访问）、<em>温（warm）</em>（向量存储，可快速检索）和<em>冷（cold）</em>（归档，容量无上限）。条目根据访问频率与重要性自动升级或降级——类似 CPU 的缓存层级。</p>

<p class="code-caption"><span class="code-number">代码 17.2　</span>实现热/温/冷三层、并支持自动升降级的分层记忆管理器。</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">enum</span><span class="w"> </span><span class="kn">import</span> <span class="n">Enum</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">collections</span><span class="w"> </span><span class="kn">import</span> <span class="n">OrderedDict</span>

<span class="k">class</span><span class="w"> </span><span class="nc">MemoryTier</span><span class="p">(</span><span class="n">Enum</span><span class="p">):</span>
    <span class="c1"># HOT  = "hot"    # In-context: immediate access</span>
    <span class="n">HOT</span>  <span class="o">=</span> <span class="s2">"hot"</span>    <span class="c1"># 在上下文内：立即可访问</span>
    <span class="c1"># WARM = "warm"   # Vector store: fast retrieval</span>
    <span class="n">WARM</span> <span class="o">=</span> <span class="s2">"warm"</span>   <span class="c1"># 向量存储：可快速检索</span>
    <span class="c1"># COLD = "cold"   # Archival: slow but unlimited</span>
    <span class="n">COLD</span> <span class="o">=</span> <span class="s2">"cold"</span>   <span class="c1"># 归档：慢但容量无上限</span>

<span class="k">class</span><span class="w"> </span><span class="nc">HierarchicalMemoryManager</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    Three-tier memory manager inspired by MemGPT.</span>
<span class="sd">    Hot tier is an LRU cache; warm is a vector store;</span>
<span class="sd">    cold is append-only archival storage.</span>
<span class="sd">    受 MemGPT 启发的三层记忆管理器。</span>
<span class="sd">    热层是 LRU 缓存；温层是向量存储；</span>
<span class="sd">    冷层是只追加的归档存储。</span>
<span class="sd">    """</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span>
        <span class="n">vector_store</span><span class="p">:</span> <span class="n">VectorMemoryStore</span><span class="p">,</span>
        <span class="c1"># max entries in hot tier</span>
        <span class="n">hot_capacity</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span>     <span class="c1"># 热层最大条目数</span>
        <span class="n">warm_capacity</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5_000</span><span class="p">,</span>
        <span class="c1"># callable for summarization</span>
        <span class="n">llm_summarize_fn</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>      <span class="c1"># 用于摘要的可调用对象</span>
    <span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">vector_store</span> <span class="o">=</span> <span class="n">vector_store</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">hot_capacity</span> <span class="o">=</span> <span class="n">hot_capacity</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">warm_capacity</span> <span class="o">=</span> <span class="n">warm_capacity</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">summarize</span> <span class="o">=</span> <span class="n">llm_summarize_fn</span>

        <span class="c1"># Hot tier: ordered dict for LRU semantics</span>
        <span class="c1"># 热层：用 OrderedDict 实现 LRU 语义</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="p">:</span> <span class="n">OrderedDict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">MemoryEntry</span><span class="p">]</span> <span class="o">=</span> <span class="n">OrderedDict</span><span class="p">()</span>
        <span class="c1"># Cold tier: append-only list (would be a DB in production)</span>
        <span class="c1"># 冷层：只追加列表（生产环境中通常是一个数据库）</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">cold</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">MemoryEntry</span><span class="p">]</span> <span class="o">=</span> <span class="p">[]</span>

    <span class="c1"># -- Page-in: promote warm -&gt; hot ---------------------------------------</span>
    <span class="c1"># -- 页入（Page-in）：把温层提升到热层 ---------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">page_in</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">k</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">3</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="n">MemoryEntry</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""</span>
<span class="sd">        Retrieve from warm store and promote to hot tier.</span>
<span class="sd">        Evicts least-recently-used hot entries if needed.</span>
<span class="sd">        从温层检索并提升到热层。</span>
<span class="sd">        如有必要会驱逐热层中最近最少使用的条目。</span>
<span class="sd">        """</span>
        <span class="n">candidates</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">vector_store</span><span class="o">.</span><span class="n">retrieve</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">k</span><span class="o">=</span><span class="n">k</span><span class="p">)</span>
        <span class="n">promoted</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">entry</span> <span class="ow">in</span> <span class="n">candidates</span><span class="p">:</span>
            <span class="c1"># use prefix as key</span>
            <span class="n">key</span> <span class="o">=</span> <span class="n">entry</span><span class="o">.</span><span class="n">content</span><span class="p">[:</span><span class="mi">64</span><span class="p">]</span>  <span class="c1"># 用前缀作为键</span>
            <span class="k">if</span> <span class="n">key</span> <span class="ow">not</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="p">:</span>
                <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="p">)</span> <span class="o">&gt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">hot_capacity</span><span class="p">:</span>
                    <span class="bp">self</span><span class="o">.</span><span class="n">_evict_hot</span><span class="p">()</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="p">[</span><span class="n">key</span><span class="p">]</span> <span class="o">=</span> <span class="n">entry</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="o">.</span><span class="n">move_to_end</span><span class="p">(</span><span class="n">key</span><span class="p">)</span>
            <span class="n">promoted</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">entry</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">promoted</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_evict_hot</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="c1"># """Evict LRU entry from hot tier back to warm."""</span>
<span class="w">        </span><span class="sd">"""把热层中的 LRU 条目驱逐回温层。"""</span>
        <span class="c1"># OrderedDict: first item is LRU</span>
        <span class="c1"># OrderedDict：第一个元素即为 LRU</span>
        <span class="n">key</span><span class="p">,</span> <span class="n">entry</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="o">.</span><span class="n">popitem</span><span class="p">(</span><span class="n">last</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
        <span class="c1"># Re-insert into warm store (already there, just update access)</span>
        <span class="c1"># In a real system, we'd update the warm store's metadata</span>
        <span class="c1"># 重新插入温层（实际上已存在，只需更新访问信息）</span>
        <span class="c1"># 在真实系统中我们会更新温层中的元数据</span>

    <span class="c1"># -- Write with tier assignment ------------------------------------------</span>
    <span class="c1"># -- 带层级分配的写入 ------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">write</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span>
        <span class="n">content</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
        <span class="n">importance</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">,</span>
        <span class="n">tier</span><span class="p">:</span> <span class="n">MemoryTier</span> <span class="o">=</span> <span class="n">MemoryTier</span><span class="o">.</span><span class="n">WARM</span><span class="p">,</span>
    <span class="p">)</span> <span class="o">-&gt;</span> <span class="n">MemoryEntry</span><span class="p">:</span>
        <span class="c1"># """Write to the appropriate tier."""</span>
<span class="w">        </span><span class="sd">"""写入到合适的层级。"""</span>
        <span class="k">if</span> <span class="n">tier</span> <span class="o">==</span> <span class="n">MemoryTier</span><span class="o">.</span><span class="n">HOT</span><span class="p">:</span>
            <span class="n">entry</span> <span class="o">=</span> <span class="n">MemoryEntry</span><span class="p">(</span>
                <span class="n">content</span><span class="o">=</span><span class="n">content</span><span class="p">,</span>
                <span class="n">embedding</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">vector_store</span><span class="o">.</span><span class="n">embed_fn</span><span class="p">(</span><span class="n">content</span><span class="p">),</span>
                <span class="n">importance</span><span class="o">=</span><span class="n">importance</span><span class="p">,</span>
            <span class="p">)</span>
            <span class="n">key</span> <span class="o">=</span> <span class="n">content</span><span class="p">[:</span><span class="mi">64</span><span class="p">]</span>
            <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="p">)</span> <span class="o">&gt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">hot_capacity</span><span class="p">:</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">_evict_hot</span><span class="p">()</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="p">[</span><span class="n">key</span><span class="p">]</span> <span class="o">=</span> <span class="n">entry</span>
            <span class="k">return</span> <span class="n">entry</span>

        <span class="k">elif</span> <span class="n">tier</span> <span class="o">==</span> <span class="n">MemoryTier</span><span class="o">.</span><span class="n">WARM</span><span class="p">:</span>
            <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">vector_store</span><span class="o">.</span><span class="n">write</span><span class="p">(</span><span class="n">content</span><span class="p">,</span> <span class="n">importance</span><span class="o">=</span><span class="n">importance</span><span class="p">)</span>

        <span class="k">else</span><span class="p">:</span>  <span class="c1"># COLD  # 冷层</span>
            <span class="n">entry</span> <span class="o">=</span> <span class="n">MemoryEntry</span><span class="p">(</span>
                <span class="n">content</span><span class="o">=</span><span class="n">content</span><span class="p">,</span>
                <span class="c1"># no embedding for cold</span>
                <span class="n">embedding</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([]),</span>  <span class="c1"># 冷层不存 Embedding</span>
                <span class="n">importance</span><span class="o">=</span><span class="n">importance</span><span class="p">,</span>
            <span class="p">)</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">cold</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">entry</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">entry</span>

    <span class="c1"># -- Summarize and compress ---------------------------------------------</span>
    <span class="c1"># -- 摘要与压缩 ---------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">compress_hot_to_warm</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""</span>
<span class="sd">        Summarize hot tier contents and write summary to warm.</span>
<span class="sd">        Called when hot tier is full and new important content arrives.</span>
<span class="sd">        对热层内容做摘要并把摘要写入温层。</span>
<span class="sd">        在热层已满且有新的重要内容到来时调用。</span>
<span class="sd">        """</span>
        <span class="k">if</span> <span class="ow">not</span> <span class="bp">self</span><span class="o">.</span><span class="n">hot</span> <span class="ow">or</span> <span class="ow">not</span> <span class="bp">self</span><span class="o">.</span><span class="n">summarize</span><span class="p">:</span>
            <span class="k">return</span> <span class="kc">None</span>

        <span class="n">hot_contents</span> <span class="o">=</span> <span class="s2">"</span><span class="se">\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">"- </span><span class="si">{</span><span class="n">e</span><span class="o">.</span><span class="n">content</span><span class="si">}</span><span class="s2">"</span> <span class="k">for</span> <span class="n">e</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="o">.</span><span class="n">values</span><span class="p">()</span>
        <span class="p">)</span>
        <span class="n">summary</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">summarize</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">"Summarize these memory entries concisely:</span><span class="se">\n</span><span class="si">{</span><span class="n">hot_contents</span><span class="si">}</span><span class="s2">"</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">vector_store</span><span class="o">.</span><span class="n">write</span><span class="p">(</span><span class="n">summary</span><span class="p">,</span> <span class="n">importance</span><span class="o">=</span><span class="mf">0.7</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">summary</span>

    <span class="c1"># -- Unified retrieval --------------------------------------------------</span>
    <span class="c1"># -- 统一检索 --------------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">retrieve</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">k</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="n">MemoryEntry</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""</span>
<span class="sd">        Retrieve from all tiers, prioritizing hot.</span>
<span class="sd">        Returns up to k entries sorted by relevance.</span>
<span class="sd">        从所有层级检索，热层优先。</span>
<span class="sd">        最多返回按相关性排序的 k 条。</span>
<span class="sd">        """</span>
        <span class="n">results</span> <span class="o">=</span> <span class="p">[]</span>

        <span class="c1"># 1. Check hot tier (exact + semantic)</span>
        <span class="c1"># 1. 检查热层（精确匹配 + 语义匹配）</span>
        <span class="n">q_emb</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">vector_store</span><span class="o">.</span><span class="n">embed_fn</span><span class="p">(</span><span class="n">query</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">entry</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="o">.</span><span class="n">values</span><span class="p">():</span>
            <span class="k">if</span> <span class="n">entry</span><span class="o">.</span><span class="n">embedding</span><span class="o">.</span><span class="n">size</span> <span class="o">&gt;</span> <span class="mi">0</span><span class="p">:</span>
                <span class="n">sim</span> <span class="o">=</span> <span class="nb">float</span><span class="p">(</span>
                    <span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">q_emb</span><span class="p">,</span> <span class="n">entry</span><span class="o">.</span><span class="n">embedding</span><span class="p">)</span>
                    <span class="o">/</span> <span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="n">q_emb</span><span class="p">)</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="n">entry</span><span class="o">.</span><span class="n">embedding</span><span class="p">)</span> <span class="o">+</span> <span class="mf">1e-8</span><span class="p">)</span>
                <span class="p">)</span>
                <span class="k">if</span> <span class="n">sim</span> <span class="o">&gt;</span> <span class="mf">0.7</span><span class="p">:</span>
                    <span class="c1"># +1 hot bonus</span>
                    <span class="n">results</span><span class="o">.</span><span class="n">append</span><span class="p">((</span><span class="n">sim</span> <span class="o">+</span> <span class="mf">1.0</span><span class="p">,</span> <span class="n">entry</span><span class="p">))</span>  <span class="c1"># +1 的热层加成</span>

        <span class="c1"># 2. Retrieve from warm store</span>
        <span class="c1"># 2. 从温层检索</span>
        <span class="n">warm_results</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">vector_store</span><span class="o">.</span><span class="n">retrieve</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">k</span><span class="o">=</span><span class="n">k</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">entry</span> <span class="ow">in</span> <span class="n">warm_results</span><span class="p">:</span>
            <span class="n">results</span><span class="o">.</span><span class="n">append</span><span class="p">((</span><span class="mf">0.5</span><span class="p">,</span> <span class="n">entry</span><span class="p">))</span>

        <span class="c1"># 3. Deduplicate and sort</span>
        <span class="c1"># 3. 去重并排序</span>
        <span class="n">seen</span> <span class="o">=</span> <span class="nb">set</span><span class="p">()</span>
        <span class="n">final</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">score</span><span class="p">,</span> <span class="n">entry</span> <span class="ow">in</span> <span class="nb">sorted</span><span class="p">(</span><span class="n">results</span><span class="p">,</span> <span class="n">reverse</span><span class="o">=</span><span class="kc">True</span><span class="p">):</span>
            <span class="n">key</span> <span class="o">=</span> <span class="n">entry</span><span class="o">.</span><span class="n">content</span><span class="p">[:</span><span class="mi">64</span><span class="p">]</span>
            <span class="k">if</span> <span class="n">key</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">seen</span><span class="p">:</span>
                <span class="n">seen</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">key</span><span class="p">)</span>
                <span class="n">final</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">entry</span><span class="p">)</span>
            <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">final</span><span class="p">)</span> <span class="o">&gt;=</span> <span class="n">k</span><span class="p">:</span>
                <span class="k">break</span>

        <span class="k">return</span> <span class="n">final</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">get_hot_context</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="c1"># """Return hot tier as a formatted context string."""</span>
<span class="w">        </span><span class="sd">"""把热层格式化为上下文字符串返回。"""</span>
        <span class="k">if</span> <span class="ow">not</span> <span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="p">:</span>
            <span class="k">return</span> <span class="s2">""</span>
        <span class="n">lines</span> <span class="o">=</span> <span class="p">[</span><span class="s2">"[Memory Context]"</span><span class="p">]</span>
        <span class="c1"># last 10</span>
        <span class="k">for</span> <span class="n">entry</span> <span class="ow">in</span> <span class="nb">list</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">hot</span><span class="o">.</span><span class="n">values</span><span class="p">())[</span><span class="o">-</span><span class="mi">10</span><span class="p">:]:</span>  <span class="c1"># 最后 10 条</span>
            <span class="n">lines</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="sa">f</span><span class="s2">"  - </span><span class="si">{</span><span class="n">entry</span><span class="o">.</span><span class="n">content</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
        <span class="k">return</span> <span class="s2">"</span><span class="se">\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">lines</span><span class="p">)</span>
</code></pre></div>

### 17.10.3　记忆增强的 Agent 循环 {#memory-augmented-agent-loop}

<p>这一模式由 MemGPT <a class="book-citation" id="cite-packer2023memgpt-2" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a> 提出，并在 CoALA 框架 <a class="book-citation" id="cite-sumers2023coala-1" href="../../references.md#ref-sumers2023coala" aria-label="参考文献 344">[344]</a> 中得到形式化。它通过一个<em>读—行动—反思—写入</em>的循环把记忆系统接入 Agent 的推理流程：响应之前，Agent 检索相关记忆；响应之后，Agent 决定要存什么。LLM 输出中的特殊 Token 会触发记忆操作，从而把对自身持久化的控制权交给模型本身。</p>

<p class="code-caption"><span class="code-number">代码 17.3　</span>完整的记忆增强 Agent 循环，含读—行动—反思—写入四个阶段。</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">re</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Any</span>

<span class="k">class</span><span class="w"> </span><span class="nc">MemoryAugmentedAgent</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    An LLM agent with a full read-act-reflect-write memory cycle.</span>
<span class="sd">    Implements the MemGPT-style self-directed memory management.</span>
<span class="sd">    一个具有完整“读—行动—反思—写入”记忆循环的 LLM Agent。</span>
<span class="sd">    实现 MemGPT 风格的自驱动记忆管理。</span>
<span class="sd">    """</span>

    <span class="n">SYSTEM_PROMPT</span> <span class="o">=</span> <span class="s2">"""You are a memory-augmented AI assistant.</span>
<span class="s2">You have access to persistent memory across conversations.</span>
<span class="s2">At each turn you may issue memory commands:</span>
<span class="s2">  [MEMORY_SEARCH: &lt;query&gt;]  - retrieve relevant memories</span>
<span class="s2">  [MEMORY_WRITE: &lt;content&gt;] - store important information</span>
<span class="s2">  [MEMORY_REFLECT]          - synthesize insights from memory</span>

<span class="s2">Always think step by step. Use memory to avoid repeating mistakes</span>
<span class="s2">and to personalize your responses."""</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span>
        <span class="c1"># callable: messages -&gt; str</span>
        <span class="n">llm_fn</span><span class="p">,</span>                         <span class="c1"># 可调用对象：messages -&gt; str</span>
        <span class="n">memory_manager</span><span class="p">:</span> <span class="n">HierarchicalMemoryManager</span><span class="p">,</span>
        <span class="n">importance_threshold</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.6</span><span class="p">,</span>
        <span class="n">max_memory_tokens</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">1500</span><span class="p">,</span>
    <span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">llm</span> <span class="o">=</span> <span class="n">llm_fn</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">memory</span> <span class="o">=</span> <span class="n">memory_manager</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">importance_threshold</span> <span class="o">=</span> <span class="n">importance_threshold</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">max_memory_tokens</span> <span class="o">=</span> <span class="n">max_memory_tokens</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">conversation_history</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">]</span> <span class="o">=</span> <span class="p">[]</span>

    <span class="c1"># -- Main agent step ----------------------------------------------------</span>
    <span class="c1"># -- Agent 主循环单步 ----------------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">step</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">user_message</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""</span>
<span class="sd">        Full agent step:</span>
<span class="sd">        1. Retrieve relevant memories</span>
<span class="sd">        2. Construct augmented prompt</span>
<span class="sd">        3. Generate response (possibly with memory commands)</span>
<span class="sd">        4. Execute memory commands</span>
<span class="sd">        5. Reflect and consolidate</span>
<span class="sd">        6. Return response to user</span>
<span class="sd">        Agent 的完整单步：</span>
<span class="sd">        1. 检索相关记忆</span>
<span class="sd">        2. 构造增强 Prompt</span>
<span class="sd">        3. 生成响应（可能包含记忆命令）</span>
<span class="sd">        4. 执行记忆命令</span>
<span class="sd">        5. 反思与巩固</span>
<span class="sd">        6. 把响应返回给用户</span>
<span class="sd">        """</span>

        <span class="c1"># Step 1: Retrieve relevant memories</span>
        <span class="c1"># 步骤 1：检索相关记忆</span>
        <span class="n">memories</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">memory</span><span class="o">.</span><span class="n">retrieve</span><span class="p">(</span><span class="n">user_message</span><span class="p">,</span> <span class="n">k</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>
        <span class="n">memory_context</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_format_memories</span><span class="p">(</span><span class="n">memories</span><span class="p">)</span>

        <span class="c1"># Step 2: Construct augmented prompt</span>
        <span class="c1"># 步骤 2：构造增强 Prompt</span>
        <span class="n">messages</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_build_messages</span><span class="p">(</span><span class="n">user_message</span><span class="p">,</span> <span class="n">memory_context</span><span class="p">)</span>

        <span class="c1"># Step 3: Generate response</span>
        <span class="c1"># 步骤 3：生成响应</span>
        <span class="n">raw_response</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">llm</span><span class="p">(</span><span class="n">messages</span><span class="p">)</span>

        <span class="c1"># Step 4: Execute any memory commands in the response</span>
        <span class="c1"># 步骤 4：执行响应中的所有记忆命令</span>
        <span class="n">clean_response</span><span class="p">,</span> <span class="n">memory_ops</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_parse_memory_commands</span><span class="p">(</span>
            <span class="n">raw_response</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_execute_memory_ops</span><span class="p">(</span><span class="n">memory_ops</span><span class="p">,</span> <span class="n">user_message</span><span class="p">,</span> <span class="n">clean_response</span><span class="p">)</span>

        <span class="c1"># Step 5: Auto-write important information</span>
        <span class="c1"># 步骤 5：自动写入重要信息</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_auto_write</span><span class="p">(</span><span class="n">user_message</span><span class="p">,</span> <span class="n">clean_response</span><span class="p">)</span>

        <span class="c1"># Step 6: Update conversation history</span>
        <span class="c1"># 步骤 6：更新对话历史</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">conversation_history</span><span class="o">.</span><span class="n">append</span><span class="p">(</span>
            <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">user_message</span><span class="p">}</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">conversation_history</span><span class="o">.</span><span class="n">append</span><span class="p">(</span>
            <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"assistant"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">clean_response</span><span class="p">}</span>
        <span class="p">)</span>

        <span class="k">return</span> <span class="n">clean_response</span>

    <span class="c1"># -- Memory retrieval and formatting -----------------------------------</span>
    <span class="c1"># -- 记忆检索与格式化 -----------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_format_memories</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">memories</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">MemoryEntry</span><span class="p">])</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="k">if</span> <span class="ow">not</span> <span class="n">memories</span><span class="p">:</span>
            <span class="k">return</span> <span class="s2">""</span>
        <span class="n">lines</span> <span class="o">=</span> <span class="p">[</span><span class="s2">"Relevant memories:"</span><span class="p">]</span>
        <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">m</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">memories</span><span class="p">,</span> <span class="mi">1</span><span class="p">):</span>
            <span class="n">age</span> <span class="o">=</span> <span class="p">(</span><span class="n">datetime</span><span class="o">.</span><span class="n">now</span><span class="p">()</span> <span class="o">-</span> <span class="n">m</span><span class="o">.</span><span class="n">timestamp</span><span class="p">)</span><span class="o">.</span><span class="n">days</span>
            <span class="n">lines</span><span class="o">.</span><span class="n">append</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"  [</span><span class="si">{</span><span class="n">i</span><span class="si">}</span><span class="s2">] (importance=</span><span class="si">{</span><span class="n">m</span><span class="o">.</span><span class="n">importance</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2">, "</span>
                <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">age</span><span class="si">}</span><span class="s2">d ago) </span><span class="si">{</span><span class="n">m</span><span class="o">.</span><span class="n">content</span><span class="si">}</span><span class="s2">"</span>
            <span class="p">)</span>
        <span class="k">return</span> <span class="s2">"</span><span class="se">\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">lines</span><span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_build_messages</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span> <span class="n">user_message</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">memory_context</span><span class="p">:</span> <span class="nb">str</span>
    <span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">]:</span>
        <span class="n">system</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">SYSTEM_PROMPT</span>
        <span class="k">if</span> <span class="n">memory_context</span><span class="p">:</span>
            <span class="n">system</span> <span class="o">+=</span> <span class="sa">f</span><span class="s2">"</span><span class="se">\n\n</span><span class="si">{</span><span class="n">memory_context</span><span class="si">}</span><span class="s2">"</span>
        <span class="n">system</span> <span class="o">+=</span> <span class="sa">f</span><span class="s2">"</span><span class="se">\n\n</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">memory</span><span class="o">.</span><span class="n">get_hot_context</span><span class="p">()</span><span class="si">}</span><span class="s2">"</span>

        <span class="n">messages</span> <span class="o">=</span> <span class="p">[{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"system"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">system</span><span class="p">}]</span>
        <span class="c1"># Include recent conversation history (last 6 turns)</span>
        <span class="c1"># 包含最近的对话历史（最后 6 轮）</span>
        <span class="n">messages</span><span class="o">.</span><span class="n">extend</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">conversation_history</span><span class="p">[</span><span class="o">-</span><span class="mi">6</span><span class="p">:])</span>
        <span class="n">messages</span><span class="o">.</span><span class="n">append</span><span class="p">({</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">user_message</span><span class="p">})</span>
        <span class="k">return</span> <span class="n">messages</span>

    <span class="c1"># -- Memory command parsing ---------------------------------------------</span>
    <span class="c1"># -- 记忆命令解析 ---------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_parse_memory_commands</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span> <span class="n">response</span><span class="p">:</span> <span class="nb">str</span>
    <span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">]]:</span>
        <span class="c1"># """Extract and remove memory commands from response."""</span>
<span class="w">        </span><span class="sd">"""从响应中抽取并移除记忆命令。"""</span>
        <span class="n">ops</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="n">patterns</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s2">"search"</span><span class="p">:</span>  <span class="sa">r</span><span class="s2">"\[MEMORY_SEARCH:\s*(.+?)\]"</span><span class="p">,</span>
            <span class="s2">"write"</span><span class="p">:</span>   <span class="sa">r</span><span class="s2">"\[MEMORY_WRITE:\s*(.+?)\]"</span><span class="p">,</span>
            <span class="s2">"reflect"</span><span class="p">:</span> <span class="sa">r</span><span class="s2">"\[MEMORY_REFLECT\]"</span><span class="p">,</span>
        <span class="p">}</span>
        <span class="n">clean</span> <span class="o">=</span> <span class="n">response</span>
        <span class="k">for</span> <span class="n">op_type</span><span class="p">,</span> <span class="n">pattern</span> <span class="ow">in</span> <span class="n">patterns</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
            <span class="k">for</span> <span class="n">match</span> <span class="ow">in</span> <span class="n">re</span><span class="o">.</span><span class="n">finditer</span><span class="p">(</span><span class="n">pattern</span><span class="p">,</span> <span class="n">response</span><span class="p">,</span> <span class="n">re</span><span class="o">.</span><span class="n">DOTALL</span><span class="p">):</span>
                <span class="n">content</span> <span class="o">=</span> <span class="n">match</span><span class="o">.</span><span class="n">group</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span> <span class="k">if</span> <span class="n">op_type</span> <span class="o">!=</span> <span class="s2">"reflect"</span> <span class="k">else</span> <span class="kc">None</span>
                <span class="n">ops</span><span class="o">.</span><span class="n">append</span><span class="p">({</span><span class="s2">"type"</span><span class="p">:</span> <span class="n">op_type</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">content</span><span class="p">})</span>
                <span class="n">clean</span> <span class="o">=</span> <span class="n">clean</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="n">match</span><span class="o">.</span><span class="n">group</span><span class="p">(</span><span class="mi">0</span><span class="p">),</span> <span class="s2">""</span><span class="p">)</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span>
        <span class="k">return</span> <span class="n">clean</span><span class="p">,</span> <span class="n">ops</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_execute_memory_ops</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span>
        <span class="n">ops</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">],</span>
        <span class="n">user_msg</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
        <span class="n">response</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
    <span class="p">):</span>
        <span class="c1"># """Execute memory commands issued by the LLM."""</span>
<span class="w">        </span><span class="sd">"""执行 LLM 发出的记忆命令。"""</span>
        <span class="k">for</span> <span class="n">op</span> <span class="ow">in</span> <span class="n">ops</span><span class="p">:</span>
            <span class="k">if</span> <span class="n">op</span><span class="p">[</span><span class="s2">"type"</span><span class="p">]</span> <span class="o">==</span> <span class="s2">"search"</span><span class="p">:</span>
                <span class="n">results</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">memory</span><span class="o">.</span><span class="n">retrieve</span><span class="p">(</span><span class="n">op</span><span class="p">[</span><span class="s2">"content"</span><span class="p">],</span> <span class="n">k</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
                <span class="c1"># Page results into hot tier for immediate use</span>
                <span class="c1"># 把结果页入热层，以便立即使用</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">memory</span><span class="o">.</span><span class="n">page_in</span><span class="p">(</span><span class="n">op</span><span class="p">[</span><span class="s2">"content"</span><span class="p">],</span> <span class="n">k</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>

            <span class="k">elif</span> <span class="n">op</span><span class="p">[</span><span class="s2">"type"</span><span class="p">]</span> <span class="o">==</span> <span class="s2">"write"</span><span class="p">:</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">memory</span><span class="o">.</span><span class="n">write</span><span class="p">(</span>
                    <span class="n">op</span><span class="p">[</span><span class="s2">"content"</span><span class="p">],</span>
                    <span class="c1"># explicitly written = important</span>
                    <span class="n">importance</span><span class="o">=</span><span class="mf">0.8</span><span class="p">,</span>  <span class="c1"># 显式写入 = 重要</span>
                    <span class="n">tier</span><span class="o">=</span><span class="n">MemoryTier</span><span class="o">.</span><span class="n">WARM</span><span class="p">,</span>
                <span class="p">)</span>

            <span class="k">elif</span> <span class="n">op</span><span class="p">[</span><span class="s2">"type"</span><span class="p">]</span> <span class="o">==</span> <span class="s2">"reflect"</span><span class="p">:</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">_reflect</span><span class="p">()</span>

    <span class="c1"># -- Auto-write heuristic -----------------------------------------------</span>
    <span class="c1"># -- 自动写入启发式 -----------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_auto_write</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">user_msg</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">response</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
<span class="w">        </span><span class="sd">"""</span>
<span class="sd">        Automatically store important information without explicit command.</span>
<span class="sd">        Uses a simple heuristic: write if response contains facts,</span>
<span class="sd">        decisions, or user preferences.</span>
<span class="sd">        在无显式命令的情况下自动存储重要信息。</span>
<span class="sd">        使用简单的启发式：当响应包含事实、决策或用户偏好时即写入。</span>
<span class="sd">        """</span>
        <span class="n">importance_keywords</span> <span class="o">=</span> <span class="p">[</span>
            <span class="s2">"remember"</span><span class="p">,</span> <span class="s2">"important"</span><span class="p">,</span> <span class="s2">"note that"</span><span class="p">,</span> <span class="s2">"you prefer"</span><span class="p">,</span>
            <span class="s2">"your name is"</span><span class="p">,</span> <span class="s2">"decided to"</span><span class="p">,</span> <span class="s2">"the answer is"</span><span class="p">,</span>
            <span class="s2">"key insight"</span><span class="p">,</span> <span class="s2">"learned that"</span><span class="p">,</span>
        <span class="p">]</span>
        <span class="n">combined</span> <span class="o">=</span> <span class="p">(</span><span class="n">user_msg</span> <span class="o">+</span> <span class="s2">" "</span> <span class="o">+</span> <span class="n">response</span><span class="p">)</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span>
        <span class="k">if</span> <span class="nb">any</span><span class="p">(</span><span class="n">kw</span> <span class="ow">in</span> <span class="n">combined</span> <span class="k">for</span> <span class="n">kw</span> <span class="ow">in</span> <span class="n">importance_keywords</span><span class="p">):</span>
            <span class="n">summary</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">"User: </span><span class="si">{</span><span class="n">user_msg</span><span class="p">[:</span><span class="mi">100</span><span class="p">]</span><span class="si">}</span><span class="s2"> | Agent: </span><span class="si">{</span><span class="n">response</span><span class="p">[:</span><span class="mi">200</span><span class="p">]</span><span class="si">}</span><span class="s2">"</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">memory</span><span class="o">.</span><span class="n">write</span><span class="p">(</span>
                <span class="n">summary</span><span class="p">,</span>
                <span class="n">importance</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">importance_threshold</span><span class="p">,</span>
                <span class="n">tier</span><span class="o">=</span><span class="n">MemoryTier</span><span class="o">.</span><span class="n">WARM</span><span class="p">,</span>
            <span class="p">)</span>

    <span class="c1"># -- Reflection --------------------------------------------------------</span>
    <span class="c1"># -- 反思 --------------------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_reflect</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
<span class="w">        </span><span class="sd">"""</span>
<span class="sd">        Meta-cognitive reflection: synthesize insights from recent memory.</span>
<span class="sd">        Stores high-level insights back into semantic memory.</span>
<span class="sd">        元认知反思：从近期记忆中综合出洞见。</span>
<span class="sd">        将高层洞见写回语义记忆。</span>
<span class="sd">        """</span>
        <span class="n">recent</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">memory</span><span class="o">.</span><span class="n">retrieve</span><span class="p">(</span><span class="s2">"recent important events"</span><span class="p">,</span> <span class="n">k</span><span class="o">=</span><span class="mi">10</span><span class="p">)</span>
        <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">recent</span><span class="p">)</span> <span class="o">&lt;</span> <span class="mi">3</span><span class="p">:</span>
            <span class="c1"># Not enough to reflect on</span>
            <span class="k">return</span>  <span class="c1"># 不够支撑反思</span>

        <span class="n">recent_text</span> <span class="o">=</span> <span class="s2">"</span><span class="se">\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="sa">f</span><span class="s2">"- </span><span class="si">{</span><span class="n">m</span><span class="o">.</span><span class="n">content</span><span class="si">}</span><span class="s2">"</span> <span class="k">for</span> <span class="n">m</span> <span class="ow">in</span> <span class="n">recent</span><span class="p">)</span>
        <span class="n">insight_prompt</span> <span class="o">=</span> <span class="p">[</span>
            <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"system"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"You extract high-level insights."</span><span class="p">},</span>
            <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span>
                <span class="sa">f</span><span class="s2">"Based on these memories, what are 2-3 key insights?</span><span class="se">\n</span><span class="s2">"</span>
                <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">recent_text</span><span class="si">}</span><span class="se">\n</span><span class="s2">Respond with bullet points only."</span><span class="p">},</span>
        <span class="p">]</span>
        <span class="n">insights</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">llm</span><span class="p">(</span><span class="n">insight_prompt</span><span class="p">)</span>
        <span class="c1"># Store each insight as a high-importance semantic memory</span>
        <span class="c1"># 把每条洞见作为高重要性的语义记忆存储</span>
        <span class="k">for</span> <span class="n">line</span> <span class="ow">in</span> <span class="n">insights</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">"</span><span class="se">\n</span><span class="s2">"</span><span class="p">):</span>
            <span class="n">line</span> <span class="o">=</span> <span class="n">line</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span><span class="o">.</span><span class="n">lstrip</span><span class="p">(</span><span class="s2">"*-"</span><span class="p">)</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span>
            <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">line</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mi">20</span><span class="p">:</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">memory</span><span class="o">.</span><span class="n">write</span><span class="p">(</span>
                    <span class="sa">f</span><span class="s2">"[INSIGHT] </span><span class="si">{</span><span class="n">line</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span>
                    <span class="n">importance</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span>
                    <span class="n">tier</span><span class="o">=</span><span class="n">MemoryTier</span><span class="o">.</span><span class="n">WARM</span><span class="p">,</span>
                <span class="p">)</span>
</code></pre></div>

<div class="admonition tip"><p class="admonition-title">读—行动—反思—写入循环</p>
<p>记忆增强的 Agent 循环实现了一个四阶段的认知周期：</p>

<ol>
<li><p><strong>读（Read）：</strong> 在行动之前，检索相关记忆以指引响应。</p></li>
<li><p><strong>行动（Act）：</strong> 基于检索到的上下文生成响应。</p></li>
<li><p><strong>反思（Reflect）：</strong> 定期从已累积的记忆中综合出更高层的洞见。</p></li>
<li><p><strong>写入（Write）：</strong> 选择性地把重要的新信息提交到持久化存储中。</p></li>
</ol>

<p>这一循环呼应了军事战略中的 <em>observe-orient-decide-act</em>（OODA）循环，以及认知心理学中的 <em>encode-store-retrieve</em>（编码—存储—检索）模型。关键洞见在于：记忆并非被动的存储库，而是认知过程的<em>主动参与者</em>。</p>
</div>

## 17.11　智能体记忆的最新进展 {#sec:memory-recent}

<p>上文描述的记忆系统奠定了基础模式。近期若干工作进一步推进了边界：</p>

### 17.11.1　CoALA：面向语言 Agent 的认知架构 {#coala-cognitive-architectures-for-language-agents}

<p>Sumers 等人 <a class="book-citation" id="cite-sumers2023coala-1--section-11" href="../../references.md#ref-sumers2023coala" aria-label="参考文献 344">[344]</a> 提出了 <em>面向语言 Agent 的认知架构</em>（Cognitive Architectures for Language Agents, CoALA），这是一个借用认知科学与符号 AI 原则、用于组织日益庞杂的 LLM Agent 生态的统一框架。CoALA 将一个语言 Agent 拆解为：</p>

<ul>
<li><p><strong>模块化记忆</strong>：工作记忆（上下文窗口）、情景记忆（过往经验）、语义记忆（世界知识）以及程序记忆（动作 schema）——与 <a href="#sec:memory-taxonomy">17.2</a> 节的分类体系相对应。</p></li>
<li><p><strong>结构化的动作空间</strong>：内部动作（推理、检索、记忆写入）与外部动作（工具调用 Tool Calling、与环境交互）。</p></li>
<li><p><strong>决策循环</strong>：一个广义的“感知—规划—行动”循环，其中带有显式的检索与写入步骤。</p></li>
</ul>

<p>CoALA 的贡献与其说是一个新系统，不如说是一种<em>设计语言</em>：它提供了一种系统化的方式来分析已有 Agent 并识别缺失能力，对从业者而言是一个有用的参考架构。</p>

### 17.11.2　Mem0：生产级记忆层 {#mem0-production-scale-memory-layer}

<p>Mem0 <a class="book-citation" id="cite-chhikara2025mem0-1--section-11" href="../../references.md#ref-chhikara2025mem0" aria-label="参考文献 53">[53]</a> 致力于弥合研究型记忆系统与生产部署之间的鸿沟。核心思路：</p>

<ul>
<li><p><strong>自动抽取</strong>：Mem0 并不依赖 LLM 显式发出记忆写入命令，而是自动从对话轮次中抽取显著事实，并将其巩固到持久化存储中。</p></li>
<li><p><strong>基于图的记忆</strong>：除了扁平的向量存储之外，Mem0 还在抽取出的实体与事实之上维护一张<em>关系图</em>，以支持多跳记忆查询（“用户在项目 Y 的语境下，对话题 X 说过什么？”）。</p></li>
<li><p><strong>记忆压缩</strong>：冗余或被取代的事实会自动合并，使记忆存储保持紧凑与最新。</p></li>
</ul>

<p>在 LOCOMO 基准上，Mem0 相对 OpenAI 的基线记忆取得了 26% 的相对提升，与全上下文方案相比，p95 延迟降低 91%，Token 成本降低 <span class="arithmatex">\(&gt;\)</span>90%。</p>

### 17.11.3　睡眠时计算（Sleep-Time Compute）：离线记忆处理 {#sleep-time-compute-offline-memory-processing}

<p>Lin 等人 <a class="book-citation" id="cite-lin2025sleeptime-1" href="../../references.md#ref-lin2025sleeptime" aria-label="参考文献 218">[218]</a> 提出了 <em>sleep-time compute</em>（睡眠时计算）范式，让 Agent 在用户交互的<em>间隔之间</em>处理并巩固记忆，而不是只在查询时才计算。其类比是生物的睡眠——大脑在睡眠中巩固记忆并预先建立有用的联想。</p>

##### 工作机制。 {#how-it-works.}

<p>在空闲时段（“睡眠”）中，Agent 会：</p>

<ol>
<li><p>基于当前上下文预测未来可能出现的查询。</p></li>
<li><p>预先计算推理链、摘要与结构化表示。</p></li>
<li><p>把这些预计算产物存储下来，以便测试时（test-time）推理可检索并复用。</p></li>
</ol>

##### 效果。 {#results.-dup4}

<p>在推理类基准上，sleep-time compute 把达成等价准确率所需的测试时计算降低约 <span class="arithmatex">\(5\times\)</span>。如果在同一上下文的多个相关查询上进行摊销，平均每次查询的成本下降 <span class="arithmatex">\(2.5\times\)</span>。当用户查询是<em>可预测</em>的——即上下文强约束了可能被问到的问题时——该方法最为有效。</p>

<div class="admonition tip"><p class="admonition-title">把记忆巩固视作离线 RL</p>
<p>Sleep-time compute 可以看作<em>离线 Policy 改进</em>：在空闲时间，Agent 利用已经收集到的数据（过往交互）改进其记忆表示（Policy），无需与环境进行新的交互。这与离线 RL 方法（见第 <a href="../08-preference-optimization/index.md#ch:po-variants">8</a> 章）相通——后者从一个静态的轨迹数据集学习。</p>
</div>

### 17.11.4　A-MEM：受 Zettelkasten 启发的智能体记忆 {#a-mem-zettelkasten-inspired-agentic-memory}

<p>A-MEM <a class="book-citation" id="cite-xu2025amem-1--section-11" href="../../references.md#ref-xu2025amem" aria-label="参考文献 400">[400]</a> 引入了一种借鉴 <em>Zettelkasten</em> 方法的记忆系统——该方法是一种基于密集互联原子笔记的笔记系统——为 LLM Agent 实现动态、自组织的记忆。</p>

##### 关键设计原则。 {#key-design-principles.}

<ul>
<li><p><strong>结构化笔记。</strong> 每条记忆条目都不是原始文本块，而是一条带有多种结构化属性的<em>笔记</em>：上下文描述、关键词、标签，以及指向相关笔记的显式链接。这些元数据使检索比单纯的 Embedding 相似度更丰富。</p></li>
<li><p><strong>动态链接。</strong> 当一条新记忆被加入时，系统会分析已有记忆，识别语义上有意义的关联并建立双向链接。结果是一张<em>知识网络</em>，而不是扁平的列表。</p></li>
<li><p><strong>记忆演化。</strong> 关键之处在于，新增笔记可以<em>触发对已有笔记的更新</em>——随着 Agent 理解的加深，对它们的上下文表示与属性进行细化。这让记忆成为一个随时间不断演进的活体结构，而非静态归档。</p></li>
<li><p><strong>Agent 驱动的组织。</strong> 与固定 schema 的记忆系统不同，A-MEM 让 LLM 自身决定如何组织、链接与更新记忆——使组织结构能够自适应任务领域。</p></li>
</ul>

##### 效果。 {#results.-1-dup3}

<p>在六个基础模型、多种多会话推理任务上，A-MEM 稳定优于扁平向量存储、基于摘要的记忆与图数据库方案，说明记忆<em>如何</em>组织与<em>存什么</em>同等重要。</p>

## 17.12　主动记忆架构 {#sec:proactive-memory}

<p>长时程智能体任务中的一个根本性挑战是<strong>行为状态衰减（behavioral state decay）</strong>：在数十次工具调用之后，执行 Agent 会丢失对原始目标、约束与已积累上下文的追踪。标准记忆系统是反应式地解决这一问题——Agent 在需要信息时才查询记忆。<strong>主动记忆架构</strong> <a class="book-citation" id="cite-meta2026remember-1" href="../../references.md#ref-meta2026remember" aria-label="参考文献 253">[253]</a>（Meta AI，2026 年 7 月，《Remember When It Matters》）采取了不同的方法：一个专门的记忆 Agent <em>监控</em>执行者，并在执行者<em>漂移之前</em>注入约束。</p>

<div class="admonition note"><p class="admonition-title">主动记忆 Agent 架构</p>
<p>系统由两个解耦的 Agent 组成：</p>

<ol>
<li><p><strong>执行 Agent（Executor Agent）</strong>：负责任务执行——工具调用、推理、输出生成。维持一个聚焦于当前子任务的精简上下文。</p></li>
<li><p><strong>记忆 Agent（Memory Agent）</strong>：并行运行，监控执行者的动作流。维护以下内容的压缩表示：</p>

<ul>
<li><p>长期目标与成功标准</p></li>
<li><p>生效中的约束（安全规则、用户偏好、资源限制）</p></li>
<li><p>轨迹早期做出的关键决策</p></li>
</ul></li>
</ol>

<p>记忆 Agent 只在检测到漂移时——即执行者近期的动作表明它忘记了一条约束或正在偏离既定目标时——才把<strong>提醒注入</strong>执行者的上下文。这是与反应式记忆的关键区别：记忆 Agent <em>主动</em>行动，赶在错误累积之前。</p>
</div>

<div class="admonition example"><p class="admonition-title">主动记忆：实证结果</p>
<p>在长时程智能体基准上（每个任务 50+ 次工具调用）：</p>

<ul>
<li><p>在需要持续遵守约束的任务（如“绝不修改 /prod 中的文件”“始终注明来源”）上<strong>成功率显著更高</strong>。</p></li>
<li><p><strong>API 成本增加 <span class="arithmatex">\(\sim\)</span>40%</strong>：记忆 Agent 为监控与注入增加了 LLM 调用。这是主要的权衡代价。</p></li>
<li><p><strong>优雅降级</strong>：没有记忆 Agent 时，执行者在 <span class="arithmatex">\(\sim\)</span>30 次工具调用后成功率急剧下降；有了它，性能在 100+ 次调用中保持稳定。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">RL 联系：把记忆 Agent 视为状态汇总器</p>
<p>主动记忆架构在 RL 框架中有一个自然的解释：</p>

<ul>
<li><p><strong>记忆 Agent</strong> 起到<strong>状态汇总器</strong>的作用——类似于 actor-critic 架构中的注意力机制，它为价值函数选择性加权相关的历史状态。</p></li>
<li><p>在 POMDP（部分可观测 MDP）中，记忆 Agent 维护<strong>信念状态</strong>：与未来决策相关的历史压缩充分统计量。</p></li>
<li><p>注入机制类似于非对称 actor-critic 训练中的<strong>特权信息</strong>：记忆 Agent 可以访问完整轨迹历史，而执行者只操作压缩后的上下文。</p></li>
<li><p>40% 的成本增加反映了<strong>马尔可夫状态表示</strong>（昂贵但充分）与<strong>截断上下文</strong>（廉价但有损）之间的根本权衡。</p></li>
</ul>

<p>这一架构为 RL 训练指明了一个有前景的方向：联合优化执行者策略与记忆 Agent 的注入策略，让记忆 Agent 学习在<em>何时</em>注入<em>什么</em>以最大化任务成功。</p>
</div>

## 17.13　小结 {#sec:memory-summary}

<p>智能体记忆系统是有能力的 AI Agent 的基础组件，用以应对有限上下文窗口这一根本限制。我们已综述：</p>

<ul>
<li><p>一个<strong>四分类法</strong>（工作记忆、情景记忆、语义记忆、程序记忆），对应认知科学，并反映了截然不同的工程需求。</p></li>
<li><p><strong>五大架构家族</strong>：基于 RAG、基于摘要、基于图、键值网络，以及分层虚拟上下文（MemGPT）。</p></li>
<li><p><strong>四种核心操作</strong>：写入（含重要性打分与冲突检测）、读取/检索（含时间衰减与查询扩展）、更新（含冲突解决与巩固），以及反思（元认知洞见生成）。</p></li>
<li><p><strong>多轮与多智能体扩展</strong>：用户建模、会话连续性、共享记忆池与 blackboard 架构。</p></li>
<li><p><strong>记忆系统的 RL 训练</strong>：记忆操作的 Reward 信号、学习“记什么”，以及记忆增强的 Policy 优化。</p></li>
<li><p><strong>主动记忆</strong>：解耦的记忆 Agent 监控执行者行为，在漂移累积之前注入提醒——从反应式检索转向预判式干预。</p></li>
</ul>

<p>该领域仍在快速演进。关键的开放挑战包括：（1）<em>记忆落地（memory grounding）</em>——确保被检索到的记忆被忠实地纳入回答，而非被忽略或被幻觉所覆盖；（2）<em>可扩展的一致性</em>——在大型多智能体系统中维持连贯的共享记忆；（3）<em>保护隐私的记忆</em>——在不损害用户数据的前提下实现个性化；（4）<em>主动注入策略</em>——学习在何时、以何种内容提醒执行者而不淹没其上下文。随着上下文窗口的增长，上下文内记忆与外部记忆之间的边界会发生移动，但对<em>有选择、有结构、可检索</em>的信息存储的根本需求将一直存在。</p>
