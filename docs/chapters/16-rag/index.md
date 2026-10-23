# 第 16 章 · 检索增强生成（RAG） {#retrieval-augmented-generation-rag}

<p>检索增强生成（Retrieval-Augmented Generation, RAG） <a class="book-citation" id="cite-lewis2020retrieval-1" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a> 已成为在生产环境中部署大型语言模型时最具实际影响力的技术之一。RAG 不再仅依赖训练时编码进模型权重中的知识，而是为 LLM 配备一个动态、可更新的外部记忆——使其能够在广泛的知识密集型任务中给出准确、有据可查且可验证的响应。</p>

## 16.1　动机与问题陈述 {#motivation-and-problem-statement}

<div class="admonition note"><p class="admonition-title">为什么 LLM 需要外部知识</p>
<p>大型语言模型以<em>参数化</em>方式存储知识——在训练过程中压缩进数十亿个权重之中。这带来了三个根本性局限：</p>

<ol>
<li><p><strong>幻觉</strong>：当查询超出模型可靠知识边界时，模型会自信地生成听起来合理但事实错误的陈述。</p></li>
<li><p><strong>知识过时</strong>：训练数据存在截止日期；模型无法了解训练之后发生的事件、论文或产品更新。</p></li>
<li><p><strong>领域特异性</strong>：通用模型缺乏对专有代码库、内部文档、专业法规或企业数据的深入了解。</p></li>
</ol>
</div>

### 16.1.1　参数化知识 vs. 非参数化知识 {#parametric-vs.-non-parametric-knowledge}

<p>我们可以将两种知识来源之间的区别形式化。令 <span class="arithmatex">\(\mathcal{M}_\theta\)</span> 表示参数为 <span class="arithmatex">\(\theta\)</span> 的语言模型，<span class="arithmatex">\(\mathcal{D} = \{d_1, d_2, \ldots, d_N\}\)</span> 为外部文档语料库。在每种范式下，给定查询 <span class="arithmatex">\(q\)</span> 生成回答 <span class="arithmatex">\(a\)</span> 的概率为：</p>



<div class="arithmatex">\[\begin{align*}
P_{\text{parametric}}(a \mid q) &amp;= P_{\mathcal{M}_\theta}(a \mid q) \tag{16.1}\\[6pt]

  P_{\text{RAG}}(a \mid q, \mathcal{D}) &amp;= \sum_{d \in \mathcal{D}} P_{\mathcal{M}_\theta}(a \mid q, d)\,
    P_{\text{ret}}(d \mid q, \mathcal{D})
\tag{16.2}
\end{align*}\]</div>

<p>其中 <span class="arithmatex">\(P_{\text{ret}}(d \mid q, \mathcal{D})\)</span> 是关于文档的检索分布。RAG 对检索到的证据进行边缘化，使生成过程接地于非参数化知识。</p>

<div class="admonition tip"><p class="admonition-title">图书馆类比</p>
<p>可以把参数化 LLM 想象成一位记住了一整座庞大图书馆但已毕业多年的学者。RAG 为这位学者发放了一张图书证——他们可以实时查阅资料、引用来源，并在需要查证文献时坦然承认，而不是凭记忆瞎猜。</p>
</div>

### 16.1.2　何时选择 RAG vs. 微调 vs. 长上下文 {#when-to-use-rag-vs.-fine-tuning-vs.-long-context}

<p class="table-caption" id="table-16-1"><span class="table-number">表 16.1　</span>决策指南：RAG vs. 微调 vs. 长上下文</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>标准</strong></th><th scope="col"><strong>RAG</strong></th><th scope="col"><strong>微调</strong></th><th scope="col"><strong>长上下文</strong></th><th scope="col"><strong>RAG + 微调</strong></th></tr></thead>
<tbody><tr><td>知识频繁更新</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr>
<tr><td>需要引用 / 接地</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td>✓</td></tr>
<tr><td>专有大型语料库</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr>
<tr><td>适配风格 / 格式</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr>
<tr><td>教授新推理技能</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr>
<tr><td>语料库可放入上下文窗口</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td></tr>
<tr><td>要求低延迟</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td></tr></tbody>
</table>
</div>

<div class="admonition warning"><p class="admonition-title">常见误解</p>
<p>RAG <em>不是</em>微调的替代品。微调教会模型<em>如何</em>推理和响应；RAG 则提供<em>推理的内容</em>。两者是互补的。一个经过良好指令跟随微调的模型比基础模型更能有效利用检索到的上下文。</p>
</div>

## 16.2　核心 RAG 架构 {#core-rag-architecture}

<p>一个标准的 RAG 系统由两个阶段构成：处理和存储文档的<strong>离线索引流水线</strong>，以及为查询提供服务的<strong>在线检索—生成流水线</strong>。</p>

### 16.2.1　完整流水线图 {#full-pipeline-diagram}

<figure id="fig:rag_arch"><a href="../../assets/figures/fig_050_rag_arch.png" target="_blank" rel="noopener" aria-label="查看图 16.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_050_rag_arch.png" alt="端到端 RAG 架构。离线流水线（蓝色）对文档进行一次性索引；在线流水线（绿色/橙色）在推理时为每个查询提供服务。" loading="lazy"></a><figcaption><span class="figure-number">图 16.1　</span>端到端 RAG 架构。离线流水线（蓝色）对文档进行一次性索引；在线流水线（绿色/橙色）在推理时为每个查询提供服务。</figcaption></figure>

### 16.2.2　索引流水线 {#indexing-pipeline}

##### 文档加载。 {#document-loading.}

<p>文档以异构格式抵达（PDF、HTML、Markdown、DOCX、代码）。加载器提取干净的文本，并保留元数据（来源 URL、页码、章节标题、时间戳），这些元数据会与 Embedding 一并存储，用于过滤和引用。</p>

##### 分块（Chunking）。 {#chunking.}

<p>长文档必须切分成可放入 Embedding 模型上下文窗口（通常为 512 个 Token）并保持语义连贯的块。分块策略是 RAG 系统设计中影响最大的决策之一（参见 Section <a href="#sec:chunking">16.4</a>）。</p>

##### Embedding。 {#embedding.}

<p>每个块 <span class="arithmatex">\(c_i\)</span> 通过 Embedding 模型 <span class="arithmatex">\(f_\phi\)</span> 编码为稠密向量 <span class="arithmatex">\(\mathbf{e}_i = f_\phi(c_i) \in \mathbb{R}^d\)</span>。这些向量与原始文本和元数据一同存储到向量数据库中。</p>

### 16.2.3　检索 {#retrieval}

<p>给定查询 <span class="arithmatex">\(q\)</span>，检索步骤将其编码为 <span class="arithmatex">\(\mathbf{q} = f_\phi(q)\)</span>，并通过余弦相似度找出最相似的 <span class="arithmatex">\(k\)</span> 个块：</p>



<div class="arithmatex">\[\text{sim}(\mathbf{q}, \mathbf{e}_i) = \frac{\mathbf{q} \cdot \mathbf{e}_i}{\|\mathbf{q}\|\,\|\mathbf{e}_i\|}
\tag{16.3}\]</div>

<p>返回 top-<span class="arithmatex">\(k\)</span> 块 <span class="arithmatex">\(\mathcal{C}_k = \{c_{(1)}, \ldots, c_{(k)}\}\)</span> 作为上下文。</p>

### 16.2.4　生成 {#generation}

<p>检索到的块被注入到一个 Prompt 模板中：</p>

<p class="code-caption"><span class="code-number">代码 16.1　</span>标准 RAG Prompt 模板</p>
<div class="highlight"><pre><span></span><code><span class="n">SYSTEM_PROMPT</span> <span class="o">=</span> <span class="s2">"""You are a helpful assistant. Answer the question using ONLY</span>
<span class="s2">the provided context. If the context does not contain enough information,</span>
<span class="s2">say so explicitly. Cite your sources using [Doc N] notation."""</span>

<span class="k">def</span><span class="w"> </span><span class="nf">build_rag_prompt</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">chunks</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">])</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="n">context_str</span> <span class="o">=</span> <span class="s2">"</span><span class="se">\n\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span>
        <span class="sa">f</span><span class="s2">"[Doc </span><span class="si">{</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="si">}</span><span class="s2">] (Source: </span><span class="si">{</span><span class="n">c</span><span class="p">[</span><span class="s1">'source'</span><span class="p">]</span><span class="si">}</span><span class="s2">, Page: </span><span class="si">{</span><span class="n">c</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s1">'page'</span><span class="p">,</span><span class="s1">'N/A'</span><span class="p">)</span><span class="si">}</span><span class="s2">)</span><span class="se">\n</span><span class="si">{</span><span class="n">c</span><span class="p">[</span><span class="s1">'text'</span><span class="p">]</span><span class="si">}</span><span class="s2">"</span>
        <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">c</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">chunks</span><span class="p">)</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="sa">f</span><span class="s2">"""</span><span class="si">{</span><span class="n">SYSTEM_PROMPT</span><span class="si">}</span>

<span class="s2">Context:</span>
<span class="si">{</span><span class="n">context_str</span><span class="si">}</span>

<span class="s2">Question: </span><span class="si">{</span><span class="n">query</span><span class="si">}</span>

<span class="s2">Answer:"""</span>
</code></pre></div>

## 16.3　检索方法 {#retrieval-methods}

### 16.3.1　稀疏检索：BM25 与 TF-IDF {#sparse-retrieval-bm25-and-tf-idf}

<p>稀疏检索方法将文档和查询表示为词汇表上的高维稀疏向量。给定查询 <span class="arithmatex">\(q\)</span>（含词项 <span class="arithmatex">\(t_1, \ldots, t_n\)</span>）时，针对文档 <span class="arithmatex">\(d\)</span> 的经典 BM25 评分函数 <a class="book-citation" id="cite-robertson2009probabilistic-1" href="../../references.md#ref-robertson2009probabilistic" aria-label="参考文献 309">[309]</a> 为：</p>



<div class="arithmatex">\[\text{BM25}(d, q) = \sum_{i=1}^{n} \text{IDF}(t_i) \cdot
    \frac{f(t_i, d) \cdot (k_1 + 1)}{f(t_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
\tag{16.4}\]</div>

<p>其中 <span class="arithmatex">\(f(t_i, d)\)</span> 为词频，<span class="arithmatex">\(|d|\)</span> 为文档长度，<span class="arithmatex">\(\text{avgdl}\)</span> 为平均文档长度，<span class="arithmatex">\(k_1 \in [1.2, 2.0]\)</span>、<span class="arithmatex">\(b = 0.75\)</span> 为可调参数。</p>

<div class="admonition note"><p class="admonition-title">稀疏检索仍然占优的场景</p>
<ul>
<li><p><strong>精确关键词匹配</strong>：产品代号、错误代码、专有名词、罕见词项</p></li>
<li><p><strong>低资源领域</strong>：稠密模型的训练数据不足</p></li>
<li><p><strong>可解释性</strong>：容易调试某个文档为何被检索出来</p></li>
<li><p><strong>速度</strong>：无需 GPU；借助倒排索引可扩展到数十亿文档</p></li>
<li><p><strong>未登录词</strong>：Embedding 训练时未见过的新术语</p></li>
</ul>
</div>

### 16.3.2　稠密检索：DPR {#dense-retrieval-dpr}

<p>稠密段落检索（Dense Passage Retrieval, DPR） <a class="book-citation" id="cite-karpukhin2020dense-1" href="../../references.md#ref-karpukhin2020dense" aria-label="参考文献 180">[180]</a> 使用两个独立的、基于 BERT 的编码器——一个<em>查询编码器</em> <span class="arithmatex">\(E_Q\)</span> 和一个<em>段落编码器</em> <span class="arithmatex">\(E_P\)</span>——通过对比 Loss 进行训练，使相关的查询—段落对在 Embedding 空间中彼此靠近。</p>

##### 双编码器（Bi-Encoder）架构。 {#bi-encoder-architecture.}



<div class="arithmatex">\[\text{sim}(q, p) = E_Q(q)^\top E_P(p)
\tag{16.5}\]</div>

##### 使用 Batch 内负样本进行训练。 {#training-with-in-batch-negatives.}

<p>给定包含 <span class="arithmatex">\(B\)</span> 个查询—段落对的 Batch <span class="arithmatex">\(\{(q_i, p_i^+)\}_{i=1}^B\)</span>，对比 Loss 将 Batch 中所有其他段落视为负样本：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{DPR}} = -\frac{1}{B} \sum_{i=1}^{B}
    \log \frac{\exp\!\left(E_Q(q_i)^\top E_P(p_i^+) / \tau\right)}
              {\sum_{j=1}^{B} \exp\!\left(E_Q(q_i)^\top E_P(p_j) / \tau\right)}
\tag{16.6}\]</div>

<p>其中 <span class="arithmatex">\(\tau\)</span> 是温度超参数。硬负样本（词面相似但语义无关的段落）对训练强检索器至关重要。</p>

##### 近似最近邻搜索。 {#approximate-nearest-neighbor-search.}

<p>在大规模场景下，对数百万个 Embedding 进行穷举搜索不可行。FAISS <a class="book-citation" id="cite-johnson2019billion-1" href="../../references.md#ref-johnson2019billion" aria-label="参考文献 171">[171]</a>（Facebook AI Similarity Search）提供了高效的近似最近邻（Approximate Nearest Neighbor, ANN）搜索，所用方法包括：</p>

<ul>
<li><p><strong>IVF（倒排文件索引，Inverted File Index）</strong>：将向量聚类到 Voronoi 单元中；只搜索邻近的单元</p></li>
<li><p><strong>HNSW（分层可导航小世界，Hierarchical Navigable Small World）</strong> <a class="book-citation" id="cite-malkov2018efficient-1" href="../../references.md#ref-malkov2018efficient" aria-label="参考文献 248">[248]</a>：基于图的索引，搜索复杂度 <span class="arithmatex">\(O(\log N)\)</span></p></li>
<li><p><strong>PQ（乘积量化，Product Quantization）</strong>：压缩向量以减少内存占用</p></li>
</ul>

### 16.3.3　基于倒数排名融合的混合检索 {#hybrid-retrieval-with-reciprocal-rank-fusion}

<p>混合检索结合稀疏和稠密得分。一个简单的线性组合为：</p>



<div class="arithmatex">\[s_{\text{hybrid}}(d, q) = \alpha \cdot s_{\text{dense}}(d, q) + (1-\alpha) \cdot s_{\text{sparse}}(d, q)
\tag{16.7}\]</div>

<p>然而，来自不同系统的分数无法直接比较。<strong>倒数排名融合（Reciprocal Rank Fusion, RRF）</strong> <a class="book-citation" id="cite-cormack2009reciprocal-1" href="../../references.md#ref-cormack2009reciprocal" aria-label="参考文献 62">[62]</a> 通过在排名而非分数上操作来规避这个问题：</p>

<span id="eq:rrf"></span>

<div class="arithmatex">\[\text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k + \text{rank}_r(d)}

\tag{16.8}\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{R}\)</span> 是排名列表集合（如 BM25 排名和稠密排名），<span class="arithmatex">\(\text{rank}_r(d)\)</span> 是文档 <span class="arithmatex">\(d\)</span> 在列表 <span class="arithmatex">\(r\)</span> 中的排名，<span class="arithmatex">\(k = 60\)</span> 是平滑常数，用于削弱排名极高的文档带来的影响。</p>

<div class="admonition example"><p class="admonition-title">RRF 计算示例</p>
<p>假设 BM25 将文档 <span class="arithmatex">\(d\)</span> 排在第 3 位，而稠密检索将其排在第 7 位。当 <span class="arithmatex">\(k = 60\)</span> 时：</p>



<div class="arithmatex">\[\text{RRF}(d) = \frac{1}{60 + 3} + \frac{1}{60 + 7} = \frac{1}{63} + \frac{1}{67} \approx 0.0159 + 0.0149 = 0.0308\]</div>

<p>而一个在两个列表中都排第 1 位的文档得分为 <span class="arithmatex">\(\frac{1}{61} + \frac{1}{61} \approx 0.0328\)</span>。</p>
</div>

### 16.3.4　学习式稀疏检索：SPLADE 与 SPLADEv2 {#learned-sparse-retrieval-splade-and-spladev2}

<div class="admonition tip"><p class="admonition-title">为什么需要 SPLADE？</p>
<p>传统的稀疏检索（BM25）依赖精确的词面匹配——当查询说 &quot;car&quot; 而文档说 &quot;automobile&quot; 时它就会失败。稠密检索（DPR）能捕捉语义但缺乏可解释性，查询时需要 GPU，且索引体积大。<strong>SPLADE</strong> 兼具二者的优点：稀疏向量（像 BM25 一样支持快速的倒排索引查找）配合可学习的语义扩展（像稠密模型一样处理同义词和相关概念）。</p>
</div>

##### SPLADE（v1）—— 核心思想。 {#splade-v1-core-idea.}

<p>SPLADE（Sparse Lexical and Expansion Model） <a class="book-citation" id="cite-formal2021splade-1" href="../../references.md#ref-formal2021splade" aria-label="参考文献 95">[95]</a> 使用预训练的掩码语言模型（如 BERT/DistilBERT），为每个文档或查询生成一个覆盖<em>整个词汇表</em>的稀疏向量。关键洞见在于：MLM 头部已经知道文本中每个位置在语义上与哪些词相关——SPLADE 把这种知识重新利用为词项重要性权重。</p>

##### 架构。 {#architecture.}

<p>给定输入文本 <span class="arithmatex">\(x = [x_1, \ldots, x_n]\)</span>：</p>

<ol>
<li><p>通过 Transformer 编码器，并经由 MLM 头部得到上下文表示 <span class="arithmatex">\(\mathbf{H} \in \mathbb{R}^{n \times |\mathcal{V}|}\)</span></p></li>
<li><p>跨位置聚合，并应用一个饱和激活：</p></li>
</ol>



<div class="arithmatex">\[w_t(x) = \log\!\left(1 + \text{ReLU}\!\left(\max_{i \in [1,n]} \mathbf{H}_i[t]\right)\right)
\tag{16.9}\]</div>

<p>其中 <span class="arithmatex">\(\mathbf{H}_i[t]\)</span> 是输入位置 <span class="arithmatex">\(i\)</span> 处针对词汇 Token <span class="arithmatex">\(t\)</span> 的 MLM logit。</p>

<ul>
<li><p><span class="arithmatex">\(\log(1 + \cdot)\)</span> 饱和防止任何单个词项占主导（类似 BM25 中的 TF 饱和）</p></li>
<li><p>ReLU 保证稀疏性——绝大多数词汇项的权重为零</p></li>
<li><p>跨位置的 <span class="arithmatex">\(\max\)</span> 池化为每个词项从文本任意位置中捕获最强信号</p></li>
<li><p><strong>扩展</strong>：即使是原文中<em>未出现</em>的 Token 也可获得非零权重（例如关于 &quot;neural networks&quot; 的文档可能为 &quot;deep learning&quot;、&quot;AI&quot;、&quot;backpropagation&quot; 赋权）</p></li>
</ul>

##### 打分。 {#scoring.}

<p>查询和文档分别映射为稀疏向量 <span class="arithmatex">\(\mathbf{w}^q, \mathbf{w}^d \in \mathbb{R}^{|\mathcal{V}|}\)</span>。相关性得分就是一个简单的点积：</p>



<div class="arithmatex">\[s(q, d) = \sum_{t \in \mathcal{V}} w_t^q \cdot w_t^d
\tag{16.10}\]</div>

<p>由于两个向量都是稀疏的（在 30K 词汇表中通常只有 20–200 个非零项），这可以使用标准倒排索引（Lucene、Anserini）高效计算——查询时无需 GPU。</p>

##### 训练。 {#training.}

<p>SPLADE 通过对比学习（Batch 内负样本 + 硬负样本）进行训练，并附加两个正则项：</p>



<div class="arithmatex">\[\mathcal{L} = \mathcal{L}_{\text{contrastive}} + \lambda_q \|\mathbf{w}^q\|_1 + \lambda_d \|\mathbf{w}^d\|_1
\tag{16.11}\]</div>

<p>查询和文档表示上的 <span class="arithmatex">\(L_1\)</span> 惩罚鼓励稀疏性——若没有它们，模型将学到稠密表示，从而背离设计初衷。</p>

##### SPLADEv2 —— 关键改进。 {#spladev2-key-improvements.}

<p>SPLADEv2 <a class="book-citation" id="cite-formal2021spladev2-1" href="../../references.md#ref-formal2021spladev2" aria-label="参考文献 94">[94]</a> 引入了若干改进，显著提升了效率与效果：</p>

<ol>
<li><p><strong>从交叉编码器蒸馏</strong>：SPLADEv2 不只用二元相关性标签训练，而是借助交叉编码器教师（如 MonoT5 <a class="book-citation" id="cite-nogueira2020document-1" href="../../references.md#ref-nogueira2020document" aria-label="参考文献 269">[269]</a>）提供软相关性得分，从而获得更丰富的训练信号：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{distill}} = \text{KL}\!\left(\sigma(s_{\text{student}}) \,\|\, \sigma(s_{\text{teacher}})\right)
\tag{16.12}\]</div></li>
<li><p><strong>分离的查询/文档编码器</strong>：SPLADEv2 为查询与文档设置不同的稀疏度目标。查询被鼓励<em>更稀疏</em>（查找更快），而文档可以稍稠密（离线预计算）：</p>



<div class="arithmatex">\[\lambda_q &gt; \lambda_d \quad \text{(e.g., } \lambda_q = 3 \times 10^{-4},\; \lambda_d = 1 \times 10^{-4}\text{)}
\tag{16.13}\]</div></li>
<li><p><strong>FLOPS 正则化</strong>：SPLADEv2 不再使用简单的 <span class="arithmatex">\(L_1\)</span>，而是引入了一个感知 FLOPS 的正则项，直接惩罚预期检索代价：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{FLOPS}} = \sum_{t \in \mathcal{V}} \left(\overline{a}_t^q\right)^2 + \sum_{t \in \mathcal{V}} \left(\overline{a}_t^d\right)^2
\tag{16.14}\]</div>

<p>其中 <span class="arithmatex">\(\overline{a}_t\)</span> 是 Batch 中词项 <span class="arithmatex">\(t\)</span> 的平均激活。该项会惩罚在许多文档上都非零的词项（倒排列表长 = 检索慢）。</p></li>
<li><p><strong>高效骨干</strong>：使用 DistilBERT（66M 参数）替代 BERT-base（110M），编码时间减半，质量损失极小。</p></li>
</ol>

<div class="admonition note"><p class="admonition-title">SPLADE 与 SPLADEv2 对比</p>
<span id="table-16-inline-2"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方面</strong></th><th scope="col"><strong>SPLADE（v1）</strong></th><th scope="col"><strong>SPLADEv2</strong></th></tr></thead>
<tbody><tr><td>训练信号</td><td>二元相关性 + 硬负样本</td><td>交叉编码器蒸馏</td></tr>
<tr><td>稀疏度控制</td><td><span class="arithmatex">\(L_1\)</span> 正则化</td><td>感知 FLOPS 的正则化</td></tr>
<tr><td>查询/文档对称性</td><td>同一编码器、同一 <span class="arithmatex">\(\lambda\)</span></td><td>非对称（查询更稀疏）</td></tr>
<tr><td>骨干</td><td>BERT-base（110M）</td><td>DistilBERT（66M）</td></tr>
<tr><td>MRR@10（MS MARCO <a class="book-citation" id="cite-bajaj2016msmarco-1" href="../../references.md#ref-bajaj2016msmarco" aria-label="参考文献 18">[18]</a>）</td><td>34.0</td><td>36.8</td></tr>
<tr><td>平均每文档非零项数</td><td><span class="arithmatex">\(\sim\)</span>200</td><td><span class="arithmatex">\(\sim\)</span>120（更稀疏 40%）</td></tr></tbody>
</table>
</div>
</div>

<div class="admonition tip"><p class="admonition-title">何时使用 SPLADE</p>
<ul>
<li><p><strong>适合使用 SPLADE/v2 的场景</strong>：你需要在查询时不依赖 GPU 完成语义检索；你的基础设施已有倒排索引（Elasticsearch、Lucene）；或你需要可解释的相关性得分（可以查看是哪些扩展词项匹配上了）。</p></li>
<li><p><strong>更适合使用稠密检索的场景</strong>：你有 GPU 预算用于查询编码；需要多语言支持（稠密模型的迁移性更好）；或者查询非常短（1–2 个词时扩展帮助较小）。</p></li>
<li><p><strong>最佳实践</strong>：用 SPLADEv2 作为第一阶段检索器 + 在 top-<span class="arithmatex">\(k\)</span> 上使用交叉编码器重排序器。这种组合可在更低延迟下匹敌甚至超越稠密检索流水线。</p></li>
</ul>
</div>

### 16.3.5　ColBERT：晚期交互 {#colbert-late-interaction}

<p>ColBERT <a class="book-citation" id="cite-khattab2020colbert-1" href="../../references.md#ref-khattab2020colbert" aria-label="参考文献 182">[182]</a> 将查询和文档编码为 Token 级 Embedding 的<em>集合</em>，并使用 <em>MaxSim</em> 算子进行打分：</p>

<span id="eq:colbert"></span>

<div class="arithmatex">\[s(q, d) = \sum_{i \in |\mathbf{q}|} \max_{j \in |\mathbf{d}|} \mathbf{q}_i^\top \mathbf{d}_j

\tag{16.15}\]</div>

<p>这种晚期交互（Late Interaction）机制比单向量双编码器更具表达力，又因为文档 Embedding 可离线预计算而远比交叉编码器更快。</p>

##### 架构。 {#architecture.-1}

<p>查询编码器 <span class="arithmatex">\(E_Q\)</span> 和文档编码器 <span class="arithmatex">\(E_D\)</span> 都是基于 BERT 的模型，产生<em>逐 Token</em> 的 Embedding（而不是单个 [CLS] 向量）。每个 Token Embedding 通过一个线性层投影到更低维度（通常为 128）：</p>



<div class="arithmatex">\[\begin{align*}
\mathbf{q}_i &amp;= \text{Linear}(E_Q(q)_i) \in \mathbb{R}^{128}, \quad i = 1, \ldots, |q| \tag{16.16}\\

  \mathbf{d}_j &amp;= \text{Linear}(E_D(d)_j) \in \mathbb{R}^{128}, \quad j = 1, \ldots, |d|
\tag{16.17}
\end{align*}\]</div>

##### 训练。 {#training.-1}

<p>ColBERT 在正、负段落上以成对的 Softmax 交叉熵 Loss 进行训练。给定查询 <span class="arithmatex">\(q\)</span>、正段落 <span class="arithmatex">\(d^+\)</span> 和一组负段落 <span class="arithmatex">\(\{d^-_1, \ldots, d^-_N\}\)</span>：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{ColBERT}} = -\log \frac{\exp(s(q, d^+))}{\exp(s(q, d^+)) + \sum_{k=1}^{N} \exp(s(q, d^-_k))}
\tag{16.18}\]</div>

<p>其中 <span class="arithmatex">\(s(q, d)\)</span> 是 Equation <a href="#eq:colbert">16.15</a> 中的 MaxSim 分数。负样本来自：</p>

<ul>
<li><p><strong>Batch 内负样本</strong>：同一训练 Batch 中的其他段落（无额外成本、数量充足）</p></li>
<li><p><strong>硬负样本</strong>：由 BM25 检索出的、词面相似但语义无关的段落（对质量影响最大）</p></li>
<li><p><strong>蒸馏负样本</strong>（ColBERTv2 <a class="book-citation" id="cite-santhanam2022colbertv2-1" href="../../references.md#ref-santhanam2022colbertv2" aria-label="参考文献 313">[313]</a>）：使用交叉编码器教师挖掘最难的负样本，并将其得分蒸馏到 ColBERT 中</p></li>
</ul>

##### 索引与服务。 {#indexing-and-serving.}

<p>在索引阶段，所有文档 Token 的 Embedding 都被预计算并存储（在 ColBERTv2 中可选地使用残差量化压缩）。在查询阶段，只需实时编码查询 Token，并与已存储的文档 Embedding 计算 MaxSim。这种分离带来：</p>

<ul>
<li><p><strong>离线文档编码</strong>：编码一次，服务于多次查询</p></li>
<li><p><strong>PLAID 索引</strong> <a class="book-citation" id="cite-santhanam2022colbertv2-2" href="../../references.md#ref-santhanam2022colbertv2" aria-label="参考文献 313">[313]</a>：对文档 Embedding 聚类，用聚类中心进行初步候选检索，然后只在候选集上计算精确的 MaxSim——延迟可降低 5–10<span class="arithmatex">\(\times\)</span></p></li>
<li><p><strong>索引大小</strong>：每文档 <span class="arithmatex">\(|d| \times 128\)</span> 个浮点数（比单向量方法更大，但通过量化可压缩到约每维度 2 字节）</p></li>
</ul>

### 16.3.6　检索方法对比 {#retrieval-method-comparison}

<p class="table-caption" id="tab:retrieval_methods"><span class="table-number">表 16.2　</span>检索方法在关键维度上的对比</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>延迟</strong></th><th scope="col"><strong>准确率</strong></th><th scope="col"><strong>索引大小</strong></th><th scope="col"><strong>GPU</strong></th><th scope="col"><strong>最佳适用场景</strong></th></tr></thead>
<tbody><tr><td>TF-IDF <a class="book-citation" id="cite-sparckjones1972idf-1" href="../../references.md#ref-sparckjones1972idf" aria-label="参考文献 336">[336]</a></td><td>极低</td><td>低</td><td>小</td><td>无需</td><td>基线、精确匹配</td></tr>
<tr><td>BM25 <a class="book-citation" id="cite-robertson2009probabilistic-2" href="../../references.md#ref-robertson2009probabilistic" aria-label="参考文献 309">[309]</a></td><td>极低</td><td>中等</td><td>小</td><td>无需</td><td>关键词搜索、罕见词项</td></tr>
<tr><td>DPR / 双编码器 <a class="book-citation" id="cite-karpukhin2020dense-2" href="../../references.md#ref-karpukhin2020dense" aria-label="参考文献 180">[180]</a></td><td>低</td><td>高</td><td>大</td><td>需要</td><td>语义相似度</td></tr>
<tr><td>SPLADE <a class="book-citation" id="cite-formal2021splade-2" href="../../references.md#ref-formal2021splade" aria-label="参考文献 95">[95]</a></td><td>低</td><td>高</td><td>中等</td><td>需要</td><td>兼顾精度与速度的混合方案</td></tr>
<tr><td>ColBERT <a class="book-citation" id="cite-khattab2020colbert-2" href="../../references.md#ref-khattab2020colbert" aria-label="参考文献 182">[182]</a></td><td>中等</td><td>极高</td><td>极大</td><td>需要</td><td>高精度检索</td></tr>
<tr><td>交叉编码器 <a class="book-citation" id="cite-nogueira2019passage-1" href="../../references.md#ref-nogueira2019passage" aria-label="参考文献 268">[268]</a></td><td>高</td><td>最高</td><td>N/A</td><td>需要</td><td>对 top-<span class="arithmatex">\(k\)</span> 重排序</td></tr>
<tr><td>混合（RRF） <a class="book-citation" id="cite-cormack2009reciprocal-2" href="../../references.md#ref-cormack2009reciprocal" aria-label="参考文献 62">[62]</a></td><td>低</td><td>极高</td><td>大</td><td>需要</td><td>生产系统</td></tr></tbody>
</table>
</div>

## 16.4　分块策略 {#sec:chunking}

<p>分块是将文档切分为以下段落的过程：(1) 足够小，能放入 Embedding 模型的上下文窗口；(2) 语义连贯；(3) 即使被单独检索出来，也包含足够的上下文以发挥作用。</p>

### 16.4.1　带重叠的固定大小分块 {#fixed-size-chunking-with-overlap}

<p>最简单的策略：每 <span class="arithmatex">\(W\)</span> 个 Token 切分一次，相邻块之间保留 <span class="arithmatex">\(O\)</span> 个 Token 的重叠。</p>

<p class="code-caption"><span class="code-number">代码 16.2　</span>带重叠的固定大小分块</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">langchain.text_splitter</span><span class="w"> </span><span class="kn">import</span> <span class="n">RecursiveCharacterTextSplitter</span>

<span class="n">splitter</span> <span class="o">=</span> <span class="n">RecursiveCharacterTextSplitter</span><span class="p">(</span>
    <span class="n">chunk_size</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span>       <span class="c1"># 每个块的 Token 数</span>
    <span class="n">chunk_overlap</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span>     <span class="c1"># 重叠量，用于在边界处保留上下文</span>
    <span class="n">length_function</span><span class="o">=</span><span class="nb">len</span><span class="p">,</span>
    <span class="n">separators</span><span class="o">=</span><span class="p">[</span><span class="s2">"</span><span class="se">\n\n</span><span class="s2">"</span><span class="p">,</span> <span class="s2">"</span><span class="se">\n</span><span class="s2">"</span><span class="p">,</span> <span class="s2">". "</span><span class="p">,</span> <span class="s2">" "</span><span class="p">,</span> <span class="s2">""</span><span class="p">]</span>
<span class="p">)</span>
<span class="n">chunks</span> <span class="o">=</span> <span class="n">splitter</span><span class="o">.</span><span class="n">split_documents</span><span class="p">(</span><span class="n">documents</span><span class="p">)</span>
</code></pre></div>

<p><strong>重叠公式</strong>：对于长度为 <span class="arithmatex">\(L\)</span> 个 Token 的文档，块的数量为：</p>



<div class="arithmatex">\[N_{\text{chunks}} = \left\lceil \frac{L - O}{W - O} \right\rceil
\tag{16.19}\]</div>

### 16.4.2　语义分块 {#semantic-chunking}

<p>语义分块不在固定间隔处切分，而是在通过相邻句子间 Embedding 相似度检测出的<em>话题边界</em>处切分：</p>

<p class="code-caption"><span class="code-number">代码 16.3　</span>通过 Embedding 相似度进行语义分块</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">langchain_experimental.text_splitter</span><span class="w"> </span><span class="kn">import</span> <span class="n">SemanticChunker</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">OpenAIEmbeddings</span>

<span class="n">chunker</span> <span class="o">=</span> <span class="n">SemanticChunker</span><span class="p">(</span>
    <span class="n">embeddings</span><span class="o">=</span><span class="n">OpenAIEmbeddings</span><span class="p">(),</span>
    <span class="n">breakpoint_threshold_type</span><span class="o">=</span><span class="s2">"percentile"</span><span class="p">,</span>  <span class="c1"># 或 "standard_deviation"</span>
    <span class="n">breakpoint_threshold_amount</span><span class="o">=</span><span class="mi">95</span><span class="p">,</span>          <span class="c1"># 在最不相似的前 5% 处切分</span>
<span class="p">)</span>
<span class="n">chunks</span> <span class="o">=</span> <span class="n">chunker</span><span class="o">.</span><span class="n">split_documents</span><span class="p">(</span><span class="n">documents</span><span class="p">)</span>
</code></pre></div>

### 16.4.3　基于文档结构的分块 {#document-structure-aware-chunking}

<p>对于结构化文档（Markdown、HTML、代码），在自然边界处切分：</p>

<ul>
<li><p><strong>Markdown</strong>：在 <code>##</code> 标题处切分，保留章节上下文</p></li>
<li><p><strong>HTML</strong>：在 <code>&lt;section&gt;</code>、<code>&lt;article&gt;</code>、<code>&lt;p&gt;</code> 标签处切分</p></li>
<li><p><strong>代码</strong>：在函数/类定义处切分，并在每个块中保留 import 语句</p></li>
<li><p><strong>表格</strong>：将整个表格保留为单个块；切勿在行中间切分</p></li>
</ul>

### 16.4.4　父—子分块 {#parent-child-chunking}

<p>这是一种将检索粒度与生成上下文解耦的强大模式：</p>

<ol>
<li><p><strong>索引较小的子块</strong>（如 128 个 Token）以实现精确检索</p></li>
<li><p><strong>将较大的父块</strong>（如 512 个 Token）返回给 LLM 以提供更丰富的上下文</p></li>
</ol>

<p class="code-caption"><span class="code-number">代码 16.4　</span>使用 LangChain 实现父—子分块</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">langchain.retrievers</span><span class="w"> </span><span class="kn">import</span> <span class="n">ParentDocumentRetriever</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain.storage</span><span class="w"> </span><span class="kn">import</span> <span class="n">InMemoryStore</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain.text_splitter</span><span class="w"> </span><span class="kn">import</span> <span class="n">RecursiveCharacterTextSplitter</span>

<span class="n">parent_splitter</span> <span class="o">=</span> <span class="n">RecursiveCharacterTextSplitter</span><span class="p">(</span><span class="n">chunk_size</span><span class="o">=</span><span class="mi">2000</span><span class="p">)</span>
<span class="n">child_splitter</span>  <span class="o">=</span> <span class="n">RecursiveCharacterTextSplitter</span><span class="p">(</span><span class="n">chunk_size</span><span class="o">=</span><span class="mi">400</span><span class="p">)</span>

<span class="n">retriever</span> <span class="o">=</span> <span class="n">ParentDocumentRetriever</span><span class="p">(</span>
    <span class="n">vectorstore</span><span class="o">=</span><span class="n">vectorstore</span><span class="p">,</span>
    <span class="n">docstore</span><span class="o">=</span><span class="n">InMemoryStore</span><span class="p">(),</span>
    <span class="n">child_splitter</span><span class="o">=</span><span class="n">child_splitter</span><span class="p">,</span>
    <span class="n">parent_splitter</span><span class="o">=</span><span class="n">parent_splitter</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">retriever</span><span class="o">.</span><span class="n">add_documents</span><span class="p">(</span><span class="n">documents</span><span class="p">)</span>
</code></pre></div>

### 16.4.5　块大小的经验指南 {#empirical-guidelines-for-chunk-size}

<p class="table-caption" id="table-16-3"><span class="table-number">表 16.3　</span>按使用场景推荐的块大小</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>使用场景</strong></th><th scope="col"><strong>推荐块大小</strong></th><th scope="col"><strong>重叠</strong></th></tr></thead>
<tbody><tr><td>事实型问答（精确事实）</td><td>128–256 Token</td><td>20–32 Token</td></tr>
<tr><td>摘要 / 综合</td><td>512–1024 Token</td><td>64–128 Token</td></tr>
<tr><td>代码检索</td><td>完整函数</td><td>无</td></tr>
<tr><td>法律 / 监管文档</td><td>段落级</td><td>1 句</td></tr>
<tr><td>对话 / 聊天</td><td>256–512 Token</td><td>32–64 Token</td></tr></tbody>
</table>
</div>

## 16.5　高级 RAG 模式 {#advanced-rag-patterns}

### 16.5.1　查询变换 {#query-transformation}

<p>用户的原始查询往往含糊、过短，或与文档语言匹配不佳。查询变换技术在搜索之前改进检索。</p>

##### HyDE（假设性文档 Embedding，Hypothetical Document Embeddings） <a class="book-citation" id="cite-gao2022precise-1" href="../../references.md#ref-gao2022precise" aria-label="参考文献 106">[106]</a>。 {#hyde-hypothetical-document-embeddings-.}

<p>不直接对查询做 Embedding，而是生成一个<em>假设性回答</em>并对其做 Embedding：</p>



<div class="arithmatex">\[\hat{d} = \text{LLM}(q), \quad \mathbf{e}_{\text{query}} = f_\phi(\hat{d})
\tag{16.20}\]</div>

<p>直觉在于：假设性回答与真实文档处于相同的语言风格中，从而缩小查询—文档之间的分布差距。</p>

##### 后退式提示（Step-Back Prompting）。 {#step-back-prompting.}

<p>对于具体问题，先生成一个更通用的“后退”问题，对两者都进行检索，再合并上下文。例如：“乙醇在 2 atm 下的沸点是多少？” <span class="arithmatex">\(\to\)</span> 后退问题：“哪些因素影响液体的沸点？”</p>

##### 多查询生成。 {#multi-query-generation.}

<p>生成 <span class="arithmatex">\(M\)</span> 种多样化的查询改写，对每个改写都进行检索，并对结果取并集：</p>

<p class="code-caption"><span class="code-number">代码 16.5　</span>多查询检索</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">langchain.retrievers.multi_query</span><span class="w"> </span><span class="kn">import</span> <span class="n">MultiQueryRetriever</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">ChatOpenAI</span>

<span class="n">retriever</span> <span class="o">=</span> <span class="n">MultiQueryRetriever</span><span class="o">.</span><span class="n">from_llm</span><span class="p">(</span>
    <span class="n">retriever</span><span class="o">=</span><span class="n">vectorstore</span><span class="o">.</span><span class="n">as_retriever</span><span class="p">(</span><span class="n">search_kwargs</span><span class="o">=</span><span class="p">{</span><span class="s2">"k"</span><span class="p">:</span> <span class="mi">5</span><span class="p">}),</span>
    <span class="n">llm</span><span class="o">=</span><span class="n">ChatOpenAI</span><span class="p">(</span><span class="n">temperature</span><span class="o">=</span><span class="mf">0.7</span><span class="p">),</span>
    <span class="n">include_original</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>   <span class="c1"># 同时也对原始查询进行检索</span>
<span class="p">)</span>
<span class="c1"># 内部生成 3 个查询变体，对每个进行检索并去重</span>
<span class="n">docs</span> <span class="o">=</span> <span class="n">retriever</span><span class="o">.</span><span class="n">get_relevant_documents</span><span class="p">(</span><span class="n">query</span><span class="p">)</span>
</code></pre></div>

### 16.5.2　重排序 {#re-ranking}

<p>在初步检索出 top-<span class="arithmatex">\(k\)</span> 候选后，<em>交叉编码器</em>重排序器会对每个查询—文档对联合打分（同时关注两者），以更高延迟为代价产生更准确的相关性得分：</p>



<div class="arithmatex">\[s_{\text{cross}}(q, d) = \text{CrossEncoder}([q; d])
\tag{16.21}\]</div>

<p>交叉编码器无法用于第一阶段检索（没有预计算的文档 Embedding），但非常适合对小规模候选集（通常 <span class="arithmatex">\(k = 20\)</span>–<span class="arithmatex">\(100\)</span>）进行重排序。</p>

<p class="code-caption"><span class="code-number">代码 16.6　</span>使用 BGE 进行交叉编码器重排序</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">sentence_transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">CrossEncoder</span>

<span class="n">reranker</span> <span class="o">=</span> <span class="n">CrossEncoder</span><span class="p">(</span><span class="s2">"BAAI/bge-reranker-large"</span><span class="p">)</span>

<span class="k">def</span><span class="w"> </span><span class="nf">rerank</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">docs</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">],</span> <span class="n">top_n</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]:</span>
    <span class="n">pairs</span> <span class="o">=</span> <span class="p">[(</span><span class="n">query</span><span class="p">,</span> <span class="n">doc</span><span class="p">)</span> <span class="k">for</span> <span class="n">doc</span> <span class="ow">in</span> <span class="n">docs</span><span class="p">]</span>
    <span class="n">scores</span> <span class="o">=</span> <span class="n">reranker</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">pairs</span><span class="p">)</span>
    <span class="n">ranked</span> <span class="o">=</span> <span class="nb">sorted</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">scores</span><span class="p">,</span> <span class="n">docs</span><span class="p">),</span> <span class="n">reverse</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="k">return</span> <span class="p">[</span><span class="n">doc</span> <span class="k">for</span> <span class="n">_</span><span class="p">,</span> <span class="n">doc</span> <span class="ow">in</span> <span class="n">ranked</span><span class="p">[:</span><span class="n">top_n</span><span class="p">]]</span>
</code></pre></div>

### 16.5.3　上下文压缩 {#contextual-compression}

<p>检索到的块往往在相关段落周围包含许多无关句子。上下文压缩使用 LLM 仅提取相关部分：</p>

<p class="code-caption"><span class="code-number">代码 16.7　</span>基于 LLM 的上下文压缩</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">langchain.retrievers</span><span class="w"> </span><span class="kn">import</span> <span class="n">ContextualCompressionRetriever</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain.retrievers.document_compressors</span><span class="w"> </span><span class="kn">import</span> <span class="n">LLMChainExtractor</span>

<span class="n">compressor</span> <span class="o">=</span> <span class="n">LLMChainExtractor</span><span class="o">.</span><span class="n">from_llm</span><span class="p">(</span><span class="n">llm</span><span class="p">)</span>
<span class="n">compression_retriever</span> <span class="o">=</span> <span class="n">ContextualCompressionRetriever</span><span class="p">(</span>
    <span class="n">base_compressor</span><span class="o">=</span><span class="n">compressor</span><span class="p">,</span>
    <span class="n">base_retriever</span><span class="o">=</span><span class="n">vectorstore</span><span class="o">.</span><span class="n">as_retriever</span><span class="p">()</span>
<span class="p">)</span>
<span class="n">compressed_docs</span> <span class="o">=</span> <span class="n">compression_retriever</span><span class="o">.</span><span class="n">get_relevant_documents</span><span class="p">(</span><span class="n">query</span><span class="p">)</span>
</code></pre></div>

### 16.5.4　Self-RAG {#self-rag}

<p>Self-RAG <a class="book-citation" id="cite-asai2023selfrag-1" href="../../references.md#ref-asai2023selfrag" aria-label="参考文献 13">[13]</a> 训练单个模型完成以下任务：(1) 决定<em>是否</em>进行检索，(2) 在带或不带检索的条件下生成，(3) 使用特殊的反思 Token <em>批评</em>自己的输出：</p>

<ul>
<li><p><code>[Retrieve]</code>：模型是否应检索更多段落？</p></li>
<li><p><code>[IsRel]</code>：检索到的段落是否与查询相关？</p></li>
<li><p><code>[IsSup]</code>：生成的陈述是否可由检索到的段落支持？</p></li>
<li><p><code>[IsUse]</code>：整体回复是否有用？</p></li>
</ul>

<p>模型经过端到端训练，与回复一起预测这些 Token，从而实现对检索与自我评分的细粒度控制。</p>

### 16.5.5　CRAG：纠正式 RAG {#crag-corrective-rag}

<p>CRAG <a class="book-citation" id="cite-yan2024crag-1" href="../../references.md#ref-yan2024crag" aria-label="参考文献 401">[401]</a> 增加了一个<em>检索评估器</em>，对检索到的文档打分并触发纠正动作：</p>

<ol>
<li><p>检索 top-<span class="arithmatex">\(k\)</span> 文档</p></li>
<li><p>对每个文档打分：<strong>正确</strong> / <strong>模糊</strong> / <strong>错误</strong></p></li>
<li><p>如果所有文档都为错误或模糊 <span class="arithmatex">\(\to\)</span> 回退到网页搜索</p></li>
<li><p>如果部分文档正确 <span class="arithmatex">\(\to\)</span> 使用知识精炼（剔除无关句子）</p></li>
<li><p>基于精炼后的上下文生成回答</p></li>
</ol>

### 16.5.6　自适应 RAG {#adaptive-rag}

<p>自适应 RAG（Adaptive RAG） <a class="book-citation" id="cite-jeong2024adaptive-1" href="../../references.md#ref-jeong2024adaptive" aria-label="参考文献 162">[162]</a> 根据预测出的查询复杂度将其路由到不同的检索策略：</p>

<ul>
<li><p><strong>不检索</strong>：模型可凭参数化知识回答的简单事实型查询</p></li>
<li><p><strong>单步 RAG</strong>：对中等复杂度的查询采用标准的“先检索后生成”</p></li>
<li><p><strong>多步 RAG</strong>：针对复杂的多跳问题进行迭代检索</p></li>
</ul>

<p>一个在查询复杂度标签上训练的轻量级分类器对每个传入查询进行路由。</p>

### 16.5.7　Graph RAG {#graph-rag}

<p>Microsoft 的 Graph RAG <a class="book-citation" id="cite-edge2024local-1" href="../../references.md#ref-edge2024local" aria-label="参考文献 82">[82]</a> 从文档语料库构建<em>知识图谱</em>，并使用社区检测生成分层摘要：</p>

<ol>
<li><p><strong>实体抽取</strong>：LLM 从每个块中抽取实体与关系</p></li>
<li><p><strong>图构建</strong>：构建图 <span class="arithmatex">\(G = (V, E)\)</span>，其中节点是实体、边是关系</p></li>
<li><p><strong>社区检测</strong>：在多种分辨率下应用 Leiden 算法寻找社区</p></li>
<li><p><strong>社区摘要</strong>：LLM 为每个社区生成摘要</p></li>
<li><p><strong>查询</strong>：对全局查询，对社区摘要执行 map-reduce；对局部查询，使用标准向量搜索</p></li>
</ol>

<div class="admonition note"><p class="admonition-title">何时使用 Graph RAG</p>
<p>Graph RAG 擅长处理需要跨大量文档综合信息的<em>全局</em>查询（“该语料库的主要主题有哪些？”），但构建与维护成本高。标准 RAG 更适合<em>局部</em>查询（“文档 X 关于话题 Y 说了什么？”）。</p>
</div>

### 16.5.8　RAG-Fusion {#rag-fusion}

<p>RAG-Fusion <a class="book-citation" id="cite-rackauckas2023ragfusion-1" href="../../references.md#ref-rackauckas2023ragfusion" aria-label="参考文献 300">[300]</a> 从原始查询生成多条搜索查询，对每条都进行检索，再使用 RRF（Equation <a href="#eq:rrf">16.8</a>）融合排名列表：</p>

<p class="code-caption"><span class="code-number">代码 16.8　</span>使用 RRF 的 RAG-Fusion</p>
<div class="highlight"><pre><span></span><code><span class="k">def</span><span class="w"> </span><span class="nf">reciprocal_rank_fusion</span><span class="p">(</span><span class="n">ranked_lists</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]],</span> <span class="n">k</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">60</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]:</span>
<span class="w">    </span><span class="sd">"""使用 RRF 融合多个有序文档列表。"""</span>
    <span class="n">scores</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">float</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">ranked</span> <span class="ow">in</span> <span class="n">ranked_lists</span><span class="p">:</span>
        <span class="k">for</span> <span class="n">rank</span><span class="p">,</span> <span class="n">doc_id</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">ranked</span><span class="p">,</span> <span class="n">start</span><span class="o">=</span><span class="mi">1</span><span class="p">):</span>
            <span class="n">scores</span><span class="p">[</span><span class="n">doc_id</span><span class="p">]</span> <span class="o">=</span> <span class="n">scores</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="n">doc_id</span><span class="p">,</span> <span class="mf">0.0</span><span class="p">)</span> <span class="o">+</span> <span class="mf">1.0</span> <span class="o">/</span> <span class="p">(</span><span class="n">k</span> <span class="o">+</span> <span class="n">rank</span><span class="p">)</span>
    <span class="k">return</span> <span class="nb">sorted</span><span class="p">(</span><span class="n">scores</span><span class="p">,</span> <span class="n">key</span><span class="o">=</span><span class="n">scores</span><span class="o">.</span><span class="n">get</span><span class="p">,</span> <span class="n">reverse</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="k">def</span><span class="w"> </span><span class="nf">rag_fusion</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">retriever</span><span class="p">,</span> <span class="n">llm</span><span class="p">,</span> <span class="n">n_queries</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">4</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="c1"># 第 1 步：生成查询变体</span>
    <span class="n">variants</span> <span class="o">=</span> <span class="n">generate_query_variants</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">llm</span><span class="p">,</span> <span class="n">n</span><span class="o">=</span><span class="n">n_queries</span><span class="p">)</span>
    <span class="c1"># 第 2 步：对每个变体进行检索</span>
    <span class="n">all_ranked</span> <span class="o">=</span> <span class="p">[</span><span class="n">retriever</span><span class="o">.</span><span class="n">retrieve</span><span class="p">(</span><span class="n">q</span><span class="p">)</span> <span class="k">for</span> <span class="n">q</span> <span class="ow">in</span> <span class="p">[</span><span class="n">query</span><span class="p">]</span> <span class="o">+</span> <span class="n">variants</span><span class="p">]</span>
    <span class="c1"># 第 3 步：使用 RRF 融合</span>
    <span class="n">fused_docs</span> <span class="o">=</span> <span class="n">reciprocal_rank_fusion</span><span class="p">(</span><span class="n">all_ranked</span><span class="p">)</span>
    <span class="c1"># 第 4 步：生成回答</span>
    <span class="k">return</span> <span class="n">generate_answer</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">fused_docs</span><span class="p">[:</span><span class="mi">5</span><span class="p">],</span> <span class="n">llm</span><span class="p">)</span>
</code></pre></div>

## 16.6　高效 RAG 解码：REFRAG {#efficient-rag-decoding-refrag}

<p>RAG 的一个实际瓶颈是<em>解码延迟</em>：拼接到 LLM 上下文中的检索段落往往很长但相关性稀疏，从而拉高首 Token 时延（Time-to-First-Token, TTFT）和 KV-cache 内存。REFRAG <a class="book-citation" id="cite-lin2025refrag-1" href="../../references.md#ref-lin2025refrag" aria-label="参考文献 220">[220]</a> 观察到，由于检索到的段落是独立来源的（通过重排序中的多样性或去重），它们的 Attention 模式呈<em>块对角</em>——大多数跨段落 Attention 接近于零。这种稀疏性意味着解码阶段对 RAG 上下文的大部分计算其实是不必要的。</p>

##### 压缩—感知—展开（Compress–Sense–Expand）框架。 {#compresssenseexpand-framework.}

<p>REFRAG 通过三阶段解码策略利用这种结构：</p>

<ol>
<li><p><strong>压缩</strong>：用紧凑的摘要（如对每个段落块的 Key/Value 进行均值池化）替换检索段落的完整 KV 表示，大幅降低内存占用。</p></li>
<li><p><strong>感知</strong>：在每个解码步骤上，对压缩后的表示执行轻量级 Attention，识别哪些段落块与当前 Token 相关。</p></li>
<li><p><strong>展开</strong>：仅为被选中的块重建完整的 KV 项，对稀疏的活跃集合执行精确 Attention。</p></li>
</ol>

##### 结果。 {#results.-dup3}

<p>在基于 LLaMA 的模型上，REFRAG 实现了最高 <span class="arithmatex">\(30.85\times\)</span> 的 TTFT 加速（比此前的稀疏 Attention 基线提升 <span class="arithmatex">\(3.75\times\)</span>），且困惑度无损。在固定内存预算下，它还将有效上下文长度扩展了 <span class="arithmatex">\(16\times\)</span>。这些收益在 RAG、多轮对话以及长文档摘要任务中均成立。</p>

<div class="admonition tip"><p class="admonition-title">REFRAG 为何对智能体 RAG 至关重要</p>
<p>智能体 RAG（Section <a href="#sec:agentic_rag">16.7</a>）需要为每个查询进行<em>多轮</em>检索，延迟会累积。REFRAG 这类高效解码方法是必备的基础设施：通过确保每一轮的解码代价相对上下文长度是次线性的，它们使迭代式的检索—推理—生成循环在大规模下具备可行性。</p>
</div>

## 16.7　智能体 RAG {#sec:agentic_rag}

### 16.7.1　动机：静态 RAG 的局限 {#motivation-limits-of-static-rag}

<p>标准 RAG 遵循固定的“先检索后生成”模式。这在以下情况下会失败：</p>

<ul>
<li><p><strong>多跳问题</strong>：“是谁创办了在 2023 年收购 OpenAI 主要竞争对手的那家公司？”这类问题需要串联多次检索</p></li>
<li><p><strong>歧义查询</strong>：正确的检索策略取决于已找到的内容</p></li>
<li><p><strong>异构来源</strong>：不同子问题需要不同的知识库</p></li>
<li><p><strong>迭代精炼</strong>：初次检索可能揭示需要换一种查询方式</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">将 RAG 视为马尔可夫决策过程</p>
<p>智能体 RAG 将检索视为一个序贯决策问题。<em>状态</em>是当前的上下文（查询 + 至今为止检索到的文档）；<em>动作</em>包括检索、推理、生成和停止；<em>Reward</em> 是答案的正确性。Agent 学习一种关于何时检索以及检索什么的 Policy。</p>
</div>

### 16.7.2　智能体 RAG 架构 {#agentic-rag-architecture}

<figure id="fig:agentic_rag"><a href="../../assets/figures/fig_051_agentic_rag.png" target="_blank" rel="noopener" aria-label="查看图 16.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_051_agentic_rag.png" alt="智能体 RAG 控制流。Agent 在返回答案前迭代地进行规划、检索、判断充分性，并自检接地性。" loading="lazy"></a><figcaption><span class="figure-number">图 16.2　</span>智能体 RAG 控制流。Agent 在返回答案前迭代地进行规划、检索、判断充分性，并自检接地性。</figcaption></figure>

### 16.7.3　多源路由 {#multi-source-routing}

<p>智能体 RAG 系统可以将子查询路由到专门化的知识源。其核心洞见是：不同问题类型需要不同的检索后端——没有任何单一索引能在所有方面都表现出色。</p>

##### 为什么要路由？ {#why-route}

<p>设想一个金融分析师助手处理四种查询：</p>

<ul>
<li><p>“我们公司的带薪休假（PTO）政策是什么？” <span class="arithmatex">\(\rightarrow\)</span> <strong>向量数据库</strong>（内部文档）</p></li>
<li><p>“美联储昨天宣布了什么？” <span class="arithmatex">\(\rightarrow\)</span> <strong>网页搜索</strong>（实时）</p></li>
<li><p>“按地区展示 Q3 营收” <span class="arithmatex">\(\rightarrow\)</span> <strong>SQL 数据库</strong>（结构化数据）</p></li>
<li><p>“我们的鉴权中间件如何校验 Token？” <span class="arithmatex">\(\rightarrow\)</span> <strong>代码索引</strong>（代码库）</p></li>
</ul>

<p>“从单一索引扁平化检索”的做法要么错过答案，要么返回无关段落。路由在检索开始前，为<em>合适的子问题选择合适的工具</em>。</p>

##### 路由策略。 {#routing-strategies.}

<p>按复杂度递增的三种主要方法：</p>

<ol>
<li><p><strong>基于规则的路由。</strong>通过关键词触发（例如，SQL 关键字 <span class="arithmatex">\(\rightarrow\)</span> 数据库，URL 模式 <span class="arithmatex">\(\rightarrow\)</span> 网页）。快速且可解释，但对歧义查询非常脆弱。</p></li>
<li><p><strong>基于分类器的路由。</strong>由一个轻量模型（如微调过的 BERT 分类器，或在查询 Embedding 上的逻辑回归）预测最佳来源。延迟低（<span class="arithmatex">\(&lt;\)</span>10 毫秒），可基于路由日志训练，但需要带标签数据。</p></li>
<li><p><strong>基于 LLM 的路由。</strong>LLM 本身在一次结构化输出调用中决定来源（见下面的代码清单）。最为灵活——可以处理新颖的查询类型并解释其推理——但会增加一次 LLM 调用的延迟。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">路由器即可学习的 Policy</p>
<p>多源路由在最简单的形式下是一个<em>分类</em>问题，在最丰富的形式下则是一个<em>规划</em>问题。当被建模为 RL Policy——其中状态是查询加上对话历史，动作是来源的选择（以及可选的查询改写），Reward 是下游答案质量——时，路由器可以通过 Policy Gradient 技术（Chapter <a href="../08-preference-optimization/index.md#ch:po-variants">8</a>）进行端到端优化。</p>
</div>

##### 实践考量。 {#practical-considerations.}

<ul>
<li><p><strong>回退链</strong>：若主要来源返回的结果置信度较低，则尝试次优来源。</p></li>
<li><p><strong>并行扇出</strong>：对于歧义查询，同时从多个来源检索，并通过倒数排名融合（Table <a href="#tab:retrieval_methods">16.2</a>）合并结果。</p></li>
<li><p><strong>成本感知</strong>：网页搜索和 API 调用可能存在金钱开销或速率限制；路由器应将其纳入考量。</p></li>
<li><p><strong>可观测性</strong>：记录每个路由决策及其推理过程——对调试和再训练至关重要。</p></li>
</ul>

<p class="code-caption"><span class="code-number">代码 16.9　</span>多源智能体 RAG 路由器</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">enum</span><span class="w"> </span><span class="kn">import</span> <span class="n">Enum</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pydantic</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseModel</span>

<span class="k">class</span><span class="w"> </span><span class="nc">KnowledgeSource</span><span class="p">(</span><span class="nb">str</span><span class="p">,</span> <span class="n">Enum</span><span class="p">):</span>
    <span class="n">VECTOR_DB</span>   <span class="o">=</span> <span class="s2">"vector_db"</span>    <span class="c1"># 内部文档</span>
    <span class="n">WEB_SEARCH</span>  <span class="o">=</span> <span class="s2">"web_search"</span>   <span class="c1"># 实时网页</span>
    <span class="n">SQL_DB</span>      <span class="o">=</span> <span class="s2">"sql_db"</span>       <span class="c1"># 结构化数据</span>
    <span class="n">CODE_INDEX</span>  <span class="o">=</span> <span class="s2">"code_index"</span>   <span class="c1"># 代码库</span>
    <span class="n">API</span>         <span class="o">=</span> <span class="s2">"api"</span>          <span class="c1"># 外部 API</span>

<span class="k">class</span><span class="w"> </span><span class="nc">RouteDecision</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">source</span><span class="p">:</span> <span class="n">KnowledgeSource</span>
    <span class="n">refined_query</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">reasoning</span><span class="p">:</span> <span class="nb">str</span>

<span class="k">def</span><span class="w"> </span><span class="nf">route_query</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">llm</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">RouteDecision</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""使用 LLM 决定该查询哪个知识源。"""</span>
    <span class="n">prompt</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">"""Given the query: "</span><span class="si">{</span><span class="n">query</span><span class="si">}</span><span class="s2">"</span>

<span class="s2">Decide which knowledge source to use:</span>
<span class="s2">- vector_db: for internal documents, policies, past reports</span>
<span class="s2">- web_search: for current events, recent information</span>
<span class="s2">- sql_db: for numerical data, statistics, structured records</span>
<span class="s2">- code_index: for code examples, API documentation</span>
<span class="s2">- api: for real-time data (weather, stock prices, etc.)</span>

<span class="s2">Return a JSON with: source, refined_query, reasoning."""</span>

    <span class="k">return</span> <span class="n">llm</span><span class="o">.</span><span class="n">with_structured_output</span><span class="p">(</span><span class="n">RouteDecision</span><span class="p">)</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span><span class="n">prompt</span><span class="p">)</span>
</code></pre></div>

### 16.7.4　完整的智能体 RAG 实现 {#full-agentic-rag-implementation}

<p>前面几节介绍了各个独立组件——路由、检索、评估。一个完整的智能体 RAG 系统将它们编排为一个<em>有状态节点的图</em>，其中控制流取决于中间结果。下面的实现使用 LangGraph 将四个节点串成一个循环：</p>

<ol>
<li><p><strong>Plan（规划）</strong>：将用户查询分解为子查询（每个信息需求对应一个）。</p></li>
<li><p><strong>Retrieve（检索）</strong>：将每个子查询路由到合适的来源并取回文档。</p></li>
<li><p><strong>Evaluate（评估）</strong>：判断累积的上下文是否足以回答原始查询。</p></li>
<li><p><strong>Generate（生成）</strong>：综合检索到的文档生成带引用的最终答案。</p></li>
</ol>

<p>关键设计模式是<em>条件循环</em>：评估之后，Agent 要么进入生成阶段（如果上下文足够，或迭代预算耗尽），要么带着精炼后的子查询回到检索阶段。这与一个在信息收集动作上运行的 RL Agent 所遵循的“感知—行动—评估”循环如出一辙。</p>

<p class="code-caption"><span class="code-number">代码 16.10　</span>基于 LangGraph 的智能体 RAG</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">TypedDict</span><span class="p">,</span> <span class="n">Annotated</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.graph</span><span class="w"> </span><span class="kn">import</span> <span class="n">StateGraph</span><span class="p">,</span> <span class="n">END</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.prebuilt</span><span class="w"> </span><span class="kn">import</span> <span class="n">ToolNode</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">operator</span>

<span class="k">class</span><span class="w"> </span><span class="nc">AgentState</span><span class="p">(</span><span class="n">TypedDict</span><span class="p">):</span>
    <span class="n">query</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">sub_queries</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span>
    <span class="n">retrieved_docs</span><span class="p">:</span> <span class="n">Annotated</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">],</span> <span class="n">operator</span><span class="o">.</span><span class="n">add</span><span class="p">]</span>
    <span class="n">context_sufficient</span><span class="p">:</span> <span class="nb">bool</span>
    <span class="n">answer</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">iterations</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">max_iterations</span><span class="p">:</span> <span class="nb">int</span>

<span class="k">def</span><span class="w"> </span><span class="nf">plan_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">AgentState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AgentState</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""将查询分解为子查询。"""</span>
    <span class="n">sub_queries</span> <span class="o">=</span> <span class="n">decompose_query</span><span class="p">(</span><span class="n">state</span><span class="p">[</span><span class="s2">"query"</span><span class="p">])</span>
    <span class="k">return</span> <span class="p">{</span><span class="o">**</span><span class="n">state</span><span class="p">,</span> <span class="s2">"sub_queries"</span><span class="p">:</span> <span class="n">sub_queries</span><span class="p">,</span> <span class="s2">"iterations"</span><span class="p">:</span> <span class="mi">0</span><span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">retrieve_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">AgentState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AgentState</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""为当前的子查询检索文档。"""</span>
    <span class="n">new_docs</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">sq</span> <span class="ow">in</span> <span class="n">state</span><span class="p">[</span><span class="s2">"sub_queries"</span><span class="p">]:</span>
        <span class="n">source</span> <span class="o">=</span> <span class="n">route_query</span><span class="p">(</span><span class="n">sq</span><span class="p">)</span>
        <span class="n">docs</span> <span class="o">=</span> <span class="n">retrieve_from_source</span><span class="p">(</span><span class="n">sq</span><span class="p">,</span> <span class="n">source</span><span class="p">)</span>
        <span class="n">new_docs</span><span class="o">.</span><span class="n">extend</span><span class="p">(</span><span class="n">docs</span><span class="p">)</span>
    <span class="k">return</span> <span class="p">{</span><span class="o">**</span><span class="n">state</span><span class="p">,</span> <span class="s2">"retrieved_docs"</span><span class="p">:</span> <span class="n">new_docs</span><span class="p">,</span>
            <span class="s2">"iterations"</span><span class="p">:</span> <span class="n">state</span><span class="p">[</span><span class="s2">"iterations"</span><span class="p">]</span> <span class="o">+</span> <span class="mi">1</span><span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">evaluate_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">AgentState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AgentState</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""评估检索到的上下文是否充分。"""</span>
    <span class="n">sufficient</span> <span class="o">=</span> <span class="n">evaluate_context_sufficiency</span><span class="p">(</span>
        <span class="n">query</span><span class="o">=</span><span class="n">state</span><span class="p">[</span><span class="s2">"query"</span><span class="p">],</span>
        <span class="n">docs</span><span class="o">=</span><span class="n">state</span><span class="p">[</span><span class="s2">"retrieved_docs"</span><span class="p">]</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="p">{</span><span class="o">**</span><span class="n">state</span><span class="p">,</span> <span class="s2">"context_sufficient"</span><span class="p">:</span> <span class="n">sufficient</span><span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">generate_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">AgentState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AgentState</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""基于检索到的上下文生成回答。"""</span>
    <span class="n">answer</span> <span class="o">=</span> <span class="n">generate_with_citations</span><span class="p">(</span>
        <span class="n">query</span><span class="o">=</span><span class="n">state</span><span class="p">[</span><span class="s2">"query"</span><span class="p">],</span>
        <span class="n">docs</span><span class="o">=</span><span class="n">state</span><span class="p">[</span><span class="s2">"retrieved_docs"</span><span class="p">]</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="p">{</span><span class="o">**</span><span class="n">state</span><span class="p">,</span> <span class="s2">"answer"</span><span class="p">:</span> <span class="n">answer</span><span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">should_retrieve</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">AgentState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="k">if</span> <span class="n">state</span><span class="p">[</span><span class="s2">"context_sufficient"</span><span class="p">]:</span>
        <span class="k">return</span> <span class="s2">"generate"</span>
    <span class="k">if</span> <span class="n">state</span><span class="p">[</span><span class="s2">"iterations"</span><span class="p">]</span> <span class="o">&gt;=</span> <span class="n">state</span><span class="p">[</span><span class="s2">"max_iterations"</span><span class="p">]:</span>
        <span class="k">return</span> <span class="s2">"generate"</span>  <span class="c1"># 放弃继续检索，用现有内容生成</span>
    <span class="k">return</span> <span class="s2">"retrieve"</span>

<span class="c1"># 构建图</span>
<span class="n">workflow</span> <span class="o">=</span> <span class="n">StateGraph</span><span class="p">(</span><span class="n">AgentState</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"plan"</span><span class="p">,</span>     <span class="n">plan_node</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"retrieve"</span><span class="p">,</span> <span class="n">retrieve_node</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"evaluate"</span><span class="p">,</span> <span class="n">evaluate_node</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"generate"</span><span class="p">,</span> <span class="n">generate_node</span><span class="p">)</span>

<span class="n">workflow</span><span class="o">.</span><span class="n">set_entry_point</span><span class="p">(</span><span class="s2">"plan"</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"plan"</span><span class="p">,</span>     <span class="s2">"retrieve"</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"retrieve"</span><span class="p">,</span> <span class="s2">"evaluate"</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_conditional_edges</span><span class="p">(</span><span class="s2">"evaluate"</span><span class="p">,</span> <span class="n">should_retrieve</span><span class="p">,</span>
    <span class="p">{</span><span class="s2">"retrieve"</span><span class="p">:</span> <span class="s2">"retrieve"</span><span class="p">,</span> <span class="s2">"generate"</span><span class="p">:</span> <span class="s2">"generate"</span><span class="p">})</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"generate"</span><span class="p">,</span> <span class="n">END</span><span class="p">)</span>

<span class="n">agent</span> <span class="o">=</span> <span class="n">workflow</span><span class="o">.</span><span class="n">compile</span><span class="p">()</span>

<span class="c1"># 运行</span>
<span class="n">result</span> <span class="o">=</span> <span class="n">agent</span><span class="o">.</span><span class="n">invoke</span><span class="p">({</span>
    <span class="s2">"query"</span><span class="p">:</span> <span class="s2">"What were the main causes of the 2023 banking crisis?"</span><span class="p">,</span>
    <span class="s2">"max_iterations"</span><span class="p">:</span> <span class="mi">3</span><span class="p">,</span>
    <span class="s2">"retrieved_docs"</span><span class="p">:</span> <span class="p">[],</span>
    <span class="s2">"iterations"</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span>
<span class="p">})</span>
</code></pre></div>

### 16.7.5　工具增强的 RAG {#tool-augmented-rag}

<p>智能体 RAG 可以将检索与计算工具结合：</p>

<p class="code-caption"><span class="code-number">代码 16.11　</span>结合 SQL 与检索的工具增强 RAG</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">langchain.agents</span><span class="w"> </span><span class="kn">import</span> <span class="n">create_tool_calling_agent</span><span class="p">,</span> <span class="n">AgentExecutor</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain.tools</span><span class="w"> </span><span class="kn">import</span> <span class="n">tool</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">search_documents</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""搜索内部文档知识库。"""</span>
    <span class="n">docs</span> <span class="o">=</span> <span class="n">vectorstore</span><span class="o">.</span><span class="n">similarity_search</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">k</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>
    <span class="k">return</span> <span class="s2">"</span><span class="se">\n\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">d</span><span class="o">.</span><span class="n">page_content</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">docs</span><span class="p">)</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">query_database</span><span class="p">(</span><span class="n">sql</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""在分析数据库上执行 SQL 查询。"""</span>
    <span class="k">return</span> <span class="n">db</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">sql</span><span class="p">)</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">web_search</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""在网上搜索最新信息。"""</span>
    <span class="k">return</span> <span class="n">tavily_client</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="n">query</span><span class="p">)</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">execute_python</span><span class="p">(</span><span class="n">code</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""执行 Python 代码进行计算。"""</span>
    <span class="k">return</span> <span class="n">python_repl</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">code</span><span class="p">)</span>

<span class="n">tools</span> <span class="o">=</span> <span class="p">[</span><span class="n">search_documents</span><span class="p">,</span> <span class="n">query_database</span><span class="p">,</span> <span class="n">web_search</span><span class="p">,</span> <span class="n">execute_python</span><span class="p">]</span>
<span class="n">agent</span> <span class="o">=</span> <span class="n">create_tool_calling_agent</span><span class="p">(</span><span class="n">llm</span><span class="p">,</span> <span class="n">tools</span><span class="p">,</span> <span class="n">prompt</span><span class="p">)</span>
<span class="n">executor</span> <span class="o">=</span> <span class="n">AgentExecutor</span><span class="p">(</span><span class="n">agent</span><span class="o">=</span><span class="n">agent</span><span class="p">,</span> <span class="n">tools</span><span class="o">=</span><span class="n">tools</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</code></pre></div>

### 16.7.6　Search-R1：RL 训练的智能体 RAG {#search-r1-rl-trained-agentic-rag}

<p>上述智能体 RAG 方法依赖<em>Prompt 工程</em>的编排——Agent 的搜索行为由指令控制，而非通过训练学习。<strong>Search-R1</strong> <a class="book-citation" id="cite-jin2025searchr1-1" href="../../references.md#ref-jin2025searchr1" aria-label="参考文献 170">[170]</a> 采取了根本不同的方法：通过强化学习训练 LLM，使其<em>学会在推理过程中何时、检索什么、检索多少次</em>。</p>

##### 核心思想。 {#core-idea.}

<p>Search-R1 扩展了 DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 推理框架，将搜索引擎查询视为 RL 训练循环中的<strong>Action</strong>。在思维链（Chain-of-Thought，CoT）生成过程中，模型可以输出特殊 Token <code>&lt;search&gt;query&lt;/search&gt;</code>，触发对搜索引擎的实时检索。检索结果被注入回推理上下文，模型继续生成。</p>

##### 形式化设定。 {#formal-setup.}

<p>模型生成一个与搜索动作交错的推理轨迹：</p>



<div class="arithmatex">\[\underbrace{\text{think}_1}_{\text{reasoning}} \to \underbrace{\texttt{&lt;search&gt;}q_1\texttt{&lt;/search&gt;}}_{\text{action}} \to \underbrace{[\text{results}_1]}_{\text{observation}} \to \text{think}_2 \to \texttt{&lt;search&gt;}q_2\texttt{&lt;/search&gt;} \to \cdots \to \text{answer}\]</div>

<p>整条轨迹（推理 + 搜索 + 最终答案）由一个终端 Reward 打分：最终答案相对于真实标签的正确性。</p>

##### 训练算法。 {#training-algorithm.}

<p>Search-R1 使用组相对策略优化（Group Relative Policy Optimization, GRPO）：</p>

<ol>
<li><p><strong>每个问题采样 <span class="arithmatex">\(N\)</span> 条轨迹</strong>，每条可能包含 0–5 次搜索调用</p></li>
<li><p><strong>实时执行搜索</strong>——环境返回真实的搜索引擎结果</p></li>
<li><p><strong>对终端答案的正确性打分</strong>（与真实答案的精确匹配或 F1）</p></li>
<li><p><strong>计算组相对优势</strong>：<span class="arithmatex">\(\hat{A}_i = (R_i - \mu_G) / \sigma_G\)</span></p></li>
<li><p>使用 GRPO 的裁剪目标<strong>更新 Policy</strong>——强化那些有效搜索的轨迹</p></li>
</ol>

<p>模型学会：</p>

<ul>
<li><p><strong>在不确定时才搜索</strong>——避免对已掌握的知识进行不必要的搜索</p></li>
<li><p><strong>构造有效查询</strong>——学会能返回相关结果的查询措辞</p></li>
<li><p><strong>多次搜索</strong>——基于初次结果迭代精炼查询</p></li>
<li><p><strong>整合检索到的上下文</strong>——利用搜索结果支撑或修正其推理</p></li>
</ul>

##### Search-R1 与基于 Prompt 的智能体 RAG 的差异。 {#how-search-r1-differs-from-prompt-based-agentic-rag.}

<p class="table-caption" id="table-16-4"><span class="table-number">表 16.4　</span>Search-R1（RL 训练）与基于 Prompt 的智能体 RAG 对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>基于 Prompt 的智能体 RAG</strong></th><th scope="col"><strong>Search-R1</strong></th></tr></thead>
<tbody><tr><td>搜索决策</td><td>Prompt / 启发式</td><td>通过 RL 学习</td></tr>
<tr><td>查询构造</td><td>由 Prompt 指引（“改写查询”）</td><td>端到端训练</td></tr>
<tr><td>搜索次数</td><td>固定或推理时由 LLM 决定</td><td>学习到的最优次数</td></tr>
<tr><td>训练信号</td><td>无（冻结模型）</td><td>正确性 Reward</td></tr>
<tr><td>搜索结果整合</td><td>追加到上下文</td><td>交错在 CoT 中</td></tr>
<tr><td>失败恢复</td><td>启发式重试</td><td>学习到的退避 / 重构造</td></tr>
<tr><td>推理开销</td><td>框架开销（LangGraph）</td><td>原生模型行为</td></tr></tbody>
</table>
</div>

##### 结果。 {#results.-1-dup2}

<p>在开放域问答基准（NQ <a class="book-citation" id="cite-kwiatkowski2019natural-1" href="../../references.md#ref-kwiatkowski2019natural" aria-label="参考文献 194">[194]</a>、TriviaQA <a class="book-citation" id="cite-joshi2017triviaqa-1" href="../../references.md#ref-joshi2017triviaqa" aria-label="参考文献 173">[173]</a>、HotpotQA <a class="book-citation" id="cite-yang2018hotpotqa-1" href="../../references.md#ref-yang2018hotpotqa" aria-label="参考文献 405">[405]</a>）上，使用 7B 模型的 Search-R1 优于：</p>

<ul>
<li><p>标准 RAG（单次检索）准确率高出 15–20%</p></li>
<li><p>基于 Prompt 的智能体 RAG（ReAct 风格）准确率高出 8–12%</p></li>
<li><p>接近搭配标准 RAG 的大得多模型（70B）的性能</p></li>
</ul>

<p>关键洞见是：<strong>学习“何时以及如何搜索”比拥有一个知道更多内容的大模型更有价值</strong>。一个善于搜索的小模型可以击败不搜索的大模型。</p>

<div class="admonition tip"><p class="admonition-title">Search-R1：范式转变</p>
<p>传统 RAG 问的是：“对于这个查询，我应该检索什么？”（一种在生成之前做出的流水线决策）。</p>

<p>Search-R1 问的是：“基于我已经推理出的内容，我是否需要更多信息？如果需要，什么具体问题能填补这个空缺？”（一种在生成<em>过程中</em>做出的、可学习的决策）。</p>

<p>这就像考试前先翻教科书的学生，与在做题中途意识到卡住、随即去查阅参考资料的学生之间的差别。后者更高效、更有针对性。</p>
</div>

## 16.8　评测 {#evaluation-dup3}

<p>评测一个 RAG 系统比单独评测检索或生成更难，因为错误可能源自流水线的<em>任何阶段</em>——而且会逐级累积。完美的生成器无法弥补无关的检索结果，而当生成器产生幻觉或忽略上下文时，完美的检索器也会被浪费。</p>

<p>因此，有效的 RAG 评测在<strong>三个层面</strong>进行：</p>

<ol>
<li><p><strong>检索质量</strong>：检索器是否找出了正确的段落？（Recall、Precision、MRR、NDCG）</p></li>
<li><p><strong>生成质量</strong>：答案是否正确、忠实于检索上下文且完整？（正确性、忠实度、答案相关性）</p></li>
<li><p><strong>端到端质量</strong>：整个系统是否令用户满意？（人类偏好、任务成功率、按延迟调整的效用）</p></li>
</ol>

<p>一种常见的失败模式是只优化某一层——例如，通过大 <span class="arithmatex">\(K\)</span> 最大化 Recall@<span class="arithmatex">\(K\)</span> 会让上下文塞满边缘相关的段落，反而<em>降低</em>生成质量。下文的指标涵盖检索与生成两端，便于从业者诊断瓶颈所在阶段。</p>

### 16.8.1　检索指标 {#retrieval-metrics}

<p>令 <span class="arithmatex">\(\mathcal{R}_k\)</span> 为前 <span class="arithmatex">\(k\)</span> 名的检索文档集合，<span class="arithmatex">\(\mathcal{R}^*\)</span> 为相关文档集合。</p>

##### Recall@K。 {#recallk.}



<div class="arithmatex">\[\text{Recall@}K = \frac{|\mathcal{R}_K \cap \mathcal{R}^*|}{|\mathcal{R}^*|}
\tag{16.22}\]</div>

##### Precision@K。 {#precisionk.}



<div class="arithmatex">\[\text{Precision@}K = \frac{|\mathcal{R}_K \cap \mathcal{R}^*|}{K}
\tag{16.23}\]</div>

##### 平均倒数排名（Mean Reciprocal Rank, MRR）。 {#mean-reciprocal-rank-mrr.}



<div class="arithmatex">\[\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
\tag{16.24}\]</div>

<p>其中 <span class="arithmatex">\(\text{rank}_i\)</span> 是查询 <span class="arithmatex">\(i\)</span> 第一个相关文档的排名。</p>

##### 归一化折损累计增益（NDCG@K）。 {#normalized-discounted-cumulative-gain-ndcgk.}



<div class="arithmatex">\[\text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}, \quad
  \text{DCG@}K = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}
\tag{16.25}\]</div>

<p>其中 <span class="arithmatex">\(\text{rel}_i \in \{0, 1, 2, \ldots\}\)</span> 是第 <span class="arithmatex">\(i\)</span> 个结果的分级相关性，IDCG 是理想（完美）的 DCG。</p>

### 16.8.2　生成指标 {#generation-metrics}

##### 忠实度（Faithfulness）。 {#faithfulness.}

<p>衡量生成的答案是否<em>接地</em>于检索到的上下文——即答案中的每一项主张都能归因到某一篇检索到的文档。由 LLM 判官评估：</p>



<div class="arithmatex">\[\text{Faithfulness} = \frac{\text{\# claims supported by context}}{\text{\# total claims in answer}}
\tag{16.26}\]</div>

##### 答案相关性（Answer Relevance）。 {#answer-relevance.}

<p>衡量答案是否针对问题作答。计算方式是从答案中生成问题，并衡量其与原始查询的相似度：</p>



<div class="arithmatex">\[\text{AnswerRelevance} = \frac{1}{N} \sum_{i=1}^{N} \cos\!\left(E(q), E(\hat{q}_i)\right)
\tag{16.27}\]</div>

<p>其中 <span class="arithmatex">\(\hat{q}_i\)</span> 是从答案中生成的问题。</p>

##### 上下文精度与召回率。 {#context-precision-and-recall.}



<div class="arithmatex">\[\begin{align*}
\text{ContextPrecision@}K &amp;= \frac{1}{K} \sum_{k=1}^{K} \text{Precision@}k \cdot \mathbf{1}[\text{doc}_k \text{ is relevant}] \tag{16.28}\\

  \text{ContextRecall} &amp;= \frac{\text{\# ground-truth claims attributable to context}}{\text{\# total ground-truth claims}}
\tag{16.29}
\end{align*}\]</div>

### 16.8.3　RAGAs 框架 {#ragas-framework}

<p>RAGAs（Retrieval Augmented Generation Assessment） <a class="book-citation" id="cite-es2023ragas-1" href="../../references.md#ref-es2023ragas" aria-label="参考文献 85">[85]</a> 提供了一个使用 LLM 判官的无参考评测框架：</p>

<p class="code-caption"><span class="code-number">代码 16.12　</span>RAGAs 评测（v0.1 API；v0.2+ 使用 <code>user_input</code>、<code>response</code>、<code>retrieved_contexts</code>、<code>reference</code>）</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">ragas</span><span class="w"> </span><span class="kn">import</span> <span class="n">evaluate</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">ragas.metrics</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">faithfulness</span><span class="p">,</span>
    <span class="n">answer_relevancy</span><span class="p">,</span>
    <span class="n">context_precision</span><span class="p">,</span>
    <span class="n">context_recall</span><span class="p">,</span>
    <span class="n">answer_correctness</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">datasets</span><span class="w"> </span><span class="kn">import</span> <span class="n">Dataset</span>

<span class="n">eval_dataset</span> <span class="o">=</span> <span class="n">Dataset</span><span class="o">.</span><span class="n">from_dict</span><span class="p">({</span>
    <span class="s2">"question"</span><span class="p">:</span>  <span class="n">questions</span><span class="p">,</span>
    <span class="s2">"answer"</span><span class="p">:</span>    <span class="n">generated_answers</span><span class="p">,</span>
    <span class="s2">"contexts"</span><span class="p">:</span>  <span class="n">retrieved_contexts</span><span class="p">,</span>   <span class="c1"># 嵌套列表（每条样本对应一个上下文列表）</span>
    <span class="s2">"ground_truth"</span><span class="p">:</span> <span class="n">reference_answers</span><span class="p">,</span>
<span class="p">})</span>

<span class="n">results</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span>
    <span class="n">dataset</span><span class="o">=</span><span class="n">eval_dataset</span><span class="p">,</span>
    <span class="n">metrics</span><span class="o">=</span><span class="p">[</span>
        <span class="n">faithfulness</span><span class="p">,</span>
        <span class="n">answer_relevancy</span><span class="p">,</span>
        <span class="n">context_precision</span><span class="p">,</span>
        <span class="n">context_recall</span><span class="p">,</span>
        <span class="n">answer_correctness</span><span class="p">,</span>
    <span class="p">],</span>
<span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">results</span><span class="o">.</span><span class="n">to_pandas</span><span class="p">())</span>
</code></pre></div>

### 16.8.4　常见失败模式 {#common-failure-modes-dup2}

<div class="admonition warning"><p class="admonition-title">需监控的 RAG 失败模式</p>
<ol>
<li><p><strong>检索遗漏</strong>：相关文档存在于语料库中却未被检索到。原因：分块不佳、Embedding 模型不匹配、查询—文档词汇差距。</p></li>
<li><p><strong>上下文污染</strong>：检索到的文档包含误导性或矛盾的信息，导致模型生成错误答案。</p></li>
<li><p><strong>中部迷失（Lost-in-the-Middle）</strong>：LLM 对长上下文的开头和结尾关注更强；中间部分的相关信息可能被忽略 <a class="book-citation" id="cite-liu2023lost-1" href="../../references.md#ref-liu2023lost" aria-label="参考文献 225">[225]</a>。</p></li>
<li><p><strong>过度检索</strong>：检索块过多会稀释相关信号，并增加延迟与成本。</p></li>
<li><p><strong>即便检索仍幻觉</strong>：模型忽略检索到的上下文，基于参数化记忆生成内容，尤其当上下文与训练数据相冲突时。</p></li>
<li><p><strong>引用伪造</strong>：模型将主张归因到并不支持这些主张的文档。</p></li>
</ol>
</div>

## 16.9　生产考量 {#production-considerations}

### 16.9.1　Embedding 模型选型 {#embedding-model-selection}

<p>Embedding 模型是 RAG 系统中影响最大的单一组件选择——它决定了检索质量的上限。该领域发展迅速；Table <a href="#tab:embedding_models">16.5</a> 汇总了当前在成本—质量光谱上的可选项。</p>

<p class="table-caption" id="tab:embedding_models"><span class="table-number">表 16.5　</span>面向生产 RAG 的 Embedding 模型（截至 2026 年）。MTEB 分数为检索、分类、聚类与 STS 任务上的总体平均。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模型</strong></th><th scope="col"><strong>维度</strong></th><th scope="col"><strong>最大 Token</strong></th><th scope="col"><strong>MTEB 均值</strong></th><th scope="col"><strong>访问方式</strong></th><th scope="col"><strong>备注</strong></th></tr></thead>
<tbody><tr><td><em>基于 API（托管）</em></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>Voyage <code>voyage-4-large</code></td><td>1024*</td><td>32K</td><td>—</td><td>API</td><td>最佳检索质量</td></tr>
<tr><td>OpenAI <code>text-embedding-3-large</code></td><td>3072</td><td>8191</td><td>64.6</td><td>API</td><td>套娃式维度（Matryoshka）</td></tr>
<tr><td>Cohere <code>embed-english-v3.0</code></td><td>1024</td><td>512</td><td>64.5</td><td>API</td><td>支持 int8/二值</td></tr>
<tr><td>Google <code>text-embedding-005</code></td><td>768</td><td>2048</td><td>—</td><td>API</td><td>集成于 Vertex AI</td></tr>
<tr><td><em>开放权重（自托管）</em></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td><code>nvidia/NV-Embed-v2</code> <a class="book-citation" id="cite-lee2024nvembed-1" href="../../references.md#ref-lee2024nvembed" aria-label="参考文献 202">[202]</a></td><td>4096</td><td>32K</td><td>72.3</td><td>免费</td><td>MTEB #1（2024 年 9 月）</td></tr>
<tr><td><code>Alibaba-NLP/gte-Qwen2-7B</code> <a class="book-citation" id="cite-li2023gte-1" href="../../references.md#ref-li2023gte" aria-label="参考文献 213">[213]</a></td><td>3584</td><td>32K</td><td>70.2</td><td>免费</td><td>Apache-2.0，多语言</td></tr>
<tr><td><code>BAAI/bge-m3</code> <a class="book-citation" id="cite-chen2024bgem3-1" href="../../references.md#ref-chen2024bgem3" aria-label="参考文献 45">[45]</a></td><td>1024</td><td>8192</td><td>65.0</td><td>免费</td><td>稠密 + 稀疏 + 多向量</td></tr>
<tr><td><code>jinaai/jina-embeddings-v3</code></td><td>1024</td><td>8192</td><td>66.0</td><td>免费</td><td>多语言，LoRA 适配器</td></tr>
<tr><td><code>BAAI/bge-large-en-v1.5</code> <a class="book-citation" id="cite-xiao2023cpack-1" href="../../references.md#ref-xiao2023cpack" aria-label="参考文献 395">[395]</a></td><td>1024</td><td>512</td><td>64.2</td><td>免费</td><td>成熟，生态完善</td></tr></tbody>
</table>
</div>

##### 选型标准。 {#selection-criteria.}

<ul>
<li><p><strong>领域匹配</strong>：专门化模型（如代码领域的 <code>voyage-code-3</code>、金融领域的 <code>voyage-finance-2</code>）在领域任务上可比通用模型高出 5–15%。</p></li>
<li><p><strong>上下文长度</strong>：具有 32K Token 上下文的模型（Voyage-4、NV-Embed-v2）可对整篇文档进行 Embedding 而无需分块，简化流水线。</p></li>
<li><p><strong>套娃式 Embedding（Matryoshka）</strong>：支持灵活输出维度（256–4096）的模型允许你在服务时以质量换取存储/延迟，而无需重新编码。</p></li>
<li><p><strong>量化支持</strong>：模型层面支持 int8 或二值量化的（Cohere、Voyage）能将索引大小缩减 4–32<span class="arithmatex">\(\times\)</span>，召回率损失极小。</p></li>
<li><p><strong>多语言</strong>：对非英语或跨语言 RAG，优先选择明确做过多语言训练的模型（BGE-M3、Jina-v3、Voyage-4）。</p></li>
</ul>

### 16.9.2　向量数据库对比 {#vector-database-comparison}

<p class="table-caption" id="table-16-6"><span class="table-number">表 16.6　</span>面向生产 RAG 系统的向量数据库对比</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>数据库</strong></th><th scope="col"><strong>托管方式</strong></th><th scope="col"><strong>规模</strong></th><th scope="col"><strong>过滤</strong></th><th scope="col"><strong>混合检索</strong></th><th scope="col"><strong>最佳适用场景</strong></th></tr></thead>
<tbody><tr><td>FAISS1</td><td>自托管</td><td>数十亿</td><td>有限</td><td>否</td><td>研究、离线</td></tr>
<tr><td>Pinecone2</td><td>托管</td><td>数十亿</td><td>是</td><td>是</td><td>Serverless，易部署</td></tr>
<tr><td>Weaviate3</td><td>两者皆可</td><td>数十亿</td><td>是</td><td>是</td><td>GraphQL、多模态</td></tr>
<tr><td>Chroma4</td><td>自托管</td><td>百万级</td><td>是</td><td>否</td><td>本地开发、原型</td></tr>
<tr><td>Qdrant5</td><td>两者皆可</td><td>数十亿</td><td>是</td><td>是</td><td>高性能</td></tr>
<tr><td>Milvus6</td><td>两者皆可</td><td>数十亿</td><td>是</td><td>是</td><td>企业级、GPU 加速</td></tr>
<tr><td>pgvector7</td><td>自托管</td><td>百万级</td><td>是</td><td>是</td><td>已有 Postgres 用户</td></tr></tbody>
</table>
</div>

<p>来源：5 <a href="https://qdrant.tech">qdrant.tech</a>；6 <a href="https://milvus.io">milvus.io</a>；7 <a href="https://github.com/pgvector/pgvector">github.com/pgvector/pgvector</a></p>

### 16.9.3　延迟优化 {#latency-optimization}

<ol>
<li><p><strong>预过滤</strong>：在 ANN 搜索之前，使用元数据过滤器（日期范围、类别、来源）缩减搜索空间</p></li>
<li><p><strong>近似最近邻</strong>：使用 HNSW 或 IVF 索引代替精确搜索；以约 1% 的召回率损失换取 <span class="arithmatex">\(10\times\)</span> 加速</p></li>
<li><p><strong>Embedding 缓存</strong>：为频繁重复出现的查询缓存 Embedding</p></li>
<li><p><strong>异步检索</strong>：并行地从多个来源检索</p></li>
<li><p><strong>流式生成</strong>：在检索完成的同时流式输出 LLM 结果</p></li>
<li><p><strong>量化</strong>：对 Embedding 使用 int8 或二值量化，以降低内存并提升吞吐</p></li>
</ol>

##### 异步并行检索。 {#async-parallel-retrieval.}

<p>上述技术 (3) 和 (4) 天然可组合：缓存查询 Embedding，然后将检索请求扇出（fan out）到多个后端并发执行。在多源 RAG 系统（Section <a href="#sec:agentic_rag">16.7</a>）中，用户查询可能需要同时从向量数据库、关键词索引和 Web API 中获取结果。顺序检索会累加延迟；并行检索仅需支付<em>最慢</em>来源的代价。Listing <a href="#lst:async_retrieve">16.13</a> 使用 Python 的 <code>asyncio</code> 展示了这一模式——<code>lru_cache</code> 装饰器确保重复查询完全跳过 Embedding 模型，而 <code>asyncio.gather</code> 同时分派所有来源的查询。</p>

<span id="lst:async_retrieve"></span><p class="code-caption"><span class="code-number">代码 16.13　</span>面向低延迟的异步并行检索</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">asyncio</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">functools</span><span class="w"> </span><span class="kn">import</span> <span class="n">lru_cache</span>

<span class="nd">@lru_cache</span><span class="p">(</span><span class="n">maxsize</span><span class="o">=</span><span class="mi">1024</span><span class="p">)</span>
<span class="k">def</span><span class="w"> </span><span class="nf">get_cached_embedding</span><span class="p">(</span><span class="n">text</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="nb">float</span><span class="p">]:</span>
    <span class="k">return</span> <span class="n">embedding_model</span><span class="o">.</span><span class="n">embed_query</span><span class="p">(</span><span class="n">text</span><span class="p">)</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">parallel_retrieve</span><span class="p">(</span>
    <span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
    <span class="n">sources</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">],</span>
    <span class="n">k</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5</span>
<span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">]:</span>
<span class="w">    </span><span class="sd">"""并行从多个来源检索。"""</span>
    <span class="n">tasks</span> <span class="o">=</span> <span class="p">[</span>
        <span class="n">asyncio</span><span class="o">.</span><span class="n">create_task</span><span class="p">(</span><span class="n">retrieve_from_source_async</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">src</span><span class="p">,</span> <span class="n">k</span><span class="p">))</span>
        <span class="k">for</span> <span class="n">src</span> <span class="ow">in</span> <span class="n">sources</span>
    <span class="p">]</span>
    <span class="n">results</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="n">tasks</span><span class="p">,</span> <span class="n">return_exceptions</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="c1"># 扁平化并去重</span>
    <span class="n">all_docs</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">results</span><span class="p">:</span>
        <span class="k">if</span> <span class="ow">not</span> <span class="nb">isinstance</span><span class="p">(</span><span class="n">r</span><span class="p">,</span> <span class="ne">Exception</span><span class="p">):</span>
            <span class="n">all_docs</span><span class="o">.</span><span class="n">extend</span><span class="p">(</span><span class="n">r</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">deduplicate_by_content</span><span class="p">(</span><span class="n">all_docs</span><span class="p">)</span>
</code></pre></div>

### 16.9.4　增量索引与版本管理 {#incremental-indexing-and-versioning}

<p>在生产环境中，文档语料库从不是静态的——政策会被修订、新报告每天涌入、过时内容必须移除。一次完整的重新索引（重新分块、重新 Embedding、重新上传）代价高昂并会导致停机。增量索引通过在文档级别应用变更来解决这一问题。</p>

##### 核心操作。 {#core-operations.}

<ul>
<li><p><strong>Upsert（更新插入）</strong>：当一篇文档被创建或更新时，删除该 <code>doc_id</code> 对应的所有已有块，对新内容重新分块、Embedding 并插入。这保证不会有过时片段残留。</p></li>
<li><p><strong>删除 / 过期</strong>：按文档 ID 移除块（显式删除）或按 TTL（针对新闻、市场数据等时效性来源的自动垃圾回收）。</p></li>
<li><p><strong>版本追踪</strong>：在块元数据中存储 <code>version</code> 和 <code>indexed_at</code> 时间戳。这支持回滚（从来源恢复旧版本）与可审计性（“模型看到了哪个版本？”）。</p></li>
</ul>

##### 一致性挑战。 {#consistency-challenges.}

<ul>
<li><p><strong>Embedding 模型漂移</strong>：升级 Embedding 模型时，新旧向量不兼容。解决方案：(a) 为每个模型版本维护独立索引，在后台迁移；或 (b) 使用 Matryoshka 兼容的模型，通过维度截断保持兼容性。</p></li>
<li><p><strong>分块边界变化</strong>：更改分块策略会使所有现有块失效。版本元数据让你能识别并选择性地重新索引受影响的文档。</p></li>
<li><p><strong>最终一致性</strong>：在分布式向量数据库中，新 upsert 的向量可能无法立即被检索到。请将流水线设计为可容忍短暂的索引滞后（通常为秒级到分钟级）。</p></li>
</ul>

##### 实现。 {#implementation.}

<p>Listing <a href="#lst:incremental_index">16.14</a> 展示了一个最小化的 <code>RAGIndexManager</code> 类，封装了 upsert 与过期逻辑，适合包装任何支持元数据过滤的向量存储。</p>

<span id="lst:incremental_index"></span><p class="code-caption"><span class="code-number">代码 16.14　</span>带版本管理的增量索引更新</p>
<div class="highlight"><pre><span></span><code><span class="k">class</span><span class="w"> </span><span class="nc">RAGIndexManager</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">vectorstore</span><span class="p">,</span> <span class="n">metadata_store</span><span class="p">,</span> <span class="n">chunker</span><span class="p">,</span> <span class="n">embedder</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">vs</span> <span class="o">=</span> <span class="n">vectorstore</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">meta</span> <span class="o">=</span> <span class="n">metadata_store</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">chunker</span> <span class="o">=</span> <span class="n">chunker</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">embedder</span> <span class="o">=</span> <span class="n">embedder</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">upsert_document</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">doc_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">content</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                        <span class="n">metadata</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="kc">None</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""添加或更新一篇文档，替换旧块。"""</span>
        <span class="c1"># 删除该文档现有的块</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">vs</span><span class="o">.</span><span class="n">delete</span><span class="p">(</span><span class="nb">filter</span><span class="o">=</span><span class="p">{</span><span class="s2">"doc_id"</span><span class="p">:</span> <span class="n">doc_id</span><span class="p">})</span>
        <span class="c1"># 对新版本分块（向量库内部完成 Embedding）</span>
        <span class="n">chunks</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">chunker</span><span class="o">.</span><span class="n">split_text</span><span class="p">(</span><span class="n">content</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">vs</span><span class="o">.</span><span class="n">add_texts</span><span class="p">(</span>
            <span class="n">texts</span><span class="o">=</span><span class="n">chunks</span><span class="p">,</span>
            <span class="n">metadatas</span><span class="o">=</span><span class="p">[{</span><span class="o">**</span><span class="n">metadata</span><span class="p">,</span> <span class="s2">"doc_id"</span><span class="p">:</span> <span class="n">doc_id</span><span class="p">,</span>
                        <span class="s2">"version"</span><span class="p">:</span> <span class="n">metadata</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"version"</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span>
                        <span class="s2">"indexed_at"</span><span class="p">:</span> <span class="n">datetime</span><span class="o">.</span><span class="n">utcnow</span><span class="p">()</span><span class="o">.</span><span class="n">isoformat</span><span class="p">()}</span>
                       <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="n">chunks</span><span class="p">],</span>
        <span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">expire_old_documents</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">ttl_days</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">365</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">int</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""移除早于 TTL 的文档。"""</span>
        <span class="n">cutoff</span> <span class="o">=</span> <span class="p">(</span><span class="n">datetime</span><span class="o">.</span><span class="n">utcnow</span><span class="p">()</span> <span class="o">-</span> <span class="n">timedelta</span><span class="p">(</span><span class="n">days</span><span class="o">=</span><span class="n">ttl_days</span><span class="p">))</span><span class="o">.</span><span class="n">isoformat</span><span class="p">()</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">vs</span><span class="o">.</span><span class="n">delete</span><span class="p">(</span><span class="nb">filter</span><span class="o">=</span><span class="p">{</span><span class="s2">"indexed_at"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"$lt"</span><span class="p">:</span> <span class="n">cutoff</span><span class="p">}})</span>
</code></pre></div>

## 16.10　RAG 与微调的协同 {#rag-fine-tuning-synergy}

### 16.10.1　何时将 RAG 与微调结合 {#when-to-combine-rag-with-fine-tuning}

<p>微调与 RAG 解决的是互补的弱点：</p>

<ul>
<li><p><strong>仅微调</strong>：模型学到风格与格式，但可能在事实上产生幻觉</p></li>
<li><p><strong>仅 RAG</strong>：模型能访问事实，但可能不知道如何最优地使用它们</p></li>
<li><p><strong>组合</strong>：对模型进行微调，使其<em>善于使用检索到的上下文</em>——引用来源、承认不确定性、忽略无关上下文</p></li>
</ul>

### 16.10.2　RAFT：检索增强微调（Retrieval-Augmented Fine-Tuning） {#raft-retrieval-augmented-fine-tuning}

<p>RAFT <a class="book-citation" id="cite-zhang2024raft-1" href="../../references.md#ref-zhang2024raft" aria-label="参考文献 425">[425]</a> 在混合了相关文档与<em>干扰</em>文档的设置下训练模型回答问题，教会模型识别并只使用相关上下文：</p>

<ol>
<li><p>对每个训练样本 <span class="arithmatex">\((q, a, d^*)\)</span>，采样 <span class="arithmatex">\(k-1\)</span> 个干扰文档 <span class="arithmatex">\(\{d_i^-\}\)</span></p></li>
<li><p>在 <code>[q, </code><span class="arithmatex">\(d^*\)</span><code>, </code><span class="arithmatex">\(d_1^-\)</span><code>, …, </code><span class="arithmatex">\(d_{k-1}^-\)</span><code>]</code> <span class="arithmatex">\(\to\)</span> <code>[chain-of-thought + a]</code> 上微调</p></li>
<li><p>思维链显式引用 <span class="arithmatex">\(d^*\)</span> 中的内容，教模型对答案进行接地</p></li>
</ol>



<div class="arithmatex">\[\mathcal{L}_{\text{RAFT}} = -\mathbb{E}_{(q,a,d^*,\{d_i^-\})} \left[
    \log P_\theta\!\left(\text{CoT}(d^*) \oplus a \;\middle|\; q, d^*, \{d_i^-\}\right)
  \right]
\tag{16.30}\]</div>

### 16.10.3　检索器—生成器联合训练 {#joint-retriever-generator-training}

<p>为获得极致性能，可以对检索器与生成器进行联合训练。REALM <a class="book-citation" id="cite-guu2020realm-1" href="../../references.md#ref-guu2020realm" aria-label="参考文献 128">[128]</a> 与 RAG <a class="book-citation" id="cite-lewis2020retrieval-1--section-10" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a> 论文提出了端到端训练方法，让 Gradient 流经检索步骤：</p>



<div class="arithmatex">\[\nabla_\theta \mathcal{L} = \nabla_\theta \left[
    -\log \sum_{d \in \mathcal{D}} P_\theta(a \mid q, d) \cdot P_\phi(d \mid q)
  \right]
\tag{16.31}\]</div>

<p>检索器参数 <span class="arithmatex">\(\phi\)</span> 通过 REINFORCE 估计器更新，或将 <span class="arithmatex">\(P_\phi(d \mid q)\)</span> 视为关于文档的可微 Attention 进行更新。</p>

<div class="admonition warning"><p class="admonition-title">联合训练的挑战</p>
<p>检索器—生成器联合训练强大但复杂：(1) 随着 <span class="arithmatex">\(\phi\)</span> 变化，文档索引必须定期刷新（异步索引刷新），(2) 训练信号稀疏（仅 top-<span class="arithmatex">\(k\)</span> 文档参与贡献），(3) 在未从预训练检索器仔细初始化的情况下，训练不稳定。</p>
</div>

## 16.11　各种 RAG 方法的综合对比 {#comprehensive-rag-approach-comparison}

<p class="table-caption" id="table-16-7"><span class="table-number">表 16.7　</span>各种 RAG 方法在关键维度上的对比</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>准确率</strong></th><th scope="col"><strong>延迟</strong></th><th scope="col"><strong>复杂度</strong></th><th scope="col"><strong>成本</strong></th><th scope="col"><strong>最佳适用场景</strong></th></tr></thead>
<tbody><tr><td>朴素 RAG <a class="book-citation" id="cite-lewis2020retrieval-1--section-11" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a></td><td>中等</td><td>低</td><td>低</td><td>低</td><td>原型、简单问答</td></tr>
<tr><td>RAG + 重排序 <a class="book-citation" id="cite-nogueira2019passage-1--section-11" href="../../references.md#ref-nogueira2019passage" aria-label="参考文献 268">[268]</a></td><td>高</td><td>中等</td><td>中等</td><td>中等</td><td>生产级问答系统</td></tr>
<tr><td>HyDE <a class="book-citation" id="cite-gao2022precise-1--section-11" href="../../references.md#ref-gao2022precise" aria-label="参考文献 106">[106]</a></td><td>高</td><td>中等</td><td>低</td><td>中等</td><td>语义不匹配的领域</td></tr>
<tr><td>多查询 RAG</td><td>高</td><td>中等</td><td>中等</td><td>中等</td><td>歧义查询</td></tr>
<tr><td>RAG-Fusion <a class="book-citation" id="cite-rackauckas2023ragfusion-1--section-11" href="../../references.md#ref-rackauckas2023ragfusion" aria-label="参考文献 300">[300]</a></td><td>高</td><td>中等</td><td>中等</td><td>中等</td><td>多样化的查询类型</td></tr>
<tr><td>Self-RAG <a class="book-citation" id="cite-asai2023selfrag-1--section-11" href="../../references.md#ref-asai2023selfrag" aria-label="参考文献 13">[13]</a></td><td>高</td><td>中等</td><td>高</td><td>中等</td><td>选择性检索</td></tr>
<tr><td>CRAG <a class="book-citation" id="cite-yan2024crag-1--section-11" href="../../references.md#ref-yan2024crag" aria-label="参考文献 401">[401]</a></td><td>高</td><td>中等</td><td>高</td><td>高</td><td>不可靠的语料库</td></tr>
<tr><td>自适应 RAG <a class="book-citation" id="cite-jeong2024adaptive-1--section-11" href="../../references.md#ref-jeong2024adaptive" aria-label="参考文献 162">[162]</a></td><td>高</td><td>低–高</td><td>高</td><td>中等</td><td>查询复杂度混合</td></tr>
<tr><td>Graph RAG <a class="book-citation" id="cite-edge2024local-1--section-11" href="../../references.md#ref-edge2024local" aria-label="参考文献 82">[82]</a></td><td>极高</td><td>高</td><td>极高</td><td>高</td><td>全局综合查询</td></tr>
<tr><td>智能体 RAG</td><td>极高</td><td>高</td><td>极高</td><td>高</td><td>多跳推理</td></tr>
<tr><td>RAFT <a class="book-citation" id="cite-zhang2024raft-1--section-11" href="../../references.md#ref-zhang2024raft" aria-label="参考文献 425">[425]</a></td><td>极高</td><td>低</td><td>极高</td><td>极高</td><td>特定领域部署</td></tr></tbody>
</table>
</div>

<div class="admonition question"><p class="admonition-title">RAG 系统的关键设计问题</p>
<p>在为生产环境设计 RAG 系统时，请考虑以下问题：</p>

<ol>
<li><p><strong>查询分布是怎样的？</strong>事实型 vs. 分析型 vs. 多跳查询需要不同的检索策略。</p></li>
<li><p><strong>语料库有多大、变化多频繁？</strong>数百万文档且频繁更新更适合具备增量索引的托管向量数据库。</p></li>
<li><p><strong>延迟需求如何？</strong>低于 100ms 的响应排除了重排序和智能体循环；批处理或异步场景则可以承受。</p></li>
<li><p><strong>接地有多关键？</strong>高风险领域（医疗、法律、金融）需要忠实度评估与引用验证。</p></li>
<li><p><strong>词汇是否专业化？</strong>特定领域术语可能需要混合检索或经过领域适配的 Embedding 模型。</p></li>
</ol>
</div>

<div class="admonition note"><p class="admonition-title">RAG 最佳实践总结</p>
<ul>
<li><p><strong>从简单开始</strong>：分块得当的朴素 RAG 往往优于分块糟糕的复杂系统</p></li>
<li><p><strong>独立评估检索</strong>：先修好检索，再优化生成</p></li>
<li><p><strong>使用混合检索</strong>：BM25 + 稠密检索 + RRF 是一个强大的默认方案</p></li>
<li><p><strong>加上重排序</strong>：在 top-20 候选上使用交叉编码器重排序器具有很高的 ROI</p></li>
<li><p><strong>监控忠实度</strong>：在生产中使用 LLM 判官跟踪幻觉率</p></li>
<li><p><strong>积极缓存</strong>：文档只 Embedding 一次；缓存频繁查询的 Embedding</p></li>
<li><p><strong>带重叠地分块</strong>：10–15% 的重叠可防止在边界处丢失信息</p></li>
<li><p><strong>存储丰富的元数据</strong>：来源、日期、章节和文档类型能支持强大的预过滤，显著提升精度</p></li>
</ul>
</div>
