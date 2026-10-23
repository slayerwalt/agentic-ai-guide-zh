# 16.3　检索方法 {#retrieval-methods}

## 16.3.1　稀疏检索：BM25 与 TF-IDF {#sparse-retrieval-bm25-and-tf-idf}

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

## 16.3.2　稠密检索：DPR {#dense-retrieval-dpr}

<p>稠密段落检索（Dense Passage Retrieval, DPR） <a class="book-citation" id="cite-karpukhin2020dense-1" href="../../references.md#ref-karpukhin2020dense" aria-label="参考文献 180">[180]</a> 使用两个独立的、基于 BERT 的编码器——一个<em>查询编码器</em> <span class="arithmatex">\(E_Q\)</span> 和一个<em>段落编码器</em> <span class="arithmatex">\(E_P\)</span>——通过对比 Loss 进行训练，使相关的查询—段落对在 Embedding 空间中彼此靠近。</p>

#### 双编码器（Bi-Encoder）架构。 {#bi-encoder-architecture.}



<div class="arithmatex">\[\text{sim}(q, p) = E_Q(q)^\top E_P(p)
\tag{16.5}\]</div>

#### 使用 Batch 内负样本进行训练。 {#training-with-in-batch-negatives.}

<p>给定包含 <span class="arithmatex">\(B\)</span> 个查询—段落对的 Batch <span class="arithmatex">\(\{(q_i, p_i^+)\}_{i=1}^B\)</span>，对比 Loss 将 Batch 中所有其他段落视为负样本：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{DPR}} = -\frac{1}{B} \sum_{i=1}^{B}
    \log \frac{\exp\!\left(E_Q(q_i)^\top E_P(p_i^+) / \tau\right)}
              {\sum_{j=1}^{B} \exp\!\left(E_Q(q_i)^\top E_P(p_j) / \tau\right)}
\tag{16.6}\]</div>

<p>其中 <span class="arithmatex">\(\tau\)</span> 是温度超参数。硬负样本（词面相似但语义无关的段落）对训练强检索器至关重要。</p>

#### 近似最近邻搜索。 {#approximate-nearest-neighbor-search.}

<p>在大规模场景下，对数百万个 Embedding 进行穷举搜索不可行。FAISS <a class="book-citation" id="cite-johnson2019billion-1" href="../../references.md#ref-johnson2019billion" aria-label="参考文献 171">[171]</a>（Facebook AI Similarity Search）提供了高效的近似最近邻（Approximate Nearest Neighbor, ANN）搜索，所用方法包括：</p>

<ul>
<li><p><strong>IVF（倒排文件索引，Inverted File Index）</strong>：将向量聚类到 Voronoi 单元中；只搜索邻近的单元</p></li>
<li><p><strong>HNSW（分层可导航小世界，Hierarchical Navigable Small World）</strong> <a class="book-citation" id="cite-malkov2018efficient-1" href="../../references.md#ref-malkov2018efficient" aria-label="参考文献 248">[248]</a>：基于图的索引，搜索复杂度 <span class="arithmatex">\(O(\log N)\)</span></p></li>
<li><p><strong>PQ（乘积量化，Product Quantization）</strong>：压缩向量以减少内存占用</p></li>
</ul>

## 16.3.3　基于倒数排名融合的混合检索 {#hybrid-retrieval-with-reciprocal-rank-fusion}

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

## 16.3.4　学习式稀疏检索：SPLADE 与 SPLADEv2 {#learned-sparse-retrieval-splade-and-spladev2}

<div class="admonition tip"><p class="admonition-title">为什么需要 SPLADE？</p>
<p>传统的稀疏检索（BM25）依赖精确的词面匹配——当查询说 &quot;car&quot; 而文档说 &quot;automobile&quot; 时它就会失败。稠密检索（DPR）能捕捉语义但缺乏可解释性，查询时需要 GPU，且索引体积大。<strong>SPLADE</strong> 兼具二者的优点：稀疏向量（像 BM25 一样支持快速的倒排索引查找）配合可学习的语义扩展（像稠密模型一样处理同义词和相关概念）。</p>
</div>

#### SPLADE（v1）—— 核心思想。 {#splade-v1-core-idea.}

<p>SPLADE（Sparse Lexical and Expansion Model） <a class="book-citation" id="cite-formal2021splade-1" href="../../references.md#ref-formal2021splade" aria-label="参考文献 95">[95]</a> 使用预训练的掩码语言模型（如 BERT/DistilBERT），为每个文档或查询生成一个覆盖<em>整个词汇表</em>的稀疏向量。关键洞见在于：MLM 头部已经知道文本中每个位置在语义上与哪些词相关——SPLADE 把这种知识重新利用为词项重要性权重。</p>

#### 架构。 {#architecture.}

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

#### 打分。 {#scoring.}

<p>查询和文档分别映射为稀疏向量 <span class="arithmatex">\(\mathbf{w}^q, \mathbf{w}^d \in \mathbb{R}^{|\mathcal{V}|}\)</span>。相关性得分就是一个简单的点积：</p>



<div class="arithmatex">\[s(q, d) = \sum_{t \in \mathcal{V}} w_t^q \cdot w_t^d
\tag{16.10}\]</div>

<p>由于两个向量都是稀疏的（在 30K 词汇表中通常只有 20–200 个非零项），这可以使用标准倒排索引（Lucene、Anserini）高效计算——查询时无需 GPU。</p>

#### 训练。 {#training.}

<p>SPLADE 通过对比学习（Batch 内负样本 + 硬负样本）进行训练，并附加两个正则项：</p>



<div class="arithmatex">\[\mathcal{L} = \mathcal{L}_{\text{contrastive}} + \lambda_q \|\mathbf{w}^q\|_1 + \lambda_d \|\mathbf{w}^d\|_1
\tag{16.11}\]</div>

<p>查询和文档表示上的 <span class="arithmatex">\(L_1\)</span> 惩罚鼓励稀疏性——若没有它们，模型将学到稠密表示，从而背离设计初衷。</p>

#### SPLADEv2 —— 关键改进。 {#spladev2-key-improvements.}

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

## 16.3.5　ColBERT：晚期交互 {#colbert-late-interaction}

<p>ColBERT <a class="book-citation" id="cite-khattab2020colbert-1" href="../../references.md#ref-khattab2020colbert" aria-label="参考文献 182">[182]</a> 将查询和文档编码为 Token 级 Embedding 的<em>集合</em>，并使用 <em>MaxSim</em> 算子进行打分：</p>

<span id="eq:colbert"></span>

<div class="arithmatex">\[s(q, d) = \sum_{i \in |\mathbf{q}|} \max_{j \in |\mathbf{d}|} \mathbf{q}_i^\top \mathbf{d}_j

\tag{16.15}\]</div>

<p>这种晚期交互（Late Interaction）机制比单向量双编码器更具表达力，又因为文档 Embedding 可离线预计算而远比交叉编码器更快。</p>

#### 架构。 {#architecture.-1}

<p>查询编码器 <span class="arithmatex">\(E_Q\)</span> 和文档编码器 <span class="arithmatex">\(E_D\)</span> 都是基于 BERT 的模型，产生<em>逐 Token</em> 的 Embedding（而不是单个 [CLS] 向量）。每个 Token Embedding 通过一个线性层投影到更低维度（通常为 128）：</p>



<div class="arithmatex">\[\begin{align*}
\mathbf{q}_i &amp;= \text{Linear}(E_Q(q)_i) \in \mathbb{R}^{128}, \quad i = 1, \ldots, |q| \tag{16.16}\\

  \mathbf{d}_j &amp;= \text{Linear}(E_D(d)_j) \in \mathbb{R}^{128}, \quad j = 1, \ldots, |d|
\tag{16.17}
\end{align*}\]</div>

#### 训练。 {#training.-1}

<p>ColBERT 在正、负段落上以成对的 Softmax 交叉熵 Loss 进行训练。给定查询 <span class="arithmatex">\(q\)</span>、正段落 <span class="arithmatex">\(d^+\)</span> 和一组负段落 <span class="arithmatex">\(\{d^-_1, \ldots, d^-_N\}\)</span>：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{ColBERT}} = -\log \frac{\exp(s(q, d^+))}{\exp(s(q, d^+)) + \sum_{k=1}^{N} \exp(s(q, d^-_k))}
\tag{16.18}\]</div>

<p>其中 <span class="arithmatex">\(s(q, d)\)</span> 是 Equation <a href="#eq:colbert">16.15</a> 中的 MaxSim 分数。负样本来自：</p>

<ul>
<li><p><strong>Batch 内负样本</strong>：同一训练 Batch 中的其他段落（无额外成本、数量充足）</p></li>
<li><p><strong>硬负样本</strong>：由 BM25 检索出的、词面相似但语义无关的段落（对质量影响最大）</p></li>
<li><p><strong>蒸馏负样本</strong>（ColBERTv2 <a class="book-citation" id="cite-santhanam2022colbertv2-1" href="../../references.md#ref-santhanam2022colbertv2" aria-label="参考文献 313">[313]</a>）：使用交叉编码器教师挖掘最难的负样本，并将其得分蒸馏到 ColBERT 中</p></li>
</ul>

#### 索引与服务。 {#indexing-and-serving.}

<p>在索引阶段，所有文档 Token 的 Embedding 都被预计算并存储（在 ColBERTv2 中可选地使用残差量化压缩）。在查询阶段，只需实时编码查询 Token，并与已存储的文档 Embedding 计算 MaxSim。这种分离带来：</p>

<ul>
<li><p><strong>离线文档编码</strong>：编码一次，服务于多次查询</p></li>
<li><p><strong>PLAID 索引</strong> <a class="book-citation" id="cite-santhanam2022colbertv2-2" href="../../references.md#ref-santhanam2022colbertv2" aria-label="参考文献 313">[313]</a>：对文档 Embedding 聚类，用聚类中心进行初步候选检索，然后只在候选集上计算精确的 MaxSim——延迟可降低 5–10<span class="arithmatex">\(\times\)</span></p></li>
<li><p><strong>索引大小</strong>：每文档 <span class="arithmatex">\(|d| \times 128\)</span> 个浮点数（比单向量方法更大，但通过量化可压缩到约每维度 2 字节）</p></li>
</ul>

## 16.3.6　检索方法对比 {#retrieval-method-comparison}

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
