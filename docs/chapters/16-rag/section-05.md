# 16.5　高级 RAG 模式 {#advanced-rag-patterns}

## 16.5.1　查询变换 {#query-transformation}

<p>用户的原始查询往往含糊、过短，或与文档语言匹配不佳。查询变换技术在搜索之前改进检索。</p>

#### HyDE（假设性文档 Embedding，Hypothetical Document Embeddings） <a class="book-citation" id="cite-gao2022precise-1" href="../../references.md#ref-gao2022precise" aria-label="参考文献 106">[106]</a>。 {#hyde-hypothetical-document-embeddings-.}

<p>不直接对查询做 Embedding，而是生成一个<em>假设性回答</em>并对其做 Embedding：</p>



<div class="arithmatex">\[\hat{d} = \text{LLM}(q), \quad \mathbf{e}_{\text{query}} = f_\phi(\hat{d})
\tag{16.20}\]</div>

<p>直觉在于：假设性回答与真实文档处于相同的语言风格中，从而缩小查询—文档之间的分布差距。</p>

#### 后退式提示（Step-Back Prompting）。 {#step-back-prompting.}

<p>对于具体问题，先生成一个更通用的“后退”问题，对两者都进行检索，再合并上下文。例如：“乙醇在 2 atm 下的沸点是多少？” <span class="arithmatex">\(\to\)</span> 后退问题：“哪些因素影响液体的沸点？”</p>

#### 多查询生成。 {#multi-query-generation.}

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

## 16.5.2　重排序 {#re-ranking}

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

## 16.5.3　上下文压缩 {#contextual-compression}

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

## 16.5.4　Self-RAG {#self-rag}

<p>Self-RAG <a class="book-citation" id="cite-asai2023selfrag-1" href="../../references.md#ref-asai2023selfrag" aria-label="参考文献 13">[13]</a> 训练单个模型完成以下任务：(1) 决定<em>是否</em>进行检索，(2) 在带或不带检索的条件下生成，(3) 使用特殊的反思 Token <em>批评</em>自己的输出：</p>

<ul>
<li><p><code>[Retrieve]</code>：模型是否应检索更多段落？</p></li>
<li><p><code>[IsRel]</code>：检索到的段落是否与查询相关？</p></li>
<li><p><code>[IsSup]</code>：生成的陈述是否可由检索到的段落支持？</p></li>
<li><p><code>[IsUse]</code>：整体回复是否有用？</p></li>
</ul>

<p>模型经过端到端训练，与回复一起预测这些 Token，从而实现对检索与自我评分的细粒度控制。</p>

## 16.5.5　CRAG：纠正式 RAG {#crag-corrective-rag}

<p>CRAG <a class="book-citation" id="cite-yan2024crag-1" href="../../references.md#ref-yan2024crag" aria-label="参考文献 401">[401]</a> 增加了一个<em>检索评估器</em>，对检索到的文档打分并触发纠正动作：</p>

<ol>
<li><p>检索 top-<span class="arithmatex">\(k\)</span> 文档</p></li>
<li><p>对每个文档打分：<strong>正确</strong> / <strong>模糊</strong> / <strong>错误</strong></p></li>
<li><p>如果所有文档都为错误或模糊 <span class="arithmatex">\(\to\)</span> 回退到网页搜索</p></li>
<li><p>如果部分文档正确 <span class="arithmatex">\(\to\)</span> 使用知识精炼（剔除无关句子）</p></li>
<li><p>基于精炼后的上下文生成回答</p></li>
</ol>

## 16.5.6　自适应 RAG {#adaptive-rag}

<p>自适应 RAG（Adaptive RAG） <a class="book-citation" id="cite-jeong2024adaptive-1" href="../../references.md#ref-jeong2024adaptive" aria-label="参考文献 162">[162]</a> 根据预测出的查询复杂度将其路由到不同的检索策略：</p>

<ul>
<li><p><strong>不检索</strong>：模型可凭参数化知识回答的简单事实型查询</p></li>
<li><p><strong>单步 RAG</strong>：对中等复杂度的查询采用标准的“先检索后生成”</p></li>
<li><p><strong>多步 RAG</strong>：针对复杂的多跳问题进行迭代检索</p></li>
</ul>

<p>一个在查询复杂度标签上训练的轻量级分类器对每个传入查询进行路由。</p>

## 16.5.7　Graph RAG {#graph-rag}

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

## 16.5.8　RAG-Fusion {#rag-fusion}

<p>RAG-Fusion <a class="book-citation" id="cite-rackauckas2023ragfusion-1" href="../../references.md#ref-rackauckas2023ragfusion" aria-label="参考文献 300">[300]</a> 从原始查询生成多条搜索查询，对每条都进行检索，再使用 RRF（Equation <a href="section-03.md#eq:rrf">16.8</a>）融合排名列表：</p>

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
