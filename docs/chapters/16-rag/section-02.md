# 16.2　核心 RAG 架构 {#core-rag-architecture}

<p>一个标准的 RAG 系统由两个阶段构成：处理和存储文档的<strong>离线索引流水线</strong>，以及为查询提供服务的<strong>在线检索—生成流水线</strong>。</p>

## 16.2.1　完整流水线图 {#full-pipeline-diagram}

<figure id="fig:rag_arch"><a href="../../assets/figures/fig_050_rag_arch.png" target="_blank" rel="noopener" aria-label="查看图 16.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_050_rag_arch.png" alt="端到端 RAG 架构。离线流水线（蓝色）对文档进行一次性索引；在线流水线（绿色/橙色）在推理时为每个查询提供服务。" loading="lazy"></a><figcaption><span class="figure-number">图 16.1　</span>端到端 RAG 架构。离线流水线（蓝色）对文档进行一次性索引；在线流水线（绿色/橙色）在推理时为每个查询提供服务。</figcaption></figure>

## 16.2.2　索引流水线 {#indexing-pipeline}

#### 文档加载。 {#document-loading.}

<p>文档以异构格式抵达（PDF、HTML、Markdown、DOCX、代码）。加载器提取干净的文本，并保留元数据（来源 URL、页码、章节标题、时间戳），这些元数据会与 Embedding 一并存储，用于过滤和引用。</p>

#### 分块（Chunking）。 {#chunking.}

<p>长文档必须切分成可放入 Embedding 模型上下文窗口（通常为 512 个 Token）并保持语义连贯的块。分块策略是 RAG 系统设计中影响最大的决策之一（参见 Section <a href="section-04.md#sec:chunking">16.4</a>）。</p>

#### Embedding。 {#embedding.}

<p>每个块 <span class="arithmatex">\(c_i\)</span> 通过 Embedding 模型 <span class="arithmatex">\(f_\phi\)</span> 编码为稠密向量 <span class="arithmatex">\(\mathbf{e}_i = f_\phi(c_i) \in \mathbb{R}^d\)</span>。这些向量与原始文本和元数据一同存储到向量数据库中。</p>

## 16.2.3　检索 {#retrieval}

<p>给定查询 <span class="arithmatex">\(q\)</span>，检索步骤将其编码为 <span class="arithmatex">\(\mathbf{q} = f_\phi(q)\)</span>，并通过余弦相似度找出最相似的 <span class="arithmatex">\(k\)</span> 个块：</p>



<div class="arithmatex">\[\text{sim}(\mathbf{q}, \mathbf{e}_i) = \frac{\mathbf{q} \cdot \mathbf{e}_i}{\|\mathbf{q}\|\,\|\mathbf{e}_i\|}
\tag{16.3}\]</div>

<p>返回 top-<span class="arithmatex">\(k\)</span> 块 <span class="arithmatex">\(\mathcal{C}_k = \{c_{(1)}, \ldots, c_{(k)}\}\)</span> 作为上下文。</p>

## 16.2.4　生成 {#generation}

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
