# 16.4　分块策略 {#sec:chunking}

<p>分块是将文档切分为以下段落的过程：(1) 足够小，能放入 Embedding 模型的上下文窗口；(2) 语义连贯；(3) 即使被单独检索出来，也包含足够的上下文以发挥作用。</p>

## 16.4.1　带重叠的固定大小分块 {#fixed-size-chunking-with-overlap}

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

## 16.4.2　语义分块 {#semantic-chunking}

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

## 16.4.3　基于文档结构的分块 {#document-structure-aware-chunking}

<p>对于结构化文档（Markdown、HTML、代码），在自然边界处切分：</p>

<ul>
<li><p><strong>Markdown</strong>：在 <code>##</code> 标题处切分，保留章节上下文</p></li>
<li><p><strong>HTML</strong>：在 <code>&lt;section&gt;</code>、<code>&lt;article&gt;</code>、<code>&lt;p&gt;</code> 标签处切分</p></li>
<li><p><strong>代码</strong>：在函数/类定义处切分，并在每个块中保留 import 语句</p></li>
<li><p><strong>表格</strong>：将整个表格保留为单个块；切勿在行中间切分</p></li>
</ul>

## 16.4.4　父—子分块 {#parent-child-chunking}

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

## 16.4.5　块大小的经验指南 {#empirical-guidelines-for-chunk-size}

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
