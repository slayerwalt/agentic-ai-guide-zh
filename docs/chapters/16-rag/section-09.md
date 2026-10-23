# 16.9　生产考量 {#production-considerations}

## 16.9.1　Embedding 模型选型 {#embedding-model-selection}

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

#### 选型标准。 {#selection-criteria.}

<ul>
<li><p><strong>领域匹配</strong>：专门化模型（如代码领域的 <code>voyage-code-3</code>、金融领域的 <code>voyage-finance-2</code>）在领域任务上可比通用模型高出 5–15%。</p></li>
<li><p><strong>上下文长度</strong>：具有 32K Token 上下文的模型（Voyage-4、NV-Embed-v2）可对整篇文档进行 Embedding 而无需分块，简化流水线。</p></li>
<li><p><strong>套娃式 Embedding（Matryoshka）</strong>：支持灵活输出维度（256–4096）的模型允许你在服务时以质量换取存储/延迟，而无需重新编码。</p></li>
<li><p><strong>量化支持</strong>：模型层面支持 int8 或二值量化的（Cohere、Voyage）能将索引大小缩减 4–32<span class="arithmatex">\(\times\)</span>，召回率损失极小。</p></li>
<li><p><strong>多语言</strong>：对非英语或跨语言 RAG，优先选择明确做过多语言训练的模型（BGE-M3、Jina-v3、Voyage-4）。</p></li>
</ul>

## 16.9.2　向量数据库对比 {#vector-database-comparison}

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

## 16.9.3　延迟优化 {#latency-optimization}

<ol>
<li><p><strong>预过滤</strong>：在 ANN 搜索之前，使用元数据过滤器（日期范围、类别、来源）缩减搜索空间</p></li>
<li><p><strong>近似最近邻</strong>：使用 HNSW 或 IVF 索引代替精确搜索；以约 1% 的召回率损失换取 <span class="arithmatex">\(10\times\)</span> 加速</p></li>
<li><p><strong>Embedding 缓存</strong>：为频繁重复出现的查询缓存 Embedding</p></li>
<li><p><strong>异步检索</strong>：并行地从多个来源检索</p></li>
<li><p><strong>流式生成</strong>：在检索完成的同时流式输出 LLM 结果</p></li>
<li><p><strong>量化</strong>：对 Embedding 使用 int8 或二值量化，以降低内存并提升吞吐</p></li>
</ol>

#### 异步并行检索。 {#async-parallel-retrieval.}

<p>上述技术 (3) 和 (4) 天然可组合：缓存查询 Embedding，然后将检索请求扇出（fan out）到多个后端并发执行。在多源 RAG 系统（Section <a href="section-07.md#sec:agentic_rag">16.7</a>）中，用户查询可能需要同时从向量数据库、关键词索引和 Web API 中获取结果。顺序检索会累加延迟；并行检索仅需支付<em>最慢</em>来源的代价。Listing <a href="#lst:async_retrieve">16.13</a> 使用 Python 的 <code>asyncio</code> 展示了这一模式——<code>lru_cache</code> 装饰器确保重复查询完全跳过 Embedding 模型，而 <code>asyncio.gather</code> 同时分派所有来源的查询。</p>

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

## 16.9.4　增量索引与版本管理 {#incremental-indexing-and-versioning}

<p>在生产环境中，文档语料库从不是静态的——政策会被修订、新报告每天涌入、过时内容必须移除。一次完整的重新索引（重新分块、重新 Embedding、重新上传）代价高昂并会导致停机。增量索引通过在文档级别应用变更来解决这一问题。</p>

#### 核心操作。 {#core-operations.}

<ul>
<li><p><strong>Upsert（更新插入）</strong>：当一篇文档被创建或更新时，删除该 <code>doc_id</code> 对应的所有已有块，对新内容重新分块、Embedding 并插入。这保证不会有过时片段残留。</p></li>
<li><p><strong>删除 / 过期</strong>：按文档 ID 移除块（显式删除）或按 TTL（针对新闻、市场数据等时效性来源的自动垃圾回收）。</p></li>
<li><p><strong>版本追踪</strong>：在块元数据中存储 <code>version</code> 和 <code>indexed_at</code> 时间戳。这支持回滚（从来源恢复旧版本）与可审计性（“模型看到了哪个版本？”）。</p></li>
</ul>

#### 一致性挑战。 {#consistency-challenges.}

<ul>
<li><p><strong>Embedding 模型漂移</strong>：升级 Embedding 模型时，新旧向量不兼容。解决方案：(a) 为每个模型版本维护独立索引，在后台迁移；或 (b) 使用 Matryoshka 兼容的模型，通过维度截断保持兼容性。</p></li>
<li><p><strong>分块边界变化</strong>：更改分块策略会使所有现有块失效。版本元数据让你能识别并选择性地重新索引受影响的文档。</p></li>
<li><p><strong>最终一致性</strong>：在分布式向量数据库中，新 upsert 的向量可能无法立即被检索到。请将流水线设计为可容忍短暂的索引滞后（通常为秒级到分钟级）。</p></li>
</ul>

#### 实现。 {#implementation.}

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
