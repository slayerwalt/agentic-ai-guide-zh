# 17.10　实现模式 {#sec:memory-implementation}

## 17.10.1　基于 Embedding 的向量存储记忆 {#vector-store-memory-with-embeddings}

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

## 17.10.2　分层记忆管理器 {#hierarchical-memory-manager}

<p>受 MemGPT <a class="book-citation" id="cite-packer2023memgpt-1" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a> 启发，这一模式把记忆组织为三层：<em>热（hot）</em>（在上下文内，立即可访问）、<em>温（warm）</em>（向量存储，可快速检索）和<em>冷（cold）</em>（归档，容量无上限）。条目根据访问频率与重要性自动升级或降级——类似 CPU 的缓存层级。</p>

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

## 17.10.3　记忆增强的 Agent 循环 {#memory-augmented-agent-loop}

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
