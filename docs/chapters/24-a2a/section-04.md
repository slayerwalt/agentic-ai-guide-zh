# 24.4　Agent 发现与路由 {#sec:a2a:discovery}

<p>在一个 Agent 能够与另一个 Agent 通信之前，它必须先<em>找到</em>对方。Agent 发现就是定位可以处理特定任务的 Agent 的过程。</p>

## 24.4.1　Agent 注册中心 {#agent-registries}

<p><strong>Agent 注册中心</strong>是一种目录服务，用于索引 Agent Card 并提供搜索与查找 API。常见的部署模型有两种：</p>

<p>中心化注册中心（Centralized Registry）</p>

<p>单一权威注册中心（例如企业服务目录）索引所有 Agent。运维简单，但会形成单点故障，且可能无法扩展到跨组织部署。</p>

<p>联合注册中心（Federated Registry）</p>

<p>由多个注册中心组成，每个对特定领域或组织具有权威性，并通过跨注册中心搜索协议互通。更具韧性和隐私保护，但需要标准化的联合协议。</p>

## 24.4.2　基于能力的路由 {#capability-based-routing}

<p>编排器不会硬编码 Agent 的 URL，而是进行<strong>基于能力的路由</strong>：先向注册中心查询匹配所需技能的 Agent，再从中选择最合适的。</p>

<div class="highlight"><pre><span></span><code><span class="k">class</span><span class="w"> </span><span class="nc">AgentRouter</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""根据能力匹配将任务路由到 Agent。"""</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">registry_url</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">registry_url</span> <span class="o">=</span> <span class="n">registry_url</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_cache</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">list</span><span class="p">[</span><span class="n">AgentCard</span><span class="p">]]</span> <span class="o">=</span> <span class="p">{}</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">find_agents</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">required_skill</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                          <span class="n">tags</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="n">AgentCard</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""向注册中心查询具备所需技能的 Agent。"""</span>
        <span class="n">params</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"skill"</span><span class="p">:</span> <span class="n">required_skill</span><span class="p">}</span>
        <span class="k">if</span> <span class="n">tags</span><span class="p">:</span>
            <span class="n">params</span><span class="p">[</span><span class="s2">"tags"</span><span class="p">]</span> <span class="o">=</span> <span class="s2">","</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">tags</span><span class="p">)</span>
        <span class="k">async</span> <span class="k">with</span> <span class="n">httpx</span><span class="o">.</span><span class="n">AsyncClient</span><span class="p">()</span> <span class="k">as</span> <span class="n">client</span><span class="p">:</span>
            <span class="n">resp</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">registry_url</span><span class="si">}</span><span class="s2">/agents"</span><span class="p">,</span> <span class="n">params</span><span class="o">=</span><span class="n">params</span><span class="p">)</span>
            <span class="k">return</span> <span class="p">[</span><span class="n">AgentCard</span><span class="p">(</span><span class="o">**</span><span class="n">card</span><span class="p">)</span> <span class="k">for</span> <span class="n">card</span> <span class="ow">in</span> <span class="n">resp</span><span class="o">.</span><span class="n">json</span><span class="p">()[</span><span class="s2">"agents"</span><span class="p">]]</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">route</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task_description</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AgentCard</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""在语义上将任务描述匹配到最合适的可用 Agent。"""</span>
        <span class="c1"># 对任务描述进行嵌入</span>
        <span class="n">task_embedding</span> <span class="o">=</span> <span class="k">await</span> <span class="n">embed</span><span class="p">(</span><span class="n">task_description</span><span class="p">)</span>

        <span class="c1"># 获取所有已注册的 Agent</span>
        <span class="n">all_agents</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">find_agents</span><span class="p">(</span><span class="n">required_skill</span><span class="o">=</span><span class="s2">"*"</span><span class="p">)</span>

        <span class="c1"># 按任务与 Agent 描述的余弦相似度对每个 Agent 评分</span>
        <span class="n">scored</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">agent</span> <span class="ow">in</span> <span class="n">all_agents</span><span class="p">:</span>
            <span class="n">agent_embedding</span> <span class="o">=</span> <span class="k">await</span> <span class="n">embed</span><span class="p">(</span><span class="n">agent</span><span class="o">.</span><span class="n">description</span><span class="p">)</span>
            <span class="n">score</span> <span class="o">=</span> <span class="n">cosine_similarity</span><span class="p">(</span><span class="n">task_embedding</span><span class="p">,</span> <span class="n">agent_embedding</span><span class="p">)</span>
            <span class="n">scored</span><span class="o">.</span><span class="n">append</span><span class="p">((</span><span class="n">score</span><span class="p">,</span> <span class="n">agent</span><span class="p">))</span>

        <span class="c1"># 返回得分最高的 Agent</span>
        <span class="n">scored</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">key</span><span class="o">=</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">x</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">reverse</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">scored</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span>
</code></pre></div>

## 24.4.3　等价 Agent 间的负载均衡 {#load-balancing-across-equivalent-agents}

<p>当多个 Agent 提供相同能力时，路由器必须分配负载。常用策略包括：</p>

<ul>
<li><p><strong>轮询（Round-robin）</strong>：在所有可用 Agent 之间平均分发任务。</p></li>
<li><p><strong>最少负载（Least-loaded）</strong>：路由到活跃任务最少的 Agent（需要健康/指标端点）。</p></li>
<li><p><strong>延迟感知（Latency-aware）</strong>：路由到近期响应时间最低的 Agent。</p></li>
<li><p><strong>亲和性路由（Affinity-based）</strong>：将相关任务路由到同一个 Agent，以利用已缓存的上下文。</p></li>
</ul>

## 24.4.4　版本管理与兼容性 {#version-management-and-compatibility}

<p>Agent Card 包含 <code>version</code> 字段。编排器应指定最低版本要求，并在只有旧版本可用时进行优雅降级。推荐使用语义化版本（Semantic Versioning） <a class="book-citation" id="cite-preston2024semver-1" href="../../references.md#ref-preston2024semver" aria-label="参考文献 293">[293]</a>（<code>MAJOR.MINOR.PATCH</code>）：破坏性接口变更递增 <code>MAJOR</code>，新增能力递增 <code>MINOR</code>。</p>

<div class="admonition warning"><p class="admonition-title">长生命周期系统中的版本偏斜</p>
<p>在生产环境的多 Agent 系统中，不同 Agent 可能在不同时间被升级，从而产生版本偏斜（Version Skew）。一个针对 Agent Card v2.1 编译的编排器，可能遇到仍运行 v1.3 的 Agent。务必实现向后兼容的消息处理，并明确测试跨版本场景。</p>
</div>
