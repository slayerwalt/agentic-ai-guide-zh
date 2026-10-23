# 26.7　生产部署模式 {#subsec:production-deployment}

<p>要大规模部署 Agent，必须在执行模型、状态管理和资源分配三方面格外审慎。</p>

<figure id="fig:deployment-arch"><a href="../../assets/figures/fig_076_deployment-arch.png" target="_blank" rel="noopener" aria-label="查看图 26.5 原图" title="点击查看原图"><img src="../../assets/figures/fig_076_deployment-arch.png" alt="基于队列的异步 Agent 部署。worker 从队列拉取任务，并独立持久化各自状态。" loading="lazy"></a><figcaption><span class="figure-number">图 26.5　</span>基于队列的异步 Agent 部署。worker 从队列拉取任务，并独立持久化各自状态。</figcaption></figure>

## 26.7.1　异步 Agent 执行 {#async-agent-execution}

<p>长时间运行的 Agent 应异步执行，以避免阻塞 API 连接。Celery20 是 Python 生态中广泛使用的分布式任务队列，能处理重试、worker 扩缩容与结果持久化：</p>

<p class="code-caption"><span class="code-number">代码 26.23　</span>基于 Celery 的异步 Agent 执行</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">celery</span><span class="w"> </span><span class="kn">import</span> <span class="n">Celery</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">myagent</span><span class="w"> </span><span class="kn">import</span> <span class="n">ResearchAgent</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">redis</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">time</span>

<span class="n">app</span> <span class="o">=</span> <span class="n">Celery</span><span class="p">(</span><span class="s2">"agent_tasks"</span><span class="p">,</span> <span class="n">broker</span><span class="o">=</span><span class="s2">"redis://localhost:6379/0"</span><span class="p">)</span>
<span class="n">state_store</span> <span class="o">=</span> <span class="n">redis</span><span class="o">.</span><span class="n">Redis</span><span class="p">(</span><span class="n">host</span><span class="o">=</span><span class="s2">"localhost"</span><span class="p">,</span> <span class="n">port</span><span class="o">=</span><span class="mi">6379</span><span class="p">,</span> <span class="n">db</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="nd">@app</span><span class="o">.</span><span class="n">task</span><span class="p">(</span><span class="n">bind</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">max_retries</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">default_retry_delay</span><span class="o">=</span><span class="mi">60</span><span class="p">)</span>
<span class="k">def</span><span class="w"> </span><span class="nf">run_agent_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">task_input</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">config</span><span class="p">:</span> <span class="nb">dict</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""异步执行一个 Agent 任务。"""</span>
    <span class="k">try</span><span class="p">:</span>
        <span class="c1"># 更新任务状态</span>
        <span class="n">state_store</span><span class="o">.</span><span class="n">hset</span><span class="p">(</span><span class="sa">f</span><span class="s2">"task:</span><span class="si">{</span><span class="n">task_id</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span> <span class="n">mapping</span><span class="o">=</span><span class="p">{</span>
            <span class="s2">"status"</span><span class="p">:</span> <span class="s2">"running"</span><span class="p">,</span>
            <span class="s2">"started_at"</span><span class="p">:</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">(),</span>
            <span class="s2">"worker"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">request</span><span class="o">.</span><span class="n">hostname</span><span class="p">,</span>
        <span class="p">})</span>

        <span class="n">agent</span> <span class="o">=</span> <span class="n">ResearchAgent</span><span class="p">(</span><span class="o">**</span><span class="n">config</span><span class="p">)</span>
        <span class="n">result</span> <span class="o">=</span> <span class="n">agent</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">task_input</span><span class="p">)</span>

        <span class="c1"># 存储结果</span>
        <span class="n">state_store</span><span class="o">.</span><span class="n">hset</span><span class="p">(</span><span class="sa">f</span><span class="s2">"task:</span><span class="si">{</span><span class="n">task_id</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span> <span class="n">mapping</span><span class="o">=</span><span class="p">{</span>
            <span class="s2">"status"</span><span class="p">:</span> <span class="s2">"completed"</span><span class="p">,</span>
            <span class="s2">"result"</span><span class="p">:</span> <span class="n">result</span><span class="o">.</span><span class="n">to_json</span><span class="p">(),</span>
            <span class="s2">"completed_at"</span><span class="p">:</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">(),</span>
            <span class="s2">"cost_usd"</span><span class="p">:</span> <span class="n">result</span><span class="o">.</span><span class="n">cost_usd</span><span class="p">,</span>
        <span class="p">})</span>
        <span class="k">return</span> <span class="p">{</span><span class="s2">"task_id"</span><span class="p">:</span> <span class="n">task_id</span><span class="p">,</span> <span class="s2">"status"</span><span class="p">:</span> <span class="s2">"completed"</span><span class="p">}</span>

    <span class="k">except</span> <span class="ne">Exception</span> <span class="k">as</span> <span class="n">exc</span><span class="p">:</span>
        <span class="n">state_store</span><span class="o">.</span><span class="n">hset</span><span class="p">(</span><span class="sa">f</span><span class="s2">"task:</span><span class="si">{</span><span class="n">task_id</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span> <span class="n">mapping</span><span class="o">=</span><span class="p">{</span>
            <span class="s2">"status"</span><span class="p">:</span> <span class="s2">"failed"</span><span class="p">,</span>
            <span class="s2">"error"</span><span class="p">:</span> <span class="nb">str</span><span class="p">(</span><span class="n">exc</span><span class="p">),</span>
            <span class="s2">"failed_at"</span><span class="p">:</span> <span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">(),</span>
        <span class="p">})</span>
        <span class="k">raise</span> <span class="bp">self</span><span class="o">.</span><span class="n">retry</span><span class="p">(</span><span class="n">exc</span><span class="o">=</span><span class="n">exc</span><span class="p">)</span>

<span class="c1"># API 端点（独立的 Flask/FastAPI 应用）</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">flask</span><span class="w"> </span><span class="kn">import</span> <span class="n">Flask</span><span class="p">,</span> <span class="n">request</span><span class="p">,</span> <span class="n">jsonify</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">uuid</span>

<span class="n">web_app</span> <span class="o">=</span> <span class="n">Flask</span><span class="p">(</span><span class="vm">__name__</span><span class="p">)</span>

<span class="nd">@web_app</span><span class="o">.</span><span class="n">route</span><span class="p">(</span><span class="s2">"/tasks"</span><span class="p">,</span> <span class="n">methods</span><span class="o">=</span><span class="p">[</span><span class="s2">"POST"</span><span class="p">])</span>
<span class="k">def</span><span class="w"> </span><span class="nf">submit_task</span><span class="p">():</span>
    <span class="n">task_id</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">())</span>
    <span class="n">task</span> <span class="o">=</span> <span class="n">run_agent_task</span><span class="o">.</span><span class="n">delay</span><span class="p">(</span>
        <span class="n">task_id</span><span class="o">=</span><span class="n">task_id</span><span class="p">,</span>
        <span class="n">task_input</span><span class="o">=</span><span class="n">request</span><span class="o">.</span><span class="n">json</span><span class="p">[</span><span class="s2">"input"</span><span class="p">],</span>
        <span class="n">config</span><span class="o">=</span><span class="n">request</span><span class="o">.</span><span class="n">json</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"config"</span><span class="p">,</span> <span class="p">{}),</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="n">jsonify</span><span class="p">({</span><span class="s2">"task_id"</span><span class="p">:</span> <span class="n">task_id</span><span class="p">,</span> <span class="s2">"celery_id"</span><span class="p">:</span> <span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">}),</span> <span class="mi">202</span>
</code></pre></div>

## 26.7.2　多租户隔离 {#multi-tenant-isolation}

<p>面向多个客户的生产 Agent 系统必须实现严格隔离：</p>

<ul>
<li><p><strong>命名空间隔离</strong>：每个租户的状态、记忆和 Tool 配置存储在独立命名空间中</p></li>
<li><p><strong>限流</strong>：按租户限定 LLM 调用、Tool 调用与计算时间的速率</p></li>
<li><p><strong>资源配额</strong>：为每个租户设置最大并发 Agent 数、Token 预算和存储上限</p></li>
<li><p><strong>审计日志</strong>：所有 Agent 动作连同租户 ID 一并记录，用于合规与计费</p></li>
</ul>

## 26.7.3　成本优化策略 {#cost-optimization-strategies-dup2}

<ul>
<li><p><strong>模型路由</strong>：在简单子任务（分类、抽取）上使用更小更便宜的模型，把大模型留给复杂推理</p></li>
<li><p><strong>Prompt 缓存</strong>：OpenAI 与 Anthropic 都提供对重复系统 Prompt 的缓存，可为高流量 Agent 降低多达 90% 的成本</p></li>
<li><p><strong>结果缓存</strong>：在一定时间窗口内对相同输入的 Tool 结果进行缓存</p></li>
<li><p><strong>批处理</strong>：在延迟允许时，把多次独立的 LLM 调用合并批处理</p></li>
<li><p><strong>提前终止</strong>：检测到 Agent 已掌握足够信息可作答时，及早结束循环</p></li>
</ul>

<p class="code-caption"><span class="code-number">代码 26.24　</span>面向成本优化的模型路由</p>
<div class="highlight"><pre><span></span><code><span class="k">class</span><span class="w"> </span><span class="nc">CostOptimizedRouter</span><span class="p">:</span>
    <span class="n">TASK_MODEL_MAP</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">"classification"</span><span class="p">:</span> <span class="s2">"gpt-4o-mini"</span><span class="p">,</span>
        <span class="s2">"extraction"</span><span class="p">:</span> <span class="s2">"gpt-4o-mini"</span><span class="p">,</span>
        <span class="s2">"summarization"</span><span class="p">:</span> <span class="s2">"gpt-4o-mini"</span><span class="p">,</span>
        <span class="s2">"reasoning"</span><span class="p">:</span> <span class="s2">"gpt-4o"</span><span class="p">,</span>
        <span class="s2">"code_generation"</span><span class="p">:</span> <span class="s2">"gpt-4o"</span><span class="p">,</span>
        <span class="s2">"complex_analysis"</span><span class="p">:</span> <span class="s2">"o1"</span><span class="p">,</span>
    <span class="p">}</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">route</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task_type</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">complexity</span><span class="p">:</span> <span class="nb">float</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="n">base_model</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">TASK_MODEL_MAP</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="n">task_type</span><span class="p">,</span> <span class="s2">"gpt-4o-mini"</span><span class="p">)</span>
        <span class="c1"># 对高复杂度任务升级到更强模型</span>
        <span class="k">if</span> <span class="n">complexity</span> <span class="o">&gt;</span> <span class="mf">0.8</span> <span class="ow">and</span> <span class="n">base_model</span> <span class="o">==</span> <span class="s2">"gpt-4o-mini"</span><span class="p">:</span>
            <span class="k">return</span> <span class="s2">"gpt-4o"</span>
        <span class="k">return</span> <span class="n">base_model</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">estimate_cost</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">model</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">input_tokens</span><span class="p">:</span> <span class="nb">int</span><span class="p">,</span> <span class="n">output_tokens</span><span class="p">:</span> <span class="nb">int</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
        <span class="n">pricing</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s2">"gpt-4o-mini"</span><span class="p">:</span> <span class="p">(</span><span class="mf">0.15e-6</span><span class="p">,</span> <span class="mf">0.60e-6</span><span class="p">),</span>
            <span class="s2">"gpt-4o"</span><span class="p">:</span>      <span class="p">(</span><span class="mf">2.50e-6</span><span class="p">,</span> <span class="mf">10.0e-6</span><span class="p">),</span>
            <span class="s2">"o1"</span><span class="p">:</span>          <span class="p">(</span><span class="mf">15.0e-6</span><span class="p">,</span> <span class="mf">60.0e-6</span><span class="p">),</span>
        <span class="p">}</span>
        <span class="n">in_price</span><span class="p">,</span> <span class="n">out_price</span> <span class="o">=</span> <span class="n">pricing</span><span class="p">[</span><span class="n">model</span><span class="p">]</span>
        <span class="k">return</span> <span class="n">input_tokens</span> <span class="o">*</span> <span class="n">in_price</span> <span class="o">+</span> <span class="n">output_tokens</span> <span class="o">*</span> <span class="n">out_price</span>
</code></pre></div>

## 26.7.4　自动扩缩容策略 {#auto-scaling-strategies}

<p>Agent 工作负载呈现突发且难以预测的特征。有效的自动扩缩容需要：</p>

<ul>
<li><p><strong>基于队列深度的扩缩</strong>：根据任务队列深度而非 CPU 利用率来调整 worker 数量</p></li>
<li><p><strong>预测式扩缩</strong>：利用历史模式（时段、星期）在需求高峰之前提前扩容</p></li>
<li><p><strong>抢占式实例</strong>：长任务可结合检查点使用 spot/可抢占（preemptible）实例以节省成本</p></li>
<li><p><strong>优雅停机</strong>：worker 在缩容前完成当前任务，避免状态损坏</p></li>
</ul>
