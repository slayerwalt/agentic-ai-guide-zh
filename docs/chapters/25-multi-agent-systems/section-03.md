# 25.3　协调机制 {#subsec:coordination-mechanisms}

<p>Agent 如何协调——如何共享信息、分工以及解决冲突——与拓扑同等重要。六种典型的协调机制适用于基于 LLM 的多智能体系统。</p>

## 25.3.1　共享状态（全局黑板） {#subsubsec:shared-state}

<p><strong>黑板架构（blackboard architecture）</strong> <a class="book-citation" id="cite-hayes1985blackboard-1" href="../../references.md#ref-hayes1985blackboard" aria-label="参考文献 135">[135]</a> 提供一个所有 Agent 都可以读写的共享数据结构。在 LLM 系统中，它通常实现为一个共享字典、数据库或结构化文档。</p>

<p class="code-caption"><span class="code-number">代码 25.1　</span>具备冲突解决的共享黑板</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">threading</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">dataclasses</span><span class="w"> </span><span class="kn">import</span> <span class="n">dataclass</span><span class="p">,</span> <span class="n">field</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Any</span><span class="p">,</span> <span class="n">Callable</span><span class="p">,</span> <span class="n">Dict</span><span class="p">,</span> <span class="n">List</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">BlackboardEntry</span><span class="p">:</span>
    <span class="n">value</span><span class="p">:</span> <span class="n">Any</span>
    <span class="n">author</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">timestamp</span><span class="p">:</span> <span class="nb">float</span>
    <span class="n">confidence</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1.0</span>

<span class="k">class</span><span class="w"> </span><span class="nc">Blackboard</span><span class="p">:</span>
    <span class="c1"># 用于多 agent 协调的线程安全共享状态。</span>
<span class="w">    </span><span class="sd">"""Thread-safe shared state for multi-agent coordination."""</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_data</span><span class="p">:</span> <span class="n">Dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">BlackboardEntry</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_lock</span> <span class="o">=</span> <span class="n">threading</span><span class="o">.</span><span class="n">RLock</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_subscribers</span><span class="p">:</span> <span class="n">Dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">List</span><span class="p">[</span><span class="n">Callable</span><span class="p">]]</span> <span class="o">=</span> <span class="p">{}</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">write</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">key</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">value</span><span class="p">:</span> <span class="n">Any</span><span class="p">,</span> <span class="n">author</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
              <span class="n">confidence</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1.0</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">bool</span><span class="p">:</span>
        <span class="c1"># 写入黑板；置信度更高的条目在冲突中获胜。</span>
<span class="w">        </span><span class="sd">"""Write to blackboard; higher-confidence entries win conflicts."""</span>
        <span class="k">with</span> <span class="bp">self</span><span class="o">.</span><span class="n">_lock</span><span class="p">:</span>
            <span class="n">existing</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_data</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="n">key</span><span class="p">)</span>
            <span class="k">if</span> <span class="n">existing</span> <span class="ow">and</span> <span class="n">existing</span><span class="o">.</span><span class="n">confidence</span> <span class="o">&gt;</span> <span class="n">confidence</span><span class="p">:</span>
                <span class="k">return</span> <span class="kc">False</span>  <span class="c1"># 冲突：已有条目获胜</span>
            <span class="kn">import</span><span class="w"> </span><span class="nn">time</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">_data</span><span class="p">[</span><span class="n">key</span><span class="p">]</span> <span class="o">=</span> <span class="n">BlackboardEntry</span><span class="p">(</span>
                <span class="n">value</span><span class="o">=</span><span class="n">value</span><span class="p">,</span> <span class="n">author</span><span class="o">=</span><span class="n">author</span><span class="p">,</span>
                <span class="n">timestamp</span><span class="o">=</span><span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">(),</span> <span class="n">confidence</span><span class="o">=</span><span class="n">confidence</span>
            <span class="p">)</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">_notify</span><span class="p">(</span><span class="n">key</span><span class="p">,</span> <span class="n">value</span><span class="p">)</span>
            <span class="k">return</span> <span class="kc">True</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">read</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">key</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Any</span><span class="p">:</span>
        <span class="k">with</span> <span class="bp">self</span><span class="o">.</span><span class="n">_lock</span><span class="p">:</span>
            <span class="n">entry</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_data</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="n">key</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">entry</span><span class="o">.</span><span class="n">value</span> <span class="k">if</span> <span class="n">entry</span> <span class="k">else</span> <span class="kc">None</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">subscribe</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">key</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">callback</span><span class="p">:</span> <span class="n">Callable</span><span class="p">):</span>
        <span class="c1"># Agent 订阅特定 key 上的变化。</span>
<span class="w">        </span><span class="sd">"""Agents subscribe to changes on specific keys."""</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_subscribers</span><span class="o">.</span><span class="n">setdefault</span><span class="p">(</span><span class="n">key</span><span class="p">,</span> <span class="p">[])</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">callback</span><span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_notify</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">key</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">value</span><span class="p">:</span> <span class="n">Any</span><span class="p">):</span>
        <span class="k">for</span> <span class="n">cb</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">_subscribers</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="n">key</span><span class="p">,</span> <span class="p">[]):</span>
            <span class="n">cb</span><span class="p">(</span><span class="n">key</span><span class="p">,</span> <span class="n">value</span><span class="p">)</span>
</code></pre></div>

## 25.3.2　消息传递（Message Passing） {#subsubsec:message-passing}

<p>消息传递是 LLM Agent 最自然的协调机制：Agent 通过相互发送结构化文本消息来通信。关键的设计决策包括：</p>

<ul>
<li><p><strong>消息格式</strong>：结构化（JSON schema）、自然语言或混合形式</p></li>
<li><p><strong>路由</strong>：直接（Agent 到 Agent）、广播或基于主题的发布/订阅</p></li>
<li><p><strong>会话线程</strong>：在多轮对话（Multi-Turn）交流中维持上下文</p></li>
<li><p><strong>确认（Acknowledgment）</strong>：发送方是否需要接收/处理的确认</p></li>
</ul>

## 25.3.3　规划与分解 {#subsubsec:planning-decomp}

<p>Manager Agent 将高层任务分解为子任务的<strong>有向无环图（directed acyclic graph, DAG）</strong>，将每个子任务分配给合适的 Worker，并跟踪依赖关系。这是经典分层任务网络（hierarchical task network, HTN）规划在多 Agent 场景下的类比。</p>

<p class="code-caption"><span class="code-number">代码 25.2　</span>任务 DAG 分解</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">dataclasses</span><span class="w"> </span><span class="kn">import</span> <span class="n">dataclass</span><span class="p">,</span> <span class="n">field</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">List</span><span class="p">,</span> <span class="n">Optional</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">asyncio</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">Task</span><span class="p">:</span>
    <span class="nb">id</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">description</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">assigned_to</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">dependencies</span><span class="p">:</span> <span class="n">List</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="n">field</span><span class="p">(</span><span class="n">default_factory</span><span class="o">=</span><span class="nb">list</span><span class="p">)</span>
    <span class="n">status</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">"pending"</span>   <span class="c1"># pending | running | done | failed</span>
    <span class="n">result</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>

<span class="k">class</span><span class="w"> </span><span class="nc">TaskDAG</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">Task</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">add_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task</span><span class="p">:</span> <span class="n">Task</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">]</span> <span class="o">=</span> <span class="n">task</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">ready_tasks</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">List</span><span class="p">[</span><span class="n">Task</span><span class="p">]:</span>
        <span class="c1"># 返回依赖已全部完成的任务。</span>
<span class="w">        </span><span class="sd">"""Return tasks whose dependencies are all completed."""</span>
        <span class="k">return</span> <span class="p">[</span>
            <span class="n">t</span> <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="o">.</span><span class="n">values</span><span class="p">()</span>
            <span class="k">if</span> <span class="n">t</span><span class="o">.</span><span class="n">status</span> <span class="o">==</span> <span class="s2">"pending"</span>
            <span class="ow">and</span> <span class="nb">all</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">d</span><span class="p">]</span><span class="o">.</span><span class="n">status</span> <span class="o">==</span> <span class="s2">"done"</span>
                    <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">t</span><span class="o">.</span><span class="n">dependencies</span><span class="p">)</span>
        <span class="p">]</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">execute</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">agent_pool</span><span class="p">:</span> <span class="nb">dict</span><span class="p">):</span>
        <span class="k">while</span> <span class="nb">any</span><span class="p">(</span><span class="n">t</span><span class="o">.</span><span class="n">status</span> <span class="o">!=</span> <span class="s2">"done"</span> <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="o">.</span><span class="n">values</span><span class="p">()):</span>
            <span class="n">ready</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">ready_tasks</span><span class="p">()</span>
            <span class="k">if</span> <span class="ow">not</span> <span class="n">ready</span><span class="p">:</span>
                <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="mf">0.1</span><span class="p">)</span>
                <span class="k">continue</span>
            <span class="c1"># 并行执行就绪任务</span>
            <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">_run_task</span><span class="p">(</span><span class="n">t</span><span class="p">,</span> <span class="n">agent_pool</span><span class="p">[</span><span class="n">t</span><span class="o">.</span><span class="n">assigned_to</span><span class="p">])</span>
                <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="n">ready</span>
            <span class="p">])</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_run_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task</span><span class="p">:</span> <span class="n">Task</span><span class="p">,</span> <span class="n">agent</span><span class="p">):</span>
        <span class="n">task</span><span class="o">.</span><span class="n">status</span> <span class="o">=</span> <span class="s2">"running"</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">task</span><span class="o">.</span><span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="n">agent</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="n">task</span><span class="o">.</span><span class="n">description</span><span class="p">)</span>
            <span class="n">task</span><span class="o">.</span><span class="n">status</span> <span class="o">=</span> <span class="s2">"done"</span>
        <span class="k">except</span> <span class="ne">Exception</span> <span class="k">as</span> <span class="n">e</span><span class="p">:</span>
            <span class="n">task</span><span class="o">.</span><span class="n">status</span> <span class="o">=</span> <span class="s2">"failed"</span>
            <span class="k">raise</span>
</code></pre></div>

## 25.3.4　投票与共识 {#subsubsec:voting}

<p>当多个 Agent 产生相互冲突的输出时，投票机制将它们的响应聚合为一个决策。常见方案包括：</p>

<ul>
<li><p><strong>多数投票（Majority voting）</strong>：出现最多的答案获胜；适用于事实性问题</p></li>
<li><p><strong>加权投票（Weighted voting）</strong>：历史记录或置信度更高的 Agent 获得更大权重</p></li>
<li><p><strong>基于辩论的裁决</strong>：Agent 为各自立场辩论；由一个裁判 Agent 做出决定</p></li>
<li><p><strong>Delphi 方法</strong>：迭代轮次中，Agent 在看到他人的推理后修改自己的答案</p></li>
</ul>

<p>形式化地，给定 <span class="arithmatex">\(n\)</span> 个 Agent 产生输出 <span class="arithmatex">\(\{o_1, \ldots, o_n\}\)</span> 及其权重 <span class="arithmatex">\(\{w_1, \ldots, w_n\}\)</span>，加权共识为：</p>



<div class="arithmatex">\[o^* = \arg\max_{o} \sum_{i=1}^{n} w_i \cdot \mathbf{1}[o_i = o]
\tag{25.1}\]</div>

<p>对于连续输出（例如概率估计），适用加权平均：</p>



<div class="arithmatex">\[\hat{p} = \frac{\sum_{i=1}^{n} w_i \cdot p_i}{\sum_{i=1}^{n} w_i}
\tag{25.2}\]</div>

## 25.3.5　基于市场的协调 {#subsubsec:market-based}

<p>市场机制通过<strong>拍卖与竞标</strong>分配任务和资源。合同网协议（Contract Net Protocol） <a class="book-citation" id="cite-smith1980contract-1" href="../../references.md#ref-smith1980contract" aria-label="参考文献 334">[334]</a> 是最古老的多 Agent 协调机制之一，本质上是一种任务拍卖：</p>

<ol>
<li><p><em>Manager</em> 广播一条带有需求的任务公告</p></li>
<li><p><em>Contractor</em>（承包者）Agent 提交竞标（能力声明 + 成本估算）</p></li>
<li><p>Manager 将合同授予最佳竞标者</p></li>
<li><p>中标的承包者执行并报告结果</p></li>
</ol>

<p>在 LLM 系统中，竞标可以用自然语言（如“我能用 3 步以高置信度完成此任务”）或结构化格式表达。市场机制在资源受限、需要最小化 API 成本的场景下尤为有效。</p>

## 25.3.6　Stigmergy：通过环境进行的间接通信 {#subsubsec:stigmergy}

<p><strong>Stigmergy（共识主动性）</strong> <a class="book-citation" id="cite-grassé1959reconstruction-1" href="../../references.md#ref-grassé1959reconstruction" aria-label="参考文献 117">[117]</a> 用一种更简单的机制取代显式的 Agent 间消息传递：每个 Agent 在工作时以副作用方式修改共享环境，其他 Agent 对这些修改而非直接信号做出反应。经典例证是觅食的蚂蚁在返程路径上沉积信息素；后续蚂蚁强化成功路径，无需任何蚂蚁“开口”说话。</p>

<p>在 LLM 多智能体系统中，Stigmergy 表现为：</p>

<ul>
<li><p><strong>共享文档</strong>：Agent 写入共享文档；其他 Agent 读取并在其基础上扩展</p></li>
<li><p><strong>代码仓库</strong>：一个 Agent 提交代码；另一个 Agent 读取并扩展</p></li>
<li><p><strong>标注层</strong>：Agent 对共享产物进行标注（高亮错误、添加评论）</p></li>
<li><p><strong>任务队列</strong>：Agent 从共享队列中添加和消费任务</p></li>
</ul>

<p>Stigmergy 实现了无显式通信开销的协调——Agent 只需观察共享环境的状态并据此行动。</p>
