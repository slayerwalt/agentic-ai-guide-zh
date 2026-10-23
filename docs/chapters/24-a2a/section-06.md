# 24.6　协调协议 {#sec:a2a:coordination}

<p>除了点对点通信，多 Agent 系统还能从更高层次的<strong>协调协议</strong>中受益——这些是支持集体决策与问题求解的结构化交互模式。</p>

## 24.6.1　合同网协议（Contract Net Protocol） {#contract-net-protocol}

<p><strong>合同网协议（Contract Net Protocol, CNP）</strong> <a class="book-citation" id="cite-smith1980contract-1" href="../../references.md#ref-smith1980contract" aria-label="参考文献 334">[334]</a> 是一种经典的多 Agent 协调机制，可被改造用于基于 LLM 的系统：</p>

<ol>
<li><p><strong>宣告（Announcement）</strong>：管理者 Agent 向所有潜在承包者 Agent 广播任务公告，其中包含任务需求和评估标准。</p></li>
<li><p><strong>竞标（Bidding）</strong>：承包者 Agent 根据自身能力评估任务，并提交包含预计完成时间、置信度和资源需求的竞标。</p></li>
<li><p><strong>授标（Award）</strong>：管理者选择中标方（对于并行子任务则可选择多个中标方），并授予合同。</p></li>
<li><p><strong>执行与汇报（Execution and Reporting）</strong>：承包者执行任务并将结果汇报给管理者。</p></li>
</ol>

<div class="admonition example"><p class="admonition-title">合同网协议实现</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">dataclasses</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ContractNetManager</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""为任务分配实现合同网协议（Contract Net Protocol）。"""</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">allocate_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task</span><span class="p">:</span> <span class="n">Task</span><span class="p">,</span>
                            <span class="n">candidate_agents</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">AgentCard</span><span class="p">])</span> <span class="o">-&gt;</span> <span class="n">AgentCard</span><span class="p">:</span>
        <span class="c1"># 阶段 1：向所有候选 Agent 宣告任务</span>
        <span class="n">announcement</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"task-announcement"</span><span class="p">,</span>
            <span class="s2">"task"</span><span class="p">:</span> <span class="n">dataclasses</span><span class="o">.</span><span class="n">asdict</span><span class="p">(</span><span class="n">task</span><span class="p">),</span>
            <span class="s2">"deadline"</span><span class="p">:</span> <span class="p">(</span><span class="n">datetime</span><span class="o">.</span><span class="n">now</span><span class="p">(</span><span class="n">timezone</span><span class="o">.</span><span class="n">utc</span><span class="p">)</span> <span class="o">+</span> <span class="n">timedelta</span><span class="p">(</span><span class="n">seconds</span><span class="o">=</span><span class="mi">10</span><span class="p">))</span><span class="o">.</span><span class="n">isoformat</span><span class="p">(),</span>
            <span class="s2">"evaluationCriteria"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"confidence"</span><span class="p">,</span> <span class="s2">"estimatedTime"</span><span class="p">,</span> <span class="s2">"cost"</span><span class="p">]</span>
        <span class="p">}</span>

        <span class="c1"># 阶段 2：收集竞标</span>
        <span class="n">bids</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">_request_bid</span><span class="p">(</span><span class="n">agent</span><span class="p">,</span> <span class="n">announcement</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">agent</span> <span class="ow">in</span> <span class="n">candidate_agents</span>
        <span class="p">],</span> <span class="n">return_exceptions</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

        <span class="n">valid_bids</span> <span class="o">=</span> <span class="p">[(</span><span class="n">agent</span><span class="p">,</span> <span class="n">bid</span><span class="p">)</span> <span class="k">for</span> <span class="n">agent</span><span class="p">,</span> <span class="n">bid</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">candidate_agents</span><span class="p">,</span> <span class="n">bids</span><span class="p">)</span>
                      <span class="k">if</span> <span class="ow">not</span> <span class="nb">isinstance</span><span class="p">(</span><span class="n">bid</span><span class="p">,</span> <span class="ne">Exception</span><span class="p">)</span> <span class="ow">and</span> <span class="n">bid</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">]</span>

        <span class="k">if</span> <span class="ow">not</span> <span class="n">valid_bids</span><span class="p">:</span>
            <span class="k">raise</span> <span class="ne">RuntimeError</span><span class="p">(</span><span class="sa">f</span><span class="s2">"No agents bid on task </span><span class="si">{</span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

        <span class="c1"># 阶段 3：授予最佳竞标方（置信度最高、用时最少）</span>
        <span class="k">def</span><span class="w"> </span><span class="nf">score_bid</span><span class="p">(</span><span class="n">agent_bid</span><span class="p">):</span>
            <span class="n">_</span><span class="p">,</span> <span class="n">bid</span> <span class="o">=</span> <span class="n">agent_bid</span>
            <span class="k">return</span> <span class="n">bid</span><span class="p">[</span><span class="s2">"confidence"</span><span class="p">]</span> <span class="o">-</span> <span class="mf">0.1</span> <span class="o">*</span> <span class="n">bid</span><span class="p">[</span><span class="s2">"estimatedSeconds"</span><span class="p">]</span>

        <span class="n">winner_agent</span><span class="p">,</span> <span class="n">winning_bid</span> <span class="o">=</span> <span class="nb">max</span><span class="p">(</span><span class="n">valid_bids</span><span class="p">,</span> <span class="n">key</span><span class="o">=</span><span class="n">score_bid</span><span class="p">)</span>

        <span class="c1"># 通知中标方与落标方</span>
        <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_award_contract</span><span class="p">(</span><span class="n">winner_agent</span><span class="p">,</span> <span class="n">task</span><span class="p">)</span>
        <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">_reject_bid</span><span class="p">(</span><span class="n">agent</span><span class="p">,</span> <span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">agent</span><span class="p">,</span> <span class="n">_</span> <span class="ow">in</span> <span class="n">valid_bids</span> <span class="k">if</span> <span class="n">agent</span> <span class="o">!=</span> <span class="n">winner_agent</span>
        <span class="p">])</span>

        <span class="k">return</span> <span class="n">winner_agent</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_request_bid</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">agent</span><span class="p">:</span> <span class="n">AgentCard</span><span class="p">,</span>
                           <span class="n">announcement</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span> <span class="o">|</span> <span class="kc">None</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""请求某个 Agent 对任务进行竞标。"""</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">client</span><span class="o">.</span><span class="n">send_task</span><span class="p">(</span>
                <span class="n">agent_url</span><span class="o">=</span><span class="n">agent</span><span class="o">.</span><span class="n">url</span><span class="p">,</span>
                <span class="n">message</span><span class="o">=</span><span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>
                         <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"data"</span><span class="p">,</span> <span class="s2">"data"</span><span class="p">:</span> <span class="n">announcement</span><span class="p">}]}</span>
            <span class="p">)</span>
            <span class="k">return</span> <span class="n">result</span><span class="o">.</span><span class="n">artifacts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">parts</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s2">"data"</span><span class="p">]</span>
        <span class="k">except</span> <span class="ne">Exception</span><span class="p">:</span>
            <span class="k">return</span> <span class="kc">None</span>
</code></pre></div>
</div>

## 24.6.2　黑板系统（Blackboard Systems） {#blackboard-systems}

<p><strong>黑板系统</strong> <a class="book-citation" id="cite-hayes1985blackboard-1" href="../../references.md#ref-hayes1985blackboard" aria-label="参考文献 135">[135]</a>提供一个共享工作区（即“黑板”），Agent 在上面发布部分解、观察和假设。其他 Agent 监视黑板，在能够提供价值时进行贡献——这是一种<em>机会主义</em>的问题求解方式。</p>

<p>黑板系统非常适合那些求解路径事先未知、不同 Agent 在不同阶段做出贡献的问题——例如科学假设生成、复杂调试或多源情报分析。</p>

## 24.6.3　共识协议（Consensus Protocols） {#consensus-protocols}

<p>当多个 Agent 必须就一项决策达成一致（例如执行哪个计划、结果是否正确），<strong>共识协议</strong>提供了结构化的投票机制：</p>

<p>简单多数投票（Simple Majority Voting）</p>

<p>每个 Agent 投票；获得 <span class="arithmatex">\(&gt; 50\%\)</span> 选票的选项胜出。速度快，但当 Agent 共享同一基础模型时，容易受到相关性错误的影响。</p>

<p>加权投票（Weighted Voting）</p>

<p>按 Agent 的置信度或历史准确率对选票加权。更稳健，但需要经过良好校准的置信度估计。</p>

<p>基于法定人数（Quorum-Based）</p>

<p>一项决策需要 <span class="arithmatex">\(n\)</span> 个 Agent 中至少 <span class="arithmatex">\(k\)</span> 个达成一致。提供容错能力：最多可有 <span class="arithmatex">\(n-k\)</span> 个 Agent 失败或不同意而不会阻塞流程。</p>

<p>德尔菲方法（Delphi Method）</p>

<p>Agent 先投票，看到匿名化的结果后修改投票，并反复迭代直至收敛。可减少锚定偏差，鼓励真正的深入审议。</p>

<div class="highlight"><pre><span></span><code><span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">quorum_vote</span><span class="p">(</span><span class="n">agents</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">AgentCard</span><span class="p">],</span> <span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                      <span class="n">options</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">],</span> <span class="n">quorum</span><span class="p">:</span> <span class="nb">int</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""在 Agent 之间进行法定人数投票。返回获胜选项或 None。"""</span>
    <span class="n">votes</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span>
        <span class="n">ask_agent_to_vote</span><span class="p">(</span><span class="n">agent</span><span class="p">,</span> <span class="n">question</span><span class="p">,</span> <span class="n">options</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">agent</span> <span class="ow">in</span> <span class="n">agents</span>
    <span class="p">])</span>

    <span class="n">counts</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">int</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">vote</span> <span class="ow">in</span> <span class="n">votes</span><span class="p">:</span>
        <span class="k">if</span> <span class="n">vote</span> <span class="ow">in</span> <span class="n">options</span><span class="p">:</span>
            <span class="n">counts</span><span class="p">[</span><span class="n">vote</span><span class="p">]</span> <span class="o">=</span> <span class="n">counts</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="n">vote</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span> <span class="o">+</span> <span class="mi">1</span>

    <span class="c1"># 返回第一个达到法定人数的选项</span>
    <span class="k">for</span> <span class="n">option</span><span class="p">,</span> <span class="n">count</span> <span class="ow">in</span> <span class="nb">sorted</span><span class="p">(</span><span class="n">counts</span><span class="o">.</span><span class="n">items</span><span class="p">(),</span> <span class="n">key</span><span class="o">=</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="o">-</span><span class="n">x</span><span class="p">[</span><span class="mi">1</span><span class="p">]):</span>
        <span class="k">if</span> <span class="n">count</span> <span class="o">&gt;=</span> <span class="n">quorum</span><span class="p">:</span>
            <span class="k">return</span> <span class="n">option</span>
    <span class="k">return</span> <span class="kc">None</span>  <span class="c1"># 未达到法定人数</span>
</code></pre></div>

## 24.6.4　领导者选举（Leader Election） {#leader-election}

<p>在动态的多 Agent 系统中，可能需要在运行时选举一位<strong>领导者</strong>（编排器）——例如原编排器失败时，或当 Agent 在没有预先指派协调者的情况下自组织时。经典的分布式系统算法（Bully、Ring）可被改造用于 Agent 网络：Agent 间交换能力分数或优先级令牌，以选举出当前可用 Agent 中能力最强的作为领导者。</p>
