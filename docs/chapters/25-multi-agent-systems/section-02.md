# 25.2　多智能体架构 {#subsec:mas-architectures}

<p>多智能体系统的拓扑结构——Agent 之间如何连接、权威如何在它们之间流动——是最具决定性的架构选择。目前已经形成了四种典型模式，各有不同的权衡。</p>

## 25.2.1　集中式（Supervisor/Manager）架构 {#subsubsec:centralized}

<p>在集中式架构中，由单一的<em>Orchestrator</em> Agent（也称为 supervisor、manager 或 planner）持有全局状态，分解任务，将子任务委派给工作 Agent，并汇总它们的结果。其拓扑是<strong>中心辐射式（hub-and-spoke）</strong>：所有通信都通过中心节点流动。</p>

<figure id="fig:centralized-arch"><a href="../../assets/figures/fig_069_centralized-arch.png" target="_blank" rel="noopener" aria-label="查看图 25.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_069_centralized-arch.png" alt="集中式（Supervisor）架构。Manager 将任务委派给专门化的 Worker，并汇总它们的输出。所有通信均通过中央枢纽流动。" loading="lazy"></a><figcaption><span class="figure-number">图 25.1　</span>集中式（Supervisor）架构。Manager 将任务委派给专门化的 Worker，并汇总它们的输出。所有通信均通过中央枢纽流动。</figcaption></figure>

<p>Manager 的职责包括：</p>

<ul>
<li><p><strong>任务路由</strong>：决定哪个 Worker 最适合每个子任务</p></li>
<li><p><strong>上下文管理</strong>：为每个 Worker 提供全局上下文中相关的子集</p></li>
<li><p><strong>结果聚合</strong>：将 Worker 输出综合为一个连贯整体</p></li>
<li><p><strong>错误处理</strong>：检测 Worker 故障并重新路由或重试</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">LangGraph 中的 Supervisor 模式</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.graph</span><span class="w"> </span><span class="kn">import</span> <span class="n">StateGraph</span><span class="p">,</span> <span class="n">START</span><span class="p">,</span> <span class="n">END</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">TypedDict</span><span class="p">,</span> <span class="n">Literal</span>

<span class="k">class</span><span class="w"> </span><span class="nc">TeamState</span><span class="p">(</span><span class="n">TypedDict</span><span class="p">):</span>
    <span class="n">task</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">plan</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">code</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">tests</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">review</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">next_agent</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">final_output</span><span class="p">:</span> <span class="nb">str</span>

<span class="k">def</span><span class="w"> </span><span class="nf">supervisor_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">TeamState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">TeamState</span><span class="p">:</span>
    <span class="c1"># 中央 orchestrator：决定下一步应调用哪个 agent。</span>
<span class="w">    </span><span class="sd">"""Central orchestrator: decides which agent to invoke next."""</span>
    <span class="n">messages</span> <span class="o">=</span> <span class="p">[</span>
        <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"system"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">SUPERVISOR_PROMPT</span><span class="p">},</span>
        <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>   <span class="s2">"content"</span><span class="p">:</span> <span class="sa">f</span><span class="s2">"Task: </span><span class="si">{</span><span class="n">state</span><span class="p">[</span><span class="s1">'task'</span><span class="p">]</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
                                      <span class="sa">f</span><span class="s2">"Plan: </span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s1">'plan'</span><span class="p">,</span><span class="s1">''</span><span class="p">)</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
                                      <span class="sa">f</span><span class="s2">"Code: </span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s1">'code'</span><span class="p">,</span><span class="s1">''</span><span class="p">)</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
                                      <span class="sa">f</span><span class="s2">"Tests: </span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s1">'tests'</span><span class="p">,</span><span class="s1">''</span><span class="p">)</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
                                      <span class="s2">"Which agent should act next? "</span>
                                      <span class="s2">"Options: planner, coder, tester, reviewer, FINISH"</span><span class="p">}</span>
    <span class="p">]</span>
    <span class="n">response</span> <span class="o">=</span> <span class="n">llm</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span><span class="n">messages</span><span class="p">)</span>
    <span class="k">return</span> <span class="p">{</span><span class="o">**</span><span class="n">state</span><span class="p">,</span> <span class="s2">"next_agent"</span><span class="p">:</span> <span class="n">response</span><span class="o">.</span><span class="n">content</span><span class="o">.</span><span class="n">strip</span><span class="p">()}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">route</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">TeamState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Literal</span><span class="p">[</span><span class="s2">"planner"</span><span class="p">,</span><span class="s2">"coder"</span><span class="p">,</span><span class="s2">"tester"</span><span class="p">,</span><span class="s2">"reviewer"</span><span class="p">,</span><span class="s2">"__end__"</span><span class="p">]:</span>
    <span class="k">return</span> <span class="n">state</span><span class="p">[</span><span class="s2">"next_agent"</span><span class="p">]</span> <span class="k">if</span> <span class="n">state</span><span class="p">[</span><span class="s2">"next_agent"</span><span class="p">]</span> <span class="o">!=</span> <span class="s2">"FINISH"</span> <span class="k">else</span> <span class="n">END</span>

<span class="n">builder</span> <span class="o">=</span> <span class="n">StateGraph</span><span class="p">(</span><span class="n">TeamState</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"supervisor"</span><span class="p">,</span> <span class="n">supervisor_node</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"planner"</span><span class="p">,</span>    <span class="n">planner_node</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"coder"</span><span class="p">,</span>      <span class="n">coder_node</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"tester"</span><span class="p">,</span>     <span class="n">tester_node</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"reviewer"</span><span class="p">,</span>   <span class="n">reviewer_node</span><span class="p">)</span>

<span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="n">START</span><span class="p">,</span> <span class="s2">"supervisor"</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_conditional_edges</span><span class="p">(</span><span class="s2">"supervisor"</span><span class="p">,</span> <span class="n">route</span><span class="p">)</span>
<span class="k">for</span> <span class="n">agent</span> <span class="ow">in</span> <span class="p">[</span><span class="s2">"planner"</span><span class="p">,</span> <span class="s2">"coder"</span><span class="p">,</span> <span class="s2">"tester"</span><span class="p">,</span> <span class="s2">"reviewer"</span><span class="p">]:</span>
    <span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="n">agent</span><span class="p">,</span> <span class="s2">"supervisor"</span><span class="p">)</span>   <span class="c1"># 始终回到 supervisor</span>

<span class="n">graph</span> <span class="o">=</span> <span class="n">builder</span><span class="o">.</span><span class="n">compile</span><span class="p">()</span>
</code></pre></div>
</div>

<div class="admonition warning"><p class="admonition-title">集中式架构的权衡</p>
<p><strong>优点：</strong>控制流简单；责任清晰；易于调试（所有决策集中在一处）；实现直接。<br><strong>缺点：</strong>单点故障——如果 Manager 产生幻觉或混乱，整个系统就会失败；在高负载下 Manager 会成为瓶颈；Manager 的上下文窗口必须容纳全局状态，限制了可扩展性。</p>
</div>

## 25.2.2　去中心化（Peer-to-Peer）架构 {#subsubsec:decentralized}

<p>在去中心化架构中，Agent 之间直接交互，没有中央协调者。其拓扑是<strong>网状（mesh）</strong>：任意 Agent 可以与任意其他 Agent 通信。协调通过局部交互涌现，而非通过全局规划。</p>

<figure id="fig:decentralized-arch"><a href="../../assets/figures/fig_070_decentralized-arch.png" target="_blank" rel="noopener" aria-label="查看图 25.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_070_decentralized-arch.png" alt="去中心化（peer-to-peer）架构。Agent 直接通信；协调从局部交互中涌现。" loading="lazy"></a><figcaption><span class="figure-number">图 25.2　</span>去中心化（peer-to-peer）架构。Agent 直接通信；协调从局部交互中涌现。</figcaption></figure>

<p>对等系统中的涌现式协调通过以下机制产生：</p>

<ul>
<li><p><strong>协商（Negotiation）</strong>：Agent 为任务或资源进行竞标</p></li>
<li><p><strong>Stigmergy（共识主动性）</strong>：Agent 修改共享状态，其他 Agent 观察该状态（见 <a href="section-03.md#subsubsec:stigmergy">25.3.6</a> 节）</p></li>
<li><p><strong>流言协议（Gossip protocols）</strong>：Agent 在网络中传播信息</p></li>
<li><p><strong>局部共识（Local consensus）</strong>：小组 Agent 在无全局协调的情况下达成一致</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">去中心化架构的权衡</p>
<p><strong>优点：</strong>对个别 Agent 故障具有韧性；随着 Agent 数量增加可自然扩展；无瓶颈。<br><strong>缺点：</strong>难以调试——涌现行为难以追踪；当 Agent 对状态的视图不一致时可能产生冲突；朴素消息传递下协调开销以 <span class="arithmatex">\(O(n^2)\)</span> 增长；难以保证全局一致性。</p>
</div>

## 25.2.3　层级式（Hierarchical）架构 {#subsubsec:hierarchical}

<p>层级式架构将集中式模式推广为具有多层管理的<strong>树形结构</strong>。顶层 Orchestrator 委派给特定领域的子 Manager，后者再委派给专门化 Worker。这映射了大型企业的组织结构。</p>

<figure id="fig:hierarchical-arch"><a href="../../assets/figures/fig_071_hierarchical-arch.png" target="_blank" rel="noopener" aria-label="查看图 25.3 原图" title="点击查看原图"><img src="../../assets/figures/fig_071_hierarchical-arch.png" alt="层级式架构。顶层 Orchestrator 委派给领域子 Manager，后者再委派给专门化 Worker。虚线箭头表示升级路径。" loading="lazy"></a><figcaption><span class="figure-number">图 25.3　</span>层级式架构。顶层 Orchestrator 委派给领域子 Manager，后者再委派给专门化 Worker。虚线箭头表示升级路径。</figcaption></figure>

<p>层级式系统的关键特征：</p>

<ul>
<li><p><strong>委派链（Delegation chains）</strong>：权威与上下文沿树向下流动；结果向上流动</p></li>
<li><p><strong>升级路径（Escalation paths）</strong>：Worker 可以将无法解决的问题升级给其 Manager</p></li>
<li><p><strong>领域隔离（Domain isolation）</strong>：子 Manager 维护特定领域的上下文，减轻顶层 Orchestrator 的认知负担</p></li>
<li><p><strong>作用域限制（Scope limitation）</strong>：每个 Agent 只需了解其直接上级与下级</p></li>
</ul>

<p>企业的类比十分贴切：CEO（顶层 Orchestrator）制定战略；VP（子 Manager）将战略转化为领域计划；一线贡献者（Worker）执行。层级结构在实现规模化的同时保留了责任归属。</p>

## 25.2.4　蜂群（Swarm）架构 {#subsubsec:swarm}

<p>受生物系统（蚁群、鸟群）启发的蜂群架构，由许多遵循简单局部规则的<strong>松耦合 Agent</strong>组成，能在没有任何中央协调者或全局状态的情况下产生复杂的全局行为。</p>

<p>OpenAI 的 <strong>Swarm</strong> 框架 <a class="book-citation" id="cite-openai2024swarm-1" href="../../references.md#ref-openai2024swarm" aria-label="参考文献 276">[276]</a>（现已被 OpenAI Agents SDK 取代，但其概念原语依然有影响力）通过两个原语将其落地：</p>

<ul>
<li><p><strong>Routines（例程）</strong>：Agent 为完成子任务所遵循的指令序列</p></li>
<li><p><strong>Handoffs（移交）</strong>：一个 Agent 将控制权（以及相关上下文）转交给另一个 Agent</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">OpenAI Swarm：Routines 与 Handoffs</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">swarm</span><span class="w"> </span><span class="kn">import</span> <span class="n">Swarm</span><span class="p">,</span> <span class="n">Agent</span>

<span class="n">client</span> <span class="o">=</span> <span class="n">Swarm</span><span class="p">()</span>

<span class="k">def</span><span class="w"> </span><span class="nf">transfer_to_billing</span><span class="p">():</span>
    <span class="c1"># Handoff：将控制权移交给计费专员。</span>
<span class="w">    </span><span class="sd">"""Handoff: transfer control to the billing specialist."""</span>
    <span class="k">return</span> <span class="n">billing_agent</span>

<span class="k">def</span><span class="w"> </span><span class="nf">transfer_to_technical</span><span class="p">():</span>
    <span class="c1"># Handoff：将控制权移交给技术支持 agent。</span>
<span class="w">    </span><span class="sd">"""Handoff: transfer control to the technical support agent."""</span>
    <span class="k">return</span> <span class="n">technical_agent</span>

<span class="n">triage_agent</span> <span class="o">=</span> <span class="n">Agent</span><span class="p">(</span>
    <span class="n">name</span><span class="o">=</span><span class="s2">"Triage Agent"</span><span class="p">,</span>
    <span class="n">instructions</span><span class="o">=</span><span class="s2">"""You are a customer service triage agent.</span>
<span class="s2">    Determine the nature of the customer's issue:</span>
<span class="s2">    - For billing questions, transfer to billing.</span>
<span class="s2">    - For technical issues, transfer to technical support.</span>
<span class="s2">    - For general questions, answer directly."""</span><span class="p">,</span>
    <span class="n">functions</span><span class="o">=</span><span class="p">[</span><span class="n">transfer_to_billing</span><span class="p">,</span> <span class="n">transfer_to_technical</span><span class="p">],</span>
<span class="p">)</span>

<span class="n">billing_agent</span> <span class="o">=</span> <span class="n">Agent</span><span class="p">(</span>
    <span class="n">name</span><span class="o">=</span><span class="s2">"Billing Specialist"</span><span class="p">,</span>
    <span class="n">instructions</span><span class="o">=</span><span class="s2">"You handle billing inquiries. "</span>
                 <span class="s2">"Access account data and resolve payment issues."</span><span class="p">,</span>
    <span class="n">functions</span><span class="o">=</span><span class="p">[</span><span class="n">lookup_account</span><span class="p">,</span> <span class="n">process_refund</span><span class="p">],</span>
<span class="p">)</span>

<span class="n">technical_agent</span> <span class="o">=</span> <span class="n">Agent</span><span class="p">(</span>
    <span class="n">name</span><span class="o">=</span><span class="s2">"Technical Support"</span><span class="p">,</span>
    <span class="n">instructions</span><span class="o">=</span><span class="s2">"You resolve technical issues. "</span>
                 <span class="s2">"Diagnose problems and provide step-by-step solutions."</span><span class="p">,</span>
    <span class="n">functions</span><span class="o">=</span><span class="p">[</span><span class="n">run_diagnostics</span><span class="p">,</span> <span class="n">escalate_to_engineering</span><span class="p">],</span>
<span class="p">)</span>

<span class="c1"># 无全局状态——每个 agent 仅在其局部上下文中运行</span>
<span class="n">response</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
    <span class="n">agent</span><span class="o">=</span><span class="n">triage_agent</span><span class="p">,</span>
    <span class="n">messages</span><span class="o">=</span><span class="p">[{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"My invoice is wrong"</span><span class="p">}]</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">Swarm 的性质</p>
<ul>
<li><p><strong>无全局状态</strong>：每个 Agent 只维护自己的局部上下文窗口</p></li>
<li><p><strong>局部决策</strong>：路由决策由当前 Agent 做出，而非由中央规划者</p></li>
<li><p><strong>通过集体行为完成任务</strong>：复杂任务通过一连串 handoff 完成，每个 Agent 贡献其专长</p></li>
<li><p><strong>轻量</strong>：无编排开销；Agent 在 handoff 之间是无状态的</p></li>
</ul>
</div>
