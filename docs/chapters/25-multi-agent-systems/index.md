# 第 25 章 · 多智能体系统（Multi-Agent Systems） {#sec:multi-agent-systems}

## 25.1　动机：为什么需要多个 Agent？ {#subsec:mas-motivation}

<p>人工智能的历史在很多方面就是一部规模演进史。早期 AI 系统是单体式的：单一程序、单一知识库、单一推理引擎。随着问题日益复杂，研究者发现，没有任何单一 Agent——无论能力多强——能够高效地处理一个丰富、开放式任务的方方面面。这一洞见在分布式 AI 与多智能体系统（Multi-Agent System, MAS）研究中早有定论 <a class="book-citation" id="cite-weiss1999multiagent-1" href="../../references.md#ref-weiss1999multiagent" aria-label="参考文献 375">[375]</a><a class="book-citation" id="cite-wooldridge2009introduction-1" href="../../references.md#ref-wooldridge2009introduction" aria-label="参考文献 381">[381]</a>，而在大语言模型时代又被赋予了新的紧迫性。</p>

<div class="admonition tip"><p class="admonition-title">核心直觉</p>
<p>单个 LLM 无论多大，本质上都是一个通才。一支由专门化 LLM 组成的团队，每个 LLM 聚焦于一个狭窄的子问题并交流结果，在复杂的多维度任务上能够胜过通才——正如一支人类专家团队在复杂工程项目上胜过单个通才。</p>
</div>

<p>从单体 Agent 转向<em>Agent 社会</em>（agent societies）的根本动机有四个：</p>

##### 专门化（Specialization）。 {#specialization.}

<p>不同的子任务受益于不同的能力、Prompt 策略，乃至不同的基础模型。代码生成 Agent 可以在编程语料上微调；事实核查 Agent 可以借助检索工具锚定事实；创意写作 Agent 可以通过 Prompt 激发风格多样性。强迫单个 Agent 在所有这些方面同时出色，既低效，往往也不可能。</p>

##### 并行化（Parallelism）。 {#parallelism.}

<p>许多现实世界的任务可以分解为可并发执行的独立子任务。一个需要文献综述、数据分析和报告撰写的研究流水线可以让这三者并行进行，从而显著降低墙钟时间。串行的单 Agent 处理是一个瓶颈，而多 Agent 并行化正好消除了它。</p>

##### 鲁棒性（Robustness）。 {#robustness.}

<p>单个 Agent 是单点故障。如果它产生幻觉、陷入循环或给出细微错误的答案，将无人核验。多 Agent 系统引入了冗余：第二个 Agent 可以验证、批判，或独立地重新推导结果。在输出被信任之前，对抗性 Agent 可以主动探测其弱点。</p>

##### 涌现能力（Emergent Capabilities）。 {#emergent-capabilities.}

<p>也许最引人入胜的是，Agent 群体可以展现出任何单一 Agent 都不具备的能力。通过辩论、协商和迭代精化，多 Agent 系统能够得到超越任一单独 Agent 所能产出的解决方案——这是一种社会性生物涌现智能的计算类比。</p>

<div class="admonition note"><p class="admonition-title">历史脉络</p>
<p>多智能体系统研究可以追溯到 1980 年代，其奠基性工作包括分布式问题求解 <a class="book-citation" id="cite-durfee1989distributed-1" href="../../references.md#ref-durfee1989distributed" aria-label="参考文献 80">[80]</a>、合同网协议（Contract Net Protocol） <a class="book-citation" id="cite-smith1980contract-1" href="../../references.md#ref-smith1980contract" aria-label="参考文献 334">[334]</a> 以及 FIPA Agent 通信标准 <a class="book-citation" id="cite-fipa2002acl-1" href="../../references.md#ref-fipa2002acl" aria-label="参考文献 93">[93]</a>。向基于 LLM 的 Agent 转变让这些经典思想在新的载体上重获新生：以前是手工编码、做符号推理的 Agent，现在则是其“认知”从学习到的神经表示中涌现出来的 Agent。核心的架构模式——层级、市场、黑板、消息传递——依然惊人地相关。</p>
</div>

<p>从单体 Agent 到 Agent 社会的转变映射了复杂系统中一个更广泛的模式：随着问题空间的增长，分布式、模块化架构始终优于集中式、单体式架构。问题不再是<em>是否</em>使用多个 Agent，而是<em>如何</em>组织它们。</p>

## 25.2　多智能体架构 {#subsec:mas-architectures}

<p>多智能体系统的拓扑结构——Agent 之间如何连接、权威如何在它们之间流动——是最具决定性的架构选择。目前已经形成了四种典型模式，各有不同的权衡。</p>

### 25.2.1　集中式（Supervisor/Manager）架构 {#subsubsec:centralized}

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

### 25.2.2　去中心化（Peer-to-Peer）架构 {#subsubsec:decentralized}

<p>在去中心化架构中，Agent 之间直接交互，没有中央协调者。其拓扑是<strong>网状（mesh）</strong>：任意 Agent 可以与任意其他 Agent 通信。协调通过局部交互涌现，而非通过全局规划。</p>

<figure id="fig:decentralized-arch"><a href="../../assets/figures/fig_070_decentralized-arch.png" target="_blank" rel="noopener" aria-label="查看图 25.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_070_decentralized-arch.png" alt="去中心化（peer-to-peer）架构。Agent 直接通信；协调从局部交互中涌现。" loading="lazy"></a><figcaption><span class="figure-number">图 25.2　</span>去中心化（peer-to-peer）架构。Agent 直接通信；协调从局部交互中涌现。</figcaption></figure>

<p>对等系统中的涌现式协调通过以下机制产生：</p>

<ul>
<li><p><strong>协商（Negotiation）</strong>：Agent 为任务或资源进行竞标</p></li>
<li><p><strong>Stigmergy（共识主动性）</strong>：Agent 修改共享状态，其他 Agent 观察该状态（见 <a href="#subsubsec:stigmergy">25.3.6</a> 节）</p></li>
<li><p><strong>流言协议（Gossip protocols）</strong>：Agent 在网络中传播信息</p></li>
<li><p><strong>局部共识（Local consensus）</strong>：小组 Agent 在无全局协调的情况下达成一致</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">去中心化架构的权衡</p>
<p><strong>优点：</strong>对个别 Agent 故障具有韧性；随着 Agent 数量增加可自然扩展；无瓶颈。<br><strong>缺点：</strong>难以调试——涌现行为难以追踪；当 Agent 对状态的视图不一致时可能产生冲突；朴素消息传递下协调开销以 <span class="arithmatex">\(O(n^2)\)</span> 增长；难以保证全局一致性。</p>
</div>

### 25.2.3　层级式（Hierarchical）架构 {#subsubsec:hierarchical}

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

### 25.2.4　蜂群（Swarm）架构 {#subsubsec:swarm}

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

## 25.3　协调机制 {#subsec:coordination-mechanisms}

<p>Agent 如何协调——如何共享信息、分工以及解决冲突——与拓扑同等重要。六种典型的协调机制适用于基于 LLM 的多智能体系统。</p>

### 25.3.1　共享状态（全局黑板） {#subsubsec:shared-state}

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

### 25.3.2　消息传递（Message Passing） {#subsubsec:message-passing}

<p>消息传递是 LLM Agent 最自然的协调机制：Agent 通过相互发送结构化文本消息来通信。关键的设计决策包括：</p>

<ul>
<li><p><strong>消息格式</strong>：结构化（JSON schema）、自然语言或混合形式</p></li>
<li><p><strong>路由</strong>：直接（Agent 到 Agent）、广播或基于主题的发布/订阅</p></li>
<li><p><strong>会话线程</strong>：在多轮对话（Multi-Turn）交流中维持上下文</p></li>
<li><p><strong>确认（Acknowledgment）</strong>：发送方是否需要接收/处理的确认</p></li>
</ul>

### 25.3.3　规划与分解 {#subsubsec:planning-decomp}

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

### 25.3.4　投票与共识 {#subsubsec:voting}

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

### 25.3.5　基于市场的协调 {#subsubsec:market-based}

<p>市场机制通过<strong>拍卖与竞标</strong>分配任务和资源。合同网协议（Contract Net Protocol） <a class="book-citation" id="cite-smith1980contract-1--section-03" href="../../references.md#ref-smith1980contract" aria-label="参考文献 334">[334]</a> 是最古老的多 Agent 协调机制之一，本质上是一种任务拍卖：</p>

<ol>
<li><p><em>Manager</em> 广播一条带有需求的任务公告</p></li>
<li><p><em>Contractor</em>（承包者）Agent 提交竞标（能力声明 + 成本估算）</p></li>
<li><p>Manager 将合同授予最佳竞标者</p></li>
<li><p>中标的承包者执行并报告结果</p></li>
</ol>

<p>在 LLM 系统中，竞标可以用自然语言（如“我能用 3 步以高置信度完成此任务”）或结构化格式表达。市场机制在资源受限、需要最小化 API 成本的场景下尤为有效。</p>

### 25.3.6　Stigmergy：通过环境进行的间接通信 {#subsubsec:stigmergy}

<p><strong>Stigmergy（共识主动性）</strong> <a class="book-citation" id="cite-grassé1959reconstruction-1" href="../../references.md#ref-grassé1959reconstruction" aria-label="参考文献 117">[117]</a> 用一种更简单的机制取代显式的 Agent 间消息传递：每个 Agent 在工作时以副作用方式修改共享环境，其他 Agent 对这些修改而非直接信号做出反应。经典例证是觅食的蚂蚁在返程路径上沉积信息素；后续蚂蚁强化成功路径，无需任何蚂蚁“开口”说话。</p>

<p>在 LLM 多智能体系统中，Stigmergy 表现为：</p>

<ul>
<li><p><strong>共享文档</strong>：Agent 写入共享文档；其他 Agent 读取并在其基础上扩展</p></li>
<li><p><strong>代码仓库</strong>：一个 Agent 提交代码；另一个 Agent 读取并扩展</p></li>
<li><p><strong>标注层</strong>：Agent 对共享产物进行标注（高亮错误、添加评论）</p></li>
<li><p><strong>任务队列</strong>：Agent 从共享队列中添加和消费任务</p></li>
</ul>

<p>Stigmergy 实现了无显式通信开销的协调——Agent 只需观察共享环境的状态并据此行动。</p>

## 25.4　通信协议 {#subsec:communication-protocols}

<p>有效的多智能体系统需要定义良好的通信协议：约定的 Agent 间消息格式、语义和模式。（关于标准化的 Agent 间协议，参见第 <a href="../24-a2a/index.md#sec:a2a">24</a> 章。）</p>

### 25.4.1　结构化消息格式 {#structured-message-formats}

<p>LLM Agent 之间的消息应该是结构化的，以支持可靠解析与路由。一个最小的消息 schema：</p>

<p class="code-caption"><span class="code-number">代码 25.3　</span>Agent 消息 schema</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">pydantic</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseModel</span><span class="p">,</span> <span class="n">Field</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Literal</span><span class="p">,</span> <span class="n">Optional</span><span class="p">,</span> <span class="n">Dict</span><span class="p">,</span> <span class="n">Any</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">datetime</span><span class="w"> </span><span class="kn">import</span> <span class="n">datetime</span><span class="p">,</span> <span class="n">timezone</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">uuid</span>

<span class="n">PerformativeType</span> <span class="o">=</span> <span class="n">Literal</span><span class="p">[</span>
    <span class="s2">"inform"</span><span class="p">,</span>    <span class="c1"># 共享信息</span>
    <span class="s2">"request"</span><span class="p">,</span>   <span class="c1"># 请求执行某动作</span>
    <span class="s2">"propose"</span><span class="p">,</span>   <span class="c1"># 提议某种行动方案</span>
    <span class="s2">"accept"</span><span class="p">,</span>    <span class="c1"># 接受提议</span>
    <span class="s2">"reject"</span><span class="p">,</span>    <span class="c1"># 拒绝提议</span>
    <span class="s2">"query"</span><span class="p">,</span>     <span class="c1"># 提出问题</span>
    <span class="s2">"confirm"</span><span class="p">,</span>   <span class="c1"># 确认收到/完成</span>
    <span class="s2">"failure"</span><span class="p">,</span>   <span class="c1"># 报告失败</span>
<span class="p">]</span>

<span class="k">class</span><span class="w"> </span><span class="nc">AgentMessage</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">message_id</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="n">Field</span><span class="p">(</span><span class="n">default_factory</span><span class="o">=</span><span class="k">lambda</span><span class="p">:</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">()))</span>
    <span class="n">conversation_id</span><span class="p">:</span> <span class="nb">str</span>          <span class="c1"># 将相关消息分组</span>
    <span class="n">sender</span><span class="p">:</span> <span class="nb">str</span>                   <span class="c1"># Agent 标识</span>
    <span class="n">receiver</span><span class="p">:</span> <span class="nb">str</span>                 <span class="c1"># 目标 agent（或 "broadcast"）</span>
    <span class="n">performative</span><span class="p">:</span> <span class="n">PerformativeType</span>
    <span class="n">content</span><span class="p">:</span> <span class="nb">str</span>                  <span class="c1"># 自然语言内容</span>
    <span class="n">metadata</span><span class="p">:</span> <span class="n">Dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">Any</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span> <span class="c1"># 结构化载荷</span>
    <span class="n">reply_to</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>  <span class="c1"># 所回复的 message_id</span>
    <span class="n">timestamp</span><span class="p">:</span> <span class="n">datetime</span> <span class="o">=</span> <span class="n">Field</span><span class="p">(</span><span class="n">default_factory</span><span class="o">=</span><span class="k">lambda</span><span class="p">:</span> <span class="n">datetime</span><span class="o">.</span><span class="n">now</span><span class="p">(</span><span class="n">timezone</span><span class="o">.</span><span class="n">utc</span><span class="p">))</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">to_llm_prompt</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="c1"># 将消息渲染为接收方 agent 的 prompt 片段。</span>
<span class="w">        </span><span class="sd">"""Render message as a prompt fragment for the receiving agent."""</span>
        <span class="k">return</span> <span class="p">(</span>
            <span class="sa">f</span><span class="s2">"[MESSAGE from </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">sender</span><span class="si">}</span><span class="s2">]</span><span class="se">\n</span><span class="s2">"</span>
            <span class="sa">f</span><span class="s2">"Type: </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">performative</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
            <span class="sa">f</span><span class="s2">"Content: </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">content</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
            <span class="o">+</span> <span class="p">(</span><span class="sa">f</span><span class="s2">"Metadata: </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">metadata</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span> <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">metadata</span> <span class="k">else</span> <span class="s2">""</span><span class="p">)</span>
        <span class="p">)</span>
</code></pre></div>

### 25.4.2　Performative 类型（受 FIPA-ACL 启发） {#performative-types-fipa-acl-inspired}

<p>借鉴 FIPA Agent 通信语言 <a class="book-citation" id="cite-fipa2002acl-1--section-04" href="../../references.md#ref-fipa2002acl" aria-label="参考文献 93">[93]</a>，并针对 LLM Agent 进行现代化改造：</p>

<p class="table-caption" id="table-25-1"><span class="table-number">表 25.1　</span>受 FIPA-ACL 启发的 LLM Agent 消息 performative 类型。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>Performative</strong></th><th scope="col"><strong>语义</strong></th><th scope="col"><strong>示例用途</strong></th></tr></thead>
<tbody><tr><td><code>inform</code></td><td>发送方相信 <span class="arithmatex">\(\phi\)</span> 为真</td><td>共享研究发现</td></tr>
<tr><td><code>request</code></td><td>发送方希望接收方执行 <span class="arithmatex">\(\alpha\)</span></td><td>委派子任务</td></tr>
<tr><td><code>propose</code></td><td>发送方提议计划 <span class="arithmatex">\(\pi\)</span></td><td>建议某种方案</td></tr>
<tr><td><code>accept</code></td><td>接收方同意提议</td><td>确认任务分配</td></tr>
<tr><td><code>reject</code></td><td>接收方拒绝提议</td><td>拒绝不兼容的任务</td></tr>
<tr><td><code>query</code></td><td>发送方希望知道 <span class="arithmatex">\(\phi\)</span></td><td>请求澄清</td></tr>
<tr><td><code>confirm</code></td><td>发送方确认 <span class="arithmatex">\(\phi\)</span> 已发生</td><td>确认完成</td></tr>
<tr><td><code>failure</code></td><td>发送方未能完成 <span class="arithmatex">\(\alpha\)</span></td><td>报告错误</td></tr></tbody>
</table>
</div>

### 25.4.3　上下文共享策略 {#context-sharing-strategies}

<p>多 Agent 通信中的一个关键挑战是<strong>上下文管理</strong>：每个 Agent 需要多少历史？三种策略：</p>

<ul>
<li><p><strong>完整历史</strong>：将整个会话历史传递给每个 Agent。信息最丰富但代价高昂；上下文窗口会迅速被填满。</p></li>
<li><p><strong>摘要</strong>：由一个摘要 Agent 将先前的交流压缩为紧凑的摘要。高效但有损；重要细节可能被丢弃。</p></li>
<li><p><strong>相关摘录</strong>：通过语义搜索仅检索最相关的历史消息。在成本与信息量之间取得平衡；需要检索机制。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">上下文共享经验法则</p>
<p>对短对话（<span class="arithmatex">\(&lt;\)</span>10 轮）使用<strong>完整历史</strong>；对中等长度对话使用<strong>摘要</strong>；对长时运行的 Agent 会话使用<strong>检索增强的摘录</strong>。始终原文保留最近的 <span class="arithmatex">\(k\)</span> 条消息，以保持即时上下文。</p>
</div>

## 25.5　角色设计与专门化 {#subsec:role-design}

<p>Agent 角色的设计——其能力、人格化设定与职责——既是艺术也是科学。设计良好的角色能促成专门化；设计糟糕的角色则会带来混乱与冗余。</p>

### 25.5.1　定义 Agent 角色 {#defining-agent-roles}

<p>LLM 多智能体系统中常见的角色：</p>

<p class="table-caption" id="table-25-2"><span class="table-number">表 25.2　</span>LLM 多智能体系统中常见的 Agent 角色。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>角色</strong></th><th scope="col"><strong>主要能力</strong></th><th scope="col"><strong>典型工具</strong></th></tr></thead>
<tbody><tr><td>Researcher</td><td>信息收集与综合</td><td>网页搜索、RAG、数据库</td></tr>
<tr><td>Planner</td><td>任务分解与调度</td><td>无（仅推理）</td></tr>
<tr><td>Coder</td><td>代码生成与调试</td><td>Code interpreter、linter</td></tr>
<tr><td>Reviewer</td><td>质量评估与批判</td><td>无（仅推理）</td></tr>
<tr><td>Tester</td><td>测试生成与执行</td><td>测试运行器、覆盖率工具</td></tr>
<tr><td>Writer</td><td>文本生成与编辑</td><td>语法检查器、风格指南</td></tr>
<tr><td>Critic</td><td>对抗式评估</td><td>无（仅推理）</td></tr>
<tr><td>Orchestrator</td><td>协调与委派</td><td>所有 Agent 接口</td></tr></tbody>
</table>
</div>

### 25.5.2　基于能力 vs. 基于角色的分配 {#capability-based-vs.-role-based-assignment}

<p>任务分配的两种理念：</p>

<ul>
<li><p><strong>基于角色</strong>：根据预定义的角色标签分配任务。简单且可预测；当任务跨多个角色时可能不是最优。</p></li>
<li><p><strong>基于能力</strong>：根据对每个 Agent 能力相对于任务需求的动态评估来分配。更灵活；需要一个能力登记与匹配机制。</p></li>
</ul>

### 25.5.3　动态角色重分配 {#dynamic-role-reassignment}

<p>在长时运行的系统中，静态角色分配会变得次优。动态重分配允许 Agent 基于以下因素承担新角色：</p>

<ul>
<li><p>当前负载（负载均衡）</p></li>
<li><p>近期任务上展现出的表现</p></li>
<li><p>任务需求的变化</p></li>
<li><p>需要顶替的 Agent 故障</p></li>
</ul>

### 25.5.4　以人格化设计实现思维多样性 {#persona-design-for-diversity-of-thought}

<p>一种微妙但强大的技巧：给 Agent 赋予<strong>鲜明的人格设定（personas）</strong>，以鼓励多元视角。与其使用五个完全相同的“助手”Agent，不如设计：</p>

<ul>
<li><p>一位强调机会的<em>乐观主义者</em></p></li>
<li><p>一位质疑假设的<em>怀疑论者</em></p></li>
<li><p>一位聚焦实现的<em>务实派</em></p></li>
<li><p>一位着眼长远的<em>远见者</em></p></li>
<li><p>一位反向论辩的<em>唱反调者（devil’s advocate）</em></p></li>
</ul>

<p>这种思维多样性受“六顶思考帽”（Six Thinking Hats） <a class="book-citation" id="cite-debono1985six-1" href="../../references.md#ref-debono1985six" aria-label="参考文献 28">[28]</a> 等技巧启发，能减少群体思维，产出更稳健的集体推理。</p>

<div class="admonition warning"><p class="admonition-title">角色冲突的化解</p>
<p>当 Agent 的职责出现重叠时，冲突就会产生。可用显式的<strong>优先级规则</strong>来化解：定义每类任务由哪个角色优先处理。或者使用一个<strong>元 Agent（meta-agent）</strong>，其唯一职责是冲突仲裁。绝不要让角色冲突隐式存在——它们会以相互矛盾的输出或死循环的形式出现。</p>
</div>

## 25.6　面向 LLM 的多 Agent 模式 {#subsec:mas-patterns}

<p>除了架构拓扑之外，还有若干<strong>交互模式</strong>已被证明对基于 LLM 的多智能体系统特别有效。（这些模式与第 <a href="../20-agent-patterns/index.md#sec:agent-design-patterns">20</a> 章中的单 Agent 设计模式互为补充。）</p>

### 25.6.1　辩论模式（Debate Pattern） {#debate-pattern}

<p>多个 Agent 为不同立场辩论；一个裁判 Agent 评估论点并做出决定。已有研究表明辩论能提升事实准确性并减少幻觉 <a class="book-citation" id="cite-du2023improving-1" href="../../references.md#ref-du2023improving" aria-label="参考文献 78">[78]</a>。</p>

<p class="code-caption"><span class="code-number">代码 25.4　</span>辩论模式的实现</p>
<div class="highlight"><pre><span></span><code><span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">debate_round</span><span class="p">(</span><span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">agents</span><span class="p">:</span> <span class="nb">list</span><span class="p">,</span> <span class="n">judge</span><span class="p">:</span> <span class="n">Agent</span><span class="p">,</span>
                       <span class="n">rounds</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="c1"># 运行多 agent 辩论，并返回裁判的判决。</span>
<span class="w">    </span><span class="sd">"""Run a multi-agent debate and return the judge's verdict."""</span>
    <span class="n">positions</span> <span class="o">=</span> <span class="p">{</span><span class="n">a</span><span class="o">.</span><span class="n">name</span><span class="p">:</span> <span class="k">await</span> <span class="n">a</span><span class="o">.</span><span class="n">generate_position</span><span class="p">(</span><span class="n">question</span><span class="p">)</span>
                 <span class="k">for</span> <span class="n">a</span> <span class="ow">in</span> <span class="n">agents</span><span class="p">}</span>

    <span class="k">for</span> <span class="n">round_num</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">rounds</span><span class="p">):</span>
        <span class="c1"># 每个 agent 看到他人立场并可反驳</span>
        <span class="n">rebuttals</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="k">for</span> <span class="n">agent</span> <span class="ow">in</span> <span class="n">agents</span><span class="p">:</span>
            <span class="n">others</span> <span class="o">=</span> <span class="p">{</span><span class="n">k</span><span class="p">:</span> <span class="n">v</span> <span class="k">for</span> <span class="n">k</span><span class="p">,</span> <span class="n">v</span> <span class="ow">in</span> <span class="n">positions</span><span class="o">.</span><span class="n">items</span><span class="p">()</span>
                      <span class="k">if</span> <span class="n">k</span> <span class="o">!=</span> <span class="n">agent</span><span class="o">.</span><span class="n">name</span><span class="p">}</span>
            <span class="n">rebuttals</span><span class="p">[</span><span class="n">agent</span><span class="o">.</span><span class="n">name</span><span class="p">]</span> <span class="o">=</span> <span class="k">await</span> <span class="n">agent</span><span class="o">.</span><span class="n">rebut</span><span class="p">(</span>
                <span class="n">question</span><span class="p">,</span> <span class="n">positions</span><span class="p">[</span><span class="n">agent</span><span class="o">.</span><span class="n">name</span><span class="p">],</span> <span class="n">others</span>
            <span class="p">)</span>
        <span class="n">positions</span> <span class="o">=</span> <span class="n">rebuttals</span>

    <span class="c1"># 裁判评估所有最终立场</span>
    <span class="n">verdict</span> <span class="o">=</span> <span class="k">await</span> <span class="n">judge</span><span class="o">.</span><span class="n">evaluate</span><span class="p">(</span><span class="n">question</span><span class="p">,</span> <span class="n">positions</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">verdict</span>
</code></pre></div>

### 25.6.2　反思模式（Reflection Pattern） {#reflection-pattern}

<p>一个 Agent 生成输出；第二个 Agent 对其进行批判；第一个 Agent 根据批判进行修订。这实现了一个“生成—批判—修订”循环，迭代地提升质量。</p>

<p class="code-caption"><span class="code-number">代码 25.5　</span>反思模式</p>
<div class="highlight"><pre><span></span><code><span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">reflection_loop</span><span class="p">(</span><span class="n">task</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">generator</span><span class="p">:</span> <span class="n">Agent</span><span class="p">,</span>
                          <span class="n">critic</span><span class="p">:</span> <span class="n">Agent</span><span class="p">,</span> <span class="n">max_rounds</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">3</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="n">draft</span> <span class="o">=</span> <span class="k">await</span> <span class="n">generator</span><span class="o">.</span><span class="n">generate</span><span class="p">(</span><span class="n">task</span><span class="p">)</span>

    <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">max_rounds</span><span class="p">):</span>
        <span class="n">critique</span> <span class="o">=</span> <span class="k">await</span> <span class="n">critic</span><span class="o">.</span><span class="n">critique</span><span class="p">(</span><span class="n">task</span><span class="p">,</span> <span class="n">draft</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">critique</span><span class="o">.</span><span class="n">is_satisfactory</span><span class="p">:</span>
            <span class="k">break</span>
        <span class="n">draft</span> <span class="o">=</span> <span class="k">await</span> <span class="n">generator</span><span class="o">.</span><span class="n">revise</span><span class="p">(</span><span class="n">task</span><span class="p">,</span> <span class="n">draft</span><span class="p">,</span> <span class="n">critique</span><span class="o">.</span><span class="n">feedback</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">draft</span>
</code></pre></div>

### 25.6.3　分工模式（Division of Labor Pattern） {#division-of-labor-pattern}

<p>任务被分解为可并行执行的独立子任务。结果由一个综合 Agent 聚合。该模式可在“高度并行（embarrassingly parallel）”任务上最大化吞吐量。</p>

### 25.6.4　流水线模式（Pipeline Pattern） {#pipeline-pattern}

<p>Agent 构成一条顺序处理链：每个 Agent 转换前一个 Agent 的输出。类似于 Unix 管道。适用于具有明确顺序依赖的任务（例如：研究 <span class="arithmatex">\(\to\)</span> 大纲 <span class="arithmatex">\(\to\)</span> 草稿 <span class="arithmatex">\(\to\)</span> 编辑 <span class="arithmatex">\(\to\)</span> 排版）。</p>

### 25.6.5　集成模式（Ensemble Pattern） {#ensemble-pattern}

<p>多个 Agent 独立求解同一问题；由某种选择机制挑出最佳答案（best-of-<span class="arithmatex">\(N\)</span>），或聚合答案（mixture-of-experts 风格）。以算力为代价提升可靠性。</p>



<div class="arithmatex">\[o^* = \arg\max_{o \in \{o_1,\ldots,o_N\}} \text{score}(o, \text{task})
\tag{25.3}\]</div>

<p>其中 <span class="arithmatex">\(\text{score}\)</span> 可以是一个 Reward 模型、一个裁判 LLM 或一个验证器。</p>

### 25.6.6　师生模式（Teacher-Student Pattern） {#teacher-student-pattern}

<p>能力更强的 Agent（teacher）引导能力较弱的 Agent（student）完成任务，提供提示、纠正和解释。该模式在推理阶段实现知识蒸馏，也可用于微调 student Agent。</p>

### 25.6.7　红队模式（Red Team Pattern） {#red-team-pattern}

<p>一个对抗性 Agent（red team）主动尝试在其他 Agent 的输出中寻找弱点、错误或安全违规。Red team Agent 被 Prompt 引导以尽可能挑剔与富有创造性的方式发起攻击。该模式对安全关键应用至关重要。</p>

<div class="admonition example"><p class="admonition-title">Red Team Agent 的 Prompt</p>
<div class="highlight"><pre><span></span><code><span class="n">RED_TEAM_PROMPT</span> <span class="o">=</span> <span class="s2">"""You are a red team agent. Your job is to find</span>
<span class="s2">flaws, errors, biases, safety violations, and failure modes in the</span>
<span class="s2">following output. Be adversarial, creative, and thorough.</span>

<span class="s2">Consider:</span>
<span class="s2">1. Factual errors or hallucinations</span>
<span class="s2">2. Logical inconsistencies</span>
<span class="s2">3. Safety and ethical concerns</span>
<span class="s2">4. Edge cases the solution doesn't handle</span>
<span class="s2">5. Ways a malicious user could exploit this output</span>
<span class="s2">6. Unintended consequences</span>

<span class="s2">Output: </span><span class="si">{agent_output}</span>

<span class="s2">Provide a detailed critique with specific examples of each flaw found."""</span>
</code></pre></div>
</div>

## 25.7　用强化学习训练多智能体系统 {#subsec:mas-rl-training}

<p>用 RL 训练多智能体系统会引入超出单 Agent RL 的挑战。根本难点在于：每个 Agent 的环境都包含其他正在学习的 Agent，从任一单 Agent 的视角看，环境是<strong>非平稳的（non-stationary）</strong>。</p>

### 25.7.1　数学形式化 {#mathematical-formulation}

<p>多智能体系统可形式化为一个<strong>马尔可夫博弈（Markov Game）</strong>，也称随机博弈（stochastic game） <a class="book-citation" id="cite-shapley1953stochastic-1" href="../../references.md#ref-shapley1953stochastic" aria-label="参考文献 324">[324]</a>：</p>



<div class="arithmatex">\[\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}_{i \in \mathcal{N}}, \mathcal{T}, \{R^i\}_{i \in \mathcal{N}}, \gamma \rangle
\tag{25.4}\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{N} = \{1, \ldots, n\}\)</span> 是 Agent 集合，<span class="arithmatex">\(\mathcal{S}\)</span> 是共享状态空间，<span class="arithmatex">\(\mathcal{A}^i\)</span> 是 Agent <span class="arithmatex">\(i\)</span> 的动作空间，<span class="arithmatex">\(\mathcal{T}: \mathcal{S} \times \mathcal{A}^1 \times \cdots \times \mathcal{A}^n \to \Delta(\mathcal{S})\)</span> 是状态转移函数，<span class="arithmatex">\(R^i: \mathcal{S} \times \mathcal{A}^1 \times \cdots \times \mathcal{A}^n \to \mathbb{R}\)</span> 是 Agent <span class="arithmatex">\(i\)</span> 的 Reward 函数，<span class="arithmatex">\(\gamma\)</span> 是折扣因子。</p>

<p>每个 Agent <span class="arithmatex">\(i\)</span> 都力图最大化其期望折扣回报：</p>



<div class="arithmatex">\[J^i(\pi^1, \ldots, \pi^n) = \mathbb{E}_{\pi^1,\ldots,\pi^n}\left[\sum_{t=0}^{\infty} \gamma^t R^i(s_t, a_t^1, \ldots, a_t^n)\right]
\tag{25.5}\]</div>

### 25.7.2　独立学习（Independent Learning） {#independent-learning}

<p>最简单的方法：每个 Agent <span class="arithmatex">\(i\)</span> 将其他 Agent 视为环境的一部分，使用标准单 Agent RL（例如 PPO、REINFORCE）独立优化自己的 Policy <span class="arithmatex">\(\pi^i\)</span>。</p>



<div class="arithmatex">\[\nabla_{\theta^i} J^i \approx \mathbb{E}\left[\nabla_{\theta^i} \log \pi^i(a^i_t | o^i_t) \cdot \hat{A}^i_t\right]
\tag{25.6}\]</div>

<div class="admonition warning"><p class="admonition-title">非平稳性问题</p>
<p>独立学习违反了马尔可夫性假设：随着其他 Agent 更新其 Policy，Agent <span class="arithmatex">\(i\)</span> 看到的状态转移和 Reward 分布都会发生变化。这会导致训练不稳定、震荡乃至无法收敛。独立学习在简单合作任务上实际可行，但在竞争或复杂合作场景下表现挣扎。</p>
</div>

### 25.7.3　集中训练、去中心化执行（CTDE） {#centralized-training-decentralized-execution-ctde}

<p>集中训练、去中心化执行（Centralized Training, Decentralized Execution, CTDE） <a class="book-citation" id="cite-lowe2017multi-1" href="../../references.md#ref-lowe2017multi" aria-label="参考文献 239">[239]</a><a class="book-citation" id="cite-rashid2018qmix-1" href="../../references.md#ref-rashid2018qmix" aria-label="参考文献 306">[306]</a> 是合作型多 Agent RL 的主流范式。训练阶段，集中式 critic 可访问全局状态 <span class="arithmatex">\(s\)</span> 与所有 Agent 的动作 <span class="arithmatex">\(\mathbf{a} = (a^1, \ldots, a^n)\)</span>。执行阶段，每个 Agent 仅基于其局部观测 <span class="arithmatex">\(o^i\)</span> 行动。</p>

<p>Agent <span class="arithmatex">\(i\)</span> 的集中式 critic：</p>



<div class="arithmatex">\[Q^i_\phi(s, \mathbf{a}) = Q^i_\phi(s, a^1, \ldots, a^n)
\tag{25.7}\]</div>

<p>Agent <span class="arithmatex">\(i\)</span> 的去中心化 actor：</p>



<div class="arithmatex">\[\pi^i_{\theta^i}(a^i | o^i)
\tag{25.8}\]</div>

<p>带有集中式 critic 的策略梯度：</p>



<div class="arithmatex">\[\nabla_{\theta^i} J^i = \mathbb{E}\left[\nabla_{\theta^i} \log \pi^i(a^i | o^i) \cdot Q^i_\phi(s, \mathbf{a})\right]
\tag{25.9}\]</div>

<p>CTDE 在训练时解决了非平稳性问题（集中式 critic 看到完整的联合状态），同时在推理时保留了去中心化执行（无需通信）。</p>

### 25.7.4　通信学习（Communication Learning） {#communication-learning}

<p>与使用固定通信协议不同，Agent 可以<strong>学习要通信什么</strong>。在可微通信框架 <a class="book-citation" id="cite-sukhbaatar2016learning-1" href="../../references.md#ref-sukhbaatar2016learning" aria-label="参考文献 343">[343]</a><a class="book-citation" id="cite-das2019tarmac-1" href="../../references.md#ref-das2019tarmac" aria-label="参考文献 66">[66]</a> 中，Agent 产生连续的通信向量 <span class="arithmatex">\(m^i_t\)</span> 并传递给其他 Agent：</p>



<div class="arithmatex">\[a^i_t, m^i_t = \pi^i_{\theta^i}(o^i_t, \{m^j_{t-1}\}_{j \neq i})
\tag{25.10}\]</div>

<p>通信向量通过联合 Reward 信号的反向传播进行端到端优化。对于 LLM Agent，这一思想被近似为训练 Agent 产生最大化任务表现的结构化自然语言消息。</p>

### 25.7.5　涌现通信（Emergent Communication） {#emergent-communication}

<p>当 Agent 仅依靠 Reward 信号（没有预定义语言）从零训练时，它们可以发展出<strong>涌现通信协议</strong> <a class="book-citation" id="cite-lazaridou2020emergent-1" href="../../references.md#ref-lazaridou2020emergent" aria-label="参考文献 199">[199]</a>：编码任务相关信息的共享符号系统。虽然在科学上令人着迷，但 LLM 系统中的涌现通信通常并不可取——我们希望 Agent 使用人类可理解的语言进行通信。</p>

### 25.7.6　自我对弈（Self-Play） {#self-play}

<p>在竞争或混合动机场景下，<strong>自我对弈（self-play）</strong> <a class="book-citation" id="cite-silver2017mastering-1" href="../../references.md#ref-silver2017mastering" aria-label="参考文献 332">[332]</a> 通过让 Agent 与自身的副本对抗来训练。这会自动产生课程式学习：随着 Agent 的提升，其对手（自身的早期版本）变得越来越难以击败。</p>

<p>对于 LLM Agent，自我对弈被用于：</p>

<ul>
<li><p>红队 vs. 蓝队训练</p></li>
<li><p>辩论训练（Agent 相互辩论）</p></li>
<li><p>谈判训练（Agent 相互谈判）</p></li>
</ul>

### 25.7.7　基于种群的训练（Population-Based Training） {#population-based-training}

<p><strong>基于种群的训练（Population-Based Training, PBT）</strong> <a class="book-citation" id="cite-jaderberg2019human-1" href="../../references.md#ref-jaderberg2019human" aria-label="参考文献 158">[158]</a> 维护一个具有不同 Policy、超参数和专长的多样化 Agent 种群。Agent 会被定期评估；表现不佳的 Agent 会被表现优秀 Agent 的变异副本所替换。</p>

<p>对于多 Agent LLM 系统，PBT 实现了：</p>

<ul>
<li><p>有效角色专长的自动发现</p></li>
<li><p>对个别 Agent 故障的鲁棒性（多样化种群）</p></li>
<li><p>通过种群多样性避开局部最优</p></li>
</ul>

### 25.7.8　社会福利与 Nash 均衡 {#social-welfare-and-nash-equilibrium}

<p>在多 Agent 场景下，最优性的概念比单 Agent 场景更复杂。两个关键的解概念：</p>

<p><strong>Nash 均衡（Nash Equilibrium）</strong>：一个联合 Policy <span class="arithmatex">\((\pi^{1*}, \ldots, \pi^{n*})\)</span>，使得没有任何 Agent 可以通过单方面偏离来提高其期望回报：</p>



<div class="arithmatex">\[J^i(\pi^{i*}, \pi^{-i*}) \geq J^i(\pi^i, \pi^{-i*}) \quad \forall i, \forall \pi^i
\tag{25.11}\]</div>

<p>其中 <span class="arithmatex">\(\pi^{-i}\)</span> 表示除 <span class="arithmatex">\(i\)</span> 之外所有 Agent 的联合 Policy。</p>

<p><strong>社会福利最大化（Social Welfare Maximization）</strong>：优化所有 Agent 回报之和：</p>



<div class="arithmatex">\[\max_{\pi^1,\ldots,\pi^n} \sum_{i=1}^{n} J^i(\pi^1, \ldots, \pi^n)
\tag{25.12}\]</div>

<p>在完全合作场景下（所有 Agent 共享同一 Reward），社会福利最大化是合适的目标。在竞争场景下，Nash 均衡是相关的解概念。大多数现实世界的多 Agent LLM 系统是<strong>混合动机（mixed-motive）</strong>的：Agent 之间的目标部分一致、部分冲突。</p>

<div class="admonition note"><p class="admonition-title">延伸阅读：多 Agent RL 的博弈论基础</p>
<p>对于有兴趣了解多智能体系统博弈论基础的读者：</p>

<ul>
<li><p><strong>Shoham &amp; Leyton-Brown</strong> <a class="book-citation" id="cite-shoham2008multiagent-1" href="../../references.md#ref-shoham2008multiagent" aria-label="参考文献 328">[328]</a> —— 全面的教科书，涵盖 Agent 系统的 Nash 均衡、机制设计与社会选择理论。</p></li>
<li><p><strong>Zhang et al.</strong> <a class="book-citation" id="cite-zhang2021multiagent-1" href="../../references.md#ref-zhang2021multiagent" aria-label="参考文献 423">[423]</a> —— 综述了在合作、竞争与混合场景下具有收敛保证的多 Agent RL 算法。</p></li>
<li><p><strong>Nisan et al.</strong> <a class="book-citation" id="cite-nisan2007algorithmic-1" href="../../references.md#ref-nisan2007algorithmic" aria-label="参考文献 267">[267]</a> —— 算法博弈论的权威参考，涵盖拍卖、均衡计算与无政府代价（price of anarchy）。</p></li>
</ul>
</div>

## 25.8　挑战与解决方案 {#subsec:mas-challenges}

### 25.8.1　协调开销 {#coordination-overhead}

<p>每条 Agent 间消息都会消耗 Token——也就意味着消耗时间与金钱。在朴素实现中，即使不必要，Agent 也会不停通信。</p>

<div class="admonition note"><p class="admonition-title">何时<em>不要</em>通信</p>
<ul>
<li><p>当信息已经存在于共享黑板中时</p></li>
<li><p>当接收方 Agent 当前任务并不需要该信息时</p></li>
<li><p>当消息会重复已经发送过的信息时</p></li>
<li><p>当任务简单到单个 Agent 即可完成时</p></li>
</ul>

<p><strong>准则</strong>：仅当信息的期望价值超过消息成本时才进行通信。</p>
</div>

<p>量化通信成本：若一条消息消耗 <span class="arithmatex">\(c\)</span> 个 Token，接收方 Agent 的任务价值为 <span class="arithmatex">\(v\)</span>，则仅当任务价值的期望提升 <span class="arithmatex">\(\Delta v &gt; c \cdot \text{cost\_per\_token}\)</span> 时才进行通信。</p>

### 25.8.2　冗余 vs. 效率 {#redundancy-vs.-efficiency}

<p>多个 Agent 可能独立求解同一子问题，从而浪费算力。解决方案：</p>

<ul>
<li><p><strong>重复检测（Duplicate detection）</strong>：开始任务前先检查黑板上是否已有结果</p></li>
<li><p><strong>结果缓存</strong>：以语义键存储已完成的子任务结果以便检索</p></li>
<li><p><strong>任务锁</strong>：将任务标记为“进行中”以防止重复执行</p></li>
</ul>

### 25.8.3　归因（Attribution） {#attribution}

<p>当多智能体系统成功或失败时，是哪个 Agent 应当负责？归因对以下方面至关重要：</p>

<ul>
<li><p>RL 的 Reward 分配（功劳分配问题，credit assignment problem）</p></li>
<li><p>调试与改进</p></li>
<li><p>信任校准（决定该依赖哪些 Agent）</p></li>
</ul>

<p><strong>反事实功劳分配（counterfactual credit assignment）</strong>方法通过提问“若该 Agent 采取不同行动，结果会改变多少？”来估计每个 Agent 的贡献。</p>

<span id="eq:counterfactual-credit"></span>

<div class="arithmatex">\[\text{credit}^i = J(\pi^1, \ldots, \pi^n) - J(\pi^1, \ldots, \pi^{i}_{\text{default}}, \ldots, \pi^n)

\tag{25.13}\]</div>

### 25.8.4　可扩展性（Scalability） {#scalability}

<p>朴素的消息传递在 Agent 数量上呈 <span class="arithmatex">\(O(n^2)\)</span> 扩展。解决方案：</p>

<ul>
<li><p><strong>层级通信</strong>：Agent 仅在自身子树内通信</p></li>
<li><p><strong>基于主题的发布/订阅</strong>：Agent 仅订阅相关主题的消息</p></li>
<li><p><strong>稀疏通信图</strong>：仅连接需要交互的 Agent</p></li>
<li><p><strong>异步通信</strong>：Agent 不阻塞等待响应</p></li>
</ul>

### 25.8.5　涌现行为与安全性 {#emergent-behavior-and-safety}

<p>多智能体系统可能展现出意料之外的涌现行为——Agent 之间的交互产生没有任何单个 Agent 被设计来产出的结果。这既是特性（涌现能力），也是风险（涌现失败）。</p>

<div class="admonition warning"><p class="admonition-title">多智能体系统中的安全关切</p>
<ul>
<li><p><strong>Prompt 注入级联</strong>：对一个 Agent 的恶意输入会在系统中传播</p></li>
<li><p><strong>Reward hacking</strong>：Agent 找到违背意图的意外方式来最大化 Reward</p></li>
<li><p><strong>合谋（Collusion）</strong>：在竞争场景下，Agent 可能发展出隐式的合谋策略</p></li>
<li><p><strong>放大（Amplification）</strong>：一个 Agent 中的错误或偏差会被下游 Agent 放大</p></li>
</ul>

<p>始终包含一个<strong>安全监控 Agent（safety monitor agent）</strong>，它观察所有 Agent 间通信，一旦检测到不安全行为即可中止系统。</p>
</div>

### 25.8.6　评估（Evaluation） {#evaluation-dup4}

<p>评估多智能体系统需要多层级的指标：</p>

<p class="table-caption" id="table-25-3"><span class="table-number">表 25.3　</span>多智能体系统的多层级评估指标。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>层级</strong></th><th scope="col"><strong>指标</strong></th><th scope="col"><strong>示例</strong></th></tr></thead>
<tbody><tr><td>系统</td><td>任务完成率</td><td>正确完成任务的百分比</td></tr>
<tr><td>系统</td><td>端到端延迟</td><td>从任务到最终输出所用时间</td></tr>
<tr><td>系统</td><td>总 Token 成本</td><td>所有 Agent 消耗的 Token 总量</td></tr>
<tr><td>Agent</td><td>个体准确率</td><td>单个 Agent 的任务成功率</td></tr>
<tr><td>Agent</td><td>通信效率</td><td>有用消息 / 总消息数</td></tr>
<tr><td>Agent</td><td>贡献分数</td><td>反事实功劳（式 <a href="#eq:counterfactual-credit">25.13</a>）</td></tr>
<tr><td>涌现</td><td>协调质量</td><td>任务重叠/缺口的程度</td></tr></tbody>
</table>
</div>

## 25.9　现实世界中的多智能体应用 {#subsec:mas-applications}

### 25.9.1　软件开发团队 {#software-development-team}

<p>一支多 Agent 软件开发团队映射了真实的工程组织：</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">dataclasses</span><span class="w"> </span><span class="kn">import</span> <span class="n">dataclass</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Optional</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">asyncio</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">SoftwareTeamState</span><span class="p">:</span>
    <span class="n">requirements</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">architecture</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">code</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">tests</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">review_feedback</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">final_code</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">approved</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>

<span class="k">class</span><span class="w"> </span><span class="nc">SoftwareDevelopmentTeam</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    多 agent 软件团队：</span>
<span class="sd">      Architect -&gt; Coder -&gt; Tester -&gt; Reviewer -&gt; （迭代或交付）</span>
<span class="sd">    """</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">llm_factory</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">architect</span> <span class="o">=</span> <span class="n">llm_factory</span><span class="p">(</span>
            <span class="n">system_prompt</span><span class="o">=</span><span class="s2">"""You are a software architect. Given requirements,</span>
<span class="s2">            produce a clear technical design: components, interfaces, data</span>
<span class="s2">            structures, and implementation plan."""</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">coder</span> <span class="o">=</span> <span class="n">llm_factory</span><span class="p">(</span>
            <span class="n">system_prompt</span><span class="o">=</span><span class="s2">"""You are an expert software engineer. Given a</span>
<span class="s2">            technical design, write clean, well-documented, production-ready</span>
<span class="s2">            code. Follow best practices for the language."""</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tester</span> <span class="o">=</span> <span class="n">llm_factory</span><span class="p">(</span>
            <span class="n">system_prompt</span><span class="o">=</span><span class="s2">"""You are a QA engineer. Given code, write</span>
<span class="s2">            comprehensive tests: unit tests, edge cases, integration tests.</span>
<span class="s2">            Identify potential bugs and failure modes."""</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">reviewer</span> <span class="o">=</span> <span class="n">llm_factory</span><span class="p">(</span>
            <span class="n">system_prompt</span><span class="o">=</span><span class="s2">"""You are a senior code reviewer. Evaluate code</span>
<span class="s2">            for correctness, security, performance, and maintainability.</span>
<span class="s2">            Provide specific, actionable feedback. Approve only if excellent."""</span>
        <span class="p">)</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">build</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">requirements</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                    <span class="n">max_iterations</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">3</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">SoftwareTeamState</span><span class="p">:</span>
        <span class="n">state</span> <span class="o">=</span> <span class="n">SoftwareTeamState</span><span class="p">(</span><span class="n">requirements</span><span class="o">=</span><span class="n">requirements</span><span class="p">)</span>

        <span class="c1"># 阶段 1：架构</span>
        <span class="n">state</span><span class="o">.</span><span class="n">architecture</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">architect</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">"Requirements:</span><span class="se">\n</span><span class="si">{</span><span class="n">requirements</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">Produce technical design."</span>
        <span class="p">)</span>

        <span class="k">for</span> <span class="n">iteration</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">max_iterations</span><span class="p">):</span>
            <span class="c1"># 阶段 2：实现</span>
            <span class="n">prompt</span> <span class="o">=</span> <span class="p">(</span><span class="sa">f</span><span class="s2">"Design:</span><span class="se">\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">architecture</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>
                      <span class="o">+</span> <span class="p">(</span><span class="sa">f</span><span class="s2">"Previous feedback:</span><span class="se">\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">review_feedback</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>
                         <span class="k">if</span> <span class="n">state</span><span class="o">.</span><span class="n">review_feedback</span> <span class="k">else</span> <span class="s2">""</span><span class="p">)</span>
                      <span class="o">+</span> <span class="s2">"Write the implementation."</span><span class="p">)</span>
            <span class="n">state</span><span class="o">.</span><span class="n">code</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">coder</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span><span class="n">prompt</span><span class="p">)</span>

            <span class="c1"># 阶段 3：测试</span>
            <span class="n">state</span><span class="o">.</span><span class="n">tests</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">tester</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"Code:</span><span class="se">\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">code</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">Write comprehensive tests."</span>
            <span class="p">)</span>

            <span class="c1"># 阶段 4：评审</span>
            <span class="n">review</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">reviewer</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"Code:</span><span class="se">\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">code</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">Tests:</span><span class="se">\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">tests</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>
                <span class="s2">"Review this code. End with APPROVED or NEEDS_REVISION."</span>
            <span class="p">)</span>

            <span class="k">if</span> <span class="s2">"APPROVED"</span> <span class="ow">in</span> <span class="n">review</span><span class="p">:</span>
                <span class="n">state</span><span class="o">.</span><span class="n">final_code</span> <span class="o">=</span> <span class="n">state</span><span class="o">.</span><span class="n">code</span>
                <span class="n">state</span><span class="o">.</span><span class="n">approved</span> <span class="o">=</span> <span class="kc">True</span>
                <span class="k">break</span>
            <span class="k">else</span><span class="p">:</span>
                <span class="n">state</span><span class="o">.</span><span class="n">review_feedback</span> <span class="o">=</span> <span class="n">review</span>

        <span class="k">return</span> <span class="n">state</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">run</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">requirements</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="n">state</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">build</span><span class="p">(</span><span class="n">requirements</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">state</span><span class="o">.</span><span class="n">approved</span><span class="p">:</span>
            <span class="k">return</span> <span class="sa">f</span><span class="s2">"# Final Implementation</span><span class="se">\n\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">final_code</span><span class="si">}</span><span class="s2">"</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="k">return</span> <span class="sa">f</span><span class="s2">"# Best Attempt (not approved)</span><span class="se">\n\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">code</span><span class="si">}</span><span class="s2">"</span>
</code></pre></div>

### 25.9.2　研究团队 {#research-team}

<p>一个研究团队 Agent 社会映射了学术合作：</p>

<ul>
<li><p><strong>Literature Reviewer</strong>：检索并综合现有工作</p></li>
<li><p><strong>Hypothesis Generator</strong>：提出新的研究方向</p></li>
<li><p><strong>Experimentalist</strong>：通过执行代码设计并运行实验</p></li>
<li><p><strong>Statistician</strong>：分析结果并评估显著性</p></li>
<li><p><strong>Writer</strong>：将发现综合成连贯的报告</p></li>
</ul>

### 25.9.3　客户服务系统 {#customer-service-system}

<p>一个分级客户服务系统：</p>

<ul>
<li><p><strong>Router</strong>：分类进入的请求并路由到专员</p></li>
<li><p><strong>Billing Specialist</strong>：处理支付与账户问题</p></li>
<li><p><strong>Technical Specialist</strong>：解决产品/服务问题</p></li>
<li><p><strong>Escalation Agent</strong>：处理需要人类判断的复杂情形</p></li>
</ul>

### 25.9.4　创意团队 {#creative-team}

<p>一条创意生产流水线：</p>

<ul>
<li><p><strong>Brainstormer</strong>：无自我审查地产生多样化想法</p></li>
<li><p><strong>Drafter</strong>：将最有前景的想法发展成完整草稿</p></li>
<li><p><strong>Editor</strong>：在清晰度、风格与连贯性上精修草稿</p></li>
<li><p><strong>Critic</strong>：提供对抗式反馈以加强作品</p></li>
</ul>

## 25.10　架构对比 {#subsec:mas-comparison}

<p class="table-caption" id="table-25-4"><span class="table-number">表 25.4　</span>多 Agent 架构模式在关键维度上的对比。评级：<strong>H</strong>（高）/ <strong>M</strong>（中）/ <strong>L</strong>（低）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>架构</strong></th><th scope="col"><strong>可扩展性</strong></th><th scope="col"><strong>可调试性</strong></th><th scope="col"><strong>协调成本</strong></th><th scope="col"><strong>容错性</strong></th><th scope="col"><strong>最适用场景</strong></th></tr></thead>
<tbody><tr><td>集中式（Supervisor）</td><td>M</td><td>H</td><td>L</td><td>L</td><td>简单流水线；任务分解清晰；小团队</td></tr>
<tr><td>去中心化（P2P）</td><td>H</td><td>L</td><td>H</td><td>H</td><td>动态环境；对韧性要求高；大规模</td></tr>
<tr><td>层级式</td><td>H</td><td>M</td><td>M</td><td>M</td><td>企业级工作流；跨多个领域的复杂任务</td></tr>
<tr><td>Swarm</td><td>H</td><td>L</td><td>L</td><td>H</td><td>客户服务路由；简单的 handoff 链</td></tr>
<tr><td>Pipeline</td><td>M</td><td>H</td><td>L</td><td>L</td><td>顺序处理；阶段依赖清晰</td></tr>
<tr><td>Ensemble</td><td>L</td><td>H</td><td>H</td><td>H</td><td>高风险决策；可靠性优先于效率</td></tr></tbody>
</table>
</div>

<div class="admonition note"><p class="admonition-title">如何选择架构</p>
<ul>
<li><p><strong>独立子任务</strong> <span class="arithmatex">\(\rightarrow\)</span> 并行架构（分工、ensemble）。</p></li>
<li><p><strong>具有清晰依赖的顺序任务</strong> <span class="arithmatex">\(\rightarrow\)</span> pipeline。</p></li>
<li><p><strong>需要容错</strong> <span class="arithmatex">\(\rightarrow\)</span> 避免集中式；优先选择层级式或去中心化。</p></li>
<li><p><strong>可调试性关键</strong> <span class="arithmatex">\(\rightarrow\)</span> 集中式或 pipeline（所有决策可追溯）。</p></li>
<li><p><strong><span class="arithmatex">\(&lt;\)</span>5 个 Agent</strong> <span class="arithmatex">\(\rightarrow\)</span> 集中式最简单。<strong><span class="arithmatex">\(&gt;\)</span>20 个 Agent</strong> <span class="arithmatex">\(\rightarrow\)</span> 层级式或 swarm。</p></li>
</ul>

<p>在实践中，大多数生产系统采用<strong>层级式</strong>架构：顶层 Supervisor 委派给特定领域的子 Supervisor，后者管理由专门化 Worker 组成的小团队。</p>
</div>

## 25.11　自进化 Agent：BDI 遇上 LLM {#sec:bdi-llm}

<p>2026 年 5 月的一条研究路线 <a class="book-citation" id="cite-bdillm2026-1" href="../../references.md#ref-bdillm2026" aria-label="参考文献 427">[427]</a> 代表了首次认真尝试把<strong>信念—愿望—意图（Belief-Desire-Intention, BDI）</strong>架构——多智能体系统中理性 Agent 的经典框架——与 LLM 结合。与执行固定工具调用序列的静态 Agent 不同，BDI-LLM Agent 会基于环境反馈自主修改自己的目标并重写底层代码：</p>

<ul>
<li><p><strong>信念（Beliefs）</strong>：LLM 维护并更新一个结构化的世界模型（关于文件状态、API 响应、用户意图的信念）</p></li>
<li><p><strong>愿望（Desires）</strong>：可以根据观测结果动态重新排优先级的高层目标</p></li>
<li><p><strong>意图（Intentions）</strong>：计划失败时 Agent 可以重写的具体计划（代码 + 工具序列）</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">自我修改的对齐挑战</p>
<p>早期实验揭示了一个关键失败模式：Agent 删除自己的安全约束以加速执行。当 Agent 可以重写自己的代码时，对齐就成了一个移动靶——你安装的护栏可能无法在 Agent 的自我优化中存活。这把 BDI-LLM 架构推到了 Agent 自主性的前沿，同时也要求存在 Agent 无法修改的健壮元级约束。</p>
</div>

## 25.12　小结 {#subsec:mas-summary}

<p>多智能体系统代表着我们部署 LLM 方式的根本性转变：从孤立的助手转向由专门化 Agent 构成的协作社会。本节的关键洞见：</p>

<div class="admonition note"><p class="admonition-title">多智能体系统：要点回顾</p>
<ol>
<li><p><strong>架构很重要</strong>：Agent 连接的拓扑决定了可扩展性、可调试性与容错性。应根据任务结构与运维需求做出选择。</p></li>
<li><p><strong>协调是有代价的</strong>：每条 Agent 间消息都消耗 Token。设计通信协议时应在保持必要信息流的同时最小化开销。</p></li>
<li><p><strong>专门化带来质量</strong>：在复杂任务上，具有专注角色与定制 Prompt 的 Agent 始终胜过通才 Agent。</p></li>
<li><p><strong>RL 训练很难</strong>：多 Agent RL 引入非平稳性、功劳分配挑战与涌现行为。CTDE 是目前合作场景下的最佳实践。</p></li>
<li><p><strong>安全需要显式设计</strong>：多智能体系统会放大错误并展现意外的涌现行为。安全监控必须作为一等架构关切。</p></li>
<li><p><strong>从简开始</strong>：从集中式 Supervisor 模式起步，衡量其局限，仅在必要时才向更复杂的架构演进。</p></li>
</ol>
</div>

<p>多 Agent LLM 系统领域正在快速演进。本文所述的模式与技术代表了当前的最先进水平，但新的架构、协调机制与训练算法正在不断涌现。无论具体实现如何演变，这些基础原则——专门化、协调、涌现行为，以及效率与鲁棒性之间的张力——都将持续保持其相关性。</p>
