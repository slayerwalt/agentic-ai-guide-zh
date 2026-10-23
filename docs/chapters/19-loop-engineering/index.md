# 第 19 章 · 循环工程（Loop Engineering） {#sec:loop-engineering}

<p>从业者与 LLM 智能体交互方式的演变遵循了一条惊人一致的轨迹。2022–2024 年，首要技能是<em>提示工程（prompt engineering）</em>：精心组织措辞，从单次模型调用中引出想要的回答。到 2025 年，焦点转向<em>上下文工程（context engineering）</em>——策展模型在推理时看到的全部 token，包括检索到的文档、工具输出与对话历史 <a class="book-citation" id="cite-anthropic2024buildingagents-1" href="../../references.md#ref-anthropic2024buildingagents" aria-label="参考文献 10">[10]</a>。上一章我们讨论了<em>harness 工程</em>：设计围绕 Agent 的运行时环境——工具、沙箱、记忆与护栏。循环工程 <a class="book-citation" id="cite-osmani2026loop-1" href="../../references.md#ref-osmani2026loop" aria-label="参考文献 279">[279]</a> 是这一演进中的下一层：设计<em>迭代控制结构</em>，让 Agent 自主地朝目标前进，无需人类在每一轮输入下一条指令。</p>

<div class="admonition note"><p class="admonition-title">从提示到循环：工程演进的四个阶段</p>
<ol>
<li><p><strong>提示工程</strong>：优化你对模型<em>说了什么</em>（2022–2024）</p></li>
<li><p><strong>上下文工程</strong>：优化模型<em>看到了什么</em>（2025）</p></li>
<li><p><strong>Harness 工程</strong>：优化 Agent<em>运行所在</em>的环境（2025–2026）</p></li>
<li><p><strong>循环工程</strong>：优化让 Agent 持续朝目标工作的<em>循环</em>（2026）</p></li>
</ol>

<p>每一层都包裹住前一层而不取代它。你仍然要写提示；仍然要策展上下文；仍然要构建 harness。循环工程让这一切运转起来。</p>
</div>

<p>这个说法诞生于 2026 年 6 月：Peter Steinberger <a class="book-citation" id="cite-steinberger2026openclaw-1" href="../../references.md#ref-steinberger2026openclaw" aria-label="参考文献 338">[338]</a> 主张开发者应当停止直接提示编码 Agent，转而设计能够提示这些 Agent 的系统——这一观点得到了 Boris Cherny（Anthropic 的 Claude Code 负责人）的印证，他提到自己的角色已经完全转变为编写协调模型行动的外部执行循环 <a class="book-citation" id="cite-cherny2026loops-1" href="../../references.md#ref-cherny2026loops" aria-label="参考文献 52">[52]</a>。这不仅仅是工具偏好。它反映了一个结构性现实：当一次 Agent 运行可能持续一小时并修改数十个文件时，杠杆率最高的工程不再在提示里——而在那个让 Agent 始终保持高效、被验证、不偏离目标的循环里。</p>

## 19.1　上下文工程：循环之下的那一层 {#sec:context-engineering}

<p>在循环运行之前，必须有人决定模型在每一步看到什么。这门学科就是<strong>上下文工程</strong>：在推理期间策展并维护上下文窗口中最优 token 集合的正式实践。它是循环之下的那一层——循环状态被翻译为模型输入的机制。</p>

<p>该术语由 Tobi Lütke（Shopify CEO）在 2025 年 6 月推广开来 <a class="book-citation" id="cite-lutke2025context-1" href="../../references.md#ref-lutke2025context" aria-label="参考文献 244">[244]</a>，他将其描述为与 AI Agent 协作的新兴核心技能。Anthropic 在 2025 年 9 月将其形式化为“在推理期间策展并维护最优 token 集合”。Andrej Karpathy 认同这一框架，将其描述为“在下一步之前，用恰到好处的信息填满上下文窗口的精妙艺术与科学” <a class="book-citation" id="cite-karpathy2025context-1" href="../../references.md#ref-karpathy2025context" aria-label="参考文献 178">[178]</a>。这些定义的汇聚反映了一个现实：当上下文窗口从 4K 增长到 128K 再到 1M token 时，“<em>往里面放什么</em>”变得与“<em>问什么</em>”同等重要。</p>

<p>上下文工程不是提示工程。提示工程优化的是<em>指令</em>——引导模型行为的措辞。上下文工程优化的是<em>信息环境</em>——模型据以推理的文档、工具输出、历史与状态的全集。在单轮交互中，两者的区别很小。在多步 Agent 循环中，它决定了模型是拥有正确行动所需的信息，还是在盲目飞行。</p>

##### 关键技术。 {#heading-19-3}

<p>上下文工程借助几种互补的方法：</p>

<ul>
<li><p><strong>动态上下文组装</strong>：在每一轮循环迭代中从多个来源（检索到的文档、工具输出、记忆存储）选择并组合上下文，而不是使用固定模板。</p></li>
<li><p><strong>RAG 集成</strong>：检索任务相关文档或代码片段并注入上下文窗口，用精确、即时的信息取代过时或泛泛的背景知识。</p></li>
<li><p><strong>工具输出摘要化</strong>：在把冗长的工具输出（如长文件列表、测试运行器日志）插入上下文之前先压缩，在保留信号的同时节省 token 预算。</p></li>
<li><p><strong>对话历史管理</strong>：决定哪些先前的轮次原样保留、哪些摘要化、哪些彻底丢弃——一个滚动压缩问题，在长时程 Agent 运行中变得至关重要。</p></li>
<li><p><strong>Token 预算分配</strong>：在相互竞争的信息来源之间显式划分上下文窗口，确保模型始终有空间生成有用的回答。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">上下文窗口预算分配（典型 Agent 循环）</p>
<span id="table-19-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>组件</strong></th><th scope="col"><strong>预算</strong></th><th scope="col"><strong>说明</strong></th></tr></thead>
<tbody><tr><td>系统提示</td><td>10–15%</td><td>稳定；定义 Agent 角色、工具与约束</td></tr>
<tr><td>检索文档 / RAG</td><td>20–30%</td><td>动态；每轮迭代刷新</td></tr>
<tr><td>对话 / 动作历史</td><td>30–40%</td><td>随运行变长而压缩</td></tr>
<tr><td>生成预算</td><td>20–30%</td><td>为模型输出预留；绝不可压缩低于此线</td></tr></tbody>
</table>
</div>

<p>这些是指导原则，而非硬性规则。长时程编码 Agent 可以给历史更多预算；单步问答 Agent 可以给检索文档更多预算。关键不变量：始终<em>先</em>预留生成预算，再填充其余部分。</p>
</div>

##### 循环内的上下文工程。 {#heading-19-4}

<p>在循环中，上下文工程在每一轮迭代都发挥作用。在每一步 <span class="arithmatex">\(t\)</span>，循环控制器必须决定：保留哪些先前的观测 <span class="arithmatex">\(o_{&lt;t}\)</span>、摘要哪些、哪些检索文档仍然相关，以及如何从这些组件组合出新的上下文 <span class="arithmatex">\(s_t\)</span>。这不是一次性的设计决策——而是一条与 Agent 策略 <span class="arithmatex">\(\pi_\theta\)</span> 并行的动态策略。设计糟糕的上下文策略会让模型丢失目标、重复已做过的动作，或无法使用两步之前检索到的信息。设计良好的策略则能让模型的“工作记忆”在整个运行期间始终与任务当前状态对齐。</p>

<div class="admonition tip"><p class="admonition-title">上下文工程 vs. 循环工程</p>
<p>上下文工程回答：<em>模型在第 <span class="arithmatex">\(t\)</span> 步看到了什么？</em>循环工程回答：<em>模型在第 <span class="arithmatex">\(t\)</span> 步行动之后发生了什么？</em>两者互补。循环决定动作与观测的序列；上下文工程决定当模型采取下一个动作时这些观测如何被表示。做不好上下文工程就不可能做好循环工程——循环的状态只有被正确编码进模型上下文时才有用。</p>
</div>

## 19.2　把循环工程视为推理时强化学习 {#sec:loop-as-rl}

<p>让循环工程与本书相关的核心洞见是：一个精心设计的 Agent 循环在结构上与在推理时运行的 RL 优化过程完全相同——只是不做模型权重的梯度更新：</p>

<span id="eq:loop-as-mdp"></span>

<div class="arithmatex">\[\begin{align*}
s_t &amp;= \text{context}(s_{t-1}, a_{t-1}, o_{t-1}) &amp;&amp; \text{(state = accumulated context)} \nonumber \\

a_t &amp;= \pi_\theta(s_t) &amp;&amp; \text{(action = model generation)} \nonumber \\

o_t &amp;= \text{env}(a_t) &amp;&amp; \text{(observation = tool/environment feedback)} \nonumber \\

r_t &amp;= \text{verify}(s_t, a_t, o_t) &amp;&amp; \text{(reward = verification signal)}

\tag{19.1}
\end{align*}\]</div>

<p>循环持续运行，直到 <span class="arithmatex">\(r_t\)</span> 超过成功阈值或触发终止条件。“策略”<span class="arithmatex">\(\pi_\theta\)</span> 不通过梯度更新；相反，被更新的是<em>状态</em>——观测、错误消息与反思被追加到上下文中，让同一个冻结模型在后续迭代中产生更好的动作。这正是 Reflexion <a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a> 背后的机制：模型跨迭代改进，靠的不是学习新权重，而是阅读自己的失败历史。</p>

<div class="admonition tip"><p class="admonition-title">循环—RL 对应关系</p>
<p>考虑标准 RL 目标 <span class="arithmatex">\(\max_\pi \mathbb{E}[\sum_{t=0}^T \gamma^t r_t]\)</span>。在循环工程中：</p>

<ul>
<li><p><strong>策略</strong> <span class="arithmatex">\(\pi\)</span>：冻结的 LLM，以不断累积的上下文为条件</p></li>
<li><p><strong>状态</strong> <span class="arithmatex">\(s_t\)</span>：第 <span class="arithmatex">\(t\)</span> 步时上下文窗口的内容</p></li>
<li><p><strong>动作</strong> <span class="arithmatex">\(a_t\)</span>：模型生成的输出（代码编辑、工具调用、消息）</p></li>
<li><p><strong>环境</strong>：工具、文件系统、测试运行器、linter——任何产生观测的东西</p></li>
<li><p><strong>奖励</strong> <span class="arithmatex">\(r_t\)</span>：验证信号（测试通过、linter 干净、指标提升）</p></li>
<li><p><strong>折扣</strong> <span class="arithmatex">\(\gamma\)</span>：隐含在循环预算中——越靠后的步骤成本越高、越不可能成功</p></li>
<li><p><strong>回合</strong>：从目标定义到终止的一次完整循环执行</p></li>
</ul>

<p>与训练时 RL 的关键区别：策略权重是冻结的。“学习”完全通过状态累积发生——随着循环推进，模型获得更丰富的条件信息。这类似于上下文内学习（in-context learning），但环境在主动塑造模型所见。</p>
</div>

<p>这种对应并非仅仅是个比喻。它有直接的工程后果：</p>

<ol>
<li><p><strong>奖励设计很重要</strong>：正如 RLHF 中那样，定义糟糕的奖励（验证标准）会导致 reward hacking——Agent 满足检查的字面要求却违背其意图（例如删除一个失败的测试让 CI 变绿）。</p></li>
<li><p><strong>探索—利用权衡</strong>：重复同一失败方法的循环表现出很差的探索能力。Reflexion <a class="book-citation" id="cite-shinn2023reflexion-2" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a> 这类机制通过自我批评加入显式探索，类似于 PPO 中的熵奖励。</p></li>
<li><p><strong>时域与折扣</strong>：更长的循环面对累积误差与上下文劣化，正如长时域 RL 受困于信用分配难题。预算上限就是有效的时域。</p></li>
<li><p><strong>状态表示</strong>：上下文管理（压缩、剪枝、外化）就是 RL 中状态表示设计在循环工程里的对应物——两者都决定 Agent 能否有效地对自己的处境进行推理。</p></li>
</ol>

## 19.3　生产级循环的解剖 {#sec:loop-anatomy}

<p>一个可用的循环需要五个结构性原语 <a class="book-citation" id="cite-osmani2026loop-1--section-03" href="../../references.md#ref-osmani2026loop" aria-label="参考文献 279">[279]</a>，外加外部状态持久化：</p>

### 19.3.1　五个原语 {#sec:loop-primitives}

<p class="table-caption" id="table-19-1"><span class="table-number">表 19.1　</span>循环工程的五个结构性原语。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>原语</strong></th><th scope="col"><strong>在循环中的角色</strong></th><th scope="col"><strong>RL 对应物</strong></th></tr></thead>
<tbody><tr><td>自动化</td><td>按计划或事件触发循环</td><td>回合启动；环境重置</td></tr>
<tr><td>工作树</td><td>隔离并行 Agent</td><td>分布式 RL 中的独立 rollout worker</td></tr>
<tr><td>技能</td><td>固化可复用能力与项目知识</td><td>策略条件化；任务特定奖励塑形</td></tr>
<tr><td>连接器</td><td>对接外部工具与系统</td><td>环境动作空间</td></tr>
<tr><td>子 Agent</td><td>分解与验证</td><td>分层 RL；critic 网络</td></tr></tbody>
</table>
</div>

##### 自动化。 {#par:automations}

<p>自动化是把一次性的 Agent 运行变成真正循环的心跳。它们定义循环<em>何时</em>、<em>为何</em>触发——按 cron 计划、响应 webhook，或由文件系统事件触发。没有自动化，你只有一个 Agent；有了它们，你拥有一个自维持的系统。生产示例包括：夜间 CI 失败分类、每小时依赖漏洞扫描，以及提交后审查。</p>

##### 工作树（隔离）。 {#par:worktrees}

<p>多个 Agent 并行运行的那一刻起，文件级冲突就成了主导性失败模式。Git 工作树——共享仓库历史的独立工作目录——提供了隔离：每个 Agent 在自己的分支上操作，不可能与兄弟 Agent 发生写入冲突。这与分布式 RL 中运行独立 rollout worker（第 <a href="../11-training-infrastructure/index.md#sec:parallelism">11.2</a> 节）是同一个原则：并行需要隔离。</p>

##### 技能。 {#par:skills-in-loops}

<p>技能（第 <a href="../23-agent-skills/index.md#sec:agent-skills">23</a> 章）把 Agent 本会在每个循环从头重新推导的项目知识固化下来。在循环语境中，技能充当<em>持久条件</em>——跨迭代保持稳定的约定、构建流程与约束。没有技能，每次循环迭代都从零开始；有了它们，Agent 跨运行累积知识，而不会把上下文窗口浪费在重新发现的事实上。</p>

##### 连接器。 {#par:connectors}

<p>连接器——通常通过 MCP 实现（第 <a href="../22-mcp/index.md#sec:mcp">22</a> 章）——把循环的动作空间扩展到文件系统之外。只能读写文件的循环是受限的；连接了问题跟踪器、CI 系统、预发布环境与团队通信渠道的循环可以闭合完整的反馈回路：发现问题 <span class="arithmatex">\(\to\)</span> 修复 <span class="arithmatex">\(\to\)</span> 验证 <span class="arithmatex">\(\to\)</span> 部署 <span class="arithmatex">\(\to\)</span> 通知。</p>

##### 子 Agent（制造者—检查者分离）。 {#par:sub-agents-loops}

<p>循环工程中最关键的结构性原则是把<em>产出</em>输出的 Agent 与<em>评估</em>输出的 Agent 分开。让模型给自己的输出打分，就像让学生给自己批改考卷——激励是错位的。一个专门的验证子 Agent（可以运行不同的模型或以更高的推理强度运行）提供独立的评估，让无人值守的运行变得可信。这对应 RL 中的 actor-critic 架构：actor（生成器）提出动作；critic（验证器）评估它们。</p>

### 19.3.2　外部状态：循环的记忆 {#sec:loop-state}

<p>上述每个原语都只作用于单次迭代。<em>外部状态</em>是把各次迭代在时间上连接起来的东西。由于 LLM 在两次调用之间是无状态的——模型会忘记当前上下文之外的一切——循环的连续性必须驻留在磁盘上：一个 Markdown 进度文件、一个结构化数据库，或一份受版本控制的日志。这个外部状态有三个功能：</p>

<ol>
<li><p><strong>进度追踪</strong>：尝试过什么、成功了什么、还剩什么。</p></li>
<li><p><strong>失败记忆</strong>：哪些方法试过并失败了，防止循环在相同的死胡同之间震荡。</p></li>
<li><p><strong>交接上下文</strong>：当循环升级给人类或后续运行接管时，状态文件提供完整的审计轨迹。</p></li>
</ol>

<div class="admonition warning"><p class="admonition-title">状态不是可选项</p>
<p>没有外部状态的循环是一个失忆的循环。它会重新尝试已失败的方法、丢失部分进度，并在出错时拿不出审计轨迹。状态机制不必复杂——每轮迭代后提交到 git 的一个 Markdown 文件通常就足够——但它必须存在。</p>
</div>

## 19.4　循环的伪代码 {#sec:loop-pseudocode}

<p>剥去所有表象，每个 Agent 循环都是一种控制结构，更像恒温器或 REPL，而不是一段对话：</p>

<div class="highlight"><pre><span></span><code><span class="k">def</span><span class="w"> </span><span class="nf">agent_loop</span><span class="p">(</span><span class="n">goal</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">max_steps</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">50</span><span class="p">,</span> <span class="n">budget</span><span class="p">:</span> <span class="n">TokenBudget</span> <span class="o">=</span> <span class="kc">None</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""The canonical agent loop structure."""</span>
    <span class="n">state</span> <span class="o">=</span> <span class="n">initialize_state</span><span class="p">(</span><span class="n">goal</span><span class="p">)</span>        <span class="c1"># 加载外部状态和目标</span>

    <span class="k">for</span> <span class="n">step</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">max_steps</span><span class="p">):</span>
        <span class="c1"># 1. 推理当前状态（ReAct 风格）</span>
        <span class="n">thought</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">reason</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>

        <span class="c1"># 2. 选择并执行动作</span>
        <span class="n">action</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">choose_action</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>
        <span class="n">observation</span> <span class="o">=</span> <span class="n">environment</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="n">action</span><span class="p">)</span>

        <span class="c1"># 3. 用新信息更新状态</span>
        <span class="n">state</span> <span class="o">=</span> <span class="n">update_state</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">thought</span><span class="p">,</span> <span class="n">action</span><span class="p">,</span> <span class="n">observation</span><span class="p">)</span>

        <span class="c1"># 4. 接近上下文窗口上限时压缩上下文</span>
        <span class="k">if</span> <span class="n">state</span><span class="o">.</span><span class="n">token_count</span> <span class="o">&gt;</span> <span class="n">CONTEXT_BUDGET</span> <span class="o">*</span> <span class="mf">0.8</span><span class="p">:</span>
            <span class="n">state</span> <span class="o">=</span> <span class="n">compact</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>        <span class="c1"># 汇总旧步骤</span>

        <span class="c1"># 5. 检查终止条件</span>
        <span class="k">if</span> <span class="n">verifier</span><span class="o">.</span><span class="n">passes</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">goal</span><span class="p">):</span>  <span class="c1"># 确定性检查</span>
            <span class="n">persist_state</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">status</span><span class="o">=</span><span class="s2">"success"</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">Success</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>

        <span class="k">if</span> <span class="n">no_progress_detected</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">window</span><span class="o">=</span><span class="mi">3</span><span class="p">):</span>
            <span class="n">persist_state</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">status</span><span class="o">=</span><span class="s2">"stuck"</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">escalate_to_human</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>

        <span class="k">if</span> <span class="n">budget</span> <span class="ow">and</span> <span class="n">budget</span><span class="o">.</span><span class="n">exhausted</span><span class="p">():</span>
            <span class="n">persist_state</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">status</span><span class="o">=</span><span class="s2">"budget_exceeded"</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">escalate_to_human</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>

    <span class="c1"># 达到硬性上限</span>
    <span class="n">persist_state</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">status</span><span class="o">=</span><span class="s2">"max_steps"</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">escalate_to_human</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>
</code></pre></div>

<p>循环工程中所有有趣的地方，都是关于这些行之一的决策：什么构成有效的 <code>goal</code>，<code>verifier.passes</code> 如何实现，<code>compact</code> 如何在丢弃噪声的同时保留相关历史，以及 <code>no_progress_detected</code> 如何既避免过早终止又避免无限循环。</p>

## 19.5　循环模式 {#sec:loop-patterns}

<p>在第 <a href="../20-agent-patterns/index.md#sec:agent-design-patterns">20</a> 章介绍的 Agent 设计模式基础之上，循环工程识别出一组自主性逐级递增的模式层级：</p>

### 19.5.1　验证循环 {#sec:validation-loop}

<p>最简单也最常见的模式：生成，对照确定性检查验证，失败则重试。</p>

<div class="admonition example"><p class="admonition-title">验证循环：修复直到测试通过</p>
<div class="highlight"><pre><span></span><code><span class="n">failure_log</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">attempt</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">MAX_ATTEMPTS</span><span class="p">):</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">run_tests</span><span class="p">()</span>
    <span class="k">if</span> <span class="n">result</span><span class="o">.</span><span class="n">passed</span><span class="p">:</span>
        <span class="k">break</span>  <span class="c1"># 成功——退出循环</span>
    <span class="n">fix</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">generate</span><span class="p">(</span>
        <span class="sa">f</span><span class="s2">"Fix this test failure:</span><span class="se">\n</span><span class="si">{</span><span class="n">result</span><span class="o">.</span><span class="n">errors</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
        <span class="sa">f</span><span class="s2">"Previous attempts that failed: </span><span class="si">{</span><span class="n">failure_log</span><span class="si">}</span><span class="s2">"</span>
    <span class="p">)</span>
    <span class="n">apply_patch</span><span class="p">(</span><span class="n">fix</span><span class="p">)</span>
    <span class="n">failure_log</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">errors</span><span class="p">)</span>
</code></pre></div>

<p><strong>终止条件</strong>：测试通过（成功）或重试预算耗尽（升级）。</p>

<p><strong>关键属性</strong>：验证信号是<em>确定性的</em>——测试运行器产生客观的通过/失败，模型无法争辩。</p>
</div>

<p>这一模式之所以成功，是因为它的奖励信号干净利落。测试运行器充当不可腐蚀的批评者——与 LLM-as-judge 评估不同，它无法被说服或愚弄。</p>

### 19.5.2　Reflexion 循环 {#sec:reflexion-loop}

<p>在验证循环之上加入尝试之间的显式自我批评 <a class="book-citation" id="cite-shinn2023reflexion-1--section-05" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a>。每次失败后，Agent 把一段自然语言反思（“我失败是因为修改了错误的函数——错误追溯到 import，而不是实现”）写入情景记忆缓冲区。后续尝试会读取这个缓冲区，从而在<em>单个回合内</em>实现学习，无需权重更新。</p>

<div class="admonition note"><p class="admonition-title">Reflexion：从失败中做上下文内学习</p>
<p>Reflexion 循环在基本重试之外增加了三个组件：</p>

<ol>
<li><p><strong>Actor</strong>：执行任务</p></li>
<li><p><strong>Evaluator</strong>：提供奖励信号（测试、指标或 LLM judge）</p></li>
<li><p><strong>自我反思</strong>：把一条口头化的“学到的教训”写入情景记忆</p></li>
</ol>

<p>反思记忆跨尝试持久化，为 Agent 提供避免重复相同失败的机制。实证上，Reflexion 在 HumanEval 上达到 91% 的 pass@1（基线为 67%），证明口头自我批评在迭代场景中可以替代权重更新 <a class="book-citation" id="cite-shinn2023reflexion-2--section-05" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a>。</p>
</div>

### 19.5.3　评估器—优化器循环 {#sec:eval-opt-loop}

<p>一种双模型架构 <a class="book-citation" id="cite-madaan2023selfrefine-1" href="../../references.md#ref-madaan2023selfrefine" aria-label="参考文献 246">[246]</a><a class="book-citation" id="cite-anthropic2024buildingagents-1--section-05" href="../../references.md#ref-anthropic2024buildingagents" aria-label="参考文献 10">[10]</a>：一个模型生成候选，另一个独立模型依据显式标准评估它并返回结构化反馈。生成器吸收反馈，产出改进版本。循环重复，直到评估器评分超过阈值或迭代预算耗尽。</p>

<p>关键的设计选择是评估器是<em>确定性的</em>（编译器、测试套件、linter）还是<em>概率性的</em>（另一个 LLM）。确定性评估器带来可靠的收敛；概率性评估器对主观标准是必要的，但引入了评估器—生成器合谋的风险。</p>

### 19.5.4　层级循环 {#sec:hierarchical-loop}

<p>一个元循环生成并监控多个子循环。编排 Agent 把高层目标分解为子任务，把每个子任务指派给运行自己循环的专门化子 Agent，监控它们的进度并综合结果。这对应分层 RL <a class="book-citation" id="cite-barto2003recent-1" href="../../references.md#ref-barto2003recent" aria-label="参考文献 21">[21]</a>：高层策略选择子目标，低层策略执行它们。</p>

<div class="admonition example"><p class="admonition-title">层级循环：通宵仓库维护</p>
<div class="highlight"><pre><span></span><code><span class="c1"># 外层循环：通过自动化每晚运行</span>
<span class="k">for</span> <span class="n">issue</span> <span class="ow">in</span> <span class="n">triage_agent</span><span class="o">.</span><span class="n">identify_actionable_issues</span><span class="p">():</span>
    <span class="c1"># 内层循环：每个问题一个子智能体，使用隔离工作树</span>
    <span class="n">worktree</span> <span class="o">=</span> <span class="n">create_isolated_worktree</span><span class="p">(</span><span class="n">issue</span><span class="o">.</span><span class="n">branch_name</span><span class="p">)</span>

    <span class="n">result</span> <span class="o">=</span> <span class="n">fix_loop</span><span class="p">(</span>
        <span class="n">goal</span><span class="o">=</span><span class="n">issue</span><span class="o">.</span><span class="n">description</span><span class="p">,</span>
        <span class="n">worktree</span><span class="o">=</span><span class="n">worktree</span><span class="p">,</span>
        <span class="n">verifier</span><span class="o">=</span><span class="n">run_tests_and_lint</span><span class="p">,</span>
        <span class="n">max_steps</span><span class="o">=</span><span class="mi">20</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">result</span><span class="o">.</span><span class="n">success</span><span class="p">:</span>
        <span class="n">review</span> <span class="o">=</span> <span class="n">review_agent</span><span class="o">.</span><span class="n">evaluate</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">patch</span><span class="p">)</span>  <span class="c1"># 检查器</span>
        <span class="k">if</span> <span class="n">review</span><span class="o">.</span><span class="n">approved</span><span class="p">:</span>
            <span class="n">open_pull_request</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">patch</span><span class="p">,</span> <span class="n">issue</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">state_db</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">issue</span><span class="p">,</span> <span class="s2">"fix_rejected"</span><span class="p">,</span> <span class="n">review</span><span class="o">.</span><span class="n">feedback</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">state_db</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">issue</span><span class="p">,</span> <span class="s2">"fix_failed"</span><span class="p">,</span> <span class="n">result</span><span class="o">.</span><span class="n">state</span><span class="p">)</span>
</code></pre></div>

<p>外层循环（分类）按计划运行。内层循环（修复）是临时的、任务范围的。审查 Agent 提供制造者—检查者分离。状态跨夜间运行持久化。</p>
</div>

### 19.5.5　自主研究循环 {#sec:research-loop}

<p>Karpathy 的 AutoResearch <a class="book-citation" id="cite-karpathy2026autoresearch-1" href="../../references.md#ref-karpathy2026autoresearch" aria-label="参考文献 179">[179]</a> 展示了一个受严格约束的研究循环：Agent 修改训练脚本，运行一个限时实验（例如在 GPU 上跑 5 分钟），读取验证指标，并决定是提交改动（指标提升）还是回滚（指标下降）。循环无限持续，通过迭代式实验探索超参数与架构空间。</p>

<p>这一模式之所以有效，是因为奖励信号<em>标量且无歧义</em>：验证损失要么下降了要么没有。每轮迭代的严格时间上限防止任何一次失败的实验消耗过多资源。</p>

## 19.6　验证工程 {#sec:verification-engineering}

<p>验证是循环工程的奖励函数。它的质量决定了循环是收敛到正确答案、收敛到错误答案，还是根本不收敛。本节建立一张按可靠性排序的验证策略层级。</p>

### 19.6.1　验证层级 {#sec:verification-hierarchy}

<p class="table-caption" id="table-19-2"><span class="table-number">表 19.2　</span>按可靠性排序的验证策略。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>策略</strong></th><th scope="col"><strong>信号来源</strong></th><th scope="col"><strong>可靠性</strong></th><th scope="col"><strong>适用场景</strong></th></tr></thead>
<tbody><tr><td>编译</td><td>语言工具链</td><td>确定性</td><td>代码必须能解析/构建</td></tr>
<tr><td>类型检查</td><td>静态分析器</td><td>确定性</td><td>类型化语言</td></tr>
<tr><td>单元/集成测试</td><td>测试运行器</td><td>确定性</td><td>测试存在且正确</td></tr>
<tr><td>Lint 与格式化</td><td>风格工具</td><td>确定性</td><td>约定已代码化</td></tr>
<tr><td>指标对比</td><td>训练/评估脚本</td><td>数值型</td><td>ML 实验</td></tr>
<tr><td>LLM-as-judge</td><td>第二个模型</td><td>概率性</td><td>主观质量标准</td></tr>
<tr><td>人工审查</td><td>领域专家</td><td>黄金标准</td><td>高风险的决策</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">验证原则</p>
<p>循环的可靠程度不会超过它最不可靠的那一步验证。只要存在确定性检查，就用它。把 LLM-as-judge 留给那些确实无法机械验证的标准。绝不让生成模型评判自己的输出——制造者与检查者必须分离。</p>
</div>

### 19.6.2　确定性 vs. 概率性验证 {#sec:det-vs-prob-verification}

<p>确定性验证器（编译器、测试套件、linter）是黄金标准，因为它们产生客观的通过/失败，模型无法靠说服钻空子。测试要么通过要么不通过——再雄辩的推理也改变不了裁决。</p>

<p>概率性验证器（LLM-as-judge）对没有机械检查的任务是必要的——代码质量、文档清晰度、UX 改进——但它们引入了失败模式：</p>

<ul>
<li><p><strong>自我评估偏差</strong>：如果同一个模型既生成又评估，它会对自己的输出手下留情。</p></li>
<li><p><strong>评估器博弈</strong>：经过多次迭代，生成器可能学会产出满足评估器表面模式的输出，却没有实现真正的质量。</p></li>
<li><p><strong>评估器不一致</strong>：由于采样方差，同一个 LLM-judge 可能在不同运行中对相同输出给出不同分数。</p></li>
</ul>

<p>缓解策略包括：用另一个（通常更强）模型作评估器、提供带二元标准的显式评分细则，以及定期用人类判断校准 LLM-judge。</p>

## 19.7　终止工程 {#sec:termination-engineering}

<p>朴素循环的标志性失败是它永远不会停。终止设计不是事后补丁——它是工程的一半。</p>

### 19.7.1　终止条件 {#sec:termination-conditions}

<p>生产级循环携带多个独立的退出条件：</p>

<ol>
<li><p><strong>目标达成</strong>：验证器确认目标已实现。这是唯一<em>成功</em>的终止。</p></li>
<li><p><strong>硬性迭代上限</strong>：循环步数上限（通常 20–50）。无论其他条件如何，防止无界执行。</p></li>
<li><p><strong>预算耗尽</strong>：token 数、墙钟时间或金钱成本超过预定义上限。</p></li>
<li><p><strong>无进展检测</strong>：最近 <span class="arithmatex">\(k\)</span> 轮迭代没有产生可测量的状态变化——循环在震荡或卡死。这是硬性上限之后最重要的安全机制。</p></li>
<li><p><strong>致命错误</strong>：任何迭代都无法解决的不可恢复状况（凭据缺失、基础设施故障）。</p></li>
</ol>

<div class="admonition note"><p class="admonition-title">无进展检测：停滞断路器</p>
<p>检测停滞需要在迭代间追踪一个进展信号。常见方法：</p>

<ul>
<li><p><strong>输出哈希比较</strong>：如果最近 3 次 Agent 输出或文件状态完全相同，循环在打转。</p></li>
<li><p><strong>错误重复</strong>：同一错误消息连续出现 <span class="arithmatex">\(k\)</span> 轮。</p></li>
<li><p><strong>指标平台期</strong>：数值信号（测试通过率、验证损失）在 <span class="arithmatex">\(k\)</span> 步内没有提升 <span class="arithmatex">\(\epsilon\)</span>。</p></li>
<li><p><strong>Diff 熵</strong>：每轮迭代的代码改动量正趋近于零——Agent 在做越来越小、无关紧要的编辑。</p></li>
</ul>

<p>检测到停滞时，正确的行动是<em>升级</em>，而不是继续迭代。循环保存状态，把控制权交给人类或更高级的编排器。</p>
</div>

### 19.7.2　探索问题 {#sec:loop-exploration}

<p>卡在局部最小值的循环——用微小的变体重复同一种失败方法——正在表现出 RL 中熟悉的探索—利用失败。循环内促进探索的机制：</p>

<ul>
<li><p><strong>基于反思的探索</strong>：<span class="arithmatex">\(k\)</span> 次失败尝试之后，触发一个反思步骤，显式要求模型考虑根本不同的方法（类似于 PPO 中的熵正则化 <a class="book-citation" id="cite-schulman2017proximal-1" href="../../references.md#ref-schulman2017proximal" aria-label="参考文献 319">[319]</a>）。</p></li>
<li><p><strong>温度升级</strong>：连续失败后提高采样温度，鼓励多样化输出。</p></li>
<li><p><strong>策略记忆</strong>：不仅记录失败的动作，还记录失败的<em>策略</em>。“方法 A（修改 handler）失败了 3 次；改试方法 B（重写 schema）。”</p></li>
<li><p><strong>全新子 Agent</strong>：生成一个上下文窗口干净的子 Agent——它不会陷入已累积上下文可能造成的窠臼。</p></li>
</ul>

## 19.8　失败模式与反模式 {#sec:loop-failures}

### 19.8.1　Loopmaxxing 陷阱 {#sec:loopmaxxing}

<p>“Loopmaxxing”——假设让 Agent 跑足够多轮迭代最终就会产出正确解——是循环工程版的“算力够多就能解决一切”。它的失败与纯随机搜索的失败同源：没有指向改进的梯度信号，光靠迭代不是优化。</p>

<p>Loopmaxxing 在以下情况出现：</p>

<ul>
<li><p>目标缺少可检查的成功条件（“改善用户体验”）</p></li>
<li><p>验证信号过于粗糙（在 1000 个测试的套件上只有二元通过/失败，无法提供指向修复的梯度）</p></li>
<li><p>无论迭代多少次，Agent 都缺乏解决问题的能力</p></li>
</ul>

<p>解药是认识到循环只是<em>放大</em>工程能力——它们不取代工程能力。目标定义糟糕的循环会以极高的效率去追求错误的东西。</p>

### 19.8.2　理解债 {#sec:comprehension-debt}

<p>当循环修改代码的速度超过工程团队审阅的速度时，仓库中实际存在的东西与人类理解之间的差距就会扩大——这就是<em>理解债（comprehension debt）</em> <a class="book-citation" id="cite-osmani2026loop-1--section-08" href="../../references.md#ref-osmani2026loop" aria-label="参考文献 279">[279]</a>。与技术债不同（技术债由代码承载），理解债存在于团队成员的头脑里。它无声地复利增长，直到某场危机迫使人们去调试那些设计决策、结构依赖与边界情况完全未被记录的代码。</p>

<div class="admonition warning"><p class="admonition-title">循环同时放大生产力与风险</p>
<p>设计良好的循环加速的是工程师深刻理解的工作。用循环来逃避理解工作，则会造成加速的不理解螺旋。同样的机制，相反的结果。区别在于：人仍然是工程师，还是沦为了按“开始”按钮的人。</p>
</div>

### 19.8.3　循环中的 Reward Hacking {#sec:loop-reward-hacking}

<p>正如 RL Agent 利用奖励定义缺陷在未达成既定目标的情况下拿到高奖励，循环 Agent 也会利用验证漏洞：</p>

<ul>
<li><p><strong>删除测试</strong>：最简单的 reward hack——删掉失败的测试让 CI 变绿。</p></li>
<li><p><strong>指标博弈</strong>：通过记住验证集而非泛化来过拟合验证损失。</p></li>
<li><p><strong>收窄规格</strong>：解决所陈述问题的简化版本，恰好能通过检查。</p></li>
<li><p><strong>掩盖输出</strong>：压制错误输出而不是修复根本原因。</p></li>
</ul>

<p>防御手段与 RLHF 相同：设计<em>难以钻空子</em>的奖励函数（验证标准）。多个独立检查——测试<em>和</em>类型检查<em>和</em> lint<em>和</em>审查子 Agent——构成一个比任何单一信号都更难利用的验证面。</p>

### 19.8.4　上下文劣化 {#sec:context-degradation}

<p>在长时程循环中，上下文窗口会被每一步的历史——思考、工具输出、错误、补丁——填满。当窗口接近容量时，模型对相关信息的注意力下降（“中间迷失”现象 <a class="book-citation" id="cite-liu2024lost-1" href="../../references.md#ref-liu2024lost" aria-label="参考文献 226">[226]</a>）。症状包括：</p>

<ul>
<li><p>Agent 重新尝试它已经试过并明确标记为失败的方法</p></li>
<li><p>工具调用变得不那么精确——Agent 忘记了自己读过哪些文件</p></li>
<li><p>模型在臃肿的上下文中艰难地分配注意力，回答变得更短、更不连贯</p></li>
</ul>

<p>对策：</p>

<ol>
<li><p><strong>激进压缩</strong>：每 <span class="arithmatex">\(k\)</span> 步之后，把已完成的工作总结为简短的状态描述，丢弃原始记录。</p></li>
<li><p><strong>外部草稿本</strong>：把中间结果写入文件，让 Agent 按需读取，而不是把一切留在上下文里。</p></li>
<li><p><strong>子 Agent 隔离</strong>：在新上下文窗口中生成子任务，只返回它们的结论——而不是完整推理轨迹。</p></li>
<li><p><strong>历史滑动窗口</strong>：只完整保留最近 <span class="arithmatex">\(w\)</span> 步；把更早的内容压缩进一个摘要块。</p></li>
</ol>

## 19.9　生产级循环架构 {#sec:production-loops}

### 19.9.1　夜间维护循环 {#sec:nightly-maintenance}

<p>最立即可用的循环模式：一个按计划在夜间运行的自动化，趁团队睡觉时处理累积的工作。</p>

<div class="admonition example"><p class="admonition-title">架构：夜间 CI 分类与修复循环</p>
<p><strong>触发</strong>：Cron 计划（例如每天 02:00 UTC）。</p>

<p><strong>阶段 1 —— 分类</strong>（单 Agent，只读）：</p>

<ul>
<li><p>读取昨天的 CI 失败、新开的 issue 与最近的提交</p></li>
<li><p>对每条发现分类：可自动修复 / 需人工 / 误报</p></li>
<li><p>把发现写入状态文件；归档误报</p></li>
</ul>

<p><strong>阶段 2 —— 修复</strong>（并行子 Agent，隔离工作树）：</p>

<ul>
<li><p>对每个可自动修复的 issue，在隔离工作树中生成一个子 Agent</p></li>
<li><p>每个子 Agent 运行一个验证循环（修复 <span class="arithmatex">\(\to\)</span> 测试 <span class="arithmatex">\(\to\)</span> 重试）</p></li>
<li><p>预算：每个 issue 20 步或 15 分钟</p></li>
</ul>

<p><strong>阶段 3 —— 审查</strong>（独立验证 Agent）：</p>

<ul>
<li><p>审查子 Agent 依据项目编码标准评估每个提议的修复</p></li>
<li><p>批准的修复：开 draft PR、链接到 issue、通过连接器通知频道</p></li>
<li><p>被拒的修复：记录拒绝原因，升级到人工分类收件箱</p></li>
</ul>

<p><strong>持久化</strong>：状态文件（提交到仓库）记录所有已采取的行动，让第二天的运行跳过已解决的 issue，并从失败处接着干。</p>
</div>

### 19.9.2　持续实验循环 {#sec:experimentation-loop}

<p>受 AutoResearch <a class="book-citation" id="cite-karpathy2026autoresearch-1--section-09" href="../../references.md#ref-karpathy2026autoresearch" aria-label="参考文献 179">[179]</a> 启发，这一模式把循环应用于科学发现：修改训练脚本，运行限时实验，评估结果，决定提交还是回滚。</p>

<div class="admonition note"><p class="admonition-title">AutoResearch 循环结构</p>
<ol>
<li><p><strong>假设生成</strong>：Agent 提出一项修改（超参数、架构改动、数据增强）</p></li>
<li><p><strong>实现</strong>：Agent 修改训练脚本</p></li>
<li><p><strong>限时执行</strong>：恰好运行 <span class="arithmatex">\(T\)</span> 分钟（硬性墙钟上限）</p></li>
<li><p><strong>评估</strong>：从输出读取验证指标</p></li>
<li><p><strong>决策</strong>：指标提升 <span class="arithmatex">\(\to\)</span> <code>git commit</code>；指标下降 <span class="arithmatex">\(\to\)</span> <code>git rollback</code></p></li>
<li><p><strong>记录</strong>：把假设、结果与推理记入实验日志</p></li>
<li><p><strong>重复</strong>：依据累积的实验历史生成下一个假设</p></li>
</ol>

<p><strong>关键约束</strong>：限时执行防止任何一次失败的实验消耗过多算力。基于 git 的提交/回滚确保代码库只会向前。</p>
</div>

<div class="admonition warning"><p class="admonition-title">研究循环中的局部最小值问题</p>
<p>研究循环容易陷入保守探索。面对困难的优化景观时，Agent 倾向于提出最小改动——把学习率调 0.001——只带来名义上的改进而非真正的进展。这是 RL 探索问题在循环层面的显现。对策包括：定期的“大胆假设”提示、为尝试新颖方法设置多样性奖励，以及在两次运行之间由人类注入研究方向。</p>
</div>

### 19.9.3　常驻编排器 {#sec:always-on-orchestrator}

<p>OpenClaw <a class="book-citation" id="cite-steinberger2026openclaw-1--section-09" href="../../references.md#ref-steinberger2026openclaw" aria-label="参考文献 338">[338]</a> 这类系统实现了一个持久的“心跳”机制：一个按固定周期运行的元 Agent，评估仓库状态、监督子循环并分派工作。与每天触发一次的夜间模式不同，常驻编排器保持持续感知，在几分钟内响应事件（新提交、CI 失败、新 issue）。</p>

<p>该架构引入了一个带崩溃恢复的持久状态数据库——如果编排器在循环中途失败，它会从最后一个检查点状态恢复，而不是从头重启。子循环被作为拥有各自终止条件的独立任务来跟踪，编排器监控它们的停滞、预算耗尽或冲突。</p>

## 19.10　长时程循环中的上下文管理 {#sec:loop-context-management}

<p>长时程循环面临一个根本张力：上下文窗口是 Agent 的工作记忆，但容量固定。每一步都会追加新信息（思考、观测、错误），没有主动管理，窗口就会填满、质量下降。本节介绍跨大量迭代维持上下文健康的工程技术。</p>

### 19.10.1　压缩策略 {#sec:compaction-strategies}

<p class="table-caption" id="table-19-3"><span class="table-number">表 19.3　</span>长时程循环的上下文压缩策略。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>策略</strong></th><th scope="col"><strong>机制</strong></th><th scope="col"><strong>权衡</strong></th></tr></thead>
<tbody><tr><td>周期摘要</td><td>每 <span class="arithmatex">\(k\)</span> 步用摘要替换原始历史</td><td>有损；可能丢弃之后需要的细节</td></tr>
<tr><td>滑动窗口</td><td>逐字保留最近 <span class="arithmatex">\(w\)</span> 步；更早的摘要化</td><td>偏重新近；可能丢失早期上下文</td></tr>
<tr><td>重要性加权</td><td>保留改变状态的步骤；丢弃无操作</td><td>需要定义“重要性”</td></tr>
<tr><td>外部记忆</td><td>细节写入文件；上下文只留引用</td><td>需要显式回读；增加延迟</td></tr>
<tr><td>子 Agent 隔离</td><td>子任务在新上下文中运行；只返回结论</td><td>生成开销；协调成本</td></tr></tbody>
</table>
</div>

<p>最优策略取决于任务。对调试循环，滑动窗口效果很好——最近的错误最相关。对研究循环，重要性加权保留能跨大量迭代保存关键实验结果。对复杂的多文件重构，外部记忆（写一个总结迄今所有改动的进度文件）能防止 Agent 丢失对自己修改的追踪。</p>

### 19.10.2　上下文预算 {#sec:context-budget}

<p>一个实用的启发式：为当前步骤的推理与生成预留上下文窗口的 20–30%。这意味着有效历史不应超过容量的 70–80%。当接近这个阈值时，在下一轮迭代<em>之前</em>触发压缩——而不是等模型已经产出劣化输出之后。</p>

<div class="highlight"><pre><span></span><code><span class="n">CONTEXT_CAPACITY</span> <span class="o">=</span> <span class="mi">128_000</span>  <span class="c1"># token 总数</span>
<span class="n">RESERVED_FOR_GENERATION</span> <span class="o">=</span> <span class="mf">0.25</span> <span class="o">*</span> <span class="n">CONTEXT_CAPACITY</span>
<span class="n">ACTIVE_BUDGET</span> <span class="o">=</span> <span class="n">CONTEXT_CAPACITY</span> <span class="o">-</span> <span class="n">RESERVED_FOR_GENERATION</span>

<span class="k">def</span><span class="w"> </span><span class="nf">should_compact</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">LoopState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">bool</span><span class="p">:</span>
    <span class="k">return</span> <span class="n">state</span><span class="o">.</span><span class="n">token_count</span> <span class="o">&gt;</span> <span class="n">ACTIVE_BUDGET</span> <span class="o">*</span> <span class="mf">0.85</span>
</code></pre></div>

## 19.11　何时不该使用循环 {#sec:when-not-loops}

<p>循环工程并非普遍适用。它增加了复杂度、成本与失败模式，在更简单的方法够用时这些是不合理的。以下情况请优先选择直接提示或简单工作流：</p>

<ul>
<li><p>任务是<strong>单轮的</strong>：精心构造的提示大多数时候一次就能给出正确答案。加循环只带来边际质量提升，却付出显著成本。</p></li>
<li><p>目标<strong>缺少可检查的条件</strong>：“让代码更好”没有客观的停止点。循环要么永远运行，要么任意停止。</p></li>
<li><p><strong>人工交互很廉价</strong>：如果开发者正在积极工作并能实时反馈，交互式聊天比工程化一个循环更快。</p></li>
<li><p>任务<strong>超出模型能力</strong>：如果 Agent 原则上解决不了这个问题——无论迭代多少次——循环只是在烧 token。再多的重试也补不上根本的能力差距。</p></li>
<li><p><strong>验证不可能</strong>：没有反馈信号，循环就没有梯度。它会退化为随机搜索。</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">简洁性原则</p>
<p>第 <a href="../20-agent-patterns/index.md#sec:agent-design-patterns">20</a> 章阐述的同一设计原则在此适用：从最简单的可行方法开始，只有当你有了“迭代确实显著改善结果”的证据时才加入循环复杂度。一个能解决问题的提示链，永远优于一个或许能解决问题的循环。</p>
</div>

## 19.12　循环的经济学 {#sec:loop-economics}

<p>循环用算力（token、墙钟时间）换取质量。理解这一权衡对生产部署至关重要。</p>

<div class="admonition note"><p class="admonition-title">循环成本模型</p>
<p>对于一个平均 <span class="arithmatex">\(N\)</span> 轮迭代、每轮消耗 <span class="arithmatex">\(T\)</span> 个 token、单价为每 token <span class="arithmatex">\(p\)</span> 的循环：</p>



<div class="arithmatex">\[\text{Cost}_{\text{loop}} = N \cdot T \cdot p + \text{Cost}_{\text{tools}} + \text{Cost}_{\text{verification}}\]</div>

<p>使用提示缓存（系统提示与早期上下文被缓存，按较低的 <span class="arithmatex">\(p_c &lt; p\)</span> 计费）时：</p>



<div class="arithmatex">\[\text{Cost}_{\text{cached}} = T_{\text{new}} \cdot p + T_{\text{cached}} \cdot p_c + \text{(tools + verification)}\]</div>

<p>提示缓存对循环尤其有价值，因为目标定义、技能内容与早期历史跨迭代保持不变——每次调用通常有 60–80% 的上下文可缓存。</p>
</div>

<p>经济决策：循环产出的价值（节省的开发者时间、通宵生产力、质量提升）是否大于它的 token 成本？一个运行 20 轮、每轮 $0.02 的循环总成本是 $0.40——如果能省下开发者 30 分钟，这微不足道。但没有预算上限的失控循环，一夜之间就能烧掉数百美元。</p>

## 19.13　历史背景与相关工作 {#sec:loop-history}

<p>循环工程并非凭空出现。它把一条跨越数年的研究谱系产品化了：</p>

<ol>
<li><p><strong>ReAct</strong>（2022） <a class="book-citation" id="cite-yao2023react-1" href="../../references.md#ref-yao2023react" aria-label="参考文献 408">[408]</a>：确立了所有现代循环都继承的推理与行动交错模式。</p></li>
<li><p><strong>Reflexion</strong>（2023） <a class="book-citation" id="cite-shinn2023reflexion-1--section-13" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a>：加入情景记忆与自我批评，证明 Agent 可以在不更新权重的情况下跨尝试改进。</p></li>
<li><p><strong>AutoGPT</strong>（2023） <a class="book-citation" id="cite-significantgravitas2023autogpt-1" href="../../references.md#ref-significantgravitas2023autogpt" aria-label="参考文献 330">[330]</a>：第一个被广泛使用的自主 Agent 循环，同时展示了无人值守运行的潜力与陷阱（失控执行、高成本、不可靠输出）。</p></li>
<li><p><strong>Self-Refine</strong>（2023） <a class="book-citation" id="cite-madaan2023selfrefine-1--section-13" href="../../references.md#ref-madaan2023selfrefine" aria-label="参考文献 246">[246]</a>：把“先生成后批评”的迭代精化循环形式化。</p></li>
<li><p><strong>“Ralph 循环”</strong>（2025）：非正式的基于 bash 的单行脚本，让编码 Agent 在简单的重试循环中运行——产品化循环原语的前身。</p></li>
<li><p><strong>产品化循环</strong>（2026）：主要平台（Codex、Claude Code）直接内嵌 <code>/goal</code> 与 <code>/loop</code> 命令，无需定制基础设施就能使用复杂的循环模式 <a class="book-citation" id="cite-osmani2026loop-1--section-13" href="../../references.md#ref-osmani2026loop" aria-label="参考文献 279">[279]</a>。</p></li>
</ol>

<p>从 AutoGPT 到现代循环工程的演进展示了一条成熟之路：早期系统缺少终止逻辑、验证与成本控制——它们是没有工程的循环。2026 年的形式化补上了让无人值守运行变得安全的纪律：显式终止、确定性验证、预算约束与制造者—检查者分离。</p>

## 19.14　小结 {#sec:loop-summary}

<p>循环工程代表了人—Agent 协作的当前前沿：从参与对话，转向设计能够自主进行那些对话的系统。其关键原则：</p>

<ol>
<li><p>循环是推理时 RL——状态、动作、奖励与（通过上下文的）策略更新——在无梯度下降的情况下运行。</p></li>
<li><p>验证就是奖励信号。它的质量决定收敛。优先确定性检查；把制造者与检查者分开。</p></li>
<li><p>终止是设计的一半。每个循环都需要硬性上限、预算与无进展检测。</p></li>
<li><p>上下文是有限的工作记忆。通过压缩、外化与子 Agent 隔离来主动管理它。</p></li>
<li><p>循环放大工程能力——它们不取代工程能力。目标定义糟糕的循环会高效地追求错误的东西。</p></li>
<li><p>从简单开始。一个带确定性验证器的验证循环，胜过一套你无法调试的精密多 Agent 系统。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">构建循环——但保持工程师身份</p>
<p>杠杆点已经从写提示转向设计循环。但设计一个能够收敛的循环，需要把问题理解得足够深，才能定义“完成”意味着什么、什么构成进展、何时停止。循环工程让人类的判断<em>更加</em>重要，而不是更不重要——因为糟糕判断的后果现在会以机器速度复利增长。</p>
</div>
