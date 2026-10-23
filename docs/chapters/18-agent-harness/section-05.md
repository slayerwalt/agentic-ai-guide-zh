# 18.5　编排模式 {#subsec:orchestration-patterns}

<p>编排定义了 Agent <em>如何</em>决定下一步该做什么。不同的模式适用于不同的任务结构。</p>

## 18.5.1　ReAct 循环（Reason + Act） {#react-loop-reason-act}

<p><strong>ReAct</strong> 模式 <a class="book-citation" id="cite-yao2023react-1" href="../../references.md#ref-yao2023react" aria-label="参考文献 408">[408]</a> 在一个紧凑循环中交替进行推理（&quot;Thought&quot;）、行动（&quot;Act&quot;）与观察（&quot;Observe&quot;）：</p>



<div class="arithmatex">\[\text{Thought}_t \to \text{Action}_t \to \text{Observation}_t \to \text{Thought}_{t+1} \to \cdots
\tag{18.10}\]</div>

<figure id="fig:react-loop"><a href="../../assets/figures/fig_057_react-loop.png" target="_blank" rel="noopener" aria-label="查看图 18.5 原图" title="点击查看原图"><img src="../../assets/figures/fig_057_react-loop.png" alt="ReAct 循环：Agent 在推理与行动之间交替，直到满足终止条件。" loading="lazy"></a><figcaption><span class="figure-number">图 18.5　</span>ReAct 循环：Agent 在推理与行动之间交替，直到满足终止条件。</figcaption></figure>

#### 实现细节。 {#implementation-details.}

<ul>
<li><p>&quot;Thought&quot; 步骤通常是一个 scratchpad——一条思维链（Chain-of-Thought，CoT）推理轨迹 <a class="book-citation" id="cite-wei2022chain-1" href="../../references.md#ref-wei2022chain" aria-label="参考文献 374">[374]</a>，<em>不会</em>展示给用户。</p></li>
<li><p>harness 解析模型输出，提取动作（工具名 + 参数）。</p></li>
<li><p><strong>最大迭代次数</strong>守卫可防止无限循环。</p></li>
<li><p>当模型输出 &quot;Final Answer&quot; 动作或停止 Token 时循环终止。</p></li>
</ul>

## 18.5.2　Plan-and-Execute {#plan-and-execute}

<p>Agent 不是逐步决策，而是先生成一个完整的计划，然后依次执行每一步 <a class="book-citation" id="cite-wang2023planandsolve-1" href="../../references.md#ref-wang2023planandsolve" aria-label="参考文献 364">[364]</a>：</p>

<ol>
<li><p><strong>规划阶段：</strong> 给定任务，生成结构化计划（带依赖关系的子任务列表）。</p></li>
<li><p><strong>执行阶段：</strong> 执行每个子任务，可能使用不同（更便宜）的模型。</p></li>
<li><p><strong>计划修订：</strong> 如果某步骤失败或产生意外结果，则从当前状态重新规划。</p></li>
</ol>



<div class="arithmatex">\[\text{Plan} = \text{Planner}(q), \quad \text{Result} = \prod_{i=1}^{|\text{Plan}|} \text{Executor}(\text{Plan}[i],\, \text{context}_i)
\tag{18.11}\]</div>

<p>Plan-and-Execute 对长时程任务更高效（LLM 调用更少），但对意外观察的适应性较差。</p>

## 18.5.3　多 Agent 编排 {#multi-agent-orchestration}

<p>复杂任务受益于多个专业化 Agent 协同工作。四种典型模式：</p>

#### Supervisor（监督者）模式。 {#supervisor-pattern.}

<p>一个中央&quot;supervisor&quot; LLM 接收用户请求，将其分解，并将子任务路由到专门的 Agent。结果由 supervisor 聚合。</p>

<figure id="figure-18-6"><a href="../../assets/figures/fig_058_supervisor.png" target="_blank" rel="noopener" aria-label="查看图 18.6 原图" title="点击查看原图"><img src="../../assets/figures/fig_058_supervisor.png" alt="Supervisor 模式：一个编排者将任务路由至专门的 Agent。" loading="lazy"></a><figcaption><span class="figure-number">图 18.6　</span>Supervisor 模式：一个编排者将任务路由至专门的 Agent。</figcaption></figure>

#### Peer-to-Peer（点对点）。 {#peer-to-peer.}

<p>Agent 之间直接通信，无中央协调者。每个 Agent 都可以将任何其他 Agent 作为工具调用。灵活但更难调试，且容易出现循环依赖。</p>

#### 分层（Tree of Agents）。 {#hierarchical-tree-of-agents.}

<p>一种树状结构，其中高层 Agent 委派给中层 Agent，中层再委派给叶子 Agent。可实现递归任务分解。AutoGen 的 nested chat 等系统采用此模式。</p>

#### Swarm 模式。 {#swarm-pattern.}

<p>该模式由 OpenAI 的 Swarm 库 <a class="book-citation" id="cite-openai2024swarm-1" href="../../references.md#ref-openai2024swarm" aria-label="参考文献 276">[276]</a> 推广，使用 <strong>handoff（交接）</strong>：一个 Agent 可以将控制权连同完整对话 Context 一起转交给另一个 Agent。核心概念：</p>

<ul>
<li><p><strong>Agent</strong> 拥有指令与工具。</p></li>
<li><p><strong>Handoff</strong> 是用于转移控制权的特殊工具。</p></li>
<li><p><strong>Context 变量</strong> 是 Agent 之间传递的共享状态。</p></li>
<li><p>活动 Agent 根据任务需要动态切换。</p></li>
</ul>

## 18.5.4　人机协同（Human-in-the-Loop） {#human-in-the-loop}

<p>生产级 Agent 必须知道何时暂停并请求人类输入：</p>

<ul>
<li><p><strong>审批门：</strong> 在不可逆操作（发送邮件、删除文件、付款购买）之前，要求显式的人类确认。</p></li>
<li><p><strong>升级标准：</strong> 当置信度低于阈值、任务超出已定义范围或触发安全规则时进行升级。</p></li>
<li><p><strong>反馈整合：</strong> 人类更正被插入 Context，并可以更新 Agent 的计划。</p></li>
<li><p><strong>异步审批：</strong> 对长时间任务，Agent 可以暂停，通过 email/Slack 通知人类，并在审批通过后继续。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">升级决策规则</p>


<div class="arithmatex">\[\text{Escalate} \iff \underbrace{p_{\text{success}} &lt; \tau_{\text{conf}}}_{\text{低置信度}} \;\lor\; \underbrace{\text{action} \in \mathcal{A}_{\text{irreversible}}}_{\text{不可逆}} \;\lor\; \underbrace{\text{cost} &gt; B_{\text{auto}}}_{\text{超出预算}}
\tag{18.12}\]</div>

<p>其中 <span class="arithmatex">\(\tau_{\text{conf}}\)</span> 是置信度阈值，<span class="arithmatex">\(\mathcal{A}_{\text{irreversible}}\)</span> 是不可逆操作集合，<span class="arithmatex">\(B_{\text{auto}}\)</span> 是自主消费上限。</p>
</div>

## 18.5.5　工作流图 {#workflow-graphs}

<p>对于复杂的、结构化的工作流，编排逻辑被表达为<strong>有向无环图（directed acyclic graph, DAG）</strong>或状态机：</p>

<ul>
<li><p><strong>LangGraph</strong> <a class="book-citation" id="cite-langchain2024langgraph-1" href="../../references.md#ref-langchain2024langgraph" aria-label="参考文献 154">[154]</a>：在 LangChain 之上扩展，提供基于图的执行模型。节点是 Agent 步骤；边是条件转移。支持环路（用于 ReAct 循环）和并行分支。</p></li>
<li><p><strong>AutoGen</strong> <a class="book-citation" id="cite-wu2023autogen-1" href="../../references.md#ref-wu2023autogen" aria-label="参考文献 385">[385]</a>：Microsoft 的多 Agent 对话图框架。支持嵌套对话、群聊以及人机协同模式。</p></li>
<li><p><strong>状态机：</strong> 显式状态（例如 <code>PLANNING</code>、<code>EXECUTING</code>、<code>WAITING_FOR_HUMAN</code>、<code>DONE</code>）与已定义的转移。比隐式循环逻辑更易推理和测试。</p></li>
</ul>



<div class="arithmatex">\[G = (V, E, \sigma_0), \quad v \in V: \text{agent step}, \quad e \in E: \text{conditional transition}, \quad \sigma_0: \text{initial state}
\tag{18.13}\]</div>

<figure id="figure-18-7"><a href="../../assets/figures/fig_059_workflow-graph.png" target="_blank" rel="noopener" aria-label="查看图 18.7 原图" title="点击查看原图"><img src="../../assets/figures/fig_059_workflow-graph.png" alt="一个人机协同 Agent 的示例工作流图。状态与条件转移都是显式的，使控制流可审计。" loading="lazy"></a><figcaption><span class="figure-number">图 18.7　</span>一个人机协同 Agent 的示例工作流图。状态与条件转移都是显式的，使控制流可审计。</figcaption></figure>
