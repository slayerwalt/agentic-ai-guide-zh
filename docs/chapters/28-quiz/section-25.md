# 28.25　Agent 通信（A2A）问题 {#agent-communication-a2a-questions}

<section class="quiz-question" id="q-source-95"><h2 class="question-title">Q：Google 的 A2A 协议与 MCP 有何不同，何时两者都需要？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>核心区分</strong>：</p>

<ul>
<li><p><strong>MCP</strong>：Agent <span class="arithmatex">\(\leftrightarrow\)</span> 工具（具有定义 schema 的结构化函数调用）</p></li>
<li><p><strong>A2A</strong>：Agent <span class="arithmatex">\(\leftrightarrow\)</span> Agent（不透明任务委派——你不知道对方 Agent 如何工作）</p></li>
</ul>

<p><strong>A2A 关键概念</strong>：</p>

<ul>
<li><p><strong>Agent Cards</strong>：描述 Agent 能力的 JSON（类似简历）。发现机制。</p></li>
<li><p><strong>不透明执行</strong>：请求者看不到受托方的内部推理。只是发送任务并获取结果。</p></li>
<li><p><strong>任务生命周期</strong>：submitted <span class="arithmatex">\(\to\)</span> working <span class="arithmatex">\(\to\)</span> completed/failed（通过 SSE 提供流式更新）</p></li>
</ul>

<p><strong>何时两者都需要</strong>：</p>

<ol>
<li><p>编排 Agent 用 <strong>A2A</strong> 将“研究这个话题”委派给一个研究 Agent</p></li>
<li><p>研究 Agent 用 <strong>MCP</strong> 调用网页搜索、文件读取和数据库工具</p></li>
<li><p>结果经 A2A 流回编排 Agent</p></li>
</ol>

<p><strong>架构</strong>：A2A 位于<em>Agent 间</em>层；MCP 位于<em>Agent–工具</em>层。完整系统两者并用：A2A 用于 Agent 间协调，MCP 用于每个 Agent 的工具访问。</p>

<p><em><strong>复习：</strong>第 <a href="../22-mcp/index.md">22</a> 章与第 <a href="../24-a2a/index.md">24</a> 章（MCP；Agent 到 Agent 通信）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-96"><h2 class="question-title">Q：什么是合同网协议（Contract Net Protocol），它如何应用于 LLM Agent？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>合同网协议</strong>（Contract Net Protocol，CNP）是来自分布式 AI 的任务分配机制：</p>

<p><strong>步骤</strong>：</p>

<ol>
<li><p><strong>宣告</strong>：管理者向所有可用 Agent 广播任务描述</p></li>
<li><p><strong>投标</strong>：Agent 评估自身能力并提交投标（置信度；预估成本；预估时间）</p></li>
<li><p><strong>授标</strong>：管理者按标准（能力/成本/可用性）选出最佳投标</p></li>
<li><p><strong>执行</strong>：中标 Agent 执行任务</p></li>
<li><p><strong>回报</strong>：Agent 将结果报告给管理者</p></li>
</ol>

<p><strong>对 LLM Agent 而言</strong>：</p>

<ul>
<li><p><strong>投标 = 自我评估</strong>：每个 Agent LLM 评估“我能把这个任务做好吗？”并给出置信度分数。这需要校准过的自我认知。</p></li>
<li><p><strong>专业化涌现</strong>：代码 Agent 在代码任务上出高价；研究 Agent 在研究任务上出高价。无需中心路由逻辑。</p></li>
<li><p><strong>负载均衡</strong>：若一个 Agent 忙（预估时间长），其他 Agent 胜出。</p></li>
<li><p><strong>失败处理</strong>：若中标 Agent 失败，则向剩余 Agent 重新宣告（自动故障切换）。</p></li>
</ul>

<p><strong>LLM 的局限</strong>：LLM 常常高估自身能力（幻觉置信度）。投标应纳入历史记录（相似任务的历史成功率），而不仅是自报置信度。</p>

<p><em><strong>复习：</strong>第 <a href="../24-a2a/index.md">24</a> 章与第 <a href="../25-multi-agent-systems/index.md">25</a> 章（A2A；多 Agent 系统）。</em></p>
</details></section>
