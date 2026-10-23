# 第 28 章 · Agent 系统与应用 {#reading-reading-05}

## 28.22　Agent 记忆问题 {#agentic-memory-questions}

<section class="quiz-question" id="q-source-87"><h3 class="question-title">Q：比较 Agent 的四种记忆类型，以及各自在何时至关重要</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<span id="table-28-inline-3"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>类型</strong></th><th scope="col"><strong>存储内容</strong></th><th scope="col"><strong>访问方式</strong></th><th scope="col"><strong>何时关键</strong></th></tr></thead>
<tbody><tr><td>工作记忆</td><td>当前上下文/草稿板</td><td>始终在上下文中</td><td>复杂多步推理</td></tr>
<tr><td>情景记忆</td><td>过往经验</td><td>按相似度检索</td><td>从过去错误中学习</td></tr>
<tr><td>语义记忆</td><td>事实与知识</td><td>按概念检索</td><td>领域特定任务</td></tr>
<tr><td>程序性记忆</td><td>技能与模式</td><td>按任务类型触发</td><td>重复性工具调用</td></tr></tbody>
</table>
</div>

<p><strong>关键洞见</strong>：它们并非独立——彼此交互。情景记忆喂养语义记忆（从事件泛化为事实）。程序性记忆被情景反馈细化（学习哪些工具序列有效）。工作记忆协调对所有其他类型的检索。</p>

<p><strong>MemGPT 类比</strong>：工作记忆 = 热（在上下文），情景/语义记忆 = 温（向量库），程序性记忆 = 冷（归档的 Policy）。Agent 自身决定何时换入/换出信息。</p>

<p><em><strong>复习：</strong>第 <a href="../17-memory/index.md">17</a> 章（Agent 记忆系统）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-88"><h3 class="question-title">Q：记忆检索中的时间衰减如何工作，为何重要？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>时间衰减</strong>在检索时给较旧记忆降权：</p>



<div class="arithmatex">\[\text{score}(m) = \alpha \cdot \text{similarity}(q, m) + (1 - \alpha) \cdot \text{recency}(m)\]</div>

<p>其中 <span class="arithmatex">\(\text{recency}(m) = e^{-\lambda \cdot \Delta t}\)</span>，<span class="arithmatex">\(\Delta t\)</span> = 自上次访问以来的时间。</p>

<p><strong>为什么重要</strong>：</p>

<ol>
<li><p><strong>相关性衰减</strong>：用户偏好会变化。6 个月前的偏好可能已过时。</p></li>
<li><p><strong>矛盾消解</strong>：当新旧信息冲突时，近期偏好自然倾向当前真相。</p></li>
<li><p><strong>检索效率</strong>：没有衰减时记忆无界增长，检索会返回越来越不相关的远古条目。</p></li>
<li><p><strong>认知合理性</strong>：人类也会遗忘——近期事件更易获取。这与间隔效应（spacing effect）相符。</p></li>
</ol>

<p><strong>基于访问的刷新</strong>：当记忆被检索并使用时，其时间戳更新（类似 LRU 缓存）。频繁访问的记忆无论创建日期如何都保持“新鲜”。</p>

<p><strong>衰减率调优</strong>：<span class="arithmatex">\(\lambda\)</span> 取决于领域。客户服务：高衰减（偏好变化快）。法律/医疗：低衰减（事实持久）。可通过 RL 学习。</p>

<p><em><strong>复习：</strong>第 <a href="../17-memory/index.md">17</a> 章（Agent 记忆系统）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-89"><h3 class="question-title">Q：如何用 RL 训练记忆操作？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：记忆操作（写入/读取/更新/删除）可作为 Agent MDP 中的动作：</p>

<p><strong>建模</strong>：</p>

<ul>
<li><p><strong>状态</strong>：当前上下文 + 记忆状态</p></li>
<li><p><strong>动作</strong>：标准动作 + <code>memory_write(key/value)</code>、<code>memory_read(query)</code>、<code>memory_delete(key)</code></p></li>
<li><p><strong>Reward</strong>：任务成功（记忆是否有帮助？）+ 记忆效率惩罚（读取越少越好）</p></li>
</ul>

<p><strong>RL 学到什么</strong>：</p>

<ol>
<li><p><strong>存什么</strong>：重要信息（API key / 用户偏好）vs 临时细节</p></li>
<li><p><strong>何时检索</strong>：回答领域问题前 vs 一般聊天中</p></li>
<li><p><strong>压缩策略</strong>：何时对旧记忆做摘要 vs 原样保留</p></li>
<li><p><strong>遗忘</strong>：何时旧信息陈旧应被删除</p></li>
</ol>

<p><strong>训练信号</strong>：反事实——“如果 Agent 没有存储/检索这条记忆，它还会成功吗？”通过轨迹比较实现：记忆使用得当的轨迹获得更高 Reward。</p>

<p><strong>挑战</strong>：延迟 Reward——现在存储的信息可能 100 步后才有用。需要长期信用分配（高 <span class="arithmatex">\(\lambda\)</span> 的 GAE）。</p>

<p><em><strong>复习：</strong>第 <a href="../12-agentic-training/index.md">12</a> 章与第 <a href="../17-memory/index.md">17</a> 章（LLM Agent 训练；Agent 记忆系统）。</em></p>
</details></section>

## 28.23　Agent 编排问题 {#agent-orchestration-questions}

<section class="quiz-question" id="q-source-90"><h3 class="question-title">Q：解释上下文预算问题，以及如何用动态分配解决</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：Agent 有上下文窗口 <span class="arithmatex">\(L\)</span> 个 token，但需要为以下部分留空间：</p>



<div class="arithmatex">\[C = S + M + T + H + R \leq L\]</div>

<p>其中 <span class="arithmatex">\(S\)</span> = 系统 Prompt，<span class="arithmatex">\(M\)</span> = 记忆/检索上下文，<span class="arithmatex">\(T\)</span> = 工具描述，<span class="arithmatex">\(H\)</span> = 对话历史，<span class="arithmatex">\(R\)</span> = 为响应预留。</p>

<p>随着对话增长，<span class="arithmatex">\(H\)</span> 增大并挤掉其他组件。</p>

<p><strong>动态分配策略</strong>：</p>

<ol>
<li><p><strong>固定下限</strong>：<span class="arithmatex">\(S_{\min}\)</span>、<span class="arithmatex">\(R_{\min}\)</span> 不可妥协</p></li>
<li><p><strong>自适应历史</strong>：当 <span class="arithmatex">\(H &gt; H_{\max}\)</span> 时摘要旧轮次。保留最近 <span class="arithmatex">\(k\)</span> 轮原样；其余做摘要。</p></li>
<li><p><strong>按需工具</strong>：仅包含与当前查询相关的工具描述（而非全部 50 个）。用分类器或 Embedding 相似度选 top-<span class="arithmatex">\(k\)</span> 工具。</p></li>
<li><p><strong>惰性记忆</strong>：仅在需要时（分析查询之后）检索记忆，而非预先加载。</p></li>
</ol>

<p><strong>溢出处理</strong>：当压缩后总量仍超过 <span class="arithmatex">\(L\)</span>：</p>

<ul>
<li><p>丢弃最不重要的工具描述</p></li>
<li><p>激进地把历史摘要到每轮一句</p></li>
<li><p>减少记忆槽</p></li>
<li><p>若仍超：截断并向用户提示</p></li>
</ul>

<p><strong>飞行前检查</strong>：调用 LLM 之前始终先统计 token 数。绝不要在推理时才发现溢出。</p>

<p><em><strong>复习：</strong>第 <a href="../18-agent-harness/index.md">18</a> 章（Agent Harness——上下文管理与编排）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-91"><h3 class="question-title">Q：比较 ReAct 与 Plan-and-Execute 两种编排模式</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>ReAct</strong>（Reason + Act）：</p>

<ul>
<li><p>循环：Thought <span class="arithmatex">\(\to\)</span> Action <span class="arithmatex">\(\to\)</span> Observation <span class="arithmatex">\(\to\)</span> Thought <span class="arithmatex">\(\to\)</span> …</p></li>
<li><p>每一步基于此前所有观察决定下一动作</p></li>
<li><p><strong>优点</strong>：自适应——可根据工具输出改变方向</p></li>
<li><p><strong>缺点</strong>：短视——无前置规划；可能陷入循环；每次 LLM 调用都看到完整历史（昂贵）</p></li>
</ul>

<p><strong>Plan-and-Execute</strong>：</p>

<ul>
<li><p>阶段 1：生成完整计划（步骤列表）</p></li>
<li><p>阶段 2：顺序执行步骤（更简单的执行器；可能用更便宜的模型）</p></li>
<li><p>阶段 3：执行失败则重新规划</p></li>
<li><p><strong>优点</strong>：高效（一次规划比每步推理便宜）；独立步骤可并行</p></li>
<li><p><strong>缺点</strong>：计划脆弱——若早期步骤失败，整个计划可能失效。重规划增加延迟。</p></li>
</ul>

<p><strong>何时选择</strong>：</p>

<ul>
<li><p>ReAct：探索性任务；未知环境；每步结果决定下一步的任务</p></li>
<li><p>Plan-and-Execute：定义清晰的任务；已知工具集；可并行子任务；成本敏感的部署</p></li>
<li><p><strong>混合</strong>：高层做规划，在每个规划步内用 ReAct（LangGraph 推荐模式）</p></li>
</ul>

<p><em><strong>复习：</strong>第 <a href="../18-agent-harness/index.md">18</a> 章与第 <a href="../20-agent-patterns/index.md">20</a> 章（Agent Harness；Agent 设计模式）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-92"><h3 class="question-title">Q：如何检测并防止 Agent 执行中的无限循环？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：当 Agent 重复同一动作期待不同结果时，会进入无限循环。</p>

<p><strong>检测方法</strong>：</p>

<ol>
<li><p><strong>最大迭代守卫</strong>：硬上限（例如 25 步）。简单但会在真正长任务上丢失进展。</p></li>
<li><p><strong>动作哈希窗口</strong>：对最近 <span class="arithmatex">\(k\)</span> 个（动作/观察）对求哈希。若当前哈希与最近 <span class="arithmatex">\(w\)</span> 步内的哈希匹配则检测到循环。</p></li>
<li><p><strong>语义相似度</strong>：对近期动作做 Embedding。若相邻动作余弦相似度超过阈值（<span class="arithmatex">\(&gt;\)</span>0.95）则可能卡住。</p></li>
<li><p><strong>进度监控</strong>：定义任务特定的进度指标。若 <span class="arithmatex">\(N\)</span> 步内无进展则干预。</p></li>
</ol>

<p><strong>恢复策略</strong>：</p>

<ol>
<li><p><strong>注入提示</strong>：加入系统消息：“你似乎在重复动作。尝试不同方法。”</p></li>
<li><p><strong>强制不同动作</strong>：下一步在动作空间中屏蔽重复动作。</p></li>
<li><p><strong>升级</strong>：返回部分结果给用户并请求指导。</p></li>
<li><p><strong>回溯</strong>：重置到循环开始前的 checkpoint 并尝试备选路径。</p></li>
</ol>

<p><strong>最佳实践</strong>：组合最大迭代（安全网）+ 基于哈希的检测（早期干预）+ 优雅升级（维护用户信任）。</p>

<p><em><strong>复习：</strong>第 <a href="../18-agent-harness/index.md">18</a> 章与第 <a href="../20-agent-patterns/index.md">20</a> 章（Agent Harness；Agent 设计模式）。</em></p>
</details></section>

## 28.24　MCP 协议问题 {#mcp-protocol-questions}

<section class="quiz-question" id="q-source-93"><h3 class="question-title">Q：解释 MCP 的 N+M 架构，以及它对 Agent 生态系统为何重要</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>N<span class="arithmatex">\(\times\)</span>M 问题</strong>：没有 MCP 时，<span class="arithmatex">\(N\)</span> 个 Agent 框架必须各自实现与 <span class="arithmatex">\(M\)</span> 个工具的集成 = 总共 <span class="arithmatex">\(N \times M\)</span> 次集成。新增一个工具需要 <span class="arithmatex">\(N\)</span> 次实现。</p>

<p><strong>MCP 的 N+M 方案</strong>：标准化接口。每个 Agent 实现一个 MCP 客户端（共 <span class="arithmatex">\(N\)</span> 个）。每个工具实现一个 MCP 服务器（共 <span class="arithmatex">\(M\)</span> 个）。总集成数 = <span class="arithmatex">\(N + M\)</span>。</p>

<p><strong>具体示例</strong>：5 个 Agent 框架（LangChain/AutoGen/CrewAI/Claude/custom）<span class="arithmatex">\(\times\)</span> 20 个工具（GitHub/Slack/DB/filesystem/…）= 没有 MCP 时 100 次集成。有 MCP 时：5 个客户端 + 20 个服务器 = 25 次实现。</p>

<p><strong>为何重要</strong>：</p>

<ol>
<li><p><strong>工具复用</strong>：构建一次工具服务器；可被任何兼容 MCP 的 Agent 使用</p></li>
<li><p><strong>Agent 可移植性</strong>：从 Claude 切换到自定义 Agent 而无需重写工具集成</p></li>
<li><p><strong>生态成长</strong>：降低新增工具的门槛激励社区构建更多</p></li>
<li><p><strong>可组合性</strong>：运行时将多个服务器动态连接到一个 Agent</p></li>
</ol>

<p><strong>类比</strong>：USB 标准化了外设连接。USB 之前：每种设备都有专有接口。USB 之后：一个口适配所有。MCP 对 Agent–工具连接做了同样的事。</p>

<p><em><strong>复习：</strong>第 <a href="../22-mcp/index.md">22</a> 章（模型上下文协议）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-94"><h3 class="question-title">Q：MCP 的四个核心原语是什么，何时分别使用？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<span id="table-28-inline-4"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>原语</strong></th><th scope="col"><strong>方向</strong></th><th scope="col"><strong>用途</strong></th><th scope="col"><strong>示例</strong></th></tr></thead>
<tbody><tr><td>Tools</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务器</td><td>执行动作</td><td><code>create_issue</code>；<code>query_db</code></td></tr>
<tr><td>Resources</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务器</td><td>读取数据</td><td>文件内容；数据库记录</td></tr>
<tr><td>Prompts</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务器</td><td>获取模板</td><td>“总结这个 PR”模板</td></tr>
<tr><td>Sampling</td><td>服务器 <span class="arithmatex">\(\to\)</span> 客户端</td><td>请求 LLM 生成</td><td>服务器请求 LLM 分类</td></tr></tbody>
</table>
</div>

<p><strong>关键区分</strong>：</p>

<ul>
<li><p><strong>Tools vs Resources</strong>：Tools 有<em>副作用</em>（创建/修改/删除）。Resources 是<em>只读</em>的。这对安全很重要——Agent 可自由读取 Resources，但调用 Tools 必须获得批准。</p></li>
<li><p><strong>Sampling</strong> 方向反转：通常是客户端（Agent）调用服务器（工具）。在 Sampling 中，服务器请求客户端的 LLM 帮忙。用例：代码分析服务器需要 LLM 解释一段代码。</p></li>
<li><p><strong>Prompts</strong> 是元数据（可复用模板），不是执行。它们帮助 Agent 构造更好的工具调用。</p></li>
</ul>

<p><em><strong>复习：</strong>第 <a href="../22-mcp/index.md">22</a> 章（模型上下文协议）。</em></p>
</details></section>

## 28.25　Agent 通信（A2A）问题 {#agent-communication-a2a-questions}

<section class="quiz-question" id="q-source-95"><h3 class="question-title">Q：Google 的 A2A 协议与 MCP 有何不同，何时两者都需要？</h3>
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

<section class="quiz-question" id="q-source-96"><h3 class="question-title">Q：什么是合同网协议（Contract Net Protocol），它如何应用于 LLM Agent？</h3>
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

## 28.26　多 Agent 系统问题 {#multi-agent-systems-questions}

<section class="quiz-question" id="q-source-97"><h3 class="question-title">Q：比较 LLM 多 Agent 系统的中心化 vs 去中心化架构</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>中心化（Supervisor）</strong>：</p>

<ul>
<li><p>一个编排 LLM 将任务路由给专家 worker</p></li>
<li><p>控制流清晰；易于调试（检查 supervisor 决策）</p></li>
<li><p>单点故障；supervisor 成为 token 瓶颈</p></li>
<li><p>最适合：定义清晰的工作流；小型 Agent 团队（3–5 个 Agent）</p></li>
</ul>

<p><strong>去中心化（点对点）</strong>：</p>

<ul>
<li><p>Agent 直接通信；无中央协调者</p></li>
<li><p>弹性（无单点故障）；可水平扩展</p></li>
<li><p>难以调试（涌现行为）；可能出现冲突和死锁</p></li>
<li><p>通信无结构时按 <span class="arithmatex">\(O(n^2)\)</span> 扩展</p></li>
<li><p>最适合：弹性系统；大量 Agent 群体；期望涌现行为的创造性任务</p></li>
</ul>

<p><strong>混合（层次化）</strong>：带子管理者的树形结构。结合优点：组内本地自治 + 顶层全局协调。通信按 <span class="arithmatex">\(O(n \log n)\)</span> 扩展。</p>

<p><strong>决策框架</strong>：需要可预测性和可审计性时用中心化。需要弹性和创造性时用去中心化。大型（<span class="arithmatex">\(&gt;\)</span>10 个 Agent）系统用层次化。</p>

<p><em><strong>复习：</strong>第 <a href="../25-multi-agent-systems/index.md">25</a> 章（多 Agent 系统）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-98"><h3 class="question-title">Q：什么是 CTDE，为什么它对训练多 Agent LLM 系统重要？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>CTDE</strong> = 中心化训练；去中心化执行（Centralized Training; Decentralized Execution）。</p>

<p><strong>问题</strong>：多 Agent RL 中，每个 Agent 的环境都是非平稳的（其他 Agent 同时在改变各自的 Policy）。这使独立训练不稳定。</p>

<p><strong>CTDE 方案</strong>：</p>

<ul>
<li><p><strong>训练时</strong>：一个中心化 critic 可访问所有 Agent 的观察和动作：<span class="arithmatex">\(V(s_1, s_2, \ldots, s_n, a_1, a_2, \ldots, a_n)\)</span>。这通过把非平稳性从 Value 函数中剔除来稳定训练。</p></li>
<li><p><strong>执行时</strong>：每个 Agent 仅基于自身观察行动：<span class="arithmatex">\(a_i = \pi_i(o_i)\)</span>。推理时没有通信开销。</p></li>
</ul>

<p><strong>对 LLM Agent 而言</strong>：中心化 critic 可以是一个 Reward 模型，评估所有 Agent 的<em>联合</em>输出（例如，Agent 团队是否产出了一个正确的软件系统？），而每个 Agent 通过反事实信用分配被训练以最大化其对团队 Reward 的贡献。</p>

<p><strong>实际挑战</strong>：完整 CTDE 要求所有 Agent 在共享状态下同时训练——对 LLM 而言代价昂贵。近似方法：分轮训练 Agent（冻结其余 Agent 训练一个），或使用具有周期同步的种群训练。</p>

<p><em><strong>复习：</strong>第 <a href="../25-multi-agent-systems/index.md">25</a> 章（多 Agent 系统）。</em></p>
</details></section>

## 28.27　Agent 开发框架问题 {#agent-development-framework-questions}

<section class="quiz-question" id="q-source-99"><h3 class="question-title">Q：比较 LangGraph、AutoGen、CrewAI 用于构建多 Agent 系统</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<span id="table-28-inline-5"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>LangGraph</strong></th><th scope="col"><strong>AutoGen / CrewAI</strong></th></tr></thead>
<tbody><tr><td>编排</td><td>显式状态图（节点 + 边）</td><td>隐式（基于对话/角色）</td></tr>
<tr><td>状态管理</td><td>TypedDict schema；checkpoint</td><td>以对话历史为状态</td></tr>
<tr><td>多 Agent</td><td>带条件路由的图</td><td>GroupChat / Crew</td></tr>
<tr><td>调试</td><td>图可视化；步骤回放</td><td>聊天日志</td></tr>
<tr><td>人机协同（Human-in-the-Loop）</td><td>一等公民（中断节点）</td><td>通过审批工具</td></tr>
<tr><td>生产部署</td><td>LangGraph Cloud；持久化</td><td>有限（AutoGen）；成长中（CrewAI）</td></tr>
<tr><td>学习曲线</td><td>高（图概念）</td><td>低（AutoGen）；极低（CrewAI）</td></tr></tbody>
</table>
</div>

<p><strong>选 LangGraph 当</strong>：需要细粒度控制；复杂条件流；带持久化和人机协同的生产部署。</p>

<p><strong>选 AutoGen 当</strong>：多 Agent 对话的快速原型；代码执行 Agent；研究实验。</p>

<p><strong>选 CrewAI 当</strong>：简单的基于角色的团队；顺序任务执行；快速演示；代码量最小。</p>

<p><strong>都不选（自定义）当</strong>：需要最大性能/控制；不希望被框架锁定；或有非标准的编排模式。</p>

<p><em><strong>复习：</strong>第 <a href="../26-development-frameworks/index.md">26</a> 章（Agent 开发框架）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-100"><h3 class="question-title">Q：如何在生产中测试和评估一个 Agent 系统？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：Agent 测试遵循一个<strong>测试金字塔</strong>：</p>

<p><strong>第 1 层——单元测试</strong>（快；多）：</p>

<ul>
<li><p>独立测试单个工具（mock LLM；验证工具逻辑）</p></li>
<li><p>测试 Prompt 模板（给定上下文；验证正确的 Prompt 构造）</p></li>
<li><p>测试解析器（给定 LLM 输出；验证正确提取）</p></li>
</ul>

<p><strong>第 2 层——集成测试</strong>（中等速度）：</p>

<ul>
<li><p>用确定性输入测试完整的 Agent 循环</p></li>
<li><p>“黄金轨迹”测试：必须能复现的已知良好执行轨迹</p></li>
<li><p>工具链测试：验证多工具序列端到端可工作</p></li>
</ul>

<p><strong>第 3 层——行为测试</strong>（慢；少）：</p>

<ul>
<li><p>Agent 是否遵守安全约束？（对抗性输入）</p></li>
<li><p>它是否在合适时请求澄清？</p></li>
<li><p>它是否保持在 token/成本预算内？</p></li>
</ul>

<p><strong>生产评估</strong>：</p>

<ul>
<li><p><strong>A/B 测试</strong>：将 5% 流量路由到新 Agent 版本</p></li>
<li><p><strong>影子模式</strong>：与旧 Agent 并行运行新 Agent；比较输出但不对外服务</p></li>
<li><p><strong>LLM-as-judge</strong>：自动化打分评估 Agent 响应质量</p></li>
<li><p><strong>用户满意度</strong>：点赞/点踩；任务完成率；解决时间</p></li>
</ul>

<p><strong>关键指标</strong>：<strong>任务成功率</strong>（Task Success Rate, TSR）——Agent 在无人干预下正确完成任务的比例。</p>

<p><em><strong>复习：</strong>第 <a href="../14-evaluation/index.md">14</a> 章与第 <a href="../26-development-frameworks/index.md">26</a> 章（LLM 评估；Agent 开发框架）。</em></p>
</details></section>

## 28.28　Agent 环境问题 {#agentic-environments-questions}

<section class="quiz-question" id="q-source-101"><h3 class="question-title">Q：为网页浏览 Agent 环境设计一个 Reward 函数</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：对于 WebArena 风格任务（例如“找到 12 月 15 日从 NYC 到 SF 的最便宜航班”）：</p>

<p><strong>稀疏 Reward</strong>（简单但难以学习）：</p>



<div class="arithmatex">\[r = \begin{cases} 1 &amp; \text{if final page/state matches ground truth} \\ 0 &amp; \text{otherwise} \end{cases}\]</div>

<p><strong>密集 Reward</strong>（更利于训练；设计更难）：</p>

<ol>
<li><p><strong>进度 Reward</strong>：每个让 Agent 更接近目标的页面 <span class="arithmatex">\(+0.1\)</span>（用与目标状态的文本相似度衡量）</p></li>
<li><p><strong>效率惩罚</strong>：每个动作 <span class="arithmatex">\(-0.01\)</span>（鼓励更短轨迹）</p></li>
<li><p><strong>里程碑 Reward</strong>：到达中间目标（如导航到航班搜索页）<span class="arithmatex">\(+0.3\)</span></p></li>
<li><p><strong>无效动作惩罚</strong>：产生错误的动作（404；表单校验失败）<span class="arithmatex">\(-0.05\)</span></p></li>
</ol>

<p><strong>基于势能的 reward shaping</strong>（保留最优 Policy）：</p>



<div class="arithmatex">\[r_{\text{shaped}}(s, a, s&#x27;) = r(s, a, s&#x27;) + \gamma \Phi(s&#x27;) - \Phi(s)\]</div>

<p>其中 <span class="arithmatex">\(\Phi(s) = -\text{min\_steps\_to\_goal}(s)\)</span>（由启发式或学习到的 Value 函数估计）。</p>

<p><strong>挑战</strong>：部分可观察（无法总是判断是否更接近目标）；随机环境（页面内容会变）；Reward hacking（Agent 找到满足 Reward 但不满足用户意图的捷径）。</p>

<p><em><strong>复习：</strong>第 <a href="../12-agentic-training/index.md">12</a> 章与第 <a href="../21-environments/index.md">21</a> 章（LLM Agent 训练；Agent 环境）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-102"><h3 class="question-title">Q：是什么让 SWE-bench 成为一个特别具有挑战性的 Agent 基准？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：SWE-bench 在来自流行 Python 仓库的真实 GitHub issue 上测试 Agent：</p>

<p><strong>为什么难</strong>：</p>

<ol>
<li><p><strong>仓库级上下文</strong>：Agent 必须理解 10 万+ 行的代码库。无法装入上下文窗口——必须探索、搜索和导航。</p></li>
<li><p><strong>规范不充分的任务</strong>：Issue 由人类带着隐含上下文写就。Agent 必须推断真正需要什么。</p></li>
<li><p><strong>跨文件编辑</strong>：解决方案通常跨越多个文件，并有级联依赖。</p></li>
<li><p><strong>测试验证</strong>：必须通过现有测试 <em>以及</em>验证修复的新测试。</p></li>
<li><p><strong>无人辅助</strong>：不像 HumanEval（单函数），SWE-bench 需要完整的软件工程工作流：读 issue <span class="arithmatex">\(\to\)</span> 探索代码 <span class="arithmatex">\(\to\)</span> 定位 bug <span class="arithmatex">\(\to\)</span> 实现修复 <span class="arithmatex">\(\to\)</span> 验证。</p></li>
</ol>

<p><strong>当前 SOTA</strong>（2024–2025）：最佳 Agent 解决 SWE-bench Verified（精选子集）的约 <span class="arithmatex">\(\sim\)</span>50%。完整 SWE-bench：约 <span class="arithmatex">\(\sim\)</span>30%。</p>

<p><strong>对训练的关键洞见</strong>：SWE-bench 揭示了“编程能力”（编写正确函数）与“软件工程能力”（理解系统；导航代码库；做最小改动）之间的差距。在 SWE-bench 风格环境上的 RL 训练教会 Agent 探索和规划策略，而不仅是代码生成。</p>

<p><em><strong>复习：</strong>第 <a href="../21-environments/index.md">21</a> 章（Agent 环境与基准）。</em></p>
</details></section>

## 28.29　Agent UI 框架问题 {#agentic-ui-framework-questions}

<section class="quiz-question" id="q-source-103"><h3 class="question-title">Q：比较 Agent 的聊天式 vs 画布式 UI 范式</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>聊天式</strong>（ChatGPT；Claude 默认）：</p>

<ul>
<li><p>线性消息流：用户 <span class="arithmatex">\(\to\)</span> 助手 <span class="arithmatex">\(\to\)</span> 用户 <span class="arithmatex">\(\to\)</span> …</p></li>
<li><p><strong>优点</strong>：UX 熟悉；适合探索和问答；易于实现</p></li>
<li><p><strong>缺点</strong>：生成的产物（代码/文档）埋没在对话中。难以针对特定产物迭代。长对话中上下文易丢失。</p></li>
</ul>

<p><strong>画布/Artifact 式</strong>（Claude Artifacts；ChatGPT Canvas；Cursor）：</p>

<ul>
<li><p>侧边栏展示生成内容；聊天栏发送指令</p></li>
<li><p>Agent 可以对持久化的产物创建、编辑和迭代</p></li>
<li><p><strong>优点</strong>：产物独立于聊天持久化。用户可直接编辑。具有版本历史。</p></li>
<li><p><strong>缺点</strong>：UI 更复杂；需要 artifact 类型检测；对两个面板的流式输出实现更难。</p></li>
</ul>

<p><strong>何时选择</strong>：</p>

<ul>
<li><p>聊天：头脑风暴；问答；快速任务；移动端界面</p></li>
<li><p>画布：代码生成；文档写作；数据分析——任何需要迭代、有持久化输出的任务</p></li>
<li><p><strong>混合</strong>（多数现代 UI）：默认聊天；检测到代码/文档/可视化输出时自动提升到画布</p></li>
</ul>

<p><strong>对 Agent 训练而言</strong>：UI 范式影响 Reward 信号。画布式 UI 提供显式的编辑反馈（用户修改 artifact），可用于在线学习。</p>

<p><em><strong>复习：</strong>第 <a href="../27-agentic-ui/index.md">27</a> 章（Agent UI 框架）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-104"><h3 class="question-title">Q：如何为人机协同（Human-in-the-Loop）的 Agent 系统设计审批关卡？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：审批关卡在关键点暂停 Agent 执行以供人工审核。</p>

<p><strong>三层模型</strong>：</p>

<ol>
<li><p><strong>自动批准</strong>（无关卡）：安全可逆动作。读取操作；搜索；计算。</p></li>
<li><p><strong>通知</strong>（软关卡）：可能有影响但可恢复。发送邮件；创建草稿；修改文件。Agent 继续，但用户被通知，可撤销。</p></li>
<li><p><strong>阻断</strong>（硬关卡）：不可逆或高风险。删数据；汇款；发布内容；执行有副作用的代码。Agent 必须等待显式批准。</p></li>
</ol>

<p><strong>设计原则</strong>：</p>

<ul>
<li><p><strong>最小化打断</strong>：关卡过多 = 用户放弃 Agent。三层模型让大多数动作通行，同时拦截危险动作。</p></li>
<li><p><strong>显示上下文</strong>：审批关卡展示：什么动作；为什么（Agent 的推理）；将改变什么；如何撤销。</p></li>
<li><p><strong>批量审批</strong>：如果 Agent 需要 5 次文件写入，一起呈现而不是逐个。</p></li>
<li><p><strong>超时处理</strong>：如果用户在 <span class="arithmatex">\(T\)</span> 分钟内未响应，则重试通知、按安全默认继续或优雅中止。</p></li>
<li><p><strong>从审批中学习</strong>：跟踪审批/拒绝模式。若用户总是批准某类动作，考虑自动提升其级别。</p></li>
</ul>

<p><strong>实现</strong>：工具注解（MCP 的 <code>destructiveHint</code> 和 <code>readOnlyHint</code>）驱动关卡的自动分配。自定义规则可根据上下文覆盖。</p>

<p><em><strong>复习：</strong>第 <a href="../18-agent-harness/index.md">18</a> 章与第 <a href="../27-agentic-ui/index.md">27</a> 章（Agent Harness；Agent UI 框架）。</em></p>
</details></section>

## 28.30　RAG 与 Agent 化 RAG 问题 {#rag-and-agentic-rag-questions}

<section class="quiz-question" id="q-source-105"><h3 class="question-title">Q：解释互逆排名融合（Reciprocal Rank Fusion, RRF），以及它为何对混合检索有效</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：RRF 合并多个检索系统的排名，无需对分数做校准：</p>



<div class="arithmatex">\[\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}\]</div>

<p>其中 <span class="arithmatex">\(r(d)\)</span> 是文档 <span class="arithmatex">\(d\)</span> 在检索器 <span class="arithmatex">\(r\)</span> 中的排名，<span class="arithmatex">\(k=60\)</span> 是防止高排名文档主导的常数。</p>

<p><strong>为何有效</strong>：</p>

<ol>
<li><p><strong>无需分数归一化</strong>：BM25 分数无界；稠密相似度在 <span class="arithmatex">\([-1, 1]\)</span>。RRF 只使用排名，使二者可直接比较。</p></li>
<li><p><strong>对离群值鲁棒</strong>：单个检索器给出异常高分也不会主导，因为即便是排名 1，<span class="arithmatex">\(1/(k+1) \approx 0.016\)</span>。</p></li>
<li><p><strong>互补信号</strong>：BM25 捕捉精确关键词匹配；稠密检索捕捉语义相似度。两者都排名高的文档得到加成。</p></li>
</ol>

<p><strong>示例</strong>：文档 <span class="arithmatex">\(d\)</span> 在 BM25 中排名 3，在稠密中排名 7。RRF 分数 <span class="arithmatex">\(= 1/(60+3) + 1/(60+7) = 0.0159 + 0.0149 = 0.0308\)</span>。在某一边排名 1 但在另一边排名 100 的文档得到 <span class="arithmatex">\(1/61 + 1/160 = 0.0226\)</span>——尽管有排名 1，但更低。</p>

<p><strong>实践中</strong>：混合（BM25 + 稠密 + RRF）在 85%+ 的基准上优于单独使用任一种。</p>

<p><em><strong>复习：</strong>第 <a href="../16-rag/index.md">16</a> 章（检索增强生成）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-106"><h3 class="question-title">Q：什么是 Agent 化 RAG，它与标准 RAG 有何不同？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>标准 RAG</strong> 遵循固定流水线：查询 <span class="arithmatex">\(\to\)</span> 检索 <span class="arithmatex">\(\to\)</span> 生成。它没有能力：</p>

<ul>
<li><p>判断是否根本需要检索</p></li>
<li><p>评估检索到的文档是否足够</p></li>
<li><p>在检索失败时重新表述查询</p></li>
<li><p>整合多步检索的信息</p></li>
</ul>

<p><strong>Agent 化 RAG</strong> 将检索视为 Agent MDP 中的一个<em>动作</em>：</p>

<ul>
<li><p><strong>是否检索决策</strong>：Agent 评估它是否已经知道答案（对训练数据内的事实性问题跳过检索）</p></li>
<li><p><strong>查询规划</strong>：把复杂问题分解为子查询（“X 发生在哪一年？” + “那时谁是总统？”）</p></li>
<li><p><strong>自我评估</strong>：检索后评估相关性。若不足，则重新表述查询或尝试不同来源。</p></li>
<li><p><strong>多跳推理</strong>：检索 <span class="arithmatex">\(\to\)</span> 推理 <span class="arithmatex">\(\to\)</span> 识别知识缺口 <span class="arithmatex">\(\to\)</span> 再次检索</p></li>
<li><p><strong>来源路由</strong>：将查询路由到合适的知识库（时事查网络；公司信息查内部文档；编程查代码搜索）</p></li>
</ul>

<p><strong>关键架构差异</strong>：标准 RAG = 确定性流水线。Agent 化 RAG = 带条件转移的状态机（LangGraph 的 retrieve/grade/rewrite/generate 节点模式）。</p>

<p><strong>权衡</strong>：Agent 化 RAG 在复杂查询上更准确，但增加延迟（多次 LLM 调用用于路由/评分）。简单事实查询用标准 RAG；多跳或歧义查询用 Agent 化 RAG。</p>

<p><em><strong>复习：</strong>第 <a href="../16-rag/index.md">16</a> 章与第 <a href="../18-agent-harness/index.md">18</a> 章（RAG；Agent Harness）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-107"><h3 class="question-title">Q：比较 Self-RAG 与 CRAG 两种改进检索质量的方法</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>Self-RAG</strong>（Asai 等，2023）：</p>

<ul>
<li><p>将特殊的<em>反思 token</em>训练进 LLM 词表</p></li>
<li><p>推理时，模型输出 [Retrieve]、[IsRel]、[IsSup]、[IsUse] 等 token</p></li>
<li><p>模型决定<em>何时</em>检索（并非每个查询都需要）</p></li>
<li><p>检索后，模型自我评分：检索段落是否相关？我的答案是否从中得出？</p></li>
<li><p><strong>训练</strong>：在用 GPT-4 反思标签增强的数据上做 SFT</p></li>
<li><p><strong>优点</strong>：单一模型处理一切。<strong>缺点</strong>：需要定制训练。</p></li>
</ul>

<p><strong>CRAG</strong>（纠正式 RAG，Corrective RAG，Yan 等，2024）：</p>

<ul>
<li><p>使用一个轻量的<em>检索评估器</em>（独立模型）来给检索到的文档评分</p></li>
<li><p>根据置信度有三种动作：<code>Correct</code>（按原样使用）、<code>Ambiguous</code>（用网络搜索增强）、<code>Incorrect</code>（丢弃；回退到网络）</p></li>
<li><p>增加<em>知识精炼</em>步骤：仅从检索文档中抽取相关句子</p></li>
<li><p><strong>优点</strong>：可与任何冻结 LLM 配合。<strong>缺点</strong>：需要额外的评估模型；增加延迟。</p></li>
</ul>

<p><strong>关键差异</strong>：Self-RAG 把检索决策嵌入 LLM 自身（需要训练）。CRAG 是包裹在任意 LLM 周围的流水线方法（无需训练）。Self-RAG 更优雅；CRAG 在现有模型的生产环境中更实用。</p>

<p><em><strong>复习：</strong>第 <a href="../16-rag/index.md">16</a> 章（检索增强生成）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-108"><h3 class="question-title">Q：什么是“中间迷失”（lost-in-the-middle）问题，如何缓解？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：当检索到的上下文很长（很多段落）时，LLM 不成比例地关注上下文的<em>开头</em>和<em>结尾</em>，而忽略中间的信息。如果答案在 10 段中的第 5 段，模型可能漏掉。</p>

<p><strong>经验证据</strong>：Liu 等（2023）表明，对于 20 文档检索，相关文档位于位置 5–15 时，相比位置 1–3，准确率下降 15%–20%。</p>

<p><strong>缓解策略</strong>：</p>

<ol>
<li><p><strong>重排序并截断</strong>：用 cross-encoder 重排序，然后只保留 top-3 最相关段落（更少 = 更少中间迷失）。</p></li>
<li><p><strong>策略性排序</strong>：把最相关的段落同时放在上下文的开头和结尾，低相关的放中间。</p></li>
<li><p><strong>上下文压缩</strong>：插入前将每段摘要到 1–2 句。文本越少 = 位置偏倚越小。</p></li>
<li><p><strong>Map-Reduce</strong>：独立处理每段（map），再合并答案（reduce）。完全消除位置效应。</p></li>
<li><p><strong>引用 Prompt</strong>：要求模型引用其使用的段落。这迫使 Attention 关注所有段落。</p></li>
<li><p><strong>减小块大小</strong>：更小的块意味着覆盖答案所需的总块数更少。</p></li>
</ol>

<p><strong>最佳实践</strong>：检索很多（20+），重排序到 top 3–5，按相关性排序（最佳在前）。这在大多数用例中完全绕开此问题。</p>

<p><em><strong>复习：</strong>第 <a href="../16-rag/index.md">16</a> 章（检索增强生成）。</em></p>
</details></section>
