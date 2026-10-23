# 28.29　Agent UI 框架问题 {#agentic-ui-framework-questions}

<section class="quiz-question" id="q-source-103"><h2 class="question-title">Q：比较 Agent 的聊天式 vs 画布式 UI 范式</h2>
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

<section class="quiz-question" id="q-source-104"><h2 class="question-title">Q：如何为人机协同（Human-in-the-Loop）的 Agent 系统设计审批关卡？</h2>
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
