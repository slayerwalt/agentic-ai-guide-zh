# 28.23　Agent 编排问题 {#agent-orchestration-questions}

<section class="quiz-question" id="q-source-90"><h2 class="question-title">Q：解释上下文预算问题，以及如何用动态分配解决</h2>
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

<section class="quiz-question" id="q-source-91"><h2 class="question-title">Q：比较 ReAct 与 Plan-and-Execute 两种编排模式</h2>
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

<section class="quiz-question" id="q-source-92"><h2 class="question-title">Q：如何检测并防止 Agent 执行中的无限循环？</h2>
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
