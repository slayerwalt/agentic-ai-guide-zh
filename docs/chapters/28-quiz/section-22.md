# 28.22　Agent 记忆问题 {#agentic-memory-questions}

<section class="quiz-question" id="q-source-87"><h2 class="question-title">Q：比较 Agent 的四种记忆类型，以及各自在何时至关重要</h2>
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

<section class="quiz-question" id="q-source-88"><h2 class="question-title">Q：记忆检索中的时间衰减如何工作，为何重要？</h2>
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

<section class="quiz-question" id="q-source-89"><h2 class="question-title">Q：如何用 RL 训练记忆操作？</h2>
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
