# 28.26　多 Agent 系统问题 {#multi-agent-systems-questions}

<section class="quiz-question" id="q-source-97"><h2 class="question-title">Q：比较 LLM 多 Agent 系统的中心化 vs 去中心化架构</h2>
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

<section class="quiz-question" id="q-source-98"><h2 class="question-title">Q：什么是 CTDE，为什么它对训练多 Agent LLM 系统重要？</h2>
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
