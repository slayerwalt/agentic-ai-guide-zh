# 28.27　Agent 开发框架问题 {#agent-development-framework-questions}

<section class="quiz-question" id="q-source-99"><h2 class="question-title">Q：比较 LangGraph、AutoGen、CrewAI 用于构建多 Agent 系统</h2>
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

<section class="quiz-question" id="q-source-100"><h2 class="question-title">Q：如何在生产中测试和评估一个 Agent 系统？</h2>
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
