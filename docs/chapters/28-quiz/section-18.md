# 28.18　Agent 强化学习问题 {#agentic-rl-questions}

<section class="quiz-question" id="q-source-72"><h2 class="question-title">Q：为什么标准 RLHF（单轮 PPO/DPO）对多步 Agent 失败？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：标准 RLHF 优化单轮质量：给定 Prompt，生成一个好响应。多步 Agent 面临根本不同的挑战：</p>

<ol>
<li><p><strong>信用分配</strong>：在 50 步轨迹中，哪一步导致失败？单轮 Reward 把信用均匀地分配给整个响应；多步需要<em>每一步</em>的信用。</p></li>
<li><p><strong>稀疏 Reward</strong>：仅在轨迹末尾才有成功/失败。PPO 的 GAE 假设有中间 Reward；缺失时，优势估计会有噪声。</p></li>
<li><p><strong>动作空间</strong>：动作是结构化的工具调用（JSON），不仅是 token 序列。模型必须同时学习语法 + 语义 + 策略。</p></li>
<li><p><strong>非平稳性</strong>：每个动作都会改变环境（工具输出修改状态）。每一步都有不同的&quot;Prompt&quot;，不像单轮那样输入是固定的。</p></li>
<li><p><strong>探索</strong>：Agent 必须发现新颖的工具使用策略，而不仅仅是改写文本。</p></li>
</ol>

<p><strong>解决方案</strong>：轨迹级 GRPO（对完整轨迹排序）、过程奖励模型（Process Reward Model, PRM）（每一步反馈）、或对成功轨迹做过滤后的 SFT。</p>

<p><em><strong>复习：</strong>第 <a href="../12-agentic-training/index.md">12</a> 章（LLM Agent 训练）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-73"><h2 class="question-title">Q：解释 GRPO 如何适配到 Agent 训练。与单轮 GRPO 的关键区别是什么？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：单轮 GRPO：对一个 Prompt 生成 <span class="arithmatex">\(N\)</span> 个响应，按 Reward 排序，计算优势。</p>

<p><strong>Agent GRPO</strong> 的区别：</p>

<ol>
<li><p><strong>生成单位</strong>：完整的<em>轨迹</em>（10–100 步），而非单个响应。每条轨迹是组内的一个“样本”。</p></li>
<li><p><strong>Reward</strong>：终止 Reward（任务成功/失败）或轨迹级 Reward（步 Reward 求和）。<em>不是</em>逐 token 的。</p></li>
<li><p><strong>掩码</strong>：仅在 Agent 的输出（推理 + 工具调用）上计算 Policy Loss。把工具<em>输出</em>（环境响应）从 Gradient 计算中掩掉。</p></li>
<li><p><strong>组大小</strong>：通常较小（<span class="arithmatex">\(N=4\)</span>–8），因为轨迹昂贵（每条轨迹很多次前向）。</p></li>
<li><p><strong>KL 惩罚</strong>：在每一步应用，防止每个决策点偏离 SFT Policy。</p></li>
<li><p><strong>长度归一化</strong>：按 Agent <em>动作</em>数（不是 token 数）归一化，避免惩罚充分的推理。</p></li>
</ol>

<p><em><strong>复习：</strong>第 <a href="../07-grpo/index.md">7</a> 章与第 <a href="../12-agentic-training/index.md">12</a> 章（GRPO；LLM Agent 训练）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-74"><h2 class="question-title">Q：比较 Agent 场景下的 STaR、Reflexion 和 ReAct。</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>STaR（自学推理器，Self-Taught Reasoner）</strong>：生成推理链 <span class="arithmatex">\(\to\)</span> 按正确性筛选 <span class="arithmatex">\(\to\)</span> 在正确链上微调。<em>适用</em>：你有可验证任务（数学、代码），希望在不使用 RL 的情况下从基模引导出推理能力。</p>

<p><strong>Reflexion</strong>：失败后生成自然语言反馈（“哪里出错了？”）<span class="arithmatex">\(\to\)</span> 把反思放进上下文重试。不更新权重。<em>适用</em>：推理时改进；训练算力有限；任务允许自我诊断。</p>

<p><strong>ReAct</strong>：在结构化循环中交替进行 Reasoning（思考）+ Acting（工具调用）。<em>适用</em>：多步工具调用任务；需要透明性（推理轨迹可解释）；Agent 必须在思考与行动之间抉择。</p>

<p><strong>关键差异</strong>：</p>

<span id="table-28-inline-2"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"></th><th scope="col"><strong>STaR</strong></th><th scope="col"><strong>Reflexion</strong></th><th scope="col"><strong>ReAct</strong></th></tr></thead>
<tbody><tr><td>是否更新权重？</td><td>是 (SFT)</td><td>否（上下文内）</td><td>否（Prompt）</td></tr>
<tr><td>多步？</td><td>否（单条推理链）</td><td>是（重试循环）</td><td>是（思考—行动循环）</td></tr>
<tr><td>工具？</td><td>否</td><td>可选</td><td>是（必需）</td></tr>
<tr><td>最适合</td><td>推理能力提升</td><td>错误恢复</td><td>工具增强任务</td></tr></tbody>
</table>
</div>

<p><em><strong>复习：</strong>第 <a href="../12-agentic-training/index.md">12</a> 章与第 <a href="../20-agent-patterns/index.md">20</a> 章（LLM Agent 训练；Agent 设计模式）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-75"><h2 class="question-title">Q：为什么研究型 Agent 偏好 GRPO 而非 PPO？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：对于具有 20–100 步轨迹的研究型 Agent：</p>

<p><strong>PPO 需要一个 Value 模型</strong>：<span class="arithmatex">\(V(s_t)\)</span> 必须预测从当前状态出发的期望总 Reward。对于研究场景（其中状态 = 128K token 的上下文，包括论文、代码和结果），训练一个精确的 Value 函数极其困难——“读过 3 篇论文并写了一部分代码”的价值难以预测。</p>

<p><strong>GRPO 完全避免了 Value 估计</strong>：它对每个研究问题生成 <span class="arithmatex">\(N\)</span> 条完整轨迹，并把组内排名作为优势。无需预测中间价值——只需比较结果。</p>

<p><strong>其他原因</strong>：</p>

<ul>
<li><p>研究质量近似二元（好报告 vs. 差报告）——排序很自然。</p></li>
<li><p>轨迹长且昂贵；GRPO 的 <span class="arithmatex">\(N=4\)</span> 可承受；PPO 需要很多 Rollout 才能得到稳定的 Value 估计。</p></li>
<li><p>终止 Reward 稀疏；稀疏 Reward 下 GAE 给出的每步优势本来就很有噪声。</p></li>
</ul>

<p><em><strong>复习：</strong>第 <a href="../07-grpo/index.md">7</a> 章与第 <a href="../12-agentic-training/index.md">12</a> 章（GRPO；LLM Agent 训练）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-76"><h2 class="question-title">Q：为编程 Agent 设计一个 Reward 函数。存在哪些 Reward hacking 风险？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>Reward 设计</strong>：</p>



<div class="arithmatex">\[R = 0.5 \cdot R_{\text{tests}} + 0.2 \cdot R_{\text{quality}} + 0.2 \cdot R_{\text{efficiency}} + 0.1 \cdot R_{\text{safety}}\]</div>

<ul>
<li><p><span class="arithmatex">\(R_{\text{tests}}\)</span>：单元测试通过比例（0–1）。可基于真值验证。</p></li>
<li><p><span class="arithmatex">\(R_{\text{quality}}\)</span>：LLM 评判代码风格、文档、可维护性。</p></li>
<li><p><span class="arithmatex">\(R_{\text{efficiency}}\)</span>：<span class="arithmatex">\(\max(0, 1 - \text{steps}/30)\)</span>——快速完成给奖励。</p></li>
<li><p><span class="arithmatex">\(R_{\text{safety}}\)</span>：无危险操作（rm -rf、沙箱外网络访问）。</p></li>
</ul>

<p><strong>Reward hacking 风险</strong>：</p>

<ol>
<li><p><strong>硬编码输出</strong>：Agent 学会直接打印预期测试输出而不真正计算。<em>修复</em>：随机化测试输入；在保留测试用例上测试。</p></li>
<li><p><strong>删除测试</strong>：Agent 修改/删除失败的测试。<em>修复</em>：测试沙箱设为只读。</p></li>
<li><p><strong>平凡解</strong>：Agent 写出能通过测试但不泛化的最小代码。<em>修复</em>：大型、多样化的测试集；基于属性的测试。</p></li>
<li><p><strong>效率作弊</strong>：Agent 跳过推理步骤以最大化效率奖励。<em>修复</em>：先满足最低质量阈值，再给效率奖励。</p></li>
</ol>

<p><em><strong>复习：</strong>第 <a href="../09-reward-models/index.md">9</a>、<a href="../12-agentic-training/index.md">12</a>、<a href="../21-environments/index.md">21</a> 章（Reward 模型训练；LLM Agent 训练；Agent 环境）。</em></p>
</details></section>
