# 第 28 章 · Agent 训练、推理与评估 {#reading-reading-04}

## 28.18　Agent 强化学习问题 {#agentic-rl-questions}

<section class="quiz-question" id="q-source-72"><h3 class="question-title">Q：为什么标准 RLHF（单轮 PPO/DPO）对多步 Agent 失败？</h3>
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

<section class="quiz-question" id="q-source-73"><h3 class="question-title">Q：解释 GRPO 如何适配到 Agent 训练。与单轮 GRPO 的关键区别是什么？</h3>
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

<section class="quiz-question" id="q-source-74"><h3 class="question-title">Q：比较 Agent 场景下的 STaR、Reflexion 和 ReAct。</h3>
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

<section class="quiz-question" id="q-source-75"><h3 class="question-title">Q：为什么研究型 Agent 偏好 GRPO 而非 PPO？</h3>
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

<section class="quiz-question" id="q-source-76"><h3 class="question-title">Q：为编程 Agent 设计一个 Reward 函数。存在哪些 Reward hacking 风险？</h3>
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

## 28.19　列表式 Reward 与高级 RM 问题 {#listwise-rewards-and-advanced-rm-questions}

<section class="quiz-question" id="q-source-77"><h3 class="question-title">Q：解释 Plackett-Luce 模型。它如何推广 Bradley-Terry？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：Bradley-Terry 建模<em>成对</em>偏好：<span class="arithmatex">\(P(y_1 \succ y_2) = \sigma(r(y_1) - r(y_2))\)</span>。</p>

<p>Plackett-Luce 把 <span class="arithmatex">\(K\)</span> 项的<em>完整排序</em>建模为顺序选择：</p>



<div class="arithmatex">\[P(\pi) = \prod_{i=1}^K \frac{e^{r(y_{\pi(i)})}}{\sum_{j=i}^K e^{r(y_{\pi(j)})}}\]</div>

<p>解释：依次挑选剩余项中最好的。位置 1 = 对全部 <span class="arithmatex">\(K\)</span> 做 Softmax；位置 2 = 对剩余 <span class="arithmatex">\(K-1\)</span> 做 Softmax；以此类推。</p>

<p><strong>推广关系</strong>：当 <span class="arithmatex">\(K=2\)</span> 时，PL 恰好退化为 BT：<span class="arithmatex">\(P(y_1 \succ y_2) = \frac{e^{r(y_1)}}{e^{r(y_1)} + e^{r(y_2)}} = \sigma(r(y_1) - r(y_2))\)</span>。</p>

<p><strong>优势</strong>：<span class="arithmatex">\(K=8\)</span> 项的排序提供 <span class="arithmatex">\(\binom{8}{2} = 28\)</span> 个隐式成对比较，外加相对差距信息——比单一成对样本丰富得多。</p>

<p><em><strong>复习：</strong>第 <a href="../09-reward-models/index.md">9</a> 章（Reward 模型训练）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-78"><h3 class="question-title">Q：什么是过程奖励模型（PRM），它在什么情况下优于结果奖励模型（Outcome Reward Model, ORM）？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>ORM</strong>：仅对最终输出打分。<span class="arithmatex">\(r(x, y_{\text{final}})\)</span> = 整个响应一个标量。</p>

<p><strong>PRM</strong>：对每一<em>步</em>推理打分。<span class="arithmatex">\(r(x, y_{\text{step } t})\)</span> = 每一中间步一个标量。</p>

<p><strong>PRM 更好的场景</strong>：</p>

<ol>
<li><p><strong>长推理链</strong>：10+ 步的数学题。ORM 无法判断哪一步错了；PRM 提供逐步信用分配。</p></li>
<li><p><strong>搜索/验证</strong>：PRM 支持树搜索（对推理步做 beam search，剪掉步 Reward 低的分支）。</p></li>
<li><p><strong>训练信号密度</strong>：PRM 每条轨迹给 <span class="arithmatex">\(T\)</span> 个 Reward（每步一个），而 ORM 只有一个 <span class="arithmatex">\(\to\)</span> 更低方差的优势估计。</p></li>
</ol>

<p><strong>ORM 更好的场景</strong>：任务短（单轮）；步骤边界不清晰；每步标注成本过高。</p>

<p><strong>PRM 标注</strong>：可通过&quot;Math-Shepherd&quot;方法自动化：对每一步，从该点多次补全整个解。如果从步骤 <span class="arithmatex">\(t\)</span> 出发的补全成功而从步骤 <span class="arithmatex">\(t+1\)</span> 出发的补全失败，则步骤 <span class="arithmatex">\(t+1\)</span> 很可能是错的。</p>

<p><em><strong>复习：</strong>第 <a href="../09-reward-models/index.md">9</a> 章与第 <a href="../13-reasoning/index.md">13</a> 章（Reward 模型训练；大型推理模型的 RL）。</em></p>
</details></section>

## 28.20　大型推理模型的 RL 问题 {#rl-for-large-reasoning-models-questions}

<section class="quiz-question" id="q-source-79"><h3 class="question-title">Q：DeepSeek-R1 在长推理链上训练，却为何不使用过程奖励模型？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：DeepSeek-R1 仅使用<strong>基于结果的 Reward</strong>（准确率 + 格式），原因如下：</p>

<ol>
<li><p><strong>可验证任务</strong>：数学和代码有确定的真值答案。即便是长链，二元的准确率 Reward 也能提供足够信号。</p></li>
<li><p><strong>PRM 失效模式</strong>：步骤级 Reward 模型自身会引入 Reward hacking——模型可学到产出“在 PRM 看来正确”但实际不正确的步骤。</p></li>
<li><p><strong>GRPO 的组归一化</strong>：通过每个 Prompt 采样 <span class="arithmatex">\(G\)</span> 个补全并在组内归一化优势，GRPO 即使没有每步 Reward 也能自然提供关于哪些推理<em>策略</em>有效的相对信号。</p></li>
<li><p><strong>涌现的自我纠正</strong>：在仅有结果 Reward 时，模型学会在链内自我纠正（“aha 时刻”），如果每步 Reward 对推理过程微观管理，这种行为将无法涌现。</p></li>
</ol>

<p><strong>关键洞见</strong>：任务领域的可验证性使 PRM 变得不必要——对于主观任务（创意写作），仅有结果 Reward 可能不够。</p>

<p><em><strong>复习：</strong>第 <a href="../13-reasoning/index.md">13</a> 章（大型推理模型的 RL）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-80"><h3 class="question-title">Q：解释测试时计算扩展律（Test-Time Compute Scaling Law）及其对模型部署的启示</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：测试时计算扩展律表述为：</p>



<div class="arithmatex">\[\text{Accuracy}(C_{\text{train}}, C_{\text{test}}) \approx f(\alpha \log C_{\text{train}} + \beta \log C_{\text{test}})\]</div>

<p><strong>启示</strong>：</p>

<ol>
<li><p><strong>算力等价</strong>：在推理任务上，使用 64<span class="arithmatex">\(\times\)</span> 推理 token 的 7B 模型可以与使用 1<span class="arithmatex">\(\times\)</span> token 的 70B 模型相匹敌。</p></li>
<li><p><strong>自适应分配</strong>：简单问题用短链（便宜）；困难问题用长链（贵）。平均成本低于始终使用大模型。</p></li>
<li><p><strong>部署灵活性</strong>：不再部署一个大模型，而是部署一个较小的推理模型，并按查询难度对推理算力进行扩展。</p></li>
<li><p><strong>收益递减</strong>：对数关系意味着测试时计算翻倍带来的准确率提升递减——训练与推理算力之间存在一个最优分配。</p></li>
</ol>

<p><strong>“过度思考”失败模式</strong>：非常长的链可能因错误累积和 Attention 稀释而<em>降低</em>准确率。最优链长取决于问题难度。</p>

<p><em><strong>复习：</strong>第 <a href="../13-reasoning/index.md">13</a> 章（大型推理模型的 RL）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-81"><h3 class="question-title">Q：MCTS（蒙特卡洛树搜索，Monte Carlo Tree Search）如何应用于 LLM 推理？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：用于推理的 MCTS 将每个部分解视为一个树节点：</p>

<p><strong>每次迭代四个阶段</strong>：</p>

<ol>
<li><p><strong>选择</strong>：用 UCB 从根开始导航：<span class="arithmatex">\(\text{UCB}(s) = Q(s) + c\sqrt{\frac{\ln N(\text{parent})}{N(s)}}\)</span></p></li>
<li><p><strong>扩展</strong>：从 LLM 生成新的推理步（子节点）</p></li>
<li><p><strong>模拟</strong>：从新节点完成解（Rollout）</p></li>
<li><p><strong>反向传播</strong>：根据最终正确性更新路径上的 Q 值</p></li>
</ol>

<p><strong>与博弈 MCTS 的关键差异</strong>：</p>

<ul>
<li><p><strong>分支因子</strong>：推理具有巨大的分支因子（任何下一句都可能）。实际实现使用 LLM 的 top-k 输出来限制分支。</p></li>
<li><p><strong>Value 函数</strong>：训练好的 PRM 估计部分解质量，替代随机 Rollout。</p></li>
<li><p><strong>步骤粒度</strong>：每“步”可能是一句话、一个方程或一次逻辑推断——粒度选择很关键。</p></li>
</ul>

<p><strong>使用案例</strong>：AlphaProof（数学奥林匹克），以及据推测 OpenAI o1/o3 的隐藏推理。</p>

<p><em><strong>复习：</strong>第 <a href="../13-reasoning/index.md">13</a> 章（大型推理模型的 RL）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-82"><h3 class="question-title">Q：比较通过蒸馏 vs 直接 RL 来打造小型推理模型</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>蒸馏</strong>（DeepSeek-R1-Distill 路线）：</p>

<ul>
<li><p>从大模型（R1-671B）生成推理链</p></li>
<li><p>在这些链上对小模型做 SFT</p></li>
<li><p>结果：小模型模仿大模型的推理<em>格式</em></p></li>
<li><p>优点：便宜（仅 SFT）。缺点：可能学到表层模式而非真正推理。</p></li>
</ul>

<p><strong>直接对小模型做 RL</strong>：</p>

<ul>
<li><p>用 GRPO/PPO 在可验证 Reward 上训练小模型</p></li>
<li><p>模型发现自己的推理策略</p></li>
<li><p>优点：真实能力。缺点：算力消耗大得多；对于非常小的模型可能不收敛。</p></li>
</ul>

<p><strong>经验发现</strong>：R1-Distill-7B（蒸馏）在大多数基准上优于 direct-RL-7B。大模型的推理链提供了非常强的监督信号，单 SFT 就具有竞争力。然而，蒸馏模型对真正新颖的问题类型泛化较差。</p>

<p><strong>最佳实践</strong>：先蒸馏（便宜的基线），再可选地在蒸馏模型上跑 RL 以获得进一步收益（Qwen 使用的“蒸馏 + RL”组合）。</p>

<p><em><strong>复习：</strong>第 <a href="../13-reasoning/index.md">13</a> 章（大型推理模型的 RL）。</em></p>
</details></section>

## 28.21　LLM 评估问题 {#llm-evaluation-questions}

<section class="quiz-question" id="q-source-83"><h3 class="question-title">Q：推导 ELO 评分更新规则，并解释 Chatbot Arena 为何使用它</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>ELO 推导</strong>：</p>

<p>玩家 A 对 B 的期望得分：<span class="arithmatex">\(E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}\)</span>（逻辑斯蒂模型）。</p>

<p>对局后，实际得分 <span class="arithmatex">\(S_A \in \{0, 0.5, 1\}\)</span>：<span class="arithmatex">\(R_A&#x27; = R_A + K(S_A - E_A)\)</span></p>

<p><span class="arithmatex">\(K\)</span> 因子控制更新幅度（<span class="arithmatex">\(K\)</span> 越大对近期结果反应越敏感）。</p>

<p><strong>为什么 Chatbot Arena 使用 ELO</strong>：</p>

<ol>
<li><p><strong>传递性</strong>：若 A 胜 B 且 B 胜 C，ELO 预测 A 胜 C。这从成对比较给出全序。</p></li>
<li><p><strong>在线更新</strong>：新增模型无需重新评估所有对。每次新比较都增量更新评分。</p></li>
<li><p><strong>置信度</strong>：在 <span class="arithmatex">\(N\)</span> 次比较后，评分不确定性以 <span class="arithmatex">\(O(1/\sqrt{N})\)</span> 收缩。标准误：<span class="arithmatex">\(\text{SE} \approx \frac{400}{\sqrt{N}}\)</span>。</p></li>
<li><p><strong>捕捉人类偏好</strong>：真实用户无需阐明标准即可提供诚实偏好。聚合揭示了真实模型质量。</p></li>
</ol>

<p><strong>Chatbot Arena 细节</strong>：使用 Bradley-Terry MLE（收敛时等价于 ELO），并配合 Bootstrap 置信区间。风格控制 ELO 去除长度/格式偏倚。</p>

<p><em><strong>复习：</strong>第 <a href="../14-evaluation/index.md">14</a> 章（LLM 评估）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-84"><h3 class="question-title">Q：代码生成的 pass@k 指标是什么，为什么无偏估计很重要？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>pass@k</strong> = 生成的 <span class="arithmatex">\(k\)</span> 个样本中至少有一个通过全部测试用例的概率。</p>

<p><strong>朴素（有偏）估计器</strong>：生成 <span class="arithmatex">\(k\)</span> 个样本，看是否有任意一个通过。问题：高方差、昂贵（每题需要很多次试验）。</p>

<p><strong>无偏估计器</strong>（Chen 等，2021）：生成 <span class="arithmatex">\(n \geq k\)</span> 个样本，统计其中通过的数量 <span class="arithmatex">\(c\)</span>：</p>



<div class="arithmatex">\[\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\]</div>

<p><strong>无偏为何重要</strong>：</p>

<ol>
<li><p>一次性生成 <span class="arithmatex">\(n\)</span> 个样本（例如 <span class="arithmatex">\(n=200\)</span>），从同一批样本算出 pass@1、pass@10、pass@100</p></li>
<li><p>无需将整个评估重复 <span class="arithmatex">\(k\)</span> 次</p></li>
<li><p>统计上精确（组合论证：不含任何正确样本的 <span class="arithmatex">\(k\)</span>-子集占比）</p></li>
<li><p>通过对数空间数值稳定计算：<span class="arithmatex">\(\text{pass@}k = 1 - \exp\left(\sum_{i=0}^{k-1} \log(n-c-i) - \log(n-i)\right)\)</span></p></li>
</ol>

<p><strong>直觉</strong>：若 50/200 个样本通过（<span class="arithmatex">\(c=50\)</span>，<span class="arithmatex">\(n=200\)</span>），pass@1 <span class="arithmatex">\(\approx 0.25\)</span>，pass@10 <span class="arithmatex">\(\approx 0.94\)</span>。该估计器统计大小为 <span class="arithmatex">\(k\)</span> 的抽取中至少包含一个成功的比例。</p>

<p><em><strong>复习：</strong>第 <a href="../14-evaluation/index.md">14</a> 章与第 <a href="../21-environments/index.md">21</a> 章（LLM 评估；Agent 环境）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-85"><h3 class="question-title">Q：如何检测和缓解基准污染（Benchmark Contamination）？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>污染</strong>：训练数据中包含基准测试样例（或其近似改写），使分数虚高。</p>

<p><strong>检测方法</strong>：</p>

<ol>
<li><p><strong>N-gram 重叠</strong>：检查训练数据是否与测试条目精确或近似匹配。8-gram 重叠且覆盖率 <span class="arithmatex">\(&gt;\)</span>80% 即可疑。</p></li>
<li><p><strong>Canary 字符串</strong>：在测试集中插入唯一标识符；检查模型是否能复现它们。</p></li>
<li><p><strong>改写基准</strong>：创建语义等价但文本不同的基准版本。准确率大幅下降表明存在记忆。</p></li>
<li><p><strong>时序分析</strong>：比较模型在训练截止前 vs 截止后测试条目上的表现。在旧条目上异常高表现暗示污染。</p></li>
<li><p><strong>成员推断</strong>：统计检验判断特定样例是否出现在训练数据中。</p></li>
</ol>

<p><strong>缓解</strong>：</p>

<ul>
<li><p><strong>动态基准</strong>：定期生成新测试条目（LiveCodeBench、Chatbot Arena）</p></li>
<li><p><strong>私有测试集</strong>：保持测试条目保密（LMSYS）</p></li>
<li><p><strong>训练期去污染</strong>：从训练数据中删除检测到的重叠</p></li>
<li><p><strong>报告污染分析</strong>：在基准分数旁披露重叠度指标</p></li>
</ul>

<p><em><strong>复习：</strong>第 <a href="../14-evaluation/index.md">14</a> 章（LLM 评估）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-86"><h3 class="question-title">Q：解释 LLM-as-Judge 中的位置偏倚（Position Bias），以及如何缓解</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>位置偏倚</strong>：当用 LLM 评判两个响应（A vs B）时，模型系统性地偏好特定位置的响应（通常是第一个或最后一个），与质量无关。</p>

<p><strong>经验幅度</strong>：GPT-4 表现出 10%–15% 的位置偏倚；Claude 为 5%–10%。较小的模型偏倚更大。</p>

<p><strong>缓解策略</strong>：</p>

<ol>
<li><p><strong>位置交换</strong>：每对评判两次（A-B 与 B-A）。最终决定 = 多数。若不一致，则标为“平局”。这能消除系统性位置偏倚，但成本翻倍。</p></li>
<li><p><strong>多评委合议</strong>：使用 3 个及以上不同模型作为评委。多数投票减少单一模型的偏倚。</p></li>
<li><p><strong>参考引导</strong>：提供评分量表或参考答案。评委按量表独立为每个响应打分，再比较分数（完全消除成对比较）。</p></li>
<li><p><strong>校准 Prompt</strong>：加入明确指令：“呈现顺序是随机的，不应影响你的判断。”</p></li>
</ol>

<p><strong>其他偏倚</strong>：冗长偏倚（偏好更长响应）、自我增强偏倚（模型偏好自身输出）、权威偏倚（顺从于引用来源的响应）。</p>

<p><em><strong>复习：</strong>第 <a href="../14-evaluation/index.md">14</a> 章（LLM 评估）。</em></p>
</details></section>
