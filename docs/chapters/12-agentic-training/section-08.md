# 12.8　LLM Agent 的 SOTA RL {#state-of-the-art-rl-for-llm-agents}

<p>针对 LLM Agent 的 RL 技术聚焦于<strong>在策略 policy 梯度</strong>与<strong>细粒度信用分配</strong>的结合。因为 Agent 执行涉及工具交互、API 查询和代码执行的复杂多轮（Multi-Turn）轨迹，标准的单轮对齐算法必须经过大幅修改。</p>

## 12.8.1　主导基线：用于 Agent 的 GRPO {#dominant-baseline-grpo-for-agents}

<p>由 DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 推广，<strong>GRPO</strong> <a class="book-citation" id="cite-shao2024deepseekmath-1" href="../../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a> 正迅速成为 Agent 训练的标准。它为每个任务采样一组 <span class="arithmatex">\(N\)</span> 条完整轨迹，从而消除了内存密集的 critic 网络：</p>

<p>对于任务 Prompt <span class="arithmatex">\(q\)</span>，GRPO 从 <span class="arithmatex">\(\pi_{\theta_{\text{old}}}\)</span> 采样 <span class="arithmatex">\(N\)</span> 条 Agent 轨迹 <span class="arithmatex">\(\{o_1, o_2, \dots, o_N\}\)</span>。每条轨迹的优势通过将其 reward 相对于组归一化来计算：</p>



<div class="arithmatex">\[\boxed{A_i = \frac{r(o_i) - \frac{1}{N}\sum_{j=1}^N r(o_j)}{\text{std}(r(o_1), \dots, r(o_N))}}
\tag{12.18}\]</div>

<p>带 KL 正则的 GRPO 目标：</p>



<div class="arithmatex">\[L_{\text{GRPO}}(\theta) = \frac{1}{N} \sum_{i=1}^N \min\!\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i,\; \text{clip}\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1{-}\epsilon, 1{+}\epsilon\right) A_i \right) - \beta\, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
\tag{12.19}\]</div>

<div class="admonition tip"><p class="admonition-title">为什么 GRPO 在 Agent 训练中占优</p>
<ul>
<li><p><strong>无 critic</strong>：节省 50% GPU 内存——当 Agent 轨迹已经占用了庞大的上下文窗口（32K–128K Token）时至关重要。</p></li>
<li><p><strong>自然契合</strong>：Agent 任务通常有二元可验证 reward（测试通过/失败、目标达成/未达成）——非常适合组相对归一化。</p></li>
<li><p><strong>探索</strong>：每个任务采样 <span class="arithmatex">\(N\)</span> 条多样化轨迹，自然地探索不同的工具使用策略。</p></li>
</ul>
</div>

## 12.8.2　用于交互式 Agent 的 PPO {#ppo-for-interactive-agents}

<p>对于在高度随机环境中运行、且步骤级 Value 估计有所帮助的 Agent，<strong>PPO</strong> <a class="book-citation" id="cite-schulman2017proximal-1" href="../../references.md#ref-schulman2017proximal" aria-label="参考文献 319">[319]</a> 仍然有价值。Critic 提供每步优势信号，在工具输出不可预测时实现更精细的信用分配：</p>

<ul>
<li><p>通过 GAE 进行步骤级优势估计，处理可变长度的工具输出</p></li>
<li><p>Value 头学习预测“从这里开始这条轨迹有多大可能成功”</p></li>
<li><p>当外部工具返回灾难性错误导致 reward 方差激增时更稳定</p></li>
<li><p>权衡：需要 <span class="arithmatex">\(2\times\)</span> 内存（critic）但提供更密集的学习信号</p></li>
</ul>

## 12.8.3　细粒度的回合级信用分配 {#fine-grained-turn-level-credit-assignment}

<p>Agent RL 的核心挑战是<strong>稀疏 reward 问题</strong>。如果 Agent 执行 20 个工具动作并最终未通过一个单元测试，终端 reward 为 <span class="arithmatex">\(0\)</span> 会平等地惩罚所有 20 个动作。现代解决方案：</p>

<div class="admonition note"><p class="admonition-title">可验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards, RLVR）</p>
<p>在<strong>确定性的中间检查点</strong>对模型给予 reward：</p>

<ul>
<li><p>Bash 命令成功编译 <span class="arithmatex">\(\rightarrow\)</span> +0.1</p></li>
<li><p>浏览器 Agent 定位到正确的 HTML 元素 <span class="arithmatex">\(\rightarrow\)</span> +0.2</p></li>
<li><p>SQL 查询返回非空结果 <span class="arithmatex">\(\rightarrow\)</span> +0.1</p></li>
<li><p>最终测试套件通过 <span class="arithmatex">\(\rightarrow\)</span> +1.0（终端）</p></li>
</ul>

<p>中间 reward 为<em>每一</em>步都提供梯度信号，而不仅是最后一步。这相比仅有稀疏 reward 的情况，将学习速度显著加速 3–5<span class="arithmatex">\(\times\)</span>。</p>
</div>

<div class="admonition note"><p class="admonition-title">多轮轨迹切片</p>
<p>框架将多轮 Agent 运行拆分为独立的单个步骤。一个信用分配模块隔离出<strong>破坏轨迹的确切子步骤</strong>：</p>

<ol>
<li><p>重放成功前缀（步骤 1–<span class="arithmatex">\(k\)</span>）</p></li>
<li><p>识别第一个偏离点（步骤 <span class="arithmatex">\(k+1\)</span>，出错处）</p></li>
<li><p>仅对该特定步骤分配负 reward</p></li>
<li><p>对正确的前缀步骤分配中性/正 reward</p></li>
</ol>

<p>这使得在不破坏已正确行为的前提下，进行外科手术式的 Policy 更新成为可能。</p>
</div>

## 12.8.4　替代范式 {#alternative-paradigms}

<ul>
<li><p><strong>迭代式 STaR（Self-Taught Reasoner）</strong> <a class="book-citation" id="cite-zelikman2022star-1" href="../../references.md#ref-zelikman2022star" aria-label="参考文献 418">[418]</a>：不进行连续 RL，而是使用迭代离线循环。生成轨迹 <span class="arithmatex">\(\rightarrow\)</span> 过滤失败 <span class="arithmatex">\(\rightarrow\)</span> 在成功上 SFT <span class="arithmatex">\(\rightarrow\)</span> 重复。易于扩展，避免 RL 不稳定性。每次迭代都自举推理能力。</p></li>
<li><p><strong>强化世界模型学习（Reinforcement World Model Learning, RWML）</strong> <a class="book-citation" id="cite-yu2026rwml-1" href="../../references.md#ref-yu2026rwml" aria-label="参考文献 414">[414]</a>：为对抗 reward hacking，训练 Agent 预测其动作的<em>语义后果</em>。Agent 因准确预测环境状态如何变化而获得辅助 reward（例如，在执行 SQL 之前预测数据库表的变化）。这迫使真正理解，而非表面的 reward 博弈。</p></li>
<li><p><strong>LATS（语言 Agent 树搜索）</strong> <a class="book-citation" id="cite-zhou2024lats-1" href="../../references.md#ref-zhou2024lats" aria-label="参考文献 438">[438]</a>：在 Agent 动作序列上应用蒙特卡洛树搜索。在每一步，扩展多个候选动作，模拟其结果，并通过树反向传播 reward。将 RL 价值估计与搜索时算力扩展相结合。</p></li>
</ul>

## 12.8.5　核心方法论比较 {#core-methodology-comparison}

<p class="table-caption" id="table-12-9"><span class="table-number">表 12.9　</span>LLM Agent 的 RL 范式比较。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>Reward 密度</strong></th><th scope="col"><strong>内存成本</strong></th><th scope="col"><strong>主要优势</strong></th></tr></thead>
<tbody><tr><td><strong>GRPO</strong> <a class="book-citation" id="cite-shao2024deepseekmath-2" href="../../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a></td><td>序列 / 最终指标</td><td>低（无 critic）</td><td>显著降低 GPU 内存；实现简单</td></tr>
<tr><td><strong>PPO</strong> <a class="book-citation" id="cite-schulman2017proximal-2" href="../../references.md#ref-schulman2017proximal" aria-label="参考文献 319">[319]</a></td><td>逐步（GAE）</td><td>高（需 critic）</td><td>细粒度信用分配；在噪声环境中稳定</td></tr>
<tr><td><strong>迭代式 STaR</strong> <a class="book-citation" id="cite-zelikman2022star-2" href="../../references.md#ref-zelikman2022star" aria-label="参考文献 418">[418]</a></td><td>稀疏（过滤后的二元）</td><td>极小（仅 SFT）</td><td>易于扩展；避免 RL 优化不稳定性</td></tr>
<tr><td><strong>RWML</strong> <a class="book-citation" id="cite-yu2026rwml-2" href="../../references.md#ref-yu2026rwml" aria-label="参考文献 414">[414]</a></td><td>密集（预测式）</td><td>中</td><td>通过世界建模缓解 reward hacking</td></tr>
<tr><td><strong>LATS</strong> <a class="book-citation" id="cite-zhou2024lats-2" href="../../references.md#ref-zhou2024lats" aria-label="参考文献 438">[438]</a></td><td>反向传播</td><td>高（树扩展）</td><td>每个任务的质量最佳；随推理算力扩展</td></tr></tbody>
</table>
</div>
