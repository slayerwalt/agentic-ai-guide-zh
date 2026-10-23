# 9.4　过程奖励模型 vs 结果奖励模型 {#sec:prm-orm}

<div class="admonition note"><p class="admonition-title">PRM 与 ORM 对比</p>
<span id="table-9-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>属性</strong></th><th scope="col"><strong>ORM</strong></th><th scope="col"><strong>PRM</strong></th></tr></thead>
<tbody><tr><td>Reward 信号</td><td>仅最终答案</td><td>每一步推理</td></tr>
<tr><td>训练数据</td><td>(prompt, answer, correct?)</td><td>(prompt, steps, step labels)</td></tr>
<tr><td>标注成本</td><td>低</td><td>高</td></tr>
<tr><td>信用分配</td><td>稀疏</td><td>稠密</td></tr>
<tr><td>Reward 黑客</td><td>较易被攻击</td><td>较难被攻击</td></tr>
<tr><td>最适合</td><td>简单任务</td><td>多步推理</td></tr>
<tr><td>推理成本</td><td>低</td><td>高（需对每步打分）</td></tr></tbody>
</table>
</div>
</div>

<div class="admonition tip"><p class="admonition-title">何时使用 PRM</p>
<p>过程奖励模型（PRM）在以下场景中最有价值：</p>

<ul>
<li><p>任务需要多步推理（数学、代码、逻辑）。</p></li>
<li><p>最终答案是二元的（正确/错误），但中间步骤质量参差不齐。</p></li>
<li><p>你希望将奖励模型用于<em>搜索</em>（例如带步骤分数的 beam search）。</p></li>
<li><p>你能获得步骤级标注（或可自动生成）。</p></li>
</ul>

<p>对于简单任务（情感、毒性、事实性），ORM 已经足够，且成本低得多。</p>
</div>

<div class="admonition example"><p class="admonition-title">LLM RLHF 中的 PBRS</p>
<p><strong>原始 Reward</strong>：二元正确性（最终答案正确为 1，否则为 0）—— 对多步推理而言极其稀疏。</p>

<p><strong>势函数（Potential function）</strong>：<span class="arithmatex">\(\Phi(s) =\)</span> 来自验证器的部分得分（例如，逻辑上有效的中间推理步骤的比例）。</p>

<p><strong>塑形后的 Reward</strong>：Agent 对每个有效推理步骤都获得增量信号，同时保留“最优 Policy 仍然最大化最终答案正确率”的理论保证。</p>

<p><strong>实际实现</strong>：</p>

<ul>
<li><p>对思维链（Chain-of-Thought, CoT）中每一步打分的过程奖励模型（PRM）</p></li>
<li><p>代码生成中的中间编译检查</p></li>
<li><p>多部分答案的部分匹配分数</p></li>
</ul>

<p>这是基于势函数的 Reward 塑形（Potential-Based Reward Shaping, PBRS） <a class="book-citation" id="cite-ng1999policy-1" href="../../references.md#ref-ng1999policy" aria-label="参考文献 265">[265]</a> 在 LLM 场景下的直接应用 —— “塑形 Reward 保留最优 Policy”这一理论保证使得 PRM 成为在推理任务中获取稠密 Reward 的有原则方法。</p>
</div>

### 自动 PRM 标注 {#automatic-prm-annotation}

<p>步骤级标注可通过以下方法自动生成：</p>

<ol>
<li><p><strong>蒙特卡洛 rollout</strong>：对每个中间步骤，采样多个补全（completion），将到达正确答案的比例作为该步的 Reward。</p></li>
<li><p><strong>LLM-as-judge</strong>：使用一个强大的 LLM 来评估每一步。</p></li>
<li><p><strong>形式化验证</strong>：对数学/代码任务，使用验证器逐步检查。</p></li>
</ol>
