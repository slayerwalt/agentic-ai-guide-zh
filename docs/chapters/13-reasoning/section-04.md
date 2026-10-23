# 13.4　OpenAI o1/o3 系列 {#subsec:openai_o1}

<p>OpenAI 的 o1 <a class="book-citation" id="cite-openai2024o1-1" href="../../references.md#ref-openai2024o1" aria-label="参考文献 275">[275]</a>（2024 年 9 月发布）以及后续的 o3/o4-mini <a class="book-citation" id="cite-openai2025o3-1" href="../../references.md#ref-openai2025o3" aria-label="参考文献 278">[278]</a> 模型代表了推理模型开发的商业前沿。尽管完整的技术细节仍属专有，但已公开的系统卡（system card）、技术报告和实证观察为理解其方法论提供了充足的线索。</p>

## 13.4.1　带隐藏推理 Token 的思维链 RL {#chain-of-thought-rl-with-hidden-reasoning-tokens}

<p>o1 最具标志性的架构选择是使用<strong>隐藏推理 Token（hidden reasoning tokens）</strong>：模型生成一段不向用户展示的内部思维链（称为“推理轨迹（reasoning trace）”或“思考 Token（thinking tokens）”）。最终只返回答案本身。这一设计具有若干意涵：</p>

<ul>
<li><p><strong>无格式约束</strong>：隐藏推理可以使用任意格式，包括草稿纸式记法、伪代码，甚至非英文推理。</p></li>
<li><p><strong>风格上无奖励作弊（reward hacking）</strong>：由于用户从未看到推理过程，模型没有为了“好看”而牺牲“有用”的压力。</p></li>
<li><p><strong>专有保护</strong>：推理过程不被暴露，防止直接模仿。</p></li>
</ul>

<p>训练流程被描述为“使用 RL 训练模型进行推理”，其 RL 目标作用于完整的（隐藏推理 + 最终答案）序列，奖励仅基于最终答案的质量。</p>

## 13.4.2　过程奖励模型与结果奖励模型 {#process-reward-models-vs.-outcome-reward-models}

<p>OpenAI 的方法据信在结果奖励之外还使用了<strong>过程奖励模型（Process Reward Model, PRM）</strong> <a class="book-citation" id="cite-lightman2023lets-1" href="../../references.md#ref-lightman2023lets" aria-label="参考文献 215">[215]</a>，这与 DeepSeek-R1 仅采用结果奖励的方式形成对比。这一推断基于 OpenAI 已发表的 PRM 研究（PRM800K 数据集，&quot;Let’s Verify Step by Step&quot;）以及 o1 系统卡中对推理链 RL 训练的描述，尽管 o1/o3 的精确训练配方并未公开披露。</p>

#### 结果奖励模型（ORM） {#outcome-reward-model-orm}

<p>结果奖励模型（Outcome Reward Model, ORM）对完整回复 <span class="arithmatex">\((q, y)\)</span> 打分：</p>



<div class="arithmatex">\[R_{\text{ORM}}(q, y) \in [0, 1]
\tag{13.25}\]</div>

<p>对于可验证任务（数学、代码），它简化为精确匹配验证；对于开放式任务，则使用一个学习得到的奖励模型。</p>

#### 过程奖励模型（PRM） {#process-reward-model-prm}

<p>PRM 为推理链 <span class="arithmatex">\(y = (s_1, s_2, \ldots, s_K)\)</span> 中的每一个推理步骤 <span class="arithmatex">\(s_k\)</span> 分配奖励：</p>



<div class="arithmatex">\[R_{\text{PRM}}(q, y) = \sum_{k=1}^{K} \gamma^{K-k} \cdot r_k(q, s_1, \ldots, s_k)
\tag{13.26}\]</div>

<p>其中 <span class="arithmatex">\(r_k \in [0,1]\)</span> 是步骤级奖励，<span class="arithmatex">\(\gamma \in (0,1]\)</span> 是折扣因子。步骤级奖励 <span class="arithmatex">\(r_k\)</span> 估计部分解 <span class="arithmatex">\((s_1, \ldots, s_k)\)</span> 能够导向正确最终答案的概率：</p>



<div class="arithmatex">\[r_k(q, s_1, \ldots, s_k) = P(\text{correct final answer} \mid q, s_1, \ldots, s_k)
\tag{13.27}\]</div>

<div class="admonition tip"><p class="admonition-title">PRM 与 ORM：信用分配的权衡</p>
<p><strong>ORM</strong> 提供干净、明确的奖励，但存在严重的信用分配问题：在一条 50 步推理链早期的单个错误步骤，所获得的零奖励与完全随机的回复并无区别。</p>

<p><strong>PRM</strong> 提供稠密奖励以直接处理信用分配，但也引入了新的挑战：</p>

<ul>
<li><p><strong>训练数据</strong>：步骤级标签需要人工标注或自动化生成（Math-Shepherd，见 <a href="section-06.md#subsubsec:prm_methods">13.6.2</a> 节）。</p></li>
<li><p><strong>奖励作弊</strong>：模型可能学会产生在 PRM 眼中<em>看起来</em>正确、但实际上并不正确的步骤。</p></li>
<li><p><strong>分布漂移</strong>：在某一推理链分布上训练的 PRM 可能无法泛化到 RL 产生的新颖推理链。</p></li>
</ul>

<p>经验证据表明，PRM 对<em>搜索</em>（在候选解中选择）是有益的，但对<em>训练</em>的收益则不那么明确。</p>
</div>

## 13.4.3　推理时计算扩展 {#inference-time-compute-scaling}

<p>o1 技术报告展示了一条清晰的扩展律：在困难推理任务上，<strong>更多的思考 Token 单调地提升性能</strong>。这通过一个控制隐藏推理 Token 数量上限的“思考预算（thinking budget）”参数得以实现。</p>

<p>设 <span class="arithmatex">\(T\)</span> 为思考 Token 预算。经验上观察到的扩展律近似为：</p>



<div class="arithmatex">\[\text{Pass@1}(T) \approx a - b \cdot T^{-c}
\tag{13.28}\]</div>

<p>其中 <span class="arithmatex">\(a, b, c &gt; 0\)</span> 为常数，<span class="arithmatex">\(a\)</span> 表示渐近准确率上限，<span class="arithmatex">\(c\)</span> 刻画提升速率。在 AIME 2024 上，配备完整思考预算的 o1 达到约 83% 的准确率，而未使用扩展思考的 GPT-4o 仅约 13%。</p>

## 13.4.4　训练计算量与测试时计算量 {#training-compute-vs.-test-time-compute}

<p>o1/o3 系列带来的一个根本性洞见是<strong>计算等价原理（compute equivalence principle）</strong>：训练计算量 <span class="arithmatex">\(C_{\text{train}}\)</span> 与测试时计算量 <span class="arithmatex">\(C_{\text{test}}\)</span> 之间存在一条权衡曲线，曲线上的各点能够达到相近的性能：</p>



<div class="arithmatex">\[\text{Performance}(C_{\text{train}}, C_{\text{test}}) = g\!\left(\alpha C_{\text{train}}^{p} + \beta C_{\text{test}}^{q}\right)
\tag{13.29}\]</div>

<p>经验上，对于推理任务 <span class="arithmatex">\(p \approx q\)</span>，这意味着训练计算量与测试时计算量大致可互相替代。这对部署有深远影响：一个更小、更便宜但拥有扩展思考能力的模型，可以在困难问题上匹敌更大的模型，代价是更高的延迟。</p>

## 13.4.5　o3 与 o4-mini 的架构洞察 {#o3-and-o4-mini-architecture-insights}

<p>尽管 o3 和 o4-mini 的细节大多仍属专有，但已经浮现出一些观察：</p>

<ul>
<li><p><strong>o3</strong>：思考预算显著大于 o1；在 ARC-AGI 上达到接近人类的表现（高算力下 87.5%）。据信在推理时使用了更复杂的搜索策略。</p></li>
<li><p><strong>o4-mini</strong>：证明了经过 RL 训练推理能力的<em>更小</em>模型同样极具竞争力。在 AIME 2025 上配合扩展思考达到 93%，表明在数学任务上模型规模的重要性不及推理能力。</p></li>
<li><p><strong>工具调用（Tool Calling）</strong>：o3/o4-mini 将工具调用（代码执行、网络搜索）集成到推理过程中，使模型能够以程序化方式验证中间步骤。</p></li>
</ul>
