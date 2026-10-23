# 第 13 章 · 代表性推理模型 {#reading-reading-02}

## 13.3　DeepSeek-R1 {#subsec:deepseek_r1}

<p>DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 是首个在主要基准上达到或超越 OpenAI o1 的完全开源大型推理模型。其训练流程在技术上完全透明，已成为基于 RL 的推理的事实参考实现。</p>

### 13.3.1　两阶段训练流程 {#two-stage-training-pipeline}

##### 阶段 1：冷启动监督微调 {#stage-1-cold-start-supervised-fine-tuning}

<p>基础模型（DeepSeek-V3）首先在一个经过精心整理的小规模长思维链示例数据集上做微调。这个“冷启动”阶段服务于两个目的：</p>

<ol>
<li><p><strong>格式初始化</strong>：模型学会在给出最终答案之前，以 <code>&lt;think&gt;...&lt;/think&gt;</code> 的格式输出推理。</p></li>
<li><p><strong>稳定性</strong>：若没有冷启动 SFT，直接在基础模型上从零进行纯 RL 会导致训练动力学不稳定与退化输出（例如混语、重复循环）。</p></li>
</ol>

<p>冷启动数据集仅包含 <span class="arithmatex">\(\sim\)</span> 数千条样本，刻意保持较小规模，以避免过度约束 RL 后续将发现的推理风格。</p>

##### 阶段 2：基于 GRPO 的强化学习 {#stage-2-grpo-based-reinforcement-learning}

<p>冷启动 SFT 之后，模型进入使用组相对策略优化（Group Relative Policy Optimization, GRPO）的大规模 RL 阶段。R1 中使用的完整 GRPO 目标见第 <a href="#subsubsec:grpo_r1_math">13.3.3</a> 节。</p>

<div class="admonition note"><p class="admonition-title">R1 训练流程总结</p>
<ol>
<li><p><strong>基础模型</strong>：DeepSeek-V3（671B MoE，37B 激活参数）</p></li>
<li><p><strong>冷启动 SFT</strong>：<span class="arithmatex">\(\sim\)</span> 数千条长 CoT 样本，格式：<code>&lt;think&gt;...&lt;/think&gt;&lt;answer&gt;...&lt;/answer&gt;</code></p></li>
<li><p><strong>RL 阶段</strong>：在数学与代码问题上使用带可验证奖励的 GRPO</p></li>
<li><p><strong>拒绝采样（Rejection Sampling）</strong>：生成多个解答，保留正确的</p></li>
<li><p><strong>在 RL 输出上做 SFT</strong>：在高质量的 RL 生成链上微调</p></li>
<li><p><strong>最终 RL</strong>：用于对齐与有用性的第二轮 RL</p></li>
</ol>
</div>

### 13.3.2　奖励设计：准确率奖励与格式奖励 {#reward-design-accuracy-and-format-rewards}

<p>R1 的一个关键设计选择是<strong>不使用过程奖励模型（PRM）</strong>。取而代之，R1 使用两种简单且可自动计算的奖励：</p>

##### 准确率奖励 {#accuracy-reward}

<p>对答案可验证的数学问题：</p>



<div class="arithmatex">\[r_{\text{acc}}(y, y^*) = \begin{cases} 1 &amp; \text{if } \texttt{verify}(y, y^*) = \texttt{True} \\ 0 &amp; \text{otherwise} \end{cases}
\tag{13.16}\]</div>

<p>其中 <span class="arithmatex">\(y\)</span> 是模型的最终答案（从 <code>&lt;answer&gt;</code> 标签中提取），<span class="arithmatex">\(y^*\)</span> 是真实答案。<code>verify</code> 函数使用符号数学比较（如 SymPy）来处理等价形式。</p>

<p>对于代码问题，准确率奖励由通过测试用例决定：</p>



<div class="arithmatex">\[r_{\text{acc}}^{\text{code}}(y, \mathcal{T}) = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \mathbf{1}[\texttt{execute}(y, t) = \texttt{expected}(t)]
\tag{13.17}\]</div>

##### 格式奖励 {#format-reward}

<p>用于强制 <code>&lt;think&gt;...&lt;/think&gt;</code> 结构：</p>



<div class="arithmatex">\[r_{\text{fmt}}(y) = \begin{cases} 1 &amp; y \text{ 含有有效的 &lt;think&gt; 与 &lt;answer&gt; 标签} \\ 0 &amp; \text{否则} \end{cases}
\tag{13.18}\]</div>

##### 组合奖励 {#combined-reward}



<div class="arithmatex">\[r(y, y^*) = r_{\text{acc}}(y, y^*) + \lambda_{\text{fmt}} \cdot r_{\text{fmt}}(y)
\tag{13.19}\]</div>

<p>在原始实现中 <span class="arithmatex">\(\lambda_{\text{fmt}} = 0.1\)</span>（足够小以不至于主导优化，足够大以防止格式崩塌）。</p>

<div class="admonition warning"><p class="admonition-title">没有过程奖励模型</p>
<p>R1 一个引人注目且令人意外的发现是：<strong>不需要过程奖励模型（PRM）</strong>。尽管推理链很长，仅靠结果奖励就足以让 RL 发现高质量的推理策略。作者推测，数学/代码奖励的可验证性提供了足够的信号，而 PRM 自身又会引入新的失效模式（步级别的奖励黑客）。这与 OpenAI 所采取的方法形成对比（见第 <a href="#subsec:openai_o1">13.4</a> 节）。</p>
</div>

### 13.3.3　R1 中的 GRPO 公式 {#subsubsec:grpo_r1_math}

<p>GRPO <a class="book-citation" id="cite-shao2024deepseekmath-1" href="../../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a> 是一种策略梯度方法，它通过对一<em>组</em>采样回答估计优势，从而避免训练独立的价值网络。对于一个问题 <span class="arithmatex">\(q\)</span>，GRPO 从当前 Policy <span class="arithmatex">\(\pi_\theta\)</span> 采样 <span class="arithmatex">\(G\)</span> 个回答 <span class="arithmatex">\(\{y_1, y_2, \ldots, y_G\}\)</span>，并相对组均值计算优势。</p>

##### 分组采样与优势归一化 {#group-sampling-and-advantage-normalization}

<p>给定问题 <span class="arithmatex">\(q\)</span>，采样 <span class="arithmatex">\(G\)</span> 个输出：</p>



<div class="arithmatex">\[\{y_i\}_{i=1}^G \sim \pi_\theta(\cdot \mid q)
\tag{13.20}\]</div>

<p>使用式 [eq:r1_combined_reward] 的奖励函数计算 Reward <span class="arithmatex">\(\{r_i\}_{i=1}^G\)</span>。第 <span class="arithmatex">\(i\)</span> 个回答的归一化优势为：</p>



<div class="arithmatex">\[\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}
\tag{13.21}\]</div>

<p>其中 <span class="arithmatex">\(\mu_r = \frac{1}{G}\sum_{i=1}^G r_i\)</span>，<span class="arithmatex">\(\sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \mu_r)^2}\)</span>，<span class="arithmatex">\(\epsilon = 10^{-8}\)</span> 用于数值稳定。</p>

##### GRPO 目标 {#grpo-objective}

<p>GRPO 目标对概率比做截断（与近端策略优化（Proximal Policy Optimization, PPO）相同），并对参考 Policy <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 加上 KL 惩罚：</p>

<span id="eq:grpo_full"></span>

<div class="arithmatex">\[\begin{multline*}
\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{q \sim \mathcal{D},\, \{y_i\} \sim \pi_\theta(\cdot|q)} \Bigg[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \\
\min\!\left( \rho_{i,t}\, \hat{A}_i,\; \text{clip}(\rho_{i,t}, 1{-}\varepsilon, 1{+}\varepsilon)\, \hat{A}_i \right) - \beta\, \mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right] \Bigg]

\tag{13.22}
\end{multline*}\]</div>

<p>其中：</p>

<ul>
<li><p><span class="arithmatex">\(\rho_{i,t} = \dfrac{\pi_\theta(y_{i,t} \mid q, y_{i,&lt;t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid q, y_{i,&lt;t})}\)</span> 是逐 Token 的概率比</p></li>
<li><p><span class="arithmatex">\(\varepsilon \in \{0.1, 0.2\}\)</span> 是 PPO 的截断参数</p></li>
<li><p><span class="arithmatex">\(\beta &gt; 0\)</span> 控制 KL 惩罚的强度</p></li>
<li><p><span class="arithmatex">\(|y_i|\)</span> 是第 <span class="arithmatex">\(i\)</span> 个回答的长度（长度归一化可防止偏向较短回答）</p></li>
</ul>

##### KL 惩罚的具体形式 {#kl-penalty-formulation}

<p>KL 散度项逐 Token 计算：</p>



<div class="arithmatex">\[\mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right] = \mathbb{E}_{y \sim \pi_\theta(\cdot|q)} \left[ \sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t \mid q, y_{&lt;t})}{\pi_{\text{ref}}(y_t \mid q, y_{&lt;t})} \right]
\tag{13.23}\]</div>

<p>在实践中，R1 使用一种 KL 的无偏估计器，通过下列近似避免在每一步都计算 <span class="arithmatex">\(\pi_{\text{ref}}\)</span>：</p>



<div class="arithmatex">\[\mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right] \approx \frac{\pi_{\text{ref}}(y_t \mid q, y_{&lt;t})}{\pi_\theta(y_t \mid q, y_{&lt;t})} - \log \frac{\pi_{\text{ref}}(y_t \mid q, y_{&lt;t})}{\pi_\theta(y_t \mid q, y_{&lt;t})} - 1
\tag{13.24}\]</div>

<p>该估计器恒为非负，且当 <span class="arithmatex">\(\pi_\theta = \pi_{\text{ref}}\)</span> 时为零。</p>

<div class="admonition example"><p class="admonition-title">GRPO 实战：组大小与稳定性</p>
<p>在 R1 的训练中，每个问题采样 <span class="arithmatex">\(G = 8\)</span> 个回答。这是一个关键超参数：</p>

<ul>
<li><p>过小（<span class="arithmatex">\(G=2\)</span>）：优势估计方差高，训练噪声大。</p></li>
<li><p>过大（<span class="arithmatex">\(G=32\)</span>）：计算成本线性增长，收益递减。</p></li>
<li><p><span class="arithmatex">\(G=8\)</span>：经验上在方差缩减与计算成本之间取得平衡。</p></li>
</ul>

<p>分组采样还提供了天然的<strong>课程学习信号</strong>：随着训练推进，模型的平均奖励 <span class="arithmatex">\(\mu_r\)</span> 上升，方差 <span class="arithmatex">\(\sigma_r\)</span> 下降。所有 <span class="arithmatex">\(G\)</span> 个回答全对（或全错）的问题贡献零梯度，从而自然地将学习聚焦于模型能力前沿的问题上。</p>
</div>

### 13.3.4　蒸馏：R1-Distill 系列 {#distillation-the-r1-distill-series}

<p>R1 的一项重要实践贡献在于证明：<strong>推理能力可以通过在 R1 生成链上进行监督微调蒸馏到小得多的模型中</strong>。R1-Distill 系列（1.5B、7B、8B、14B、32B、70B 参数）的训练方式为：</p>

<ol>
<li><p>使用 R1（671B）为一个大规模问题集生成长 CoT 解答</p></li>
<li><p>过滤，仅保留正确的解答</p></li>
<li><p>在这些解答上对较小的基础模型（Qwen2.5、Llama-3）进行微调</p></li>
</ol>

<div class="admonition note"><p class="admonition-title">小模型：蒸馏 vs. RL</p>
<p>一个引人注目的发现：<strong>在小模型上，蒸馏胜过从零开始的 RL 训练</strong>。DeepSeek-R1-Distill-Qwen-7B 在 MATH 基准上的得分高于直接用 GRPO 训练的 7B 模型。这表明：</p>

<ul>
<li><p>小模型缺乏通过 RL 探索去发现推理策略的能力</p></li>
<li><p>但它们<em>可以</em>学会模仿大模型发现的推理策略</p></li>
<li><p>小模型的瓶颈是<em>探索</em>，而非<em>表示</em></p></li>
</ul>
</div>

<p>蒸馏路径引出了一个关于推理本质的重要问题：小模型究竟是真的在“推理”，还是只是在对推理链的表面形式进行模式匹配？经验上，蒸馏模型对新颖问题类型表现出一定的泛化能力，提示其确实在某种程度上内化了推理策略，而非纯粹的记忆。</p>

## 13.4　OpenAI o1/o3 系列 {#subsec:openai_o1}

<p>OpenAI 的 o1 <a class="book-citation" id="cite-openai2024o1-1" href="../../references.md#ref-openai2024o1" aria-label="参考文献 275">[275]</a>（2024 年 9 月发布）以及后续的 o3/o4-mini <a class="book-citation" id="cite-openai2025o3-1" href="../../references.md#ref-openai2025o3" aria-label="参考文献 278">[278]</a> 模型代表了推理模型开发的商业前沿。尽管完整的技术细节仍属专有，但已公开的系统卡（system card）、技术报告和实证观察为理解其方法论提供了充足的线索。</p>

### 13.4.1　带隐藏推理 Token 的思维链 RL {#chain-of-thought-rl-with-hidden-reasoning-tokens}

<p>o1 最具标志性的架构选择是使用<strong>隐藏推理 Token（hidden reasoning tokens）</strong>：模型生成一段不向用户展示的内部思维链（称为“推理轨迹（reasoning trace）”或“思考 Token（thinking tokens）”）。最终只返回答案本身。这一设计具有若干意涵：</p>

<ul>
<li><p><strong>无格式约束</strong>：隐藏推理可以使用任意格式，包括草稿纸式记法、伪代码，甚至非英文推理。</p></li>
<li><p><strong>风格上无奖励作弊（reward hacking）</strong>：由于用户从未看到推理过程，模型没有为了“好看”而牺牲“有用”的压力。</p></li>
<li><p><strong>专有保护</strong>：推理过程不被暴露，防止直接模仿。</p></li>
</ul>

<p>训练流程被描述为“使用 RL 训练模型进行推理”，其 RL 目标作用于完整的（隐藏推理 + 最终答案）序列，奖励仅基于最终答案的质量。</p>

### 13.4.2　过程奖励模型与结果奖励模型 {#process-reward-models-vs.-outcome-reward-models}

<p>OpenAI 的方法据信在结果奖励之外还使用了<strong>过程奖励模型（Process Reward Model, PRM）</strong> <a class="book-citation" id="cite-lightman2023lets-1" href="../../references.md#ref-lightman2023lets" aria-label="参考文献 215">[215]</a>，这与 DeepSeek-R1 仅采用结果奖励的方式形成对比。这一推断基于 OpenAI 已发表的 PRM 研究（PRM800K 数据集，&quot;Let’s Verify Step by Step&quot;）以及 o1 系统卡中对推理链 RL 训练的描述，尽管 o1/o3 的精确训练配方并未公开披露。</p>

##### 结果奖励模型（ORM） {#outcome-reward-model-orm}

<p>结果奖励模型（Outcome Reward Model, ORM）对完整回复 <span class="arithmatex">\((q, y)\)</span> 打分：</p>



<div class="arithmatex">\[R_{\text{ORM}}(q, y) \in [0, 1]
\tag{13.25}\]</div>

<p>对于可验证任务（数学、代码），它简化为精确匹配验证；对于开放式任务，则使用一个学习得到的奖励模型。</p>

##### 过程奖励模型（PRM） {#process-reward-model-prm}

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
<li><p><strong>训练数据</strong>：步骤级标签需要人工标注或自动化生成（Math-Shepherd，见 <a href="reading-03.md#subsubsec:prm_methods">13.6.2</a> 节）。</p></li>
<li><p><strong>奖励作弊</strong>：模型可能学会产生在 PRM 眼中<em>看起来</em>正确、但实际上并不正确的步骤。</p></li>
<li><p><strong>分布漂移</strong>：在某一推理链分布上训练的 PRM 可能无法泛化到 RL 产生的新颖推理链。</p></li>
</ul>

<p>经验证据表明，PRM 对<em>搜索</em>（在候选解中选择）是有益的，但对<em>训练</em>的收益则不那么明确。</p>
</div>

### 13.4.3　推理时计算扩展 {#inference-time-compute-scaling}

<p>o1 技术报告展示了一条清晰的扩展律：在困难推理任务上，<strong>更多的思考 Token 单调地提升性能</strong>。这通过一个控制隐藏推理 Token 数量上限的“思考预算（thinking budget）”参数得以实现。</p>

<p>设 <span class="arithmatex">\(T\)</span> 为思考 Token 预算。经验上观察到的扩展律近似为：</p>



<div class="arithmatex">\[\text{Pass@1}(T) \approx a - b \cdot T^{-c}
\tag{13.28}\]</div>

<p>其中 <span class="arithmatex">\(a, b, c &gt; 0\)</span> 为常数，<span class="arithmatex">\(a\)</span> 表示渐近准确率上限，<span class="arithmatex">\(c\)</span> 刻画提升速率。在 AIME 2024 上，配备完整思考预算的 o1 达到约 83% 的准确率，而未使用扩展思考的 GPT-4o 仅约 13%。</p>

### 13.4.4　训练计算量与测试时计算量 {#training-compute-vs.-test-time-compute}

<p>o1/o3 系列带来的一个根本性洞见是<strong>计算等价原理（compute equivalence principle）</strong>：训练计算量 <span class="arithmatex">\(C_{\text{train}}\)</span> 与测试时计算量 <span class="arithmatex">\(C_{\text{test}}\)</span> 之间存在一条权衡曲线，曲线上的各点能够达到相近的性能：</p>



<div class="arithmatex">\[\text{Performance}(C_{\text{train}}, C_{\text{test}}) = g\!\left(\alpha C_{\text{train}}^{p} + \beta C_{\text{test}}^{q}\right)
\tag{13.29}\]</div>

<p>经验上，对于推理任务 <span class="arithmatex">\(p \approx q\)</span>，这意味着训练计算量与测试时计算量大致可互相替代。这对部署有深远影响：一个更小、更便宜但拥有扩展思考能力的模型，可以在困难问题上匹敌更大的模型，代价是更高的延迟。</p>

### 13.4.5　o3 与 o4-mini 的架构洞察 {#o3-and-o4-mini-architecture-insights}

<p>尽管 o3 和 o4-mini 的细节大多仍属专有，但已经浮现出一些观察：</p>

<ul>
<li><p><strong>o3</strong>：思考预算显著大于 o1；在 ARC-AGI 上达到接近人类的表现（高算力下 87.5%）。据信在推理时使用了更复杂的搜索策略。</p></li>
<li><p><strong>o4-mini</strong>：证明了经过 RL 训练推理能力的<em>更小</em>模型同样极具竞争力。在 AIME 2025 上配合扩展思考达到 93%，表明在数学任务上模型规模的重要性不及推理能力。</p></li>
<li><p><strong>工具调用（Tool Calling）</strong>：o3/o4-mini 将工具调用（代码执行、网络搜索）集成到推理过程中，使模型能够以程序化方式验证中间步骤。</p></li>
</ul>

## 13.5　QwQ 与 Qwen 推理模型 {#subsec:qwq_qwen}

<p>阿里巴巴 Qwen 团队开发了一系列推理模型（QwQ-32B <a class="book-citation" id="cite-qwen2024qwq-1" href="../../references.md#ref-qwen2024qwq" aria-label="参考文献 351">[351]</a>、Qwen3 <a class="book-citation" id="cite-qwen2025qwen3-1" href="../../references.md#ref-qwen2025qwen3" aria-label="参考文献 352">[352]</a>），与 DeepSeek-R1 一同代表了开源前沿。其方法在若干关键方面有所不同。</p>

### 13.5.1　多阶段 RL 流水线 {#multi-stage-rl-pipeline}

<p>Qwen 的推理流水线采用了更精细的多阶段方法：</p>

<ol>
<li><p><strong>基础预训练</strong>：具备强大数学与编码能力的 Qwen2.5 基础模型</p></li>
<li><p><strong>多样推理任务上的 SFT</strong>：在涵盖广泛推理任务（数学、代码、科学、逻辑）的混合数据上进行监督微调（Supervised Fine-Tuning, SFT）</p></li>
<li><p><strong>拒绝采样微调（Rejection Sampling Fine-Tuning, RFT）</strong>：对每个问题生成 <span class="arithmatex">\(N\)</span> 个解，保留正确的解，再进行微调</p></li>
<li><p><strong>RL 第一阶段</strong>：在数学和代码上使用可验证奖励的组相对策略优化（Group Relative Policy Optimization, GRPO）</p></li>
<li><p><strong>RL 第二阶段</strong>：更广泛的 RL，包括指令遵循与安全</p></li>
</ol>

### 13.5.2　拒绝采样与 RL 的结合 {#rejection-sampling-rl-combination}

<p>Qwen 方法的一项关键创新是<strong>拒绝采样与 RL 的迭代结合</strong>：</p>

<ol>
<li><p><strong>初始化</strong>：以 SFT 模型作为 Policy <span class="arithmatex">\(\pi_0\)</span>。</p></li>
<li><p><strong>拒绝采样（Rejection Sampling）</strong>：采样 <span class="arithmatex">\(N\)</span> 个解 <span class="arithmatex">\(\{y_i\}_{i=1}^N \sim \pi_{k-1}(\cdot \mid q)\)</span>，保留正确的解 <span class="arithmatex">\(\mathcal{Y}^+(q) = \{y_i : r(y_i, y^*) = 1\}\)</span>。</p></li>
<li><p><strong>SFT 更新</strong>：<span class="arithmatex">\(\pi_k^{\text{SFT}} \leftarrow \text{SFT}(\pi_{k-1}, \bigcup_q \mathcal{Y}^+(q))\)</span></p></li>
<li><p><strong>RL 更新</strong>：<span class="arithmatex">\(\pi_k \leftarrow \text{GRPO}(\pi_k^{\text{SFT}}, \mathcal{D})\)</span></p></li>
<li><p><strong>重复</strong>步骤 2–4 共 <span class="arithmatex">\(K\)</span> 次迭代，得到最终 Policy <span class="arithmatex">\(\pi_K\)</span>。</p></li>
</ol>

<p>拒绝采样步骤提供高质量的正样本来锚定 Policy，而 RL 则在当前分布之外进行探索。这种组合比纯 RL 更稳定，又比纯 SFT 更有能力。</p>

### 13.5.3　工具集成推理 {#tool-integrated-reasoning}

<p>QwQ-32B 与 Qwen3 模型支持<strong>工具集成推理（tool-integrated reasoning）</strong>：模型可以在其推理链中调用外部工具（Python 解释器、搜索引擎、计算器）。这通过特殊 Token 实现：</p>

<p class="code-caption"><span class="code-number">代码 13.1　</span>QwQ 中的工具集成推理格式</p>
<div class="highlight"><pre><span></span><code><span class="o">&lt;</span><span class="n">think</span><span class="o">&gt;</span>
<span class="n">Let</span> <span class="n">me</span> <span class="n">solve</span> <span class="n">this</span> <span class="n">step</span> <span class="n">by</span> <span class="n">step</span><span class="o">.</span>
<span class="n">First</span><span class="p">,</span> <span class="n">I</span><span class="s1">'ll compute the eigenvalues of the matrix.</span>

<span class="o">&lt;</span><span class="n">tool_call</span><span class="o">&gt;</span>
<span class="p">{</span><span class="s2">"name"</span><span class="p">:</span> <span class="s2">"python"</span><span class="p">,</span> <span class="s2">"arguments"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"code"</span><span class="p">:</span> <span class="s2">"import numpy as np</span><span class="se">\n</span><span class="s2">A = np.array([[2,1],[1,3]])</span><span class="se">\n</span><span class="s2">eigenvalues = np.linalg.eigvals(A)</span><span class="se">\n</span><span class="s2">print(eigenvalues)"</span><span class="p">}}</span>
<span class="o">&lt;/</span><span class="n">tool_call</span><span class="o">&gt;</span>

<span class="o">&lt;</span><span class="n">tool_response</span><span class="o">&gt;</span>
<span class="p">[</span><span class="mf">1.38196601</span> <span class="mf">3.61803399</span><span class="p">]</span>
<span class="o">&lt;/</span><span class="n">tool_response</span><span class="o">&gt;</span>

<span class="n">The</span> <span class="n">eigenvalues</span> <span class="n">are</span> <span class="n">approximately</span> <span class="mf">1.382</span> <span class="ow">and</span> <span class="mf">3.618</span><span class="o">.</span>
<span class="n">These</span> <span class="n">are</span> <span class="p">(</span><span class="mi">5</span> <span class="o">+/-</span> <span class="n">sqrt</span><span class="p">(</span><span class="mi">5</span><span class="p">))</span><span class="o">/</span><span class="mi">2</span><span class="p">,</span> <span class="n">which</span> <span class="n">are</span> <span class="n">the</span> <span class="n">golden</span> <span class="n">ratio</span> <span class="ow">and</span> <span class="n">its</span> <span class="n">conjugate</span><span class="o">...</span>
<span class="o">&lt;/</span><span class="n">think</span><span class="o">&gt;</span>
<span class="o">&lt;</span><span class="n">answer</span><span class="o">&gt;</span><span class="n">The</span> <span class="n">eigenvalues</span> <span class="n">are</span> <span class="p">(</span><span class="mi">5</span> <span class="o">+/-</span> <span class="n">sqrt</span><span class="p">(</span><span class="mi">5</span><span class="p">))</span><span class="o">/</span><span class="mi">2</span><span class="o">&lt;/</span><span class="n">answer</span><span class="o">&gt;</span>
</code></pre></div>

<p>RL 训练奖励基于最终答案计算，但模型会学会策略性地使用工具，因为工具调用能提升得到正确答案的概率。</p>
