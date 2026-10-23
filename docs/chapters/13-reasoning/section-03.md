# 13.3　DeepSeek-R1 {#subsec:deepseek_r1}

<p>DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 是首个在主要基准上达到或超越 OpenAI o1 的完全开源大型推理模型。其训练流程在技术上完全透明，已成为基于 RL 的推理的事实参考实现。</p>

## 13.3.1　两阶段训练流程 {#two-stage-training-pipeline}

#### 阶段 1：冷启动监督微调 {#stage-1-cold-start-supervised-fine-tuning}

<p>基础模型（DeepSeek-V3）首先在一个经过精心整理的小规模长思维链示例数据集上做微调。这个“冷启动”阶段服务于两个目的：</p>

<ol>
<li><p><strong>格式初始化</strong>：模型学会在给出最终答案之前，以 <code>&lt;think&gt;...&lt;/think&gt;</code> 的格式输出推理。</p></li>
<li><p><strong>稳定性</strong>：若没有冷启动 SFT，直接在基础模型上从零进行纯 RL 会导致训练动力学不稳定与退化输出（例如混语、重复循环）。</p></li>
</ol>

<p>冷启动数据集仅包含 <span class="arithmatex">\(\sim\)</span> 数千条样本，刻意保持较小规模，以避免过度约束 RL 后续将发现的推理风格。</p>

#### 阶段 2：基于 GRPO 的强化学习 {#stage-2-grpo-based-reinforcement-learning}

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

## 13.3.2　奖励设计：准确率奖励与格式奖励 {#reward-design-accuracy-and-format-rewards}

<p>R1 的一个关键设计选择是<strong>不使用过程奖励模型（PRM）</strong>。取而代之，R1 使用两种简单且可自动计算的奖励：</p>

#### 准确率奖励 {#accuracy-reward}

<p>对答案可验证的数学问题：</p>



<div class="arithmatex">\[r_{\text{acc}}(y, y^*) = \begin{cases} 1 &amp; \text{if } \texttt{verify}(y, y^*) = \texttt{True} \\ 0 &amp; \text{otherwise} \end{cases}
\tag{13.16}\]</div>

<p>其中 <span class="arithmatex">\(y\)</span> 是模型的最终答案（从 <code>&lt;answer&gt;</code> 标签中提取），<span class="arithmatex">\(y^*\)</span> 是真实答案。<code>verify</code> 函数使用符号数学比较（如 SymPy）来处理等价形式。</p>

<p>对于代码问题，准确率奖励由通过测试用例决定：</p>



<div class="arithmatex">\[r_{\text{acc}}^{\text{code}}(y, \mathcal{T}) = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \mathbf{1}[\texttt{execute}(y, t) = \texttt{expected}(t)]
\tag{13.17}\]</div>

#### 格式奖励 {#format-reward}

<p>用于强制 <code>&lt;think&gt;...&lt;/think&gt;</code> 结构：</p>



<div class="arithmatex">\[r_{\text{fmt}}(y) = \begin{cases} 1 &amp; y \text{ 含有有效的 &lt;think&gt; 与 &lt;answer&gt; 标签} \\ 0 &amp; \text{否则} \end{cases}
\tag{13.18}\]</div>

#### 组合奖励 {#combined-reward}



<div class="arithmatex">\[r(y, y^*) = r_{\text{acc}}(y, y^*) + \lambda_{\text{fmt}} \cdot r_{\text{fmt}}(y)
\tag{13.19}\]</div>

<p>在原始实现中 <span class="arithmatex">\(\lambda_{\text{fmt}} = 0.1\)</span>（足够小以不至于主导优化，足够大以防止格式崩塌）。</p>

<div class="admonition warning"><p class="admonition-title">没有过程奖励模型</p>
<p>R1 一个引人注目且令人意外的发现是：<strong>不需要过程奖励模型（PRM）</strong>。尽管推理链很长，仅靠结果奖励就足以让 RL 发现高质量的推理策略。作者推测，数学/代码奖励的可验证性提供了足够的信号，而 PRM 自身又会引入新的失效模式（步级别的奖励黑客）。这与 OpenAI 所采取的方法形成对比（见第 <a href="section-04.md#subsec:openai_o1">13.4</a> 节）。</p>
</div>

## 13.3.3　R1 中的 GRPO 公式 {#subsubsec:grpo_r1_math}

<p>GRPO <a class="book-citation" id="cite-shao2024deepseekmath-1" href="../../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a> 是一种策略梯度方法，它通过对一<em>组</em>采样回答估计优势，从而避免训练独立的价值网络。对于一个问题 <span class="arithmatex">\(q\)</span>，GRPO 从当前 Policy <span class="arithmatex">\(\pi_\theta\)</span> 采样 <span class="arithmatex">\(G\)</span> 个回答 <span class="arithmatex">\(\{y_1, y_2, \ldots, y_G\}\)</span>，并相对组均值计算优势。</p>

#### 分组采样与优势归一化 {#group-sampling-and-advantage-normalization}

<p>给定问题 <span class="arithmatex">\(q\)</span>，采样 <span class="arithmatex">\(G\)</span> 个输出：</p>



<div class="arithmatex">\[\{y_i\}_{i=1}^G \sim \pi_\theta(\cdot \mid q)
\tag{13.20}\]</div>

<p>使用式 [eq:r1_combined_reward] 的奖励函数计算 Reward <span class="arithmatex">\(\{r_i\}_{i=1}^G\)</span>。第 <span class="arithmatex">\(i\)</span> 个回答的归一化优势为：</p>



<div class="arithmatex">\[\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}
\tag{13.21}\]</div>

<p>其中 <span class="arithmatex">\(\mu_r = \frac{1}{G}\sum_{i=1}^G r_i\)</span>，<span class="arithmatex">\(\sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \mu_r)^2}\)</span>，<span class="arithmatex">\(\epsilon = 10^{-8}\)</span> 用于数值稳定。</p>

#### GRPO 目标 {#grpo-objective}

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

#### KL 惩罚的具体形式 {#kl-penalty-formulation}

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

## 13.3.4　蒸馏：R1-Distill 系列 {#distillation-the-r1-distill-series}

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
