# 28.5　GRPO 变体与高级 RL 题 {#grpo-variants-and-advanced-rl-questions}

<section class="quiz-question" id="q-21"><h2 class="question-title">Q21：什么是 DAPO？它相对标准 GRPO 有哪些改进？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：DAPO（动态自适应策略优化，Dynamic Adaptive Policy Optimization）引入了 5 项关键改动：</p>

<p><strong>1. Clip-Higher（非对称裁剪）</strong>：标准 PPO/GRPO 两侧都按 <span class="arithmatex">\(\epsilon=0.2\)</span> 等量裁剪。DAPO 使用 <span class="arithmatex">\(\epsilon_\text{low}=0.2\)</span> 但 <span class="arithmatex">\(\epsilon_\text{high}=0.28\)</span>。这允许模型<em>更激进地提升</em>好动作的概率，同时仍限制对坏动作的抑制幅度。直觉：探索比利用需要更大的空间。</p>

<p><strong>2. 超长过滤（Overlong Filtering）</strong>：若回答达到最大长度上限（被截断，没有 EOS Token），就将其完全从 loss 中屏蔽。理由：截断的回答不包含自然停止信号——在其上训练会教会模型“句子中间停下”是可接受的。</p>

<p><strong>3. Token 级 Loss</strong>：loss 按所有序列的总 Token 数归一，而非按序列数。这防止更长的序列主导梯度。</p>

<p><strong>4. 软超长惩罚（Soft Overlong Punishment）</strong>：不再是二元截断过滤，而是在回答接近最大长度时施加渐进惩罚。<span class="arithmatex">\(r_\text{soft} = -c \cdot \max(0, \text{len} - L_\text{soft})/(L_\text{max} - L_\text{soft})\)</span>。</p>

<p><strong>5. 动态采样（Dynamic Sampling）</strong>：训练中重采样 prompt，确保每个 batch 都有成功/失败的混合（TRL 中尚未实现）。</p>

<p><strong>适用场景</strong>：需要极大探索与超长补全（32K+ Token）的大规模推理 RL。非对称裁剪尤其有价值。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../08-preference-optimization/index.md">8</a> 章（GRPO；偏好优化变体）。</em></p>
</details></section>

<section class="quiz-question" id="q-22"><h2 class="question-title">Q22：解释 vLLM 的训练-推理不一致。为何发生，TIS/MIS 如何修复？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：当用 vLLM 做生成、用训练框架（DeepSpeed/FSDP）做更新时，同一模型、同一权重会产生<em>不同</em>的 Token 概率。原因：</p>

<ul>
<li><p>数值 kernel 不同（vLLM 使用针对吞吐优化的自定义 CUDA kernel）。</p></li>
<li><p>attention 实现不同（训练用 FlashAttention，vLLM 用 PagedAttention）。</p></li>
<li><p>精度处理不同（vLLM 用 FP8/INT8，训练用 BF16）。</p></li>
<li><p>批处理差异影响 layer normalization 的数值。</p></li>
</ul>

<p>这会悄无声息地破坏 PPO 的 on-policy 假设：我们计算 <span class="arithmatex">\(\pi_\theta/\pi_\text{old}\)</span> 时，<span class="arithmatex">\(\pi_\text{old}\)</span> 来自 vLLM 而 <span class="arithmatex">\(\pi_\theta\)</span> 来自训练框架。从第零步起比率就是错的！</p>

<p><strong>TIS（截断重要性采样，Truncated Importance Sampling）</strong>：通过乘以 <span class="arithmatex">\(\min(\pi_\text{train}/\pi_\text{inference}, C)\)</span> 修正梯度。<span class="arithmatex">\(\min\)</span> 与上限 <span class="arithmatex">\(C\)</span> 防止极端修正破坏训练稳定性。典型 <span class="arithmatex">\(C=2.0\)</span>。</p>

<p><strong>MIS（掩码重要性采样，Masked Importance Sampling）</strong>：更激进——直接丢弃任何 <span class="arithmatex">\(\pi_\text{train}/\pi_\text{inference} &gt; C\)</span> 的 Token，对梯度贡献为零。防止任何估计糟糕的 Token 影响更新。</p>

<p><strong>序列级 vs Token 级</strong>：序列级 IS 理论上正确（无偏）；Token 级 IS 有偏但方差更低。实践中，序列级带截断效果最好。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（GRPO；系统架构）。</em></p>
</details></section>

<section class="quiz-question" id="q-23"><h2 class="question-title">Q23：GSPO vs GRPO——根本区别在哪？什么时候重要？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>GRPO</strong>：<em>逐 Token</em>计算重要性比 <span class="arithmatex">\(w_{i,t} = \pi_\theta(o_{i,t}|q, o_{i,&lt;t}) / \pi_\text{old}(o_{i,t}|q, o_{i,&lt;t})\)</span>，然后独立裁剪每个 Token。</p>

<p><strong>GSPO</strong>：在<em>序列级</em>计算重要性比：<span class="arithmatex">\(s_i(\theta) = (\pi_\theta(o_i|q)/\pi_\text{old}(o_i|q))^{1/|o_i|}\)</span>——Token 概率的几何均值。裁剪这单一的序列级比率。</p>

<p><strong>为何重要</strong>：GRPO 的逐 Token 裁剪把每个 Token 视为独立，但语言中它们高度相关。序列前段的微小逐 Token 变化在多个 Token 上指数级放大。GSPO 通过审视完整序列概率来捕捉这一点。</p>

<p><strong>长度归一化</strong>：<span class="arithmatex">\(1/|o_i|\)</span> 指数保证不同长度序列间的公平比较。否则更长的序列总是有更低的概率比。</p>

<p><strong>何时用 GSPO</strong>：当训练变为 off-policy 时（<code>steps_per_generation &gt; 1</code> 或 <code>num_iterations &gt; 1</code>）。如果完全 on-policy（比率 <span class="arithmatex">\(\approx 1\)</span>），GRPO 与 GSPO 等价。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../08-preference-optimization/index.md">8</a> 章（GRPO；偏好优化变体）。</em></p>
</details></section>

<section class="quiz-question" id="q-24"><h2 class="question-title">Q24：论文 “It Takes Two” 表明 G=2 与 G=16 效果相当。这怎么可能？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：关键洞见是 GRPO 的有效性并非来自精确的 advantage 估计（那需要大 <span class="arithmatex">\(G\)</span>），而来自一个<strong>隐式的对比目标</strong>。</p>

<p><span class="arithmatex">\(G=2\)</span> 加二元 reward（一个对一个错）时：归一化后 <span class="arithmatex">\(\hat{A}_\text{correct} = +1\)</span>，<span class="arithmatex">\(\hat{A}_\text{wrong} = -1\)</span>。loss 变成：提升正确回答的概率、降低错误回答的概率。这本质上就是一个 DPO 风格的对比 loss！</p>

<p><strong>为什么大 <span class="arithmatex">\(G\)</span> 帮助不大</strong>：归一化 advantage <span class="arithmatex">\(\hat{A}_i = (r_i - \mu)/\sigma\)</span> 本身已制造了好坏之间的对比。更多样本能更准确估计 <span class="arithmatex">\(\mu\)</span>，但梯度方向由最好与最差之间的<em>对比</em>主导，而非 <span class="arithmatex">\(\mu\)</span> 的精度。</p>

<p><strong>算力节省</strong>：<span class="arithmatex">\(G=2\)</span> 意味着生成算力比 <span class="arithmatex">\(G=16\)</span> 少 8<span class="arithmatex">\(\times\)</span>。由于生成占训练时间 60%，整体训练加速 <span class="arithmatex">\(\sim\)</span>4<span class="arithmatex">\(\times\)</span>。</p>

<p><strong>注意</strong>：通过率在 30–70% 时效果最佳。如果通过率极低（<span class="arithmatex">\(&lt;\)</span>10%），<span class="arithmatex">\(G=2\)</span> 常常给出两个失败（无信号）。难题需要更大的 <span class="arithmatex">\(G\)</span>。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 章（GRPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-25"><h2 class="question-title">Q25：什么是 SAPO？它的软门控与硬裁剪有何不同？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：标准 PPO/GRPO 使用硬裁剪：<span class="arithmatex">\(\text{clip}(r, 1-\epsilon, 1+\epsilon)\)</span>。在边界处梯度突然降为零。这制造了一个“死区”，模型在其中收不到任何学习信号。</p>

<p><strong>SAPO</strong> 用平滑的 sigmoid 门控取代它：随着比率偏离 1，梯度被逐渐衰减，绝不会突兀归零。它使用非对称温度：</p>

<ul>
<li><p>对正 advantage 用 <span class="arithmatex">\(\tau_+ = 1.0\)</span>（标准衰减）。</p></li>
<li><p>对负 advantage 用 <span class="arithmatex">\(\tau_- = 1.05\)</span>（对抑制略激进一些的衰减）。</p></li>
</ul>

<p><strong>收益</strong>：(1) 梯度地形上没有“悬崖”。(2) 略超出裁剪范围的 Token 仍有贡献（衰减而非归零）。(3) 优化轨迹更稳定。(4) 序列连贯——考虑了完整序列上下文。</p>

<p><strong>权衡</strong>：信任域比硬裁剪略宽松，因此需要仔细调温度。但整体对超参更鲁棒。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../08-preference-optimization/index.md">8</a> 章（GRPO；偏好优化变体）。</em></p>
</details></section>
