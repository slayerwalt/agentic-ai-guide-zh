# 1.11　LLM 训练中的多样性 {#diversity-in-llm-training}

<p>多样性——在训练数据、模型输出和优化轨迹上的多样性——对于防止模式坍缩并确保鲁棒的通用 LLM 至关重要。本节介绍适用于所有 LLM 训练阶段的关键多样性机制。</p>

## 1.11.1　采样多样性 {#sampling-diversity}

<div class="admonition note"><p class="admonition-title">用于多样化生成的采样策略</p>
<ul>
<li><p><strong>温度 <span class="arithmatex">\(\tau\)</span></strong>：<span class="arithmatex">\(P(x_i) \propto \exp(\text{logit}_i / \tau)\)</span>。更高的 <span class="arithmatex">\(\tau\)</span> = 更均匀的分布 = 更多样化。典型值：RLHF 生成中使用 <span class="arithmatex">\(\tau=0.7\)</span>–<span class="arithmatex">\(1.0\)</span>。</p></li>
<li><p><strong>Top-<span class="arithmatex">\(k\)</span></strong>：只从概率最高的 <span class="arithmatex">\(k\)</span> 个 Token 中采样。防止退化的低概率 Token。</p></li>
<li><p><strong>Top-<span class="arithmatex">\(p\)</span>（核采样，Nucleus Sampling）</strong>：从累计概率 <span class="arithmatex">\(\geq p\)</span> 的最小 Token 集合中采样。自适应：模型不确定时更多样。</p></li>
<li><p><strong>Min-<span class="arithmatex">\(p\)</span></strong>：仅保留满足 <span class="arithmatex">\(P \geq p_{\min} \times P_{\max}\)</span> 的 Token。比 top-<span class="arithmatex">\(k\)</span> 更具原理性。</p></li>
<li><p><strong>频率/出现惩罚（Frequency/presence penalty）</strong>：对已在回复中出现过的 Token 进行惩罚。鼓励词汇多样性。</p></li>
</ul>
</div>

## 1.11.2　训练数据多样性 {#training-data-diversity}

<ul>
<li><p><strong>提示多样性</strong>：覆盖不同的领域、难度等级和格式。Goldilocks 原则：提示的成功率应在 20–80% 之间。</p></li>
<li><p><strong>去重</strong>：移除近似重复的训练样本（MinHash、n-gram 重叠）。重复样本会导致对特定模式的过拟合。</p></li>
<li><p><strong>数据混合</strong>：使用温度加权采样或课程策略，在任务/领域之间进行平衡。</p></li>
</ul>

## 1.11.3　促进多样性的方法 {#diversity-promoting-methods}

<span id="table-1-inline-21"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>它如何促进多样性</strong></th></tr></thead>
<tbody><tr><td>温度缩放（Temperature scaling）</td><td>更高的 <span class="arithmatex">\(\tau\)</span> 会让分布变得更平坦；更多 Token 变得合理。</td></tr>
<tr><td>Top-<span class="arithmatex">\(p\)</span> / Min-<span class="arithmatex">\(p\)</span></td><td>自适应阈值允许模型在不确定时进行更广的采样。</td></tr>
<tr><td>频率惩罚（Frequency penalty）</td><td>惩罚重复出现的 Token，强制在一次回复内产生词汇多样性。</td></tr>
<tr><td>数据去重</td><td>从训练数据中移除近似重复样本，防止对特定模式的过拟合。</td></tr>
<tr><td>多领域混合</td><td>跨领域的温度加权采样确保了广泛的覆盖。</td></tr>
<tr><td>口头化采样（Verbalized sampling）</td><td>提示模型显式地用语言表达出对回复的概率分布 <a class="book-citation" id="cite-zhang2025verbalized-1" href="../../references.md#ref-zhang2025verbalized" aria-label="参考文献 422">[422]</a>。见 §<a href="../07-grpo/section-05.md#grpo-variants-and-extensions">7.5</a>。</td></tr></tbody>
</table>
</div>
