# 6.6　DPO 变体及其各自的失败场景 {#dpo-variants-and-when-each-fails}

<div class="admonition warning"><p class="admonition-title">DPO 何时会失败</p>
<p><strong>1. 分布漂移</strong>：偏好数据来自旧模型。当前 policy 生成的文本截然不同 <span class="arithmatex">\(\rightarrow\)</span> loss 在无关样本上做优化。</p>

<p><strong>2. 无探索</strong>：无法发现数据集之外的行为。陷入局部最优。</p>

<p><strong>3. Reference 崩塌</strong>：Reference 太强，policy 动弹不得；太弱则失去正则化作用。</p>

<p><strong>4. 数据质量</strong>：噪声标签会毒化训练。与对大量样本平均的 PPO 不同，DPO 会“记住”单个的样本对。</p>

<p><strong>5. 偏好数据多样性</strong>：要确保被选/被拒成对样本覆盖质量差异的完整谱系（而不仅是“好 vs 极差”）。在<em>解法思路</em>上有差异的样本对（而不仅是质量差异）能教出更丰富的 policy 区分能力。</p>
</div>
