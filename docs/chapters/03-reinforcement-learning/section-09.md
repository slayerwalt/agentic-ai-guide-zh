# 3.9　同策略 vs 异策略——详细对比 {#on-policy-vs-off-policy-detailed-comparison}

<span id="table-3-inline-2"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"></th><th scope="col"><strong>On-Policy</strong></th><th scope="col"><strong>Off-Policy</strong></th></tr></thead>
<tbody><tr><td><strong>数据来源</strong></td><td>仅当前策略 <span class="arithmatex">\(\pi_\theta\)</span></td><td>任意策略（重放缓冲区）</td></tr>
<tr><td><strong>更新之后</strong></td><td>旧数据失效，必须重新生成</td><td>旧数据仍可使用</td></tr>
<tr><td><strong>样本效率</strong></td><td>低（数据仅用一次）</td><td>高（数据可多次复用）</td></tr>
<tr><td><strong>稳定性</strong></td><td>更稳定（分布一致）</td><td>可能发散（分布不匹配）</td></tr>
<tr><td><strong>示例</strong></td><td>REINFORCE、PPO、A2C、GRPO</td><td>Q-Learning、DQN、SAC、DPO</td></tr>
<tr><td><strong>用于 LLM</strong></td><td>PPO、GRPO（每步重新生成）</td><td>DPO（静态偏好数据集）</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">RLHF 方法的同/异策略属性</p>
<p><strong>PPO/GRPO 是同策略</strong>：用当前策略生成回复，计算优势，更新，丢弃数据，再次生成。这就是为何生成占据 60% 算力——每一步都要重新生成。</p>

<p><strong>DPO 是异策略</strong>：在固定的偏好数据集上训练。训练过程中无需生成。便宜得多，但会遭受分布偏移问题（随策略变化数据逐渐过时）。</p>

<p><strong>在线 DPO 是混合体</strong>：生成新鲜数据（同策略生成），但使用 DPO 的监督损失（异策略式优化）。兼具两者优势。</p>

<p><strong>PPO 的巧妙之处</strong>：使用裁剪比率 <span class="arithmatex">\(r = \pi_\text{new}/\pi_\text{old}\)</span>，从同一批同策略数据中榨取多步梯度更新（通常 4 个 epoch），以可控方式使其变得「略带异策略」。</p>
</div>
