# 9.3　奖励模型训练技巧 {#sec:rm-tricks}

### Reward 中心化 {#reward-centering}

<p>原始的奖励模型输出可能存在任意尺度和偏移。对 Reward 进行中心化（减去均值）可以稳定 RL 训练：</p>



<div class="arithmatex">\[r_{\text{centered}}(y, q) = r_\phi(y, q) - \mathbb{E}_{y&#x27; \sim \pi_\theta}[r_\phi(y&#x27;, q)].\]</div>

<p>在 TRL 中，这通过 <code>center_rewards_coefficient</code> 参数实现，它会向奖励模型 Loss 中添加一个正则项，惩罚均值非零的 Reward。</p>

### 长度偏差校正 {#length-bias-correction}

<p>奖励模型往往存在<em>长度偏差</em>：无论质量如何，都倾向于给较长的回答分配更高的 Reward。可通过以下方式校正：</p>

<ol>
<li><p><strong>长度归一化（Length normalisation）</strong>：将 Reward 除以回答长度。</p></li>
<li><p><strong>长度受控训练（Length-controlled training）</strong>：将长度作为特征，训练模型使其对长度不变。</p></li>
<li><p><strong>校准（Calibration）</strong>：事后回归以剔除长度效应。</p></li>
</ol>

### Margin Loss {#margin-losses}

<p>在 Bradley-Terry Loss 中加入 margin <span class="arithmatex">\(m\)</span>，可确保奖励模型对偏好回答与非偏好回答打出具有显著区分度的分数：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{margin}} = \max\!\bigl(0,\; m - (r_w - r_l)\bigr).\]</div>
