# 29.6　超参数范围 {#hyperparameter-ranges}

<span id="table-29-inline-4"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>参数</strong></th><th scope="col"><strong>典型范围</strong></th><th scope="col"><strong>默认值</strong></th><th scope="col"><strong>说明</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(\beta\)</span> (DPO/KTO)</td><td>0.05–0.5</td><td>0.1</td><td>越大越保守</td></tr>
<tr><td><span class="arithmatex">\(\epsilon\)</span> (PPO clip)</td><td>0.1–0.3</td><td>0.2</td><td>越大更新越激进</td></tr>
<tr><td><span class="arithmatex">\(\gamma\)</span> (GAE 折扣)</td><td>0.99–1.0</td><td>1.0</td><td>情景式任务使用 1.0</td></tr>
<tr><td><span class="arithmatex">\(\lambda\)</span> (GAE)</td><td>0.9–0.99</td><td>0.95</td><td>越小偏差越大、方差越小</td></tr>
<tr><td>KL 系数 (<span class="arithmatex">\(\beta_\text{KL}\)</span>)</td><td>0.01–0.2</td><td>0.05</td><td>自适应目标 KL <span class="arithmatex">\(\approx\)</span> 5–8</td></tr>
<tr><td>学习率 (RLHF)</td><td>1e-7 – 5e-6</td><td>5e-7</td><td>远低于预训练</td></tr>
<tr><td>学习率 (SFT)</td><td>1e-5 – 5e-5</td><td>2e-5</td><td>标准微调范围</td></tr>
<tr><td>LoRA 秩 <span class="arithmatex">\(r\)</span></td><td>8–128</td><td>16–64</td><td>越大容量越大、显存越多</td></tr>
<tr><td>LoRA alpha <span class="arithmatex">\(\alpha\)</span></td><td><span class="arithmatex">\(r\)</span> – <span class="arithmatex">\(2r\)</span></td><td><span class="arithmatex">\(2r\)</span></td><td>缩放因子；<span class="arithmatex">\(\alpha/r\)</span> 为有效尺度</td></tr>
<tr><td>温度（生成）</td><td>0.6–1.0</td><td>0.7</td><td>越低候选越同质</td></tr>
<tr><td>生成数量 <span class="arithmatex">\(K\)</span></td><td>4–64</td><td>4–16</td><td>用于 GRPO / Online DPO / Best-of-N</td></tr>
<tr><td>梯度裁剪范数</td><td>0.5–2.0</td><td>1.0</td><td>防止梯度爆炸</td></tr></tbody>
</table>
</div>
