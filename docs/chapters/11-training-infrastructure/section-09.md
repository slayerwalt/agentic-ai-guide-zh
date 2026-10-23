# 11.9　监控与可观测性 {#monitoring-and-observability}

<div class="admonition note"><p class="admonition-title">RLHF 训练中要追踪的关键指标</p>
<p><strong>质量指标</strong>（每 10 步记录）：</p>

<ul>
<li><p>平均 reward（应当先上升后趋稳）</p></li>
<li><p>与 reference 的 KL 散度（应当保持在 3–10）</p></li>
<li><p>响应长度分布（警惕 length hacking）</p></li>
<li><p>Entropy（应缓慢下降，不能崩塌）</p></li>
</ul>

<p><strong>系统指标</strong>（每步记录）：</p>

<ul>
<li><p>GPU 利用率（目标：训练 <span class="arithmatex">\(&gt;\)</span>80%，生成 <span class="arithmatex">\(&gt;\)</span>60%）</p></li>
<li><p>每 GPU 的显存水位（在 OOM 发生前发现）</p></li>
<li><p>生成吞吐（tokens/秒，应稳定）</p></li>
<li><p>Gradient norm（尖峰 = 即将出现不稳定）</p></li>
<li><p>NCCL 通信时间（侦测网络劣化）</p></li>
</ul>
</div>
