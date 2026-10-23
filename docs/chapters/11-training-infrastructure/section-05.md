# 11.5　权重同步策略 {#weight-synchronization-strategies}

<span id="table-11-inline-6"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>策略</strong></th><th scope="col"><strong>滞后</strong></th><th scope="col"><strong>带宽</strong></th><th scope="col"><strong>质量影响</strong></th></tr></thead>
<tbody><tr><td>同步（每步）</td><td>0 步</td><td>140 GB/步</td><td>完美但太慢</td></tr>
<tr><td>周期性（每 50 步）</td><td>平均 25 步</td><td>摊销后 2.8 GB/步</td><td>质量损失 <span class="arithmatex">\(&lt;\)</span>2%</td></tr>
<tr><td>Delta 压缩（INT8）</td><td>平均 25 步</td><td>0.4 GB/步</td><td>质量损失 <span class="arithmatex">\(&lt;\)</span>3%</td></tr>
<tr><td>异步流式</td><td>5–10 步</td><td>14 GB/步（后台）</td><td>质量损失 <span class="arithmatex">\(&lt;\)</span>1%</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">为什么 PPO/GRPO 能容忍滞后</p>
<p>PPO 的裁剪目标本来就是<em>为</em>离策略数据设计的！裁剪区间 <span class="arithmatex">\([1-\epsilon, 1+\epsilon]\)</span> 限制了陈旧数据的影响。当滞后在 10–50 步时：</p>

<ul>
<li><p>每步 policy 大约变化 <span class="arithmatex">\(\sim\)</span>0.1–1%（合理学习率下）</p></li>
<li><p>50 步累计：<span class="arithmatex">\(\sim\)</span>5% 的 policy 漂移</p></li>
<li><p>PPO 裁剪本身就能处理高达 20% 的漂移</p></li>
<li><p>经验上：50 步滞后的质量损失 <span class="arithmatex">\(&lt;\)</span>2%</p></li>
</ul>

<p><strong>带宽算账</strong>：70B BF16 = 140GB。InfiniBand 400Gb/s = 50GB/s <span class="arithmatex">\(\rightarrow\)</span> 完整同步 2.8s。配合 delta 压缩：<span class="arithmatex">\(&lt;\)</span>0.5s。异步同步 = 免费（在后台跑）。</p>
</div>
