# 8.6　总结：如何选择对齐方法 {#sec:method-comparison}

<p>我们已经纵览了偏好优化与基于 RL 的对齐方法的完整全景。本节将关键权衡整合为一份参考，帮助实践者根据自身约束选择合适方案。</p>

<p class="table-caption" id="table-8-1"><span class="table-number">表 8.1　</span>各对齐方法的横向对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>模型</strong></th><th scope="col"><strong>数据</strong></th><th scope="col"><strong>算力</strong></th><th scope="col"><strong>稳定性</strong></th><th scope="col"><strong>最适合</strong></th></tr></thead>
<tbody><tr><td>PPO</td><td>4</td><td>在线（生成）</td><td>非常高</td><td>低</td><td>极致质量、复杂推理</td></tr>
<tr><td>GRPO</td><td>2（无 critic）</td><td>在线（生成）</td><td>高</td><td>中</td><td>数学/代码（可验证 reward）</td></tr>
<tr><td>DPO</td><td>2</td><td>离线 pair</td><td>低</td><td>高</td><td>风格/安全、算力有限</td></tr>
<tr><td>Online DPO</td><td>3</td><td>在线（生成）</td><td>中</td><td>中-高</td><td>无分布偏移的 DPO</td></tr>
<tr><td>KTO</td><td>2</td><td>未配对二值</td><td>低</td><td>高</td><td>生产反馈、点赞/点踩</td></tr>
<tr><td>IPO</td><td>2</td><td>离线 pair</td><td>低</td><td>非常高</td><td>噪声标签、抗过拟合</td></tr>
<tr><td>ORPO</td><td>1</td><td>离线 pair</td><td>非常低</td><td>高</td><td>显存受限、SFT+对齐合并</td></tr>
<tr><td>Best-of-N</td><td>1+RM</td><td>在线（生成）</td><td>中</td><td>完美</td><td>强基线、数据生成</td></tr></tbody>
</table>
</div>

<figure id="figure-8-1"><a href="../../assets/figures/fig_029_fig29.png" target="_blank" rel="noopener" aria-label="查看图 8.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_029_fig29.png" alt="近似的质量 vs. 算力前沿。位于 SFT 上限线之上的方法超越了单纯监督微调所能达到的水平。位置仅为示意，与模型有关。" loading="lazy"></a><figcaption><span class="figure-number">图 8.1　</span>近似的质量 vs. 算力前沿。位于 SFT 上限线之上的方法超越了单纯监督微调所能达到的水平。位置仅为示意，与模型有关。</figcaption></figure>

<div class="admonition note"><p class="admonition-title">决策树：该用哪种方法？</p>
<ol>
<li><p><strong>有可验证 reward 吗？</strong>（数学/代码）<span class="arithmatex">\(\rightarrow\)</span> <strong>GRPO</strong></p></li>
<li><p><strong>在复杂任务上需要极致质量吗？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>PPO</strong></p></li>
<li><p><strong>有成对偏好吗？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>DPO</strong>（若噪声大则用 IPO）</p></li>
<li><p><strong>仅有未配对的二值反馈？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>KTO</strong></p></li>
<li><p><strong>显存受限、从 base 模型起步？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>ORPO</strong></p></li>
<li><p><strong>DPO 停滞、想要 on-policy？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>Online DPO</strong></p></li>
<li><p><strong>需要快速建立强基线？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>Best-of-N / RFT</strong></p></li>
</ol>
</div>
