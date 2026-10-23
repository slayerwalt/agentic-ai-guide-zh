# 11.8　端到端延迟拆解 {#end-to-end-latency-breakdown}

<figure id="figure-11-11"><a href="../../assets/figures/fig_041_fig41.png" target="_blank" rel="noopener" aria-label="查看图 11.11 原图" title="点击查看原图"><img src="../../assets/figures/fig_041_fig41.png" alt="不重叠（单体式）。解耦后：生成与训练重叠，实际加速 1.4\(\times\)。" loading="lazy"></a><figcaption><span class="figure-number">图 11.11　</span>不重叠（单体式）。解耦后：生成与训练重叠，实际加速 1.4<span class="arithmatex">\(\times\)</span>。</figcaption></figure>

<span id="table-11-inline-8"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>阶段</strong></th><th scope="col"><strong>用时（70B）</strong></th><th scope="col"><strong>受限于</strong></th><th scope="col"><strong>优化手段</strong></th></tr></thead>
<tbody><tr><td>生成（128<span class="arithmatex">\(\times\)</span>512 tok）</td><td>30–45s</td><td>显存带宽</td><td>vLLM、推测解码、INT8</td></tr>
<tr><td>Reward 打分</td><td>5–8s</td><td>算力（批量前向）</td><td>INT8 RM、batch=128</td></tr>
<tr><td>Reference log-prob</td><td>4–6s</td><td>算力（批量前向）</td><td>INT8 ref，或 LoRA（免费）</td></tr>
<tr><td>PPO 更新（4 epoch）</td><td>8–12s</td><td>算力（反向）</td><td>FSDP、FlashAttention</td></tr>
<tr><td>权重同步</td><td>0–3s</td><td>网络（异步）</td><td>Delta 压缩、异步</td></tr>
<tr><td><strong>合计（单体式）</strong></td><td><strong>50–75s</strong></td><td></td><td></td></tr>
<tr><td><strong>合计（解耦+重叠）</strong></td><td><strong>35–50s</strong></td><td></td><td>生成与上一步训练重叠</td></tr></tbody>
</table>
</div>
