# 11.1　4 模型显存挑战 {#the-4-model-memory-challenge}

<figure id="figure-11-1"><a href="../../assets/figures/fig_031_fig31.png" target="_blank" rel="noopener" aria-label="查看图 11.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_031_fig31.png" alt="70B PPO 显存预算：RLHF 所需的四个模型及其显存占用。合计 1470–1560GB。朴素方案最少需要 19–20 块 A100-80GB。使用 ZeRO-3 时：8 个节点即可装下。" loading="lazy"></a><figcaption><span class="figure-number">图 11.1　</span>70B PPO 显存预算：RLHF 所需的四个模型及其显存占用。合计 1470–1560GB。朴素方案最少需要 19–20 块 A100-80GB。使用 ZeRO-3 时：8 个节点即可装下。</figcaption></figure>

<div class="admonition warning"><p class="admonition-title">显存预算现实核对——70B BF16</p>
<span id="table-11-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col">Policy 权重（BF16）</th><th scope="col">140 GB</th></tr></thead>
<tbody><tr><td>FP32 master 权重</td><td>280 GB</td></tr>
<tr><td>Adam 优化器（m + v，FP32）</td><td>560 GB</td></tr>
<tr><td>梯度（BF16）</td><td>140 GB</td></tr>
<tr><td>Reference model</td><td>140 GB（INT8 时 70 GB）</td></tr>
<tr><td>Reward model</td><td>140 GB（INT8 时 70 GB）</td></tr>
<tr><td>激活值（batch 128，seq 2048）</td><td>50–100 GB</td></tr>
<tr><td>生成用 KV cache</td><td>20–60 GB</td></tr>
<tr><td><strong>合计</strong></td><td><strong>1470–1560 GB</strong></td></tr></tbody>
</table>
</div>

<p><span class="arithmatex">\(\div\)</span> 80 GB/GPU = <strong>至少 19–20 块 A100</strong>（尚未计入任何并行开销）。</p>
</div>
