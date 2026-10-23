# 11.14　硬件选型指南 {#sec:hardware-guide}

<p>合适的硬件选择取决于模型规模、预算和训练阶段。</p>

<p class="table-caption" id="table-11-11"><span class="table-number">表 11.11　</span>按模型规模与训练阶段给出的硬件建议</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模型规模</strong></th><th scope="col"><strong>训练阶段</strong></th><th scope="col"><strong>推荐</strong></th><th scope="col"><strong>配置</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(\leq\)</span>7B</td><td>SFT + RLHF</td><td>1–2<span class="arithmatex">\(\times\)</span> A100</td><td>单节点，无需并行</td></tr>
<tr><td>7–13B</td><td>SFT + RLHF</td><td>4–8<span class="arithmatex">\(\times\)</span> A100</td><td>FSDP，可选生成 TP=2</td></tr>
<tr><td>13–34B</td><td>SFT + RLHF</td><td>8–16<span class="arithmatex">\(\times\)</span> A100/H100</td><td>FSDP + 生成 TP=4</td></tr>
<tr><td>70B</td><td>RLHF（完整）</td><td>32–64<span class="arithmatex">\(\times\)</span> A100/H100</td><td>解耦，FSDP + TP=8</td></tr>
<tr><td>70B</td><td>RLHF（LoRA）</td><td>8–16<span class="arithmatex">\(\times\)</span> A100/H100</td><td>无 reference，LoRA adapter</td></tr>
<tr><td><span class="arithmatex">\(&gt;\)</span>100B</td><td>RLHF</td><td>128+<span class="arithmatex">\(\times\)</span> H100</td><td>3D 并行（TP+PP+DP）</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">H100 vs A100：何时升级才值得？</p>
<p>H100 提供：</p>

<ul>
<li><p>峰值 FLOPS 约 <span class="arithmatex">\(\sim\)</span>1.6<span class="arithmatex">\(\times\)</span>（带稀疏 BF16：989 vs 624 TFLOPS；不带稀疏：495 vs 312）</p></li>
<li><p>显存带宽约 <span class="arithmatex">\(\sim\)</span>2<span class="arithmatex">\(\times\)</span>（3.35 vs 2.0 TB/s）</p></li>
<li><p>支持 FP8（推理再快 2<span class="arithmatex">\(\times\)</span>）</p></li>
<li><p>NVLink 4.0（900 vs 600 GB/s）</p></li>
</ul>

<p><strong>训练上</strong>：端到端约快 <span class="arithmatex">\(\sim\)</span>1.8–2.2<span class="arithmatex">\(\times\)</span>（FP8 支持与更高带宽放大了原始 FLOPS 优势）。</p>

<p><strong>生成上</strong>：约快 <span class="arithmatex">\(\sim\)</span>1.7<span class="arithmatex">\(\times\)</span>（带宽受限，2<span class="arithmatex">\(\times\)</span> BW 加上开销 <span class="arithmatex">\(\approx\)</span> 1.7<span class="arithmatex">\(\times\)</span> 吞吐）。</p>

<p><strong>性价比</strong>：在 1.5<span class="arithmatex">\(\times\)</span> 价格下，H100 在训练上几乎总是更划算。仅推理（生成集群）时，A100 在 spot 价格下可能更划算。</p>
</div>
