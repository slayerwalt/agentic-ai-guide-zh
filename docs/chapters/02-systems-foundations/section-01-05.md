# 2.1.5　各代际 GPU 芯片的扩展 {#gpu-chip-scaling-across-generations}

<p>NVIDIA GPU 架构的演进显示出在计算密度、片上内存和深度学习专用单元方面持续扩展：</p>

<p class="table-caption" id="table-2-3"><span class="table-number">表 2.3　</span>各代 NVIDIA 架构在 SM 层面的扩展。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>架构</strong></th><th scope="col"><strong>SM 数</strong></th><th scope="col"><strong>TCs/SM</strong></th><th scope="col"><strong>SRAM/SM</strong></th><th scope="col"><strong>L2</strong></th><th scope="col"><strong>关键变化</strong></th></tr></thead>
<tbody><tr><td>Volta (V100)</td><td>80</td><td>8</td><td>128 KB</td><td>6 MB</td><td>引入 Tensor Cores</td></tr>
<tr><td>Ampere (A100)</td><td>108</td><td>4</td><td>192 KB</td><td>40 MB</td><td>BF16/TF32；更大的 L2</td></tr>
<tr><td>Hopper (H100)</td><td>132</td><td>4</td><td>256 KB</td><td>50 MB</td><td>TMA；FP8；Thread Block Clusters</td></tr>
<tr><td>Blackwell (B200)</td><td>148</td><td>4</td><td>256 KB</td><td>128 MB</td><td>2<span class="arithmatex">\(\times\)</span> 裸片；FP4；TMEM；NVLink 5</td></tr></tbody>
</table>
</div>
