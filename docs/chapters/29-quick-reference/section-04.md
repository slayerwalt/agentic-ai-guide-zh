# 29.4　系统与并行 {#systems-parallelism}

<span id="table-29-inline-2"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>公式</strong></th><th scope="col"><strong>取值 (70B, BF16)</strong></th><th scope="col"><strong>说明</strong></th></tr></thead>
<tbody><tr><td>模型显存</td><td><span class="arithmatex">\(2P\)</span> 字节</td><td><span class="arithmatex">\(140\)</span> GB（仅权重）</td></tr>
<tr><td>Adam 优化器</td><td><span class="arithmatex">\(2P \times 4\)</span> 字节 (m + v)</td><td><span class="arithmatex">\(280\)</span> GB</td></tr>
<tr><td>完整训练占用</td><td><span class="arithmatex">\(\sim 8P\)</span> 字节</td><td><span class="arithmatex">\(560\)</span> GB（权重 + 优化器 + 梯度）</td></tr>
<tr><td>FSDP 每 GPU 显存</td><td><span class="arithmatex">\(8P / N_\text{GPUs}\)</span></td><td>8 卡时为 <span class="arithmatex">\(70\)</span> GB</td></tr>
<tr><td>生成算术强度</td><td><span class="arithmatex">\(2P / 2P = 1\)</span> FLOP/byte</td><td>严重内存瓶颈</td></tr>
<tr><td>Token 速率（生成）</td><td>HBM_BW <span class="arithmatex">\(/ (2P)\)</span></td><td><span class="arithmatex">\(\sim\)</span>14 tok/s (A100, batch=1)</td></tr>
<tr><td>TP AllReduce / 层</td><td><span class="arithmatex">\(2 \times 2 \cdot \frac{T-1}{T} \cdot bsd\)</span> 字节</td><td><span class="arithmatex">\(\sim\)</span>188 MB (70B, TP=8)</td></tr>
<tr><td>PP 气泡占比</td><td><span class="arithmatex">\((P-1)/(P+M-1)\)</span></td><td><span class="arithmatex">\(P\)</span>=阶段数，<span class="arithmatex">\(M\)</span>=微批数</td></tr>
<tr><td>MFU</td><td>observed_toks <span class="arithmatex">\(\times\)</span> 6<span class="arithmatex">\(P\)</span> / peak_FLOPS</td><td>目标：<span class="arithmatex">\(&gt;40\%\)</span></td></tr></tbody>
</table>
</div>
