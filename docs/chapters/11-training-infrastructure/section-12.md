# 11.12　成本分析与云端部署 {#cost-analysis-and-cloud-deployment}

<p>理解 RLHF 训练的经济账，对规划至关重要。</p>

## 11.12.1　硬件成本对比 {#hardware-cost-comparison}

<p class="table-caption" id="table-11-9"><span class="table-number">表 11.9　</span>RLHF 训练大致云端 GPU 成本（2024–2025 价格）</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>GPU</strong></th><th scope="col"><strong>按需/小时</strong></th><th scope="col"><strong>Spot/小时</strong></th><th scope="col"><strong>显存</strong></th><th scope="col"><strong>适用场景</strong></th></tr></thead>
<tbody><tr><td>A100 80GB</td><td>$2.50–3.50</td><td>$1.00–1.50</td><td>80 GB HBM2e</td><td>经济型训练、生成集群</td></tr>
<tr><td>H100 80GB</td><td>$4.00–6.00</td><td>$2.00–3.00</td><td>80 GB HBM3</td><td>生产级训练</td></tr>
<tr><td>H200 141GB</td><td>$6.00–8.00</td><td>—</td><td>141 GB HBM3e</td><td>大上下文、少卡配置</td></tr>
<tr><td>MI300X 192GB</td><td>$3.50–5.00</td><td>$1.50–2.50</td><td>192 GB HBM3</td><td>性价比替代方案</td></tr></tbody>
</table>
</div>

## 11.12.2　RLHF 训练成本估算 {#rlhf-training-cost-estimation}



<div class="arithmatex">\[\text{成本} = \frac{N_\text{steps} \times T_\text{step}}{3600} \times N_\text{GPUs} \times C_\text{GPU/hr}
\tag{11.9}\]</div>

<div class="admonition example"><p class="admonition-title">成本示例：70B 模型 RLHF（10K 步）</p>
<span id="table-11-inline-15"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col">步数</th><th scope="col">10,000</th></tr></thead>
<tbody><tr><td>每步用时（解耦）</td><td>45 秒</td></tr>
<tr><td>总训练时间</td><td><span class="arithmatex">\(10000 \times 45 / 3600 = 125\)</span> 小时</td></tr>
<tr><td>GPU 数（生成 + 训练）</td><td>64 块 A100-80GB</td></tr>
<tr><td>每 GPU-小时成本（spot）</td><td>$1.20</td></tr>
<tr><td><strong>总成本</strong></td><td><span class="arithmatex">\(125 \times 64 \times \$1.20 =\)</span> <strong>$9,600</strong></td></tr></tbody>
</table>
</div>

<p><strong>按阶段拆分</strong>：</p>

<ul>
<li><p>生成集群（32 GPU）：$4,800（占 60% 时间）</p></li>
<li><p>训练集群（32 GPU）：$4,800（可重叠 <span class="arithmatex">\(\rightarrow\)</span> 实际 $3,400）</p></li>
<li><p>打分（与生成共享 GPU）：已含在上文中</p></li>
</ul>

<p><strong>重叠后</strong>：完整对齐 70B 模型的实际成本约为 <strong>$7,500</strong>。</p>
</div>

## 11.12.3　成本优化策略 {#cost-optimization-strategies}

<ul>
<li><p><strong>Spot/可抢占实例</strong>：节省 50–70%。要求 checkpoint 机制健壮（每 5 分钟保存一次）。</p></li>
<li><p><strong>合理配型</strong>：不要用 H100 做生成（显存带宽瓶颈）；推理时 A100 的 tokens/$ 相近。</p></li>
<li><p><strong>量化推理</strong>：生成与打分用 INT8/FP8 可让相应集群的 GPU 数减半。</p></li>
<li><p><strong>渐进式训练</strong>：先用 8B 代理模型做 reward 工程/调试（约 $200），再扩到 70B。</p></li>
<li><p><strong>用 LoRA 取消 reference</strong>：彻底移除 reference 模型（显存减少 50%）。</p></li>
<li><p><strong>先短后长</strong>：按 256<span class="arithmatex">\(\rightarrow\)</span>512<span class="arithmatex">\(\rightarrow\)</span>1024 token 的 curriculum 生成可节省 40% 算力。</p></li>
</ul>
