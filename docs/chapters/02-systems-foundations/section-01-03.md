# 2.1.3　LLM 训练与推理常用 GPU {#common-gpus-for-llm-training-and-inference}

<p class="table-caption" id="table-2-2"><span class="table-number">表 2.2　</span>与 LLM 工作负载相关的 GPU 规格。所有带宽数字均为双向。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>GPU</strong></th><th scope="col"><strong>架构</strong></th><th scope="col"><strong>HBM</strong></th><th scope="col"><strong>BF16 TF</strong></th><th scope="col"><strong>HBM 带宽</strong></th><th scope="col"><strong>NVLink</strong></th><th scope="col"><strong>LLM 角色</strong></th></tr></thead>
<tbody><tr><td>V100-32GB</td><td>Volta</td><td>32 GB</td><td>125 TF*</td><td>900 GB/s</td><td>300 GB/s</td><td>旧型号；小模型微调</td></tr>
<tr><td>A100-40GB</td><td>Ampere</td><td>40 GB</td><td>312 TF</td><td>1.5 TB/s</td><td>600 GB/s</td><td>经济型训练/推理</td></tr>
<tr><td>A100-80GB</td><td>Ampere</td><td>80 GB</td><td>312 TF</td><td>2.0 TB/s</td><td>600 GB/s</td><td>标准 RLHF（70B 模型用 8–64 卡）</td></tr>
<tr><td>H100 SXM</td><td>Hopper</td><td>80 GB</td><td>990 TF</td><td>3.35 TB/s</td><td>900 GB/s</td><td>训练快 3<span class="arithmatex">\(\times\)</span></td></tr>
<tr><td>H200 SXM</td><td>Hopper</td><td>141 GB</td><td>990 TF</td><td>4.8 TB/s</td><td>900 GB/s</td><td>用更少 GPU 装下 70B 策略+参考模型</td></tr>
<tr><td>B200 SXM</td><td>Blackwell</td><td>192 GB</td><td>2250 TF</td><td>8.0 TB/s</td><td>1800 GB/s</td><td>新一代；比 H100 快 2<span class="arithmatex">\(\times\)</span></td></tr>
<tr><td colspan="7"><em>AMD 和 Google 替代方案：</em></td></tr>
<tr><td>MI300X</td><td>CDNA3</td><td>192 GB</td><td>1300 TF</td><td>5.3 TB/s</td><td>N/A</td><td>显存最大；ROCm</td></tr>
<tr><td>TPU v5e</td><td>Google</td><td>16 GB</td><td>197 TF</td><td>1.6 TB/s</td><td>ICI 1.6 TB/s</td><td>仅云端；JAX/XLA</td></tr></tbody>
</table>
</div>

<div class="admonition warning"><p class="admonition-title">如何选择 GPU？</p>
<ul>
<li><p><strong>训练 70B+ 模型</strong>：带 NVLink 的 H100/B200 节点（张量并行需要快速互联）。每个实例至少 8<span class="arithmatex">\(\times\)</span>H100。</p></li>
<li><p><strong>推理（延迟敏感）</strong>：H100/H200 适合高带宽场景；MI300X 适合内存密集（Memory-Bound）场景（巨大的键值缓存（KV Cache））。</p></li>
<li><p><strong>微调 7B–13B 模型</strong>：A100-80GB 性价比高。用 LoRA 单卡即可。</p></li>
<li><p><strong>预算紧张</strong>：A100-40GB，甚至用 A10（24GB）在 7B 模型上跑 LoRA。</p></li>
</ul>
</div>
