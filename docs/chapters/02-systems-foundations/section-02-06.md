# 2.2.6　具体显存节约——大规模 70B 模型 {#concrete-memory-savings-70b-model-at-scale}

<div class="admonition example"><p class="admonition-title">显存预算——70B BF16 推理</p>
<p><strong>设定：</strong>Llama-3 70B，BF16，单个 A100 80GB 节点（8 GPU，张量并行）。</p>

<p><strong>模型权重：</strong><span class="arithmatex">\(70 \times 10^9 \times 2\)</span> 字节 <span class="arithmatex">\(= 140\)</span> GB <span class="arithmatex">\(\div\)</span> 8 GPU <span class="arithmatex">\(= 17.5\)</span> GB/GPU。</p>

<p><strong>KV cache 可用部分：</strong><span class="arithmatex">\(80 - 17.5 - 3\)</span>（开销）<span class="arithmatex">\(= 59.5\)</span> GB/GPU。</p>

<p><strong>每块 GPU 每 token 的 KV cache</strong>（TP=8 时每块 GPU 持有 <span class="arithmatex">\(1/8\)</span> 的头）：<span class="arithmatex">\(2 \times 80 \times 1 \times 128 \times 2 = 40{,}960\)</span> 字节 <span class="arithmatex">\(\approx 40\)</span> KB/token。</p>

<p><strong>KV cache 最大 token 数：</strong><span class="arithmatex">\(59.5 \times 10^9 / 40{,}960 \approx 145\)</span> 万 token。</p>

<p><strong>当 128 条并发序列每条 4096 个 token 时：</strong><span class="arithmatex">\(128 \times 4096 = 524{,}288\)</span> 个 token——完全在预算之内。</p>

<p><strong>若不使用 PagedAttention</strong>（为每条序列预分配最大长度 4096）：同样的计算，但碎片化平均浪费约 50% <span class="arithmatex">\(\to\)</span> 只能容纳 64 条序列。</p>
</div>

<div class="admonition warning"><p class="admonition-title">块大小的权衡</p>
<p>更大的块大小可减少块表开销，并提高内存访问局部性（更少的分散读取）。更小的块大小可减少内部碎片，并支持更细粒度的前缀共享。vLLM 默认每块 16 个 token，是一个不错的平衡。对于非常长的序列（10 万+ token），更大的块（32–64）可能更合适。</p>
</div>
