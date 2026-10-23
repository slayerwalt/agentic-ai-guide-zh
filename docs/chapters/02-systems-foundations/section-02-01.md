# 2.2.1　KV cache 碎片化问题 {#the-kv-cache-fragmentation-problem}

<div class="admonition note"><p class="admonition-title">KV cache 显存计算公式</p>
<p>对于具有 <span class="arithmatex">\(L\)</span> 层、<span class="arithmatex">\(H\)</span> 个头、头维度 <span class="arithmatex">\(d\)</span>、序列长度为 <span class="arithmatex">\(n\)</span> 个 token 的模型：</p>



<div class="arithmatex">\[\text{KV cache 大小} = 2 \times L \times H \times d \times n \times \text{每元素字节数}\]</div>

<p>对于 Llama-3 70B（BF16）：<span class="arithmatex">\(L=80\)</span>、<span class="arithmatex">\(H=8\)</span>（GQA）、<span class="arithmatex">\(d=128\)</span>：</p>



<div class="arithmatex">\[= 2 \times 80 \times 8 \times 128 \times n \times 2 = 327{,}680 \times n \text{ bytes}\]</div>

<p>当 <span class="arithmatex">\(n=4096\)</span> token 时：每条序列约 1.3 GB。</p>
</div>

<div class="admonition tip"><p class="admonition-title">内部碎片与外部碎片</p>
<p>传统推理系统会为每条序列的 KV cache 预先分配一块连续内存，大小为<em>可能的最大序列长度</em>。这会造成两类浪费：</p>

<p><strong>内部碎片：</strong>一条只生成 500 token 的序列仍然占据为 4096 token 预留的块。未使用的 3596 个 token 槽位被浪费。</p>

<p><strong>外部碎片：</strong>许多序列完成后，剩余空闲内存由许多小的不连续空隙组成。即使总空闲内存足够，新的长序列也无法被分配，因为没有任何一块连续区域足够大。</p>

<p>实际中，朴素分配下的 GPU 显存利用率通常只有 20–40%。</p>
</div>
