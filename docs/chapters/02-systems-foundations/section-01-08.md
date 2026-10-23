# 2.1.8　Attention 是内存密集的；FFN 是计算密集的 {#attention-is-memory-bound-ffn-is-compute-bound}

<div class="admonition tip"><p class="admonition-title">Transformer 中的两种区间</p>
<p>一个 transformer 块有两个主要组件，它们的算术强度截然不同：</p>

<ul>
<li><p><strong>Attention：</strong>作用于 <span class="arithmatex">\(n \times d\)</span> 张量。<span class="arithmatex">\(QK^T\)</span> 乘积是 <span class="arithmatex">\(O(n^2 d)\)</span> FLOPs，但注意力分数需要 <span class="arithmatex">\(O(n^2)\)</span> 内存。在长序列下，内存流量占主导地位——attention 是<em>内存密集（Memory-Bound）</em>的。</p></li>
<li><p><strong>FFN（MLP）：</strong>两个大型线性层，权重矩阵形状为 <span class="arithmatex">\([d_{\text{model}}, 4d_{\text{model}}]\)</span>。这些是高算术强度的大型 GEMM——FFN 是<em>计算密集（Compute-Bound）</em>的。</p></li>
</ul>

<p>这就是为什么 FlashAttention（内存优化）对 attention 有效但对 FFN 无效，而量化（缩小权重体积）对 FFN 的帮助比对 attention 更大。</p>
</div>
