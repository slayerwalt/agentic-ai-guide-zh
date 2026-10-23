# 2.1.7　算术强度与 Roofline 模型 {#arithmetic-intensity-and-the-roofline-model}

<div class="admonition note"><p class="admonition-title">算术强度（Arithmetic Intensity）</p>


<div class="arithmatex">\[I = \frac{\text{FLOPs}}{\text{从 HBM 访问的字节数}}
  \quad \text{(FLOPs / Byte)}\]</div>

<p>当 <span class="arithmatex">\(I &lt; I_{\text{ridge}}\)</span> 时内核是<strong>内存密集（Memory-Bound）</strong>的，当 <span class="arithmatex">\(I &gt; I_{\text{ridge}}\)</span> 时则是<strong>计算密集（Compute-Bound）</strong>的，其中</p>



<div class="arithmatex">\[I_{\text{ridge}} = \frac{\text{峰值 FLOP/s}}{\text{峰值带宽}}
  = \frac{312 \times 10^{12}}{2 \times 10^{12}} = 156 \text{ FLOP/Byte (A100 BF16)}\]</div>
</div>

<figure id="figure-2-2"><a href="../../assets/figures/fig_017_fig17.png" target="_blank" rel="noopener" aria-label="查看图 2.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_017_fig17.png" alt="A100 BF16 的 Roofline 模型（Roofline Model）。Attention 深陷于内存密集区间；大型 GEMM（FFN 层）则是计算密集的。" loading="lazy"></a><figcaption><span class="figure-number">图 2.2　</span>A100 BF16 的 Roofline 模型（Roofline Model）。Attention 深陷于内存密集区间；大型 GEMM（FFN 层）则是计算密集的。</figcaption></figure>

<div class="admonition example"><p class="admonition-title">Attention 的算术强度</p>
<p>对于序列长度 <span class="arithmatex">\(n=4096\)</span>、头维度 <span class="arithmatex">\(d=128\)</span> 的单个注意力头：</p>

<p><strong>FLOPs</strong>：<span class="arithmatex">\(QK^T\)</span> 耗费 <span class="arithmatex">\(2n^2d\)</span>，softmax 为 <span class="arithmatex">\(O(n^2)\)</span>，<span class="arithmatex">\(\text{Attn} \times V\)</span> 耗费 <span class="arithmatex">\(2n^2d\)</span>。总计：<span class="arithmatex">\(\approx 4n^2 d = 4 \times 4096^2 \times 128 \approx 8.6\)</span> GFLOP。</p>

<p><strong>内存流量</strong>（标准实现，非 Flash 版）：</p>

<ul>
<li><p>读取 <span class="arithmatex">\(Q, K\)</span>：<span class="arithmatex">\(2 \times n \times d \times 2 = 2\)</span> MB</p></li>
<li><p>写入注意力分数 <span class="arithmatex">\(S = QK^T\)</span>：<span class="arithmatex">\(n^2 \times 2 = 33.5\)</span> MB</p></li>
<li><p>为 softmax 读取 <span class="arithmatex">\(S\)</span>：<span class="arithmatex">\(n^2 \times 2 = 33.5\)</span> MB</p></li>
<li><p>写入 softmax 输出 <span class="arithmatex">\(P\)</span>：<span class="arithmatex">\(n^2 \times 2 = 33.5\)</span> MB</p></li>
<li><p>为最终矩阵乘读取 <span class="arithmatex">\(P\)</span> 和 <span class="arithmatex">\(V\)</span>：<span class="arithmatex">\(n^2 \times 2 + n \times d \times 2 = 34.5\)</span> MB</p></li>
<li><p>写入输出 <span class="arithmatex">\(O\)</span>：<span class="arithmatex">\(n \times d \times 2 = 1\)</span> MB</p></li>
</ul>

<p><strong>总内存</strong>：<span class="arithmatex">\(\approx 138\)</span> MB（由对 <span class="arithmatex">\(n^2\)</span> 注意力矩阵的 4 次遍历主导）。</p>

<p><strong>算术强度</strong>：</p>



<div class="arithmatex">\[I = \frac{8.6 \times 10^9}{138 \times 10^6} \approx 62 \text{ FLOP/Byte}\]</div>

<p>这是 A100 脊点的 <span class="arithmatex">\(62/156 = 40\%\)</span>——<strong>明确属于内存密集</strong>。GPU 有 60% 的时间在等内存空转。</p>

<p><strong>FlashAttention 解决方案</strong>：通过永远不物化 <span class="arithmatex">\(n \times n\)</span> 矩阵（在 SRAM 中对 <span class="arithmatex">\(Q, K, V\)</span> 分块），FlashAttention 把 HBM 流量降至仅读取 <span class="arithmatex">\(Q, K, V\)</span> 并写入 <span class="arithmatex">\(O\)</span>：<span class="arithmatex">\(4 \times n \times d \times 2 = 4\)</span> MB。每个加载的字节被复用于 <span class="arithmatex">\(O(n)\)</span> 次计算（每个 query 都要 attend 所有 key），因此：</p>



<div class="arithmatex">\[I = \frac{4n^2 d}{4 \cdot n \cdot d \cdot 2} = \frac{n}{2} = \frac{4096}{2} = 2048 \text{ FLOP/Byte}\]</div>

<p>这比脊点（156）高出 <span class="arithmatex">\(13\times\)</span>——<strong>深度计算密集</strong>。GPU 跑满峰值 312 TFLOPS，只需要 <span class="arithmatex">\(312\text{T}/2048 \approx 152\)</span> GB/s 带宽（HBM 容量的 7.6%）。内存不再是瓶颈。</p>
</div>
