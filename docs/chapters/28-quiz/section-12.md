# 28.12　FlashAttention 问题 {#flash-attention-questions}

<section class="quiz-question" id="q-source-56"><h2 class="question-title">Q：FlashAttention 计算的结果与标准 attention 相同，但快 2–4 倍。如果 FLOPs 数相同，这怎么可能？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：FlashAttention 之所以更快，是因为它减少了<em>HBM 显存流量</em>，而非 FLOPs。标准 attention 在 HBM（慢）中物化 <span class="arithmatex">\(n \times n\)</span> 的 attention 矩阵，再读回来做 Softmax，再读一次做 <span class="arithmatex">\(PV\)</span> 乘法——总共在 <span class="arithmatex">\(O(n^2)\)</span> 数据上有 4 次 HBM 往返。</p>

<p>FlashAttention 对计算进行分块，使 <span class="arithmatex">\(n \times n\)</span> 矩阵完全在 SRAM（快，19 TB/s）中计算并消费，从不写入 HBM（2 TB/s）。“在线 Softmax”技巧通过维护滚动统计量来实现这一点。</p>

<p>结果：HBM 流量从 <span class="arithmatex">\(O(n^2 d)\)</span> 降到 <span class="arithmatex">\(O(n^2 d / M)\)</span>，其中 <span class="arithmatex">\(M\)</span> 为 SRAM 大小。相同 FLOPs，显存流量减少 10–50<span class="arithmatex">\(\times\)</span> <span class="arithmatex">\(\to\)</span> 实际墙钟加速 2–4<span class="arithmatex">\(\times\)</span>。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章与第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 架构；系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-57"><h2 class="question-title">Q：为什么 FlashAttention 对 FFN 层没帮助？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：FFN 层是<em>计算瓶颈</em>，而非显存瓶颈。其算术强度（大 Batch GEMM 下 <span class="arithmatex">\(I \approx 300\)</span> FLOP/byte）已经高于 roofline 拐点（A100 上为 156 FLOP/byte）。</p>

<p>FlashAttention 对 attention 有帮助是因为 attention 严重受限于显存（<span class="arithmatex">\(I \approx 1\)</span>–<span class="arithmatex">\(60\)</span> FLOP/byte）。通过将数据保留在 SRAM 中，它消除了显存瓶颈。</p>

<p>对 FFN 而言：瓶颈已经是 Tensor Core（而非显存带宽），所以减少显存流量没用。FFN 的收益来自量化（减小权重大小 <span class="arithmatex">\(\to\)</span> 提高算术强度）和更大的 Batch。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章与第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 架构；系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-58"><h2 class="question-title">Q：解释在线 Softmax 技巧，以及它对 FlashAttention 为何不可或缺。</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：标准 Softmax 在计算任何输出之前都需要全局最大值 <span class="arithmatex">\(m = \max_j x_j\)</span>——这要求先看到所有 <span class="arithmatex">\(n\)</span> 个 attention 分数，迫使物化完整的 <span class="arithmatex">\(n \times n\)</span> 矩阵。</p>

<p>在线 Softmax 技巧按顺序处理块，维护一个滚动的 <span class="arithmatex">\((m, \ell, O)\)</span> 状态：</p>

<ol>
<li><p>处理新块 <span class="arithmatex">\(\to\)</span> 更新滚动最大值：<span class="arithmatex">\(m_{\text{new}} = \max(m_{\text{old}}, \max(s_{\text{new}}))\)</span></p></li>
<li><p>重新缩放旧的求和：<span class="arithmatex">\(\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} + \text{new terms}\)</span></p></li>
<li><p>重新缩放输出：<span class="arithmatex">\(O_{\text{new}} = \text{rescaled}(O_{\text{old}}) + \text{new contribution}\)</span></p></li>
</ol>

<p>这在数学上是完全精确的——没有任何近似。它使得逐块处理成为可能，每块都能放进 SRAM，永远不需要在显存中保留完整的 <span class="arithmatex">\(n \times n\)</span> 矩阵。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章与第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 架构；系统基础）。</em></p>
</details></section>
