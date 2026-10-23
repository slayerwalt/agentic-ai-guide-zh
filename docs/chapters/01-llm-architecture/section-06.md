# 1.6　FlashAttention——算法与硬件感知 {#flash-attention-algorithm-and-hardware-awareness}

<p>FlashAttention <a class="book-citation" id="cite-dao2022flashattention-1" href="../../references.md#ref-dao2022flashattention" aria-label="参考文献 65">[65]</a><a class="book-citation" id="cite-dao2023flashattention2-1" href="../../references.md#ref-dao2023flashattention2" aria-label="参考文献 64">[64]</a>是自 Transformer 本身以来深度学习领域最具影响力的算法创新之一。它不改变 Attention 的数学结果——计算出的输出与原始 Attention <em>完全相同</em>——但它重构了内存访问模式，让 GPU 上容量有限的高速 SRAM 承担所有重活，从而将高带宽内存（HBM）占用从 <span class="arithmatex">\(O(n^2)\)</span> 降至 <span class="arithmatex">\(O(n)\)</span>，并在典型工作负载上带来 2–4<span class="arithmatex">\(\times\)</span> 的端到端实际墙钟时间加速。</p>

## 1.6.1　标准 Attention 的内存问题 {#the-standard-attention-memory-problem}

<p>标准的缩放点积 Attention 为：</p>



<div class="arithmatex">\[\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V\]</div>

<div class="admonition note"><p class="admonition-title">标准 Attention 的内存复杂度</p>
<p>对于序列长度 <span class="arithmatex">\(n\)</span> 和头维度 <span class="arithmatex">\(d\)</span>：</p>

<ul>
<li><p><span class="arithmatex">\(Q, K, V \in \mathbb{R}^{n \times d}\)</span>：<span class="arithmatex">\(O(nd)\)</span> 内存</p></li>
<li><p><span class="arithmatex">\(S = QK^T \in \mathbb{R}^{n \times n}\)</span>：<strong><span class="arithmatex">\(O(n^2)\)</span> 内存</strong>——瓶颈所在</p></li>
<li><p><span class="arithmatex">\(P = \text{softmax}(S) \in \mathbb{R}^{n \times n}\)</span>：又一个 <span class="arithmatex">\(O(n^2)\)</span></p></li>
<li><p><span class="arithmatex">\(O = PV \in \mathbb{R}^{n \times d}\)</span>：<span class="arithmatex">\(O(nd)\)</span></p></li>
</ul>

<p>在 <span class="arithmatex">\(n=8192\)</span>、<span class="arithmatex">\(d=128\)</span>、BF16 条件下：仅 Attention 矩阵本身就达到 <span class="arithmatex">\(8192^2 \times 2 \approx 134\)</span> MB <em>每头</em>。若有 32 个头，单单一层的 Attention 分数就要占用 4.3 GB。</p>
</div>

<div class="admonition tip"><p class="admonition-title">为什么 <span class="arithmatex">\(O(n^2)\)</span> 是灾难性的</p>
<p>Attention 矩阵必须写入 HBM（对长序列而言它放不进 SRAM），随后再读出来做 Softmax，然后再读一次用于 <span class="arithmatex">\(PV\)</span> 乘法。每一次 HBM 往返都很慢。当 <span class="arithmatex">\(n=32768\)</span>（32K 上下文）时，Attention 矩阵达 <span class="arithmatex">\(32768^2 \times 2 \approx 2\)</span> GB <em>每头</em>——完全无法存储。</p>
</div>

## 1.6.2　FlashAttention 的核心洞察——分块（Tiling）与在线 Softmax（Online Softmax） {#the-flash-attention-key-insight-tiling-and-online-softmax}

<p>核心洞察是：<strong>我们从不需要把完整的 <span class="arithmatex">\(n \times n\)</span> 矩阵一次性放入内存</strong>。如果使用<em>在线 Softmax</em>技巧，就可以按块逐步计算输出 <span class="arithmatex">\(O\)</span>。</p>

#### 在线 Softmax（Online Softmax）。 {#online-softmax.}

<p>回想一下，为了数值稳定性，Softmax 需要一个全局最大值：</p>



<div class="arithmatex">\[\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j\]</div>

<p>诀窍在于：我们可以在处理新块时<em>更新</em>运行中的最大值和归一化因子，而无需在任何时刻物化整行。</p>

<div class="admonition note"><p class="admonition-title">在线 Softmax 更新规则</p>
<p>给定运行状态 <span class="arithmatex">\((m_{\text{old}}, \ell_{\text{old}}, O_{\text{old}})\)</span> 和一个新的分数块 <span class="arithmatex">\(s_{\text{new}}\)</span>：</p>

<ol>
<li><p><span class="arithmatex">\(m_{\text{new}} = \max(m_{\text{old}},\; \max(s_{\text{new}}))\)</span></p></li>
<li><p><span class="arithmatex">\(\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}}
        + \sum_j e^{s_{\text{new},j} - m_{\text{new}}}\)</span></p></li>
<li><p><span class="arithmatex">\(O_{\text{new}} = \frac{1}{\ell_{\text{new}}} \left(
        e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} \cdot O_{\text{old}}
        + e^{s_{\text{new}} - m_{\text{new}}} \cdot V_{\text{new}} \right)\)</span></p></li>
</ol>

<p>这在数学上等价于一次性对所有块计算 Softmax。</p>
</div>

## 1.6.3　FlashAttention 算法 {#the-flash-attention-algorithm}

<div class="admonition example"><p class="admonition-title">FlashAttention 前向传播——分块（Block Tiling）</p>
<p><strong>设定：</strong> SRAM 大小 <span class="arithmatex">\(M\)</span>。块大小 <span class="arithmatex">\(B_r = \lceil M / (4d) \rceil\)</span>，<span class="arithmatex">\(B_c = \min(\lceil M / (4d) \rceil, d)\)</span>。</p>

<ol>
<li><p>将 <span class="arithmatex">\(Q\)</span> 划分为 <span class="arithmatex">\(T_r = \lceil n / B_r \rceil\)</span> 块 <span class="arithmatex">\(Q_1, \ldots, Q_{T_r}\)</span></p></li>
<li><p>将 <span class="arithmatex">\(K, V\)</span> 划分为 <span class="arithmatex">\(T_c = \lceil n / B_c \rceil\)</span> 块 <span class="arithmatex">\(K_1, \ldots, K_{T_c}\)</span></p></li>
<li><p>初始化输出 <span class="arithmatex">\(O \in \mathbb{R}^{n \times d}\)</span>、运行最大值 <span class="arithmatex">\(m \in \mathbb{R}^n\)</span>、运行求和 <span class="arithmatex">\(\ell \in \mathbb{R}^n\)</span>（均在 HBM 中）</p></li>
<li><p>对 <span class="arithmatex">\(j = 1, \ldots, T_c\)</span> 进行 <strong>外循环</strong>：</p>

<ol>
<li><p>将 <span class="arithmatex">\(K_j, V_j\)</span> 从 HBM 加载到 SRAM</p></li>
<li><p>对 <span class="arithmatex">\(i = 1, \ldots, T_r\)</span> 进行 <strong>内循环</strong>：</p>

<ol>
<li><p>将 <span class="arithmatex">\(Q_i, O_i, m_i, \ell_i\)</span> 从 HBM 加载到 SRAM</p></li>
<li><p>计算 <span class="arithmatex">\(S_{ij} = Q_i K_j^T / \sqrt{d}\)</span>（保留在 SRAM 中）</p></li>
<li><p>应用在线 Softmax 更新得到新的 <span class="arithmatex">\(m_i, \ell_i, O_i\)</span></p></li>
<li><p>将 <span class="arithmatex">\(O_i, m_i, \ell_i\)</span> 写回 HBM</p></li>
</ol></li>
</ol></li>
<li><p>返回 <span class="arithmatex">\(O\)</span></p></li>
</ol>

<p><strong>关键：</strong> <span class="arithmatex">\(S_{ij}\)</span>（Attention 块）在 SRAM 中计算并丢弃。它<em>从不写入 HBM</em>。</p>
</div>

<div class="admonition note"><p class="admonition-title">FlashAttention 复杂度</p>
<span id="table-1-inline-14"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"></th><th scope="col"><strong>标准 Attention</strong></th><th scope="col"><strong>FlashAttention</strong></th></tr></thead>
<tbody><tr><td>内存（HBM）</td><td><span class="arithmatex">\(O(n^2)\)</span></td><td><span class="arithmatex">\(O(n)\)</span></td></tr>
<tr><td>HBM 读写量</td><td><span class="arithmatex">\(O(n^2 d)\)</span></td><td><span class="arithmatex">\(O(n^2 d / M)\)</span></td></tr>
<tr><td>FLOPs</td><td><span class="arithmatex">\(O(n^2 d)\)</span></td><td><span class="arithmatex">\(O(n^2 d)\)</span>（相同）</td></tr>
<tr><td>加速比</td><td>1<span class="arithmatex">\(\times\)</span></td><td>2–4<span class="arithmatex">\(\times\)</span></td></tr></tbody>
</table>
</div>

<p>在前向传播中，总 FLOPs 仍为 <span class="arithmatex">\(O(n^2 d)\)</span>——与标准 Attention 完全相同。FlashAttention 的加速完全来自于削减缓慢的 HBM 流量，而不是减少运算。（反向传播由于重计算实际上执行了<em>更多</em> FLOPs，但墙钟时间仍然更短，因为节省下来的内存带宽占主导。）</p>
</div>

## 1.6.4　FlashAttention 2——更好的并行性 {#flash-attention-2-better-parallelism}

<p>FlashAttention 2 <a class="book-citation" id="cite-dao2023flashattention2-2" href="../../references.md#ref-dao2023flashattention2" aria-label="参考文献 64">[64]</a> 做出了三项关键改进：</p>

<ol>
<li><p><strong>减少非矩阵乘法 FLOPs：</strong> 原版 FA 在内循环中存在不必要的重缩放操作。FA2 重构循环以最小化这些操作。在 A100 上，Tensor Core 矩阵乘法比标量运算快约 16<span class="arithmatex">\(\times\)</span>，因此即便内循环中只有少量非矩阵乘法工作，也会成为延迟瓶颈。</p></li>
<li><p><strong>序列维度上更好的并行性：</strong> FA1 仅在 batch 和 head 上并行。FA2 还在 query 序列维度上并行，使长序列、小批量场景下的 GPU 利用率显著提升。</p></li>
<li><p><strong>因果掩码优化：</strong> 对于自回归（因果）Attention，大约一半的块是完全被掩码的。FA2 完全跳过这些块，相对于双向 Attention 带来约 2<span class="arithmatex">\(\times\)</span> 的加速。</p></li>
</ol>

## 1.6.5　FlashAttention 3——Hopper 架构 {#flash-attention-3-hopper-architecture}

<p>FlashAttention 3 <a class="book-citation" id="cite-shah2024flashattention3-1" href="../../references.md#ref-shah2024flashattention3" aria-label="参考文献 322">[322]</a> 专为 H100 设计，并利用了三项 Hopper 特有的特性：</p>

<ul>
<li><p><strong>TMA（Tensor Memory Accelerator，张量内存加速器）：</strong> H100 有一个专用硬件单元用于在 HBM 和 SRAM 之间进行异步批量数据搬移。FA3 利用 TMA 让数据加载与计算重叠，从而隐藏内存延迟。</p></li>
<li><p><strong>Warp 专用化（Warp-specialization）：</strong> FA3 给不同的 warp 分配不同角色（生产者 warp 通过 TMA 加载数据；消费者 warp 计算 MMA）。这是一种软件流水线技术，使内存系统和 Tensor Core 同时保持繁忙。</p></li>
<li><p><strong>FP8 支持：</strong> H100 以 BF16 两倍的吞吐量支持 FP8（E4M3/E5M2）Tensor Core 运算。FA3 支持 FP8 Attention，并通过逐块量化来保持精度。</p></li>
</ul>

<p>FA3 在 FP16 Attention 上可达 <strong>H100 理论峰值的 75%</strong>，相比之下 FA2 约为 35%。</p>

## 1.6.6　FlashAttention 4——Blackwell 架构 {#flash-attention-4-blackwell-architecture}

<p>FlashAttention 4 <a class="book-citation" id="cite-zadouri2026flashattention4-1" href="../../references.md#ref-zadouri2026flashattention4" aria-label="参考文献 415">[415]</a> 面向 NVIDIA 的 Blackwell GPU（B200/GB200），这类 GPU 将 Tensor Core 吞吐量提高到 2.25 PFLOP/s（BF16），但非矩阵乘法单元（指数运算、共享内存带宽）的扩展速度较慢。这种<em>非对称的硬件扩展</em>意味着瓶颈发生了转移：在 Blackwell 上，Attention 的限制因素不再是矩阵乘法，而是 Softmax 指数运算以及围绕它们的共享内存流量。</p>

<p>FA4 通过四项关键技术来应对这一问题：</p>

<ul>
<li><p><strong>完全异步的 MMA 流水线：</strong> Blackwell 的 MMA 指令是完全异步的（不像 Hopper 的 wgmma 仍会阻塞等待完成）。FA4 重新设计了流水线，在更大的分块尺寸上让 MMA、TMA 加载和 Softmax 重缩放相互重叠，使所有硬件单元保持饱和。</p></li>
<li><p><strong>软件模拟指数运算：</strong> FA4 不调用硬件的 <code>ex2</code> 单元（吞吐量瓶颈），而是用多项式近似在快得多的 Tensor Core 上模拟 <span class="arithmatex">\(e^x\)</span>。这相当于用额外的矩阵乘法指令换取消除指数单元的停顿。</p></li>
<li><p><strong>条件式 Softmax 重缩放：</strong> 标准 FlashAttention 每个分块都会重缩放运行 <span class="arithmatex">\(\max\)</span>。FA4 在新分块的最大值不超过当前运行最大值时（实际中很常见）跳过重缩放，从而节省寄存器搬移和同步屏障。</p></li>
<li><p><strong>Tensor Memory + 2-CTA MMA 模式（反向传播）：</strong> 反向传播使用 Blackwell 的<em>张量内存（Tensor Memory）</em>（一种比共享内存更大的、每 SM 私有的暂存区）以及 2-CTA 协作模式，在两个线程块簇之间融合 <span class="arithmatex">\(dQ\)</span> 累加，使共享内存往返次数减半。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">FA4 实现：CuTe-DSL</p>
<p>FA4 是首个用 <strong>CuTe-DSL</strong> 编写的 FlashAttention 版本，CuTe-DSL 是一种嵌入在 Python 中、面向 GPU 内核的领域特定语言（CUTLASS 4.x 的一部分）。CuTe-DSL 的编译速度比 C++ CUTLASS 模板快 20–30<span class="arithmatex">\(\times\)</span>，同时保留对寄存器分配和流水线调度的完全控制。这显著缩短了内核开发的迭代时间。</p>
</div>

#### 结果。 {#results.}

<p>在 B200 上，BF16、head 维度 128（因果，序列长度 8K）：</p>

<ul>
<li><p><strong>1613 TFLOP/s</strong>——达到 Blackwell 峰值利用率的 71%</p></li>
<li><p>比 cuDNN 9.13（NVIDIA 的专有融合内核）快 <strong>1.3<span class="arithmatex">\(\times\)</span></strong></p></li>
<li><p>在同一硬件上比 Triton 快 <strong>2.7<span class="arithmatex">\(\times\)</span></strong></p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">软硬件协同演化</p>
<p>FlashAttention 系列体现了一条关键原则：每一代 GPU 都会让瓶颈发生转移，因此需要新的算法思想而不仅是重新编译。A80 <span class="arithmatex">\(\to\)</span> 内存带宽受限（FA1/FA2：分块 + 重计算）。H100 <span class="arithmatex">\(\to\)</span> 数据搬移受限（FA3：TMA + warp 专用化）。B200 <span class="arithmatex">\(\to\)</span> 非矩阵乘法计算受限（FA4：软件模拟 exp + 条件式重缩放）。理解<em>硬件瓶颈究竟在哪</em>是编写高效内核的前提。</p>
</div>
