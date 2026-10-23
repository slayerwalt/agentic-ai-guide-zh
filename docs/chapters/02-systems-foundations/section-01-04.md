# 2.1.4　GPU 内部架构——流式多处理器（SM） {#gpu-internal-architecture-the-streaming-multiprocessor-sm}

<p>GPU 由一组<strong>流式多处理器（Streaming Multiprocessor，SM）</strong>阵列组成，每个 SM 都是独立的处理器，拥有自己的寄存器文件、共享内存和执行单元。理解 SM 是理解 GPU 性能的关键。</p>

<figure id="figure-2-1"><a href="../../assets/figures/fig_016_fig16.png" target="_blank" rel="noopener" aria-label="查看图 2.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_016_fig16.png" alt="左：A100 单个流式多处理器（SM）的内部结构——64 个 FP32 CUDA cores、4 个 Tensor Cores、4 个 warp 调度器、256 KB 寄存器文件以及 192 KB 共享内存/L1 缓存。右：完整 A100 芯片包含 108 个 SM，共享 40 MB L2 缓存和 80 GB HBM2e。左侧边栏的带宽标注显示了从寄存器到 HBM 的急剧下降。" loading="lazy"></a><figcaption><span class="figure-number">图 2.1　</span>左：A100 单个流式多处理器（SM）的内部结构——64 个 FP32 CUDA cores、4 个 Tensor Cores、4 个 warp 调度器、256 KB 寄存器文件以及 192 KB 共享内存/L1 缓存。右：完整 A100 芯片包含 108 个 SM，共享 40 MB L2 缓存和 80 GB HBM2e。左侧边栏的带宽标注显示了从寄存器到 HBM 的急剧下降。</figcaption></figure>

<div class="admonition note"><p class="admonition-title">SM 的关键组件</p>
<ul>
<li><p><strong>CUDA Cores：</strong>用于 FP32/INT32 运算的标量 ALU。A100 上每个 SM 有 64 个。用于逐元素运算、归约和非矩阵运算。</p></li>
<li><p><strong>Tensor Cores：</strong>专用的矩阵乘加（matrix-multiply-accumulate，MMA）单元。每个单元每周期执行一次 <span class="arithmatex">\(4{\times}4{\times}4\)</span> 融合乘加运算。A100 每个 SM 有 4 个，在支持的精度下提供相对 CUDA cores <span class="arithmatex">\(16\times\)</span> 的吞吐量。</p></li>
<li><p><strong>寄存器文件：</strong>最快的存储（延迟 1 周期）。所有活动线程共享。溢出到 L1 会显著降速。</p></li>
<li><p><strong>共享内存 / L1：</strong>由程序员显式管理的片上 SRAM。这是 FlashAttention 性能的关键（数据块完全装入共享内存）。</p></li>
<li><p><strong>Warp 调度器：</strong>每个 SM 有 4 个 warp 调度器（A100）。一个 <em>warp</em> = 32 个步调一致执行的线程（SIMT 模型）。调度器通过在 warp 之间切换来隐藏内存延迟。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">SIMT 执行模型</p>
<p>GPU 采用单指令多线程（Single Instruction, Multiple Threads，SIMT）执行模型。在一个 warp（32 个线程）内，所有线程执行相同指令但作用于不同数据。当线程出现分支（例如 <code>if/else</code>）时，两条路径都会被串行执行——称为 <em>warp 分支（warp divergence）</em>。这就是为什么 GPU 内核必须最小化分支的原因。</p>

<p>对于 LLM 工作负载，主要运算（GEMM、attention、softmax）在线程间具有统一的控制流，非常适合 SIMT 执行。</p>
</div>
