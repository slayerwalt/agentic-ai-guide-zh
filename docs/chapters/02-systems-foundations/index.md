# 第 2 章 · 面向 LLM 的系统基础 {#systems-foundations-for-llms}

## 2.1　GPU 架构——从硅片到 LLM 训练 {#gpu-architecture-from-silicon-to-llm-training}

<p>现代大型语言模型几乎完全依赖 GPU（图形处理器，Graphics Processing Unit）进行训练和服务。理解 GPU 架构对于在并行策略、内存管理、内核优化和基础设施规模等方面做出明智决策至关重要。本节针对 LLM 工作负载，系统地介绍 GPU 硬件相关知识。</p>

### 2.1.1　为什么深度学习要用 GPU？ {#why-gpus-for-deep-learning}

<p>GPU 和 CPU 代表了根本不同的硬件设计哲学。理解这种差异能解释为何 LLM 训练在 GPU 上能快 100–1000 倍。</p>

<div class="admonition tip"><p class="admonition-title">CPU 与 GPU——基本设计哲学</p>
<ul>
<li><p><strong>CPU</strong> 针对<em>延迟</em>进行优化——它们以尽可能快的速度执行少量线程，配备大型缓存、分支预测器和乱序执行单元。现代 CPU 拥有 8–96 个核心。</p></li>
<li><p><strong>GPU</strong> 针对<em>吞吐量</em>进行优化——它们并行执行数千个线程，每个线程只做简单工作。现代 GPU 拥有数千个“核心”（执行单元），它们被组织成流式多处理器（Streaming Multiprocessor，SM）。</p></li>
</ul>

<p>深度学习工作负载以矩阵乘法（在 <span class="arithmatex">\(O(n^2)\)</span> 数据上执行 <span class="arithmatex">\(O(n^3)\)</span> 操作）为主，这是天然并行的。70B 模型的单次 transformer 前向传播每个 token 需要约 140 TFLOP 计算量——非常适合 GPU 的吞吐能力。</p>
</div>

### 2.1.2　NVIDIA GPU 微架构代际演进 {#nvidia-gpu-microarchitecture-generations}

<p>NVIDIA 已经发布了一系列 GPU 架构，每一代都为深度学习带来关键创新：</p>

<p class="table-caption" id="table-2-1"><span class="table-number">表 2.1　</span>面向深度学习的 NVIDIA GPU 微架构时间线。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>架构</strong></th><th scope="col"><strong>年份</strong></th><th scope="col"><strong>旗舰</strong></th><th scope="col"><strong>深度学习关键创新</strong></th></tr></thead>
<tbody><tr><td>Pascal</td><td>2016</td><td>P100</td><td>首款 HBM GPU；FP16 支持；NVLink 1</td></tr>
<tr><td>Volta</td><td>2017</td><td>V100</td><td><strong>Tensor Cores</strong>（第一代）；混合精度训练</td></tr>
<tr><td>Turing</td><td>2018</td><td>T4</td><td>INT8 推理；RT cores（非 ML 用途）</td></tr>
<tr><td>Ampere</td><td>2020</td><td>A100</td><td>BF16 Tensor Cores；TF32；第 3 代 NVLink；MIG</td></tr>
<tr><td>Hopper</td><td>2022</td><td>H100</td><td>FP8 Tensor Cores；TMA；Transformer Engine；NVLink 4</td></tr>
<tr><td>Blackwell</td><td>2024</td><td>B200</td><td>第 2 代 Transformer Engine；NVLink 5（1.8 TB/s）；FP4</td></tr></tbody>
</table>
</div>

### 2.1.3　LLM 训练与推理常用 GPU {#common-gpus-for-llm-training-and-inference}

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

### 2.1.4　GPU 内部架构——流式多处理器（SM） {#gpu-internal-architecture-the-streaming-multiprocessor-sm}

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

### 2.1.5　各代际 GPU 芯片的扩展 {#gpu-chip-scaling-across-generations}

<p>NVIDIA GPU 架构的演进显示出在计算密度、片上内存和深度学习专用单元方面持续扩展：</p>

<p class="table-caption" id="table-2-3"><span class="table-number">表 2.3　</span>各代 NVIDIA 架构在 SM 层面的扩展。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>架构</strong></th><th scope="col"><strong>SM 数</strong></th><th scope="col"><strong>TCs/SM</strong></th><th scope="col"><strong>SRAM/SM</strong></th><th scope="col"><strong>L2</strong></th><th scope="col"><strong>关键变化</strong></th></tr></thead>
<tbody><tr><td>Volta (V100)</td><td>80</td><td>8</td><td>128 KB</td><td>6 MB</td><td>引入 Tensor Cores</td></tr>
<tr><td>Ampere (A100)</td><td>108</td><td>4</td><td>192 KB</td><td>40 MB</td><td>BF16/TF32；更大的 L2</td></tr>
<tr><td>Hopper (H100)</td><td>132</td><td>4</td><td>256 KB</td><td>50 MB</td><td>TMA；FP8；Thread Block Clusters</td></tr>
<tr><td>Blackwell (B200)</td><td>148</td><td>4</td><td>256 KB</td><td>128 MB</td><td>2<span class="arithmatex">\(\times\)</span> 裸片；FP4；TMEM；NVLink 5</td></tr></tbody>
</table>
</div>

### 2.1.6　GPU 内存层级与带宽 {#gpu-memory-hierarchy-and-bandwidth}

<p>现代 GPU 训练和推理性能几乎完全取决于<em>你如何管理数据在内存层级之间的搬运</em>。理解内存层级不是可选项——它是后续章节讨论每一种优化技术的基础。</p>

<div class="admonition note"><p class="admonition-title">GPU 内存层级——A100 80GB 参考数字</p>
<span id="table-2-inline-4"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>层级</strong></th><th scope="col"><strong>容量</strong></th><th scope="col"><strong>带宽</strong></th><th scope="col"><strong>延迟</strong></th><th scope="col"><strong>位置</strong></th></tr></thead>
<tbody><tr><td>寄存器</td><td><span class="arithmatex">\(\sim\)</span>256 KB/SM</td><td><span class="arithmatex">\(&gt;\)</span>100 TB/s</td><td>1 周期</td><td>片上，每线程</td></tr>
<tr><td>SRAM (shared)</td><td>164 KB/SM</td><td><span class="arithmatex">\(\sim\)</span>19 TB/s</td><td><span class="arithmatex">\(\sim\)</span>20 周期</td><td>片上，每 SM</td></tr>
<tr><td>L2 缓存</td><td>共 40 MB</td><td><span class="arithmatex">\(\sim\)</span>5 TB/s</td><td><span class="arithmatex">\(\sim\)</span>200 周期</td><td>片上，共享</td></tr>
<tr><td>HBM2e (VRAM)</td><td>80 GB</td><td>2 TB/s</td><td><span class="arithmatex">\(\sim\)</span>200 ns</td><td>封装上（5 个堆栈）</td></tr>
<tr><td>CPU DRAM</td><td>512 GB+</td><td><span class="arithmatex">\(\sim\)</span>25 GB/s</td><td><span class="arithmatex">\(\sim\)</span>10 <span class="arithmatex">\(\mu\)</span>s</td><td>主机（PCIe 4）</td></tr>
<tr><td>NVMe SSD</td><td>TB 级</td><td>7 GB/s</td><td><span class="arithmatex">\(\sim\)</span>100 <span class="arithmatex">\(\mu\)</span>s</td><td>主机存储</td></tr></tbody>
</table>
</div>
</div>

<div class="admonition tip"><p class="admonition-title">为何各层级差距如此巨大</p>
<p>内存层级中每一级大约比上一级<strong>慢 10<span class="arithmatex">\(\times\)</span>，大 100–1000<span class="arithmatex">\(\times\)</span></strong>。A100 有 312 TFLOP/s 的 BF16 tensor-core 吞吐，但 HBM 带宽只有 2 TB/s。这意味着从 HBM 每加载一字节，你可以在下一字节到达之前完成 <span class="arithmatex">\(312 \times 10^{12} / (2 \times 10^{12}) \approx 156\)</span> 次浮点运算。如果你的内核每字节运算少于 156 FLOPs，它就是<em>内存密集（Memory-Bound）</em>的——计算单元在空等数据。</p>
</div>

##### 寄存器。 {#registers.}

<p>每个 CUDA 线程都有自己的私有寄存器文件。寄存器是芯片上最快的存储——读写在一个时钟周期内完成且无需仲裁。A100 每个 SM 有 65,536 个 32 位寄存器。寄存器溢出到本地内存（L1/L2）是主要的性能隐患。</p>

##### SRAM——共享内存 / L1。 {#sram-shared-memory-l1.}

<p>A100 上每个 SM 拥有 192 KB 的 L1/共享内存池（H100 为 256 KB），其中最多 164 KB 可配置为共享内存。共享内存由程序员（或较新版本 CUDA 中由编译器）显式管理。例如，FlashAttention 的整个设计都基于一个洞察：注意力分块计算可以装入 SRAM。</p>

##### L2 缓存。 {#l2-cache.}

<p>A100 上的 40 MB L2 由所有 108 个 SM 共享。它充当 SRAM 与 HBM 之间的中转区。对于具有良好空间局部性的工作负载（例如在一个 batch 内被反复访问的权重矩阵），L2 命中率可以大幅降低实际 HBM 流量。</p>

##### HBM——高带宽显存。 {#hbm-high-bandwidth-memory.}

<p>高带宽显存（High Bandwidth Memory，HBM）是直接安装在 GPU 封装上的堆叠 DRAM，通过宽带中介层连接。A100 SXM 拥有 80 GB HBM2e，带宽 2 TB/s；H100 SXM5 拥有 80 GB HBM3，带宽 3.35 TB/s。它是模型权重、KV cache、激活和优化器状态的主要工作内存。</p>

##### 通过 PCIe 访问 CPU DRAM。 {#cpu-dram-via-pcie.}

<p>GPU HBM 与 CPU DRAM 之间的数据传输需经过 PCIe 总线。PCIe Gen4 <span class="arithmatex">\(\times\)</span>16 每方向提供约 32 GB/s（双向 64 GB/s）；Gen5 翻倍。这相对于 HBM（每方向）是约 60<span class="arithmatex">\(\times\)</span> 的带宽下降。CPU 卸载（ZeRO-Infinity、DeepSpeed）利用这条链路，但必须谨慎使用以免成为瓶颈。</p>

##### NVMe。 {#nvme.}

<p>NVMe SSD（例如 Samsung 990 Pro）顺序读取可达约 7 GB/s。ZeRO-Infinity 可以将优化器状态卸载到 NVMe，但只有在计算与 I/O 比率非常高（大 batch 尺寸、训练步骤较慢）时才可行。</p>

### 2.1.7　算术强度与 Roofline 模型 {#arithmetic-intensity-and-the-roofline-model}

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

### 2.1.8　Attention 是内存密集的；FFN 是计算密集的 {#attention-is-memory-bound-ffn-is-compute-bound}

<div class="admonition tip"><p class="admonition-title">Transformer 中的两种区间</p>
<p>一个 transformer 块有两个主要组件，它们的算术强度截然不同：</p>

<ul>
<li><p><strong>Attention：</strong>作用于 <span class="arithmatex">\(n \times d\)</span> 张量。<span class="arithmatex">\(QK^T\)</span> 乘积是 <span class="arithmatex">\(O(n^2 d)\)</span> FLOPs，但注意力分数需要 <span class="arithmatex">\(O(n^2)\)</span> 内存。在长序列下，内存流量占主导地位——attention 是<em>内存密集（Memory-Bound）</em>的。</p></li>
<li><p><strong>FFN（MLP）：</strong>两个大型线性层，权重矩阵形状为 <span class="arithmatex">\([d_{\text{model}}, 4d_{\text{model}}]\)</span>。这些是高算术强度的大型 GEMM——FFN 是<em>计算密集（Compute-Bound）</em>的。</p></li>
</ul>

<p>这就是为什么 FlashAttention（内存优化）对 attention 有效但对 FFN 无效，而量化（缩小权重体积）对 FFN 的帮助比对 attention 更大。</p>
</div>

### 2.1.9　Tensor Cores {#tensor-cores}

<div class="admonition note"><p class="admonition-title">什么是 Tensor Cores？</p>
<p>张量核心（Tensor Core）是 Volta 架构（2017）引入的专用矩阵乘加（MMA）单元。每个 Tensor Core 在单个时钟周期内执行一次 <span class="arithmatex">\(4\times4\times4\)</span> 矩阵乘法：</p>



<div class="arithmatex">\[D = A \times B + C \quad (4\times4 \text{ matrices})\]</div>

<p>A100 在 108 个 SM 中共有 <strong>432 个 Tensor Cores</strong>（每个 SM 4 个，每个子分区 1 个）。在 BF16 精度下提供 312 TFLOP/s 吞吐——约是 FP32 CUDA cores 吞吐的 <span class="arithmatex">\(16\times\)</span>。</p>
</div>

<ul>
<li><p><strong>支持的精度：</strong>FP64、TF32、BF16、FP16、INT8、FP8（H100+）。</p></li>
<li><p><strong>累加：</strong>内部始终使用 FP32，即使输入为 BF16。这能防止点积过程中的灾难性抵消。</p></li>
<li><p><strong>要求：</strong>当矩阵维度是 8（BF16）或 16（FP8）的倍数时，Tensor Cores 效率最高。为此进行填充通常是值得的。</p></li>
<li><p><strong>WGMMA（H100）：</strong>Hopper 引入了 warpgroup 级别的 MMA 指令，作用于更大的分块（64<span class="arithmatex">\(\times\)</span>256<span class="arithmatex">\(\times\)</span>16），并可与 TMA（Tensor Memory Accelerator）数据搬运形成流水线。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">Tensor Core 陷阱</p>
<p>只有当你的内核是<em>计算密集（Compute-Bound）</em>时，Tensor Cores 才有用。如果你跑的是小 batch（batch size 1 的推理），GEMM 分块很小，Tensor Core 利用率很低，你又回到内存密集区间。这就是为什么推理引擎要激进地对请求进行 batch。</p>
</div>

### 2.1.10　通信架构——NVLink、InfiniBand 与 PCIe {#communication-architecture-nvlink-infiniband-and-pcie}

<p>分布式 LLM 训练和推理需要在 GPU、节点和存储之间搬运海量数据。通信结构往往是大规模训练的瓶颈。</p>

##### PCIe——主机与设备的链路。 {#pcie-the-host-device-link.}

<div class="admonition note"><p class="admonition-title">PCIe 各代</p>
<span id="table-2-inline-5"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>代际</strong></th><th scope="col"><strong>x16 带宽（每方向）</strong></th><th scope="col"><strong>双向</strong></th><th scope="col"><strong>备注</strong></th></tr></thead>
<tbody><tr><td>PCIe Gen3</td><td>16 GB/s</td><td>32 GB/s</td><td>常见于较旧服务器</td></tr>
<tr><td>PCIe Gen4</td><td>32 GB/s</td><td>64 GB/s</td><td>A100 PCIe、当前主流服务器</td></tr>
<tr><td>PCIe Gen5</td><td>64 GB/s</td><td>128 GB/s</td><td>H100 PCIe，正在普及</td></tr></tbody>
</table>
</div>
</div>

<p>PCIe 用于：</p>

<ul>
<li><p>CPU <span class="arithmatex">\(\leftrightarrow\)</span> GPU 数据传输（模型加载、CPU 卸载）</p></li>
<li><p>当 NVLink 不可用时的跨节点 GPU 通信（罕见，且非常慢）</p></li>
<li><p>NVMe 存储访问（经由 CPU）</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">PCIe 不应用于 GPU 间通信</p>
<p>如果有 NVLink 可用，绝不要让 GPU-GPU 通信走 PCIe。PCIe 带宽（32 GB/s）比 NVLink 4（900 GB/s）低 28<span class="arithmatex">\(\times\)</span>。在没有 NVLink 的多 GPU 服务器中（例如消费级 GPU），GPU 间带宽被限制在 PCIe 水平，使得张量并行极其缓慢。</p>
</div>

##### NVLink——节点内高速互联。 {#nvlink-intra-node-high-speed-interconnect.}

<div class="admonition note"><p class="admonition-title">NVLink 各代</p>
<span id="table-2-inline-6"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>代际</strong></th><th scope="col"><strong>链路数</strong></th><th scope="col"><strong>总带宽</strong></th><th scope="col"><strong>GPU</strong></th></tr></thead>
<tbody><tr><td>NVLink 2</td><td>6</td><td>300 GB/s</td><td>V100</td></tr>
<tr><td>NVLink 3</td><td>12</td><td>600 GB/s</td><td>A100</td></tr>
<tr><td>NVLink 4</td><td>18</td><td>900 GB/s</td><td>H100</td></tr>
<tr><td>NVLink 5</td><td>18</td><td>1800 GB/s</td><td>B200 (Blackwell)</td></tr></tbody>
</table>
</div>
</div>

<p>NVLink 是同一节点内 GPU 之间的点对点互联。每条链路都是双向的。H100 SXM5 拥有 18 条 NVLink 4 链路，每条提供 50 GB/s 双向带宽，总计 900 GB/s。</p>

##### NVSwitch。 {#nvswitch.}

<p>在 DGX H100 系统中，所有 8 块 GPU 都通过 NVSwitch 互联——这是一种提供<em>全二分带宽（full bisection bandwidth）</em>的专用交换芯片。这意味着任意 GPU 都能以完整 NVLink 速度同时与任何其他 GPU 通信，而不仅仅是环上邻居。</p>

<div class="admonition tip"><p class="admonition-title">环形拓扑 vs. 全二分</p>
<p>在环形拓扑（8 GPU）中，AllReduce 需要数据绕环传输。每条链路必须承载总数据的 <span class="arithmatex">\(\frac{2(N-1)}{N}\)</span>，因此算法带宽为 <span class="arithmatex">\(B_{\text{link}} \times \frac{N}{2(N-1)}\)</span>（<span class="arithmatex">\(N=8\)</span> 时约为 <span class="arithmatex">\(0.57 \times B_{\text{link}}\)</span>）。借助 NVSwitch 的全二分带宽，AllReduce 可以使用基于树的算法同时使用所有链路，达到接近峰值的带宽。在 DGX H100 实际测试中：ring 达到约 700 GB/s 总线带宽，NVSwitch 达到约 900 GB/s。</p>
</div>

##### InfiniBand——节点间通信。 {#infiniband-inter-node-communication.}

<p>对于节点（服务器）之间的通信，InfiniBand 提供高带宽、低延迟的网络，并支持直接 GPU 内存访问。</p>

<div class="admonition note"><p class="admonition-title">InfiniBand NDR</p>
<ul>
<li><p><strong>NDR 400Gb/s</strong> = 每端口 50 GB/s（单向）</p></li>
<li><p><strong>HDR 200Gb/s</strong> = 每端口 25 GB/s（上一代）</p></li>
<li><p><strong>RDMA：</strong>远程直接内存访问（Remote Direct Memory Access）——GPU 可读写远端 GPU 内存而无需远端 CPU 参与</p></li>
<li><p><strong>GPUDirect RDMA：</strong>数据直接经由 HBM <span class="arithmatex">\(\to\)</span> NIC <span class="arithmatex">\(\to\)</span> 网络 <span class="arithmatex">\(\to\)</span> NIC <span class="arithmatex">\(\to\)</span> HBM，完全绕过 CPU 和系统 DRAM</p></li>
<li><p><strong>延迟：</strong>小消息约 1–2 <span class="arithmatex">\(\mu\)</span>s（对比 TCP/IP 的约 100 <span class="arithmatex">\(\mu\)</span>s）</p></li>
</ul>
</div>

##### 胖树拓扑（Fat-Tree Topology）。 {#fat-tree-topology.}

<p>大型 GPU 集群使用胖树（Fat-Tree Topology）网络拓扑。使用 <span class="arithmatex">\(k\)</span> 端口交换机的 3 层胖树可支持 <span class="arithmatex">\(k^3/4\)</span> 个节点并保持全二分带宽。对于 <span class="arithmatex">\(k=64\)</span> 端口的 400Gb/s NDR 交换机：<span class="arithmatex">\(64^3/4 = 65{,}536\)</span> 个节点。</p>

##### 轨道优化拓扑（Rail-Optimized Topology）。 {#rail-optimized-topology.}

<p>实践中，集群使用<em>轨道优化拓扑（Rail-Optimized Topology）</em>，节点中的每块 GPU 连接到不同的机架顶端（top-of-rack）交换机。这样可以保证 AllReduce 操作（涉及所有 GPU）同时使用所有网络链路，从而最大化带宽。</p>

##### 分布式 LLM 训练中的通信模式。 {#communication-patterns-in-distributed-llm-training.}

<p>分布式训练依赖集体通信原语。原语的选择决定了带宽需求和扩展行为。</p>

<div class="admonition note"><p class="admonition-title">通信原语</p>
<span id="table-2-inline-7"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>原语</strong></th><th scope="col"><strong>使用场景</strong></th><th scope="col"><strong>通信量</strong></th></tr></thead>
<tbody><tr><td>AllReduce</td><td>梯度同步（DDP、FSDP）</td><td><span class="arithmatex">\(2(N-1)/N \times\)</span> 参数大小</td></tr>
<tr><td>AllGather</td><td>收集分片权重（FSDP）</td><td><span class="arithmatex">\((N-1)/N \times\)</span> 参数大小</td></tr>
<tr><td>ReduceScatter</td><td>分散梯度（FSDP）</td><td><span class="arithmatex">\((N-1)/N \times\)</span> 参数大小</td></tr>
<tr><td>AllGather</td><td>张量并行激活</td><td>激活大小</td></tr>
<tr><td>Point-to-Point</td><td>流水线并行（send/recv）</td><td>micro-batch 激活</td></tr>
<tr><td>Broadcast</td><td>权重同步（新 worker）</td><td>完整模型大小</td></tr></tbody>
</table>
</div>
</div>

<div class="admonition example"><p class="admonition-title">带宽计算——70B 模型的梯度 AllReduce</p>
<p><strong>设定：</strong>70B 参数模型，BF16 梯度，8 节点 <span class="arithmatex">\(\times\)</span> 8 GPU = 64 GPU。数据并行度 = 64。</p>

<p><strong>梯度大小：</strong><span class="arithmatex">\(70 \times 10^9 \times 2\)</span> 字节 <span class="arithmatex">\(= 140\)</span> GB。</p>

<p><strong>每块 GPU 的 AllReduce 量</strong>（环形）：<span class="arithmatex">\(2 \times (64-1)/64 \times 140 \approx 275\)</span> GB。</p>

<p><strong>可用节点间带宽：</strong>8 GPU/节点 <span class="arithmatex">\(\times\)</span> 50 GB/s/GPU <span class="arithmatex">\(= 400\)</span> GB/s（采用轨道优化拓扑，8 块网卡全部启用）。</p>

<p><strong>AllReduce 耗时：</strong>每步 <span class="arithmatex">\(275 / 400 \approx 0.69\)</span> 秒。</p>

<p><strong>含义：</strong>对于一个 1 秒的计算步骤，通信增加 0.69 秒（占总步骤时间 41%）。这就是为什么梯度压缩、混合精度、以及把通信与计算重叠的 FSDP 至关重要。</p>
</div>

##### 网络拓扑示意图。 {#network-topology-diagram.}

<p>下图展示了典型的双节点 GPU 集群拓扑，同时显示节点内（NVLink）和节点间（InfiniBand）通信路径。</p>

<figure id="figure-2-3"><a href="../../assets/figures/fig_018_fig18.png" target="_blank" rel="noopener" aria-label="查看图 2.3 原图" title="点击查看原图"><img src="../../assets/figures/fig_018_fig18.png" alt="两节点 8-GPU 拓扑。节点内：通过 NVSwitch 的 NVLink 4（总计 900 GB/s）。节点间：通过机架顶端交换机的 InfiniBand NDR 400Gb/s。每个节点配有 8 块 IB 网卡（每块 GPU 一块），用于轨道优化 AllReduce。" loading="lazy"></a><figcaption><span class="figure-number">图 2.3　</span>两节点 8-GPU 拓扑。节点内：通过 NVSwitch 的 NVLink 4（总计 900 GB/s）。节点间：通过机架顶端交换机的 InfiniBand NDR 400Gb/s。每个节点配有 8 块 IB 网卡（每块 GPU 一块），用于轨道优化 AllReduce。</figcaption></figure>

<div class="admonition tip"><p class="admonition-title">根据带宽选择并行策略</p>
<ul>
<li><p><strong>张量并行（TP）：</strong>每层都需要 all-reduce——仅在节点内通过 NVLink 使用。TP=8 是 H100 DGX 节点的标准。</p></li>
<li><p><strong>流水线并行（PP）：</strong>阶段之间的点对点通信——可以跨节点，但会引入流水线气泡开销。当模型大到单靠 TP 装不下时使用。</p></li>
<li><p><strong>数据并行（DP）：</strong>梯度的 AllReduce——可以通过 IB 跨节点。在快速 IB 下扩展性良好。</p></li>
<li><p><strong>FSDP/ZeRO：</strong>AllGather + ReduceScatter——类似于 DP 但会分片优化器状态。对于大模型优于 DP。</p></li>
</ul>
</div>

## 2.2　vLLM——分页注意力与高吞吐推理 {#vllm-pagedattention-and-high-throughput-inference}

<p>vLLM <a class="book-citation" id="cite-kwon2023efficient-1" href="../../references.md#ref-kwon2023efficient" aria-label="参考文献 195">[195]</a> 引入了分页注意力（PagedAttention），它借鉴操作系统用于管理 RAM 的分页抽象，并将其应用于 GPU 的键值缓存（KV Cache）。在 LLM 推理过程中，<em>KV cache</em>——存储了所有先前 token 的 key 和 value 张量——是最大的内存消耗者。高效管理 KV cache 是高吞吐推理的核心挑战。</p>

### 2.2.1　KV cache 碎片化问题 {#the-kv-cache-fragmentation-problem}

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

### 2.2.2　分页注意力——为 KV cache 引入虚拟内存 {#pagedattention-virtual-memory-for-kv-caches}

<p>分页注意力（PagedAttention，Kwon 等，2023）借鉴了操作系统的<em>分页</em>抽象。它不再为每条序列分配一块连续内存，而是把 KV cache 切分成固定大小的<strong>页（page）</strong>（即块），并通过一个间接表——类似 CPU 页表——将每条序列的逻辑 token 位置翻译为分散的物理 GPU 内存地址。</p>

<div class="admonition note"><p class="admonition-title">PagedAttention 核心概念</p>
<ul>
<li><p><strong>块大小：</strong>通常每块 16 个 token（可调）。每个块存储 <span class="arithmatex">\(16 \times 2 \times L \times H \times d\)</span> 个元素。</p></li>
<li><p><strong>块表（Block table）：</strong>每条序列一份的映射，把逻辑块索引映射到 GPU 内存池中的物理块索引。</p></li>
<li><p><strong>物理块池：</strong>预先分配好的固定大小块池。分配为 <span class="arithmatex">\(O(1)\)</span>——只需从空闲列表弹出。</p></li>
<li><p><strong>注意力内核：</strong>做了修改，在注意力计算时使用块表从不连续的物理位置 gather KV 块。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">块表示例</p>
<p>假设块大小 = 4 个 token，我们有两条序列：</p>

<ul>
<li><p>序列 A（7 个 token）：逻辑块 [0,1] <span class="arithmatex">\(\to\)</span> 物理块 [3, 7]</p></li>
<li><p>序列 B（5 个 token）：逻辑块 [0,1] <span class="arithmatex">\(\to\)</span> 物理块 [1, 5]</p></li>
</ul>

<p>物理块 3 持有序列 A 的 token 0–3。物理块 7 持有序列 A 的 token 4–6（部分填充）。序列 A 的注意力内核按顺序从物理块 3 和 7 读取，并以块表作为间接层。</p>
</div>

### 2.2.3　PagedAttention 的收益 {#benefits-of-pagedattention}

##### 近零浪费。 {#near-zero-waste.}

<p>内部碎片至多为每条序列一个部分填充块（最后一个块）。当块大小为 16 时，最差浪费为每条序列 15 个 token——可忽略不计。外部碎片被消除，因为块是固定大小、可互换的。</p>

##### 动态分配。 {#dynamic-allocation.}

<p>块在序列增长时按需分配。无需预先知道最终序列长度。这对生成任务至关重要，因为输出长度未知。</p>

##### 前缀共享（写时复制）。 {#prefix-sharing-copy-on-write.}

<p>多条共享同一前缀（例如同一系统提示）的序列可以<em>共享相同的物理块</em>来表示该前缀。块表只需让多条序列指向相同的物理块。当某条序列需要写入共享块（与前缀分叉）时，会触发写时复制（Copy-on-Write）。</p>

<div class="admonition tip"><p class="admonition-title">前缀共享带来的节约</p>
<p>在一个拥有 1000-token 系统提示、服务 128 位并发用户的聊天机器人中：</p>

<ul>
<li><p>无前缀共享：仅系统提示 KV cache 就需 <span class="arithmatex">\(128 \times 1000 \times 327{,}680 / 10^9 \approx 42\)</span> GB</p></li>
<li><p>有前缀共享：<span class="arithmatex">\(1 \times 1000 \times 327{,}680 / 10^9 \approx 0.33\)</span> GB</p></li>
<li><p>节约：共享前缀部分约 128<span class="arithmatex">\(\times\)</span></p></li>
</ul>
</div>

##### 通过 swap 抢占。 {#preemption-via-swap.}

<p>当 GPU 显存耗尽时，vLLM 可以通过把某条序列的 KV 块换出到 CPU DRAM（或者直接丢弃，之后重新计算）来<em>抢占</em>该序列。这之所以可行，是因为块是自包含且非连续的——若是连续分配则需要复制整个缓冲区。</p>

### 2.2.4　连续批处理 {#continuous-batching}

<p>传统的批处理（“静态批处理”）会等到一批中<em>所有</em>序列都完成后才开始新批次。如果一条序列生成 500 个 token、另一条生成 10 个，那么 GPU 在短序列上有 490 步是空闲的。这极其浪费。</p>

<div class="admonition note"><p class="admonition-title">连续批处理（Continuous Batching）</p>
<p>连续批处理（Continuous Batching，也叫迭代级调度）一次处理一个<em>解码步</em>。每一步之后：</p>

<ol>
<li><p>检查哪些序列已完成（生成了 EOS token）</p></li>
<li><p>把已完成的序列从批中移除，释放其 KV 块</p></li>
<li><p>加入新的等待序列以填补空出的槽位</p></li>
<li><p>在更新后的批上运行下一解码步</p></li>
</ol>

<p>批次组成每一步都在变化——序列动态地加入和离开。这让 GPU 利用率接近 100%，并大幅提升吞吐（相对静态批处理提升 1.5–3<span class="arithmatex">\(\times\)</span>）。PagedAttention 在这里至关重要：在批次中动态添加/移除序列需要动态分配/释放 KV 块，只有在分页内存下才高效。</p>
</div>

### 2.2.5　vLLM 中的投机解码 {#speculative-decoding-integration}

<p>投机解码（Speculative Decoding）使用一个小型<em>草稿模型</em>（draft model，例如 1B 参数）快速提出 <span class="arithmatex">\(k\)</span> 个候选 token，再由大型<em>目标模型</em>（target model）在一次前向传播中验证。从第一个被拒绝处之前的所有 token 都会被接受（预期接受数：每次验证 3–5 个 token）。这为延迟敏感的单序列生成带来 2–3<span class="arithmatex">\(\times\)</span> 提速，且无质量损失。</p>

<p>vLLM 将投机解码与 PagedAttention 集成：</p>

<ul>
<li><p>草稿 token 被分配投机性 KV 块</p></li>
<li><p>拒绝时，投机块被释放（在分页分配下成本极低）</p></li>
<li><p>接受时，投机块被提升为主序列的一部分</p></li>
<li><p>块表更新为 <span class="arithmatex">\(O(k)\)</span>——只需更新少量表项</p></li>
</ul>

### 2.2.6　具体显存节约——大规模 70B 模型 {#concrete-memory-savings-70b-model-at-scale}

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

### 2.2.7　vLLM：端到端系统 {#vllm-end-to-end-system}

<p>vLLM 将 PagedAttention 封装在一整套服务栈中：连续批处理（Continuous Batching）、前缀缓存（Prefix Caching）、投机解码（Speculative Decoding）以及张量并行的模型分片协同工作，最大化每美元 GPU 的吞吐。</p>

### 2.2.8　架构总览 {#architecture-overview}

<figure id="figure-2-4"><a href="../../assets/figures/fig_019_fig19.png" target="_blank" rel="noopener" aria-label="查看图 2.4 原图" title="点击查看原图"><img src="../../assets/figures/fig_019_fig19.png" alt="vLLM 架构：请求自上而下流动。Scheduler 管理准入与抢占，Block Manager 负责虚拟到物理 KV cache 映射（像操作系统页表一样），Model Executor 在 GPU HBM 中从预分配的块池读取数据并执行批量推理。" loading="lazy"></a><figcaption><span class="figure-number">图 2.4　</span>vLLM 架构：请求自上而下流动。Scheduler 管理准入与抢占，Block Manager 负责虚拟到物理 KV cache 映射（像操作系统页表一样），Model Executor 在 GPU HBM 中从预分配的块池读取数据并执行批量推理。</figcaption></figure>

### 2.2.9　核心组件 {#core-components}

<ul>
<li><p><strong>API Server</strong>：接收 OpenAI 兼容的请求（completions、chat）。对输入进行 tokenize，并创建“序列组（sequence groups）”（用于 beam search 或多样本）。</p></li>
<li><p><strong>Scheduler</strong>：vLLM 的大脑。维护三个队列：</p>

<ul>
<li><p><code>waiting</code>：尚未开始的新请求（等待 prefill）</p></li>
<li><p><code>running</code>：正在生成 token（decode 阶段）</p></li>
<li><p><code>swapped</code>：被抢占、KV cache 已卸载到 CPU 的请求</p></li>
</ul>

<p>每次迭代，调度器根据可用 GPU 内存块决定运行哪些请求。</p></li>
<li><p><strong>Block Manager</strong>：为 KV cache 实现虚拟内存抽象。将逻辑块（按序列）映射到物理块（GPU 内存池中）。负责处理：</p>

<ul>
<li><p>分配（生成新 token <span class="arithmatex">\(\rightarrow\)</span> 需要新块）</p></li>
<li><p>写时复制（Copy-on-Write）（用于 beam search：多 beam 共享前缀块，仅在分叉时复制）</p></li>
<li><p>Swap（抢占/恢复时在 GPU <span class="arithmatex">\(\leftrightarrow\)</span> CPU 之间迁移）</p></li>
<li><p>前缀缓存（Prefix Caching）（当 prompt 共享相同前缀时复用已缓存块）</p></li>
</ul></li>
<li><p><strong>Model Executor</strong>：执行实际的 LLM 前向传播。管理跨 GPU 的张量并行，调度从分页 KV cache 块读取的注意力内核。</p></li>
<li><p><strong>KV Cache Pool</strong>：预分配的 GPU 内存，划分为固定大小的块（默认每块 16 token <span class="arithmatex">\(\times\)</span> num_heads <span class="arithmatex">\(\times\)</span> head_dim <span class="arithmatex">\(\times\)</span> 2 字节）。运行时无动态分配 <span class="arithmatex">\(\rightarrow\)</span> 零碎片。</p></li>
</ul>

### 2.2.10　请求生命周期（端到端流程） {#request-lifecycle-end-to-end-flow}

<ol>
<li><p><strong>到达</strong>：客户端发送 prompt。API server 将其 tokenize，创建一个 <code>SequenceGroup</code>，放入 <code>waiting</code> 队列。</p></li>
<li><p><strong>调度</strong>：每一步，调度器执行：</p>

<ol>
<li><p>检查是否有 <code>swapped</code> 序列可以恢复（有足够空闲块）。</p></li>
<li><p>检查是否有 <code>waiting</code> 序列可以开始 prefill（有足够块容纳完整 prompt）。</p></li>
<li><p>在 <code>running</code> 序列之间预算剩余块（若当前块已满，每条序列每步需要 1 个新块）。</p></li>
<li><p>如果超出预算：<strong>抢占</strong>优先级最低的运行中序列（把 KV 换出到 CPU 或之后重新计算）。</p></li>
</ol></li>
<li><p><strong>Prefill</strong>（请求的第一次迭代）：整个 prompt 在一次前向传播中被处理。所有 prompt token 的 KV cache 被计算并存入已分配的块。这是计算密集（Compute-Bound）的（大批 token）。</p></li>
<li><p><strong>Decode</strong>（后续迭代）：每条序列每步生成一个新 token。所有运行中的序列被一起 batch（连续批处理）。这是内存密集（Memory-Bound）的（读取完整 KV cache，生成 1 个 token）。</p></li>
<li><p><strong>块分配</strong>：每个 decode 步后，如果某条序列的最后一个块已满，Block Manager 分配一个新的物理块并将其映射到下一逻辑块。</p></li>
<li><p><strong>完成</strong>：当一条序列遇到 EOS 或达到最大长度时，它会从 <code>running</code> 中移除。其物理块立即被释放 <span class="arithmatex">\(\rightarrow\)</span> 可供其他序列使用。响应被流式回传给客户端。</p></li>
</ol>

### 2.2.11　前缀缓存（自动 Prompt 缓存） {#prefix-caching-automatic-prompt-caching}

<p>当多个请求共享相同前缀（系统 prompt、few-shot 示例）时：</p>

<ol>
<li><p>对每个逻辑块的 token 内容计算哈希。</p></li>
<li><p>新请求到达时，检查是否有前缀块已在缓存中。</p></li>
<li><p>命中时：跳过这些 token 的 prefill，直接复用物理 KV 块。首 token 延迟（Time-to-first-token）大幅下降。</p></li>
<li><p>驱逐：LRU 策略。缓存块仅在内存压力需要时才被释放。</p></li>
</ol>

<p><strong>影响</strong>：对于具有长系统提示（2K+ token、所有用户共享）的聊天应用，前缀缓存可降低 TTFT 60–80%。</p>

### 2.2.12　vLLM 中的引导（受约束）解码 {#sec:vllm-guided-decoding}

<p>vLLM 通过可插拔的后端原生支持受约束解码（参见 Section <a href="../01-llm-architecture/reading-04.md#sec:constrained-decoding">1.12.11</a>），可以在服务期间以极小的性能开销<em>保证</em>结构化输出。</p>

##### 支持的约束类型。 {#supported-constraint-types.}

<p>OpenAI 兼容 API 通过 <code>guided_*</code> 参数或 <code>response_format</code> 字段接受约束：</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">OpenAI</span>
<span class="n">client</span> <span class="o">=</span> <span class="n">OpenAI</span><span class="p">(</span><span class="n">base_url</span><span class="o">=</span><span class="s2">"http://localhost:8000/v1"</span><span class="p">)</span>

<span class="c1"># --- JSON Schema 约束 ---</span>
<span class="n">response</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">chat</span><span class="o">.</span><span class="n">completions</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Llama-3-70B-Instruct"</span><span class="p">,</span>
    <span class="n">messages</span><span class="o">=</span><span class="p">[{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>
               <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"Extract: name, age, city from: "</span>
                          <span class="s2">"'John is 30 and lives in NYC'"</span><span class="p">}],</span>
    <span class="n">extra_body</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"guided_json"</span><span class="p">:</span> <span class="p">{</span>
            <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"object"</span><span class="p">,</span>
            <span class="s2">"properties"</span><span class="p">:</span> <span class="p">{</span>
                <span class="s2">"name"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"string"</span><span class="p">},</span>
                <span class="s2">"age"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"integer"</span><span class="p">},</span>
                <span class="s2">"city"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"string"</span><span class="p">}</span>
            <span class="p">},</span>
            <span class="s2">"required"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"name"</span><span class="p">,</span> <span class="s2">"age"</span><span class="p">,</span> <span class="s2">"city"</span><span class="p">]</span>
        <span class="p">}</span>
    <span class="p">}</span>
<span class="p">)</span>
<span class="c1"># 输出保证是符合该 schema 的合法 JSON</span>

<span class="c1"># --- 正则表达式约束 ---</span>
<span class="n">response</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">completions</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Llama-3-70B-Instruct"</span><span class="p">,</span>
    <span class="n">prompt</span><span class="o">=</span><span class="s2">"Generate an IPv4 address: "</span><span class="p">,</span>
    <span class="n">extra_body</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"guided_regex"</span><span class="p">:</span> <span class="sa">r</span><span class="s2">"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"</span>
    <span class="p">}</span>
<span class="p">)</span>

<span class="c1"># --- 选项约束 ---</span>
<span class="n">response</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">completions</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Llama-3-70B-Instruct"</span><span class="p">,</span>
    <span class="n">prompt</span><span class="o">=</span><span class="s2">"Sentiment: "</span><span class="p">,</span>
    <span class="n">extra_body</span><span class="o">=</span><span class="p">{</span><span class="s2">"guided_choice"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"positive"</span><span class="p">,</span> <span class="s2">"negative"</span><span class="p">,</span> <span class="s2">"neutral"</span><span class="p">]}</span>
<span class="p">)</span>
</code></pre></div>

##### 后端架构。 {#backend-architecture.}

<p>vLLM 将掩码计算委托给后端引擎：</p>

<ul>
<li><p><strong>XGrammar</strong>（自 v0.7 起为默认）：基于下推自动机的引擎，支持 JSON schema、正则和任意 EBNF 语法。得益于高效的 C++ 核心，对复杂 schema 而言最快。</p></li>
<li><p><strong>Outlines</strong> <a class="book-citation" id="cite-willard2023outlines-1" href="../../references.md#ref-willard2023outlines" aria-label="参考文献 378">[378]</a>：基于 FSM；支持 JSON 和正则。在 XGrammar 不可用时作为回退方案。</p></li>
</ul>

<p>掩码在模型前向传播产生 logits <em>之后</em>、采样<em>之前</em>施加——实际中每步只增加不到 1 ms，因为 FSM/PDA 状态转移和预先计算的索引查找均为 <span class="arithmatex">\(O(1)\)</span>。</p>

##### 性能影响。 {#performance-impact.}

<p>由于约束仅对 logits 进行掩码（不重新计算 attention 或 FFN），吞吐损失可忽略（基准测试中 <span class="arithmatex">\(&lt;\)</span>2%）。主要成本来自把 schema <em>编译</em>成 FSM/PDA 索引，耗时 0.5–5 秒，取决于 schema 复杂度。vLLM 会跨请求缓存已编译的 schema，因此每个独特 schema 只需支付一次成本。</p>

<div class="admonition warning"><p class="admonition-title">结构化输出 <span class="arithmatex">\(\neq\)</span> 正确输出</p>
<p>受约束解码保证输出<em>语法上</em>有效（能解析为 JSON、符合 schema 的类型）。但它<em>不</em>保证<em>语义上</em>的正确性——模型仍可能产生能成功解析但事实错误的幻觉值。务必在下游验证业务逻辑。</p>
</div>

<p class="table-caption" id="table-2-4"><span class="table-number">表 2.4　</span>vLLM 与其他方案的性能对比（70B 模型，A100 <span class="arithmatex">\(\times\)</span> 4，TP=4）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>指标</strong></th><th scope="col"><strong>vLLM</strong></th><th scope="col"><strong>HF Generate</strong></th><th scope="col"><strong>原因</strong></th></tr></thead>
<tbody><tr><td>吞吐（tok/s）</td><td>2,500–4,000</td><td>300–600</td><td>连续批处理 + PagedAttention</td></tr>
<tr><td>显存利用率</td><td>90–95%</td><td>50–60%</td><td>零碎片，动态块分配</td></tr>
<tr><td>最大并发序列数</td><td>200–500</td><td>16–32</td><td>分页 KV 消除每序列预留</td></tr>
<tr><td>首 token 延迟</td><td>100–300ms</td><td>500–2000ms</td><td>对重复系统提示启用前缀缓存</td></tr></tbody>
</table>
</div>

##### Dynamo：数据中心级推理编排。 {#heading-2-47}

<p>Dynamo <a class="book-citation" id="cite-dynamo2025nvidia-1" href="../../references.md#ref-dynamo2025nvidia" aria-label="参考文献 270">[270]</a> 是 NVIDIA 的开源编排层，位于各推理引擎（SGLang、TensorRT-LLM、vLLM）之<em>上</em>，把它们组织成协同的多节点服务系统。关键能力包括：分离式服务（把 prefill 与 decode 阶段拆到专用节点池）、基于 KV 缓存局部性的智能请求路由、多级 KV 缓存（GPU <span class="arithmatex">\(\to\)</span> CPU <span class="arithmatex">\(\to\)</span> SSD），以及自动扩缩容。它填补了单 GPU 推理优化（由 vLLM 覆盖）与数据中心级生产服务之间的空白。</p>
