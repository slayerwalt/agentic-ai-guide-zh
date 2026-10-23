# 2.1.6　GPU 内存层级与带宽 {#gpu-memory-hierarchy-and-bandwidth}

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

### 寄存器。 {#registers.}

<p>每个 CUDA 线程都有自己的私有寄存器文件。寄存器是芯片上最快的存储——读写在一个时钟周期内完成且无需仲裁。A100 每个 SM 有 65,536 个 32 位寄存器。寄存器溢出到本地内存（L1/L2）是主要的性能隐患。</p>

### SRAM——共享内存 / L1。 {#sram-shared-memory-l1.}

<p>A100 上每个 SM 拥有 192 KB 的 L1/共享内存池（H100 为 256 KB），其中最多 164 KB 可配置为共享内存。共享内存由程序员（或较新版本 CUDA 中由编译器）显式管理。例如，FlashAttention 的整个设计都基于一个洞察：注意力分块计算可以装入 SRAM。</p>

### L2 缓存。 {#l2-cache.}

<p>A100 上的 40 MB L2 由所有 108 个 SM 共享。它充当 SRAM 与 HBM 之间的中转区。对于具有良好空间局部性的工作负载（例如在一个 batch 内被反复访问的权重矩阵），L2 命中率可以大幅降低实际 HBM 流量。</p>

### HBM——高带宽显存。 {#hbm-high-bandwidth-memory.}

<p>高带宽显存（High Bandwidth Memory，HBM）是直接安装在 GPU 封装上的堆叠 DRAM，通过宽带中介层连接。A100 SXM 拥有 80 GB HBM2e，带宽 2 TB/s；H100 SXM5 拥有 80 GB HBM3，带宽 3.35 TB/s。它是模型权重、KV cache、激活和优化器状态的主要工作内存。</p>

### 通过 PCIe 访问 CPU DRAM。 {#cpu-dram-via-pcie.}

<p>GPU HBM 与 CPU DRAM 之间的数据传输需经过 PCIe 总线。PCIe Gen4 <span class="arithmatex">\(\times\)</span>16 每方向提供约 32 GB/s（双向 64 GB/s）；Gen5 翻倍。这相对于 HBM（每方向）是约 60<span class="arithmatex">\(\times\)</span> 的带宽下降。CPU 卸载（ZeRO-Infinity、DeepSpeed）利用这条链路，但必须谨慎使用以免成为瓶颈。</p>

### NVMe。 {#nvme.}

<p>NVMe SSD（例如 Samsung 990 Pro）顺序读取可达约 7 GB/s。ZeRO-Infinity 可以将优化器状态卸载到 NVMe，但只有在计算与 I/O 比率非常高（大 batch 尺寸、训练步骤较慢）时才可行。</p>
