# 2.1.10　通信架构——NVLink、InfiniBand 与 PCIe {#communication-architecture-nvlink-infiniband-and-pcie}

<p>分布式 LLM 训练和推理需要在 GPU、节点和存储之间搬运海量数据。通信结构往往是大规模训练的瓶颈。</p>

### PCIe——主机与设备的链路。 {#pcie-the-host-device-link.}

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

### NVLink——节点内高速互联。 {#nvlink-intra-node-high-speed-interconnect.}

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

### NVSwitch。 {#nvswitch.}

<p>在 DGX H100 系统中，所有 8 块 GPU 都通过 NVSwitch 互联——这是一种提供<em>全二分带宽（full bisection bandwidth）</em>的专用交换芯片。这意味着任意 GPU 都能以完整 NVLink 速度同时与任何其他 GPU 通信，而不仅仅是环上邻居。</p>

<div class="admonition tip"><p class="admonition-title">环形拓扑 vs. 全二分</p>
<p>在环形拓扑（8 GPU）中，AllReduce 需要数据绕环传输。每条链路必须承载总数据的 <span class="arithmatex">\(\frac{2(N-1)}{N}\)</span>，因此算法带宽为 <span class="arithmatex">\(B_{\text{link}} \times \frac{N}{2(N-1)}\)</span>（<span class="arithmatex">\(N=8\)</span> 时约为 <span class="arithmatex">\(0.57 \times B_{\text{link}}\)</span>）。借助 NVSwitch 的全二分带宽，AllReduce 可以使用基于树的算法同时使用所有链路，达到接近峰值的带宽。在 DGX H100 实际测试中：ring 达到约 700 GB/s 总线带宽，NVSwitch 达到约 900 GB/s。</p>
</div>

### InfiniBand——节点间通信。 {#infiniband-inter-node-communication.}

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

### 胖树拓扑（Fat-Tree Topology）。 {#fat-tree-topology.}

<p>大型 GPU 集群使用胖树（Fat-Tree Topology）网络拓扑。使用 <span class="arithmatex">\(k\)</span> 端口交换机的 3 层胖树可支持 <span class="arithmatex">\(k^3/4\)</span> 个节点并保持全二分带宽。对于 <span class="arithmatex">\(k=64\)</span> 端口的 400Gb/s NDR 交换机：<span class="arithmatex">\(64^3/4 = 65{,}536\)</span> 个节点。</p>

### 轨道优化拓扑（Rail-Optimized Topology）。 {#rail-optimized-topology.}

<p>实践中，集群使用<em>轨道优化拓扑（Rail-Optimized Topology）</em>，节点中的每块 GPU 连接到不同的机架顶端（top-of-rack）交换机。这样可以保证 AllReduce 操作（涉及所有 GPU）同时使用所有网络链路，从而最大化带宽。</p>

### 分布式 LLM 训练中的通信模式。 {#communication-patterns-in-distributed-llm-training.}

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

### 网络拓扑示意图。 {#network-topology-diagram.}

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
