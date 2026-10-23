# 11.10　网络拓扑与通信模式 {#sec:network-topology}

<p>高效的分布式训练要求理解连接 GPU 的分层通信网络。现代集群采用两级架构：超高速的节点内链路，以及较慢但可扩展的节点间网络。</p>

## 11.10.1　节点内：NVLink 与 NVSwitch {#intra-node-nvlink-and-nvswitch}

<p class="table-caption" id="table-11-5"><span class="table-number">表 11.5　</span>NVLink 各代及其对 LLM 训练的影响</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>代际</strong></th><th scope="col"><strong>每链路带宽</strong></th><th scope="col"><strong>每 GPU 链路数</strong></th><th scope="col"><strong>总带宽</strong></th><th scope="col"><strong>平台</strong></th></tr></thead>
<tbody><tr><td>NVLink 3.0</td><td>50 GB/s</td><td>12</td><td>600 GB/s</td><td>A100（DGX A100）</td></tr>
<tr><td>NVLink 4.0</td><td>50 GB/s</td><td>18</td><td>900 GB/s</td><td>H100（DGX H100）</td></tr>
<tr><td>NVLink 5.0</td><td>100 GB/s</td><td>18</td><td>1800 GB/s</td><td>B200（DGX B200）</td></tr></tbody>
</table>
</div>

<p>在单个节点内（通常 8 块 GPU），<strong>NVSwitch</strong> 在任意 GPU 对之间提供全二分带宽（full-bisection bandwidth）。这意味着任意 GPU 可以同时以全 NVLink 速率与任意其他 GPU 通信——这对张量并行至关重要，因为每层都需要在 8 块 GPU 间做 AllReduce。</p>

<div class="admonition note"><p class="admonition-title">NVSwitch vs PCIe 拓扑</p>
<p><strong>有 NVSwitch</strong>（DGX/HGX）：8 块 GPU 两两全互联，速率 600–1800 GB/s。TP 的 AllReduce 每层 <span class="arithmatex">\(\sim\)</span>0.2ms。</p>

<p><strong>无 NVSwitch</strong>（仅 PCIe 的服务器）：GPU 经由 CPU 的 PCIe root complex 通信，带宽仅 32–64 GB/s。8 卡 TP 慢上 10–30<span class="arithmatex">\(\times\)</span>。<strong>在仅 PCIe 的系统上绝不要用 TP<span class="arithmatex">\(&gt;\)</span>2。</strong></p>
</div>

## 11.10.2　节点间：InfiniBand 与 RoCE {#inter-node-infiniband-and-roce}

<p>跨节点做 FSDP/ZeRO-3 的 AllGather 与 ReduceScatter 时，节点间网络成为主要瓶颈。</p>

<p class="table-caption" id="table-11-6"><span class="table-number">表 11.6　</span>LLM 训练集群的节点间网络选项</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>技术</strong></th><th scope="col"><strong>带宽</strong></th><th scope="col"><strong>延迟</strong></th><th scope="col"><strong>说明</strong></th></tr></thead>
<tbody><tr><td>InfiniBand NDR</td><td>400 Gb/s（50 GB/s）</td><td>1–2 <span class="arithmatex">\(\mu\)</span>s</td><td>黄金标准，RDMA，无损</td></tr>
<tr><td>InfiniBand NDR（双轨）</td><td>800 Gb/s（100 GB/s）</td><td>1–2 <span class="arithmatex">\(\mu\)</span>s</td><td>用于 H100 集群</td></tr>
<tr><td>RoCE v2</td><td>100–400 Gb/s</td><td>2–5 <span class="arithmatex">\(\mu\)</span>s</td><td>更便宜，需要调优 PFC/ECN</td></tr>
<tr><td>Ethernet（TCP）</td><td>100–400 Gb/s</td><td>10–50 <span class="arithmatex">\(\mu\)</span>s</td><td>不适合 <span class="arithmatex">\(&gt;\)</span>16 GPU 训练</td></tr></tbody>
</table>
</div>

## 11.10.3　通信原语及其代价 {#communication-primitives-and-their-costs}

<p>理解每种集合通信原语的使用场景有助于诊断瓶颈：</p>

<p class="table-caption" id="table-11-7"><span class="table-number">表 11.7　</span>分布式 LLM 训练中的 NCCL 集合通信操作</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>集合通信</strong></th><th scope="col"><strong>搬运数据量</strong></th><th scope="col"><strong>被谁使用</strong></th><th scope="col"><strong>使用时机</strong></th></tr></thead>
<tbody><tr><td>AllReduce</td><td><span class="arithmatex">\(2 \cdot \frac{N-1}{N} \cdot M\)</span></td><td>TP、DP</td><td>跨 GPU 求和梯度或激活</td></tr>
<tr><td>AllGather</td><td><span class="arithmatex">\(\frac{N-1}{N} \cdot M\)</span></td><td>FSDP 前向</td><td>矩阵乘前重建完整参数张量</td></tr>
<tr><td>ReduceScatter</td><td><span class="arithmatex">\(\frac{N-1}{N} \cdot M\)</span></td><td>FSDP 反向</td><td>反向后分发梯度分片</td></tr>
<tr><td>Broadcast</td><td><span class="arithmatex">\(M\)</span></td><td>PP</td><td>把激活发到下一个流水线 stage</td></tr>
<tr><td>Send/Recv</td><td><span class="arithmatex">\(M\)</span></td><td>PP</td><td>相邻 stage 间的点对点</td></tr></tbody>
</table>
</div>

<p>其中 <span class="arithmatex">\(M\)</span> 是消息大小（字节），<span class="arithmatex">\(N\)</span> 是参与者数量。</p>

<div class="admonition tip"><p class="admonition-title">通信-计算重叠</p>
<p>现代框架（FSDP、DeepSpeed）会积极地让通信与计算重叠：</p>

<p><strong>前向传播</strong>：当第 <span class="arithmatex">\(i\)</span> 层在计算时，AllGather 预取第 <span class="arithmatex">\(i+1\)</span> 层的参数。第 <span class="arithmatex">\(i\)</span> 层一结束，其参数立刻被丢弃（“free-after-forward”）。</p>

<p><strong>反向传播</strong>：当第 <span class="arithmatex">\(i\)</span> 层在算梯度时，ReduceScatter 发送第 <span class="arithmatex">\(i+1\)</span> 层的梯度。调优得当时，这种重叠可隐藏 70–90% 的通信延迟。</p>

<p><strong>调优旋钮</strong>：<code>prefetch_factor</code>（提前预取多少层）、<code>reduce_bucket_size</code>（梯度归约的粒度）、<code>backward_prefetch</code>（反向预取的 “pre” 还是 “post” 策略）。</p>
</div>

## 11.10.4　网络拓扑设计 {#network-topology-design}

<p>生产集群采用 <strong>fat-tree</strong> 或 <strong>rail-optimized</strong>（按轨优化）拓扑：</p>

<ul>
<li><p><strong>Fat-tree</strong>：每一级都具备全二分带宽。任一节点都能以满速与任一其他节点通信。代价昂贵（需要大量交换机），但灵活性最高。</p></li>
<li><p><strong>Rail-optimized</strong>：每个节点的 GPU <span class="arithmatex">\(i\)</span> 连接到同一台叶子交换机（“轨 <span class="arithmatex">\(i\)</span>”）。轨内 AllReduce 便宜，跨轨流量昂贵。Meta 的 RSC 和 Google 的 TPU pod 都采用此拓扑。</p></li>
<li><p><strong>3D torus / Dragonfly</strong>：用于 HPC 集群（Frontier、Aurora）。拓扑感知的作业放置至关重要。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">作业放置很重要</p>
<p>在 512 卡集群上，随机分配节点会因网络拥塞导致 2–3<span class="arithmatex">\(\times\)</span> 减速。<strong>始终请求连续的节点块。</strong>生产调度器（Slurm、Kubernetes）应当强制本地性：一个训练作业的所有节点应处于同一叶子交换机下，或彼此距离不超过一跳。</p>
</div>
