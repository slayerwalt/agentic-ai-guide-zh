# 11.7　大规模容错 {#fault-tolerance-at-scale}

<div class="admonition warning"><p class="admonition-title">硬件故障现实</p>
<p><strong>单卡 MTBF</strong>：约 10,000 小时。<br><strong>512 卡集群 MTBF</strong>：<span class="arithmatex">\(10000/512 \approx 20\)</span> 小时。算上软件/网络故障，<strong>实际 4–8 小时</strong>。<br><strong>多日训练</strong>：会遇到 5–15 次故障。没有容错机制，一次故障就会让整次训练报废。</p>
</div>

<p><strong>生产级容错栈</strong>：</p>

<ol>
<li><p><strong>探测</strong>：NCCL 超时（60s）、GPU 心跳（10s）、NVML 健康监控、ECC 错误计数。</p></li>
<li><p><strong>Checkpoint</strong>：每 50–100 步异步保存。非阻塞（后台线程）。保存内容：模型权重、优化器状态（Adam m/v）、调度器状态、RNG 状态、KL 系数、replay buffer。保留最近 3 个 checkpoint。70B 用时约 30s（并行写入 NVMe）。</p></li>
<li><p><strong>恢复</strong>：(a) 生成集群无状态，直接重启并加载最新权重。(b) 训练集群：加载 checkpoint，剔除故障节点重建 NCCL 进程组，重新分配 FSDP 分片，从最近 checkpoint 续训。</p></li>
<li><p><strong>弹性训练</strong>：Torch Elastic / Kubernetes 自动扩缩。几分钟内替换故障节点。训练以 <span class="arithmatex">\(N-1\)</span> 块 GPU 暂时继续。</p></li>
<li><p><strong>预防</strong>：训练前做 GPU 健康预筛（跑 GEMM 压测）。热备机待命。双轨 InfiniBand 等冗余网络通路。</p></li>
</ol>
