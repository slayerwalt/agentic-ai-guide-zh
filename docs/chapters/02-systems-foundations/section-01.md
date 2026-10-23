# 2.1　GPU 架构——从硅片到 LLM 训练 {#gpu-architecture-from-silicon-to-llm-training}

<p>现代大型语言模型几乎完全依赖 GPU（图形处理器，Graphics Processing Unit）进行训练和服务。理解 GPU 架构对于在并行策略、内存管理、内核优化和基础设施规模等方面做出明智决策至关重要。本节针对 LLM 工作负载，系统地介绍 GPU 硬件相关知识。</p>

## 本节内容

<ul class="chapter-outline">
<li><a href="section-01-01.md">2.1.1　为什么深度学习要用 GPU？</a></li>
<li><a href="section-01-02.md">2.1.2　NVIDIA GPU 微架构代际演进</a></li>
<li><a href="section-01-03.md">2.1.3　LLM 训练与推理常用 GPU</a></li>
<li><a href="section-01-04.md">2.1.4　GPU 内部架构——流式多处理器（SM）</a></li>
<li><a href="section-01-05.md">2.1.5　各代际 GPU 芯片的扩展</a></li>
<li><a href="section-01-06.md">2.1.6　GPU 内存层级与带宽</a></li>
<li><a href="section-01-07.md">2.1.7　算术强度与 Roofline 模型</a></li>
<li><a href="section-01-08.md">2.1.8　Attention 是内存密集的；FFN 是计算密集的</a></li>
<li><a href="section-01-09.md">2.1.9　Tensor Cores</a></li>
<li><a href="section-01-10.md">2.1.10　通信架构——NVLink、InfiniBand 与 PCIe</a></li>
</ul>
