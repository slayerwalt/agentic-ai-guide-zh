# 第 11 章 · 大规模系统架构与基础设施 {#system-architecture-infrastructure-at-scale}

<p>用基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）训练 LLM，既是一道算法题，也是一道系统工程题。与标准 SFT 不同——后者只涉及单一模型、单次前向-反向传播、扩展规律也已被充分理解——RLHF 需要<em>同时</em>加载<em>多个模型</em>（policy、reference、reward model、value head），通过复杂的 rollout-打分-训练循环协调起来，并分布到数十乃至数百块 GPU 上。本章覆盖让大规模 RLHF 训练成为可能的系统级细节：显存预算、并行策略（Data、Tensor、Pipeline、Sequence 及其组合）、生成瓶颈、解耦式架构、权重同步、容错以及生产监控。</p>

## 本章内容

<ul class="chapter-outline">
<li id="the-4-model-memory-challenge"><a href="section-01.md">11.1　4 模型显存挑战</a></li>
<li id="sec:parallelism"><a href="section-02.md">11.2　并行策略详解</a></li>
<li id="the-generation-bottleneck-quantitative-analysis"><a href="section-03.md">11.3　生成瓶颈：定量分析</a></li>
<li id="decoupled-architecture-production-design"><a href="section-04.md">11.4　解耦式架构：生产级设计</a></li>
<li id="weight-synchronization-strategies"><a href="section-05.md">11.5　权重同步策略</a></li>
<li id="memory-optimization-techniques"><a href="section-06.md">11.6　显存优化技术</a></li>
<li id="fault-tolerance-at-scale"><a href="section-07.md">11.7　大规模容错</a></li>
<li id="end-to-end-latency-breakdown"><a href="section-08.md">11.8　端到端延迟拆解</a></li>
<li id="monitoring-and-observability"><a href="section-09.md">11.9　监控与可观测性</a></li>
<li id="sec:network-topology"><a href="section-10.md">11.10　网络拓扑与通信模式</a></li>
<li id="sec:mfu"><a href="section-11.md">11.11　训练吞吐与模型 FLOPs 利用率</a></li>
<li id="cost-analysis-and-cloud-deployment"><a href="section-12.md">11.12　成本分析与云端部署</a></li>
<li id="sec:checkpointing"><a href="section-13.md">11.13　分布式 Checkpointing</a></li>
<li id="sec:hardware-guide"><a href="section-14.md">11.14　硬件选型指南</a></li>
<li id="sec:rl-optimizer-config"><a href="section-15.md">11.15　RL 训练的优化器配置</a></li>
</ul>
