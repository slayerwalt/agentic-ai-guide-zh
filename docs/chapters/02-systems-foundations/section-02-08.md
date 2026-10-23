# 2.2.8　架构总览 {#architecture-overview}

<figure id="figure-2-4"><a href="../../assets/figures/fig_019_fig19.png" target="_blank" rel="noopener" aria-label="查看图 2.4 原图" title="点击查看原图"><img src="../../assets/figures/fig_019_fig19.png" alt="vLLM 架构：请求自上而下流动。Scheduler 管理准入与抢占，Block Manager 负责虚拟到物理 KV cache 映射（像操作系统页表一样），Model Executor 在 GPU HBM 中从预分配的块池读取数据并执行批量推理。" loading="lazy"></a><figcaption><span class="figure-number">图 2.4　</span>vLLM 架构：请求自上而下流动。Scheduler 管理准入与抢占，Block Manager 负责虚拟到物理 KV cache 映射（像操作系统页表一样），Model Executor 在 GPU HBM 中从预分配的块池读取数据并执行批量推理。</figcaption></figure>
