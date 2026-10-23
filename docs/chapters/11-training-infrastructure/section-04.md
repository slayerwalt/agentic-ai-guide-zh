# 11.4　解耦式架构：生产级设计 {#decoupled-architecture-production-design}

<p>诸如 DeepSpeed-Chat <a class="book-citation" id="cite-yao2023deepspeedchat-1" href="../../references.md#ref-yao2023deepspeedchat" aria-label="参考文献 410">[410]</a> 与 OpenRLHF <a class="book-citation" id="cite-hu2024openrlhf-1" href="../../references.md#ref-hu2024openrlhf" aria-label="参考文献 148">[148]</a> 这样的生产级 RLHF 系统采用<strong>解耦式架构</strong>，将生成、打分、训练拆成三个可独立扩展的集群。</p>

<figure id="figure-11-10"><a href="../../assets/figures/fig_040_fig40.png" target="_blank" rel="noopener" aria-label="查看图 11.10 原图" title="点击查看原图"><img src="../../assets/figures/fig_040_fig40.png" alt="解耦式 RLHF 架构。每个集群针对自己的负载做优化。打分后的 rollout 先在经验缓冲（experience buffer）中累积，再被训练消费。" loading="lazy"></a><figcaption><span class="figure-number">图 11.10　</span>解耦式 RLHF 架构。每个集群针对自己的负载做优化。打分后的 rollout 先在经验缓冲（experience buffer）中累积，再被训练消费。</figcaption></figure>

<div class="admonition note"><p class="admonition-title">为什么要解耦？</p>
<p><strong>生成</strong>受显存带宽约束（需要高速 HBM，浪费算力）。<br><strong>训练</strong>受算力约束（需要 tensor core，反向时浪费带宽）。<br><strong>同一硬件无法同时优化两者</strong>：若把所有事情放在一起，要么生成时浪费算力，要么训练时浪费带宽。解耦让每个集群用最优的硬件/配置。</p>

<p><strong>实际好处</strong>：</p>

<ul>
<li><p>生成和训练可以独立扩展</p></li>
<li><p>生成集群无状态 <span class="arithmatex">\(\rightarrow\)</span> 容错极简</p></li>
<li><p>gen(step <span class="arithmatex">\(N+1\)</span>) 与 train(step <span class="arithmatex">\(N\)</span>) 可重叠 <span class="arithmatex">\(\rightarrow\)</span> 30–40% 加速</p></li>
<li><p>不同精度：生成用 INT8（带宽优先），训练用 BF16（精度优先）</p></li>
</ul>
</div>
