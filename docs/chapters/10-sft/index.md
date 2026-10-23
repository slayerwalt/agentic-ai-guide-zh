# 第 10 章 · SFT 最佳实践与技巧 {#sec:sft-best-practices}

<p>监督微调（Supervised Fine-Tuning, SFT）是 RLHF 流水线的基础。SFT 模型的质量决定了 RL 能达到的上限：RL 可以精炼和提升某种行为，但无法可靠地引入 SFT 模型中完全不存在的行为。本章涵盖高效 SFT 的关键技术。</p>

## 本章内容

<ul class="chapter-outline">
<li id="sec:sequence-packing"><a href="section-01.md">10.1　序列打包以提升效率</a></li>
<li id="sec:chat-templates"><a href="section-02.md">10.2　Chat 模板与格式化</a></li>
<li id="sec:completion-masking"><a href="section-03.md">10.3　仅对补全部分进行 mask</a></li>
<li id="sec:data-mixing"><a href="section-04.md">10.4　多任务 SFT 的数据混合策略</a></li>
<li id="sec:sft-pitfalls"><a href="section-05.md">10.5　当 SFT 反而有害 —— 灾难性遗忘与对齐税</a></li>
<li id="sec:sft-rl-connection"><a href="section-06.md">10.6　与 RL 的关系 —— SFT 质量决定 RL 上限</a></li>
</ul>
