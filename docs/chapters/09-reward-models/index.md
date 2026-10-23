# 第 9 章 · 奖励模型训练 {#sec:reward-models}

<p>奖励模型是连接人类偏好与 RL 训练信号的桥梁。训练良好的奖励模型对成功的 RLHF 至关重要；训练不佳的奖励模型会导致奖励黑客攻击（reward hacking）和行为失准。本章涵盖奖励模型的理论基础、实用训练技术以及架构选择。</p>

## 本章内容

<ul class="chapter-outline">
<li id="sec:bradley-terry"><a href="section-01.md">9.1　Bradley-Terry 模型 —— 完整推导</a></li>
<li id="sec:rm-architectures"><a href="section-02.md">9.2　奖励模型架构</a></li>
<li id="sec:rm-tricks"><a href="section-03.md">9.3　奖励模型训练技巧</a></li>
<li id="sec:prm-orm"><a href="section-04.md">9.4　过程奖励模型 vs 结果奖励模型</a></li>
<li id="sec:rule-based-rewards"><a href="section-05.md">9.5　RLVR 中基于规则的 Reward</a></li>
<li id="sec:multi-objective-rewards"><a href="section-06.md">9.6　多目标 Reward —— 组合策略</a></li>
<li id="listwise-rank-based-rewards"><a href="section-07.md">9.7　基于列表排序的 Reward</a></li>
</ul>
