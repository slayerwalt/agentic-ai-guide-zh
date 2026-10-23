# 4.4　本部分路线图 {#what-this-part-covers}

<p>后续章节将构建出完整的“面向大语言模型的 RL”工具箱：</p>

<ol>
<li><p><strong>PPO</strong>（第 5 章）——clipped surrogate 目标、用于优势估计的 GAE、critic 网络，以及完整的 RLHF 训练循环。它是 GPT-4 与 Claude 背后的主力。</p></li>
<li><p><strong>DPO</strong>（第 6 章）——通过将偏好转化为对比式有监督 loss，从而完全绕开 RL。比 online RL 更简单，但灵活性较低。</p></li>
<li><p><strong>GRPO</strong>（第 7 章）——DeepSeek 提出的无 critic 算法，使用组级 reward 归一化。它是 DeepSeek-R1 背后的方法，也是推理模型训练的主导选择。</p></li>
<li><p><strong>偏好优化变体</strong>（第 8 章）——Online DPO、KTO、Best-of-N 采样（Rejection Sampling），以及方法选型指南。</p></li>
<li><p><strong>Reward 建模</strong>（第 9 章）——Bradley-Terry 模型、过程 reward 与结果 reward 的对比、面向 RLVR 的基于规则的 reward，以及多目标组合。</p></li>
<li><p><strong>SFT 最佳实践</strong>（第 10 章）——序列打包（Sequence Packing）、对话模板、数据混合，以及 SFT 质量如何决定 RL 的上限。</p></li>
<li><p><strong>系统工程</strong>（第 11 章）——大规模分布式训练：并行策略、生成与训练解耦，以及面向数百块 GPU 的基础设施。</p></li>
</ol>
