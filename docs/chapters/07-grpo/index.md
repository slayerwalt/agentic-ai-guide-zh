# 第 7 章 · GRPO —— 组相对策略优化 {#grpo-group-relative-policy-optimization}

<p>组相对策略优化（Group Relative Policy Optimization, GRPO） <a class="book-citation" id="cite-shao2024deepseekmath-1" href="../../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a> 是一种专为语言模型设计的强化学习算法，它消除了对独立 value network（critic）的需求。GRPO 由 DeepSeek 在 DeepSeekMath 工作中提出，随后在 DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 中被扩展到更大规模，已经迅速成为 LLM 训练中的主流 RL 方法——被大多数开源对齐框架（TRL、OpenRLHF、veRL）作为默认算法采用。</p>

<p>其核心思想看似简单却出人意料地有效：与其训练一个神经网络去预测期望 reward（即 PPO 中的 critic），GRPO 通过对同一 prompt 生成多条响应、利用该组的 reward 统计量作为 baseline，从经验上 <em>估计</em> 这一基线。这从显存中去掉了一整个模型，将工程复杂度减半，而且——令人意外的是——往往优于 PPO，因为经验基线比训练不充分的 value function 更准确。</p>

<p>GRPO 在下列场景中尤其有效：</p>

<ul>
<li><p><strong>推理任务</strong>：具有可验证 reward（数学、代码），二值正确性提供干净信号。</p></li>
<li><p><strong>大模型</strong>（70B 及以上）：去掉 critic 带来的显存节省至关重要。</p></li>
<li><p><strong>多轮对话（Multi-Turn）与 Agent 场景</strong>：跨工具调用（Tool Calling）的价值估计本就难以处理。</p></li>
</ul>

<p>本章涵盖 GRPO 的动机、算法、关键变体（Dr. GRPO、DAPO、2-GRPO、GDPO）以及基于 TRL 的实战实现。</p>

## 本章内容

<ul class="chapter-outline">
<li id="motivation-dup2"><a href="section-01.md">7.1　动机</a></li>
<li id="algorithm"><a href="section-02.md">7.2　算法</a></li>
<li id="trl-implementation-dup3"><a href="section-03.md">7.3　TRL 实现</a></li>
<li id="group-size-analysis"><a href="section-04.md">7.4　组大小分析</a></li>
<li id="grpo-variants-and-extensions"><a href="section-05.md">7.5　GRPO 变体与扩展</a></li>
</ul>
