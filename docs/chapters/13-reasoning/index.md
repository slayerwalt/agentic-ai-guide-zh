# 第 13 章 · 面向大型推理模型的强化学习 {#sec:rl_reasoning}

<p>大型推理模型的出现是现代 AI 最重要的进展之一。与优化下一个 Token 预测的标准语言模型训练不同，面向推理的强化学习教会模型在<em>回答之前先思考</em>——在推理时分配额外算力以探索、验证并精炼中间步骤。本节系统地介绍支撑这一范式的方法、架构与扩展律。</p>

## 本章内容

<ul class="chapter-outline">
<li id="subsec:reasoning_motivation"><a href="section-01.md">13.1　动机与背景</a></li>
<li id="sec:test_time_methods"><a href="section-02.md">13.2　测试时扩展方法</a></li>
<li id="subsec:deepseek_r1"><a href="section-03.md">13.3　DeepSeek-R1</a></li>
<li id="subsec:openai_o1"><a href="section-04.md">13.4　OpenAI o1/o3 系列</a></li>
<li id="subsec:qwq_qwen"><a href="section-05.md">13.5　QwQ 与 Qwen 推理模型</a></li>
<li id="subsec:reasoning_methods"><a href="section-06.md">13.6　关键方法及其数学基础</a></li>
<li id="subsec:reasoning_scaling"><a href="section-07.md">13.7　推理的扩展律</a></li>
<li id="subsec:reasoning_comparison"><a href="section-08.md">13.8　推理模型比较</a></li>
<li id="sec:inference-time-scaling"><a href="section-09.md">13.9　推理时算力扩展定律</a></li>
<li id="sec:on-policy-self-distillation"><a href="section-10.md">13.10　面向推理的同策略自蒸馏</a></li>
<li id="subsec:reasoning_summary"><a href="section-11.md">13.11　总结与开放问题</a></li>
</ul>
