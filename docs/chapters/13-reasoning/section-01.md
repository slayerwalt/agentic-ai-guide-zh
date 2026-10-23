# 13.1　动机与背景 {#subsec:reasoning_motivation}

## 13.1.1　为什么推理需要不同的强化学习方法 {#why-reasoning-requires-different-rl-approaches}

<p>标准的基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）（见第<a href="../04-rl-for-language-models/section-03.md#the-rlhf-pipeline">4.3</a>节）针对一段完整回答优化一个标量奖励。对于需要多步推理的任务——数学、形式化验证、竞赛编程、科学推导——这种表述在以下几个方面不够充分：</p>

<ul>
<li><p><strong>稀疏奖励</strong>：一道数学题可能需要 20 个中间步骤；单一的结果奖励无法为导致错误的中间步骤提供梯度信号。</p></li>
<li><p><strong>长时序</strong>：推理链可能跨越数百到数千个 Token，造成严重的信用分配问题。</p></li>
<li><p><strong>组合搜索</strong>：有效推理路径的空间呈指数级膨胀；模型必须学会高效搜索这一空间。</p></li>
<li><p><strong>可验证性</strong>：与主观的文本质量不同，数学与逻辑的正确性是可客观验证的，从而无需人工标注即可自动计算奖励。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">关键洞见：推理即搜索问题</p>
<p>多步推理可以被刻画为在部分解构成的树上进行的<strong>搜索问题</strong>。树中每个节点是一个推理状态（思维链的前缀），每条边是一个推理步骤（一个 Token 或一个句子），叶子节点是最终答案。面向推理的强化学习教会模型高效地在这棵树中导航——探索有前途的分支、从死胡同回溯，并把算力分配到最重要的地方。</p>
</div>

## 13.1.2　思维链：涌现行为 vs. 训练得到的能力 {#chain-of-thought-emergent-behavior-vs.-trained-capability}

<p>思维链（Chain-of-Thought，CoT）推理最初是在足够大的语言模型中被观察到的一种<em>涌现</em>能力 <a class="book-citation" id="cite-wei2022chain-1" href="../../references.md#ref-wei2022chain" aria-label="参考文献 374">[374]</a>：当用逐步示例进行 Prompt 时，大型模型（通常 <span class="arithmatex">\(\geq\)</span> 100B 参数）会自发地产生中间推理步骤，并由此提升准确率。这引出了一个基本问题：CoT 是规模带来的涌现性质，还是可以被显式训练？</p>

<p>正如 DeepSeek-R1 及相关工作所表明的，答案是<strong>两者皆是</strong>——但有若干重要的微妙之处：</p>

<ul>
<li><p><strong>涌现的 CoT</strong> 源于上下文学习并需要大型基础模型。它脆弱、对 Prompt 敏感，且泛化不稳定。</p></li>
<li><p>通过 RL <strong>训练得到的 CoT</strong> 让模型<em>内在地</em>把生成推理链作为生成过程的一部分，与提示风格无关。这些链条更长、更具探索性，并表现出质上不同的行为（自我纠正、回溯、验证）。</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">“顿悟时刻”现象（DeepSeek-AI et al. 2025）</p>
<p>在对推理模型进行 RL 训练时，DeepSeek 的研究者观察到一个引人注目的涌现行为：训练到某一阶段，模型开始在推理链中途自发地<em>重新审视</em>自己最初的思路，使用诸如“Wait, let me reconsider…”或“Actually, I think I made an error…”这样的措辞。这种自我纠正行为——<em>并未</em>被显式训练——纯粹由“最大化最终答案准确率”的 RL 目标涌现而出。它表明 RL 能够发现对求解难题工具性有用的元认知策略。</p>
</div>

## 13.1.3　测试时计算的扩展律 {#test-time-compute-scaling-laws}

<p>推动推理模型研究的一个核心经验发现是：<strong>测试时算力与性能之间存在可预测的扩展关系</strong>。记 <span class="arithmatex">\(C_{\text{train}}\)</span> 为训练算力（FLOPs），<span class="arithmatex">\(C_{\text{test}}\)</span> 为推理算力（生成的 Token 数）。关键观察是：</p>



<div class="arithmatex">\[\text{Accuracy}(C_{\text{train}}, C_{\text{test}}) \approx f\!\left(\alpha \log C_{\text{train}} + \beta \log C_{\text{test}}\right)
\tag{13.1}\]</div>

<p>其中 <span class="arithmatex">\(f\)</span> 是某个单调函数，<span class="arithmatex">\(\alpha, \beta &gt; 0\)</span> 为常数。这意味着：使用更多推理算力的较小模型可以匹敌使用较少推理算力的较大模型——这是算力与性能权衡的根本性转变。</p>

<figure id="fig:test_time_scaling"><a href="../../assets/figures/fig_044_test_time_scaling.png" target="_blank" rel="noopener" aria-label="查看图 13.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_044_test_time_scaling.png" alt="测试时算力扩展曲线示意图。在各种模型规模下，性能随推理 Token 数对数线性提升；较小模型在更多算力下可以逼近较大模型在更少算力下的表现。" loading="lazy"></a><figcaption><span class="figure-number">图 13.1　</span>测试时算力扩展曲线示意图。在各种模型规模下，性能随推理 Token 数对数线性提升；较小模型在更多算力下可以逼近较大模型在更少算力下的表现。</figcaption></figure>

<p>其实践含义深远：<strong>推理模型用推理算力换取训练算力</strong>。不必总是部署尽可能大的模型，而是可以部署一个具备推理能力的较小模型，并在难题上分配更多 Token 用于“思考”。</p>
