# 28.20　大型推理模型的 RL 问题 {#rl-for-large-reasoning-models-questions}

<section class="quiz-question" id="q-source-79"><h2 class="question-title">Q：DeepSeek-R1 在长推理链上训练，却为何不使用过程奖励模型？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：DeepSeek-R1 仅使用<strong>基于结果的 Reward</strong>（准确率 + 格式），原因如下：</p>

<ol>
<li><p><strong>可验证任务</strong>：数学和代码有确定的真值答案。即便是长链，二元的准确率 Reward 也能提供足够信号。</p></li>
<li><p><strong>PRM 失效模式</strong>：步骤级 Reward 模型自身会引入 Reward hacking——模型可学到产出“在 PRM 看来正确”但实际不正确的步骤。</p></li>
<li><p><strong>GRPO 的组归一化</strong>：通过每个 Prompt 采样 <span class="arithmatex">\(G\)</span> 个补全并在组内归一化优势，GRPO 即使没有每步 Reward 也能自然提供关于哪些推理<em>策略</em>有效的相对信号。</p></li>
<li><p><strong>涌现的自我纠正</strong>：在仅有结果 Reward 时，模型学会在链内自我纠正（“aha 时刻”），如果每步 Reward 对推理过程微观管理，这种行为将无法涌现。</p></li>
</ol>

<p><strong>关键洞见</strong>：任务领域的可验证性使 PRM 变得不必要——对于主观任务（创意写作），仅有结果 Reward 可能不够。</p>

<p><em><strong>复习：</strong>第 <a href="../13-reasoning/index.md">13</a> 章（大型推理模型的 RL）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-80"><h2 class="question-title">Q：解释测试时计算扩展律（Test-Time Compute Scaling Law）及其对模型部署的启示</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：测试时计算扩展律表述为：</p>



<div class="arithmatex">\[\text{Accuracy}(C_{\text{train}}, C_{\text{test}}) \approx f(\alpha \log C_{\text{train}} + \beta \log C_{\text{test}})\]</div>

<p><strong>启示</strong>：</p>

<ol>
<li><p><strong>算力等价</strong>：在推理任务上，使用 64<span class="arithmatex">\(\times\)</span> 推理 token 的 7B 模型可以与使用 1<span class="arithmatex">\(\times\)</span> token 的 70B 模型相匹敌。</p></li>
<li><p><strong>自适应分配</strong>：简单问题用短链（便宜）；困难问题用长链（贵）。平均成本低于始终使用大模型。</p></li>
<li><p><strong>部署灵活性</strong>：不再部署一个大模型，而是部署一个较小的推理模型，并按查询难度对推理算力进行扩展。</p></li>
<li><p><strong>收益递减</strong>：对数关系意味着测试时计算翻倍带来的准确率提升递减——训练与推理算力之间存在一个最优分配。</p></li>
</ol>

<p><strong>“过度思考”失败模式</strong>：非常长的链可能因错误累积和 Attention 稀释而<em>降低</em>准确率。最优链长取决于问题难度。</p>

<p><em><strong>复习：</strong>第 <a href="../13-reasoning/index.md">13</a> 章（大型推理模型的 RL）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-81"><h2 class="question-title">Q：MCTS（蒙特卡洛树搜索，Monte Carlo Tree Search）如何应用于 LLM 推理？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：用于推理的 MCTS 将每个部分解视为一个树节点：</p>

<p><strong>每次迭代四个阶段</strong>：</p>

<ol>
<li><p><strong>选择</strong>：用 UCB 从根开始导航：<span class="arithmatex">\(\text{UCB}(s) = Q(s) + c\sqrt{\frac{\ln N(\text{parent})}{N(s)}}\)</span></p></li>
<li><p><strong>扩展</strong>：从 LLM 生成新的推理步（子节点）</p></li>
<li><p><strong>模拟</strong>：从新节点完成解（Rollout）</p></li>
<li><p><strong>反向传播</strong>：根据最终正确性更新路径上的 Q 值</p></li>
</ol>

<p><strong>与博弈 MCTS 的关键差异</strong>：</p>

<ul>
<li><p><strong>分支因子</strong>：推理具有巨大的分支因子（任何下一句都可能）。实际实现使用 LLM 的 top-k 输出来限制分支。</p></li>
<li><p><strong>Value 函数</strong>：训练好的 PRM 估计部分解质量，替代随机 Rollout。</p></li>
<li><p><strong>步骤粒度</strong>：每“步”可能是一句话、一个方程或一次逻辑推断——粒度选择很关键。</p></li>
</ul>

<p><strong>使用案例</strong>：AlphaProof（数学奥林匹克），以及据推测 OpenAI o1/o3 的隐藏推理。</p>

<p><em><strong>复习：</strong>第 <a href="../13-reasoning/index.md">13</a> 章（大型推理模型的 RL）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-82"><h2 class="question-title">Q：比较通过蒸馏 vs 直接 RL 来打造小型推理模型</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>蒸馏</strong>（DeepSeek-R1-Distill 路线）：</p>

<ul>
<li><p>从大模型（R1-671B）生成推理链</p></li>
<li><p>在这些链上对小模型做 SFT</p></li>
<li><p>结果：小模型模仿大模型的推理<em>格式</em></p></li>
<li><p>优点：便宜（仅 SFT）。缺点：可能学到表层模式而非真正推理。</p></li>
</ul>

<p><strong>直接对小模型做 RL</strong>：</p>

<ul>
<li><p>用 GRPO/PPO 在可验证 Reward 上训练小模型</p></li>
<li><p>模型发现自己的推理策略</p></li>
<li><p>优点：真实能力。缺点：算力消耗大得多；对于非常小的模型可能不收敛。</p></li>
</ul>

<p><strong>经验发现</strong>：R1-Distill-7B（蒸馏）在大多数基准上优于 direct-RL-7B。大模型的推理链提供了非常强的监督信号，单 SFT 就具有竞争力。然而，蒸馏模型对真正新颖的问题类型泛化较差。</p>

<p><strong>最佳实践</strong>：先蒸馏（便宜的基线），再可选地在蒸馏模型上跑 RL 以获得进一步收益（Qwen 使用的“蒸馏 + RL”组合）。</p>

<p><em><strong>复习：</strong>第 <a href="../13-reasoning/index.md">13</a> 章（大型推理模型的 RL）。</em></p>
</details></section>
