# 4.3　RLHF 流水线 {#the-rlhf-pipeline}

<p>经典的 RLHF 流水线 <a class="book-citation" id="cite-ouyang2022training-1" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a> 包含四个阶段：</p>

<ol>
<li><p><strong>监督微调（SFT）</strong>：在高质量示例上训练 base 模型，得到一个能够遵循指令的 policy <span class="arithmatex">\(\pi_{\text{SFT}}\)</span>。</p></li>
<li><p><strong>Reward 模型训练</strong>：收集人类偏好比较（对同一 prompt 满足 <span class="arithmatex">\(y_w \succ y_l\)</span>），并使用 Bradley-Terry 模型（Bradley-Terry Model）目标训练 reward 模型 <span class="arithmatex">\(R_\phi(x, y)\)</span>。</p></li>
<li><p><strong>RL 优化</strong>：以 reward 模型作为信号，通过 PPO 或 GRPO 优化 policy，并对 <span class="arithmatex">\(\pi_{\text{SFT}}\)</span> 施加 KL 约束。</p></li>
<li><p><strong>评估与迭代</strong>：评估对齐后的模型，收集新的失败案例，并进行迭代。</p></li>
</ol>

<p>对于 RLVR（推理/agentic 训练），阶段 1–2 被替换：SFT 模型在推理轨迹上训练，reward 模型被替换为验证器（例如检查数学正确性）。阶段 3 不变——使用 PPO 或 GRPO 针对 reward 信号进行优化。</p>

<div class="admonition tip"><p class="admonition-title">LLM RL 与经典 RL 的不同之处</p>
<p>大语言模型场景与经典 RL 在若干重要方面有所不同：</p>

<ul>
<li><p><strong>确定性 transition</strong>：所谓“下一个 state”只是先前 token 的拼接——没有随机环境。</p></li>
<li><p><strong>稀疏 reward</strong>：反馈通常仅在生成结束时给出一次（结果 reward），或在关键步骤给出（过程 reward）。</p></li>
<li><p><strong>巨大的 action 空间</strong>：每一步有 32K–128K 个候选 token，但探索通过温度采样隐式完成。</p></li>
<li><p><strong>KL 锚</strong>：LLM RL 被约束保持在 SFT policy 附近，以防止 reward hacking，代价是探索受限。</p></li>
<li><p><strong>无需 value function</strong>：GRPO 完全去掉了 critic 网络，转而采用 reward 的组相对归一化。</p></li>
</ul>

<p>这些差异解释了为何在大语言模型上 PPO 与 GRPO 占据主导，而非 DQN 式方法。</p>
</div>
