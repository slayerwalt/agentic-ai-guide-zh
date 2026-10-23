# 28.8　优化与训练题 {#optimization-and-training-questions}

<section class="quiz-question" id="q-34"><h2 class="question-title">Q34：解释 Adam 与 AdamW。这一差异对 LLM 为何重要？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>Adam 加 L2 正则化</strong>：<span class="arithmatex">\(\theta_{t+1} = \theta_t - \alpha \cdot (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda\theta_t)\)</span>。权重衰减项 <span class="arithmatex">\(\lambda\theta_t\)</span> <em>位于</em>自适应缩放<em>内部</em>。梯度大（<span class="arithmatex">\(v_t\)</span> 大）的参数<em>衰减更少</em>（除以 <span class="arithmatex">\(\sqrt{v_t}\)</span>）。这不是真正的权重衰减——它与尺度相关。</p>

<p><strong>AdamW（解耦权重衰减）</strong>：<span class="arithmatex">\(\theta_{t+1} = (1 - \alpha\lambda)\theta_t - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)\)</span>。权重衰减在自适应更新<em>之外</em>且<em>之前</em>施加。无论梯度历史如何，每个参数都获得相同比例的衰减。</p>

<p><strong>对 LLM 为何重要</strong>：</p>

<ol>
<li><p>LLM 的参数跨越多个数量级（embedding 层 vs attention vs FFN）。Adam 的耦合 L2 实际上对小梯度参数惩罚得比大梯度参数更多——错误行为。</p></li>
<li><p>解耦 WD 在所有层提供统一正则化，防止某些层无界增长而另一些层过度收缩。</p></li>
<li><p>经验：在长预训练上，AdamW 相比相同有效正则强度的 Adam+L2 给出 2–5% 更好的 perplexity。</p></li>
</ol>

<p><strong>对 RL 而言</strong>：通常使用 <span class="arithmatex">\(\lambda = 0\)</span>（无权重衰减）。由 KL 惩罚提供正则化。但对 SFT：AdamW 配合 <span class="arithmatex">\(\lambda = 0.01\)</span>–<span class="arithmatex">\(0.1\)</span> 是标准做法。</p>

<p><em><strong>复习：</strong> 第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

<section class="quiz-question" id="q-35"><h2 class="question-title">Q35：为什么学习率 warmup 是必要的？不做 warmup 会发生什么？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：Adam 的二阶矩估计 <span class="arithmatex">\(v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2\)</span> 从 <span class="arithmatex">\(v_0 = 0\)</span> 起步。偏置修正 <span class="arithmatex">\(\hat{v}_t = v_t/(1-\beta_2^t)\)</span> 在数学上做了补偿，但在实践中：</p>

<ul>
<li><p>头几步：<span class="arithmatex">\(v_t\)</span> 基于 1–5 个梯度样本。对真实方差的估计极不准确。</p></li>
<li><p>若某参数初期恰好得到小梯度，<span class="arithmatex">\(v_t\)</span> 微小 <span class="arithmatex">\(\rightarrow\)</span> 有效 LR 巨大 <span class="arithmatex">\(\rightarrow\)</span> 灾难性更新。</p></li>
<li><p>偏置修正放大早期更新：第 1 步时 <span class="arithmatex">\(\hat{v}_1 = v_1/(1-0.999) = 1000 \cdot v_1\)</span>。</p></li>
</ul>

<p><strong>无 warmup 时</strong>：头 10–100 步经常出现梯度尖峰，永久性损坏模型。在优化器稳定之前，早期表征就被打乱。</p>

<p><strong>Warmup 的修复</strong>：从 LR <span class="arithmatex">\(\approx 0\)</span> 起步，在 <span class="arithmatex">\(W\)</span> 步（通常为训练量的 3–10%）内线性升至目标值。当 LR 到达满值时，<span class="arithmatex">\(v_t\)</span> 已积累了足够样本以变得准确。</p>

<p><strong>典型设置</strong>：</p>

<ul>
<li><p>预训练：2000 步 warmup（占 200K 步的 <span class="arithmatex">\(\sim\)</span>1%）。</p></li>
<li><p>SFT：100 步 warmup（占 2000 步的 <span class="arithmatex">\(\sim\)</span>5%）。</p></li>
<li><p>RL（PPO/GRPO）：20–50 步 warmup（短，模型经 SFT 已稳定）。</p></li>
</ul>

<p><em><strong>复习：</strong> 第 <a href="../01-llm-architecture/index.md">1</a> 与 <a href="../10-sft/index.md">10</a> 章（LLM 架构；SFT 最佳实践）。</em></p>
</details></section>

<section class="quiz-question" id="q-36"><h2 class="question-title">Q36：比较学习率调度策略。RL 微调你会选哪种？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>余弦衰减（Cosine decay）</strong>：<span class="arithmatex">\(\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})(1 + \cos(\pi t/T))\)</span>。预训练与 SFT 的标准选择。衰减平滑，大部分时间停留在中等 LR。</p>

<p><strong>线性衰减</strong>：<span class="arithmatex">\(\eta_t = \eta_\text{max}(1 - t/T)\)</span>。更简单，短训练下与余弦衰减效果相近。</p>

<p><strong>预热-稳定-衰减（Warmup-Stable-Decay, WSD）</strong>：warmup <span class="arithmatex">\(\rightarrow\)</span> 80% 时间常数 LR <span class="arithmatex">\(\rightarrow\)</span> 最后 20% 快速衰减。预训练的新标准。“稳定”阶段提供一致学习；最终衰减挤出剩余收益。</p>

<p><strong>常数</strong>：无衰减。warmup 之后 <span class="arithmatex">\(\eta_t = \eta_\text{max}\)</span>。</p>

<p><strong>对 RL 微调（PPO/GRPO），我会选择：短 warmup + 常数</strong>。理由：</p>

<ol>
<li><p>RL 训练长度高度不可预测（按胜率停训，而非 epoch）。</p></li>
<li><p>余弦/线性衰减假设你提前知道总步数。</p></li>
<li><p>LR 已经非常低（<span class="arithmatex">\(10^{-6}\)</span>），再衰减会让更新几乎不可见。</p></li>
<li><p>PPO 的自适应 KL 控制器已在调控有效步长。</p></li>
<li><p>如必须衰减：在宽裕预算上做线性衰减，指标停滞时早停。</p></li>
</ol>

<p><em><strong>复习：</strong> 第 <a href="../01-llm-architecture/index.md">1</a> 与 <a href="../05-ppo/index.md">5</a> 章（LLM 架构；PPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-37"><h2 class="question-title">Q37：为何梯度裁剪（Gradient Clipping）对 RL 训练至关重要、对 SFT 却不那么重要？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>SFT</strong>：监督 loss 平滑且行为良好。梯度范数在 batch 间一致（通常 0.1–1.0）。在 1.0 处裁剪很少触发——只是安全网。</p>

<p><strong>RL（PPO/GRPO）</strong>：梯度范数高度多变，因为：</p>

<ol>
<li><p><strong>Reward 方差</strong>：一个 batch 可能全是高 reward 回答，下一个全是低的。advantage <span class="arithmatex">\(\hat{A}\)</span> 剧烈摆动。</p></li>
<li><p><strong>比率爆炸</strong>：若某个稀有 Token 的概率变化很大，<span class="arithmatex">\(r_t = \pi_\text{new}/\pi_\text{old}\)</span> 可能极大 <span class="arithmatex">\(\rightarrow\)</span> 在裁剪生效前就产生大梯度。</p></li>
<li><p><strong>稀疏 reward</strong>：在二元 reward 的 GRPO 中，有些 prompt 全部正确（advantage <span class="arithmatex">\(\approx 0\)</span>），突然一道难题给出极端 advantage。</p></li>
<li><p><strong>KL 项</strong>：当 policy 偏离时，KL 惩罚的梯度可能尖峰。</p></li>
</ol>

<p><strong>无裁剪时</strong>：一个坏 batch 就能产生比正常大 100<span class="arithmatex">\(\times\)</span> 的梯度 <span class="arithmatex">\(\rightarrow\)</span> 一步毁掉模型。无法恢复（所有预训练遭灾难性遗忘）。</p>

<p><strong>典型设置</strong>：<code>max_grad_norm=1.0</code>。一些人在 RL 训练早期用 0.5 以更安全。范数在所有参数上全局计算（非逐层）。</p>

<p><strong>监控</strong>：若裁剪在超过 20% 步触发，你的 LR 可能过高，或 batch size 过小。</p>

<p><em><strong>复习：</strong> 第 <a href="../05-ppo/index.md">5</a> 与 <a href="../07-grpo/index.md">7</a> 章（PPO；GRPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-38"><h2 class="question-title">Q38：训练中 BF16 vs FP16。这个选择何时重要？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>FP16</strong>：1 符号 + 5 指数 + 10 尾数位。范围：<span class="arithmatex">\(\pm 65504\)</span>。精度：<span class="arithmatex">\(\sim\)</span>3.3 十进制位。</p>

<p><strong>BF16</strong>：1 符号 + 8 指数 + 7 尾数位。范围：<span class="arithmatex">\(\pm 3.4 \times 10^{38}\)</span>（与 FP32 相同！）。精度：<span class="arithmatex">\(\sim\)</span>2.4 十进制位。</p>

<p><strong>为什么 BF16 在 LLM 上胜出</strong>：</p>

<ol>
<li><p><strong>无需 loss scaling</strong>：FP16 范围窄（<span class="arithmatex">\(\pm\)</span>65K）导致梯度和激活频繁溢出/下溢。需要动态 loss scaling（loss 乘 1024，梯度再除回去）。BF16 拥有 FP32 的范围——溢出基本不可能。</p></li>
<li><p><strong>代码更简单</strong>：无 loss scaler，无 inf/nan 检查，无动态 scaling 调整。</p></li>
<li><p><strong>对 RL 关键</strong>：RL 梯度比 SFT 更嘈杂、更易尖峰。FP16 的 loss scaling 经常失败（选错尺度，产生 NaN）。BF16 “开箱即用”。</p></li>
</ol>

<p><strong>FP16 可能更好的场景</strong>：若你需要极致精度（某些科学计算任务）并能管理好 loss scaling。FP16 多 3 个尾数位 = 结果略更精确。</p>

<p><strong>FP32 主权重</strong>：即便前向/反向使用 BF16，也要在 FP32 中累加梯度更新，以防舍入误差在数千次小步中累积。所有 LLM 训练的标准做法。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>
