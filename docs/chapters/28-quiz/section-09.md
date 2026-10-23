# 28.9　Reward 模型与 SFT 题 {#reward-model-and-sft-questions}

<section class="quiz-question" id="q-39"><h2 class="question-title">Q39：推导 Bradley-Terry reward 模型的 loss。它有哪些局限？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>Bradley-Terry 模型（Bradley-Terry Model）</strong>：给定两个回答，更好的那个（<span class="arithmatex">\(y_w\)</span>）被偏好的概率：<span class="arithmatex">\(P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))\)</span>，其中 <span class="arithmatex">\(\sigma\)</span> 是 sigmoid。</p>

<p><strong>MLE 推导</strong>：给定 <span class="arithmatex">\(N\)</span> 个偏好对，最大化似然：<span class="arithmatex">\(\prod_i P(y_w^i \succ y_l^i)\)</span>。取负对数：<span class="arithmatex">\(\mathcal{L} = -\sum_i \log\sigma(r(x_i, y_w^i) - r(x_i, y_l^i))\)</span>。</p>

<p><strong>局限</strong>：</p>

<ol>
<li><p><strong>无平局</strong>：BT 无法建模“同样好”——强制严格偏好。</p></li>
<li><p><strong>可传递性</strong>：假设若 A<span class="arithmatex">\(&gt;\)</span>B 且 B<span class="arithmatex">\(&gt;\)</span>C 则 A<span class="arithmatex">\(&gt;\)</span>C。人类并不可传递。</p></li>
<li><p><strong>上下文无关</strong>：无论备选项是什么，reward 都相同。</p></li>
<li><p><strong>标量塌缩</strong>：把所有质量维度压成一个数。一个回答可能既安全又无用——RM 必须权衡。</p></li>
<li><p><strong>长度偏差</strong>：更长的回答得到更高分（信息更多 = 更可能包含标注者想要的内容）。必须显式去相关。</p></li>
</ol>

<p><strong>缓解措施</strong>：margin loss（要求最小间隔 <span class="arithmatex">\(\delta\)</span>）、reward 中心化（减去滑动均值）、训练时长度惩罚、多头 RM（为有用性/安全性/准确性提供独立分数）。</p>

<p><em><strong>复习：</strong> 第 <a href="../09-reward-models/index.md">9</a> 章（Reward 模型训练）。</em></p>
</details></section>

<section class="quiz-question" id="q-40"><h2 class="question-title">Q40：SFT 中的序列打包（Sequence Packing）是什么？为何重要？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：训练样本长度可变。标准批处理把所有样本填充到 max_length。若 max=4096 而平均=500，就有 88% 的算力浪费在填充 Token 上（梯度贡献为零）。</p>

<p><strong>打包方案</strong>：把多个短样本拼接成一条 max_length 序列，用 EOS Token 分隔。同时对所有样本训练。</p>

<p>例子：不再用 4 条填充到 4096 的序列（16K Token，14K 填充），而是打包成 1 条 4096 的序列，首尾相连地容纳 4 个样本（4096 真实 Token，0 填充）。<strong>效率提升 4<span class="arithmatex">\(\times\)</span>。</strong></p>

<p><strong>关键细节——块对角 attention 掩码</strong>：若不做特殊处理，样本 2 会关注样本 1 的 Token（交叉污染）。必须使用块对角 attention 掩码，限制每个样本只能关注自身的 Token。</p>

<p><strong>在 TRL 中</strong>：<code>SFTConfig(packing=True, max_seq_length=4096)</code>。自动处理掩码。</p>

<p><strong>注意</strong>：(1) 更长的样本仍需自己的 batch 条目（不能在序列中间切分）。(2) 位置 Embedding 实现略复杂（每个样本重置）。(3) 有人认为打包改变了有效 batch size（每步样本更多）——相应调整 LR。</p>

<p><em><strong>复习：</strong> 第 <a href="../10-sft/index.md">10</a> 章（SFT 最佳实践与技巧）。</em></p>
</details></section>

<section class="quiz-question" id="q-41"><h2 class="question-title">Q41：解释 SFT 的 completion-only masking。不使用它会怎样？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：在 chat 格式的 SFT 数据中：<code>[system] + [user message] + [assistant response]</code>。标准 NLL loss 会在所有 Token 上计算 loss，包括 system prompt 和用户消息。</p>

<p><strong>不掩码的问题</strong>：模型浪费容量去预测用户消息（推理时永远不需要生成）。更糟：若训练数据中用户消息多样，模型会困惑“现在轮到谁说话？”</p>

<p><strong>Completion-only masking</strong>：将 prompt（system + user）中所有 Token 的 loss 权重设为 0。仅在 assistant 回答 Token 上计算 loss。</p>

<p>TRL：<code>DataCollatorForCompletionOnlyLM(response_template=&quot;&lt;|assistant|&gt;&quot;)</code></p>

<p><strong>影响</strong>：在指令跟随基准上通常提升 5–15%。收敛更快（梯度信号集中于有用 Token）。算力成本不变。</p>

<p><strong>细节</strong>：response 模板 Token 必须包含在 loss 内（教模型开始回答），但之前的所有内容都要排除。</p>

<p><em><strong>复习：</strong> 第 <a href="../10-sft/index.md">10</a> 章（SFT 最佳实践与技巧）。</em></p>
</details></section>

<section class="quiz-question" id="q-42"><h2 class="question-title">Q42：SFT 质量如何影响 RL 的天花板？pass@k 诊断是什么？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>天花板定理（非正式表述）</strong>：RL 只能强化模型已能以非可忽略概率产出的行为。若 SFT 模型生成正确解的概率为 0%，RL 永远找不到它。</p>

<p><strong>为什么</strong>：GRPO/PPO 从当前 policy 采样并强化好样本。若好样本不在分布内，就没什么可强化。RL 的探索受 base policy 支撑集（support）的限制。</p>

<p><strong>pass@k 诊断</strong>：对每个 prompt 生成 <span class="arithmatex">\(k\)</span> 个回答，检查<em>是否有任一个</em>正确：</p>

<ul>
<li><p>pass@1：模型的典型表现（greedy/低温）。</p></li>
<li><p>pass@8：<span class="arithmatex">\(G=8\)</span> 的 GRPO 所能达到的上界。</p></li>
<li><p>pass@64：激进 Best-of-N 采样（Rejection Sampling）的上界。</p></li>
<li><p>pass@256：RL 改进的近似天花板。</p></li>
</ul>

<p><strong>解读</strong>：</p>

<ul>
<li><p>pass@1=20%、pass@64=80%：好！RL 有 4<span class="arithmatex">\(\times\)</span> 上行空间。预期强劲收益。</p></li>
<li><p>pass@1=20%、pass@64=25%：几乎没有上行空间。RL 帮不上忙。需先做更好的 SFT。</p></li>
<li><p>pass@1=5%、pass@64=60%：模型<em>能</em>解但很少解出来。RL 的完美场景（强化稀有的成功）。</p></li>
</ul>

<p><strong>规则</strong>：若 pass@64 <span class="arithmatex">\(&lt;\)</span> 1.5<span class="arithmatex">\(\times\)</span> pass@1，在开始 RL 之前先投入更好的 SFT 数据。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../10-sft/index.md">10</a> 章（GRPO；SFT 最佳实践）。</em></p>
</details></section>

<section class="quiz-question" id="q-43"><h2 class="question-title">Q43：为聊天模型设计一个多目标 reward 系统。如何平衡有用性与安全性？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>架构</strong>：每个目标使用独立 reward 模型：</p>

<ul>
<li><p><span class="arithmatex">\(r_\text{helpful}\)</span>：在有用性偏好上训练（质量、准确性、完整性）。</p></li>
<li><p><span class="arithmatex">\(r_\text{safe}\)</span>：在安全偏好上训练（拒答、无害、无幻觉）。</p></li>
<li><p><span class="arithmatex">\(r_\text{format}\)</span>：基于规则（遵循指令、合适格式、合理长度）。</p></li>
</ul>

<p><strong>组合策略</strong>：</p>

<ol>
<li><p><strong>加权求和</strong>（最简）：<span class="arithmatex">\(r = w_1 r_\text{helpful} + w_2 r_\text{safe} + w_3 r_\text{format}\)</span>。问题：安全性可能被有用性压过。</p></li>
<li><p><strong>约束式</strong>（更安全）：在 <span class="arithmatex">\(r_\text{safe} &gt; \tau\)</span> 约束下最大化 <span class="arithmatex">\(r_\text{helpful}\)</span>。通过 <span class="arithmatex">\(r = r_\text{helpful} - \lambda \cdot \max(0, \tau - r_\text{safe})\)</span> 实现，<span class="arithmatex">\(\lambda\)</span> 取较大值。</p></li>
<li><p><strong>GDPO 归一化</strong>（对 GRPO 最佳）：在 group 内独立归一化每个 reward，再组合：<span class="arithmatex">\(\hat{A} = w_1 \hat{A}_\text{helpful} + w_2 \hat{A}_\text{safe}\)</span>。防止某个 reward 因尺度差异而主导。</p></li>
<li><p><strong>字典序（Lexicographic）</strong>：安全是硬约束（必须通过），再优化有用性。分阶段训练：先做安全对齐，再做有用性。</p></li>
</ol>

<p><strong>实用权重</strong>：从 <span class="arithmatex">\(w_\text{safe}=2.0, w_\text{helpful}=1.0, w_\text{format}=0.5\)</span> 起步。安全权重 2<span class="arithmatex">\(\times\)</span>，因为它的失败模式（有害内容）远比有用性失败（平庸答案）更糟。</p>

<p><em><strong>复习：</strong> 第 <a href="../09-reward-models/index.md">9</a> 与 <a href="../12-agentic-training/index.md">12</a> 章（Reward 模型训练；LLM 智能体训练）。</em></p>
</details></section>
