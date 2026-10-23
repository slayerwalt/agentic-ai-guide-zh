# 28.1　基础题 {#foundations-questions}

<section class="quiz-question" id="q-0a"><h2 class="question-title">Q0a：在 decoder-only Transformer 中，attention 机制的作用是什么？为什么它是因果的？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：attention 机制允许每个 Token 关注（即对其表示做加权组合）其他 Token 的表示。在 decoder-only Transformer 中，attention 是<strong>因果的</strong>（也称为<em>自回归</em>）：Token <span class="arithmatex">\(t\)</span> 只能关注 Token <span class="arithmatex">\(1, \ldots, t\)</span>，绝不能关注未来 Token <span class="arithmatex">\(t+1, \ldots, T\)</span>。</p>

<p><strong>为什么是因果的？</strong>因为模型从左到右生成文本。推理时，未来 Token 字面意义上还不存在。训练时的因果掩码模拟了这一约束，使模型学会仅用左侧上下文预测每个 Token。在数学上，attention 矩阵被掩盖：</p>



<div class="arithmatex">\[\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V\]</div>

<p>其中对于 <span class="arithmatex">\(j &gt; i\)</span>（未来位置）有 <span class="arithmatex">\(M_{ij} = -\infty\)</span>，强制将这些 attention 权重置零。</p>

<p><strong>实践含义</strong>：这使得推理时的 KV-cache 优化成为可能——由于过去 Token 的 keys 和 values 永远不变，可被缓存复用，将每个新 Token 的生成代价从 <span class="arithmatex">\(O(T^2)\)</span> 降至 <span class="arithmatex">\(O(T)\)</span>。</p>

<p><em><strong>复习：</strong> 第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

<section class="quiz-question" id="q-0b"><h2 class="question-title">Q0b：解释 FlashAttention。它解决什么问题，又是怎么解决的？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：标准 attention 会计算完整的 <span class="arithmatex">\(T \times T\)</span> attention 矩阵，需要 <span class="arithmatex">\(O(T^2)\)</span> 显存，并且是<strong>受显存带宽约束的</strong>——GPU 大部分时间都花在 HBM（慢、大）与 SRAM（快、小）之间搬运数据，而不是做真正的计算。</p>

<p><strong>FlashAttention 的洞见</strong>：永远不要在 HBM 中物化完整的 attention 矩阵。相反，把计算切分成可放进 SRAM 的分块（tile），用在线 Softmax（Online Softmax）算法<em>逐块</em>计算 attention，并只把最终输出写入 HBM。</p>

<p><strong>关键技术</strong>：</p>

<ol>
<li><p><strong>分块（Tiling）</strong>：将 Q、K、V 拆成大小为 <span class="arithmatex">\(B_r \times B_c\)</span>、可装入 SRAM 的块。</p></li>
<li><p><strong>在线 Softmax（Online Softmax）</strong>：维护一个滑动的 max 和 sum，以增量方式计算 softmax，无需访问完整一行。</p></li>
<li><p><strong>重算（Recomputation）</strong>：在反向传播时，从 Q、K、V 重新计算 attention（开销小），而不存储 <span class="arithmatex">\(T \times T\)</span> 矩阵（开销大）。</p></li>
</ol>

<p><strong>结果</strong>：HBM 显存复杂度为 <span class="arithmatex">\(O(T)\)</span>（而非 <span class="arithmatex">\(O(T^2)\)</span>），墙钟（wall-clock）时间加速 2–4<span class="arithmatex">\(\times\)</span>，数值输出完全一致（并非近似）。</p>

<p><em><strong>复习：</strong> 第 <a href="../01-llm-architecture/index.md">1</a>–<a href="../02-systems-foundations/index.md">2</a> 章（LLM 架构；系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-0c"><h2 class="question-title">Q0c：从高层次看，SFT、RLHF 与 DPO 有什么区别？分别在什么场景下使用？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<ul>
<li><p><strong>SFT（监督微调，Supervised Fine-Tuning）</strong>：训练模型去模仿高质量示范。Loss：在精选数据上做下一个 Token 预测。教会模型<em>格式</em>与<em>风格</em>。</p></li>
<li><p><strong>RLHF</strong>：从人类偏好训练一个 reward 模型，然后用 RL（PPO）针对它优化 policy。模型会探索示范数据之外的空间。教会模型<em>人类偏好什么</em>。</p></li>
<li><p><strong>DPO</strong>：跳过 reward 模型。直接在偏好对 <span class="arithmatex">\((y_w, y_l)\)</span> 上用对比 loss 优化 policy。目标与 RLHF 相同，但流水线更简单。</p></li>
</ul>

<p><strong>典型流水线</strong>：先做 SFT（提供良好起点），再做 RLHF 或 DPO（细化偏好）。仅做 SFT 倾向产出冗长、过度模棱两可的回答。RLHF/DPO 让输出更直接、更贴合人类意图。</p>

<p><strong>各自适用场景</strong>：有黄金标准输出 <span class="arithmatex">\(\rightarrow\)</span> SFT。有偏好对但算力有限 <span class="arithmatex">\(\rightarrow\)</span> DPO。需要极致质量且能承担基础设施成本 <span class="arithmatex">\(\rightarrow\)</span> RLHF（PPO）。</p>

<p><em><strong>复习：</strong> 第 <a href="../05-ppo/index.md">5</a>、<a href="../06-dpo/index.md">6</a> 与 <a href="../10-sft/index.md">10</a> 章（PPO；DPO；SFT 最佳实践）。</em></p>
</details></section>

<section class="quiz-question" id="q-0d"><h2 class="question-title">Q0d：什么是 reward 模型？它如何训练，可能出什么问题？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：reward 模型（Reward Model, RM）是一个神经网络，输入 (prompt, response) 对，输出一个表示质量的标量分数。它在人类偏好数据上训练：给定 <span class="arithmatex">\((y_w, y_l)\)</span> 对（其中 <span class="arithmatex">\(y_w\)</span> 被偏好），RM 学习赋予 <span class="arithmatex">\(R(y_w) &gt; R(y_l)\)</span>。</p>

<p><strong>训练</strong>：Bradley-Terry loss：<span class="arithmatex">\(\mathcal{L} = -\log\sigma(R(y_w) - R(y_l))\)</span>。架构：通常与 policy 共用同一个 Transformer，将 LM head 替换为标量投影。</p>

<p><strong>可能出什么问题</strong>：</p>

<ol>
<li><p><strong>Reward hacking</strong>：policy 找到了在 RM 上得分很高但实际质量很差的输出（例如过度冗长、重复，或包含 RM 存在偏好的特定短语）。</p></li>
<li><p><strong>分布漂移（Distribution Shift）</strong>：RM 是在更早的 policy 输出上训练的。随着训练推进，当前 policy 生成的是 RM 无法准确打分的分布外输出。</p></li>
<li><p><strong>标签噪声</strong>：人类标注者意见不一致、疲劳，或使用不一致的标准。这种噪声会传播到 RM 的预测中。</p></li>
<li><p><strong>过度自信</strong>：RM 对从未见过的输出赋予极端分数，提供具有误导性的梯度信号。</p></li>
</ol>

<p><em><strong>复习：</strong> 第 <a href="../09-reward-models/index.md">9</a> 章（Reward 模型训练）。</em></p>
</details></section>

<section class="quiz-question" id="q-0e"><h2 class="question-title">Q0e：解释 RL 中的探索-利用（exploration-exploitation）权衡。它在 LLM 训练中如何体现？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：在 RL 中，Agent 必须在以下两者间取得平衡：</p>

<ul>
<li><p><strong>利用（Exploitation）</strong>：选择已知能产生高 reward 的动作（贪婪行为）。</p></li>
<li><p><strong>探索（Exploration）</strong>：尝试可能带来更高 reward（但也可能失败）的新动作。</p></li>
</ul>

<p><strong>在 LLM 训练中</strong>：policy 就是语言模型。“动作”是 Token 的选择。“利用”指生成与已得高分相似的回答；“探索”指尝试新颖的措辞、结构或推理路径。</p>

<p><strong>表现形式</strong>：</p>

<ul>
<li><p><strong>生成时的温度</strong>：温度越高 = 探索越多。GRPO 使用温度 1.0 以在每个 group 内获得多样的样本。</p></li>
<li><p><strong>KL 惩罚</strong>：作为一种反探索的刹车——防止 policy 偏离参考模型过远。否则 policy 可能会塌缩为单一的高 reward 模板（mode collapse）。</p></li>
<li><p><strong>GRPO 中的 group 采样</strong>：每个 prompt 生成 <span class="arithmatex">\(G\)</span> 个回答以显式探索输出空间，然后强化高于平均水平的回答。</p></li>
</ul>

<p><strong>张力</strong>：探索过少 <span class="arithmatex">\(\rightarrow\)</span> 模型陷入局部最优（永远给同一个安全答案）。探索过多 <span class="arithmatex">\(\rightarrow\)</span> 训练不稳定，质量剧烈波动。</p>

<p><em><strong>复习：</strong> 第 <a href="../03-reinforcement-learning/index.md">3</a> 与 <a href="../07-grpo/index.md">7</a> 章（RL 简介；GRPO）。</em></p>
</details></section>
