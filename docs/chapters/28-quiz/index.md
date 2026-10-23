# 第 28 章 · 测验题与详细解答 {#quiz-questions-detailed-answers}

<p>本章提供一整套题目，用于检验和巩固你对本书所讲内容的理解。每道题目都瞄准一个关键概念、算法或系统设计决策——正是这种知识把表面熟悉与真正的专长区分开来。请将这些题目用于自我检验：先尝试自己作答，再阅读详细解答。题目从基础概念（LLM 架构、强化学习基础）逐步推进到核心算法（PPO、DPO、GRPO），最终覆盖高级系统设计与智能体 AI 主题。</p>

<ul class="chapter-outline" aria-label="本章阅读">
<li><a href="#foundations-questions">28.1–28.6 · 基础、算法与实战</a></li>
<li><a href="reading-02.md#gpu-architecture-and-hardware-questions">28.7–28.10 · 硬件、优化与训练系统</a></li>
<li><a href="reading-03.md#transformer-architecture-questions">28.11–28.17 · 模型架构与效率</a></li>
<li><a href="reading-04.md#agentic-rl-questions">28.18–28.21 · Agent 训练、推理与评估</a></li>
<li><a href="reading-05.md#agentic-memory-questions">28.22–28.30 · Agent 系统与应用</a></li>
</ul>

## 28.1　基础题 {#foundations-questions}

<section class="quiz-question" id="q-0a"><h3 class="question-title">Q0a：在 decoder-only Transformer 中，attention 机制的作用是什么？为什么它是因果的？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：attention 机制允许每个 Token 关注（即对其表示做加权组合）其他 Token 的表示。在 decoder-only Transformer 中，attention 是<strong>因果的</strong>（也称为<em>自回归</em>）：Token <span class="arithmatex">\(t\)</span> 只能关注 Token <span class="arithmatex">\(1, \ldots, t\)</span>，绝不能关注未来 Token <span class="arithmatex">\(t+1, \ldots, T\)</span>。</p>

<p><strong>为什么是因果的？</strong>因为模型从左到右生成文本。推理时，未来 Token 字面意义上还不存在。训练时的因果掩码模拟了这一约束，使模型学会仅用左侧上下文预测每个 Token。在数学上，attention 矩阵被掩盖：</p>



<div class="arithmatex">\[\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V\]</div>

<p>其中对于 <span class="arithmatex">\(j &gt; i\)</span>（未来位置）有 <span class="arithmatex">\(M_{ij} = -\infty\)</span>，强制将这些 attention 权重置零。</p>

<p><strong>实践含义</strong>：这使得推理时的 KV-cache 优化成为可能——由于过去 Token 的 keys 和 values 永远不变，可被缓存复用，将每个新 Token 的生成代价从 <span class="arithmatex">\(O(T^2)\)</span> 降至 <span class="arithmatex">\(O(T)\)</span>。</p>

<p><em><strong>复习：</strong> 第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

<section class="quiz-question" id="q-0b"><h3 class="question-title">Q0b：解释 FlashAttention。它解决什么问题，又是怎么解决的？</h3>
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

<section class="quiz-question" id="q-0c"><h3 class="question-title">Q0c：从高层次看，SFT、RLHF 与 DPO 有什么区别？分别在什么场景下使用？</h3>
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

<section class="quiz-question" id="q-0d"><h3 class="question-title">Q0d：什么是 reward 模型？它如何训练，可能出什么问题？</h3>
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

<section class="quiz-question" id="q-0e"><h3 class="question-title">Q0e：解释 RL 中的探索-利用（exploration-exploitation）权衡。它在 LLM 训练中如何体现？</h3>
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

## 28.2　核心算法题 {#core-algorithm-questions}

<section class="quiz-question" id="q-1"><h3 class="question-title">Q1：解释 PPO 的裁剪目标。为什么它比朴素 policy gradient 更有效？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：朴素 policy gradient：<span class="arithmatex">\(\nabla J = \mathbb{E}[\nabla\log\pi(a|s) \cdot \hat{A}]\)</span>。问题：一个幸运/不幸的样本就能产生巨大的梯度 <span class="arithmatex">\(\rightarrow\)</span> policy 跳到糟糕区域 <span class="arithmatex">\(\rightarrow\)</span> 生成乱码 <span class="arithmatex">\(\rightarrow\)</span> 下一个梯度让情况更糟 <span class="arithmatex">\(\rightarrow\)</span> 进入无法挽回的“死亡螺旋”。</p>

<p><strong>PPO 的方案</strong>：将概率比 <span class="arithmatex">\(r = \pi_\text{new}/\pi_\text{old}\)</span> 裁剪到 <span class="arithmatex">\([0.8, 1.2]\)</span>。</p>

<p><strong>机制</strong>：对好的动作（<span class="arithmatex">\(\hat{A}&gt;0\)</span>）：目标为 <span class="arithmatex">\(\min(r\hat{A}, 1.2\hat{A})\)</span>。一旦 <span class="arithmatex">\(r\)</span> 超过 1.2，便不再有额外收益——防止 policy 在单个样本上过度承诺。对坏的动作（<span class="arithmatex">\(\hat{A}&lt;0\)</span>）：目标为 <span class="arithmatex">\(\min(r\hat{A}, 0.8\hat{A})\)</span>。一旦 <span class="arithmatex">\(r\)</span> 降至 0.8 以下，惩罚便不再增长——防止灾难性遗忘（Catastrophic Forgetting）。</p>

<p><strong>关键洞见</strong>：它是对 TRPO 的 KL 约束的一阶近似，但无需昂贵的二阶优化。每次更新对 policy 的改变最多 <span class="arithmatex">\(\pm\)</span>20%。</p>

<p><strong>对 LLM 而言</strong>：Token 级比率 <span class="arithmatex">\(r_t = \pi_\theta(y_t|y_{&lt;t})/\pi_\text{old}(y_t|y_{&lt;t})\)</span> 防止任何单个 Token 的概率发生过大变化，从而维持生成的连贯性。</p>

<p><em><strong>复习：</strong> 第 <a href="../05-ppo/index.md">5</a> 章（PPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-2"><h3 class="question-title">Q2：从第一原理推导 DPO。它做了哪些假设？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：从 RLHF 目标出发：<span class="arithmatex">\(\max_\pi \mathbb{E}[r(x,y)] - \beta D_\text{KL}[\pi\|\pi_\text{ref}]\)</span>。</p>

<p><strong>步骤 1</strong>：写出 KKT 条件。最优 policy 有闭式解：<span class="arithmatex">\(\pi^*(y|x) \propto \pi_\text{ref}(y|x)\exp(r(x,y)/\beta)\)</span>。</p>

<p><strong>步骤 2</strong>：反解出 reward 的表达：<span class="arithmatex">\(r(x,y) = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z(x)\)</span>。</p>

<p><strong>步骤 3</strong>：代入 Bradley-Terry 模型 <span class="arithmatex">\(P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))\)</span>。配分函数 <span class="arithmatex">\(Z(x)\)</span> 抵消（同一 prompt）。</p>

<p><strong>步骤 4</strong>：将 <span class="arithmatex">\(\pi^*\)</span> 替换为 <span class="arithmatex">\(\pi_\theta\)</span>（我们要训练的参数化 policy）：<span class="arithmatex">\(\mathcal{L} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)})]\)</span>。</p>

<p><strong>假设</strong>：</p>

<ol>
<li><p>Bradley-Terry 偏好模型（成对、无平局、可传递）。</p></li>
<li><p>最优 policy 可被 <span class="arithmatex">\(\pi_\theta\)</span> 实现（容量足够）。</p></li>
<li><p>偏好与训练数据来自同一分布（无分布漂移）。</p></li>
<li><p>参考模型是固定且合理的。</p></li>
</ol>

<p><strong>当假设失效时</strong>：真实偏好并不可传递，数据在训练中漂移，标签有噪声 <span class="arithmatex">\(\rightarrow\)</span> 这就是 Online DPO 和 IPO 存在的原因。</p>

<p><em><strong>复习：</strong> 第 <a href="../06-dpo/index.md">6</a> 章（DPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-3"><h3 class="question-title">Q3：GRPO vs PPO——你会在什么场景下选择哪一个？权衡何在？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>GRPO 的优势</strong>：</p>

<ul>
<li><p>无需 value function：节省了一个模型规模的显存与复杂度。</p></li>
<li><p>更简单：超参更少，更直观（高于均值 = 好，低于均值 = 坏）。</p></li>
<li><p>更适合可验证奖励：数学/代码中 <span class="arithmatex">\(r \in \{0, 1\}\)</span> 给出清晰信号。</p></li>
<li><p>DeepSeek-R1 证明，仅用二元 reward 即可教出涌现的推理能力。</p></li>
</ul>

<p><strong>PPO 的优势</strong>：</p>

<ul>
<li><p>逐 Token 信用分配：value function 为每个 Token 分配 reward，而非仅在序列级。</p></li>
<li><p>样本效率更高：GAE 用 value 预测估计 advantage，无需生成 <span class="arithmatex">\(G\)</span> 个样本。</p></li>
<li><p>更适合细腻的 reward：当 reward 是连续的、在 Token 间显著变化时。</p></li>
<li><p>更成熟：在 OpenAI、Anthropic 等公司经过实战检验。</p></li>
</ul>

<p><strong>经验法则</strong>：若 reward 可验证（对/错） <span class="arithmatex">\(\rightarrow\)</span> GRPO。若 reward 细腻（RM 分数）且追求极致质量 <span class="arithmatex">\(\rightarrow\)</span> PPO。若算力有限 <span class="arithmatex">\(\rightarrow\)</span> GRPO（无需训练 critic）。</p>

<p><strong>算力对比</strong>：GRPO 为每个 prompt 生成 <span class="arithmatex">\(G\)</span> 个回答（生成量增加 8<span class="arithmatex">\(\times\)</span>），但跳过 value function 训练。净算力相近，但分布不同（生成更多，训练更少）。</p>

<p><em><strong>复习：</strong> 第 <a href="../05-ppo/index.md">5</a> 与 <a href="../07-grpo/index.md">7</a> 章（PPO；GRPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-4"><h3 class="question-title">Q4：GAE 如何工作？为 LLM 走一个具体例子。</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：GAE = <span class="arithmatex">\(n\)</span>-步 TD 误差的加权和：<span class="arithmatex">\(\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}\)</span>。</p>

<p><strong>具体例子</strong>：回答有 5 个 Token。仅在末端有 reward（<span class="arithmatex">\(r_5 = 0.8\)</span>）。Value 预测：<span class="arithmatex">\(V_1=0.5, V_2=0.55, V_3=0.6, V_4=0.65, V_5=0.7\)</span>。</p>

<p>TD 误差（<span class="arithmatex">\(\gamma=1\)</span>）：<span class="arithmatex">\(\delta_1 = 0 + V_2 - V_1 = 0.05\)</span>，<span class="arithmatex">\(\delta_2 = 0 + V_3 - V_2 = 0.05\)</span>，……，<span class="arithmatex">\(\delta_5 = 0.8 + 0 - 0.7 = 0.1\)</span>。</p>

<p>取 <span class="arithmatex">\(\lambda = 0.95\)</span>：<span class="arithmatex">\(\hat{A}_5 = 0.1\)</span>（仅是最后的 TD 误差），<span class="arithmatex">\(\hat{A}_4 = 0.05 + 0.95 \times 0.1 = 0.145\)</span>，<span class="arithmatex">\(\hat{A}_3 = 0.05 + 0.95 \times 0.145 = 0.188\)</span>，依此类推。</p>

<p><strong>解读</strong>：Token 3 获得 advantage 0.188，因为它对一个得到高于预期 reward 的序列做出了贡献。更早的 Token 通过指数衰减获得信用。</p>

<p><strong>对 LLM</strong>：<span class="arithmatex">\(\gamma=1.0\)</span>（所有 Token 都重要，且为有限视野）。Token <span class="arithmatex">\(t\)</span> 的 advantage 回答：“鉴于此 Token 之后发生的事，这个 Token 的选择比预期更好还是更差？”</p>

<p><em><strong>复习：</strong> 第 <a href="../05-ppo/index.md">5</a> 章（PPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-5"><h3 class="question-title">Q5：如何防止 reward hacking？给出一种分层防御策略。</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>检测信号</strong>：RM 分数上升但胜率持平或下降；回答长度单调增长；KL 散度 <span class="arithmatex">\(&gt;\)</span> 15；多样性（unique n-grams）下降；阅读高 reward 输出可发现作弊模式。</p>

<p><strong>分层防御（按优先级排序）</strong>：</p>

<ol>
<li><p><strong>KL 惩罚</strong>（首要）：自适应控制器以 KL <span class="arithmatex">\(\approx\)</span> 6 为目标。若 KL 上升，<span class="arithmatex">\(\beta\)</span> 自动增大。防止过度偏离参考模型。</p></li>
<li><p><strong>Reward 模型集成</strong>（3–5 个模型）：取分数的 min 或 mean。各模型有不同的盲区——能骗过一个的漏洞很少能骗过所有。</p></li>
<li><p><strong>长度惩罚</strong>：<span class="arithmatex">\(r&#x27; = r - c \cdot \max(0, \text{length} - L_\text{target})\)</span>。防止“只要更长就得高分”的作弊。</p></li>
<li><p><strong>周期性 RM 刷新</strong>：每 2000 步，从当前 policy 生成数据、重新标注，并加入 RM 训练集。模型一找到作弊路径就立即堵上。</p></li>
<li><p><strong>基于胜率的停训</strong>：跟踪相对 SFT baseline 的胜率。如果 RM 分数上升但胜率停滞 200+ 步，立即停训。模型在作弊，而非进步。</p></li>
</ol>

<p><strong>检测后恢复</strong>：回滚到最近一个“干净”的 checkpoint。将 <span class="arithmatex">\(\beta\)</span> 提升 2<span class="arithmatex">\(\times\)</span>。把发现的作弊模式加入 RM 的负样本。</p>

<p><em><strong>复习：</strong> 第 <a href="../09-reward-models/index.md">9</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（Reward 模型训练；系统架构）。</em></p>
</details></section>

## 28.3　系统设计题 {#system-design-questions}

<section class="quiz-question" id="q-6"><h3 class="question-title">Q6：为训练 70B 模型设计一个 RLHF 系统。逐组件讲解。</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：基于 72 张 A100-80GB GPU 的三集群解耦架构：</p>

<p><strong>集群 1 ——生成（32 GPU）</strong>：</p>

<ul>
<li><p>8 个 vLLM 实例，每个 TP=4。PagedAttention + 投机解码（speculative decoding，1B 草稿模型）。</p></li>
<li><p>连续批处理，最多 256 条在飞序列。INT8 权重以省带宽。</p></li>
<li><p>输出：(prompt, response, 逐 Token log-probs)。吞吐量：<span class="arithmatex">\(\sim\)</span>500 回答/分钟。</p></li>
<li><p>无状态：仅从共享存储加载最新权重。恢复极简。</p></li>
</ul>

<p><strong>集群 2 ——打分（8 GPU）</strong>：</p>

<ul>
<li><p>Reward 模型（70B，INT8 = 70GB），4 GPU（TP=4）。</p></li>
<li><p>参考模型（70B，INT8），4 GPU（TP=4）。计算逐 Token 的 log-probs 以算 KL。</p></li>
<li><p>输出：reward 分数 + 逐 Token 的 KL。轻量级，批量推理。</p></li>
</ul>

<p><strong>集群 3 ——训练（32 GPU）</strong>：</p>

<ul>
<li><p>Policy 模型用 FSDP（ZeRO-3）。FlashAttention 2。梯度检查点。</p></li>
<li><p>从 buffer 消费已打分的 experience。PPO 更新：在 mini-batch=16 上跑 4 个 epoch。</p></li>
<li><p>每 50 步将更新后的权重推送到共享存储（异步、后台传输）。</p></li>
<li><p>每 100 步做一次异步 checkpoint（非阻塞写入 NVMe + S3 备份）。</p></li>
</ul>

<p><strong>连接结构</strong>：</p>

<ul>
<li><p>生成 <span class="arithmatex">\(\rightarrow\)</span> 打分：Ray/Redis 队列（每 batch <span class="arithmatex">\(\sim\)</span>10 MB：Token ID + log-probs）。</p></li>
<li><p>打分 <span class="arithmatex">\(\rightarrow\)</span> 训练：经验缓冲（循环式，保存最近 500 步）。</p></li>
<li><p>训练 <span class="arithmatex">\(\rightarrow\)</span> 生成：共享并行文件系统（Lustre/GPFS）上的权重存储。每 50 步异步推送 140GB。</p></li>
</ul>

<p><strong>重叠</strong>：当训练处理第 <span class="arithmatex">\(N\)</span> 步时，生成已经在为第 <span class="arithmatex">\(N+1\)</span> 步生产数据。这隐藏了生成延迟，相对单体架构吞吐提升 1.3–1.5<span class="arithmatex">\(\times\)</span>。</p>

<p><em><strong>复习：</strong> 第 <a href="../11-training-infrastructure/index.md">11</a> 章（系统架构与大规模基础设施）。</em></p>
</details></section>

<section class="quiz-question" id="q-7"><h3 class="question-title">Q7：如何处理生成瓶颈？量化各种方案的收益。</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：生成占 RLHF 总墙钟时间的 60–70%。根本原因：自回归解码受显存带宽约束（算术强度 <span class="arithmatex">\(\approx 1\)</span> FLOP/byte，而 A100 的 roofline 是 156 FLOP/byte）。</p>

<p><strong>按影响力排序的方案</strong>：</p>

<p><strong>1. 解耦生成与训练</strong>（端到端 1.3–1.5<span class="arithmatex">\(\times\)</span>）：在独立硬件上运行生成，与训练重叠。最大的单一架构性胜利。</p>

<p><strong>2. vLLM + PagedAttention</strong>（2–4<span class="arithmatex">\(\times\)</span>）：消除内部碎片造成的 60–80% KV cache 显存浪费。可承载 3–4<span class="arithmatex">\(\times\)</span> 更大的 batch = 更好的带宽利用。</p>

<p><strong>3. 连续批处理</strong>（1.5–2<span class="arithmatex">\(\times\)</span>）：不等最长序列结束。立即在腾出的槽位开新序列。保持 GPU 繁忙。</p>

<p><strong>4. 投机解码（Speculative Decoding）</strong>（2–3<span class="arithmatex">\(\times\)</span>）：1B 草稿模型提议 5 个 Token。70B 模型在一次前向中并行验证全部 5 个！平均接受 3–4 个 <span class="arithmatex">\(\rightarrow\)</span> 每次前向产出 3–4 个 Token 而非 1 个。</p>

<p><strong>5. 生成权重用 INT8/FP8</strong>（2<span class="arithmatex">\(\times\)</span>）：将每 Token 的 140GB 权重读取减半。质量损失极小，因为：(a) 我们本就在用温度采样，(b) 仅生成使用 INT8，训练仍是 BF16。</p>

<p><strong>6. CUDA graphs + kernel fusion</strong>（1.1–1.3<span class="arithmatex">\(\times\)</span>）：消除 Python/CUDA 启动开销。把 layernorm+attention+MLP 融合成更少的 kernel。</p>

<p><strong>组合后</strong>：<span class="arithmatex">\(1.5 \times 3 \times 1.5 \times 2.5 \times 2 \times 1.2 = 40\times\)</span> 优于朴素实现。实际中由于边际收益递减，整体大约为朴素实现的 10–20<span class="arithmatex">\(\times\)</span>。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（系统基础；系统架构）。</em></p>
</details></section>

<section class="quiz-question" id="q-8"><h3 class="question-title">Q8：解释解耦系统中的权重同步。能容忍多少陈旧度（staleness）？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>问题</strong>：生成集群使用 policy 权重生成回答，训练集群更新这些权重。它们位于不同硬件上。如何保持同步？</p>

<p><strong>为什么完美同步是浪费</strong>：70B BF16 全量同步 = 140GB。InfiniBand 400Gb/s（50GB/s）下，每次同步 2.8s。若每步（每 50–90s）都同步，权重传输占总时间的 3–5%。可接受，但没必要。</p>

<p><strong>陈旧度容忍分析</strong>：</p>

<ul>
<li><p>每步 policy 变化：<span class="arithmatex">\(\sim\)</span>0.1%（按参数均值差衡量）。</p></li>
<li><p>50 步：<span class="arithmatex">\(\sim\)</span>5% 累计漂移。</p></li>
<li><p>PPO 裁剪范围：可处理高达 20% 的概率比偏离。</p></li>
<li><p>经验：50 步陈旧度 <span class="arithmatex">\(\rightarrow\)</span> 质量下降 <span class="arithmatex">\(&lt;\)</span>2%（按胜率衡量）。</p></li>
</ul>

<p><strong>生产策略</strong>：</p>

<ol>
<li><p>每 50 训练步：向共享存储推送一次 BF16 完整 checkpoint（传输 2.8s）。</p></li>
<li><p>生成集群：在 batch 之间做非阻塞的权重重载。</p></li>
<li><p>增量压缩（可选）：仅发送变更参数（INT8 delta <span class="arithmatex">\(\approx\)</span> 5GB），作为偏移应用。带宽减少 10<span class="arithmatex">\(\times\)</span>。</p></li>
<li><p>超大规模（256+ GPU）：流式同步——后台连续发送小块。平均陈旧度：5–10 步。</p></li>
</ol>

<p><strong>关键细节</strong>：生成阶段计算的 log-probs 使用的是陈旧权重。PPO 比率 <span class="arithmatex">\(\pi_\text{new}/\pi_\text{old}\)</span> 用这些陈旧 log-probs 作为 <span class="arithmatex">\(\pi_\text{old}\)</span>。这没问题，因为 PPO 本就被设计为处理 off-policy 修正。</p>

<p><em><strong>复习：</strong> 第 <a href="../11-training-infrastructure/index.md">11</a> 章（系统架构与大规模基础设施）。</em></p>
</details></section>

<section class="quiz-question" id="q-9"><h3 class="question-title">Q9：在 512 GPU 规模下如何做到容错？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：在 512 GPU 规模下，MTBF 为 4–8 小时。5 天的训练会经历 15–30 次故障。</p>

<p><strong>架构级韧性</strong>：</p>

<ul>
<li><p>生成集群 = 无状态。失败实例在 <span class="arithmatex">\(&lt;\)</span>60s 内重启（只需加载权重，无状态）。</p></li>
<li><p>训练集群 = 有状态。需要基于 checkpoint 的恢复。</p></li>
<li><p>打分集群 = 无状态。与生成相同。</p></li>
</ul>

<p><strong>Checkpoint 策略</strong>：</p>

<ul>
<li><p>频率：每 50–100 步（5–10 分钟训练）。</p></li>
<li><p>方法：异步（非阻塞）。后台线程在下一步进行时写入。使用 FSDP 的分布式保存（每个 rank 并行保存自己的分片）。</p></li>
<li><p>内容：模型权重、优化器状态（Adam m/v）、LR scheduler、RNG 状态、KL 自适应系数、全局 step 计数器、回放缓冲指针。</p></li>
<li><p>存储：本地 NVMe（快，70B 仅 30s）+ 异步拷贝到 S3 / 共享 FS（持久化）。</p></li>
<li><p>保留：保留最近 3 个 checkpoint。自动删除更早的。</p></li>
</ul>

<p><strong>检测与恢复流程</strong>：</p>

<ol>
<li><p>NCCL 集合通信超时（60s）或心跳丢失（10s）<span class="arithmatex">\(\rightarrow\)</span> 检测到故障。</p></li>
<li><p>通过 NVML 健康检查定位故障节点。</p></li>
<li><p>方案 A（快，<span class="arithmatex">\(&lt;\)</span>2 min）：Torch Elastic 缩减 world size、重分配分片，以 <span class="arithmatex">\(N-1\)</span> 节点继续。后台请求替换。</p></li>
<li><p>方案 B（干净，<span class="arithmatex">\(\sim\)</span>5 min）：拉起替换节点、重建进程组、加载最近 checkpoint、恢复。</p></li>
<li><p>经验缓冲已持久化——无需重新生成。</p></li>
</ol>

<p><strong>预防</strong>：上线前的压力测试（GEMM、显存、NVLink）。ECC 错误监控（错误激增时抢占式迁移）。热备节点（环境预先加载）。双轨 InfiniBand 提供网络冗余。</p>

<p><em><strong>复习：</strong> 第 <a href="../11-training-infrastructure/index.md">11</a> 章（系统架构与大规模基础设施）。</em></p>
</details></section>

<section class="quiz-question" id="q-10"><h3 class="question-title">Q10：如何从 7B 扩展到 70B 再到 405B？每个量级要变什么？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>7B（单 8-GPU 节点，数小时）</strong>：</p>

<ul>
<li><p>架构：单体（TRL 默认）。所有模型在同一组 GPU 上。</p></li>
<li><p>显存：LoRA + INT8 的 ref/RM 在 8<span class="arithmatex">\(\times\)</span>80GB 内轻松容纳。</p></li>
<li><p>并行：节点内 DP=8 或 FSDP。无网络通信。</p></li>
<li><p>超参：LR=<span class="arithmatex">\(5\times10^{-6}\)</span>，激进的 <span class="arithmatex">\(\beta\)</span>=0.02，50K–100K 步。</p></li>
<li><p>时间：每次运行 4–12 小时。快速迭代。</p></li>
</ul>

<p><strong>70B（32–64 GPU，2–5 天）</strong>：</p>

<ul>
<li><p>架构：半解耦。vLLM 生成 + FSDP 训练。</p></li>
<li><p>显存：ZeRO-3 必不可少。梯度检查点。INT8 ref/RM。</p></li>
<li><p>并行：节点内 TP=8（生成），节点间 FSDP（训练）。</p></li>
<li><p>超参：LR=<span class="arithmatex">\(1.5\times10^{-6}\)</span>，温和的 <span class="arithmatex">\(\beta\)</span>=0.05，10K–30K 步。</p></li>
<li><p>容错：异步 checkpoint、监控，但仍可人工管理。</p></li>
</ul>

<p><strong>405B（256–512 GPU，1–3 周）</strong>：</p>

<ul>
<li><p>架构：完全解耦。独立集群。权重存储 + 队列。</p></li>
<li><p>显存：训练用 ZeRO-3 + TP=8 + PP=2。生成用 INT4。</p></li>
<li><p>并行：3D 并行（TP<span class="arithmatex">\(\times\)</span>PP<span class="arithmatex">\(\times\)</span>DP = 8<span class="arithmatex">\(\times\)</span>2<span class="arithmatex">\(\times\)</span>16 = 256 GPU 训练）。</p></li>
<li><p>超参：LR=<span class="arithmatex">\(5\times10^{-7}\)</span>，非常保守的 <span class="arithmatex">\(\beta\)</span>=0.1，2K–5K 步。</p></li>
<li><p>容错：必备。弹性训练、冗余 checkpoint、热备节点。</p></li>
<li><p>关键变化：所需 RL 训练量大幅减少（模型预训练后已非常强）。但每步贵 50<span class="arithmatex">\(\times\)</span>，因此不稳定性是灾难性的。</p></li>
</ul>

<p><strong>悖论</strong>：更大的模型实际上<em>逐步 RL 训练更容易</em>（更稳定，loss 地形更平滑）。但不稳定性的代价随模型规模放大——405B 上一次失败的训练浪费 $100K+ 的算力。</p>

<p><em><strong>复习：</strong> 第 <a href="../11-training-infrastructure/index.md">11</a> 章（系统架构与大规模基础设施）。</em></p>
</details></section>

## 28.4　实战与调试题 {#practical-and-debugging-questions}

<section class="quiz-question" id="q-11"><h3 class="question-title">Q11：reward 分数上升但模型质量下降。诊断并修复。</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：典型的<strong>reward hacking / 古德哈特定律（Goodhart’s Law）</strong>。</p>

<p><strong>诊断流程</strong>：</p>

<ol>
<li><p><strong>检查回答长度</strong>：绘制训练过程中的平均长度。单调增长？= 长度作弊（RM 给更长的回答更高分）。</p></li>
<li><p><strong>检查 KL 散度</strong>：<span class="arithmatex">\(&gt;\)</span>15？= policy 偏离参考模型过远，丧失了能力。</p></li>
<li><p><strong>检查多样性</strong>：每条回答的 unique trigrams。下降？= mode collapse（重复同一高 reward 模式）。</p></li>
<li><p><strong>人工检查</strong>：阅读 20 个最高 reward 的回答。它们共享什么模式？（例如：都以&quot;Great question!&quot;开头、都用项目符号、过度模棱两可）。</p></li>
<li><p><strong>胜率</strong>：在留出 prompt 上与 SFT baseline 比较。若 RM 上升而胜率持平/下降 = 确认存在作弊。</p></li>
</ol>

<p><strong>即时修复</strong>：</p>

<ul>
<li><p>回滚到胜率还在提升的最近 checkpoint。</p></li>
<li><p>将 <span class="arithmatex">\(\beta\)</span> 提升 2–3<span class="arithmatex">\(\times\)</span>（更强的 KL 惩罚）。</p></li>
<li><p>添加显式长度惩罚：<span class="arithmatex">\(r&#x27; = r - 0.001 \cdot \max(0, \text{len} - 500)\)</span>。</p></li>
</ul>

<p><strong>结构性修复（防止复发）</strong>：</p>

<ul>
<li><p>RM 集成：在不同数据切分上训练 3–5 个 RM。取 min 或 mean。作弊往往针对特定模型。</p></li>
<li><p>RM 刷新：每 2000 步，从当前 policy 生成、获取人工标签、重训 RM。</p></li>
<li><p>多目标 reward：用独立 RM 组合“有用性 + 无害性 + 简洁性”。</p></li>
<li><p>基于胜率（而非 RM 分数）早停。你优化的指标应与训练信号不同。</p></li>
</ul>

<p><em><strong>复习：</strong> 第 <a href="../09-reward-models/index.md">9</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（Reward 模型训练；系统架构）。</em></p>
</details></section>

<section class="quiz-question" id="q-12"><h3 class="question-title">Q12：如何决定 RL 训练的 prompt 分布？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：prompt 质量是 RLHF 中最被低估的因素。差的 prompt = 没有学习信号。</p>

<p><strong>组成（我默认的配比）</strong>：</p>

<ul>
<li><p>40% 真实用户流量（代表实际用例）。</p></li>
<li><p>30% 合成（LLM 生成，填补覆盖空白——稀有主题、边角情况）。</p></li>
<li><p>20% 课程式（渐进难度——先易，随模型变强提升复杂度）。</p></li>
<li><p>10% 对抗式（red-team prompt、越狱尝试、含糊指令）。</p></li>
</ul>

<p><strong>关键：Goldilocks 过滤器</strong>：</p>

<ol>
<li><p>对每个候选 prompt，用当前模型生成 4–8 个回答。</p></li>
<li><p>用 RM 打分。计算通过率（超过阈值的比例）。</p></li>
<li><p>仅保留通过率在 20–80% 的 prompt：</p>

<ul>
<li><p><span class="arithmatex">\(&lt;\)</span>20%：太难。模型几乎总失败 <span class="arithmatex">\(\rightarrow\)</span> 全是负 advantage <span class="arithmatex">\(\rightarrow\)</span> 无有效梯度。</p></li>
<li><p><span class="arithmatex">\(&gt;\)</span>80%：太易。模型几乎总成功 <span class="arithmatex">\(\rightarrow\)</span> 全是正 advantage <span class="arithmatex">\(\rightarrow\)</span> 无对比。</p></li>
<li><p>20–80%：完美。成功与失败兼有 <span class="arithmatex">\(\rightarrow\)</span> 关于“什么有效”的清晰信号。</p></li>
</ul></li>
<li><p>每 500 训练步重新过滤（模型在进步，难度分布在漂移）。</p></li>
</ol>

<p><strong>主题多样性</strong>：确保没有单一主题主导（每类 <span class="arithmatex">\(&lt;\)</span>10%）。用 Embedding 聚类验证覆盖度。否则模型会过度优化主导主题。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../09-reward-models/index.md">9</a> 章（GRPO；Reward 模型训练）。</em></p>
</details></section>

<section class="quiz-question" id="q-13"><h3 class="question-title">Q13：RLHF 中 LoRA vs 全量微调。各自何时使用？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>LoRA</strong>（Low-Rank Adaptation，<span class="arithmatex">\(r\)</span>=64，<span class="arithmatex">\(\alpha\)</span>=16）：</p>

<ul>
<li><p>可训练参数：模型的 <span class="arithmatex">\(\sim\)</span>0.2%（70B 对应 200M）。</p></li>
<li><p>显存节省：无需独立参考模型（base 模型 = 参考）！节省 140GB。</p></li>
<li><p>稳定性：天然更稳定（低秩约束限制了 policy 的漂移幅度）。</p></li>
<li><p>速度：每步更快（更新参数更少），但可能需要更多步。</p></li>
<li><p>质量天花板：通常达到全量微调的 90–95%。</p></li>
</ul>

<p><strong>全量微调</strong>：</p>

<ul>
<li><p>所有参数都更新。表达力最强。</p></li>
<li><p>需要独立的参考模型副本（70B 为 140GB）。或者通过非常频繁的 checkpoint 作为“锚”。</p></li>
<li><p>灾难性遗忘的风险更高。需要更低的 LR（比 LoRA 低 <span class="arithmatex">\(3\times\)</span>）和更强的 <span class="arithmatex">\(\beta\)</span>。</p></li>
<li><p>更适合：需要大幅分布漂移（新语言、迥异风格），LoRA 触及容量上限的情况。</p></li>
</ul>

<p><strong>我的决策框架</strong>：</p>

<ol>
<li><p>先用 LoRA（<span class="arithmatex">\(r\)</span>=64）。便宜 3<span class="arithmatex">\(\times\)</span> 且更稳定。</p></li>
<li><p>监控 LoRA 矩阵的梯度范数。如果持续 <span class="arithmatex">\(&gt;\)</span>1.0（相对于参数量很大）：LoRA 已到容量上限。</p></li>
<li><p>仅当胜率停滞 <em>且</em> 梯度分析指示容量受限时，才切换到全量微调。</p></li>
<li><p>对全量微调：使用 <span class="arithmatex">\(\text{LR}/3\)</span>、<span class="arithmatex">\(\beta \times 2\)</span>、更频繁的 checkpoint，以及基于胜率的早停。</p></li>
</ol>

<p><strong>混合</strong>：LoRA 用于对齐/安全（小行为漂移）+ 全量微调用于能力/推理（需要大漂移）。</p>

<p><em><strong>复习：</strong> 第 <a href="../01-llm-architecture/index.md">1</a> 与 <a href="../10-sft/index.md">10</a> 章（LLM 架构；SFT 最佳实践）。</em></p>
</details></section>

<section class="quiz-question" id="q-14"><h3 class="question-title">Q14：过程奖励模型（Process Reward Model, PRM）vs 结果奖励模型（Outcome Reward Model, ORM）。设计一个 PRM 系统。</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>ORM</strong>：只给最终答案打分。“整体回答好不好？”简单但无法定位推理<em>在何处</em>出错。</p>

<p><strong>PRM</strong>：为每个中间步骤打分。“这个推导的第 3 步对吗？”信息量大得多，但更难训练。</p>

<p><strong>PRM 对推理任务的优势</strong>：</p>

<ul>
<li><p>准确定位推理失败的位置（步骤级信用分配）。</p></li>
<li><p>支持树搜索：只展开步骤分数高的分支。</p></li>
<li><p>reward hacking 更难：错误步骤 + 侥幸正确答案无法拿到高分。</p></li>
<li><p>在 MATH 基准上，PRM + Best-of-N 比 ORM + Best-of-N 高 10–20%。</p></li>
</ul>

<p><strong>训练 PRM</strong>：</p>

<ol>
<li><p><strong>数据收集</strong>（蒙特卡洛方法）：</p>

<ul>
<li><p>对每道题，逐步生成推理轨迹。</p></li>
<li><p>在每一步 <span class="arithmatex">\(k\)</span>，从该步出发把解完整补全 <span class="arithmatex">\(M\)</span> 次（<span class="arithmatex">\(M=32\)</span>）。</p></li>
<li><p>步骤分数 = 抵达正确答案的补全比例。</p></li>
<li><p>分数明显下降的步骤 = “出错”的步骤。</p></li>
</ul></li>
<li><p><strong>标注</strong>：补全率 <span class="arithmatex">\(&gt;\)</span> 50% 的步骤标为“正确”，<span class="arithmatex">\(&lt;\)</span> 20% 的标为“错误”。</p></li>
<li><p><strong>模型</strong>：与 base 模型同架构 + 每个 Token 位置上的分类头。用步骤标签做二元交叉熵训练。</p></li>
<li><p><strong>推理</strong>：为每步打分。若任一步骤分数 <span class="arithmatex">\(&lt;\)</span>0.3，则将该轨迹标记为有缺陷。</p></li>
</ol>

<p><strong>在 RLHF 中使用 PRM</strong>：PRM 给出的逐 Token reward 直接喂入 GAE。每个 Token 都获得即时反馈，而非仅在序列末端。这极大改善了长推理链上的信用分配。</p>

<p><em><strong>复习：</strong> 第 <a href="../09-reward-models/index.md">9</a> 与 <a href="../13-reasoning/index.md">13</a> 章（Reward 模型训练；大型推理模型的 RL）。</em></p>
</details></section>

<section class="quiz-question" id="q-15"><h3 class="question-title">Q15：如何评估 RL 是否真的改进了模型？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：多面向评估（单一指标无法捕捉“质量”）：</p>

<p><strong>1. 胜率</strong>（最重要、最可靠）：</p>

<ul>
<li><p>500+ 多样 prompt。用 LLM judge（GPT-4 或 Claude）在与 SFT baseline 的盲对比 A/B 中选胜者。</p></li>
<li><p>目标：<span class="arithmatex">\(&gt;\)</span>55% 胜率 = 有意义的改进；<span class="arithmatex">\(&gt;\)</span>65% = 强改进。</p></li>
<li><p>使用位置去偏（互换 A/B 顺序、取平均）。报告置信区间。</p></li>
</ul>

<p><strong>2. 能力基准</strong>（回退检测）：</p>

<ul>
<li><p>MMLU（知识）、HumanEval（代码）、MATH（推理）、MT-Bench（多轮）。</p></li>
<li><p>任意 <span class="arithmatex">\(&gt;\)</span>2% 的下降 = 值得警惕的对齐税。深入排查哪些类别退化。</p></li>
</ul>

<p><strong>3. 类别专项评测</strong>：</p>

<ul>
<li><p>安全：对有害 prompt 的拒绝率（应上升）。</p></li>
<li><p>真实性：TruthfulQA 分数（应上升或持平）。</p></li>
<li><p>有用性：指令跟随基准上的任务完成率。</p></li>
</ul>

<p><strong>4. 分布性指标</strong>：</p>

<ul>
<li><p>回答长度分布（不应发生剧烈漂移）。</p></li>
<li><p>词汇多样性（每条回答的 unique tokens）。</p></li>
<li><p>格式合规（若训练目标包含特定格式）。</p></li>
</ul>

<p><strong>5. 人工评测</strong>（金标准，昂贵）：</p>

<ul>
<li><p>每个样本由 3+ 名熟练标注者做盲 A/B。标注者间一致性 <span class="arithmatex">\(&gt;\)</span> 70%。</p></li>
<li><p>仅用于最终模型选择，训练过程中不用（太慢/太贵）。</p></li>
</ul>

<p><strong>红旗信号</strong>：RM 分数上升 + 胜率持平 = reward hacking，不是真改进。胜率上升 + 基准下降 = 对齐税过高，需降低 RL 强度。</p>

<p><em><strong>复习：</strong> 第 <a href="../14-evaluation/index.md">14</a> 章（LLM 评估）。</em></p>
</details></section>

<section class="quiz-question" id="q-16"><h3 class="question-title">Q16：端到端描述 reward 模型训练流水线。</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>阶段 1——数据生成</strong>：</p>

<ul>
<li><p>收集 50K–100K 多样 prompt（真实流量 + 合成）。</p></li>
<li><p>对每个 prompt 用不同温度（0.3、0.7、1.0）和多个模型生成 4–8 个回答（多样性是关键——如果所有回答都相似，偏好就毫无信息量）。</p></li>
<li><p>总计：200K–800K 个候选回答。</p></li>
</ul>

<p><strong>阶段 2——偏好收集</strong>：</p>

<ul>
<li><p>方案 A（昂贵，质量最佳）：人工标注者两两比较。每对 3 名标注者。成本：每次比较 $2–5。</p></li>
<li><p>方案 B（便宜，与人类一致率 85–90%）：LLM judge（GPT-4/Claude）。便宜 10<span class="arithmatex">\(\times\)</span>。适合规模化。</p></li>
<li><p>格式：(prompt, chosen response, rejected response)。标注者不一致（一致率 <span class="arithmatex">\(&lt;\)</span>70%）的对被丢弃。</p></li>
<li><p>最终数据集：100K–500K 对。</p></li>
</ul>

<p><strong>阶段 3——训练</strong>：</p>

<ul>
<li><p>架构：与 base LLM 相同 + 标量头（每个序列一个回归输出）。</p></li>
<li><p>Loss：<span class="arithmatex">\(\mathcal{L} = -\mathbb{E}[\log\sigma(r(x,y_w) - r(x,y_l))]\)</span>（Bradley-Terry）。</p></li>
<li><p>训练：<strong>仅 1 个 epoch！</strong> RM 过拟合极快。验证准确率 68–75% 即为良好（更高往往意味着过拟合到标注 artifact）。</p></li>
<li><p>技巧：将 reward 中心化到 0（减去滑动均值）。检查长度偏差（若长度与分数相关 <span class="arithmatex">\(&gt;\)</span> 0.3，在训练中加长度惩罚）。</p></li>
</ul>

<p><strong>阶段 4——验证</strong>：</p>

<ul>
<li><p>留出偏好对：准确率应在 68–75%。</p></li>
<li><p>在新数据上与人类的一致率：<span class="arithmatex">\(&gt;\)</span> 80%。</p></li>
<li><p>长度偏差检查：回答长度与 RM 分数的相关 <span class="arithmatex">\(&lt;\)</span> 0.2。</p></li>
<li><p>一致性检查：同一 prompt 下，改写后的回答应得到相近分数。</p></li>
</ul>

<p><em><strong>复习：</strong> 第 <a href="../09-reward-models/index.md">9</a> 章（Reward 模型训练）。</em></p>
</details></section>

<section class="quiz-question" id="q-17"><h3 class="question-title">Q17：KL 散度爆炸时会发生什么？根因与修复。</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>KL 衡量什么</strong>：当前 policy 与参考之间的平均对数比：<span class="arithmatex">\(D_\text{KL} = \mathbb{E}_{y\sim\pi_\theta}[\log(\pi_\theta(y|x)/\pi_\text{ref}(y|x))]\)</span>。KL=0 意味着与参考完全相同。KL=10 意味着 policy 对其偏好的输出多投放了 10 nats 的概率。</p>

<p><strong>健康区间</strong>：训练中 3–10。缓慢增长无妨。突发飙升 = 出问题。</p>

<p><strong>KL 爆炸的根因</strong>：</p>

<ol>
<li><p><strong>学习率过高</strong>：policy 步幅巨大、偏离参考。修复：将 LR 降低 2–5<span class="arithmatex">\(\times\)</span>。</p></li>
<li><p><strong>reward hacking</strong>：找到了远离参考行为的高 reward 漏洞。修复：增大 <span class="arithmatex">\(\beta\)</span>、加入 RM 集成。</p></li>
<li><p><strong>Mode collapse</strong>：policy 集中到一个回答模板上。在该模板处 KL 很高，其他地方都低。修复：加大熵奖励、提高温度。</p></li>
<li><p><strong>坏 batch</strong>：一个具有极端 advantage 的不幸 batch 推动了 policy。修复：梯度裁剪、减小 mini-batch 大小。</p></li>
<li><p><strong>Value function 偏离</strong>：错误的 advantage 估计导致错误的更新。修复：降低 value function 的 LR，或切换到 GRPO（无 value function）。</p></li>
</ol>

<p><strong>恢复流程</strong>：</p>

<ol>
<li><p>检测：KL <span class="arithmatex">\(&gt;\)</span> 15 持续 50+ 步，或单步内 KL 跳 <span class="arithmatex">\(&gt;\)</span>5。</p></li>
<li><p>立即：加载最近一个干净的 checkpoint（KL <span class="arithmatex">\(&lt;\)</span> 10）。</p></li>
<li><p>调整：LR 降低 50%。<span class="arithmatex">\(\beta\)</span> 增大 2<span class="arithmatex">\(\times\)</span>。cliprange 降至 0.1。</p></li>
<li><p>恢复：在头 200 步密切监控。</p></li>
</ol>

<p><em><strong>复习：</strong> 第 <a href="../05-ppo/index.md">5</a> 与 <a href="../07-grpo/index.md">7</a> 章（PPO；GRPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-18"><h3 class="question-title">Q18：比较单体 vs 解耦的 RLHF 架构。各自在什么场景合理？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>单体</strong>（TRL 默认：单进程，所有模型同 GPU）：</p>

<ul>
<li><p>优点：代码简单。无分布式系统复杂度。易于调试。</p></li>
<li><p>缺点：GPU 闲置 60% 时间（生成时计算闲、训练时带宽闲）。无法高效扩展到 <span class="arithmatex">\(\sim\)</span>16 GPU 以上。所有模型争抢同一显存。</p></li>
<li><p>适用：模型 <span class="arithmatex">\(\leq\)</span> 13B，单节点，研究/原型。</p></li>
</ul>

<p><strong>半解耦</strong>（vLLM 生成 + FSDP 训练，同一集群）：</p>

<ul>
<li><p>优点：利用率更好（生成与训练可部分重叠）。可扩展到 64 GPU。</p></li>
<li><p>缺点：仍共享硬件，无法独立优化。比单体更复杂。</p></li>
<li><p>适用：13B–70B，2–8 节点，生产实验。</p></li>
</ul>

<p><strong>完全解耦</strong>（独立集群通过队列连接）：</p>

<ul>
<li><p>优点：每个集群针对其工作负载优化。生成与训练独立扩展。生成集群无状态（容错极简）。可扩展到数百 GPU。</p></li>
<li><p>缺点：分布式系统复杂度。权重陈旧度。队列管理。网络开销。</p></li>
<li><p>适用：<span class="arithmatex">\(\geq\)</span> 70B 生产训练。需要规模、容错、高利用率。</p></li>
</ul>

<p><strong>关键洞见</strong>：生成受带宽约束，训练受算力约束。同一硬件无法同时为二者优化。解耦让生成节点拥有：更多显存带宽、INT8 权重、大 batch。训练节点拥有：完整 BF16 精度、FlashAttention、FSDP 分片。</p>

<p><em><strong>复习：</strong> 第 <a href="../11-training-infrastructure/index.md">11</a> 章（系统架构与大规模基础设施）。</em></p>
</details></section>

<section class="quiz-question" id="q-19"><h3 class="question-title">Q19：如何为 RL 训练设置课程学习？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：课程学习 = 逐渐提升难度，让模型循序渐进地学习。</p>

<p><strong>为什么重要</strong>：若把最难的 prompt 丢给弱模型，它会得到全负的 reward <span class="arithmatex">\(\rightarrow\)</span> 无学习信号（一切都同样糟）。若从易入手，模型先发展基础能力，再在其上构建。</p>

<p><strong>实现</strong>：</p>

<ol>
<li><p><strong>难度评分</strong>：根据当前模型的通过率（来自 Goldilocks 过滤）对每个 prompt 打分。易 = 通过 <span class="arithmatex">\(&gt;\)</span>80%，中 = 30–80%，难 = <span class="arithmatex">\(&lt;\)</span>30%。</p></li>
<li><p><strong>排程</strong>：步 0–1000：70% 易，20% 中，10% 难。步 1000–5000：30% 易，50% 中，20% 难。步 5000+：10% 易，40% 中，50% 难。</p></li>
<li><p><strong>动态调整</strong>：每 500 步重新评估难度分布。模型已“掌握”（通过率 <span class="arithmatex">\(&gt;\)</span> 95%）的 prompt 退役。引入更难的新 prompt。</p></li>
<li><p><strong>对 GRPO</strong>：课程保证 group 总是混合成功与失败。没有课程，难 prompt 会产生全零 group（无用）。</p></li>
</ol>

<p><strong>证据</strong>：DeepSeek-R1 使用隐式课程——从简单的数学/代码问题起步，模型发展出基本推理，再逐步解出更难的问题，无需显式排程。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../12-agentic-training/index.md">12</a> 章（GRPO；LLM 智能体训练）。</em></p>
</details></section>

<section class="quiz-question" id="q-20"><h3 class="question-title">Q20：你有 64 张 A100-80GB GPU 的预算，需要 RL 训练一个 70B 模型。设计资源分配。</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：8 节点 <span class="arithmatex">\(\times\)</span> 8 GPU = 64 张总。需在生成、打分、训练之间分配。</p>

<p><strong>我的分配</strong>：</p>

<ul>
<li><p><strong>生成</strong>：24 GPU（3 节点）。6 个 vLLM 实例，TP=4。INT8 权重 = 70GB/模型，留下空间给 KV cache。连续批处理，总 batch <span class="arithmatex">\(\approx\)</span> 128。</p></li>
<li><p><strong>打分</strong>：8 GPU（1 节点）。RM（INT8，TP=4）+ 参考模型（INT8，TP=4）位于同一节点。或共享 4 GPU，TP=4 在 RM 与 ref 间轮换。</p></li>
<li><p><strong>训练</strong>：32 GPU（4 节点）。在全部 32 张上做 FSDP。每张 GPU 持有 <span class="arithmatex">\(\sim\)</span>70B/32 = 2.2GB 参数 + 优化器片段。激活留有充足余量。开启梯度检查点以保险。</p></li>
</ul>

<p><strong>预期吞吐</strong>：</p>

<ul>
<li><p>生成：6 实例 <span class="arithmatex">\(\times\)</span> <span class="arithmatex">\(\sim\)</span>80 回答/分钟 = 480 回答/分钟。</p></li>
<li><p>训练：batch=128，每 <span class="arithmatex">\(\sim\)</span>15s 一步（仅训练时间，不含等待生成）。</p></li>
<li><p>重叠：当训练第 <span class="arithmatex">\(N\)</span> 步进行（15s）时，生成已为第 <span class="arithmatex">\(N+1\)</span> 步产出 <span class="arithmatex">\(\sim\)</span>120 个回答。完美流水。</p></li>
</ul>

<p><strong>瓶颈分析</strong>：128 个回答的生成需 <span class="arithmatex">\(\sim\)</span>45s。训练 <span class="arithmatex">\(\sim\)</span>15s。打分 <span class="arithmatex">\(\sim\)</span>5s。生成是瓶颈。可将 8 GPU 从训练移到生成（40 生成、24 训练）以平衡，但那样训练变成瓶颈。当前分配接近最优。</p>

<p><strong>显存吃紧时的替代方案</strong>：把打分放到训练节点（分时复用：生成时打分，训练时训练）。省下 8 GPU。流水稍差但可用。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（系统基础；系统架构）。</em></p>
</details></section>

## 28.5　GRPO 变体与高级 RL 题 {#grpo-variants-and-advanced-rl-questions}

<section class="quiz-question" id="q-21"><h3 class="question-title">Q21：什么是 DAPO？它相对标准 GRPO 有哪些改进？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：DAPO（动态自适应策略优化，Dynamic Adaptive Policy Optimization）引入了 5 项关键改动：</p>

<p><strong>1. Clip-Higher（非对称裁剪）</strong>：标准 PPO/GRPO 两侧都按 <span class="arithmatex">\(\epsilon=0.2\)</span> 等量裁剪。DAPO 使用 <span class="arithmatex">\(\epsilon_\text{low}=0.2\)</span> 但 <span class="arithmatex">\(\epsilon_\text{high}=0.28\)</span>。这允许模型<em>更激进地提升</em>好动作的概率，同时仍限制对坏动作的抑制幅度。直觉：探索比利用需要更大的空间。</p>

<p><strong>2. 超长过滤（Overlong Filtering）</strong>：若回答达到最大长度上限（被截断，没有 EOS Token），就将其完全从 loss 中屏蔽。理由：截断的回答不包含自然停止信号——在其上训练会教会模型“句子中间停下”是可接受的。</p>

<p><strong>3. Token 级 Loss</strong>：loss 按所有序列的总 Token 数归一，而非按序列数。这防止更长的序列主导梯度。</p>

<p><strong>4. 软超长惩罚（Soft Overlong Punishment）</strong>：不再是二元截断过滤，而是在回答接近最大长度时施加渐进惩罚。<span class="arithmatex">\(r_\text{soft} = -c \cdot \max(0, \text{len} - L_\text{soft})/(L_\text{max} - L_\text{soft})\)</span>。</p>

<p><strong>5. 动态采样（Dynamic Sampling）</strong>：训练中重采样 prompt，确保每个 batch 都有成功/失败的混合（TRL 中尚未实现）。</p>

<p><strong>适用场景</strong>：需要极大探索与超长补全（32K+ Token）的大规模推理 RL。非对称裁剪尤其有价值。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../08-preference-optimization/index.md">8</a> 章（GRPO；偏好优化变体）。</em></p>
</details></section>

<section class="quiz-question" id="q-22"><h3 class="question-title">Q22：解释 vLLM 的训练-推理不一致。为何发生，TIS/MIS 如何修复？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：当用 vLLM 做生成、用训练框架（DeepSpeed/FSDP）做更新时，同一模型、同一权重会产生<em>不同</em>的 Token 概率。原因：</p>

<ul>
<li><p>数值 kernel 不同（vLLM 使用针对吞吐优化的自定义 CUDA kernel）。</p></li>
<li><p>attention 实现不同（训练用 FlashAttention，vLLM 用 PagedAttention）。</p></li>
<li><p>精度处理不同（vLLM 用 FP8/INT8，训练用 BF16）。</p></li>
<li><p>批处理差异影响 layer normalization 的数值。</p></li>
</ul>

<p>这会悄无声息地破坏 PPO 的 on-policy 假设：我们计算 <span class="arithmatex">\(\pi_\theta/\pi_\text{old}\)</span> 时，<span class="arithmatex">\(\pi_\text{old}\)</span> 来自 vLLM 而 <span class="arithmatex">\(\pi_\theta\)</span> 来自训练框架。从第零步起比率就是错的！</p>

<p><strong>TIS（截断重要性采样，Truncated Importance Sampling）</strong>：通过乘以 <span class="arithmatex">\(\min(\pi_\text{train}/\pi_\text{inference}, C)\)</span> 修正梯度。<span class="arithmatex">\(\min\)</span> 与上限 <span class="arithmatex">\(C\)</span> 防止极端修正破坏训练稳定性。典型 <span class="arithmatex">\(C=2.0\)</span>。</p>

<p><strong>MIS（掩码重要性采样，Masked Importance Sampling）</strong>：更激进——直接丢弃任何 <span class="arithmatex">\(\pi_\text{train}/\pi_\text{inference} &gt; C\)</span> 的 Token，对梯度贡献为零。防止任何估计糟糕的 Token 影响更新。</p>

<p><strong>序列级 vs Token 级</strong>：序列级 IS 理论上正确（无偏）；Token 级 IS 有偏但方差更低。实践中，序列级带截断效果最好。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（GRPO；系统架构）。</em></p>
</details></section>

<section class="quiz-question" id="q-23"><h3 class="question-title">Q23：GSPO vs GRPO——根本区别在哪？什么时候重要？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>GRPO</strong>：<em>逐 Token</em>计算重要性比 <span class="arithmatex">\(w_{i,t} = \pi_\theta(o_{i,t}|q, o_{i,&lt;t}) / \pi_\text{old}(o_{i,t}|q, o_{i,&lt;t})\)</span>，然后独立裁剪每个 Token。</p>

<p><strong>GSPO</strong>：在<em>序列级</em>计算重要性比：<span class="arithmatex">\(s_i(\theta) = (\pi_\theta(o_i|q)/\pi_\text{old}(o_i|q))^{1/|o_i|}\)</span>——Token 概率的几何均值。裁剪这单一的序列级比率。</p>

<p><strong>为何重要</strong>：GRPO 的逐 Token 裁剪把每个 Token 视为独立，但语言中它们高度相关。序列前段的微小逐 Token 变化在多个 Token 上指数级放大。GSPO 通过审视完整序列概率来捕捉这一点。</p>

<p><strong>长度归一化</strong>：<span class="arithmatex">\(1/|o_i|\)</span> 指数保证不同长度序列间的公平比较。否则更长的序列总是有更低的概率比。</p>

<p><strong>何时用 GSPO</strong>：当训练变为 off-policy 时（<code>steps_per_generation &gt; 1</code> 或 <code>num_iterations &gt; 1</code>）。如果完全 on-policy（比率 <span class="arithmatex">\(\approx 1\)</span>），GRPO 与 GSPO 等价。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../08-preference-optimization/index.md">8</a> 章（GRPO；偏好优化变体）。</em></p>
</details></section>

<section class="quiz-question" id="q-24"><h3 class="question-title">Q24：论文 “It Takes Two” 表明 G=2 与 G=16 效果相当。这怎么可能？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：关键洞见是 GRPO 的有效性并非来自精确的 advantage 估计（那需要大 <span class="arithmatex">\(G\)</span>），而来自一个<strong>隐式的对比目标</strong>。</p>

<p><span class="arithmatex">\(G=2\)</span> 加二元 reward（一个对一个错）时：归一化后 <span class="arithmatex">\(\hat{A}_\text{correct} = +1\)</span>，<span class="arithmatex">\(\hat{A}_\text{wrong} = -1\)</span>。loss 变成：提升正确回答的概率、降低错误回答的概率。这本质上就是一个 DPO 风格的对比 loss！</p>

<p><strong>为什么大 <span class="arithmatex">\(G\)</span> 帮助不大</strong>：归一化 advantage <span class="arithmatex">\(\hat{A}_i = (r_i - \mu)/\sigma\)</span> 本身已制造了好坏之间的对比。更多样本能更准确估计 <span class="arithmatex">\(\mu\)</span>，但梯度方向由最好与最差之间的<em>对比</em>主导，而非 <span class="arithmatex">\(\mu\)</span> 的精度。</p>

<p><strong>算力节省</strong>：<span class="arithmatex">\(G=2\)</span> 意味着生成算力比 <span class="arithmatex">\(G=16\)</span> 少 8<span class="arithmatex">\(\times\)</span>。由于生成占训练时间 60%，整体训练加速 <span class="arithmatex">\(\sim\)</span>4<span class="arithmatex">\(\times\)</span>。</p>

<p><strong>注意</strong>：通过率在 30–70% 时效果最佳。如果通过率极低（<span class="arithmatex">\(&lt;\)</span>10%），<span class="arithmatex">\(G=2\)</span> 常常给出两个失败（无信号）。难题需要更大的 <span class="arithmatex">\(G\)</span>。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 章（GRPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-25"><h3 class="question-title">Q25：什么是 SAPO？它的软门控与硬裁剪有何不同？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：标准 PPO/GRPO 使用硬裁剪：<span class="arithmatex">\(\text{clip}(r, 1-\epsilon, 1+\epsilon)\)</span>。在边界处梯度突然降为零。这制造了一个“死区”，模型在其中收不到任何学习信号。</p>

<p><strong>SAPO</strong> 用平滑的 sigmoid 门控取代它：随着比率偏离 1，梯度被逐渐衰减，绝不会突兀归零。它使用非对称温度：</p>

<ul>
<li><p>对正 advantage 用 <span class="arithmatex">\(\tau_+ = 1.0\)</span>（标准衰减）。</p></li>
<li><p>对负 advantage 用 <span class="arithmatex">\(\tau_- = 1.05\)</span>（对抑制略激进一些的衰减）。</p></li>
</ul>

<p><strong>收益</strong>：(1) 梯度地形上没有“悬崖”。(2) 略超出裁剪范围的 Token 仍有贡献（衰减而非归零）。(3) 优化轨迹更稳定。(4) 序列连贯——考虑了完整序列上下文。</p>

<p><strong>权衡</strong>：信任域比硬裁剪略宽松，因此需要仔细调温度。但整体对超参更鲁棒。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../08-preference-optimization/index.md">8</a> 章（GRPO；偏好优化变体）。</em></p>
</details></section>

## 28.6　DPO 扩展题 {#dpo-extensions-questions}

<section class="quiz-question" id="q-26"><h3 class="question-title">Q26：比较 f-DPO 的散度选择。前向 KL、JS 与反向 KL 各自何时使用？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：标准 DPO 隐式使用反向 KL（<span class="arithmatex">\(D_\text{KL}[\pi_\theta \| \pi_\text{ref}]\)</span>）：</p>

<ul>
<li><p><strong>反向 KL</strong>（默认）：寻峰（mode-seeking）。policy 把概率集中在参考概率高的位置。避免生成参考不会生成的文本。利于安全（保守）。</p></li>
<li><p><strong>前向 KL</strong>：覆盖（mass-covering）。policy 试图覆盖参考的所有峰，甚至低概率的峰。利于多样性但可能生成低质量输出。</p></li>
<li><p><strong>Jensen-Shannon</strong>：前向与反向之间的对称折中。在覆盖与寻峰间取得平衡。对通用对齐通常最佳。</p></li>
<li><p><strong>Alpha 散度</strong>（<span class="arithmatex">\(\alpha=0.5\)</span>）：在前向（<span class="arithmatex">\(\alpha=0\)</span>）与反向（<span class="arithmatex">\(\alpha=1\)</span>）之间插值。可调。</p></li>
</ul>

<p><strong>实践建议</strong>：先用反向 KL（标准 DPO）。若模型过于保守（不愿尝试创造性方案），切换到 JS 散度。若多样性至关重要（创意写作、头脑风暴），试试前向 KL。</p>

<p><em><strong>复习：</strong> 第 <a href="../06-dpo/index.md">6</a> 与 <a href="../08-preference-optimization/index.md">8</a> 章（DPO；偏好优化变体）。</em></p>
</details></section>

<section class="quiz-question" id="q-27"><h3 class="question-title">Q27：你的 DPO 偏好数据有 15% 的标签噪声。怎么办？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：按复杂度递增的三个方案：</p>

<p><strong>1. Robust DPO</strong>（已知噪声率时最佳）：解析地去偏 loss：<span class="arithmatex">\(\mathcal{L}_\text{robust} = \frac{(1-\varepsilon)\mathcal{L}_\text{DPO}(y_w, y_l) - \varepsilon \mathcal{L}_\text{DPO}(y_l, y_w)}{1 - 2\varepsilon}\)</span>。设 <span class="arithmatex">\(\varepsilon = 0.15\)</span>。在期望意义上可证明恢复干净的 DPO 目标。TRL：<code>loss_type=&quot;robust&quot;, label_smoothing=0.15</code>。</p>

<p><strong>2. IPO（Identity Preference Optimization, IPO）</strong>（噪声率未知时最佳）：带目标 margin 的平方 loss。被误标的对影响有界（平方 loss 不发散）。对任意噪声模式更鲁棒，且无需知道 <span class="arithmatex">\(\varepsilon\)</span>。TRL：<code>loss_type=&quot;ipo&quot;</code>。</p>

<p><strong>3. TR-DPO</strong>（分布漂移时最佳）：训练中通过 EMA 更新参考模型。即便早期数据有噪声，演化的参考也能帮助模型自我修正。TRL：<code>sync_ref_model=True, ref_model_mixup_alpha=0.6</code>。</p>

<p><strong>数据侧修复</strong>：(1) 过滤标注者一致率 <span class="arithmatex">\(&lt;\)</span>70% 的对。(2) 用 RM 给对打分；丢弃 RM 与标签不一致的对。(3) 主动学习：重新标注最不确定的对。</p>

<p><em><strong>复习：</strong> 第 <a href="../06-dpo/index.md">6</a> 与 <a href="../08-preference-optimization/index.md">8</a> 章（DPO；偏好优化变体）。</em></p>
</details></section>

<section class="quiz-question" id="q-28"><h3 class="question-title">Q28：什么是 SimPO？为什么“无参考模型”是优势？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：SimPO 用回答的平均对数概率作为隐式 reward 信号：<span class="arithmatex">\(r(x,y) = \frac{1}{|y|}\sum_t \log \pi_\theta(y_t|x, y_{&lt;t})\)</span>——无需参考模型。</p>

<p>loss 中加入目标 margin <span class="arithmatex">\(\gamma\)</span>：chosen 回答的平均 log-prob 应至少比 rejected 高 <span class="arithmatex">\(\gamma\)</span>。</p>

<p><strong>为什么“无参考”重要</strong>：</p>

<ol>
<li><p><strong>显存</strong>：无参考模型 = 70B 节省 70–140GB。可在同样硬件上训练更大的模型。</p></li>
<li><p><strong>简洁性</strong>：无需管理/加载/服务第二份模型副本。</p></li>
<li><p><strong>无陈旧参考</strong>：DPO 的参考随着训练推进越来越不相关。SimPO 没有这个问题。</p></li>
<li><p><strong>内置长度归一化</strong>：<span class="arithmatex">\(1/|y|\)</span> 天然防止长度偏差（DPO 需显式处理）。</p></li>
</ol>

<p><strong>权衡</strong>：没有参考锚点，模型有更多自由去塌缩或漂移。<span class="arithmatex">\(\gamma\)</span> margin 和长度归一化部分缓解了这一点，但在激进训练时 SimPO 可能不如 DPO 稳定。</p>

<p><em><strong>复习：</strong> 第 <a href="../08-preference-optimization/index.md">8</a> 章（偏好优化变体）。</em></p>
</details></section>

<section class="quiz-question" id="q-29"><h3 class="question-title">Q29：解释 Iterative RPO。为何在推理任务中把 DPO 与 NLL loss 结合？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：用于推理的标准 DPO 有一种微妙的失效模式：它学会<em>判别</em>（给正确轨迹更高的隐式 reward），但不一定学会<em>生成</em>它们。</p>

<p><strong>为什么</strong>：DPO 的梯度推高 chosen 的概率、压低 rejected 的概率。但 chosen 回答可能与模型自身会生成的东西差异巨大，以至于提升其概率并不能教会模型产出类似的推理模式。</p>

<p><strong>RPO 的修复</strong>：在 chosen 回答上添加负对数似然（NLL/SFT）loss：<span class="arithmatex">\(\mathcal{L} = \mathcal{L}_\text{DPO} + \alpha \cdot \mathcal{L}_\text{NLL}(y_w)\)</span>。</p>

<p>NLL 项显式训练模型逐步生成获胜回答。DPO 项确保模型同时学会避免落败回答。结合起来：模型同时学到“如何正确推理”（NLL）和“该避免什么”（DPO）。</p>

<p><strong>迭代</strong>：生成回答 <span class="arithmatex">\(\rightarrow\)</span> 检查正确性 <span class="arithmatex">\(\rightarrow\)</span> 构造对 <span class="arithmatex">\(\rightarrow\)</span> 用 RPO 训练 <span class="arithmatex">\(\rightarrow\)</span> 重复。每次迭代，模型更擅长生成正确推理，为下一轮提供更高质量的训练数据。</p>

<p>TRL：<code>loss_type=[&quot;sigmoid&quot;, &quot;sft&quot;], loss_weights=[1.0, 1.0]</code></p>

<p><em><strong>复习：</strong> 第 <a href="../08-preference-optimization/index.md">8</a> 与 <a href="../13-reasoning/index.md">13</a> 章（偏好优化变体；大型推理模型的 RL）。</em></p>
</details></section>
