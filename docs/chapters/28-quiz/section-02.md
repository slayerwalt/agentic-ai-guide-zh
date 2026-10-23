# 28.2　核心算法题 {#core-algorithm-questions}

<section class="quiz-question" id="q-1"><h2 class="question-title">Q1：解释 PPO 的裁剪目标。为什么它比朴素 policy gradient 更有效？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：朴素 policy gradient：<span class="arithmatex">\(\nabla J = \mathbb{E}[\nabla\log\pi(a|s) \cdot \hat{A}]\)</span>。问题：一个幸运/不幸的样本就能产生巨大的梯度 <span class="arithmatex">\(\rightarrow\)</span> policy 跳到糟糕区域 <span class="arithmatex">\(\rightarrow\)</span> 生成乱码 <span class="arithmatex">\(\rightarrow\)</span> 下一个梯度让情况更糟 <span class="arithmatex">\(\rightarrow\)</span> 进入无法挽回的“死亡螺旋”。</p>

<p><strong>PPO 的方案</strong>：将概率比 <span class="arithmatex">\(r = \pi_\text{new}/\pi_\text{old}\)</span> 裁剪到 <span class="arithmatex">\([0.8, 1.2]\)</span>。</p>

<p><strong>机制</strong>：对好的动作（<span class="arithmatex">\(\hat{A}&gt;0\)</span>）：目标为 <span class="arithmatex">\(\min(r\hat{A}, 1.2\hat{A})\)</span>。一旦 <span class="arithmatex">\(r\)</span> 超过 1.2，便不再有额外收益——防止 policy 在单个样本上过度承诺。对坏的动作（<span class="arithmatex">\(\hat{A}&lt;0\)</span>）：目标为 <span class="arithmatex">\(\min(r\hat{A}, 0.8\hat{A})\)</span>。一旦 <span class="arithmatex">\(r\)</span> 降至 0.8 以下，惩罚便不再增长——防止灾难性遗忘（Catastrophic Forgetting）。</p>

<p><strong>关键洞见</strong>：它是对 TRPO 的 KL 约束的一阶近似，但无需昂贵的二阶优化。每次更新对 policy 的改变最多 <span class="arithmatex">\(\pm\)</span>20%。</p>

<p><strong>对 LLM 而言</strong>：Token 级比率 <span class="arithmatex">\(r_t = \pi_\theta(y_t|y_{&lt;t})/\pi_\text{old}(y_t|y_{&lt;t})\)</span> 防止任何单个 Token 的概率发生过大变化，从而维持生成的连贯性。</p>

<p><em><strong>复习：</strong> 第 <a href="../05-ppo/index.md">5</a> 章（PPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-2"><h2 class="question-title">Q2：从第一原理推导 DPO。它做了哪些假设？</h2>
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

<section class="quiz-question" id="q-3"><h2 class="question-title">Q3：GRPO vs PPO——你会在什么场景下选择哪一个？权衡何在？</h2>
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

<section class="quiz-question" id="q-4"><h2 class="question-title">Q4：GAE 如何工作？为 LLM 走一个具体例子。</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：GAE = <span class="arithmatex">\(n\)</span>-步 TD 误差的加权和：<span class="arithmatex">\(\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}\)</span>。</p>

<p><strong>具体例子</strong>：回答有 5 个 Token。仅在末端有 reward（<span class="arithmatex">\(r_5 = 0.8\)</span>）。Value 预测：<span class="arithmatex">\(V_1=0.5, V_2=0.55, V_3=0.6, V_4=0.65, V_5=0.7\)</span>。</p>

<p>TD 误差（<span class="arithmatex">\(\gamma=1\)</span>）：<span class="arithmatex">\(\delta_1 = 0 + V_2 - V_1 = 0.05\)</span>，<span class="arithmatex">\(\delta_2 = 0 + V_3 - V_2 = 0.05\)</span>，……，<span class="arithmatex">\(\delta_5 = 0.8 + 0 - 0.7 = 0.1\)</span>。</p>

<p>取 <span class="arithmatex">\(\lambda = 0.95\)</span>：<span class="arithmatex">\(\hat{A}_5 = 0.1\)</span>（仅是最后的 TD 误差），<span class="arithmatex">\(\hat{A}_4 = 0.05 + 0.95 \times 0.1 = 0.145\)</span>，<span class="arithmatex">\(\hat{A}_3 = 0.05 + 0.95 \times 0.145 = 0.188\)</span>，依此类推。</p>

<p><strong>解读</strong>：Token 3 获得 advantage 0.188，因为它对一个得到高于预期 reward 的序列做出了贡献。更早的 Token 通过指数衰减获得信用。</p>

<p><strong>对 LLM</strong>：<span class="arithmatex">\(\gamma=1.0\)</span>（所有 Token 都重要，且为有限视野）。Token <span class="arithmatex">\(t\)</span> 的 advantage 回答：“鉴于此 Token 之后发生的事，这个 Token 的选择比预期更好还是更差？”</p>

<p><em><strong>复习：</strong> 第 <a href="../05-ppo/index.md">5</a> 章（PPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-5"><h2 class="question-title">Q5：如何防止 reward hacking？给出一种分层防御策略。</h2>
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
