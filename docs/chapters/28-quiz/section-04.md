# 28.4　实战与调试题 {#practical-and-debugging-questions}

<section class="quiz-question" id="q-11"><h2 class="question-title">Q11：reward 分数上升但模型质量下降。诊断并修复。</h2>
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

<section class="quiz-question" id="q-12"><h2 class="question-title">Q12：如何决定 RL 训练的 prompt 分布？</h2>
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

<section class="quiz-question" id="q-13"><h2 class="question-title">Q13：RLHF 中 LoRA vs 全量微调。各自何时使用？</h2>
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

<section class="quiz-question" id="q-14"><h2 class="question-title">Q14：过程奖励模型（Process Reward Model, PRM）vs 结果奖励模型（Outcome Reward Model, ORM）。设计一个 PRM 系统。</h2>
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

<section class="quiz-question" id="q-15"><h2 class="question-title">Q15：如何评估 RL 是否真的改进了模型？</h2>
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

<section class="quiz-question" id="q-16"><h2 class="question-title">Q16：端到端描述 reward 模型训练流水线。</h2>
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

<section class="quiz-question" id="q-17"><h2 class="question-title">Q17：KL 散度爆炸时会发生什么？根因与修复。</h2>
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

<section class="quiz-question" id="q-18"><h2 class="question-title">Q18：比较单体 vs 解耦的 RLHF 架构。各自在什么场景合理？</h2>
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

<section class="quiz-question" id="q-19"><h2 class="question-title">Q19：如何为 RL 训练设置课程学习？</h2>
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

<section class="quiz-question" id="q-20"><h2 class="question-title">Q20：你有 64 张 A100-80GB GPU 的预算，需要 RL 训练一个 70B 模型。设计资源分配。</h2>
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
