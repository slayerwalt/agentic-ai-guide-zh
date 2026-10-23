# 第 4 章 · 大语言模型的强化学习基础 {#rl-foundations-for-language-models}

<p>监督微调（Supervised Fine-Tuning, SFT）教模型模仿示例，但模仿存在天花板：模型永远无法超越其训练数据的质量。强化学习突破了这一壁垒。通过生成新文本、接收 reward 反馈，并朝着获得更高 reward 的行为更新，经过 RL 训练的模型能够<em>发现</em>任何人类示范者都未曾写出的策略——产出更有帮助、更准确、且更契合人类偏好的输出 <a class="book-citation" id="cite-ouyang2022training-1" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a>。</p>

<p>这是每一款前沿模型背后的机制：GPT-4 <a class="book-citation" id="cite-openai2023gpt4-1" href="../../references.md#ref-openai2023gpt4" aria-label="参考文献 274">[274]</a>、Claude、Llama-3 <a class="book-citation" id="cite-grattafiori2024llama3-1" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a> 和 DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 都在 SFT 之后施加 RL，作为把一个能力强但缺乏导向的模型转化为对齐助手的关键一步。</p>

## 4.1　大语言模型 RL 的两大范式 {#two-paradigms-rl-llms}

<p>面向语言模型的 RL 方法可大致划分为两类范式，分别适用于不同的目标：</p>

##### 范式一：通过人类偏好实现对齐（RLHF/DPO）。 {#heading-4-3}

<p>将 RL 应用于大语言模型最初的动机是<strong>对齐</strong>——让模型变得有帮助、无害且诚实。<strong>基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）</strong> <a class="book-citation" id="cite-ouyang2022training-1--section-01" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a><a class="book-citation" id="cite-ziegler2019fine-1" href="../../references.md#ref-ziegler2019fine" aria-label="参考文献 442">[442]</a><a class="book-citation" id="cite-christiano2017deep-1" href="../../references.md#ref-christiano2017deep" aria-label="参考文献 57">[57]</a> 利用人类两两比较的判断（“哪个回答更好？”）来训练 reward 模型，然后优化 policy 以最大化这一学到的 reward。<strong>DPO</strong> <a class="book-citation" id="cite-rafailov2023direct-1" href="../../references.md#ref-rafailov2023direct" aria-label="参考文献 302">[302]</a> 通过彻底取消 reward 模型来简化这一过程，将偏好直接转化为有监督 loss。两种方法都能产出能够遵循指令且尊重安全约束的对齐助手。</p>

##### 范式二：通过可验证奖励增强能力（RLVR）。 {#heading-4-4}

<p>近期，RL 不仅被用于对齐，也被用于<strong>教授新能力</strong>——尤其是推理、数学和代码生成。此处的 reward 不再来自人类偏好，而来自<strong>可验证的结果</strong>：模型是否给出了正确答案？代码是否通过了所有测试？DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1--section-01" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 证明，配合基于规则的 reward（格式正确性 + 答案准确性）的 GRPO 能够在<em>完全不使用人类偏好数据</em>的情况下，训练模型发展出复杂的思维链（Chain-of-Thought，CoT）推理。这一范式——可验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards, RLVR）——如今已成为构建推理模型和 agentic 系统的主流路线。</p>

<div class="admonition note"><p class="admonition-title">共同的底层基础</p>
<p>尽管两类范式的目标不同，但它们共享同一套核心机制：</p>

<ul>
<li><p>一个自回归生成文本的 <strong>policy</strong> <span class="arithmatex">\(\pi_\theta\)</span>（即大语言模型）</p></li>
<li><p>一个 <strong>reward 信号</strong> <span class="arithmatex">\(r(x, y)\)</span>（来自偏好学习或来自验证计算）</p></li>
<li><p>相对于参考 policy 的 <strong>KL 约束</strong>，用于防止退化解</p></li>
<li><p>用于将模型朝更高 reward 方向更新的 <strong>policy gradient 优化</strong>（PPO 或 GRPO）</p></li>
</ul>

<p>本部分的各章将逐一展开每个组件的细节。</p>
</div>

## 4.2　将文本生成建模为 MDP {#text-generation-as-mdp}

<p>让 RL 得以应用于语言模型的关键洞察，是将自回归生成重述为一个 Markov 决策过程（MDP）：</p>

<div class="admonition tip"><p class="admonition-title">“LLM 即 Agent”的类比</p>
<p>把大语言模型想象成一个 <strong>Agent</strong>，正在一个 token 接一个 token 地撰写回答。在每一步，它查看到目前为止写下的全部内容（<em>state</em>）、选择下一个词（<em>action</em>），页面随之增长一个 token（<em>transition</em>）。当回答完成时，一位评判者对其打分（<em>reward</em>）。目标是：学到一种持续获得高分的写作策略（<em>policy</em>）。</p>
</div>

<p>形式化地，文本生成的 MDP 定义如下：</p>

<ul>
<li><p><strong>State</strong> <span class="arithmatex">\(s_t = (x, y_1, \ldots, y_{t-1})\)</span>：Prompt 与到目前为止已生成的所有 token 的拼接。</p></li>
<li><p><strong>Action</strong> <span class="arithmatex">\(a_t \in \{1, \ldots, |\mathcal{V}|\}\)</span>：从词表（32K–128K 选项）中选出下一个 token。</p></li>
<li><p><strong>Transition</strong> <span class="arithmatex">\(P(s_{t+1}|s_t, a_t)\)</span>：确定性的——只需追加所选 token。环境无随机性。</p></li>
<li><p><strong>Reward</strong> <span class="arithmatex">\(r\)</span>：通常仅在生成结束时给出（稀疏）。对 RLHF 而言是 reward 模型评分；对 RLVR 而言是最终答案的正确性。</p></li>
<li><p><strong>Policy</strong> <span class="arithmatex">\(\pi_\theta(a_t|s_t)\)</span>：大语言模型的下一 token 概率分布——正是 Softmax 输出已经计算出的东西。</p></li>
<li><p><strong>折扣因子</strong> <span class="arithmatex">\(\gamma = 1.0\)</span>：Episode 是有限的（即一条回答），因此无需折扣。</p></li>
</ul>

<p>这种映射之所以强大，是因为大语言模型<em>本身就已经是</em>一个 policy——其 Softmax 输出为每一个 state 定义了 <span class="arithmatex">\(\pi_\theta(a_t|s_t)\)</span>。我们无需另行构建一个 policy 网络；只需调整权重 <span class="arithmatex">\(\theta\)</span>，让模型为能获得更高 reward 的 token 序列赋予更高概率。</p>

## 4.3　RLHF 流水线 {#the-rlhf-pipeline}

<p>经典的 RLHF 流水线 <a class="book-citation" id="cite-ouyang2022training-1--section-03" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a> 包含四个阶段：</p>

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

## 4.4　本部分路线图 {#what-this-part-covers}

<p>后续章节将构建出完整的“面向大语言模型的 RL”工具箱：</p>

<ol>
<li><p><strong>PPO</strong>（第 5 章）——clipped surrogate 目标、用于优势估计的 GAE、critic 网络，以及完整的 RLHF 训练循环。它是 GPT-4 与 Claude 背后的主力。</p></li>
<li><p><strong>DPO</strong>（第 6 章）——通过将偏好转化为对比式有监督 loss，从而完全绕开 RL。比 online RL 更简单，但灵活性较低。</p></li>
<li><p><strong>GRPO</strong>（第 7 章）——DeepSeek 提出的无 critic 算法，使用组级 reward 归一化。它是 DeepSeek-R1 背后的方法，也是推理模型训练的主导选择。</p></li>
<li><p><strong>偏好优化变体</strong>（第 8 章）——Online DPO、KTO、Best-of-N 采样（Rejection Sampling），以及方法选型指南。</p></li>
<li><p><strong>Reward 建模</strong>（第 9 章）——Bradley-Terry 模型、过程 reward 与结果 reward 的对比、面向 RLVR 的基于规则的 reward，以及多目标组合。</p></li>
<li><p><strong>SFT 最佳实践</strong>（第 10 章）——序列打包（Sequence Packing）、对话模板、数据混合，以及 SFT 质量如何决定 RL 的上限。</p></li>
<li><p><strong>系统工程</strong>（第 11 章）——大规模分布式训练：并行策略、生成与训练解耦，以及面向数百块 GPU 的基础设施。</p></li>
</ol>
