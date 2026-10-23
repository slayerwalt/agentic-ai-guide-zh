# 21.1　动机：为什么 Agent 需要环境 {#sec:env-motivation}

<p>对话式语言模型的评估在原则上是直截了当的：给出一个 Prompt、收集一次回复，并依据参考答案或人工评判进行打分。Agent 评估则根本不同：Agent 必须在世界中<em>行动</em>、观察后果，并在一连串步骤中调整其行为。任何单次回复都无法刻画这一点；只有结构化的<em>环境</em>才能做到。</p>

<p><strong>范围说明。</strong> 我们在强化学习意义上使用<em>环境</em>一词：Agent 为训练或评估而与之交互的世界——而非在线服务时承载 Agent 的生产基础设施（harness、编排器）。执行沙箱出现在此处，是因为它们<em>使</em>这种环境成为可能，但 Agent harness 本身在第 18 章讨论。</p>

<div class="admonition note"><p class="admonition-title">Chatbot 与 Agent 评估的鸿沟</p>
<p><strong>Chatbot 评估</strong>衡量单次生成的质量：流畅度、事实性、有用性。<strong>Agent 评估</strong>衡量 <em>Policy</em> 的质量：Agent 能否在多样化的长时程任务上稳定达成目标？这一鸿沟不仅是量上的——它需要完全不同的基础设施。</p>
</div>

<p>三股力量驱动了对专用 Agent 环境的需求：</p>

#### 安全探索。 {#safe-exploration.}

<p>真实世界系统——生产数据库、在线网站、金融 API——无法承受训练中 Agent 的探索行为。沙箱环境提供一个忠实副本，Agent 可在其中失败、恢复、学习，且不造成不可逆的损害。安全隔离（如 Docker 容器、受限网络的虚拟机）不是可选项，而是一等设计要求。</p>

#### 可复现的评估。 {#reproducible-evaluation.}

<p>基准测试要求每个 Agent 在相同条件下面对相同任务。环境必须按需具备确定性、受版本控制并可分发，以便一个实验室报告的结果能在另一个实验室复现。该性质的缺失历来使 Agent 基准难以比较。</p>

#### 课程学习。 {#curriculum-learning.}

<p>从零开始在困难任务上训练 Agent 在样本效率上很低。提供<em>难度课程</em>的环境——随着 Agent 的进步逐步增加任务复杂度——可显著减少达到目标性能所需的环境交互次数。这与人类学习的方式相似：子技能的掌握先于整体的掌握。</p>

<div class="admonition tip"><p class="admonition-title">把环境视为 LLM 的 RL &quot;Gym&quot;</p>
<p>正如 OpenAI Gym <a class="book-citation" id="cite-brockman2016openai-1" href="../../references.md#ref-brockman2016openai" aria-label="参考文献 31">[31]</a> 标准化了 RL 算法与模拟控制任务之间的接口，Agent 环境标准化了基于 LLM 的 Agent 与其必须解决的各类任务之间的接口。这一类比相当贴切：<code>reset()</code> 初始化新的 Episode，<code>step(action)</code> 推进世界状态并返回观察和奖励，<code>render()</code> 产生当前状态的人类可读视图。</p>
</div>
