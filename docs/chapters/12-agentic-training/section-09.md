# 12.9　面向 Agent 训练的交互式 RL 环境 {#sec:interactive-rl-envs}

<p>早期智能体 RL 的主导范式依赖<strong>静态数据集</strong>：精选轨迹、离线偏好对或固定基准套件。这种方法虽然可行，但有一个根本局限——在静态数据上训练的 Agent 无法通过与环境的交互发现新策略，奖励信号也被限制在数据集创建时所能预见的范围内。该领域已决定性地转向<strong>交互式 RL 环境</strong>：Agent 在其中执行动作、接收反馈，并基于真实结果更新策略的实时模拟器。</p>

<div class="admonition note"><p class="admonition-title">静态 vs. 交互式训练</p>
<ul>
<li><p><strong>静态（SFT/DPO）</strong>：Agent 从固定轨迹中学习。没有探索。奖励隐含在数据策展之中。天花板：人类示范者的质量。</p></li>
<li><p><strong>交互式 RL</strong>：Agent 探索环境，获得可验证奖励。能够发现训练分布之外的策略。天花板：环境奖励的质量。</p></li>
</ul>

<p>这一转变呼应了机器人学中从模仿学习到 RL 的过渡——交互式环境解锁了质变级别的新能力。</p>
</div>

## 12.9.1　NeMo Gym（NVIDIA） {#subsec:nemo-gym}

<p><strong>NeMo Gym</strong> <a class="book-citation" id="cite-nemogym2025nvidia-1" href="../../references.md#ref-nemogym2025nvidia" aria-label="参考文献 271">[271]</a> 是 NVIDIA 专为 LLM Agent 训练设计的交互式 RL 环境框架。其架构干净地把<strong>Agent</strong>（LLM 策略）与<strong>环境</strong>（任务模拟器）分离，使两个组件可以独立扩展。</p>

<p>关键设计原则：</p>

<ul>
<li><p><strong>多轮 rollout</strong>：环境支持扩展的交互序列，而不只是单步查询。Agent 可以跨多轮调用工具、接收结果并继续推理。</p></li>
<li><p><strong>工具调用验证</strong>：奖励通过执行 Agent 的工具调用并与真值核对结果来计算——提供密集、可靠的信号。</p></li>
<li><p><strong>Agent/环境解耦</strong>：环境作为独立服务运行，允许异构算力分配（例如 Agent 用 GPU、环境模拟用 CPU）。</p></li>
</ul>

<p><strong>Nemotron 3 Super</strong> 使用 NeMo Gym 训练，配置了覆盖数学、代码、工具使用与多轮对话的 21 种环境，生成了 120 万次 rollout。环境的多样性至关重要：只在单一类型环境上训练的 Agent 容易过拟合该环境的奖励结构。</p>

## 12.9.2　RLFactory {#subsec:rlfactory}

<p><strong>RLFactory</strong> <a class="book-citation" id="cite-rlfactory2025-1" href="../../references.md#ref-rlfactory2025" aria-label="参考文献 369">[369]</a> 是一个面向多轮工具使用 RL 的即插即用框架，旨在把连接 LLM 策略与多样工具环境的工程开销降到最低。</p>

<div class="admonition example"><p class="admonition-title">RLFactory：Qwen3-4B 超越更大模型</p>
<p>使用 RLFactory 配合异步工具调用与多样化奖励信号：</p>

<ul>
<li><p>用 RLFactory 训练的 <strong>Qwen3-4B</strong> 在 Natural Questions 上超越了显著更大的模型。</p></li>
<li><p>通过异步执行，相对同步工具调用基线获得 <strong>6.8<span class="arithmatex">\(\times\)</span> 吞吐量</strong>提升。</p></li>
<li><p>多样化奖励（正确性、格式、工具效率）防止在任何一个单一信号上出现 reward hacking。</p></li>
</ul>

<p>关键洞见：具备高质量交互式训练的小模型可以胜过在静态数据上训练的大模型，因为交互式 RL 让模型能够发现任何人类示范者都想不到要写进数据集的工具使用策略。</p>
</div>

## 12.9.3　MOSAIC：智能体训练中的安全 {#subsec:mosaic-safety}

<p>随着智能体 RL 环境的能力越来越强，安全成为一等关注点。<strong>MOSAIC</strong> <a class="book-citation" id="cite-mosaic2026safety-1" href="../../references.md#ref-mosaic2026safety" aria-label="参考文献 50">[50]</a>（2026 年 3 月）解决的是在交互式环境中训练既强大又安全的 Agent 的问题。</p>

<p>MOSAIC 的核心框架引入了一个<strong>计划 <span class="arithmatex">\(\rightarrow\)</span> 检查 <span class="arithmatex">\(\rightarrow\)</span> 行动或拒绝</strong>的循环：</p>

<ol>
<li><p><strong>计划（Plan）</strong>：Agent 生成候选动作序列。</p></li>
<li><p><strong>检查（Check）</strong>：安全验证器依据约束集（危害类别、策略规则）评估该计划。</p></li>
<li><p><strong>行动或拒绝（Act or Refuse）</strong>：计划通过则执行；否则生成带解释的拒绝。</p></li>
</ol>

<p>安全通过<strong>轨迹级偏好学习</strong>来训练：MOSAIC 不是把单个动作标注为安全/不安全，而是标注整条轨迹，从而捕捉多步计划的累积安全影响。这一点至关重要，因为看似无害的单个动作组合起来可能形成有害序列。</p>

<div class="admonition tip"><p class="admonition-title">MOSAIC 结果</p>
<ul>
<li><p>在智能体安全基准上<strong>有害行为最多减少 50%</strong>。</p></li>
<li><p><strong>任务性能保持不变</strong>：计划—检查—行动循环只增加极小的延迟，不会降低良性任务上的成功率。</p></li>
<li><p>轨迹级偏好学习优于步骤级安全分类器，因为它能捕捉动作组合产生的涌现性危害。</p></li>
</ul>
</div>

#### 在真实 Agent 轨迹上训练。 {#heading-12-55}

<p>xAI 的 Grok 4.5（2026 年 7 月） <a class="book-citation" id="cite-grok45_2026-1" href="../../references.md#ref-grok45_2026" aria-label="参考文献 389">[389]</a> 引入了一种新颖的数据来源：来自 Cursor 的匿名开发者会话轨迹——工程师在真实代码库中工作的实际击键、导航模式与编辑序列。在这些轨迹上做中期训练后，Grok 4.5 在智能体工具使用基准上登顶。这预示着一个“算力 + 轨迹”范式：一旦算力到位，稀缺的输入就变成<em>目标行为的记录</em>——类似于 AlphaGo 先以人类棋局自举、再转向自我对弈。这一方法引发了尚未解决的数据权利问题（谁的代码、在何种同意下？），但它证明：来自专家用户的行为示范仍然是智能体能力的有力训练信号。</p>
