# 12.1　动机：从聊天机器人到自主智能体 {#motivation-from-chatbots-to-autonomous-agents}

<p>现代 LLM 越来越多地被部署，不仅作为对话助手，而且作为<strong>自主智能体（Autonomous Agents）</strong>，能够在多个步骤中与外部工具、API、数据库和环境进行交互。这种从单轮聊天机器人到多步智能体的转变，引入了根本性的新 RL 挑战，要求我们重新思考如何训练、评估和部署语言模型。</p>

<figure id="figure-12-1"><a href="../../assets/diagrams/figure-12-1.svg" target="_blank" rel="noopener" aria-label="查看图 12.1 原图" title="点击查看原图"><img src="../../assets/diagrams/figure-12-1.svg" alt="从聊天机器人到自主智能体：传统的 LLM 聊天机器人在带有即时人类反馈的单步对话循环中运行。自主智能体跨多个工具交互进行规划，从真实世界的执行环境中接收反馈，并针对稀疏的终端 reward（任务成功/失败）进行优化。" loading="lazy"></a><figcaption><span class="figure-number">图 12.1　</span>从聊天机器人到自主智能体：传统的 LLM 聊天机器人在带有即时人类反馈的单步对话循环中运行。自主智能体跨多个工具交互进行规划，从真实世界的执行环境中接收反馈，并针对稀疏的终端 reward（任务成功/失败）进行优化。</figcaption></figure>

<p>要求新 RL 方法的关键差异：</p>

<ul>
<li><p><strong>多步推理</strong>：Agent 必须跨越 10–100+ 次工具调用进行规划，而不仅仅是生成单一响应。</p></li>
<li><p><strong>外部环境反馈</strong>：Reward 来自真实世界的执行（测试套件通过、网页加载、代码编译），而不仅仅是人类偏好分数。</p></li>
<li><p><strong>结构化动作</strong>：动作不仅是 Token，而是结构化输出（JSON 工具调用、API 负载、代码块）。</p></li>
<li><p><strong>长视野与稀疏 reward</strong>：成功/失败可能只能在多个中间步骤之后才能确定。</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">为什么标准 RLHF 对 Agent 不够用</p>
<p>标准 RLHF（PPO/DPO）针对单轮质量进行优化：给定一个 Prompt，产生一个好的响应。但 Agent 必须：</p>

<ul>
<li><p>决定<em>何时</em>使用工具，还是进行内部推理</p></li>
<li><p>在轨迹中途从错误中恢复（自我纠正）</p></li>
<li><p>平衡探索（尝试新方法）与利用（使用已知良好模式）</p></li>
<li><p>处理部分可观测性（工具输出可能不完整或带噪声）</p></li>
</ul>

<p>这要求训练方法在<strong>完整轨迹</strong>上进行推理，而不是单个轮次。</p>
</div>
