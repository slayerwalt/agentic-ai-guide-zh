# 27.7　人机协同 UI 设计 {#subsec:hitl-design}

<p>人机协同（Human-in-the-Loop, HITL）交互是 Agent 化 UI 中最关键的设计挑战之一。目标是在维持有效人类监督的同时，避免形成抹杀自动化效率收益的瓶颈。</p>

## 27.7.1　何时打断 Agent {#when-to-interrupt-the-agent}

<p>并非所有 Agent 操作都需要人类审查。一项有原则的打断策略应考虑：</p>

<ul>
<li><p><strong>可逆性</strong>：不可逆操作（删除文件、发送邮件、进行采购）始终需要批准。可逆操作（读取文件、网页搜索）一般不需要。</p></li>
<li><p><strong>影响范围</strong>：影响外部系统或他人的操作比纯本地操作更需谨慎审查。</p></li>
<li><p><strong>置信度</strong>：当 Agent 对其对用户意图的解读置信度较低时，应请求澄清而非径直执行。</p></li>
<li><p><strong>成本</strong>：高成本操作（大规模 API 调用、昂贵计算）需要批准。</p></li>
<li><p><strong>新颖性</strong>：Agent 在此上下文中此前未执行过的操作，比例行操作更需审查。</p></li>
</ul>

## 27.7.2　分层审批工作流 {#tiered-approval-workflows}

<p>分层审批策略可在监督与效率之间取得平衡：</p>

<div class="admonition note"><p class="admonition-title">三层审批模型</p>
<p><strong>Tier 1（自动批准）：</strong>低风险、可逆、例行操作。例如：网页搜索、读取文件、调用只读 API。Agent 无须打断即可执行；操作被记录以便审计。</p>

<p><strong>Tier 2（通知）：</strong>中风险操作。UI 显示非阻塞通知（“Agent 已向您的草稿箱发送一封草稿邮件”），用户可异步审阅。在短窗口期（例如 30 秒）内允许在操作最终生效前取消。</p>

<p><strong>Tier 3（要求批准）：</strong>高风险、不可逆或高成本操作。Agent 暂停并呈现阻塞式审批门控。用户必须显式批准、拒绝或修改，Agent 才会继续。</p>
</div>

<p>各层之间的阈值可由用户配置（“发送邮件前总是询问”），也可从用户行为中学得（如果用户总是批准网页搜索，则将来自动批准它们）。</p>

## 27.7.3　反馈机制 {#feedback-mechanisms}

<p>除了审批门控，Agent 化 UI 还应提供丰富的反馈机制，帮助 Agent 随时间改进：</p>

<ul>
<li><p><strong>点赞/踩</strong>：对响应的简单二元反馈，被存储并用于基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）微调或偏好学习。</p></li>
<li><p><strong>内联纠正</strong>：用户可直接编辑 Agent 的输出；原始输出与纠正输出之间的差异即是一个训练信号。</p></li>
<li><p><strong>偏好选择</strong>：当 Agent 提供多个选项时，用户的选择即偏好信号。</p></li>
<li><p><strong>显式指令</strong>：“下次别这么做”、“X 之前总是询问”、“偏好方法 Y 而非 Z”——更新 Agent 行为策略的自然语言指令。</p></li>
<li><p><strong>带理由的评分</strong>：评分附带可选的自由文本解释，提供比二元反馈更丰富的信号。</p></li>
</ul>

## 27.7.4　通过 UI 交互教会 Agent {#teaching-the-agent-through-ui-interaction}

<p>最成熟的 HITL UI 将每次交互视为教学机会：</p>

<ul>
<li><p><strong>示范</strong>：用户手动执行任务；Agent 观察并学习首选方法。</p></li>
<li><p><strong>带泛化的纠正</strong>：当用户纠正某个 Agent 操作时，UI 询问“我以后是否总是这样改？”，以将纠正泛化。</p></li>
<li><p><strong>偏好引出</strong>：周期性提示用户对比两种 Agent 行为并指出更偏好哪种。</p></li>
<li><p><strong>行为配置文件</strong>：UI 维护一份可见的“偏好”档案，用户可审阅与编辑，使 Agent 习得的行为透明可控。</p></li>
</ul>
