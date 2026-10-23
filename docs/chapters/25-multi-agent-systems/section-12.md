# 25.12　小结 {#subsec:mas-summary}

<p>多智能体系统代表着我们部署 LLM 方式的根本性转变：从孤立的助手转向由专门化 Agent 构成的协作社会。本节的关键洞见：</p>

<div class="admonition note"><p class="admonition-title">多智能体系统：要点回顾</p>
<ol>
<li><p><strong>架构很重要</strong>：Agent 连接的拓扑决定了可扩展性、可调试性与容错性。应根据任务结构与运维需求做出选择。</p></li>
<li><p><strong>协调是有代价的</strong>：每条 Agent 间消息都消耗 Token。设计通信协议时应在保持必要信息流的同时最小化开销。</p></li>
<li><p><strong>专门化带来质量</strong>：在复杂任务上，具有专注角色与定制 Prompt 的 Agent 始终胜过通才 Agent。</p></li>
<li><p><strong>RL 训练很难</strong>：多 Agent RL 引入非平稳性、功劳分配挑战与涌现行为。CTDE 是目前合作场景下的最佳实践。</p></li>
<li><p><strong>安全需要显式设计</strong>：多智能体系统会放大错误并展现意外的涌现行为。安全监控必须作为一等架构关切。</p></li>
<li><p><strong>从简开始</strong>：从集中式 Supervisor 模式起步，衡量其局限，仅在必要时才向更复杂的架构演进。</p></li>
</ol>
</div>

<p>多 Agent LLM 系统领域正在快速演进。本文所述的模式与技术代表了当前的最先进水平，但新的架构、协调机制与训练算法正在不断涌现。无论具体实现如何演变，这些基础原则——专门化、协调、涌现行为，以及效率与鲁棒性之间的张力——都将持续保持其相关性。</p>
