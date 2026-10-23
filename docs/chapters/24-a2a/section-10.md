# 24.10　小结 {#sec:a2a:summary}

<div class="admonition note"><p class="admonition-title">关键要点：Agent 到 Agent 通信</p>
<ol>
<li><p><strong>A2A 让规模化的专业化成为可能</strong>：通过将任务路由给专家 Agent，多 Agent 系统可同时获得深度与广度。（第 <a href="../25-multi-agent-systems/index.md#sec:multi-agent-systems">25</a> 章将深入介绍多 Agent 架构。）</p></li>
<li><p><strong>Google 的 A2A 协议</strong>为可互操作的 Agent 通信提供了一个面向生产的开放标准，涵盖 Agent Card、任务生命周期管理、SSE 流式传输和企业级认证。</p></li>
<li><p><strong>通信模式</strong>从简单的请求-响应，到复杂的协商和基于拍卖的分配——可根据任务复杂度和延迟需求来选择。</p></li>
<li><p><strong>A2A 与 MCP 互补</strong>：A2A 连接 Agent 与 Agent；MCP 连接 Agent 与 Tool。大多数生产系统两者并用。</p></li>
<li><p><strong>安全不可妥协</strong>：Agent 身份验证、授权 scope 和审计轨迹在任何多 Agent 部署中都不可或缺。</p></li>
<li><p><strong>协调协议</strong>（合同网、黑板、共识）在简单委派之外，提供了用于集体决策的结构化机制。</p></li>
<li><p><strong>基于关联 ID 的可观测性</strong>对于调试和审计跨越多个 Agent 与 Tool 的复杂多 Agent 工作流至关重要。</p></li>
</ol>
</div>

<div class="admonition question"><p class="admonition-title">A2A 中的开放研究问题</p>
<ul>
<li><p>Agent 应如何处理来自层级中多个编排器的<em>相互冲突的指令</em>？哪些冲突解决机制最为有效？</p></li>
<li><p>Agent 能否通过经验<em>学习</em>出更好的路由与委派策略，而不再依赖静态的能力声明？</p></li>
<li><p>如何防范<em>Prompt 注入攻击</em>——即恶意 Agent 通过在消息中嵌入对抗性指令来操纵下游 Agent？</p></li>
<li><p>上下文传递的合理<em>隐私边界</em>在哪里——子 Agent 应该看到多少对话历史，又如何在技术上强制执行这些边界？</p></li>
<li><p>当 Agent 网络扩展到成百上千个 Agent 时，如何在不形成瓶颈或破坏一致性的前提下维持<em>连贯的全局状态</em>？</p></li>
</ul>
</div>
