# 29.19　智能体框架对比 {#agent-framework-comparison}

<span id="table-29-inline-14"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>框架</strong></th><th scope="col"><strong>编排方式</strong></th><th scope="col"><strong>多 Agent</strong></th><th scope="col"><strong>最适用于</strong></th></tr></thead>
<tbody><tr><td>LangGraph</td><td>显式状态图</td><td>条件路由</td><td>生产环境：持久化、HITL、精细控制</td></tr>
<tr><td>OpenAI Agents SDK</td><td>声明式 handoff</td><td>基于 handoff</td><td>简单易用：护栏、追踪、快速上手</td></tr>
<tr><td>AutoGen (AG2)</td><td>对话驱动</td><td>GroupChat</td><td>原型开发：代码执行、研究场景</td></tr>
<tr><td>CrewAI</td><td>基于角色的团队</td><td>顺序 / 并行</td><td>低代码：快速演示、简单流水线</td></tr>
<tr><td>Google ADK</td><td>会话 + 事件</td><td>原生 A2A</td><td>企业级：artifact 管理、多模态</td></tr></tbody>
</table>
</div>
