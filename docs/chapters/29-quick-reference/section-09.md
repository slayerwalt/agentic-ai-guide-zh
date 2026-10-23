# 29.9　智能体设计模式 {#agentic-design-patterns}

<span id="table-29-inline-6"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模式</strong></th><th scope="col"><strong>结构</strong></th><th scope="col"><strong>最适用于</strong></th></tr></thead>
<tbody><tr><td>ReAct</td><td>Think <span class="arithmatex">\(\to\)</span> Act <span class="arithmatex">\(\to\)</span> Observe <span class="arithmatex">\(\to\)</span> 循环</td><td>通用工具使用型 Agent</td></tr>
<tr><td>Plan-and-Execute</td><td>Plan <span class="arithmatex">\(\to\)</span> 执行步骤 <span class="arithmatex">\(\to\)</span> 修订</td><td>长程、结构化任务</td></tr>
<tr><td>Supervisor</td><td>路由器 <span class="arithmatex">\(\to\)</span> 专家 Agent</td><td>多领域、子任务边界清晰</td></tr>
<tr><td>Swarm (handoffs)</td><td>Agent 移交控制权 + 上下文</td><td>客服、问题升级流程</td></tr>
<tr><td>Hierarchical</td><td>委派 Agent 的树状结构</td><td>复杂任务分解</td></tr>
<tr><td>Human-in-the-loop</td><td>Agent <span class="arithmatex">\(\to\)</span> 审批关卡 <span class="arithmatex">\(\to\)</span> 继续</td><td>高风险、不可逆动作</td></tr></tbody>
</table>
</div>
