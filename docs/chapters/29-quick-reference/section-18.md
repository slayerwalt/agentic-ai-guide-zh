# 29.18　A2A 协议速查 {#a2a-protocol-quick-reference}

<span id="table-29-inline-13"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>概念</strong></th><th scope="col"><strong>描述</strong></th></tr></thead>
<tbody><tr><td>Agent Card</td><td>位于 <code>/.well-known/agent.json</code> 的 JSON — 名称、技能、支持的内容类型</td></tr>
<tr><td>Task</td><td>工作单元：<code>id</code>、<code>status</code>、<code>artifacts</code>。生命周期：submitted <span class="arithmatex">\(\to\)</span> working <span class="arithmatex">\(\to\)</span> completed/failed</td></tr>
<tr><td>Message</td><td>任务内的通信单元（角色：user/agent；parts：text/file/data）</td></tr>
<tr><td>Artifact</td><td>Agent 产出的输出（结构化数据、文件、生成内容）</td></tr>
<tr><td>Push Notifications</td><td>长任务的 webhook 更新（通过 <code>tasks/pushNotification/set</code>）</td></tr></tbody>
</table>
</div>

<p><strong>关键端点</strong>：<code>tasks/send</code>（创建/更新）、<code>tasks/get</code>（轮询状态）、<code>tasks/sendSubscribe</code>（SSE 流）。</p>
