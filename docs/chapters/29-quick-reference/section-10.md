# 29.10　智能体通信协议 {#agent-communication-protocols}

<span id="table-29-inline-7"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>协议</strong></th><th scope="col"><strong>范围</strong></th><th scope="col"><strong>传输方式</strong></th><th scope="col"><strong>核心概念</strong></th></tr></thead>
<tbody><tr><td>MCP</td><td>工具集成</td><td>stdio / HTTP+SSE</td><td>服务端暴露工具；客户端发现并调用</td></tr>
<tr><td>A2A</td><td>智能体到智能体</td><td>HTTP + JSON-RPC</td><td>带生命周期的任务（submitted<span class="arithmatex">\(\to\)</span>working<span class="arithmatex">\(\to\)</span>done）</td></tr>
<tr><td>OpenAI Function Calling</td><td>工具使用</td><td>API 负载</td><td>在 <code>tools[]</code> 数组中以 JSON schema 描述</td></tr></tbody>
</table>
</div>
