# 29.17　MCP 速查 {#mcp-quick-reference}

<span id="table-29-inline-12"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>原语</strong></th><th scope="col"><strong>方向</strong></th><th scope="col"><strong>有副作用？</strong></th><th scope="col"><strong>用途</strong></th></tr></thead>
<tbody><tr><td>Tools</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务端</td><td>是</td><td>执行动作（创建、修改、删除）</td></tr>
<tr><td>Resources</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务端</td><td>否（只读）</td><td>读取数据（文件、DB 记录、配置）</td></tr>
<tr><td>Prompts</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务端</td><td>否</td><td>常用任务的可复用模板</td></tr>
<tr><td>Sampling</td><td>服务端 <span class="arithmatex">\(\to\)</span> 客户端</td><td>否</td><td>服务端请求客户端调用 LLM 生成</td></tr></tbody>
</table>
</div>

<p><strong>传输方式</strong>：<code>stdio</code>（本地子进程）或 <code>HTTP+SSE</code>（远程、可流式）。<br><strong>发现机制</strong>：客户端在连接初始化时调用 <code>tools/list</code>、<code>resources/list</code>、<code>prompts/list</code>。<br><strong>工具注解</strong>：<code>readOnlyHint</code>、<code>destructiveHint</code>、<code>idempotentHint</code>、<code>openWorldHint</code>。</p>
