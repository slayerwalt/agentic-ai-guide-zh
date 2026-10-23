# 22.3　核心原语 {#sec:mcp:primitives}

<p>MCP 定义了四种核心原语，server 可将其暴露给 client。每种原语都有其独特的用途、控制方向与使用场景。</p>

## 22.3.1　Tools {#tools}

<p><strong>Tools</strong> 是最重要的原语——它们是 server 暴露给 LLM 调用的、类似函数的操作。一个 tool 具有：</p>

<ul>
<li><p>一个 <strong>name</strong>（在 server 内唯一的标识符）</p></li>
<li><p>一个 <strong>description</strong>（面向 LLM 的自然语言说明）</p></li>
<li><p>一个 <strong>inputSchema</strong>（用 JSON Schema 定义参数）</p></li>
<li><p>一个可选的 <strong>outputSchema</strong>（用 JSON Schema 定义返回值）</p></li>
</ul>

<p>Tools 表示<em>带有副作用的动作</em>：创建文件、发送消息、执行代码、查询数据库。LLM 决定何时以及如何调用 tool；server 负责执行。</p>

## 22.3.2　Resources {#resources}

<p><strong>Resources</strong> 是 server 可提供给 client 的数据。与由 LLM 调用的 tool 不同，resources 通常是<em>由 Host 应用读取</em>并用于填充 LLM 的上下文窗口。Resources 拥有 URI（例如 <code>file:///home/user/notes.txt</code>、<code>db://customers/42</code>），可以是静态的，也可以是动态的。</p>

<p>Resources 支持<strong>订阅</strong>：client 可以订阅某个 resource URI，并在底层数据变化时收到通知。这使得 Agent 可以以响应式方式对真实世界的事件做出反应。</p>

## 22.3.3　Prompts {#prompts}

<p><strong>Prompts</strong> 是 server 提供的可复用 prompt 模板。它们允许 server 作者将领域专长编码为结构化 prompt，Host 既可将其展示给用户，也可注入到对话中。例如，一个 GitHub MCP server 可能提供一个“代码评审”的 prompt 模板，接受一个 PR 编号作为输入，并生成结构化的评审请求。</p>

## 22.3.4　Sampling {#sampling}

<p><strong>Sampling</strong> 是最特殊的原语——它的方向是<em>反过来的</em>。不再是 client 请求 server 做某件事，而是<em>server 请求 client 执行一次 LLM 推理</em>。这种反向流动让工具 server 可以引入模型驱动的推理步骤（例如在返回数据前先对检索结果做摘要），而无需自行部署 LLM。Host 始终保留是否响应 sampling 请求的完全控制权，从而维持安全边界。</p>

<div class="admonition note"><p class="admonition-title">MCP 原语对比</p>
<span id="table-22-inline-2"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>原语</strong></th><th scope="col"><strong>方向</strong></th><th scope="col"><strong>使用场景</strong></th><th scope="col"><strong>示例</strong></th></tr></thead>
<tbody><tr><td><strong>Tools</strong></td><td>Client <span class="arithmatex">\(\to\)</span> Server</td><td>由 LLM 调用的带副作用动作</td><td><code>create_file</code>、<code>send_email</code>、<code>run_query</code></td></tr>
<tr><td><strong>Resources</strong></td><td>Client <span class="arithmatex">\(\leftarrow\)</span> Server</td><td>用于 LLM 上下文窗口的上下文数据</td><td>文件内容、数据库记录、API 响应</td></tr>
<tr><td><strong>Prompts</strong></td><td>Client <span class="arithmatex">\(\leftarrow\)</span> Server</td><td>可复用的 prompt 模板</td><td>“Summarize PR #id”、“Debug this error”</td></tr>
<tr><td><strong>Sampling</strong></td><td>Server <span class="arithmatex">\(\to\)</span> Client</td><td>Server 请求 LLM 推理</td><td>智能体子任务、递归推理</td></tr></tbody>
</table>
</div>
</div>
