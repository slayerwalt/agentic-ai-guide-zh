# 22.8　MCP 生态 {#sec:mcp:ecosystem}

<p>自发布以来，MCP 吸引了一个快速壮大的 server、client 与配套工具生态<sup aria-label="原文注号 2">2</sup>。</p>

## 22.8.1　常用 MCP Server {#popular-mcp-servers}

<p class="table-caption" id="tab:mcp-notable-servers"><span class="table-number">表 22.1　</span>有代表性的 MCP server（官方与社区）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>Server</strong></th><th scope="col"><strong>类别</strong></th><th scope="col"><strong>关键能力</strong></th></tr></thead>
<tbody><tr><td><code>server-filesystem</code></td><td>本地 I/O</td><td>读写文件、目录列表、搜索</td></tr>
<tr><td><code>server-github</code></td><td>版本控制</td><td>Issue、PR、提交、代码搜索、文件访问</td></tr>
<tr><td><code>server-postgres</code></td><td>数据库</td><td>只读 SQL 查询、schema 检查</td></tr>
<tr><td><code>server-sqlite</code></td><td>数据库</td><td>完整 SQLite 访问、schema 管理</td></tr>
<tr><td><code>server-brave-search</code></td><td>Web</td><td>通过 Brave API 进行网页与新闻搜索</td></tr>
<tr><td><code>server-slack</code></td><td>通信</td><td>发送消息、读取频道、搜索</td></tr>
<tr><td><code>server-google-maps</code></td><td>地理空间</td><td>地理编码、路线、地点搜索</td></tr>
<tr><td><code>server-puppeteer</code></td><td>浏览器</td><td>网页抓取、截图、表单交互</td></tr>
<tr><td><code>server-memory</code></td><td>知识</td><td>跨会话的持久化知识图谱</td></tr>
<tr><td><code>server-sequential-thinking</code></td><td>推理</td><td>结构化的多步推理脚手架</td></tr></tbody>
</table>
</div>

## 22.8.2　MCP 在生产应用中的使用 {#mcp-in-production-applications}

<p>MCP 已被多款主流 AI 开发工具采用：</p>

<p><strong>Claude Desktop</strong></p>

<p>Anthropic 的桌面应用<sup aria-label="原文注号 3">3</sup>是首个主要的 MCP host。用户在一个 JSON 配置文件中配置 server；Claude 随后即可在任意对话中使用所有已连接 server 暴露的 tool。</p>

<p><strong>Cursor</strong></p>

<p>这款由 AI 驱动的代码编辑器<sup aria-label="原文注号 4">4</sup>支持 MCP server，允许开发者将其开发工具（数据库、问题跟踪、文档系统）直接连接到编程助手。</p>

<p><strong>VS Code (GitHub Copilot)</strong></p>

<p>Microsoft 在 VS Code 的 GitHub Copilot 中加入了 MCP 支持<sup aria-label="原文注号 5">5</sup>，使编程助手能够访问项目特定的工具与数据源。</p>

<p><strong>自定义 Agent</strong></p>

<p>开源社区已为 LangChain<sup aria-label="原文注号 6">6</sup>、LlamaIndex<sup aria-label="原文注号 7">7</sup>、AutoGen<sup aria-label="原文注号 8">8</sup>等框架加入了 MCP 支持，使建立在这些框架上的任何 Agent 都能使用 MCP server。</p>

## 22.8.3　Server 注册表与发现 {#server-registries-and-discovery}

<p>MCP 生态正在发展 server 发现的基础设施：</p>

<ul>
<li><p><strong>MCP Registry</strong><sup aria-label="原文注号 9">9</sup>：由 Anthropic 维护的、经过验证的 MCP server 官方精选列表。</p></li>
<li><p><strong>npm</strong>：许多 JavaScript/TypeScript MCP server 以 npm 包形式发布在 <code>@modelcontextprotocol</code> 作用域下。</p></li>
<li><p><strong>PyPI</strong>：Python server 以 pip 包形式发布（例如 <code>pip install mcp-server-sqlite</code>）。</p></li>
<li><p><strong>GitHub</strong>：<code>modelcontextprotocol/servers</code><sup aria-label="原文注号 10">10</sup>仓库维护着一份官方 server 的参考集合。</p></li>
<li><p><strong>Python SDK 文档</strong><sup aria-label="原文注号 11">11</sup>：构建 server 与 client 的完整 API 参考与示例。</p></li>
</ul>
