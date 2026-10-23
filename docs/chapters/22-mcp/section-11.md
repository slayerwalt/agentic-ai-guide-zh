# 22.11　小结 {#sec:mcp:summary}

<p>模型上下文协议（MCP）是朝着“标准化 AI Agent 与世界交互方式”迈出的重要一步。通过把 <span class="arithmatex">\(N \times M\)</span> 的集成问题降低为 <span class="arithmatex">\(N + M\)</span>，MCP 降低了构建有能力、工具增强的 AI 系统的门槛。其关键设计决定——以 JSON-RPC 2.0 作为线协议、有状态会话、四种核心原语（tools、resources、prompts、sampling），以及清晰的安全模型——折射出 LSP 与 USB 生态中得来不易的经验。</p>

<p>对于构建经 RL 训练的 Agent 的从业者而言，MCP 提供了一项尤为有吸引力的价值主张：一个标准化、可扩展的接口，用以定义动作空间、采集训练轨迹，并将训练好的 Agent 部署到多样化的环境中。随着生态成熟、基准套件出现，MCP 有望成为工具使用 Agent 研究事实上的基础设施——LLM 时代的 gymnasium。</p>

<div class="admonition note"><p class="admonition-title">MCP 速览</p>
<span id="table-22-inline-6"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>属性</strong></th><th scope="col"><strong>取值</strong></th></tr></thead>
<tbody><tr><td>线协议</td><td>JSON-RPC 2.0</td></tr>
<tr><td>传输</td><td>stdio、Streamable HTTP</td></tr>
<tr><td>核心原语</td><td>Tools、Resources、Prompts、Sampling</td></tr>
<tr><td>会话模型</td><td>有状态（持久连接）</td></tr>
<tr><td>工具模式格式</td><td>JSON Schema（Draft 7）</td></tr>
<tr><td>安全模型</td><td>由 Host 执行同意 + 信任层级</td></tr>
<tr><td>主要用例</td><td>标准化 LLM <span class="arithmatex">\(\leftrightarrow\)</span> 工具集成</td></tr>
<tr><td>与 RL 的关联</td><td>标准化动作空间 + 轨迹记录</td></tr>
<tr><td>官方 SDK</td><td>Python、TypeScript（Node.js）</td></tr>
<tr><td>许可</td><td>开放标准（MIT）</td></tr></tbody>
</table>
</div>
</div>
