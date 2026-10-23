# 22.1　动机：工具集成问题 {#sec:mcp:motivation}

<div class="admonition tip"><p class="admonition-title">为什么标准化很重要</p>
<p>每当出现一个新的 LLM Agent 框架，开发者都不得不重新为同一批工具实现连接器：文件系统、数据库、网页搜索、代码执行、日历 API。这种做法既浪费又易出错，并且产生了一种随 Agent 数与工具数呈平方级增长的维护负担。</p>
</div>

<p>考虑任何想把 AI Agent 接入自身基础设施的组织所面临的组合爆炸。假设有 <span class="arithmatex">\(N\)</span> 种不同的 Agent 框架（LangChain、AutoGen、CrewAI、自研 Agent、…）和 <span class="arithmatex">\(M\)</span> 个不同的工具提供方（GitHub、Slack、PostgreSQL、Jira、…）。在没有标准协议的情况下，每一种组合都需要一份定制集成：</p>



<div class="arithmatex">\[\text{Integrations without standard} = N \times M
\tag{22.1}\]</div>

<p>有了通用协议，每一方都只需实现一次该协议：</p>



<div class="arithmatex">\[\text{Integrations with standard} = N + M
\tag{22.2}\]</div>

<p>对于 <span class="arithmatex">\(N = 20\)</span> 个 Agent 框架和 <span class="arithmatex">\(M = 50\)</span> 个工具提供方而言，这把集成负担从 1,000 个定制连接器降低到仅 70 个协议实现——<strong>减少了 14 倍</strong>。这正是 USB（通用设备连接）、HTTP（通用 Web 通信）和 LSP（语言服务器协议，用于 IDE 工具链）等协议背后的思想。MCP 把同样的理念应用到了 AI 工具使用上。</p>

<div class="admonition note"><p class="admonition-title"><span class="arithmatex">\(N \times M \to N+M\)</span> 的化简</p>
<span id="table-22-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>场景</strong></th><th scope="col"><strong>无 MCP</strong></th><th scope="col"><strong>有 MCP</strong></th></tr></thead>
<tbody><tr><td>20 个 Agent、50 个工具</td><td>1,000 个连接器</td><td>70 个实现</td></tr>
<tr><td>50 个 Agent、200 个工具</td><td>10,000 个连接器</td><td>250 个实现</td></tr>
<tr><td>100 个 Agent、500 个工具</td><td>50,000 个连接器</td><td>600 个实现</td></tr></tbody>
</table>
</div>

<p>MCP 把一个平方级的集成问题转化为线性问题——这与让 USB 取代数十种专有端口标准的洞见如出一辙。</p>
</div>

<p>与<strong>语言服务器协议（Language Server Protocol, LSP）</strong><sup aria-label="原文注号 1">1</sup>的类比尤为贴切。在 LSP 出现之前，每一个 IDE 都必须为每一种编程语言单独实现语言支持（自动补全、跳转定义、错误高亮）。在 LSP 之后，语言服务器与编辑器只需说一种共同的协议。MCP 之于 AI 工具使用，正如 LSP 之于开发者工具链。</p>

<figure id="figure-22-1"><a href="../../assets/figures/fig_066_mcp-flow.png" target="_blank" rel="noopener" aria-label="查看图 22.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_066_mcp-flow.png" alt="MCP 的工作方式：一次用户请求依次流经 Host、LLM 与 MCP Server。LLM 决定调用哪一个工具（步骤 3）；Host 通过 JSON-RPC 将该调用路由到对应的 server（步骤 4）；结果再回传给 LLM 进行自然语言整理（步骤 5–7）。用户看不到任何协议层的细节。" loading="lazy"></a><figcaption><span class="figure-number">图 22.1　</span>MCP 的工作方式：一次用户请求依次流经 Host、LLM 与 MCP Server。LLM 决定调用哪一个工具（步骤 3）；Host 通过 JSON-RPC 将该调用路由到对应的 server（步骤 4）；结果再回传给 LLM 进行自然语言整理（步骤 5–7）。用户看不到任何协议层的细节。</figcaption></figure>
