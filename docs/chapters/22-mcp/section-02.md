# 22.2　架构概览 {#sec:mcp:architecture}

<p>MCP 采用一种<strong>客户端-服务器架构</strong>，包含三种不同的角色，由一层定义明确的协议层加以连接。</p>

## 22.2.1　三角色模型 {#the-three-role-model}

<p><strong>MCP Host</strong></p>

<p>终端用户直接与之交互的 LLM 应用。例如 Claude Desktop、VS Code 扩展、自研聊天机器人或自主 Agent。Host 负责管理整体用户体验，决定连接哪些 MCP server，并执行安全策略。一个 Host 内部包含一个或多个 MCP client。</p>

<p><strong>MCP Client</strong></p>

<p>嵌入在 Host 应用内部的协议层组件。每个 client 与单个 MCP server 维持一条<em>有状态的一对一连接</em>。Client 负责协议协商、消息序列化以及连接的生命周期管理。一个 Host 可同时运行多个 client，每个 client 连接到不同的 server。</p>

<p><strong>MCP Server</strong></p>

<p>一个轻量级的进程或服务，向 client 暴露能力（tools、resources、prompts）。Server 通常是对既有 API、数据库或系统接口的薄包装。它们被设计得易于实现——协议的复杂性由 client/host 层承担。</p>

<div class="admonition example"><p class="admonition-title">具体示例：一个编程助手</p>
<p>一位开发者使用一个由 Claude 驱动的 VS Code 扩展（即 <strong>Host</strong>）。该扩展运行三个 <strong>Client</strong>，分别连接到不同的 <strong>Server</strong>：</p>

<ul>
<li><p>一个可读写本地文件的<em>文件系统 server</em></p></li>
<li><p>一个可查询 issue、PR 和提交历史的 <em>GitHub server</em></p></li>
<li><p>一个可对开发数据库执行只读 SQL 查询的 <em>PostgreSQL server</em></p></li>
</ul>

<p>当该开发者问“修复 <code>auth.py</code> 中导致 issue #42 所示登录失败的 bug”时，LLM 可以同时读取该文件、抓取该 GitHub issue 并查询相关数据库日志——全部通过标准化的 MCP 调用完成。</p>
</div>

## 22.2.2　传输层 {#transport-layers}

<p>MCP 在协议层与传输层无关，但定义了两种标准的传输机制：</p>

<p><strong>stdio（标准输入输出）</strong></p>

<p>Client 将 server 作为子进程启动，通过标准输入/输出流通信。这是本地工具最简单、最常见的传输方式。它提供了较强的隔离性（server 运行在独立的进程中），且无需任何网络配置。非常适合文件系统访问、本地代码执行和开发者工具。</p>

<p><strong>Streamable HTTP</strong></p>

<p>Server 作为 HTTP 服务运行。Client 通过 HTTP POST 发送 JSON-RPC 请求；server 可以返回单个 JSON 响应，也可以升级为 Server-Sent Events (SSE) 流以增量返回结果。这种传输方式支持远程 server，可启用服务端推送通知，并能穿越标准的 Web 基础设施（代理、负载均衡器、防火墙）。适用于云端托管工具与企业部署。（该传输在 2025-03-26 的协议修订版中取代了早先仅 HTTP+SSE 的传输方式。）</p>

## 22.2.3　协议生命周期 {#protocol-lifecycle}

<p>每一条 MCP 连接都遵循一个四阶段的生命周期：</p>

<ol>
<li><p><strong>初始化</strong>：Client 发送一个 <code>initialize</code> 请求，其中包含其协议版本与所支持的能力。Server 返回自己的版本与能力。这一步确立了本次会话可用的特性集合。</p></li>
<li><p><strong>能力协商</strong>：双方各自声明所支持的能力（例如 server 是否提供 tools、resources 或 prompts；client 是否支持 sampling）。未被双方都声明的能力不会被使用。</p></li>
<li><p><strong>运行</strong>：主要阶段。Client 发送请求（工具调用、资源读取、prompt 获取），server 给出响应。Server 也可在未被询问的情况下主动发送通知（例如资源变更事件）。</p></li>
<li><p><strong>关闭</strong>：任一方都可发起优雅关闭。Client 发送一个 <code>shutdown</code> 通知；server 清理资源并终止。</p></li>
</ol>

## 22.2.4　有状态会话 vs. 无状态请求 {#stateful-sessions-vs.-stateless-requests}

<p>MCP 的一项关键设计决定是：连接是<strong>有状态会话</strong>，而非无状态的 HTTP 请求。这一点之所以重要，原因有以下几点：</p>

<ul>
<li><p><strong>效率</strong>：能力协商只在建立连接时发生一次，而不是每次请求都进行。</p></li>
<li><p><strong>上下文</strong>：Server 可维护会话状态（例如一个未提交的数据库事务、一个已签出的文件锁）。</p></li>
<li><p><strong>订阅</strong>：当资源发生变化时，server 可向 client 推送通知。</p></li>
<li><p><strong>长时运行操作</strong>：在有状态会话中，进度上报变得自然。</p></li>
</ul>

<p>代价在于，有状态会话需要进行连接管理（重连逻辑、会话恢复），而这正是无状态 API 所规避的。</p>

## 22.2.5　完整架构图 {#sec:mcp:diagram}

<p>图 <a href="#fig:mcp-architecture">22.2</a> 展示了完整的 MCP 栈，从用户界面一直到外部服务。</p>

<figure id="fig:mcp-architecture"><a href="../../assets/figures/fig_067_mcp-architecture.png" target="_blank" rel="noopener" aria-label="查看图 22.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_067_mcp-architecture.png" alt="完整的 MCP 架构栈。Host 管理一个或多个 Client，每个 Client 通过传输层（stdio 或 Streamable HTTP）与一个 MCP Server 维持有状态的会话。所有 client–server 通信均使用 JSON-RPC 2.0。Server 包装外部服务，并将其暴露为标准化的 Tools、Resources 与 Prompts。" loading="lazy"></a><figcaption><span class="figure-number">图 22.2　</span>完整的 MCP 架构栈。Host 管理一个或多个 Client，每个 Client 通过传输层（stdio 或 Streamable HTTP）与一个 MCP Server 维持有状态的会话。所有 client–server 通信均使用 JSON-RPC 2.0。Server 包装外部服务，并将其暴露为标准化的 Tools、Resources 与 Prompts。</figcaption></figure>
