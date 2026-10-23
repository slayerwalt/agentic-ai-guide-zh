# 24.7　A2A 与 MCP：互补的协议 {#sec:a2a:vs_mcp}

<p>A2A 与 <strong>模型上下文协议（Model Context Protocol, MCP）</strong> <a class="book-citation" id="cite-anthropic-mcp-2024-1" href="../../references.md#ref-anthropic-mcp-2024" aria-label="参考文献 9">[9]</a> 之间的关系常被误解。这两种协议是<em>互补</em>的，并非竞争关系：</p>

<div class="admonition note"><p class="admonition-title">核心区别</p>
<ul>
<li><p><strong>MCP</strong> 是<em>纵向</em>协议：它把一个 Agent 向下延伸到数据库、API、文件系统和代码执行器的世界。只有 Agent 在推理；MCP 端点是确定性服务。</p></li>
<li><p><strong>A2A</strong> 是<em>横向</em>协议：它把一个具备推理能力的 Agent 与另一个相连。两端都是能够推理、规划和使用 Tool 的智能体。</p></li>
</ul>
</div>

<span id="table-24-inline-4"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>MCP</strong></th><th scope="col"><strong>A2A</strong></th></tr></thead>
<tbody><tr><td><strong>参与方</strong></td><td>Agent <span class="arithmatex">\(\leftrightarrow\)</span> Tool/资源</td><td>Agent <span class="arithmatex">\(\leftrightarrow\)</span> Agent</td></tr>
<tr><td><strong>智能性</strong></td><td>仅一端（Agent）具备智能</td><td>双方均具备智能</td></tr>
<tr><td><strong>状态性</strong></td><td>通常为无状态 Tool 调用</td><td>带生命周期的有状态任务</td></tr>
<tr><td><strong>流式传输</strong></td><td>有限（Tool 结果）</td><td>一等的 SSE 流式传输</td></tr>
<tr><td><strong>发现机制</strong></td><td>Tool 清单</td><td>Agent Card</td></tr>
<tr><td><strong>认证模型</strong></td><td>服务端控制</td><td>双向认证、OAuth 2.0</td></tr>
<tr><td><strong>典型延迟</strong></td><td>毫秒级</td><td>秒到分钟级</td></tr>
<tr><td><strong>典型用例</strong></td><td>“搜索网络”、“运行 SQL”</td><td>“委派给专家 Agent”</td></tr></tbody>
</table>
</div>

## 24.7.1　何时使用哪一种 {#when-to-use-which}

<ul>
<li><p>当远程端点是一个确定性函数时使用 <strong>MCP</strong>：数据库查询、API 调用、代码执行沙箱。Agent 完全掌控交互过程。</p></li>
<li><p>当远程端点需要对请求进行<em>推理</em>时使用 <strong>A2A</strong>：解释模糊指令、做出判断、使用自有 Tool 或进行多轮对话。</p></li>
<li><p>在同一系统中<strong>同时使用两者</strong>：编排 Agent 通过 A2A 将任务委派给专家 Agent，而每个专家 Agent 通过 MCP 访问自己的 Tool。</p></li>
</ul>

## 24.7.2　组合架构 {#combined-architecture}

<p>在生产环境的多 Agent 系统中，A2A 与 MCP 在不同层次上协同工作：<strong>A2A</strong> 处理 Agent 之间的委派与协调（同级之间的横向通信），而 <strong>MCP</strong> 处理每个 Agent 到其 Tool 与数据源的连接（与能力的纵向集成）。这种关注点分离是构建可扩展 Agent 架构的关键。</p>

<figure id="fig:combined-a2a-mcp"><a href="../../assets/figures/fig_068_combined-a2a-mcp.png" target="_blank" rel="noopener" aria-label="查看图 24.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_068_combined-a2a-mcp.png" alt="A2A 与 MCP 组合架构。编排器通过 A2A 将任务委派给专家 Agent；每个 Agent 则通过 MCP 服务器访问自己的 Tool。" loading="lazy"></a><figcaption><span class="figure-number">图 24.1　</span>A2A 与 MCP 组合架构。编排器通过 A2A 将任务委派给专家 Agent；每个 Agent 则通过 MCP 服务器访问自己的 Tool。</figcaption></figure>

<div class="admonition note"><p class="admonition-title">要点</p>
<ul>
<li><p><strong>用 A2A 进行委派</strong>：当一个 Agent 需要它本身不具备的能力时，会通过 A2A 任务消息委派给另一个 Agent。每个 Agent 都是带有自身 Agent Card 的自包含服务。</p></li>
<li><p><strong>用 MCP 访问 Tool</strong>：每个 Agent 通过 MCP 服务器连接其 Tool。这意味着 Tool 永远不会直接暴露给其他 Agent——只能通过所属 Agent 的接口访问。</p></li>
<li><p><strong>信任边界的分离</strong>：编排器信任专家 Agent（通过 A2A 认证进行核实）。每个专家信任其自有的 MCP 服务器（本地或已认证）。不存在可传递的 Tool 访问。</p></li>
<li><p><strong>独立扩缩容</strong>：代码密集型负载可扩展 CodeAgent 实例；数据型负载可扩展 DataAgent。编排器始终保持轻量。</p></li>
</ul>
</div>
