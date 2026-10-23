# 22.9　MCP vs. 其他方案 {#sec:mcp:alternatives}

<p class="table-caption" id="tab:mcp-vs-alternatives"><span class="table-number">表 22.2　</span>MCP 与其他工具集成方案的对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>特性</strong></th><th scope="col"><strong>MCP</strong></th><th scope="col"><strong>OpenAI Functions</strong></th><th scope="col"><strong>LangChain Tools</strong></th><th scope="col"><strong>直接 API</strong></th></tr></thead>
<tbody><tr><td>标准化</td><td>✓</td><td>部分</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td></tr>
<tr><td>跨厂商</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>部分</td><td><span class="arithmatex">\(\times\)</span></td></tr>
<tr><td>有状态会话</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td>视情况</td></tr>
<tr><td>资源流式传输</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td>视情况</td></tr>
<tr><td>服务端推送</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td>视情况</td></tr>
<tr><td>Sampling（反向）</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td></tr>
<tr><td>生态规模</td><td>成长中</td><td>大</td><td>大</td><td>无上限</td></tr>
<tr><td>搭建复杂度</td><td>中</td><td>低</td><td>低</td><td>高</td></tr>
<tr><td>厂商锁定</td><td>无</td><td>OpenAI</td><td>LangChain</td><td>无</td></tr></tbody>
</table>
</div>

## 22.9.1　何时使用 MCP vs. 自定义集成 {#when-to-use-mcp-vs.-custom-integration}

<p><strong>在以下情况使用 MCP：</strong></p>

<ul>
<li><p>希望自己的 tool 能与多个 LLM 提供方或 Agent 框架协同工作</p></li>
<li><p>正在构建供他人使用的 tool（开源或企业分发）</p></li>
<li><p>需要有状态会话、资源订阅或服务端推送能力</p></li>
<li><p>希望复用已有的 MCP server 生态</p></li>
</ul>

<p><strong>在以下情况使用自定义集成：</strong></p>

<ul>
<li><p>只有单一、紧耦合的 LLM 提供方，且无更换计划</p></li>
<li><p>需要极低延迟，无法承受协议开销</p></li>
<li><p>工具接口非常特殊，难以与 MCP 原语良好映射</p></li>
<li><p>正处于早期原型阶段，希望尽量减少依赖</p></li>
</ul>

## 22.9.2　迁移路径 {#migration-paths}

<p>从 OpenAI 函数调用迁移到 MCP 较为直接：工具参数的 JSON Schema 格式完全一致。主要的变化在于：</p>

<ol>
<li><p>将 tool 实现封装到一个 MCP server 中（使用 Python 或 TypeScript SDK）</p></li>
<li><p>在 client 中用 <code>session.call_tool()</code> 替换直接的 API 调用</p></li>
<li><p>加入能力协商与生命周期管理</p></li>
</ol>

<p>LangChain 的 tool 可以通过 <code>langchain-mcp-adapters</code> 包封装到 MCP server 中，该包在 LangChain 的 <code>BaseTool</code> 接口与 MCP tool 定义之间提供自动转换。</p>
