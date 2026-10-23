# 22.4　协议规范 {#sec:mcp:protocol}

<p>MCP 构建在 <strong>JSON-RPC 2.0</strong> <a class="book-citation" id="cite-jsonrpc2010spec-1" href="../../references.md#ref-jsonrpc2010spec" aria-label="参考文献 121">[121]</a> 之上，这是一种以 JSON 作为消息编码的轻量级远程过程调用协议。该选择提供了一个被广泛理解、与编程语言无关、且有广泛库支持的基础。</p>

## 22.4.1　JSON-RPC 2.0 消息格式 {#json-rpc-2.0-message-format}

<p>JSON-RPC 2.0 中共有三种消息类型：</p>

<p><strong>Request</strong>（client <span class="arithmatex">\(\to\)</span> server，期望收到响应）：</p>

<div class="highlight"><pre><span></span><code><span class="p">{</span>
  <span class="s2">"jsonrpc"</span><span class="p">:</span> <span class="s2">"2.0"</span><span class="p">,</span>
  <span class="s2">"id"</span><span class="p">:</span> <span class="mi">42</span><span class="p">,</span>
  <span class="s2">"method"</span><span class="p">:</span> <span class="s2">"tools/call"</span><span class="p">,</span>
  <span class="s2">"params"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"read_file"</span><span class="p">,</span>
    <span class="s2">"arguments"</span><span class="p">:</span> <span class="p">{</span> <span class="s2">"path"</span><span class="p">:</span> <span class="s2">"/home/user/notes.txt"</span> <span class="p">}</span>
  <span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<p><strong>Response</strong>（server <span class="arithmatex">\(\to\)</span> client，作为对请求的回复）：</p>

<div class="highlight"><pre><span></span><code><span class="p">{</span>
  <span class="s2">"jsonrpc"</span><span class="p">:</span> <span class="s2">"2.0"</span><span class="p">,</span>
  <span class="s2">"id"</span><span class="p">:</span> <span class="mi">42</span><span class="p">,</span>
  <span class="s2">"result"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"content"</span><span class="p">:</span> <span class="p">[</span>
      <span class="p">{</span> <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"text"</span><span class="p">,</span> <span class="s2">"text"</span><span class="p">:</span> <span class="s2">"Meeting notes: ..."</span> <span class="p">}</span>
    <span class="p">],</span>
    <span class="s2">"isError"</span><span class="p">:</span> <span class="n">false</span>
  <span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<p><strong>Notification</strong>（两个方向均可，不期望响应）：</p>

<div class="highlight"><pre><span></span><code><span class="p">{</span>
  <span class="s2">"jsonrpc"</span><span class="p">:</span> <span class="s2">"2.0"</span><span class="p">,</span>
  <span class="s2">"method"</span><span class="p">:</span> <span class="s2">"notifications/resources/updated"</span><span class="p">,</span>
  <span class="s2">"params"</span><span class="p">:</span> <span class="p">{</span> <span class="s2">"uri"</span><span class="p">:</span> <span class="s2">"file:///home/user/notes.txt"</span> <span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

## 22.4.2　能力协商握手 {#capability-negotiation-handshake}

<p>初始化握手确立双方各自能够做什么：</p>

<div class="highlight"><pre><span></span><code><span class="o">//</span> <span class="n">Client</span> <span class="n">sends</span><span class="p">:</span>
<span class="p">{</span>
  <span class="s2">"jsonrpc"</span><span class="p">:</span> <span class="s2">"2.0"</span><span class="p">,</span> <span class="s2">"id"</span><span class="p">:</span> <span class="mi">1</span><span class="p">,</span>
  <span class="s2">"method"</span><span class="p">:</span> <span class="s2">"initialize"</span><span class="p">,</span>
  <span class="s2">"params"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"protocolVersion"</span><span class="p">:</span> <span class="s2">"2024-11-05"</span><span class="p">,</span>
    <span class="s2">"capabilities"</span><span class="p">:</span> <span class="p">{</span>
      <span class="s2">"sampling"</span><span class="p">:</span> <span class="p">{},</span>          <span class="o">//</span> <span class="n">client</span> <span class="n">supports</span> <span class="n">sampling</span> <span class="n">requests</span>
      <span class="s2">"roots"</span><span class="p">:</span> <span class="p">{</span> <span class="s2">"listChanged"</span><span class="p">:</span> <span class="n">true</span> <span class="p">}</span>
    <span class="p">},</span>
    <span class="s2">"clientInfo"</span><span class="p">:</span> <span class="p">{</span> <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"MyAgent"</span><span class="p">,</span> <span class="s2">"version"</span><span class="p">:</span> <span class="s2">"1.0.0"</span> <span class="p">}</span>
  <span class="p">}</span>
<span class="p">}</span>

<span class="o">//</span> <span class="n">Server</span> <span class="n">responds</span><span class="p">:</span>
<span class="p">{</span>
  <span class="s2">"jsonrpc"</span><span class="p">:</span> <span class="s2">"2.0"</span><span class="p">,</span> <span class="s2">"id"</span><span class="p">:</span> <span class="mi">1</span><span class="p">,</span>
  <span class="s2">"result"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"protocolVersion"</span><span class="p">:</span> <span class="s2">"2024-11-05"</span><span class="p">,</span>
    <span class="s2">"capabilities"</span><span class="p">:</span> <span class="p">{</span>
      <span class="s2">"tools"</span><span class="p">:</span> <span class="p">{</span> <span class="s2">"listChanged"</span><span class="p">:</span> <span class="n">true</span> <span class="p">},</span>   <span class="o">//</span> <span class="n">server</span> <span class="n">has</span> <span class="n">tools</span>
      <span class="s2">"resources"</span><span class="p">:</span> <span class="p">{</span> <span class="s2">"subscribe"</span><span class="p">:</span> <span class="n">true</span> <span class="p">},</span> <span class="o">//</span> <span class="n">server</span> <span class="n">supports</span> <span class="n">subscriptions</span>
      <span class="s2">"prompts"</span><span class="p">:</span> <span class="p">{}</span>
    <span class="p">},</span>
    <span class="s2">"serverInfo"</span><span class="p">:</span> <span class="p">{</span> <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"filesystem"</span><span class="p">,</span> <span class="s2">"version"</span><span class="p">:</span> <span class="s2">"0.6.2"</span> <span class="p">}</span>
  <span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

## 22.4.3　错误处理 {#error-handling}

<p>JSON-RPC 错误遵循一套标准格式，带有数值型错误码。MCP 在 JSON-RPC 标准之外定义了额外的错误码：</p>

<div class="highlight"><pre><span></span><code><span class="p">{</span>
  <span class="s2">"jsonrpc"</span><span class="p">:</span> <span class="s2">"2.0"</span><span class="p">,</span> <span class="s2">"id"</span><span class="p">:</span> <span class="mi">42</span><span class="p">,</span>
  <span class="s2">"error"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"code"</span><span class="p">:</span> <span class="o">-</span><span class="mi">32602</span><span class="p">,</span>          <span class="o">//</span> <span class="n">Invalid</span> <span class="n">params</span> <span class="p">(</span><span class="n">JSON</span><span class="o">-</span><span class="n">RPC</span> <span class="n">standard</span><span class="p">)</span>
    <span class="s2">"message"</span><span class="p">:</span> <span class="s2">"Invalid file path: path must be absolute"</span><span class="p">,</span>
    <span class="s2">"data"</span><span class="p">:</span> <span class="p">{</span> <span class="s2">"path"</span><span class="p">:</span> <span class="s2">"relative/path.txt"</span> <span class="p">}</span>
  <span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<div class="admonition note"><p class="admonition-title">MCP 错误码</p>
<span id="table-22-inline-3"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>错误码</strong></th><th scope="col"><strong>名称</strong></th><th scope="col"><strong>含义</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(-32700\)</span></td><td>Parse Error</td><td>收到的 JSON 无效</td></tr>
<tr><td><span class="arithmatex">\(-32600\)</span></td><td>Invalid Request</td><td>不是合法的 JSON-RPC 对象</td></tr>
<tr><td><span class="arithmatex">\(-32601\)</span></td><td>Method Not Found</td><td>方法不存在</td></tr>
<tr><td><span class="arithmatex">\(-32602\)</span></td><td>Invalid Params</td><td>方法参数无效</td></tr>
<tr><td><span class="arithmatex">\(-32603\)</span></td><td>Internal Error</td><td>服务器内部错误</td></tr></tbody>
</table>
</div>

<p>取消操作通过 <code>notifications/cancelled</code> 处理（一种通知，而非错误响应）。按 JSON-RPC 约定，server 可在 <span class="arithmatex">\(-32000\)</span> 到 <span class="arithmatex">\(-32099\)</span> 范围内定义额外的应用级错误码。</p>
</div>

## 22.4.4　进度上报 {#progress-reporting}

<p>对于长时间运行的操作，MCP 支持进度通知。Client 在请求中带上一个 <code>progressToken</code>；server 周期性地发送 <code>notifications/progress</code> 消息：</p>

<div class="highlight"><pre><span></span><code><span class="o">//</span> <span class="n">Request</span> <span class="k">with</span> <span class="n">progress</span> <span class="n">token</span>
<span class="p">{</span>
  <span class="s2">"jsonrpc"</span><span class="p">:</span> <span class="s2">"2.0"</span><span class="p">,</span> <span class="s2">"id"</span><span class="p">:</span> <span class="mi">10</span><span class="p">,</span>
  <span class="s2">"method"</span><span class="p">:</span> <span class="s2">"tools/call"</span><span class="p">,</span>
  <span class="s2">"params"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"index_codebase"</span><span class="p">,</span>
    <span class="s2">"arguments"</span><span class="p">:</span> <span class="p">{</span> <span class="s2">"path"</span><span class="p">:</span> <span class="s2">"/repo"</span> <span class="p">},</span>
    <span class="s2">"_meta"</span><span class="p">:</span> <span class="p">{</span> <span class="s2">"progressToken"</span><span class="p">:</span> <span class="s2">"index-op-1"</span> <span class="p">}</span>
  <span class="p">}</span>
<span class="p">}</span>

<span class="o">//</span> <span class="n">Server</span> <span class="n">sends</span> <span class="n">progress</span> <span class="n">notifications</span> <span class="p">(</span><span class="n">no</span> <span class="nb">id</span> <span class="o">=</span> <span class="n">notification</span><span class="p">)</span>
<span class="p">{</span>
  <span class="s2">"jsonrpc"</span><span class="p">:</span> <span class="s2">"2.0"</span><span class="p">,</span>
  <span class="s2">"method"</span><span class="p">:</span> <span class="s2">"notifications/progress"</span><span class="p">,</span>
  <span class="s2">"params"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"progressToken"</span><span class="p">:</span> <span class="s2">"index-op-1"</span><span class="p">,</span>
    <span class="s2">"progress"</span><span class="p">:</span> <span class="mi">45</span><span class="p">,</span>
    <span class="s2">"total"</span><span class="p">:</span> <span class="mi">100</span><span class="p">,</span>
    <span class="s2">"message"</span><span class="p">:</span> <span class="s2">"Indexed 450/1000 files..."</span>
  <span class="p">}</span>
<span class="p">}</span>
</code></pre></div>
