# 22.5　Tool 定义与发现 {#sec:mcp:tools}

<p>Tools 是 MCP 的核心。把 tool 定义写好至关重要，因为 LLM 正是依靠 name 与 description 来决定<em>在什么时候调用哪一个 tool</em>。</p>

## 22.5.1　Tool 模式格式 {#tool-schema-format}

<p>一个完整的 tool 定义：</p>

<div class="highlight"><pre><span></span><code><span class="p">{</span>
  <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"search_codebase"</span><span class="p">,</span>
  <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Search for a pattern across all files in the repository.</span>
    <span class="n">Returns</span> <span class="n">matching</span> <span class="n">file</span> <span class="n">paths</span> <span class="ow">and</span> <span class="n">line</span> <span class="n">numbers</span><span class="o">.</span> <span class="n">Use</span> <span class="n">this</span> <span class="n">when</span> <span class="n">you</span> <span class="n">need</span>
    <span class="n">to</span> <span class="n">find</span> <span class="n">where</span> <span class="n">a</span> <span class="n">function</span> <span class="ow">is</span> <span class="n">defined</span><span class="p">,</span> <span class="n">where</span> <span class="n">a</span> <span class="n">variable</span> <span class="ow">is</span> <span class="n">used</span><span class="p">,</span> <span class="ow">or</span>
    <span class="n">where</span> <span class="n">a</span> <span class="n">specific</span> <span class="n">string</span> <span class="n">appears</span><span class="o">.</span> <span class="n">Supports</span> <span class="n">regex</span> <span class="n">patterns</span><span class="o">.</span><span class="s2">",</span>
  <span class="s2">"inputSchema"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"object"</span><span class="p">,</span>
    <span class="s2">"properties"</span><span class="p">:</span> <span class="p">{</span>
      <span class="s2">"pattern"</span><span class="p">:</span> <span class="p">{</span>
        <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"string"</span><span class="p">,</span>
        <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Regex pattern to search for"</span>
      <span class="p">},</span>
      <span class="s2">"path"</span><span class="p">:</span> <span class="p">{</span>
        <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"string"</span><span class="p">,</span>
        <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Directory to search in (default: repo root)"</span><span class="p">,</span>
        <span class="s2">"default"</span><span class="p">:</span> <span class="s2">"."</span>
      <span class="p">},</span>
      <span class="s2">"case_sensitive"</span><span class="p">:</span> <span class="p">{</span>
        <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"boolean"</span><span class="p">,</span>
        <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Whether the search is case-sensitive"</span><span class="p">,</span>
        <span class="s2">"default"</span><span class="p">:</span> <span class="n">false</span>
      <span class="p">}</span>
    <span class="p">},</span>
    <span class="s2">"required"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"pattern"</span><span class="p">]</span>
  <span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

## 22.5.2　动态工具注册 {#dynamic-tool-registration}

<p>Server 可在会话过程中通过发送 <code>notifications/tools/list_changed</code> 通知来增加、移除或修改 tool。Client 随后通过一次 <code>tools/list</code> 请求重新获取 tool 列表。这一机制使得：</p>

<ul>
<li><p><strong>上下文相关的 tool</strong>：一个代码编辑器 server 可能根据当前打开的文件类型暴露不同的 tool。</p></li>
<li><p><strong>受权限门控的 tool</strong>：只有在用户授予特定权限之后才可用的 tool。</p></li>
<li><p><strong>动态插件系统</strong>：在运行时从外部注册表加载的 tool。</p></li>
</ul>

## 22.5.3　Tool 标注 {#tool-annotations}

<p>MCP 引入了 <strong>tool 标注</strong>——一种帮助 Host 在工具执行时做出更好决策的元数据提示（在 2025-03-26 的协议修订版中加入）：</p>

<div class="highlight"><pre><span></span><code><span class="p">{</span>
  <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"delete_file"</span><span class="p">,</span>
  <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Permanently delete a file from the filesystem."</span><span class="p">,</span>
  <span class="s2">"inputSchema"</span><span class="p">:</span> <span class="p">{</span> <span class="o">...</span> <span class="p">},</span>
  <span class="s2">"annotations"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"readOnlyHint"</span><span class="p">:</span> <span class="n">false</span><span class="p">,</span>      <span class="o">//</span> <span class="n">This</span> <span class="n">tool</span> <span class="n">modifies</span> <span class="n">state</span>
    <span class="s2">"destructiveHint"</span><span class="p">:</span> <span class="n">true</span><span class="p">,</span>    <span class="o">//</span> <span class="n">Changes</span> <span class="n">are</span> <span class="n">irreversible</span>
    <span class="s2">"idempotentHint"</span><span class="p">:</span> <span class="n">false</span><span class="p">,</span>    <span class="o">//</span> <span class="n">Calling</span> <span class="n">twice</span> <span class="n">has</span> <span class="n">different</span> <span class="n">effects</span>
    <span class="s2">"openWorldHint"</span><span class="p">:</span> <span class="n">false</span>      <span class="o">//</span> <span class="n">Does</span> <span class="ow">not</span> <span class="n">interact</span> <span class="k">with</span> <span class="n">external</span> <span class="n">services</span>
  <span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<p><code>readOnlyHint</code></p>

<p>若为 <code>true</code>，该 tool 只读取数据、无副作用。Host 可以在无需用户确认的情况下自动批准只读 tool。</p>

<p><code>destructiveHint</code></p>

<p>若为 <code>true</code>，该 tool 执行不可逆操作。Host 应要求用户进行明确确认。</p>

<p><code>idempotentHint</code></p>

<p>若为 <code>true</code>，使用相同参数多次调用该 tool 的效果等同于只调用一次。失败时可安全重试。</p>

<p><code>openWorldHint</code></p>

<p>若为 <code>true</code>，该 tool 会与 server 直接控制之外的外部服务交互（例如发送邮件、向社交媒体发帖）。</p>

<div class="admonition warning"><p class="admonition-title">Tool 描述至关重要</p>
<p>LLM 几乎完全依据 <code>name</code> 与 <code>description</code> 字段来选择 tool。模糊或有歧义的描述会导致选错 tool、错失使用正确 tool 的机会，乃至产生幻觉式的工具调用。最佳实践：</p>

<ul>
<li><p><strong>清楚说明该 tool 能做什么、不能做什么。</strong>“按内容搜索文件”比“搜索文件”更好。</p></li>
<li><p><strong>说明何时该使用它。</strong>“当你需要查找某个符号在何处定义时使用此 tool”可以引导 LLM 的决策。</p></li>
<li><p><strong>描述输出格式。</strong>“返回一个由 {file, line, match} 对象组成的 JSON 数组”有助于 LLM 解析结果。</p></li>
<li><p><strong>说明限制。</strong>“仅搜索 <code>.py</code> 文件；其他类型请使用 <code>search_all</code>”可防止误用。</p></li>
<li><p><strong>避免使用</strong> LLM 可能无法与该 tool 实际行为关联起来的行话。</p></li>
</ul>
</div>
