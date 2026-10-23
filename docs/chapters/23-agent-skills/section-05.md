# 23.5　Skill 注册表与市场 {#skill-registries-and-marketplaces}

<p>生产级 skill 系统需要相应的基础设施：</p>

<ul>
<li><p><strong>Skill 清单（manifest）</strong>：一份结构化描述（名称、能力、所需工具、输入/输出 schema），用于支持自动发现与路由。</p></li>
<li><p><strong>版本控制</strong>：Skill 会不断演进；Agent 需要锁定特定版本以保证可复现性。</p></li>
<li><p><strong>依赖解析</strong>：Skill 可能需要特定的 MCP server、API key 或其他 skill。</p></li>
<li><p><strong>权限模型</strong>：并非所有 Agent 都应能访问所有 skill（安全性、成本与能力边界考虑）。</p></li>
<li><p><strong>市场</strong>：组织可以发布、分享与安装 skill——类似于代码的包管理器。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">Skill 清单示例</p>
<p>一份 skill 清单声明了编排器加载并调用一个 skill 所需的一切。目前尚不存在行业标准 schema；下面是一个示意性格式，它涵盖了真实实现（Anthropic MCP、OpenAI 函数规范、LangChain 工具定义）中常见的字段：</p>

<div class="highlight"><pre><span></span><code><span class="o">//</span> <span class="n">示意性</span> <span class="n">schema</span> <span class="err">——</span> <span class="n">并非任何具体</span> <span class="n">SDK</span> <span class="n">的格式</span>
<span class="p">{</span>
  <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"code-review"</span><span class="p">,</span>
  <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Review code changes for bugs, style, and security issues"</span><span class="p">,</span>
  <span class="s2">"version"</span><span class="p">:</span> <span class="s2">"2.1.0"</span><span class="p">,</span>
  <span class="s2">"requires"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"tools"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"file_read"</span><span class="p">,</span> <span class="s2">"grep"</span><span class="p">,</span> <span class="s2">"git_diff"</span><span class="p">],</span>
    <span class="s2">"mcp_servers"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"github"</span><span class="p">],</span>
    <span class="s2">"models"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"claude-sonnet-4-20250514"</span><span class="p">]</span>
  <span class="p">},</span>
  <span class="s2">"input_schema"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"object"</span><span class="p">,</span>
    <span class="s2">"properties"</span><span class="p">:</span> <span class="p">{</span>
      <span class="s2">"repo"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"string"</span><span class="p">},</span>
      <span class="s2">"pr_number"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"integer"</span><span class="p">}</span>
    <span class="p">}</span>
  <span class="p">},</span>
  <span class="s2">"prompts"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"skills/code-review/system.md"</span><span class="p">],</span>
  <span class="s2">"knowledge"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"skills/code-review/style-guide.md"</span><span class="p">]</span>
<span class="p">}</span>
</code></pre></div>
</div>
