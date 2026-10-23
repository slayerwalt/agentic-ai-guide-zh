# 22.7　实现模式 {#sec:mcp:implementation}

## 22.7.1　用 Python 构建一个 MCP Server {#building-an-mcp-server-in-python}

<p>官方 Python SDK 提供了 <code>FastMCP</code>，这是一个高级框架，可自动处理协议协商、序列化与传输。下面是一个完整的记笔记 MCP server：</p>

<p class="code-caption"><span class="code-number">代码 22.1　</span>完整 MCP Server：记笔记 Tool（FastMCP）</p>
<div class="highlight"><pre><span></span><code><span class="ch">#!/usr/bin/env python3</span>
<span class="sd">"""</span>
<span class="sd">A simple MCP server exposing note-taking tools and resources.</span>
<span class="sd">Install: pip install "mcp[cli]"</span>
<span class="sd">Run:     mcp run notes_server.py        (stdio)</span>
<span class="sd">         mcp run notes_server.py --transport streamable-http  (HTTP)</span>
<span class="sd">"""</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pathlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">Path</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">mcp.server.fastmcp</span><span class="w"> </span><span class="kn">import</span> <span class="n">FastMCP</span>

<span class="c1"># -- 服务器设置 ----------------------------------------------------------------</span>
<span class="n">mcp</span> <span class="o">=</span> <span class="n">FastMCP</span><span class="p">(</span><span class="s2">"notes-server"</span><span class="p">)</span>
<span class="n">NOTES_DIR</span> <span class="o">=</span> <span class="n">Path</span><span class="o">.</span><span class="n">home</span><span class="p">()</span> <span class="o">/</span> <span class="s2">".notes"</span>
<span class="n">NOTES_DIR</span><span class="o">.</span><span class="n">mkdir</span><span class="p">(</span><span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="c1"># -- Tools（由 LLM 调用的动作） -------------------------------------------------</span>
<span class="nd">@mcp</span><span class="o">.</span><span class="n">tool</span><span class="p">()</span>
<span class="k">def</span><span class="w"> </span><span class="nf">create_note</span><span class="p">(</span><span class="n">title</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">content</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">tags</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""Create a new text note with a given title and content.</span>

<span class="sd">    Use this when the user wants to save information for later.</span>
<span class="sd">    Returns the path where the note was saved.</span>
<span class="sd">    """</span>
    <span class="n">tags</span> <span class="o">=</span> <span class="n">tags</span> <span class="ow">or</span> <span class="p">[]</span>
    <span class="n">safe_title</span> <span class="o">=</span> <span class="s2">""</span><span class="o">.</span><span class="n">join</span><span class="p">(</span>
        <span class="n">c</span> <span class="k">if</span> <span class="n">c</span><span class="o">.</span><span class="n">isalnum</span><span class="p">()</span> <span class="ow">or</span> <span class="n">c</span> <span class="ow">in</span> <span class="s2">" -_"</span> <span class="k">else</span> <span class="s2">"_"</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">title</span>
    <span class="p">)</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span>
    <span class="n">note_path</span> <span class="o">=</span> <span class="n">NOTES_DIR</span> <span class="o">/</span> <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">safe_title</span><span class="si">}</span><span class="s2">.md"</span>

    <span class="n">frontmatter</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">"---</span><span class="se">\n</span><span class="s2">title: </span><span class="si">{</span><span class="n">title</span><span class="si">}</span><span class="se">\n</span><span class="s2">tags: </span><span class="si">{</span><span class="n">tags</span><span class="si">}</span><span class="se">\n</span><span class="s2">---</span><span class="se">\n\n</span><span class="s2">"</span>
    <span class="n">note_path</span><span class="o">.</span><span class="n">write_text</span><span class="p">(</span><span class="n">frontmatter</span> <span class="o">+</span> <span class="n">content</span><span class="p">,</span> <span class="n">encoding</span><span class="o">=</span><span class="s2">"utf-8"</span><span class="p">)</span>
    <span class="k">return</span> <span class="sa">f</span><span class="s2">"Note saved to </span><span class="si">{</span><span class="n">note_path</span><span class="si">}</span><span class="s2">"</span>

<span class="nd">@mcp</span><span class="o">.</span><span class="n">tool</span><span class="p">()</span>
<span class="k">def</span><span class="w"> </span><span class="nf">search_notes</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""Search notes by keyword. Searches both titles and content.</span>

<span class="sd">    Returns a list of matching note titles and snippets.</span>
<span class="sd">    Use this before creating a note to check if one already exists.</span>
<span class="sd">    """</span>
    <span class="n">query_lower</span> <span class="o">=</span> <span class="n">query</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span>
    <span class="n">results</span> <span class="o">=</span> <span class="p">[]</span>

    <span class="k">for</span> <span class="n">note_file</span> <span class="ow">in</span> <span class="n">NOTES_DIR</span><span class="o">.</span><span class="n">glob</span><span class="p">(</span><span class="s2">"*.md"</span><span class="p">):</span>
        <span class="n">text</span> <span class="o">=</span> <span class="n">note_file</span><span class="o">.</span><span class="n">read_text</span><span class="p">(</span><span class="n">encoding</span><span class="o">=</span><span class="s2">"utf-8"</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">query_lower</span> <span class="ow">in</span> <span class="n">text</span><span class="o">.</span><span class="n">lower</span><span class="p">():</span>
            <span class="n">idx</span> <span class="o">=</span> <span class="n">text</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span><span class="o">.</span><span class="n">find</span><span class="p">(</span><span class="n">query_lower</span><span class="p">)</span>
            <span class="n">snippet</span> <span class="o">=</span> <span class="n">text</span><span class="p">[</span><span class="nb">max</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">idx</span> <span class="o">-</span> <span class="mi">50</span><span class="p">):</span><span class="n">idx</span> <span class="o">+</span> <span class="mi">100</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s2">"</span><span class="se">\n</span><span class="s2">"</span><span class="p">,</span> <span class="s2">" "</span><span class="p">)</span>
            <span class="n">results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="sa">f</span><span class="s2">"- **</span><span class="si">{</span><span class="n">note_file</span><span class="o">.</span><span class="n">stem</span><span class="si">}</span><span class="s2">**: ...</span><span class="si">{</span><span class="n">snippet</span><span class="si">}</span><span class="s2">..."</span><span class="p">)</span>

    <span class="k">return</span> <span class="s2">"</span><span class="se">\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">results</span><span class="p">)</span> <span class="k">if</span> <span class="n">results</span> <span class="k">else</span> <span class="sa">f</span><span class="s2">"No notes found matching '</span><span class="si">{</span><span class="n">query</span><span class="si">}</span><span class="s2">'"</span>

<span class="c1"># -- Resources（供 LLM 使用的上下文数据） --------------------------------------</span>
<span class="nd">@mcp</span><span class="o">.</span><span class="n">resource</span><span class="p">(</span><span class="s2">"notes://</span><span class="si">{title}</span><span class="s2">"</span><span class="p">)</span>
<span class="k">def</span><span class="w"> </span><span class="nf">get_note</span><span class="p">(</span><span class="n">title</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""Read a note by title."""</span>
    <span class="n">note_path</span> <span class="o">=</span> <span class="n">NOTES_DIR</span> <span class="o">/</span> <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">title</span><span class="si">}</span><span class="s2">.md"</span>
    <span class="k">if</span> <span class="ow">not</span> <span class="n">note_path</span><span class="o">.</span><span class="n">exists</span><span class="p">():</span>
        <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Note not found: </span><span class="si">{</span><span class="n">title</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">note_path</span><span class="o">.</span><span class="n">read_text</span><span class="p">(</span><span class="n">encoding</span><span class="o">=</span><span class="s2">"utf-8"</span><span class="p">)</span>

<span class="c1"># -- 入口 ----------------------------------------------------------------------</span>
<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">"__main__"</span><span class="p">:</span>
    <span class="n">mcp</span><span class="o">.</span><span class="n">run</span><span class="p">()</span>  <span class="c1"># 默认使用 stdio 传输</span>
</code></pre></div>

<p>与较旧的低层 API 相比的关键差异：</p>

<ul>
<li><p><strong>声明式 tool</strong>：<code>@mcp.tool()</code> 装饰器会根据 Python 类型注解和 docstring 推断 JSON Schema——无需手写 <code>inputSchema</code>。</p></li>
<li><p><strong>自动传输</strong>：<code>mcp.run()</code> 会根据 server 的启动方式自动选择 stdio 或 Streamable HTTP。</p></li>
<li><p><strong>以函数表达 resource</strong>：<code>@mcp.resource(&quot;uri-template&quot;)</code> 通过基于 URI 的路由暴露数据。</p></li>
</ul>

## 22.7.2　构建一个 MCP Client {#building-an-mcp-client}

<p>一个最小化的 client，连接到上面的笔记 server 并调用一个 tool：</p>

<p class="code-caption"><span class="code-number">代码 22.2　</span>MCP Client：连接与调用 Tool</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">asyncio</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">mcp</span><span class="w"> </span><span class="kn">import</span> <span class="n">ClientSession</span><span class="p">,</span> <span class="n">StdioServerParameters</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">mcp.client.stdio</span><span class="w"> </span><span class="kn">import</span> <span class="n">stdio_client</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="c1"># 通过 stdio 连接到笔记 server</span>
    <span class="n">server_params</span> <span class="o">=</span> <span class="n">StdioServerParameters</span><span class="p">(</span>
        <span class="n">command</span><span class="o">=</span><span class="s2">"python"</span><span class="p">,</span>
        <span class="n">args</span><span class="o">=</span><span class="p">[</span><span class="s2">"notes_server.py"</span><span class="p">],</span>
        <span class="n">env</span><span class="o">=</span><span class="kc">None</span>  <span class="c1"># 继承环境变量</span>
    <span class="p">)</span>

    <span class="k">async</span> <span class="k">with</span> <span class="n">stdio_client</span><span class="p">(</span><span class="n">server_params</span><span class="p">)</span> <span class="k">as</span> <span class="p">(</span><span class="n">read</span><span class="p">,</span> <span class="n">write</span><span class="p">):</span>
        <span class="k">async</span> <span class="k">with</span> <span class="n">ClientSession</span><span class="p">(</span><span class="n">read</span><span class="p">,</span> <span class="n">write</span><span class="p">)</span> <span class="k">as</span> <span class="n">session</span><span class="p">:</span>
            <span class="c1"># 阶段 1：初始化</span>
            <span class="k">await</span> <span class="n">session</span><span class="o">.</span><span class="n">initialize</span><span class="p">()</span>

            <span class="c1"># 阶段 2：发现可用 tool</span>
            <span class="n">tools_result</span> <span class="o">=</span> <span class="k">await</span> <span class="n">session</span><span class="o">.</span><span class="n">list_tools</span><span class="p">()</span>
            <span class="nb">print</span><span class="p">(</span><span class="s2">"Available tools:"</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">tool</span> <span class="ow">in</span> <span class="n">tools_result</span><span class="o">.</span><span class="n">tools</span><span class="p">:</span>
                <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"  - </span><span class="si">{</span><span class="n">tool</span><span class="o">.</span><span class="n">name</span><span class="si">}</span><span class="s2">: </span><span class="si">{</span><span class="n">tool</span><span class="o">.</span><span class="n">description</span><span class="p">[:</span><span class="mi">60</span><span class="p">]</span><span class="si">}</span><span class="s2">..."</span><span class="p">)</span>

            <span class="c1"># 阶段 3：调用一个 tool</span>
            <span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="n">session</span><span class="o">.</span><span class="n">call_tool</span><span class="p">(</span>
                <span class="s2">"create_note"</span><span class="p">,</span>
                <span class="n">arguments</span><span class="o">=</span><span class="p">{</span>
                    <span class="s2">"title"</span><span class="p">:</span> <span class="s2">"MCP Architecture Notes"</span><span class="p">,</span>
                    <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"MCP uses JSON-RPC 2.0 over stdio or HTTP+SSE."</span><span class="p">,</span>
                    <span class="s2">"tags"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"mcp"</span><span class="p">,</span> <span class="s2">"architecture"</span><span class="p">]</span>
                <span class="p">}</span>
            <span class="p">)</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"</span><span class="se">\n</span><span class="s2">Tool result: </span><span class="si">{</span><span class="n">result</span><span class="o">.</span><span class="n">content</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">text</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

            <span class="c1"># 阶段 4：列出 resource</span>
            <span class="n">resources</span> <span class="o">=</span> <span class="k">await</span> <span class="n">session</span><span class="o">.</span><span class="n">list_resources</span><span class="p">()</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"</span><span class="se">\n</span><span class="s2">Available resources: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">resources</span><span class="o">.</span><span class="n">resources</span><span class="p">)</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

<span class="n">asyncio</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">main</span><span class="p">())</span>
</code></pre></div>

## 22.7.3　同时连接多个 Server {#connecting-to-multiple-servers-simultaneously}

<p>一个 Host 应用通常会管理多条 server 连接。其模式使用一个连接池：</p>

<p class="code-caption"><span class="code-number">代码 22.3　</span>多 Server 的 MCP Host 模式</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">asyncio</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">contextlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">AsyncExitStack</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">mcp</span><span class="w"> </span><span class="kn">import</span> <span class="n">ClientSession</span><span class="p">,</span> <span class="n">StdioServerParameters</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">mcp.client.stdio</span><span class="w"> </span><span class="kn">import</span> <span class="n">stdio_client</span>

<span class="k">class</span><span class="w"> </span><span class="nc">MCPHost</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""Manages connections to multiple MCP servers."""</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">sessions</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">ClientSession</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tool_registry</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">object</span><span class="p">]]</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_exit_stack</span> <span class="o">=</span> <span class="n">AsyncExitStack</span><span class="p">()</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">connect</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">params</span><span class="p">:</span> <span class="n">StdioServerParameters</span><span class="p">):</span>
<span class="w">        </span><span class="sd">"""Connect to a named MCP server and register its tools."""</span>
        <span class="n">read</span><span class="p">,</span> <span class="n">write</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_exit_stack</span><span class="o">.</span><span class="n">enter_async_context</span><span class="p">(</span>
            <span class="n">stdio_client</span><span class="p">(</span><span class="n">params</span><span class="p">)</span>
        <span class="p">)</span>
        <span class="n">session</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_exit_stack</span><span class="o">.</span><span class="n">enter_async_context</span><span class="p">(</span>
            <span class="n">ClientSession</span><span class="p">(</span><span class="n">read</span><span class="p">,</span> <span class="n">write</span><span class="p">)</span>
        <span class="p">)</span>
        <span class="k">await</span> <span class="n">session</span><span class="o">.</span><span class="n">initialize</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">sessions</span><span class="p">[</span><span class="n">name</span><span class="p">]</span> <span class="o">=</span> <span class="n">session</span>

        <span class="c1"># 注册该 server 暴露的所有 tool</span>
        <span class="n">tools</span> <span class="o">=</span> <span class="k">await</span> <span class="n">session</span><span class="o">.</span><span class="n">list_tools</span><span class="p">()</span>
        <span class="k">for</span> <span class="n">tool</span> <span class="ow">in</span> <span class="n">tools</span><span class="o">.</span><span class="n">tools</span><span class="p">:</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">tool_registry</span><span class="p">[</span><span class="n">tool</span><span class="o">.</span><span class="n">name</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="n">tool</span><span class="p">)</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Registered tool '</span><span class="si">{</span><span class="n">tool</span><span class="o">.</span><span class="n">name</span><span class="si">}</span><span class="s2">' from server '</span><span class="si">{</span><span class="n">name</span><span class="si">}</span><span class="s2">'"</span><span class="p">)</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">call_tool</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">tool_name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">arguments</span><span class="p">:</span> <span class="nb">dict</span><span class="p">):</span>
<span class="w">        </span><span class="sd">"""Route a tool call to the appropriate server."""</span>
        <span class="k">if</span> <span class="n">tool_name</span> <span class="ow">not</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">tool_registry</span><span class="p">:</span>
            <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Unknown tool: </span><span class="si">{</span><span class="n">tool_name</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

        <span class="n">server_name</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">tool_registry</span><span class="p">[</span><span class="n">tool_name</span><span class="p">]</span>
        <span class="n">session</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">sessions</span><span class="p">[</span><span class="n">server_name</span><span class="p">]</span>
        <span class="k">return</span> <span class="k">await</span> <span class="n">session</span><span class="o">.</span><span class="n">call_tool</span><span class="p">(</span><span class="n">tool_name</span><span class="p">,</span> <span class="n">arguments</span><span class="p">)</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">get_all_tools</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""Return all tools across all connected servers."""</span>
        <span class="k">return</span> <span class="p">[</span><span class="n">tool</span> <span class="k">for</span> <span class="n">_</span><span class="p">,</span> <span class="n">tool</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">tool_registry</span><span class="o">.</span><span class="n">values</span><span class="p">()]</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">close</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_exit_stack</span><span class="o">.</span><span class="n">aclose</span><span class="p">()</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">host</span> <span class="o">=</span> <span class="n">MCPHost</span><span class="p">()</span>

    <span class="c1"># 并发连接到多个 server</span>
    <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span>
        <span class="n">host</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="s2">"filesystem"</span><span class="p">,</span> <span class="n">StdioServerParameters</span><span class="p">(</span>
            <span class="n">command</span><span class="o">=</span><span class="s2">"npx"</span><span class="p">,</span> <span class="n">args</span><span class="o">=</span><span class="p">[</span><span class="s2">"-y"</span><span class="p">,</span> <span class="s2">"@modelcontextprotocol/server-filesystem"</span><span class="p">,</span>
                                  <span class="s2">"/home/user"</span><span class="p">]</span>
        <span class="p">)),</span>
        <span class="n">host</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="s2">"github"</span><span class="p">,</span> <span class="n">StdioServerParameters</span><span class="p">(</span>
            <span class="n">command</span><span class="o">=</span><span class="s2">"npx"</span><span class="p">,</span> <span class="n">args</span><span class="o">=</span><span class="p">[</span><span class="s2">"-y"</span><span class="p">,</span> <span class="s2">"@modelcontextprotocol/server-github"</span><span class="p">]</span>
        <span class="p">)),</span>
        <span class="n">host</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="s2">"notes"</span><span class="p">,</span> <span class="n">StdioServerParameters</span><span class="p">(</span>
            <span class="n">command</span><span class="o">=</span><span class="s2">"python"</span><span class="p">,</span> <span class="n">args</span><span class="o">=</span><span class="p">[</span><span class="s2">"notes_server.py"</span><span class="p">]</span>
        <span class="p">)),</span>
    <span class="p">)</span>

    <span class="c1"># 所有 tool 都通过同一个接口暴露</span>
    <span class="n">all_tools</span> <span class="o">=</span> <span class="k">await</span> <span class="n">host</span><span class="o">.</span><span class="n">get_all_tools</span><span class="p">()</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Total tools available: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">all_tools</span><span class="p">)</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

    <span class="k">await</span> <span class="n">host</span><span class="o">.</span><span class="n">close</span><span class="p">()</span>

<span class="n">asyncio</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">main</span><span class="p">())</span>
</code></pre></div>

## 22.7.4　错误恢复与重连 {#error-recovery-and-reconnection}

<p>生产级的 MCP client 必须能处理 server 崩溃与网络中断：</p>

<p class="code-caption"><span class="code-number">代码 22.4　</span>带有重试逻辑的健壮 MCP 连接</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">asyncio</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">logging</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">mcp</span><span class="w"> </span><span class="kn">import</span> <span class="n">ClientSession</span><span class="p">,</span> <span class="n">StdioServerParameters</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">mcp.client.stdio</span><span class="w"> </span><span class="kn">import</span> <span class="n">stdio_client</span>

<span class="n">logger</span> <span class="o">=</span> <span class="n">logging</span><span class="o">.</span><span class="n">getLogger</span><span class="p">(</span><span class="vm">__name__</span><span class="p">)</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">resilient_tool_call</span><span class="p">(</span>
    <span class="n">params</span><span class="p">:</span> <span class="n">StdioServerParameters</span><span class="p">,</span>
    <span class="n">tool_name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
    <span class="n">arguments</span><span class="p">:</span> <span class="nb">dict</span><span class="p">,</span>
    <span class="n">max_retries</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">3</span><span class="p">,</span>
    <span class="n">backoff_base</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1.0</span>
<span class="p">):</span>
<span class="w">    </span><span class="sd">"""Call a tool with automatic reconnection on failure."""</span>
    <span class="k">for</span> <span class="n">attempt</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">max_retries</span><span class="p">):</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="k">async</span> <span class="k">with</span> <span class="n">stdio_client</span><span class="p">(</span><span class="n">params</span><span class="p">)</span> <span class="k">as</span> <span class="p">(</span><span class="n">read</span><span class="p">,</span> <span class="n">write</span><span class="p">):</span>
                <span class="k">async</span> <span class="k">with</span> <span class="n">ClientSession</span><span class="p">(</span><span class="n">read</span><span class="p">,</span> <span class="n">write</span><span class="p">)</span> <span class="k">as</span> <span class="n">session</span><span class="p">:</span>
                    <span class="k">await</span> <span class="n">session</span><span class="o">.</span><span class="n">initialize</span><span class="p">()</span>
                    <span class="k">return</span> <span class="k">await</span> <span class="n">session</span><span class="o">.</span><span class="n">call_tool</span><span class="p">(</span><span class="n">tool_name</span><span class="p">,</span> <span class="n">arguments</span><span class="p">)</span>

        <span class="k">except</span> <span class="p">(</span><span class="ne">ConnectionError</span><span class="p">,</span> <span class="ne">TimeoutError</span><span class="p">,</span> <span class="ne">OSError</span><span class="p">)</span> <span class="k">as</span> <span class="n">e</span><span class="p">:</span>
            <span class="k">if</span> <span class="n">attempt</span> <span class="o">==</span> <span class="n">max_retries</span> <span class="o">-</span> <span class="mi">1</span><span class="p">:</span>
                <span class="k">raise</span>
            <span class="n">wait_time</span> <span class="o">=</span> <span class="n">backoff_base</span> <span class="o">*</span> <span class="p">(</span><span class="mi">2</span> <span class="o">**</span> <span class="n">attempt</span><span class="p">)</span>
            <span class="n">logger</span><span class="o">.</span><span class="n">warning</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"Tool call failed (attempt </span><span class="si">{</span><span class="n">attempt</span><span class="o">+</span><span class="mi">1</span><span class="si">}</span><span class="s2">/</span><span class="si">{</span><span class="n">max_retries</span><span class="si">}</span><span class="s2">): </span><span class="si">{</span><span class="n">e</span><span class="si">}</span><span class="s2">. "</span>
                <span class="sa">f</span><span class="s2">"Retrying in </span><span class="si">{</span><span class="n">wait_time</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2">s..."</span>
            <span class="p">)</span>
            <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="n">wait_time</span><span class="p">)</span>
</code></pre></div>
