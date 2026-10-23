# 26.9　完整实现示例：生产级研究型 Agent {#subsec:implementation-example}

<p>下面给出一个基于 LangGraph 构建的完整、生产可用的研究型 Agent，演示 Tool 定义、状态 schema、图构建、错误处理与部署配置。</p>

<div class="admonition example"><p class="admonition-title">生产级研究 Agent 架构</p>
<p>本示例实现的研究 Agent：(1) 接收研究主题，(2) 在网络上检索相关来源，(3) 阅读并综合关键文档，(4) 撰写结构化报告，(5) 通过重试逻辑优雅处理错误。Agent 使用检查点实现可恢复性，使用结构化日志实现可观测性。</p>
</div>

<p class="code-caption"><span class="code-number">代码 26.25　</span>完整生产研究 Agent：Tool 与状态</p>
<div class="highlight"><pre><span></span><code><span class="c1"># === tools.py ===</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">httpx</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">uuid</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">datetime</span><span class="w"> </span><span class="kn">import</span> <span class="n">datetime</span><span class="p">,</span> <span class="n">timezone</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">urllib.parse</span><span class="w"> </span><span class="kn">import</span> <span class="n">urlparse</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_core.tools</span><span class="w"> </span><span class="kn">import</span> <span class="n">tool</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">tenacity</span><span class="w"> </span><span class="kn">import</span> <span class="n">retry</span><span class="p">,</span> <span class="n">stop_after_attempt</span><span class="p">,</span> <span class="n">wait_exponential</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">utils</span><span class="w"> </span><span class="kn">import</span> <span class="n">extract_text</span>  <span class="c1"># HTML -&gt; 纯文本的辅助函数（例如 BeautifulSoup）</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">database</span><span class="w"> </span><span class="kn">import</span> <span class="n">db</span>          <span class="c1"># 应用数据库连接</span>

<span class="nd">@tool</span>
<span class="nd">@retry</span><span class="p">(</span><span class="n">stop</span><span class="o">=</span><span class="n">stop_after_attempt</span><span class="p">(</span><span class="mi">3</span><span class="p">),</span> <span class="n">wait</span><span class="o">=</span><span class="n">wait_exponential</span><span class="p">(</span><span class="nb">min</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="nb">max</span><span class="o">=</span><span class="mi">10</span><span class="p">))</span>
<span class="k">def</span><span class="w"> </span><span class="nf">search_web</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">num_results</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""在网络上搜索信息，返回 JSON 列表形式的结果。"""</span>
    <span class="k">if</span> <span class="ow">not</span> <span class="n">query</span><span class="o">.</span><span class="n">strip</span><span class="p">():</span>
        <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="s2">"Search query cannot be empty"</span><span class="p">)</span>
    <span class="n">response</span> <span class="o">=</span> <span class="n">httpx</span><span class="o">.</span><span class="n">get</span><span class="p">(</span>
        <span class="s2">"https://api.search.example.com/search"</span><span class="p">,</span>
        <span class="n">params</span><span class="o">=</span><span class="p">{</span><span class="s2">"q"</span><span class="p">:</span> <span class="n">query</span><span class="p">,</span> <span class="s2">"n"</span><span class="p">:</span> <span class="n">num_results</span><span class="p">},</span>
        <span class="n">headers</span><span class="o">=</span><span class="p">{</span><span class="s2">"Authorization"</span><span class="p">:</span> <span class="sa">f</span><span class="s2">"Bearer </span><span class="si">{</span><span class="n">os</span><span class="o">.</span><span class="n">environ</span><span class="p">[</span><span class="s1">'SEARCH_API_KEY'</span><span class="p">]</span><span class="si">}</span><span class="s2">"</span><span class="p">},</span>
        <span class="n">timeout</span><span class="o">=</span><span class="mf">10.0</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">response</span><span class="o">.</span><span class="n">raise_for_status</span><span class="p">()</span>
    <span class="n">results</span> <span class="o">=</span> <span class="n">response</span><span class="o">.</span><span class="n">json</span><span class="p">()[</span><span class="s2">"results"</span><span class="p">]</span>
    <span class="k">return</span> <span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">([{</span><span class="s2">"title"</span><span class="p">:</span> <span class="n">r</span><span class="p">[</span><span class="s2">"title"</span><span class="p">],</span> <span class="s2">"url"</span><span class="p">:</span> <span class="n">r</span><span class="p">[</span><span class="s2">"url"</span><span class="p">],</span>
                        <span class="s2">"snippet"</span><span class="p">:</span> <span class="n">r</span><span class="p">[</span><span class="s2">"snippet"</span><span class="p">]}</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">results</span><span class="p">])</span>

<span class="nd">@tool</span>
<span class="nd">@retry</span><span class="p">(</span><span class="n">stop</span><span class="o">=</span><span class="n">stop_after_attempt</span><span class="p">(</span><span class="mi">2</span><span class="p">),</span> <span class="n">wait</span><span class="o">=</span><span class="n">wait_exponential</span><span class="p">(</span><span class="nb">min</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="nb">max</span><span class="o">=</span><span class="mi">5</span><span class="p">))</span>
<span class="k">def</span><span class="w"> </span><span class="nf">fetch_document</span><span class="p">(</span><span class="n">url</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">max_chars</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5000</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""从 URL 拉取并抽取文本内容。"""</span>
    <span class="n">allowed_domains</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">environ</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"ALLOWED_DOMAINS"</span><span class="p">,</span> <span class="s2">""</span><span class="p">)</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">","</span><span class="p">)</span>
    <span class="n">domain</span> <span class="o">=</span> <span class="n">urlparse</span><span class="p">(</span><span class="n">url</span><span class="p">)</span><span class="o">.</span><span class="n">netloc</span>
    <span class="k">if</span> <span class="n">allowed_domains</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="ow">and</span> <span class="n">domain</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">allowed_domains</span><span class="p">:</span>
        <span class="k">raise</span> <span class="ne">PermissionError</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Domain </span><span class="si">{</span><span class="n">domain</span><span class="si">}</span><span class="s2"> not in allowed list"</span><span class="p">)</span>
    <span class="n">response</span> <span class="o">=</span> <span class="n">httpx</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="n">url</span><span class="p">,</span> <span class="n">timeout</span><span class="o">=</span><span class="mf">15.0</span><span class="p">,</span> <span class="n">follow_redirects</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">response</span><span class="o">.</span><span class="n">raise_for_status</span><span class="p">()</span>
    <span class="k">return</span> <span class="n">extract_text</span><span class="p">(</span><span class="n">response</span><span class="o">.</span><span class="n">text</span><span class="p">)[:</span><span class="n">max_chars</span><span class="p">]</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">save_report</span><span class="p">(</span><span class="n">title</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">summary</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">sections</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">])</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""将结构化研究报告保存到数据库。"""</span>
    <span class="n">report_id</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">())</span>
    <span class="n">db</span><span class="o">.</span><span class="n">reports</span><span class="o">.</span><span class="n">insert_one</span><span class="p">({</span>
        <span class="s2">"id"</span><span class="p">:</span> <span class="n">report_id</span><span class="p">,</span> <span class="s2">"title"</span><span class="p">:</span> <span class="n">title</span><span class="p">,</span>
        <span class="s2">"summary"</span><span class="p">:</span> <span class="n">summary</span><span class="p">,</span> <span class="s2">"sections"</span><span class="p">:</span> <span class="n">sections</span><span class="p">,</span>
        <span class="s2">"created_at"</span><span class="p">:</span> <span class="n">datetime</span><span class="o">.</span><span class="n">now</span><span class="p">(</span><span class="n">timezone</span><span class="o">.</span><span class="n">utc</span><span class="p">)</span><span class="o">.</span><span class="n">isoformat</span><span class="p">(),</span>
    <span class="p">})</span>
    <span class="k">return</span> <span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s2">"report_id"</span><span class="p">:</span> <span class="n">report_id</span><span class="p">,</span> <span class="s2">"status"</span><span class="p">:</span> <span class="s2">"saved"</span><span class="p">})</span>

<span class="n">TOOLS</span> <span class="o">=</span> <span class="p">[</span><span class="n">search_web</span><span class="p">,</span> <span class="n">fetch_document</span><span class="p">,</span> <span class="n">save_report</span><span class="p">]</span>
</code></pre></div>

<p class="code-caption"><span class="code-number">代码 26.26　</span>完整生产研究 Agent：状态与节点</p>
<div class="highlight"><pre><span></span><code><span class="c1"># === agent.py ===</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">TypedDict</span><span class="p">,</span> <span class="n">Annotated</span><span class="p">,</span> <span class="n">List</span><span class="p">,</span> <span class="n">Literal</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.graph.message</span><span class="w"> </span><span class="kn">import</span> <span class="n">add_messages</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.prebuilt</span><span class="w"> </span><span class="kn">import</span> <span class="n">ToolNode</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">ChatOpenAI</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_core.messages</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseMessage</span><span class="p">,</span> <span class="n">HumanMessage</span><span class="p">,</span> <span class="n">SystemMessage</span><span class="p">,</span> <span class="n">AIMessage</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">tools</span><span class="w"> </span><span class="kn">import</span> <span class="n">TOOLS</span>

<span class="n">SYSTEM_PROMPT</span> <span class="o">=</span> <span class="s2">"""You are a professional research analyst. Your task is to:</span>
<span class="s2">1. Search for relevant information on the given topic</span>
<span class="s2">2. Read and analyze key sources (aim for 3-5 sources)</span>
<span class="s2">3. Synthesize findings into a structured report using save_report</span>

<span class="s2">Guidelines:</span>
<span class="s2">- Always verify information across multiple sources</span>
<span class="s2">- Cite your sources in the report</span>
<span class="s2">- If a tool fails, try an alternative approach</span>
<span class="s2">- Complete the task in at most 15 tool calls</span>
<span class="s2">- Use save_report exactly once when you have sufficient information"""</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ResearchState</span><span class="p">(</span><span class="n">TypedDict</span><span class="p">):</span>
    <span class="n">messages</span><span class="p">:</span> <span class="n">Annotated</span><span class="p">[</span><span class="n">List</span><span class="p">[</span><span class="n">BaseMessage</span><span class="p">],</span> <span class="n">add_messages</span><span class="p">]</span>
    <span class="n">topic</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">sources_found</span><span class="p">:</span> <span class="n">List</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span>
    <span class="n">sources_read</span><span class="p">:</span> <span class="n">List</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span>
    <span class="n">report_id</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span>
    <span class="n">error_count</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">tool_call_count</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">status</span><span class="p">:</span> <span class="n">Literal</span><span class="p">[</span><span class="s2">"researching"</span><span class="p">,</span> <span class="s2">"done"</span><span class="p">,</span> <span class="s2">"failed"</span><span class="p">]</span>

<span class="n">tool_executor</span> <span class="o">=</span> <span class="n">ToolNode</span><span class="p">(</span><span class="n">TOOLS</span><span class="p">)</span>

<span class="k">def</span><span class="w"> </span><span class="nf">research_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">ResearchState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""主推理节点：由 LLM 决定下一步动作。"""</span>
    <span class="n">llm</span> <span class="o">=</span> <span class="n">ChatOpenAI</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">,</span> <span class="n">temperature</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span><span class="o">.</span><span class="n">bind_tools</span><span class="p">(</span><span class="n">TOOLS</span><span class="p">)</span>
    <span class="n">messages</span> <span class="o">=</span> <span class="p">[</span><span class="n">SystemMessage</span><span class="p">(</span><span class="n">content</span><span class="o">=</span><span class="n">SYSTEM_PROMPT</span><span class="p">)]</span> <span class="o">+</span> <span class="n">state</span><span class="p">[</span><span class="s2">"messages"</span><span class="p">]</span>
    <span class="n">response</span> <span class="o">=</span> <span class="n">llm</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span><span class="n">messages</span><span class="p">)</span>
    <span class="k">return</span> <span class="p">{</span><span class="s2">"messages"</span><span class="p">:</span> <span class="p">[</span><span class="n">response</span><span class="p">]}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">tool_node_with_error_handling</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">ResearchState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""执行 Tool 调用，附带错误处理与状态更新。"""</span>
    <span class="k">try</span><span class="p">:</span>
        <span class="n">result</span> <span class="o">=</span> <span class="n">tool_executor</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>
        <span class="k">return</span> <span class="p">{</span>
            <span class="o">**</span><span class="n">result</span><span class="p">,</span>
            <span class="s2">"tool_call_count"</span><span class="p">:</span> <span class="n">state</span><span class="p">[</span><span class="s2">"tool_call_count"</span><span class="p">]</span> <span class="o">+</span> <span class="nb">len</span><span class="p">(</span>
                <span class="n">state</span><span class="p">[</span><span class="s2">"messages"</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">tool_calls</span>
            <span class="p">),</span>
        <span class="p">}</span>
    <span class="k">except</span> <span class="ne">Exception</span> <span class="k">as</span> <span class="n">e</span><span class="p">:</span>
        <span class="c1"># 用 AIMessage 把错误信号回传给 LLM，便于其调整策略</span>
        <span class="n">error_msg</span> <span class="o">=</span> <span class="n">AIMessage</span><span class="p">(</span><span class="n">content</span><span class="o">=</span><span class="sa">f</span><span class="s2">"Tool execution failed: </span><span class="si">{</span><span class="n">e</span><span class="si">}</span><span class="s2">. Try a different approach."</span><span class="p">)</span>
        <span class="k">return</span> <span class="p">{</span>
            <span class="s2">"messages"</span><span class="p">:</span> <span class="p">[</span><span class="n">error_msg</span><span class="p">],</span>
            <span class="s2">"error_count"</span><span class="p">:</span> <span class="n">state</span><span class="p">[</span><span class="s2">"error_count"</span><span class="p">]</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span>
        <span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">check_completion</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">ResearchState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""检查报告是否已保存，并相应更新状态。"""</span>
    <span class="k">for</span> <span class="n">msg</span> <span class="ow">in</span> <span class="n">state</span><span class="p">[</span><span class="s2">"messages"</span><span class="p">][</span><span class="o">-</span><span class="mi">5</span><span class="p">:]:</span>
        <span class="n">content</span> <span class="o">=</span> <span class="nb">getattr</span><span class="p">(</span><span class="n">msg</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">,</span> <span class="s2">""</span><span class="p">)</span>
        <span class="k">if</span> <span class="s2">"report_id"</span> <span class="ow">in</span> <span class="n">content</span><span class="p">:</span>
            <span class="k">try</span><span class="p">:</span>
                <span class="n">data</span> <span class="o">=</span> <span class="n">json</span><span class="o">.</span><span class="n">loads</span><span class="p">(</span><span class="n">content</span><span class="p">)</span>
                <span class="k">return</span> <span class="p">{</span><span class="s2">"status"</span><span class="p">:</span> <span class="s2">"done"</span><span class="p">,</span> <span class="s2">"report_id"</span><span class="p">:</span> <span class="n">data</span><span class="p">[</span><span class="s2">"report_id"</span><span class="p">]}</span>
            <span class="k">except</span> <span class="p">(</span><span class="n">json</span><span class="o">.</span><span class="n">JSONDecodeError</span><span class="p">,</span> <span class="ne">KeyError</span><span class="p">):</span>
                <span class="k">pass</span>
    <span class="k">return</span> <span class="p">{}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">route_after_llm</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">ResearchState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""根据 LLM 响应决定下一步去向。"""</span>
    <span class="k">if</span> <span class="n">state</span><span class="p">[</span><span class="s2">"error_count"</span><span class="p">]</span> <span class="o">&gt;=</span> <span class="mi">5</span> <span class="ow">or</span> <span class="n">state</span><span class="p">[</span><span class="s2">"tool_call_count"</span><span class="p">]</span> <span class="o">&gt;=</span> <span class="mi">15</span><span class="p">:</span>
        <span class="k">return</span> <span class="s2">"fail"</span>
    <span class="n">last_message</span> <span class="o">=</span> <span class="n">state</span><span class="p">[</span><span class="s2">"messages"</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
    <span class="k">if</span> <span class="nb">hasattr</span><span class="p">(</span><span class="n">last_message</span><span class="p">,</span> <span class="s2">"tool_calls"</span><span class="p">)</span> <span class="ow">and</span> <span class="n">last_message</span><span class="o">.</span><span class="n">tool_calls</span><span class="p">:</span>
        <span class="k">return</span> <span class="s2">"tools"</span>
    <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">state</span><span class="p">[</span><span class="s2">"messages"</span><span class="p">])</span> <span class="o">&gt;</span> <span class="mi">30</span><span class="p">:</span>
        <span class="k">return</span> <span class="s2">"fail"</span>
    <span class="k">return</span> <span class="s2">"research"</span>  <span class="c1"># LLM 需要继续推理</span>

<span class="k">def</span><span class="w"> </span><span class="nf">fail_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">ResearchState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
    <span class="k">return</span> <span class="p">{</span><span class="s2">"status"</span><span class="p">:</span> <span class="s2">"failed"</span><span class="p">}</span>
</code></pre></div>

<p class="code-caption"><span class="code-number">代码 26.27　</span>完整生产研究 Agent：图与部署</p>
<div class="highlight"><pre><span></span><code><span class="c1"># === graph.py ===</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.graph</span><span class="w"> </span><span class="kn">import</span> <span class="n">StateGraph</span><span class="p">,</span> <span class="n">START</span><span class="p">,</span> <span class="n">END</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.graph.state</span><span class="w"> </span><span class="kn">import</span> <span class="n">CompiledStateGraph</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.checkpoint.postgres.aio</span><span class="w"> </span><span class="kn">import</span> <span class="n">AsyncPostgresSaver</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">build_graph</span><span class="p">(</span><span class="n">db_url</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">CompiledStateGraph</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""构建并编译研究 Agent 图。"""</span>
    <span class="n">checkpointer</span> <span class="o">=</span> <span class="n">AsyncPostgresSaver</span><span class="o">.</span><span class="n">from_conn_string</span><span class="p">(</span><span class="n">db_url</span><span class="p">)</span>
    <span class="k">await</span> <span class="n">checkpointer</span><span class="o">.</span><span class="n">setup</span><span class="p">()</span>  <span class="c1"># 如有需要则建表</span>

    <span class="n">builder</span> <span class="o">=</span> <span class="n">StateGraph</span><span class="p">(</span><span class="n">ResearchState</span><span class="p">)</span>

    <span class="c1"># 添加节点</span>
    <span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"research"</span><span class="p">,</span> <span class="n">research_node</span><span class="p">)</span>
    <span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"tools"</span><span class="p">,</span> <span class="n">tool_node_with_error_handling</span><span class="p">)</span>
    <span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"check"</span><span class="p">,</span> <span class="n">check_completion</span><span class="p">)</span>
    <span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"fail"</span><span class="p">,</span> <span class="n">fail_node</span><span class="p">)</span>

    <span class="c1"># 定义边</span>
    <span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="n">START</span><span class="p">,</span> <span class="s2">"research"</span><span class="p">)</span>
    <span class="n">builder</span><span class="o">.</span><span class="n">add_conditional_edges</span><span class="p">(</span>
        <span class="s2">"research"</span><span class="p">,</span>
        <span class="n">route_after_llm</span><span class="p">,</span>
        <span class="p">{</span><span class="s2">"tools"</span><span class="p">:</span> <span class="s2">"tools"</span><span class="p">,</span> <span class="s2">"research"</span><span class="p">:</span> <span class="s2">"research"</span><span class="p">,</span> <span class="s2">"fail"</span><span class="p">:</span> <span class="s2">"fail"</span><span class="p">}</span>
    <span class="p">)</span>
    <span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"tools"</span><span class="p">,</span> <span class="s2">"check"</span><span class="p">)</span>
    <span class="n">builder</span><span class="o">.</span><span class="n">add_conditional_edges</span><span class="p">(</span>
        <span class="s2">"check"</span><span class="p">,</span>
        <span class="k">lambda</span> <span class="n">s</span><span class="p">:</span> <span class="s2">"end"</span> <span class="k">if</span> <span class="n">s</span><span class="p">[</span><span class="s2">"status"</span><span class="p">]</span> <span class="o">==</span> <span class="s2">"done"</span> <span class="k">else</span> <span class="s2">"research"</span><span class="p">,</span>
        <span class="p">{</span><span class="s2">"end"</span><span class="p">:</span> <span class="n">END</span><span class="p">,</span> <span class="s2">"research"</span><span class="p">:</span> <span class="s2">"research"</span><span class="p">}</span>
    <span class="p">)</span>
    <span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"fail"</span><span class="p">,</span> <span class="n">END</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">builder</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">checkpointer</span><span class="o">=</span><span class="n">checkpointer</span><span class="p">)</span>

<span class="c1"># === deployment.py ===</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">uuid</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">contextlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">asynccontextmanager</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">fastapi</span><span class="w"> </span><span class="kn">import</span> <span class="n">FastAPI</span><span class="p">,</span> <span class="n">BackgroundTasks</span><span class="p">,</span> <span class="n">HTTPException</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pydantic</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseModel</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_core.messages</span><span class="w"> </span><span class="kn">import</span> <span class="n">HumanMessage</span>

<span class="n">graph</span><span class="p">:</span> <span class="n">CompiledStateGraph</span> <span class="o">=</span> <span class="kc">None</span>  <span class="c1"># 在启动时初始化</span>

<span class="nd">@asynccontextmanager</span>
<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">lifespan</span><span class="p">(</span><span class="n">app</span><span class="p">:</span> <span class="n">FastAPI</span><span class="p">):</span>
    <span class="k">global</span> <span class="n">graph</span>
    <span class="n">graph</span> <span class="o">=</span> <span class="k">await</span> <span class="n">build_graph</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">environ</span><span class="p">[</span><span class="s2">"DATABASE_URL"</span><span class="p">])</span>
    <span class="k">yield</span>

<span class="n">app</span> <span class="o">=</span> <span class="n">FastAPI</span><span class="p">(</span><span class="n">title</span><span class="o">=</span><span class="s2">"Research Agent API"</span><span class="p">,</span> <span class="n">lifespan</span><span class="o">=</span><span class="n">lifespan</span><span class="p">)</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ResearchRequest</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">topic</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">user_id</span><span class="p">:</span> <span class="nb">str</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ResearchResponse</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">status</span><span class="p">:</span> <span class="nb">str</span>

<span class="nd">@app</span><span class="o">.</span><span class="n">post</span><span class="p">(</span><span class="s2">"/research"</span><span class="p">,</span> <span class="n">response_model</span><span class="o">=</span><span class="n">ResearchResponse</span><span class="p">)</span>
<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">start_research</span><span class="p">(</span><span class="n">request</span><span class="p">:</span> <span class="n">ResearchRequest</span><span class="p">,</span> <span class="n">background_tasks</span><span class="p">:</span> <span class="n">BackgroundTasks</span><span class="p">):</span>
    <span class="n">task_id</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">())</span>
    <span class="n">config</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"configurable"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"thread_id"</span><span class="p">:</span> <span class="n">task_id</span><span class="p">,</span> <span class="s2">"user_id"</span><span class="p">:</span> <span class="n">request</span><span class="o">.</span><span class="n">user_id</span><span class="p">}}</span>
    <span class="n">initial_state</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">"messages"</span><span class="p">:</span> <span class="p">[</span><span class="n">HumanMessage</span><span class="p">(</span><span class="n">content</span><span class="o">=</span><span class="sa">f</span><span class="s2">"Research topic: </span><span class="si">{</span><span class="n">request</span><span class="o">.</span><span class="n">topic</span><span class="si">}</span><span class="s2">"</span><span class="p">)],</span>
        <span class="s2">"topic"</span><span class="p">:</span> <span class="n">request</span><span class="o">.</span><span class="n">topic</span><span class="p">,</span>
        <span class="s2">"sources_found"</span><span class="p">:</span> <span class="p">[],</span> <span class="s2">"sources_read"</span><span class="p">:</span> <span class="p">[],</span>
        <span class="s2">"report_id"</span><span class="p">:</span> <span class="kc">None</span><span class="p">,</span> <span class="s2">"error_count"</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span>
        <span class="s2">"tool_call_count"</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span> <span class="s2">"status"</span><span class="p">:</span> <span class="s2">"researching"</span><span class="p">,</span>
    <span class="p">}</span>
    <span class="n">background_tasks</span><span class="o">.</span><span class="n">add_task</span><span class="p">(</span><span class="n">graph</span><span class="o">.</span><span class="n">ainvoke</span><span class="p">,</span> <span class="n">initial_state</span><span class="p">,</span> <span class="n">config</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">ResearchResponse</span><span class="p">(</span><span class="n">task_id</span><span class="o">=</span><span class="n">task_id</span><span class="p">,</span> <span class="n">status</span><span class="o">=</span><span class="s2">"started"</span><span class="p">)</span>

<span class="nd">@app</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"/research/</span><span class="si">{task_id}</span><span class="s2">"</span><span class="p">)</span>
<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">get_research_status</span><span class="p">(</span><span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
    <span class="n">config</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"configurable"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"thread_id"</span><span class="p">:</span> <span class="n">task_id</span><span class="p">}}</span>
    <span class="n">state</span> <span class="o">=</span> <span class="k">await</span> <span class="n">graph</span><span class="o">.</span><span class="n">aget_state</span><span class="p">(</span><span class="n">config</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">state</span> <span class="ow">is</span> <span class="kc">None</span><span class="p">:</span>
        <span class="k">raise</span> <span class="n">HTTPException</span><span class="p">(</span><span class="n">status_code</span><span class="o">=</span><span class="mi">404</span><span class="p">,</span> <span class="n">detail</span><span class="o">=</span><span class="s2">"Task not found"</span><span class="p">)</span>
    <span class="k">return</span> <span class="p">{</span>
        <span class="s2">"task_id"</span><span class="p">:</span> <span class="n">task_id</span><span class="p">,</span>
        <span class="s2">"status"</span><span class="p">:</span> <span class="n">state</span><span class="o">.</span><span class="n">values</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"status"</span><span class="p">,</span> <span class="s2">"unknown"</span><span class="p">),</span>
        <span class="s2">"report_id"</span><span class="p">:</span> <span class="n">state</span><span class="o">.</span><span class="n">values</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"report_id"</span><span class="p">),</span>
        <span class="s2">"tool_calls"</span><span class="p">:</span> <span class="n">state</span><span class="o">.</span><span class="n">values</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"tool_call_count"</span><span class="p">,</span> <span class="mi">0</span><span class="p">),</span>
        <span class="s2">"error_count"</span><span class="p">:</span> <span class="n">state</span><span class="o">.</span><span class="n">values</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"error_count"</span><span class="p">,</span> <span class="mi">0</span><span class="p">),</span>
    <span class="p">}</span>
</code></pre></div>

<p class="code-caption"><span class="code-number">代码 26.28　</span>部署配置：Docker 与 Kubernetes</p>
<div class="highlight"><pre><span></span><code># === Dockerfile ===
# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# CMD ["uvicorn", "deployment:app", "--host", "0.0.0.0", "--port", "8000"]

# === kubernetes/deployment.yaml（此处以 Python dict 形式示意）===
k8s_deployment = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": "research-agent", "namespace": "agents"},
    "spec": {
        "replicas": 3,
        "selector": {"matchLabels": {"app": "research-agent"}},
        "template": {
            "metadata": {"labels": {"app": "research-agent"}},
            "spec": {
                "containers": [{
                    "name": "agent",
                    "image": "myregistry/research-agent:latest",
                    "ports": [{"containerPort": 8000}],
                    "resources": {
                        "requests": {"memory": "512Mi", "cpu": "250m"},
                        "limits":   {"memory": "2Gi",  "cpu": "1000m"},
                    },
                    "env": [
                        {"name": "DATABASE_URL",   "valueFrom": {
                            "secretKeyRef": {"name": "agent-secrets", "key": "db-url"}}},
                        {"name": "OPENAI_API_KEY", "valueFrom": {
                            "secretKeyRef": {"name": "agent-secrets", "key": "openai-key"}}},
                    ],
                    "livenessProbe":  {"httpGet": {"path": "/health", "port": 8000},
                                       "initialDelaySeconds": 30, "periodSeconds": 10},
                    "readinessProbe": {"httpGet": {"path": "/ready",  "port": 8000},
                                       "initialDelaySeconds": 10, "periodSeconds": 5},
                }]
            }
        }
    }
}

# HorizontalPodAutoscaler 基于队列深度指标进行扩缩
hpa_config = {
    "apiVersion": "autoscaling/v2",
    "kind": "HorizontalPodAutoscaler",
    "metadata": {"name": "research-agent-hpa", "namespace": "agents"},
    "spec": {
        "scaleTargetRef": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "name": "research-agent",
        },
        "minReplicas": 2,
        "maxReplicas": 20,
        "metrics": [{
            "type": "External",
            "external": {
                "metric": {"name": "agent_task_queue_depth"},
                "target": {"type": "AverageValue", "averageValue": "10"},
            }
        }]
    }
}
</code></pre></div>

<div class="admonition warning"><p class="admonition-title">生产检查清单</p>
<p>在将 Agent 部署到生产前，请逐项核对：</p>

<ul>
<li><p>所有 Tool 都具备重试逻辑与错误处理</p></li>
<li><p>已强制设定最大迭代次数</p></li>
<li><p>敏感数据不会被记录到 trace 中</p></li>
<li><p>已按租户配置限流</p></li>
<li><p>长任务已启用检查点</p></li>
<li><p>行为测试通过（无有害输出）</p></li>
<li><p>已校验成本与延迟上限</p></li>
<li><p>回滚流程已文档化并经过演练</p></li>
<li><p>值班手册（runbook）覆盖常见失败模式</p></li>
</ul>
</div>
