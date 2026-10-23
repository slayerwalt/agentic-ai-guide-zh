# 18.10　实现：生产级 Agent Harness {#subsec:implementation}

<p>下面是一个完整的、生产级质量的 agent harness 实现，演示了 Context 管理、工具集成、ReAct 编排循环与错误处理。</p>

<p class="code-caption"><span class="code-number">代码 18.1　</span>生产级 Agent Harness —— 核心实现</p>
<div class="highlight"><pre><span></span><code><span class="sd">"""</span>
<span class="sd">production_harness.py -- A production-quality agent harness.</span>
<span class="sd">Demonstrates: context management, tool integration,</span>
<span class="sd">ReAct loop, error handling, and observability.</span>
<span class="sd">"""</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">__future__</span><span class="w"> </span><span class="kn">import</span> <span class="n">annotations</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">asyncio</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">hashlib</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">logging</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">time</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">dataclasses</span><span class="w"> </span><span class="kn">import</span> <span class="n">dataclass</span><span class="p">,</span> <span class="n">field</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">enum</span><span class="w"> </span><span class="kn">import</span> <span class="n">Enum</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Any</span><span class="p">,</span> <span class="n">Callable</span><span class="p">,</span> <span class="n">Optional</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">tiktoken</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">AsyncOpenAI</span>

<span class="c1"># -- 日志 / 可观测性 -----------------------------------------</span>
<span class="n">logger</span> <span class="o">=</span> <span class="n">logging</span><span class="o">.</span><span class="n">getLogger</span><span class="p">(</span><span class="s2">"agent_harness"</span><span class="p">)</span>

<span class="c1"># -- 数据模型 ------------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">Role</span><span class="p">(</span><span class="nb">str</span><span class="p">,</span> <span class="n">Enum</span><span class="p">):</span>
    <span class="n">SYSTEM</span>    <span class="o">=</span> <span class="s2">"system"</span>
    <span class="n">USER</span>      <span class="o">=</span> <span class="s2">"user"</span>
    <span class="n">ASSISTANT</span> <span class="o">=</span> <span class="s2">"assistant"</span>
    <span class="n">TOOL</span>      <span class="o">=</span> <span class="s2">"tool"</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">Message</span><span class="p">:</span>
    <span class="n">role</span><span class="p">:</span>        <span class="n">Role</span>
    <span class="n">content</span><span class="p">:</span>     <span class="nb">str</span>
    <span class="n">tool_calls</span><span class="p">:</span>  <span class="n">Optional</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">]]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">tool_call_id</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span>       <span class="o">=</span> <span class="kc">None</span>
    <span class="n">metadata</span><span class="p">:</span>    <span class="nb">dict</span>                 <span class="o">=</span> <span class="n">field</span><span class="p">(</span><span class="n">default_factory</span><span class="o">=</span><span class="nb">dict</span><span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">to_api_dict</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
        <span class="n">d</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">role</span><span class="o">.</span><span class="n">value</span><span class="p">,</span>
                   <span class="s2">"content"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">content</span> <span class="ow">or</span> <span class="kc">None</span><span class="p">}</span>
        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">tool_calls</span><span class="p">:</span>
            <span class="n">d</span><span class="p">[</span><span class="s2">"tool_calls"</span><span class="p">]</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">tool_calls</span>
        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">tool_call_id</span><span class="p">:</span>
            <span class="n">d</span><span class="p">[</span><span class="s2">"tool_call_id"</span><span class="p">]</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">tool_call_id</span>
        <span class="k">return</span> <span class="n">d</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">ToolDefinition</span><span class="p">:</span>
    <span class="n">name</span><span class="p">:</span>        <span class="nb">str</span>
    <span class="n">description</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">parameters</span><span class="p">:</span>  <span class="nb">dict</span>
    <span class="n">handler</span><span class="p">:</span>     <span class="n">Callable</span>
    <span class="n">requires_approval</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">to_api_dict</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
        <span class="k">return</span> <span class="p">{</span>
            <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"function"</span><span class="p">,</span>
            <span class="s2">"function"</span><span class="p">:</span> <span class="p">{</span>
                <span class="s2">"name"</span><span class="p">:</span>        <span class="bp">self</span><span class="o">.</span><span class="n">name</span><span class="p">,</span>
                <span class="s2">"description"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">description</span><span class="p">,</span>
                <span class="s2">"parameters"</span><span class="p">:</span>  <span class="bp">self</span><span class="o">.</span><span class="n">parameters</span><span class="p">,</span>
            <span class="p">}</span>
        <span class="p">}</span>

<span class="c1"># -- Context 管理器 ------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ContextManager</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    Manages the context window with budget enforcement,</span>
<span class="sd">    compression, and token counting.</span>
<span class="sd">    """</span>
    <span class="n">BUDGET_FRACTIONS</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">"system"</span><span class="p">:</span>   <span class="mf">0.10</span><span class="p">,</span>
        <span class="s2">"memory"</span><span class="p">:</span>   <span class="mf">0.20</span><span class="p">,</span>
        <span class="s2">"tools"</span><span class="p">:</span>    <span class="mf">0.10</span><span class="p">,</span>
        <span class="s2">"history"</span><span class="p">:</span>  <span class="mf">0.50</span><span class="p">,</span>
        <span class="s2">"reserved"</span><span class="p">:</span> <span class="mf">0.10</span><span class="p">,</span>
    <span class="p">}</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">model</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">max_tokens</span><span class="p">:</span> <span class="nb">int</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">model</span>      <span class="o">=</span> <span class="n">model</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">max_tokens</span> <span class="o">=</span> <span class="n">max_tokens</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">enc</span>        <span class="o">=</span> <span class="n">tiktoken</span><span class="o">.</span><span class="n">encoding_for_model</span><span class="p">(</span><span class="n">model</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">history</span><span class="p">:</span>   <span class="nb">list</span><span class="p">[</span><span class="n">Message</span><span class="p">]</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">system_msg</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="n">Message</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">count_tokens</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">text</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">int</span><span class="p">:</span>
        <span class="k">return</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">enc</span><span class="o">.</span><span class="n">encode</span><span class="p">(</span><span class="n">text</span><span class="p">))</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">count_message_tokens</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">msg</span><span class="p">:</span> <span class="n">Message</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">int</span><span class="p">:</span>
        <span class="c1"># OpenAI 开销：每条消息 4 个 token + role</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">count_tokens</span><span class="p">(</span><span class="n">msg</span><span class="o">.</span><span class="n">content</span> <span class="ow">or</span> <span class="s2">""</span><span class="p">)</span> <span class="o">+</span> <span class="mi">4</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">total_history_tokens</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">int</span><span class="p">:</span>
        <span class="k">return</span> <span class="nb">sum</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">count_message_tokens</span><span class="p">(</span><span class="n">m</span><span class="p">)</span>
                   <span class="k">for</span> <span class="n">m</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">history</span><span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">history_budget</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">int</span><span class="p">:</span>
        <span class="k">return</span> <span class="nb">int</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">max_tokens</span>
                   <span class="o">*</span> <span class="bp">self</span><span class="o">.</span><span class="n">BUDGET_FRACTIONS</span><span class="p">[</span><span class="s2">"history"</span><span class="p">])</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">add_message</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">msg</span><span class="p">:</span> <span class="n">Message</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="kc">None</span><span class="p">:</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">history</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">msg</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_enforce_budget</span><span class="p">()</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_enforce_budget</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="kc">None</span><span class="p">:</span>
        <span class="n">budget</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">history_budget</span><span class="p">()</span>
        <span class="k">while</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">total_history_tokens</span><span class="p">()</span> <span class="o">&gt;</span> <span class="n">budget</span>
               <span class="ow">and</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">history</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mi">2</span><span class="p">):</span>
            <span class="c1"># 丢弃最旧的非固定（non-pinned）消息（索引 1）。</span>
            <span class="c1"># 如果它带有 tool_calls，同时丢弃其后的工具结果，</span>
            <span class="c1"># 以保持对话有效。</span>
            <span class="n">dropped</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">history</span><span class="o">.</span><span class="n">pop</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
            <span class="k">if</span> <span class="n">dropped</span><span class="o">.</span><span class="n">tool_calls</span><span class="p">:</span>
                <span class="k">while</span> <span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">history</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mi">1</span>
                       <span class="ow">and</span> <span class="bp">self</span><span class="o">.</span><span class="n">history</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">role</span> <span class="o">==</span> <span class="n">Role</span><span class="o">.</span><span class="n">TOOL</span><span class="p">):</span>
                    <span class="bp">self</span><span class="o">.</span><span class="n">history</span><span class="o">.</span><span class="n">pop</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
        <span class="n">logger</span><span class="o">.</span><span class="n">debug</span><span class="p">(</span>
            <span class="s2">"Context: </span><span class="si">%d</span><span class="s2">/</span><span class="si">%d</span><span class="s2"> tokens used"</span><span class="p">,</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">total_history_tokens</span><span class="p">(),</span> <span class="n">budget</span>
        <span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">preflight_check</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">tool_tokens</span><span class="p">:</span> <span class="nb">int</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">bool</span><span class="p">:</span>
        <span class="c1"># 如果当前在预算范围内则返回 True。</span>
<span class="w">        </span><span class="sd">"""Returns True if we are within budget."""</span>
        <span class="n">sys_tokens</span> <span class="o">=</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">count_message_tokens</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">system_msg</span><span class="p">)</span>
                      <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">system_msg</span> <span class="k">else</span> <span class="mi">0</span><span class="p">)</span>
        <span class="n">total</span> <span class="o">=</span> <span class="p">(</span><span class="n">sys_tokens</span>
                 <span class="o">+</span> <span class="n">tool_tokens</span>
                 <span class="o">+</span> <span class="bp">self</span><span class="o">.</span><span class="n">total_history_tokens</span><span class="p">())</span>
        <span class="n">reserved</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">max_tokens</span>
                       <span class="o">*</span> <span class="bp">self</span><span class="o">.</span><span class="n">BUDGET_FRACTIONS</span><span class="p">[</span><span class="s2">"reserved"</span><span class="p">])</span>
        <span class="n">ok</span> <span class="o">=</span> <span class="n">total</span> <span class="o">&lt;=</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">max_tokens</span> <span class="o">-</span> <span class="n">reserved</span><span class="p">)</span>
        <span class="k">if</span> <span class="ow">not</span> <span class="n">ok</span><span class="p">:</span>
            <span class="n">logger</span><span class="o">.</span><span class="n">warning</span><span class="p">(</span>
                <span class="s2">"Context overflow: </span><span class="si">%d</span><span class="s2"> &gt; </span><span class="si">%d</span><span class="s2">"</span><span class="p">,</span>
                <span class="n">total</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">max_tokens</span> <span class="o">-</span> <span class="n">reserved</span>
            <span class="p">)</span>
        <span class="k">return</span> <span class="n">ok</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">build_messages</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">]:</span>
        <span class="n">msgs</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">system_msg</span><span class="p">:</span>
            <span class="n">msgs</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">system_msg</span><span class="o">.</span><span class="n">to_api_dict</span><span class="p">())</span>
        <span class="n">msgs</span><span class="o">.</span><span class="n">extend</span><span class="p">(</span><span class="n">m</span><span class="o">.</span><span class="n">to_api_dict</span><span class="p">()</span> <span class="k">for</span> <span class="n">m</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">history</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">msgs</span>

<span class="c1"># -- 工具执行器 ----------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ToolExecutor</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    Executes tool calls with sandboxing, retry logic,</span>
<span class="sd">    and output truncation.</span>
<span class="sd">    """</span>
    <span class="n">MAX_OUTPUT_TOKENS</span> <span class="o">=</span> <span class="mi">2000</span>
    <span class="n">MAX_RETRIES</span>       <span class="o">=</span> <span class="mi">3</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">tools</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">ToolDefinition</span><span class="p">],</span>
                 <span class="n">approval_callback</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="n">Callable</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
                 <span class="n">encoding</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">"cl100k_base"</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tools</span>    <span class="o">=</span> <span class="p">{</span><span class="n">t</span><span class="o">.</span><span class="n">name</span><span class="p">:</span> <span class="n">t</span> <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="n">tools</span><span class="p">}</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">approval</span> <span class="o">=</span> <span class="n">approval_callback</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">enc</span>      <span class="o">=</span> <span class="n">tiktoken</span><span class="o">.</span><span class="n">get_encoding</span><span class="p">(</span><span class="n">encoding</span><span class="p">)</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">execute</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">tool_name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                      <span class="n">args</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="n">tool</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">tools</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="n">tool_name</span><span class="p">)</span>
        <span class="k">if</span> <span class="ow">not</span> <span class="n">tool</span><span class="p">:</span>
            <span class="k">return</span> <span class="sa">f</span><span class="s2">"Error: unknown tool '</span><span class="si">{</span><span class="n">tool_name</span><span class="si">}</span><span class="s2">'"</span>

        <span class="c1"># 人机协同（Human-in-the-loop）审批门</span>
        <span class="k">if</span> <span class="n">tool</span><span class="o">.</span><span class="n">requires_approval</span> <span class="ow">and</span> <span class="bp">self</span><span class="o">.</span><span class="n">approval</span><span class="p">:</span>
            <span class="n">approved</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">approval</span><span class="p">(</span><span class="n">tool_name</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
            <span class="k">if</span> <span class="ow">not</span> <span class="n">approved</span><span class="p">:</span>
                <span class="k">return</span> <span class="s2">"Action rejected by human reviewer."</span>

        <span class="k">for</span> <span class="n">attempt</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">MAX_RETRIES</span><span class="p">):</span>
            <span class="k">try</span><span class="p">:</span>
                <span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">wait_for</span><span class="p">(</span>
                    <span class="bp">self</span><span class="o">.</span><span class="n">_call</span><span class="p">(</span><span class="n">tool</span><span class="p">,</span> <span class="n">args</span><span class="p">),</span> <span class="n">timeout</span><span class="o">=</span><span class="mf">30.0</span>
                <span class="p">)</span>
                <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">_truncate</span><span class="p">(</span><span class="n">result</span><span class="p">)</span>
            <span class="k">except</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">TimeoutError</span><span class="p">:</span>
                <span class="n">logger</span><span class="o">.</span><span class="n">warning</span><span class="p">(</span><span class="s2">"Tool </span><span class="si">%s</span><span class="s2"> timed out (attempt </span><span class="si">%d</span><span class="s2">)"</span><span class="p">,</span>
                               <span class="n">tool_name</span><span class="p">,</span> <span class="n">attempt</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span>
                <span class="k">if</span> <span class="n">attempt</span> <span class="o">==</span> <span class="bp">self</span><span class="o">.</span><span class="n">MAX_RETRIES</span> <span class="o">-</span> <span class="mi">1</span><span class="p">:</span>
                    <span class="k">return</span> <span class="sa">f</span><span class="s2">"Error: tool '</span><span class="si">{</span><span class="n">tool_name</span><span class="si">}</span><span class="s2">' timed out"</span>
                <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="mi">2</span> <span class="o">**</span> <span class="n">attempt</span><span class="p">)</span>  <span class="c1"># 退避</span>
            <span class="k">except</span> <span class="ne">Exception</span> <span class="k">as</span> <span class="n">exc</span><span class="p">:</span>
                <span class="n">logger</span><span class="o">.</span><span class="n">error</span><span class="p">(</span><span class="s2">"Tool </span><span class="si">%s</span><span class="s2"> error: </span><span class="si">%s</span><span class="s2">"</span><span class="p">,</span> <span class="n">tool_name</span><span class="p">,</span> <span class="n">exc</span><span class="p">)</span>
                <span class="k">if</span> <span class="n">attempt</span> <span class="o">==</span> <span class="bp">self</span><span class="o">.</span><span class="n">MAX_RETRIES</span> <span class="o">-</span> <span class="mi">1</span><span class="p">:</span>
                    <span class="k">return</span> <span class="sa">f</span><span class="s2">"Error: </span><span class="si">{</span><span class="n">exc</span><span class="si">}</span><span class="s2">"</span>
                <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="mi">2</span> <span class="o">**</span> <span class="n">attempt</span><span class="p">)</span>
        <span class="k">return</span> <span class="s2">"Error: max retries exceeded"</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_call</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">tool</span><span class="p">:</span> <span class="n">ToolDefinition</span><span class="p">,</span>
                    <span class="n">args</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="k">if</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">iscoroutinefunction</span><span class="p">(</span><span class="n">tool</span><span class="o">.</span><span class="n">handler</span><span class="p">):</span>
            <span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="n">tool</span><span class="o">.</span><span class="n">handler</span><span class="p">(</span><span class="o">**</span><span class="n">args</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">get_running_loop</span><span class="p">()</span><span class="o">.</span><span class="n">run_in_executor</span><span class="p">(</span>
                <span class="kc">None</span><span class="p">,</span> <span class="k">lambda</span><span class="p">:</span> <span class="n">tool</span><span class="o">.</span><span class="n">handler</span><span class="p">(</span><span class="o">**</span><span class="n">args</span><span class="p">)</span>
            <span class="p">)</span>
        <span class="k">return</span> <span class="nb">str</span><span class="p">(</span><span class="n">result</span><span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_truncate</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">text</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="n">tokens</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">enc</span><span class="o">.</span><span class="n">encode</span><span class="p">(</span><span class="n">text</span><span class="p">)</span>
        <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">tokens</span><span class="p">)</span> <span class="o">&lt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">MAX_OUTPUT_TOKENS</span><span class="p">:</span>
            <span class="k">return</span> <span class="n">text</span>
        <span class="n">truncated</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">enc</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span>
            <span class="n">tokens</span><span class="p">[:</span><span class="bp">self</span><span class="o">.</span><span class="n">MAX_OUTPUT_TOKENS</span><span class="p">]</span>
        <span class="p">)</span>
        <span class="k">return</span> <span class="n">truncated</span> <span class="o">+</span> <span class="s2">"</span><span class="se">\n</span><span class="s2">[... output truncated ...]"</span>

<span class="c1"># -- 循环检测器 ----------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">LoopDetector</span><span class="p">:</span>
    <span class="c1"># 在滑动窗口内检测重复动作。</span>
<span class="w">    </span><span class="sd">"""Detects repeated actions within a sliding window."""</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">window</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span> <span class="n">max_repeats</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">2</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">window</span>      <span class="o">=</span> <span class="n">window</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">max_repeats</span> <span class="o">=</span> <span class="n">max_repeats</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">action_hashes</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="p">[]</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">record</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">tool_name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">args</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">bool</span><span class="p">:</span>
        <span class="c1"># 如果检测到循环则返回 True。</span>
<span class="w">        </span><span class="sd">"""Returns True if a loop is detected."""</span>
        <span class="n">h</span> <span class="o">=</span> <span class="n">hashlib</span><span class="o">.</span><span class="n">md5</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">tool_name</span><span class="si">}</span><span class="s2">:</span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">(</span><span class="n">args</span><span class="p">,</span><span class="w"> </span><span class="n">sort_keys</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="si">}</span><span class="s2">"</span>
            <span class="o">.</span><span class="n">encode</span><span class="p">()</span>
        <span class="p">)</span><span class="o">.</span><span class="n">hexdigest</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">action_hashes</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">h</span><span class="p">)</span>
        <span class="n">recent</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">action_hashes</span><span class="p">[</span><span class="o">-</span><span class="bp">self</span><span class="o">.</span><span class="n">window</span><span class="p">:]</span>
        <span class="k">if</span> <span class="n">recent</span><span class="o">.</span><span class="n">count</span><span class="p">(</span><span class="n">h</span><span class="p">)</span> <span class="o">&gt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">max_repeats</span><span class="p">:</span>
            <span class="n">logger</span><span class="o">.</span><span class="n">warning</span><span class="p">(</span><span class="s2">"Loop detected: </span><span class="si">%s</span><span class="s2"> called </span><span class="si">%d</span><span class="s2"> times"</span><span class="p">,</span>
                           <span class="n">tool_name</span><span class="p">,</span> <span class="n">recent</span><span class="o">.</span><span class="n">count</span><span class="p">(</span><span class="n">h</span><span class="p">))</span>
            <span class="k">return</span> <span class="kc">True</span>
        <span class="k">return</span> <span class="kc">False</span>

<span class="c1"># -- Agent Harness -------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">AgentHarness</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    Production agent harness implementing the ReAct loop</span>
<span class="sd">    with full context management, tool integration,</span>
<span class="sd">    error handling, and observability.</span>
<span class="sd">    """</span>
    <span class="n">MAX_ITERATIONS</span> <span class="o">=</span> <span class="mi">50</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span>
        <span class="n">model</span><span class="p">:</span>        <span class="nb">str</span><span class="p">,</span>
        <span class="n">system_prompt</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
        <span class="n">tools</span><span class="p">:</span>        <span class="nb">list</span><span class="p">[</span><span class="n">ToolDefinition</span><span class="p">],</span>
        <span class="n">max_tokens</span><span class="p">:</span>   <span class="nb">int</span> <span class="o">=</span> <span class="mi">128_000</span><span class="p">,</span>
        <span class="n">approval_cb</span><span class="p">:</span>  <span class="n">Optional</span><span class="p">[</span><span class="n">Callable</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
        <span class="n">client</span><span class="p">:</span>       <span class="n">Optional</span><span class="p">[</span><span class="n">AsyncOpenAI</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
    <span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">model</span>   <span class="o">=</span> <span class="n">model</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">client</span>  <span class="o">=</span> <span class="n">client</span> <span class="ow">or</span> <span class="n">AsyncOpenAI</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">ctx_mgr</span> <span class="o">=</span> <span class="n">ContextManager</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">max_tokens</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">executor</span> <span class="o">=</span> <span class="n">ToolExecutor</span><span class="p">(</span><span class="n">tools</span><span class="p">,</span> <span class="n">approval_cb</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">loop_det</span> <span class="o">=</span> <span class="n">LoopDetector</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tools</span>    <span class="o">=</span> <span class="n">tools</span>

        <span class="c1"># 设置 system 消息</span>
        <span class="n">sys_msg</span> <span class="o">=</span> <span class="n">Message</span><span class="p">(</span><span class="n">Role</span><span class="o">.</span><span class="n">SYSTEM</span><span class="p">,</span> <span class="n">system_prompt</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">ctx_mgr</span><span class="o">.</span><span class="n">system_msg</span> <span class="o">=</span> <span class="n">sys_msg</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">run</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">user_input</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""</span>
<span class="sd">        Execute the ReAct loop for a user request.</span>
<span class="sd">        Returns the final response string.</span>
<span class="sd">        """</span>
        <span class="n">run_id</span>   <span class="o">=</span> <span class="n">hashlib</span><span class="o">.</span><span class="n">md5</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">()</span><span class="si">}</span><span class="s2">:</span><span class="si">{</span><span class="n">user_input</span><span class="si">}</span><span class="s2">"</span><span class="o">.</span><span class="n">encode</span><span class="p">()</span>
        <span class="p">)</span><span class="o">.</span><span class="n">hexdigest</span><span class="p">()[:</span><span class="mi">8</span><span class="p">]</span>
        <span class="n">start_ts</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">monotonic</span><span class="p">()</span>
        <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="s2">"[</span><span class="si">%s</span><span class="s2">] Starting run: </span><span class="si">%s</span><span class="s2">"</span><span class="p">,</span> <span class="n">run_id</span><span class="p">,</span>
                    <span class="n">user_input</span><span class="p">[:</span><span class="mi">80</span><span class="p">])</span>

        <span class="c1"># 将 user 消息加入 context</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">ctx_mgr</span><span class="o">.</span><span class="n">add_message</span><span class="p">(</span>
            <span class="n">Message</span><span class="p">(</span><span class="n">Role</span><span class="o">.</span><span class="n">USER</span><span class="p">,</span> <span class="n">user_input</span><span class="p">)</span>
        <span class="p">)</span>

        <span class="n">tool_defs</span> <span class="o">=</span> <span class="p">[</span><span class="n">t</span><span class="o">.</span><span class="n">to_api_dict</span><span class="p">()</span> <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">tools</span><span class="p">]</span>
        <span class="n">tool_tokens</span> <span class="o">=</span> <span class="nb">sum</span><span class="p">(</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">ctx_mgr</span><span class="o">.</span><span class="n">count_tokens</span><span class="p">(</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">(</span><span class="n">t</span><span class="p">))</span>
            <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="n">tool_defs</span>
        <span class="p">)</span>

        <span class="k">for</span> <span class="n">iteration</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">MAX_ITERATIONS</span><span class="p">):</span>
            <span class="c1"># 预飞行 context 检查</span>
            <span class="k">if</span> <span class="ow">not</span> <span class="bp">self</span><span class="o">.</span><span class="n">ctx_mgr</span><span class="o">.</span><span class="n">preflight_check</span><span class="p">(</span><span class="n">tool_tokens</span><span class="p">):</span>
                <span class="n">logger</span><span class="o">.</span><span class="n">error</span><span class="p">(</span><span class="s2">"[</span><span class="si">%s</span><span class="s2">] Context overflow at iter </span><span class="si">%d</span><span class="s2">"</span><span class="p">,</span>
                             <span class="n">run_id</span><span class="p">,</span> <span class="n">iteration</span><span class="p">)</span>
                <span class="k">return</span> <span class="p">(</span><span class="s2">"I've run out of context space. "</span>
                        <span class="s2">"Please start a new conversation."</span><span class="p">)</span>

            <span class="c1"># -- LLM 调用 ---------------------------------</span>
            <span class="n">messages</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">ctx_mgr</span><span class="o">.</span><span class="n">build_messages</span><span class="p">()</span>
            <span class="k">try</span><span class="p">:</span>
                <span class="n">response</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">client</span><span class="o">.</span><span class="n">chat</span><span class="o">.</span><span class="n">completions</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
                    <span class="n">model</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">model</span><span class="p">,</span>
                    <span class="n">messages</span><span class="o">=</span><span class="n">messages</span><span class="p">,</span>
                    <span class="n">tools</span><span class="o">=</span><span class="n">tool_defs</span> <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">tools</span> <span class="k">else</span> <span class="kc">None</span><span class="p">,</span>
                    <span class="n">tool_choice</span><span class="o">=</span><span class="s2">"auto"</span><span class="p">,</span>
                    <span class="n">temperature</span><span class="o">=</span><span class="mf">0.0</span><span class="p">,</span>
                <span class="p">)</span>
            <span class="k">except</span> <span class="ne">Exception</span> <span class="k">as</span> <span class="n">exc</span><span class="p">:</span>
                <span class="n">logger</span><span class="o">.</span><span class="n">error</span><span class="p">(</span><span class="s2">"[</span><span class="si">%s</span><span class="s2">] LLM call failed: </span><span class="si">%s</span><span class="s2">"</span><span class="p">,</span>
                             <span class="n">run_id</span><span class="p">,</span> <span class="n">exc</span><span class="p">)</span>
                <span class="k">return</span> <span class="sa">f</span><span class="s2">"I encountered an error: </span><span class="si">{</span><span class="n">exc</span><span class="si">}</span><span class="s2">"</span>

            <span class="n">choice</span>  <span class="o">=</span> <span class="n">response</span><span class="o">.</span><span class="n">choices</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
            <span class="n">msg</span>     <span class="o">=</span> <span class="n">choice</span><span class="o">.</span><span class="n">message</span>
            <span class="n">finish</span>  <span class="o">=</span> <span class="n">choice</span><span class="o">.</span><span class="n">finish_reason</span>

            <span class="c1"># 保存 assistant 消息</span>
            <span class="n">assistant_msg</span> <span class="o">=</span> <span class="n">Message</span><span class="p">(</span>
                <span class="n">role</span><span class="o">=</span><span class="n">Role</span><span class="o">.</span><span class="n">ASSISTANT</span><span class="p">,</span>
                <span class="n">content</span><span class="o">=</span><span class="n">msg</span><span class="o">.</span><span class="n">content</span> <span class="ow">or</span> <span class="s2">""</span><span class="p">,</span>
                <span class="n">tool_calls</span><span class="o">=</span><span class="p">([</span><span class="n">tc</span><span class="o">.</span><span class="n">model_dump</span><span class="p">()</span>
                             <span class="k">for</span> <span class="n">tc</span> <span class="ow">in</span> <span class="n">msg</span><span class="o">.</span><span class="n">tool_calls</span><span class="p">]</span>
                            <span class="k">if</span> <span class="n">msg</span><span class="o">.</span><span class="n">tool_calls</span> <span class="k">else</span> <span class="kc">None</span><span class="p">),</span>
            <span class="p">)</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">ctx_mgr</span><span class="o">.</span><span class="n">add_message</span><span class="p">(</span><span class="n">assistant_msg</span><span class="p">)</span>

            <span class="c1"># -- 终止条件 ----------------------------------</span>
            <span class="k">if</span> <span class="n">finish</span> <span class="o">==</span> <span class="s2">"stop"</span> <span class="ow">or</span> <span class="ow">not</span> <span class="n">msg</span><span class="o">.</span><span class="n">tool_calls</span><span class="p">:</span>
                <span class="n">elapsed</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">monotonic</span><span class="p">()</span> <span class="o">-</span> <span class="n">start_ts</span>
                <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span>
                    <span class="s2">"[</span><span class="si">%s</span><span class="s2">] Done in </span><span class="si">%d</span><span class="s2"> iters, </span><span class="si">%.2f</span><span class="s2">s"</span><span class="p">,</span>
                    <span class="n">run_id</span><span class="p">,</span> <span class="n">iteration</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span> <span class="n">elapsed</span>
                <span class="p">)</span>
                <span class="k">return</span> <span class="n">msg</span><span class="o">.</span><span class="n">content</span> <span class="ow">or</span> <span class="s2">"Task complete."</span>

            <span class="c1"># -- 工具执行 ----------------------------------</span>
            <span class="n">tool_results</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_execute_tool_calls</span><span class="p">(</span>
                <span class="n">msg</span><span class="o">.</span><span class="n">tool_calls</span><span class="p">,</span> <span class="n">run_id</span>
            <span class="p">)</span>

            <span class="c1"># 检查是否进入循环</span>
            <span class="k">for</span> <span class="n">tc</span> <span class="ow">in</span> <span class="n">msg</span><span class="o">.</span><span class="n">tool_calls</span><span class="p">:</span>
                <span class="n">args</span> <span class="o">=</span> <span class="n">json</span><span class="o">.</span><span class="n">loads</span><span class="p">(</span><span class="n">tc</span><span class="o">.</span><span class="n">function</span><span class="o">.</span><span class="n">arguments</span><span class="p">)</span>
                <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">loop_det</span><span class="o">.</span><span class="n">record</span><span class="p">(</span><span class="n">tc</span><span class="o">.</span><span class="n">function</span><span class="o">.</span><span class="n">name</span><span class="p">,</span> <span class="n">args</span><span class="p">):</span>
                    <span class="k">return</span> <span class="p">(</span><span class="s2">"I seem to be stuck in a loop. "</span>
                            <span class="s2">"Please clarify your request."</span><span class="p">)</span>

            <span class="c1"># 将工具结果加入 context</span>
            <span class="k">for</span> <span class="n">tool_call_id</span><span class="p">,</span> <span class="n">result</span> <span class="ow">in</span> <span class="n">tool_results</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">ctx_mgr</span><span class="o">.</span><span class="n">add_message</span><span class="p">(</span><span class="n">Message</span><span class="p">(</span>
                    <span class="n">role</span><span class="o">=</span><span class="n">Role</span><span class="o">.</span><span class="n">TOOL</span><span class="p">,</span>
                    <span class="n">content</span><span class="o">=</span><span class="n">result</span><span class="p">,</span>
                    <span class="n">tool_call_id</span><span class="o">=</span><span class="n">tool_call_id</span><span class="p">,</span>
                <span class="p">))</span>

        <span class="c1"># 达到最大迭代次数</span>
        <span class="n">logger</span><span class="o">.</span><span class="n">warning</span><span class="p">(</span><span class="s2">"[</span><span class="si">%s</span><span class="s2">] Max iterations reached"</span><span class="p">,</span> <span class="n">run_id</span><span class="p">)</span>
        <span class="k">return</span> <span class="p">(</span><span class="s2">"I reached the maximum number of steps "</span>
                <span class="s2">"without completing the task. "</span>
                <span class="s2">"Here is what I found so far: "</span>
                <span class="o">+</span> <span class="p">(</span><span class="n">msg</span><span class="o">.</span><span class="n">content</span> <span class="ow">or</span> <span class="s2">""</span><span class="p">))</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_execute_tool_calls</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span>
        <span class="n">tool_calls</span><span class="p">:</span> <span class="nb">list</span><span class="p">,</span>
        <span class="n">run_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
    <span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">str</span><span class="p">]:</span>
        <span class="c1"># 并行执行工具调用。</span>
<span class="w">        </span><span class="sd">"""Execute tool calls in parallel."""</span>
        <span class="n">tasks</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="k">for</span> <span class="n">tc</span> <span class="ow">in</span> <span class="n">tool_calls</span><span class="p">:</span>
            <span class="n">name</span> <span class="o">=</span> <span class="n">tc</span><span class="o">.</span><span class="n">function</span><span class="o">.</span><span class="n">name</span>
            <span class="k">try</span><span class="p">:</span>
                <span class="n">args</span> <span class="o">=</span> <span class="n">json</span><span class="o">.</span><span class="n">loads</span><span class="p">(</span><span class="n">tc</span><span class="o">.</span><span class="n">function</span><span class="o">.</span><span class="n">arguments</span><span class="p">)</span>
            <span class="k">except</span> <span class="n">json</span><span class="o">.</span><span class="n">JSONDecodeError</span><span class="p">:</span>
                <span class="n">args</span> <span class="o">=</span> <span class="p">{}</span>
            <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="s2">"[</span><span class="si">%s</span><span class="s2">] Tool call: </span><span class="si">%s</span><span class="s2">(</span><span class="si">%s</span><span class="s2">)"</span><span class="p">,</span>
                        <span class="n">run_id</span><span class="p">,</span> <span class="n">name</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
            <span class="n">tasks</span><span class="p">[</span><span class="n">tc</span><span class="o">.</span><span class="n">id</span><span class="p">]</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">executor</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>

        <span class="n">results</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span>
            <span class="o">*</span><span class="n">tasks</span><span class="o">.</span><span class="n">values</span><span class="p">(),</span> <span class="n">return_exceptions</span><span class="o">=</span><span class="kc">True</span>
        <span class="p">)</span>
        <span class="n">output</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="k">for</span> <span class="n">tool_id</span><span class="p">,</span> <span class="n">result</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">tasks</span><span class="o">.</span><span class="n">keys</span><span class="p">(),</span> <span class="n">results</span><span class="p">):</span>
            <span class="k">if</span> <span class="nb">isinstance</span><span class="p">(</span><span class="n">result</span><span class="p">,</span> <span class="ne">Exception</span><span class="p">):</span>
                <span class="n">output</span><span class="p">[</span><span class="n">tool_id</span><span class="p">]</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">"Error: </span><span class="si">{</span><span class="n">result</span><span class="si">}</span><span class="s2">"</span>
            <span class="k">else</span><span class="p">:</span>
                <span class="n">output</span><span class="p">[</span><span class="n">tool_id</span><span class="p">]</span> <span class="o">=</span> <span class="n">result</span>
        <span class="k">return</span> <span class="n">output</span>

<span class="c1"># -- 示例用法 ------------------------------------------------</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="c1"># 定义工具</span>
    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">search_web</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                         <span class="n">num_results</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="c1"># 在生产环境中：调用真实的搜索 API</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"[Search results for '</span><span class="si">{</span><span class="n">query</span><span class="si">}</span><span class="s2">': ...]"</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">run_python</span><span class="p">(</span><span class="n">code</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="c1"># 在生产环境中：在沙箱容器中执行</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"[Execution result of code: ...]"</span>

    <span class="n">tools</span> <span class="o">=</span> <span class="p">[</span>
        <span class="n">ToolDefinition</span><span class="p">(</span>
            <span class="n">name</span><span class="o">=</span><span class="s2">"search_web"</span><span class="p">,</span>
            <span class="n">description</span><span class="o">=</span><span class="p">(</span>
                <span class="s2">"Search the web for current information. "</span>
                <span class="s2">"Use when the user asks about recent events "</span>
                <span class="s2">"or facts beyond your knowledge cutoff."</span>
            <span class="p">),</span>
            <span class="n">parameters</span><span class="o">=</span><span class="p">{</span>
                <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"object"</span><span class="p">,</span>
                <span class="s2">"properties"</span><span class="p">:</span> <span class="p">{</span>
                    <span class="s2">"query"</span><span class="p">:</span> <span class="p">{</span>
                        <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"string"</span><span class="p">,</span>
                        <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Search query"</span>
                    <span class="p">},</span>
                    <span class="s2">"num_results"</span><span class="p">:</span> <span class="p">{</span>
                        <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"integer"</span><span class="p">,</span>
                        <span class="s2">"default"</span><span class="p">:</span> <span class="mi">5</span>
                    <span class="p">},</span>
                <span class="p">},</span>
                <span class="s2">"required"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"query"</span><span class="p">],</span>
            <span class="p">},</span>
            <span class="n">handler</span><span class="o">=</span><span class="n">search_web</span><span class="p">,</span>
        <span class="p">),</span>
        <span class="n">ToolDefinition</span><span class="p">(</span>
            <span class="n">name</span><span class="o">=</span><span class="s2">"run_python"</span><span class="p">,</span>
            <span class="n">description</span><span class="o">=</span><span class="p">(</span>
                <span class="s2">"Execute Python code in a sandbox. "</span>
                <span class="s2">"Use for calculations, data processing, "</span>
                <span class="s2">"or generating visualizations."</span>
            <span class="p">),</span>
            <span class="n">parameters</span><span class="o">=</span><span class="p">{</span>
                <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"object"</span><span class="p">,</span>
                <span class="s2">"properties"</span><span class="p">:</span> <span class="p">{</span>
                    <span class="s2">"code"</span><span class="p">:</span> <span class="p">{</span>
                        <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"string"</span><span class="p">,</span>
                        <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Python code to execute"</span>
                    <span class="p">},</span>
                <span class="p">},</span>
                <span class="s2">"required"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"code"</span><span class="p">],</span>
            <span class="p">},</span>
            <span class="n">handler</span><span class="o">=</span><span class="n">run_python</span><span class="p">,</span>
            <span class="n">requires_approval</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>  <span class="c1"># 需要人工签核</span>
        <span class="p">),</span>
    <span class="p">]</span>

    <span class="n">harness</span> <span class="o">=</span> <span class="n">AgentHarness</span><span class="p">(</span>
        <span class="n">model</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">,</span>
        <span class="n">system_prompt</span><span class="o">=</span><span class="p">(</span>
            <span class="s2">"You are a helpful research assistant. "</span>
            <span class="s2">"Think step by step before acting. "</span>
            <span class="s2">"Always cite your sources."</span>
        <span class="p">),</span>
        <span class="n">tools</span><span class="o">=</span><span class="n">tools</span><span class="p">,</span>
        <span class="n">max_tokens</span><span class="o">=</span><span class="mi">128_000</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="n">response</span> <span class="o">=</span> <span class="k">await</span> <span class="n">harness</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
        <span class="s2">"What were the key AI research breakthroughs "</span>
        <span class="s2">"in the first half of 2025?"</span>
    <span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">response</span><span class="p">)</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">"__main__"</span><span class="p">:</span>
    <span class="n">asyncio</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">main</span><span class="p">())</span>
</code></pre></div>

<div class="admonition example"><p class="admonition-title">实现中的关键设计决策</p>
<ul>
<li><p><strong>Context 强制约束</strong> 在每次 <code>add_message</code> 调用时执行，而不仅仅是在 LLM 调用之前。这可防止静默溢出。</p></li>
<li><p><strong>并行工具执行</strong> 通过 <code>asyncio.gather</code>，在模型同时请求多个工具时降低延迟。</p></li>
<li><p><strong>循环检测</strong> 在滑动窗口上使用内容哈希，可同时捕获精确重复和近似重复。</p></li>
<li><p><strong>审批门</strong> 按工具粒度设置，而非按运行设置，从而对哪些动作需要人工签核进行细粒度控制。</p></li>
<li><p><strong>结构化日志</strong> 带有 <code>run_id</code>，便于在分布式日志中追踪单次 Agent 运行。</p></li>
<li><p><strong>指数退避</strong> 应用于工具层而非 LLM 层，因为工具失败更常见也更易恢复。</p></li>
</ul>
</div>

<div class="admonition question"><p class="admonition-title">如何测试一个 Agent Harness？</p>
<p>测试 Agent 与测试确定性软件有本质区别。关键策略：(1) <strong>单元测试</strong>：在隔离环境中以 mock 依赖测试每个组件（context manager、tool executor、loop detector）。(2) <strong>集成测试</strong>：用一个返回脚本化响应的 mock LLM 对完整 harness 进行测试。(3) <strong>评测 harness</strong>：在带有已知正确答案的任务基准上运行 Agent 并测量成功率。(4) <strong>对抗性测试</strong>：故意注入形态异常的工具输出并验证优雅失败。(5) <strong>回归测试</strong>：回放过去的生产追踪并验证更改后输出未退化。</p>
</div>

## 小结 {#summary}

<p>agent harness 是将语言模型转变为能干、可靠 Agent 的工程基础。本节的关键要点：</p>

<ul>
<li><p><strong>Context 是有限且珍贵的资源。</strong> 显式强制预算，使用模型精确的 tokenizer 计数 Token，主动压缩历史。</p></li>
<li><p><strong>Prompt 就是代码。</strong> 对其进行版本控制、测试，并从组件模块化地组装。</p></li>
<li><p><strong>工具是 Agent 的执行器。</strong> 精确地定义它们，沙箱化其执行，并以防御性方式处理其输出。</p></li>
<li><p><strong>编排模式不能一刀切。</strong> 探索性任务用 ReAct，结构化任务用 Plan-and-Execute，复杂的可分解任务用多 Agent。</p></li>
<li><p><strong>状态管理是一等关注点。</strong> 提前设计状态 schema；事后改造非常痛苦。</p></li>
<li><p><strong>错误是不可避免的；优雅恢复是一项特性。</strong> 实现重试逻辑、循环检测以及富含信息的失败消息。</p></li>
<li><p><strong>可观测性不是可选项。</strong> 你无法调试看不见的东西。从第一天就对所有事物进行埋点。</p></li>
<li><p><strong>生产环境的关注点会复合。</strong> 延迟、成本、速率限制与评测彼此交互。请系统化地解决它们，而不是事后补救。</p></li>
</ul>
