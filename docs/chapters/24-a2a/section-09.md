# 24.9　实现示例：多 Agent 研究工作流 {#sec:a2a:implementation}

<p>下面的示例演示了一个完整的、基于 A2A 的多 Agent 研究工作流：一个 <code>OrchestratorAgent</code> 分解研究问题，将子任务委派给专家 Agent，并综合它们的结果。</p>

<div class="highlight"><pre><span></span><code><span class="sd">"""</span>
<span class="sd">使用 A2A 协议实现的多 Agent 研究工作流。</span>
<span class="sd">演示内容：Agent Card、A2A 客户端/服务器、任务生命周期、</span>
<span class="sd">多轮交互以及 Agent 间交接。</span>
<span class="sd">"""</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">asyncio</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">uuid</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">collections.abc</span><span class="w"> </span><span class="kn">import</span> <span class="n">AsyncIterator</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">datetime</span><span class="w"> </span><span class="kn">import</span> <span class="n">datetime</span><span class="p">,</span> <span class="n">timedelta</span><span class="p">,</span> <span class="n">timezone</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">httpx</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">fastapi</span><span class="w"> </span><span class="kn">import</span> <span class="n">FastAPI</span><span class="p">,</span> <span class="n">HTTPException</span><span class="p">,</span> <span class="n">Request</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">fastapi.responses</span><span class="w"> </span><span class="kn">import</span> <span class="n">StreamingResponse</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pydantic</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseModel</span><span class="p">,</span> <span class="n">Field</span>

<span class="c1"># -- 数据模型 ------------------------------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">Part</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="nb">type</span><span class="p">:</span> <span class="nb">str</span>           <span class="c1"># "text" | "file" | "data"</span>
    <span class="n">text</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">data</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">mimeType</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">uri</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>

<span class="k">class</span><span class="w"> </span><span class="nc">Message</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">role</span><span class="p">:</span> <span class="nb">str</span>           <span class="c1"># "user" | "agent"</span>
    <span class="n">parts</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">Part</span><span class="p">]</span>

<span class="k">class</span><span class="w"> </span><span class="nc">TaskStatus</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">state</span><span class="p">:</span> <span class="nb">str</span>          <span class="c1"># submitted | working | input-required | completed | failed</span>
    <span class="n">message</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">timestamp</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="n">Field</span><span class="p">(</span>
        <span class="n">default_factory</span><span class="o">=</span><span class="k">lambda</span><span class="p">:</span> <span class="n">datetime</span><span class="o">.</span><span class="n">now</span><span class="p">(</span><span class="n">timezone</span><span class="o">.</span><span class="n">utc</span><span class="p">)</span><span class="o">.</span><span class="n">isoformat</span><span class="p">()</span>
    <span class="p">)</span>

<span class="k">class</span><span class="w"> </span><span class="nc">Artifact</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">parts</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">Part</span><span class="p">]</span>
    <span class="n">index</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">append</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>
    <span class="n">lastChunk</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span>

<span class="k">class</span><span class="w"> </span><span class="nc">Task</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="nb">id</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">status</span><span class="p">:</span> <span class="n">TaskStatus</span>
    <span class="n">messages</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">Message</span><span class="p">]</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">artifacts</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">Artifact</span><span class="p">]</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">metadata</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">=</span> <span class="p">{}</span>

<span class="c1"># -- A2A 客户端（HTTP/REST 绑定） ---------------------------------------------</span>
<span class="c1"># 注意：A2A v1.0 定义了三种协议绑定：JSON-RPC 2.0、gRPC，</span>
<span class="c1"># 以及 HTTP+JSON/REST。此示例为提高可读性使用 REST 绑定。</span>

<span class="k">class</span><span class="w"> </span><span class="nc">A2AClient</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""用于向兼容 A2A 的 Agent 发送任务的客户端。"""</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">agent_url</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">auth_token</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">agent_url</span> <span class="o">=</span> <span class="n">agent_url</span><span class="o">.</span><span class="n">rstrip</span><span class="p">(</span><span class="s2">"/"</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">headers</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s2">"Authorization"</span><span class="p">:</span> <span class="sa">f</span><span class="s2">"Bearer </span><span class="si">{</span><span class="n">auth_token</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span>
            <span class="s2">"Content-Type"</span><span class="p">:</span> <span class="s2">"application/json"</span>
        <span class="p">}</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">get_agent_card</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""获取该 Agent 的能力卡片。"""</span>
        <span class="k">async</span> <span class="k">with</span> <span class="n">httpx</span><span class="o">.</span><span class="n">AsyncClient</span><span class="p">()</span> <span class="k">as</span> <span class="n">client</span><span class="p">:</span>
            <span class="n">resp</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">get</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">agent_url</span><span class="si">}</span><span class="s2">/.well-known/agent.json"</span><span class="p">,</span>
                <span class="n">headers</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">headers</span>
            <span class="p">)</span>
            <span class="n">resp</span><span class="o">.</span><span class="n">raise_for_status</span><span class="p">()</span>
            <span class="k">return</span> <span class="n">resp</span><span class="o">.</span><span class="n">json</span><span class="p">()</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">send_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">message</span><span class="p">:</span> <span class="n">Message</span><span class="p">,</span>
                        <span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
                        <span class="n">metadata</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Task</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""提交一个任务，返回初始的 Task 对象。"""</span>
        <span class="n">payload</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s2">"id"</span><span class="p">:</span> <span class="n">task_id</span> <span class="ow">or</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">()),</span>
            <span class="s2">"message"</span><span class="p">:</span> <span class="n">message</span><span class="o">.</span><span class="n">model_dump</span><span class="p">(),</span>
            <span class="s2">"metadata"</span><span class="p">:</span> <span class="n">metadata</span> <span class="ow">or</span> <span class="p">{}</span>
        <span class="p">}</span>
        <span class="k">async</span> <span class="k">with</span> <span class="n">httpx</span><span class="o">.</span><span class="n">AsyncClient</span><span class="p">()</span> <span class="k">as</span> <span class="n">client</span><span class="p">:</span>
            <span class="n">resp</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">post</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">agent_url</span><span class="si">}</span><span class="s2">/tasks/send"</span><span class="p">,</span>
                <span class="n">json</span><span class="o">=</span><span class="n">payload</span><span class="p">,</span>
                <span class="n">headers</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">headers</span><span class="p">,</span>
                <span class="n">timeout</span><span class="o">=</span><span class="mf">30.0</span>
            <span class="p">)</span>
            <span class="n">resp</span><span class="o">.</span><span class="n">raise_for_status</span><span class="p">()</span>
            <span class="k">return</span> <span class="n">Task</span><span class="p">(</span><span class="o">**</span><span class="n">resp</span><span class="o">.</span><span class="n">json</span><span class="p">())</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">stream_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">message</span><span class="p">:</span> <span class="n">Message</span><span class="p">,</span>
                          <span class="n">metadata</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AsyncIterator</span><span class="p">[</span><span class="nb">dict</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""提交任务并以 SSE 事件流的形式接收结果。"""</span>
        <span class="n">payload</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s2">"id"</span><span class="p">:</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">()),</span>
            <span class="s2">"message"</span><span class="p">:</span> <span class="n">message</span><span class="o">.</span><span class="n">model_dump</span><span class="p">(),</span>
            <span class="s2">"metadata"</span><span class="p">:</span> <span class="n">metadata</span> <span class="ow">or</span> <span class="p">{}</span>
        <span class="p">}</span>
        <span class="k">async</span> <span class="k">with</span> <span class="n">httpx</span><span class="o">.</span><span class="n">AsyncClient</span><span class="p">()</span> <span class="k">as</span> <span class="n">client</span><span class="p">:</span>
            <span class="k">async</span> <span class="k">with</span> <span class="n">client</span><span class="o">.</span><span class="n">stream</span><span class="p">(</span>
                <span class="s2">"POST"</span><span class="p">,</span>
                <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">agent_url</span><span class="si">}</span><span class="s2">/tasks/sendSubscribe"</span><span class="p">,</span>
                <span class="n">json</span><span class="o">=</span><span class="n">payload</span><span class="p">,</span>
                <span class="n">headers</span><span class="o">=</span><span class="p">{</span><span class="o">**</span><span class="bp">self</span><span class="o">.</span><span class="n">headers</span><span class="p">,</span> <span class="s2">"Accept"</span><span class="p">:</span> <span class="s2">"text/event-stream"</span><span class="p">},</span>
                <span class="n">timeout</span><span class="o">=</span><span class="mf">300.0</span>
            <span class="p">)</span> <span class="k">as</span> <span class="n">response</span><span class="p">:</span>
                <span class="k">async</span> <span class="k">for</span> <span class="n">line</span> <span class="ow">in</span> <span class="n">response</span><span class="o">.</span><span class="n">aiter_lines</span><span class="p">():</span>
                    <span class="k">if</span> <span class="n">line</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"data: "</span><span class="p">):</span>
                        <span class="n">event_data</span> <span class="o">=</span> <span class="n">json</span><span class="o">.</span><span class="n">loads</span><span class="p">(</span><span class="n">line</span><span class="p">[</span><span class="mi">6</span><span class="p">:])</span>
                        <span class="k">yield</span> <span class="n">event_data</span>
                        <span class="k">if</span> <span class="n">event_data</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"final"</span><span class="p">):</span>
                            <span class="k">break</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">get_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Task</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""轮询任务状态。"""</span>
        <span class="k">async</span> <span class="k">with</span> <span class="n">httpx</span><span class="o">.</span><span class="n">AsyncClient</span><span class="p">()</span> <span class="k">as</span> <span class="n">client</span><span class="p">:</span>
            <span class="n">resp</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">get</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">agent_url</span><span class="si">}</span><span class="s2">/tasks/</span><span class="si">{</span><span class="n">task_id</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span>
                <span class="n">headers</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">headers</span>
            <span class="p">)</span>
            <span class="n">resp</span><span class="o">.</span><span class="n">raise_for_status</span><span class="p">()</span>
            <span class="k">return</span> <span class="n">Task</span><span class="p">(</span><span class="o">**</span><span class="n">resp</span><span class="o">.</span><span class="n">json</span><span class="p">())</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">wait_for_completion</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task</span><span class="p">:</span> <span class="n">Task</span><span class="p">,</span>
                                  <span class="n">poll_interval</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">2.0</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Task</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""轮询直至任务到达终止状态。"""</span>
        <span class="n">terminal_states</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"completed"</span><span class="p">,</span> <span class="s2">"failed"</span><span class="p">,</span> <span class="s2">"canceled"</span><span class="p">}</span>
        <span class="k">while</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">terminal_states</span><span class="p">:</span>
            <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="n">poll_interval</span><span class="p">)</span>
            <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">get_task</span><span class="p">(</span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">task</span>

<span class="c1"># -- A2A 服务器（FastAPI） ----------------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ResearchAgent</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    一个专门的研究 Agent，能够检索文献并对给定主题</span>
<span class="sd">    的研究结果进行总结。</span>
<span class="sd">    """</span>

    <span class="n">AGENT_CARD</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"ResearchAgent"</span><span class="p">,</span>
        <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Searches academic literature and synthesizes research findings."</span><span class="p">,</span>
        <span class="s2">"url"</span><span class="p">:</span> <span class="s2">"https://research-agent.example.com/a2a"</span><span class="p">,</span>
        <span class="s2">"version"</span><span class="p">:</span> <span class="s2">"1.0.0"</span><span class="p">,</span>
        <span class="s2">"capabilities"</span><span class="p">:</span> <span class="p">{</span>
            <span class="s2">"streaming"</span><span class="p">:</span> <span class="kc">True</span><span class="p">,</span>
            <span class="s2">"pushNotifications"</span><span class="p">:</span> <span class="kc">False</span><span class="p">,</span>
            <span class="s2">"stateTransitionHistory"</span><span class="p">:</span> <span class="kc">True</span>
        <span class="p">},</span>
        <span class="s2">"authentication"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"schemes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"Bearer"</span><span class="p">]},</span>
        <span class="s2">"skills"</span><span class="p">:</span> <span class="p">[{</span>
            <span class="s2">"id"</span><span class="p">:</span> <span class="s2">"literature-search"</span><span class="p">,</span>
            <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"Literature Search"</span><span class="p">,</span>
            <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Search and summarize academic papers on a topic."</span><span class="p">,</span>
            <span class="s2">"tags"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"research"</span><span class="p">,</span> <span class="s2">"literature"</span><span class="p">,</span> <span class="s2">"academic"</span><span class="p">,</span> <span class="s2">"papers"</span><span class="p">],</span>
            <span class="s2">"examples"</span><span class="p">:</span> <span class="p">[</span>
                <span class="s2">"Summarize recent papers on transformer attention mechanisms."</span><span class="p">,</span>
                <span class="s2">"What does the literature say about RLHF for code generation?"</span>
            <span class="p">],</span>
            <span class="s2">"inputModes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"text"</span><span class="p">],</span>
            <span class="s2">"outputModes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"text"</span><span class="p">,</span> <span class="s2">"data"</span><span class="p">]</span>
        <span class="p">}]</span>
    <span class="p">}</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">Task</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">app</span> <span class="o">=</span> <span class="n">FastAPI</span><span class="p">(</span><span class="n">title</span><span class="o">=</span><span class="s2">"ResearchAgent A2A Server"</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_register_routes</span><span class="p">()</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_register_routes</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="nd">@self</span><span class="o">.</span><span class="n">app</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"/.well-known/agent.json"</span><span class="p">)</span>
        <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">agent_card</span><span class="p">():</span>
            <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">AGENT_CARD</span>

        <span class="nd">@self</span><span class="o">.</span><span class="n">app</span><span class="o">.</span><span class="n">post</span><span class="p">(</span><span class="s2">"/tasks/send"</span><span class="p">)</span>
        <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">send_task</span><span class="p">(</span><span class="n">request</span><span class="p">:</span> <span class="n">Request</span><span class="p">):</span>
            <span class="n">body</span> <span class="o">=</span> <span class="k">await</span> <span class="n">request</span><span class="o">.</span><span class="n">json</span><span class="p">()</span>
            <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_create_and_run_task</span><span class="p">(</span><span class="n">body</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">task</span><span class="o">.</span><span class="n">model_dump</span><span class="p">()</span>

        <span class="nd">@self</span><span class="o">.</span><span class="n">app</span><span class="o">.</span><span class="n">post</span><span class="p">(</span><span class="s2">"/tasks/sendSubscribe"</span><span class="p">)</span>
        <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">send_subscribe</span><span class="p">(</span><span class="n">request</span><span class="p">:</span> <span class="n">Request</span><span class="p">):</span>
            <span class="n">body</span> <span class="o">=</span> <span class="k">await</span> <span class="n">request</span><span class="o">.</span><span class="n">json</span><span class="p">()</span>
            <span class="k">return</span> <span class="n">StreamingResponse</span><span class="p">(</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">_stream_task</span><span class="p">(</span><span class="n">body</span><span class="p">),</span>
                <span class="n">media_type</span><span class="o">=</span><span class="s2">"text/event-stream"</span>
            <span class="p">)</span>

        <span class="nd">@self</span><span class="o">.</span><span class="n">app</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"/tasks/</span><span class="si">{task_id}</span><span class="s2">"</span><span class="p">)</span>
        <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">get_task</span><span class="p">(</span><span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
            <span class="k">if</span> <span class="n">task_id</span> <span class="ow">not</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">:</span>
                <span class="k">raise</span> <span class="n">HTTPException</span><span class="p">(</span><span class="n">status_code</span><span class="o">=</span><span class="mi">404</span><span class="p">,</span> <span class="n">detail</span><span class="o">=</span><span class="s2">"Task not found"</span><span class="p">)</span>
            <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">task_id</span><span class="p">]</span><span class="o">.</span><span class="n">model_dump</span><span class="p">()</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_create_and_run_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">body</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Task</span><span class="p">:</span>
        <span class="n">task_id</span> <span class="o">=</span> <span class="n">body</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"id"</span><span class="p">,</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">()))</span>
        <span class="n">message</span> <span class="o">=</span> <span class="n">Message</span><span class="p">(</span><span class="o">**</span><span class="n">body</span><span class="p">[</span><span class="s2">"message"</span><span class="p">])</span>

        <span class="n">task</span> <span class="o">=</span> <span class="n">Task</span><span class="p">(</span>
            <span class="nb">id</span><span class="o">=</span><span class="n">task_id</span><span class="p">,</span>
            <span class="n">status</span><span class="o">=</span><span class="n">TaskStatus</span><span class="p">(</span><span class="n">state</span><span class="o">=</span><span class="s2">"submitted"</span><span class="p">),</span>
            <span class="n">messages</span><span class="o">=</span><span class="p">[</span><span class="n">message</span><span class="p">],</span>
            <span class="n">metadata</span><span class="o">=</span><span class="n">body</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"metadata"</span><span class="p">,</span> <span class="p">{})</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">task_id</span><span class="p">]</span> <span class="o">=</span> <span class="n">task</span>

        <span class="c1"># 异步执行</span>
        <span class="n">asyncio</span><span class="o">.</span><span class="n">create_task</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_execute_task</span><span class="p">(</span><span class="n">task_id</span><span class="p">))</span>
        <span class="k">return</span> <span class="n">task</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_execute_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
        <span class="n">task</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">task_id</span><span class="p">]</span>
        <span class="n">task</span><span class="o">.</span><span class="n">status</span> <span class="o">=</span> <span class="n">TaskStatus</span><span class="p">(</span><span class="n">state</span><span class="o">=</span><span class="s2">"working"</span><span class="p">)</span>

        <span class="k">try</span><span class="p">:</span>
            <span class="c1"># 从消息中提取研究问题</span>
            <span class="n">question</span> <span class="o">=</span> <span class="n">task</span><span class="o">.</span><span class="n">messages</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">parts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">text</span>

            <span class="c1"># 模拟文献检索（实际生产中替换为真实搜索 Tool）</span>
            <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>  <span class="c1"># 模拟延迟</span>
            <span class="n">findings</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_search_literature</span><span class="p">(</span><span class="n">question</span><span class="p">)</span>

            <span class="c1"># 生成产物</span>
            <span class="n">task</span><span class="o">.</span><span class="n">artifacts</span> <span class="o">=</span> <span class="p">[</span><span class="n">Artifact</span><span class="p">(</span><span class="n">parts</span><span class="o">=</span><span class="p">[</span>
                <span class="n">Part</span><span class="p">(</span><span class="nb">type</span><span class="o">=</span><span class="s2">"text"</span><span class="p">,</span> <span class="n">text</span><span class="o">=</span><span class="n">findings</span><span class="p">[</span><span class="s2">"summary"</span><span class="p">]),</span>
                <span class="n">Part</span><span class="p">(</span><span class="nb">type</span><span class="o">=</span><span class="s2">"data"</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="p">{</span><span class="s2">"papers"</span><span class="p">:</span> <span class="n">findings</span><span class="p">[</span><span class="s2">"papers"</span><span class="p">],</span>
                                        <span class="s2">"query"</span><span class="p">:</span> <span class="n">question</span><span class="p">})</span>
            <span class="p">])]</span>
            <span class="n">task</span><span class="o">.</span><span class="n">status</span> <span class="o">=</span> <span class="n">TaskStatus</span><span class="p">(</span><span class="n">state</span><span class="o">=</span><span class="s2">"completed"</span><span class="p">)</span>

        <span class="k">except</span> <span class="ne">Exception</span> <span class="k">as</span> <span class="n">e</span><span class="p">:</span>
            <span class="n">task</span><span class="o">.</span><span class="n">status</span> <span class="o">=</span> <span class="n">TaskStatus</span><span class="p">(</span><span class="n">state</span><span class="o">=</span><span class="s2">"failed"</span><span class="p">,</span> <span class="n">message</span><span class="o">=</span><span class="nb">str</span><span class="p">(</span><span class="n">e</span><span class="p">))</span>

        <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">task_id</span><span class="p">]</span> <span class="o">=</span> <span class="n">task</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_search_literature</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""占位实现：生产环境中调用真实的搜索 API。"""</span>
        <span class="k">return</span> <span class="p">{</span>
            <span class="s2">"summary"</span><span class="p">:</span> <span class="sa">f</span><span class="s2">"Based on a search of recent literature regarding "</span>
                       <span class="sa">f</span><span class="s2">"'</span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="s2">', key findings include: ..."</span><span class="p">,</span>
            <span class="s2">"papers"</span><span class="p">:</span> <span class="p">[</span>
                <span class="p">{</span><span class="s2">"title"</span><span class="p">:</span> <span class="s2">"Attention Is All You Need"</span><span class="p">,</span> <span class="s2">"year"</span><span class="p">:</span> <span class="mi">2017</span><span class="p">,</span>
                 <span class="s2">"relevance"</span><span class="p">:</span> <span class="mf">0.95</span><span class="p">},</span>
                <span class="p">{</span><span class="s2">"title"</span><span class="p">:</span> <span class="s2">"RLHF: Training Language Models to Follow Instructions"</span><span class="p">,</span>
                 <span class="s2">"year"</span><span class="p">:</span> <span class="mi">2022</span><span class="p">,</span> <span class="s2">"relevance"</span><span class="p">:</span> <span class="mf">0.88</span><span class="p">}</span>
            <span class="p">]</span>
        <span class="p">}</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_stream_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">body</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AsyncIterator</span><span class="p">[</span><span class="nb">str</span><span class="p">]:</span>
        <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_create_and_run_task</span><span class="p">(</span><span class="n">body</span><span class="p">)</span>

        <span class="c1"># 推送状态更新</span>
        <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s1">'id'</span><span class="p">:</span><span class="w"> </span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">,</span><span class="w"> </span><span class="s1">'status'</span><span class="p">:</span><span class="w"> </span><span class="p">{</span><span class="s1">'state'</span><span class="p">:</span><span class="w"> </span><span class="s1">'submitted'</span><span class="p">},</span><span class="w"> </span><span class="s1">'final'</span><span class="p">:</span><span class="w"> </span><span class="kc">False</span><span class="p">})</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>
        <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s1">'id'</span><span class="p">:</span><span class="w"> </span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">,</span><span class="w"> </span><span class="s1">'status'</span><span class="p">:</span><span class="w"> </span><span class="p">{</span><span class="s1">'state'</span><span class="p">:</span><span class="w"> </span><span class="s1">'working'</span><span class="p">},</span><span class="w"> </span><span class="s1">'final'</span><span class="p">:</span><span class="w"> </span><span class="kc">False</span><span class="p">})</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>

        <span class="c1"># 等待完成</span>
        <span class="k">while</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="ow">not</span> <span class="ow">in</span> <span class="p">(</span><span class="s2">"completed"</span><span class="p">,</span> <span class="s2">"failed"</span><span class="p">,</span> <span class="s2">"canceled"</span><span class="p">):</span>
            <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="mf">0.5</span><span class="p">)</span>
            <span class="n">task</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">]</span>

        <span class="c1"># 推送产物</span>
        <span class="k">if</span> <span class="n">task</span><span class="o">.</span><span class="n">artifacts</span><span class="p">:</span>
            <span class="k">for</span> <span class="n">part</span> <span class="ow">in</span> <span class="n">task</span><span class="o">.</span><span class="n">artifacts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">parts</span><span class="p">:</span>
                <span class="n">event</span> <span class="o">=</span> <span class="p">{</span>
                    <span class="s2">"id"</span><span class="p">:</span> <span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">,</span>
                    <span class="s2">"artifact"</span><span class="p">:</span> <span class="p">{</span>
                        <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[</span><span class="n">part</span><span class="o">.</span><span class="n">model_dump</span><span class="p">()],</span>
                        <span class="s2">"index"</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span>
                        <span class="s2">"append"</span><span class="p">:</span> <span class="kc">False</span><span class="p">,</span>
                        <span class="s2">"lastChunk"</span><span class="p">:</span> <span class="kc">True</span>
                    <span class="p">},</span>
                    <span class="s2">"final"</span><span class="p">:</span> <span class="kc">False</span>
                <span class="p">}</span>
                <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">(</span><span class="n">event</span><span class="p">)</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>

        <span class="c1"># 最终状态</span>
        <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s1">'id'</span><span class="p">:</span><span class="w"> </span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">,</span><span class="w"> </span><span class="s1">'status'</span><span class="p">:</span><span class="w"> </span><span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">model_dump</span><span class="p">(),</span><span class="w"> </span><span class="s1">'final'</span><span class="p">:</span><span class="w"> </span><span class="kc">True</span><span class="p">})</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>

<span class="c1"># -- 编排器：多 Agent 工作流 --------------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ResearchOrchestrator</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    编排一个多 Agent 研究工作流：</span>
<span class="sd">    1. 将研究问题分解为子问题</span>
<span class="sd">    2. 将每个子问题分派给 ResearchAgent</span>
<span class="sd">    3. 将结果综合为最终报告</span>
<span class="sd">    """</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">research_agent_url</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">auth_token</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">research_client</span> <span class="o">=</span> <span class="n">A2AClient</span><span class="p">(</span><span class="n">research_agent_url</span><span class="p">,</span> <span class="n">auth_token</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">())</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">run</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">research_question</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"[Orchestrator] Starting workflow </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"[Orchestrator] Question: </span><span class="si">{</span><span class="n">research_question</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

        <span class="c1"># 第 1 步：分解为子问题</span>
        <span class="n">sub_questions</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_decompose</span><span class="p">(</span><span class="n">research_question</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"[Orchestrator] Decomposed into </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">sub_questions</span><span class="p">)</span><span class="si">}</span><span class="s2"> sub-questions"</span><span class="p">)</span>

        <span class="c1"># 第 2 步：并行分派子问题</span>
        <span class="n">tasks</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">research_client</span><span class="o">.</span><span class="n">send_task</span><span class="p">(</span>
                <span class="n">message</span><span class="o">=</span><span class="n">Message</span><span class="p">(</span><span class="n">role</span><span class="o">=</span><span class="s2">"user"</span><span class="p">,</span> <span class="n">parts</span><span class="o">=</span><span class="p">[</span><span class="n">Part</span><span class="p">(</span><span class="nb">type</span><span class="o">=</span><span class="s2">"text"</span><span class="p">,</span> <span class="n">text</span><span class="o">=</span><span class="n">q</span><span class="p">)]),</span>
                <span class="n">metadata</span><span class="o">=</span><span class="p">{</span><span class="s2">"workflowId"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span><span class="p">,</span> <span class="s2">"subQuestion"</span><span class="p">:</span> <span class="n">i</span><span class="p">}</span>
            <span class="p">)</span>
            <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">q</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">sub_questions</span><span class="p">)</span>
        <span class="p">])</span>

        <span class="c1"># 第 3 步：等待所有任务完成</span>
        <span class="n">completed_tasks</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">research_client</span><span class="o">.</span><span class="n">wait_for_completion</span><span class="p">(</span><span class="n">task</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">task</span> <span class="ow">in</span> <span class="n">tasks</span>
        <span class="p">])</span>

        <span class="c1"># 第 4 步：检查失败情况</span>
        <span class="n">failed</span> <span class="o">=</span> <span class="p">[</span><span class="n">t</span> <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="n">completed_tasks</span> <span class="k">if</span> <span class="n">t</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="o">==</span> <span class="s2">"failed"</span><span class="p">]</span>
        <span class="k">if</span> <span class="n">failed</span><span class="p">:</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"[Orchestrator] Warning: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">failed</span><span class="p">)</span><span class="si">}</span><span class="s2"> sub-tasks failed"</span><span class="p">)</span>

        <span class="c1"># 第 5 步：综合结果</span>
        <span class="n">findings</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">task</span><span class="p">,</span> <span class="n">question</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">completed_tasks</span><span class="p">,</span> <span class="n">sub_questions</span><span class="p">):</span>
            <span class="k">if</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="o">==</span> <span class="s2">"completed"</span> <span class="ow">and</span> <span class="n">task</span><span class="o">.</span><span class="n">artifacts</span><span class="p">:</span>
                <span class="n">summary</span> <span class="o">=</span> <span class="n">task</span><span class="o">.</span><span class="n">artifacts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">parts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">text</span>
                <span class="n">findings</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="sa">f</span><span class="s2">"### </span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="se">\n</span><span class="si">{</span><span class="n">summary</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

        <span class="n">report</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_synthesize</span><span class="p">(</span><span class="n">research_question</span><span class="p">,</span> <span class="n">findings</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"[Orchestrator] Workflow complete. Report: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">report</span><span class="p">)</span><span class="si">}</span><span class="s2"> chars"</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">report</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_decompose</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""将复杂问题分解为聚焦的子问题。"""</span>
        <span class="c1"># 生产环境中：使用 LLM 进行分解</span>
        <span class="k">return</span> <span class="p">[</span>
            <span class="sa">f</span><span class="s2">"What are the foundational methods for: </span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="s2">?"</span><span class="p">,</span>
            <span class="sa">f</span><span class="s2">"What are the most recent advances in: </span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="s2">?"</span><span class="p">,</span>
            <span class="sa">f</span><span class="s2">"What are the open challenges and limitations in: </span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="s2">?"</span>
        <span class="p">]</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_synthesize</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">findings</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">])</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""将子结论综合为一份连贯的报告。"""</span>
        <span class="c1"># 生产环境中：使用 LLM 进行综合</span>
        <span class="n">sections</span> <span class="o">=</span> <span class="s2">"</span><span class="se">\n\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">findings</span><span class="p">)</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"# Research Report: </span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="se">\n\n</span><span class="si">{</span><span class="n">sections</span><span class="si">}</span><span class="s2">"</span>

<span class="c1"># -- 入口 ---------------------------------------------------------------------</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">orchestrator</span> <span class="o">=</span> <span class="n">ResearchOrchestrator</span><span class="p">(</span>
        <span class="n">research_agent_url</span><span class="o">=</span><span class="s2">"https://research-agent.example.com/a2a"</span><span class="p">,</span>
        <span class="n">auth_token</span><span class="o">=</span><span class="s2">"eyJhbGciOiJSUzI1NiJ9..."</span>
    <span class="p">)</span>
    <span class="n">report</span> <span class="o">=</span> <span class="k">await</span> <span class="n">orchestrator</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
        <span class="s2">"Reinforcement learning from human feedback for large language models"</span>
    <span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">report</span><span class="p">)</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">"__main__"</span><span class="p">:</span>
    <span class="n">asyncio</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">main</span><span class="p">())</span>
</code></pre></div>
