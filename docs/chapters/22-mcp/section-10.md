# 22.10　MCP 在 Agent 训练中的作用 {#sec:mcp:training}

<p>除了部署，MCP 对<em>训练</em>使用工具的 Agent 也有重要意义。本节探讨 MCP 如何作为 LLM 强化学习与监督微调的基础设施。</p>

## 22.10.1　MCP Server 作为 RL 环境接口 {#mcp-servers-as-rl-environment-interfaces}

<p>在面向 LLM 的强化学习中（见第 <a href="../03-reinforcement-learning/index.md#sec:rl">3</a> 节），Agent 必须与环境交互以获得 reward。MCP server 为此提供了一个自然且标准化的接口：</p>

<ul>
<li><p><strong>动作空间</strong>：可用 tool 的集合定义了 Agent 的动作空间。MCP 的 <code>tools/list</code> 端点提供了一个结构化、机器可读且可动态更新的动作空间。</p></li>
<li><p><strong>观测空间</strong>：MCP resource 提供结构化的观测。一个编程环境可能把当前文件内容、测试结果与错误信息都暴露为 resource。</p></li>
<li><p><strong>Reward 信号</strong>：工具调用结果可以编码 reward 信号。一个跑测试的 tool 可能会在测试输出之外返回 <code>{&quot;passed&quot;: 8, &quot;failed&quot;: 2, &quot;reward&quot;: 0.8}</code>。</p></li>
<li><p><strong>环境重置</strong>：一个 <code>reset_environment</code> tool 可以在 episode 之间把环境恢复到初始状态。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">把 SWE-bench 实现为一个 MCP 环境</p>
<p>SWE-bench 基准（来自真实 GitHub issue 的软件工程任务）可以被实现为一个 MCP server：</p>

<ul>
<li><p><strong>Tools</strong>：<code>read_file</code>、<code>write_file</code>、<code>run_tests</code>、<code>apply_patch</code>、<code>search_codebase</code></p></li>
<li><p><strong>Resources</strong>：当前文件树、失败测试输出、issue 描述</p></li>
<li><p><strong>Reward</strong>：Agent 修改后通过测试的比例</p></li>
</ul>

<p>任何能讲 MCP 的 RL 训练框架，无需任何定制环境代码即可在 SWE-bench 上训练。</p>
</div>

## 22.10.2　借助 MCP 实现标准化动作空间 {#standardized-action-spaces-via-mcp}

<p>训练使用工具的 Agent 时，一个挑战在于不同环境拥有不同的动作空间，使得已学到的 policy 难以迁移。MCP 提供了一种<strong>通用的动作空间抽象</strong>：</p>



<div class="arithmatex">\[\mathcal{A}_{\text{MCP}} = \bigcup_{s \in \mathcal{S}} \text{Tools}(s)
\tag{22.3}\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{S}\)</span> 为已连接的 MCP server 集合，<span class="arithmatex">\(\text{Tools}(s)\)</span> 是 server <span class="arithmatex">\(s\)</span> 暴露的 tool 集合。Agent 学到的是一个以可用动作集合为条件的 policy <span class="arithmatex">\(\pi(a \mid o, \mathcal{A}_{\text{MCP}})\)</span>，从而能够零样本泛化到新的 tool 集合。</p>

<p>工具参数的 JSON Schema 格式提供了一种 LLM 可以可靠解析与生成的<strong>结构化动作表示</strong>。这比自由格式的 API 文档更易处理，并使得在训练期间对动作空间进行系统性探索成为可能。</p>

## 22.10.3　为 SFT 记录工具使用轨迹 {#recording-tool-use-trajectories-for-sft}

<p>MCP 的结构化协议让记录高质量的工具使用轨迹（用于监督微调）变得很容易：</p>

<p class="code-caption"><span class="code-number">代码 22.5　</span>用于 SFT 数据采集的轨迹记录中间件</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">json</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">time</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">dataclasses</span><span class="w"> </span><span class="kn">import</span> <span class="n">dataclass</span><span class="p">,</span> <span class="n">field</span><span class="p">,</span> <span class="n">asdict</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Any</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">mcp</span><span class="w"> </span><span class="kn">import</span> <span class="n">ClientSession</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">ToolCallRecord</span><span class="p">:</span>
    <span class="n">timestamp</span><span class="p">:</span> <span class="nb">float</span>
    <span class="n">tool_name</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">arguments</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">Any</span><span class="p">]</span>
    <span class="n">result</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">Any</span><span class="p">]</span>
    <span class="n">duration_ms</span><span class="p">:</span> <span class="nb">float</span>
    <span class="n">is_error</span><span class="p">:</span> <span class="nb">bool</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">Trajectory</span><span class="p">:</span>
    <span class="n">task_description</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">tool_calls</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">ToolCallRecord</span><span class="p">]</span> <span class="o">=</span> <span class="n">field</span><span class="p">(</span><span class="n">default_factory</span><span class="o">=</span><span class="nb">list</span><span class="p">)</span>
    <span class="n">final_answer</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">""</span>
    <span class="n">success</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>
    <span class="n">total_reward</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.0</span>

<span class="k">class</span><span class="w"> </span><span class="nc">RecordingMCPClient</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""Wraps an MCP session to record all tool calls for SFT data."""</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">session</span><span class="p">:</span> <span class="n">ClientSession</span><span class="p">,</span> <span class="n">trajectory</span><span class="p">:</span> <span class="n">Trajectory</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">session</span> <span class="o">=</span> <span class="n">session</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">trajectory</span> <span class="o">=</span> <span class="n">trajectory</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">call_tool</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">arguments</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Any</span><span class="p">:</span>
        <span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">monotonic</span><span class="p">()</span>
        <span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">session</span><span class="o">.</span><span class="n">call_tool</span><span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="n">arguments</span><span class="p">)</span>
        <span class="n">duration</span> <span class="o">=</span> <span class="p">(</span><span class="n">time</span><span class="o">.</span><span class="n">monotonic</span><span class="p">()</span> <span class="o">-</span> <span class="n">start</span><span class="p">)</span> <span class="o">*</span> <span class="mi">1000</span>

        <span class="bp">self</span><span class="o">.</span><span class="n">trajectory</span><span class="o">.</span><span class="n">tool_calls</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">ToolCallRecord</span><span class="p">(</span>
            <span class="n">timestamp</span><span class="o">=</span><span class="n">time</span><span class="o">.</span><span class="n">time</span><span class="p">(),</span>
            <span class="n">tool_name</span><span class="o">=</span><span class="n">name</span><span class="p">,</span>
            <span class="n">arguments</span><span class="o">=</span><span class="n">arguments</span><span class="p">,</span>
            <span class="n">result</span><span class="o">=</span><span class="p">{</span><span class="s2">"content"</span><span class="p">:</span> <span class="p">[</span><span class="n">c</span><span class="o">.</span><span class="n">text</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">result</span><span class="o">.</span><span class="n">content</span>
                                 <span class="k">if</span> <span class="nb">hasattr</span><span class="p">(</span><span class="n">c</span><span class="p">,</span> <span class="s2">"text"</span><span class="p">)]},</span>
            <span class="n">duration_ms</span><span class="o">=</span><span class="n">duration</span><span class="p">,</span>
            <span class="n">is_error</span><span class="o">=</span><span class="n">result</span><span class="o">.</span><span class="n">isError</span>
        <span class="p">))</span>
        <span class="k">return</span> <span class="n">result</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">save_trajectory</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">path</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
        <span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">path</span><span class="p">,</span> <span class="s2">"w"</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
            <span class="n">json</span><span class="o">.</span><span class="n">dump</span><span class="p">(</span><span class="n">asdict</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">trajectory</span><span class="p">),</span> <span class="n">f</span><span class="p">,</span> <span class="n">indent</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
</code></pre></div>

<p>被记录的轨迹可以转化为遵循指令式的训练样本：</p>

<p class="code-caption"><span class="code-number">代码 22.6　</span>将 MCP 轨迹转换为 SFT 训练样本</p>
<div class="highlight"><pre><span></span><code><span class="k">def</span><span class="w"> </span><span class="nf">trajectory_to_sft_example</span><span class="p">(</span><span class="n">traj</span><span class="p">:</span> <span class="n">Trajectory</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""Convert a recorded MCP trajectory to a chat-format SFT example."""</span>
    <span class="n">messages</span> <span class="o">=</span> <span class="p">[</span>
        <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"system"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="p">(</span>
            <span class="s2">"You are a helpful assistant with access to tools. "</span>
            <span class="s2">"Use tools to complete tasks step by step."</span>
        <span class="p">)},</span>
        <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">traj</span><span class="o">.</span><span class="n">task_description</span><span class="p">}</span>
    <span class="p">]</span>

    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">call</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">traj</span><span class="o">.</span><span class="n">tool_calls</span><span class="p">):</span>
        <span class="n">call_id</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">"call_</span><span class="si">{</span><span class="n">i</span><span class="si">:</span><span class="s2">04d</span><span class="si">}</span><span class="s2">"</span>
        <span class="c1"># 助手决定调用某个 tool</span>
        <span class="n">messages</span><span class="o">.</span><span class="n">append</span><span class="p">({</span>
            <span class="s2">"role"</span><span class="p">:</span> <span class="s2">"assistant"</span><span class="p">,</span>
            <span class="s2">"content"</span><span class="p">:</span> <span class="kc">None</span><span class="p">,</span>
            <span class="s2">"tool_calls"</span><span class="p">:</span> <span class="p">[{</span>
                <span class="s2">"id"</span><span class="p">:</span> <span class="n">call_id</span><span class="p">,</span>
                <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"function"</span><span class="p">,</span>
                <span class="s2">"function"</span><span class="p">:</span> <span class="p">{</span>
                    <span class="s2">"name"</span><span class="p">:</span> <span class="n">call</span><span class="o">.</span><span class="n">tool_name</span><span class="p">,</span>
                    <span class="s2">"arguments"</span><span class="p">:</span> <span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">(</span><span class="n">call</span><span class="o">.</span><span class="n">arguments</span><span class="p">)</span>
                <span class="p">}</span>
            <span class="p">}]</span>
        <span class="p">})</span>
        <span class="c1"># 工具返回结果</span>
        <span class="n">messages</span><span class="o">.</span><span class="n">append</span><span class="p">({</span>
            <span class="s2">"role"</span><span class="p">:</span> <span class="s2">"tool"</span><span class="p">,</span>
            <span class="s2">"content"</span><span class="p">:</span> <span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">(</span><span class="n">call</span><span class="o">.</span><span class="n">result</span><span class="p">),</span>
            <span class="s2">"tool_call_id"</span><span class="p">:</span> <span class="n">call_id</span><span class="p">,</span>
        <span class="p">})</span>

    <span class="c1"># 最终答案</span>
    <span class="n">messages</span><span class="o">.</span><span class="n">append</span><span class="p">({</span>
        <span class="s2">"role"</span><span class="p">:</span> <span class="s2">"assistant"</span><span class="p">,</span>
        <span class="s2">"content"</span><span class="p">:</span> <span class="n">traj</span><span class="o">.</span><span class="n">final_answer</span>
    <span class="p">})</span>

    <span class="k">return</span> <span class="p">{</span>
        <span class="s2">"messages"</span><span class="p">:</span> <span class="n">messages</span><span class="p">,</span>
        <span class="s2">"metadata"</span><span class="p">:</span> <span class="p">{</span>
            <span class="s2">"success"</span><span class="p">:</span> <span class="n">traj</span><span class="o">.</span><span class="n">success</span><span class="p">,</span>
            <span class="s2">"reward"</span><span class="p">:</span> <span class="n">traj</span><span class="o">.</span><span class="n">total_reward</span><span class="p">,</span>
            <span class="s2">"num_tool_calls"</span><span class="p">:</span> <span class="nb">len</span><span class="p">(</span><span class="n">traj</span><span class="o">.</span><span class="n">tool_calls</span><span class="p">)</span>
        <span class="p">}</span>
    <span class="p">}</span>
</code></pre></div>

<div class="admonition question"><p class="admonition-title">MCP 能否成为工具使用 Agent 的通用 Gym？</p>
<p>MCP 是否可以成为工具使用 LLM 训练的 <code>gymnasium</code>（即旧版 OpenAI Gym）？这一类比颇具吸引力：正如 Gym 为机器人与博弈 Agent 标准化了 RL 环境，MCP 也有望为语言 Agent 标准化工具环境。关键的开放问题包括：</p>

<ul>
<li><p><strong>Reward 规范</strong>：reward 应如何在 MCP 响应中编码？在工具结果中加入一个标准的 <code>reward</code> 字段将使即插即用的 RL 训练成为可能。</p></li>
<li><p><strong>Episode 管理</strong>：MCP 会话自然地对应到 episode，但重置语义需要标准化。</p></li>
<li><p><strong>观测空间</strong>：Resources 提供观测，但结构化的观测 schema（类似 Gym 的 <code>observation_space</code>）尚未标准化。</p></li>
<li><p><strong>基准套件</strong>：一组与 MCP 兼容的基准环境（编程、网页导航、数据分析）将加速研究。</p></li>
</ul>
</div>
