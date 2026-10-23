# 26.6　可观测性与调试 {#subsec:observability}

<p>生产级 Agent 系统需要完整的可观测性，以诊断故障、优化性能并保证合规。</p>

<div class="admonition note"><p class="admonition-title">Agent 可观测性的三大支柱</p>
<ol>
<li><p><strong>Trace（轨迹）</strong>：每一次 LLM 调用、Tool 调用与状态迁移的完整执行记录</p></li>
<li><p><strong>Metric（指标）</strong>：成本、延迟、成功率和 Tool 使用情况的聚合统计</p></li>
<li><p><strong>Log（日志）</strong>：用于调试与审计的结构化事件日志</p></li>
</ol>
</div>

## 26.6.1　追踪 Agent 执行 {#tracing-agent-execution}

<p>现代 Agent 可观测平台提供了针对 LLM 工作负载定制的分布式追踪：</p>

<ul>
<li><p><strong>LangSmith</strong>16：与 LangChain/LangGraph 深度集成；在每一步捕获完整的 prompt/response 对、Token 数量与延迟</p></li>
<li><p><strong>Arize Phoenix</strong>17：带 LLM 专用指标（幻觉检测、相关性打分）的开源可观测性</p></li>
<li><p><strong>Braintrust</strong>18：聚焦评测的平台，支持 A/B 测试与 Prompt 版本管理</p></li>
<li><p><strong>Weights &amp; Biases Weave</strong>：将实验追踪扩展到 Agent 轨迹</p></li>
<li><p><strong>OpenTelemetry</strong>19：标准化的可观测性插桩协议，对 LLM 的支持日益完善</p></li>
</ul>

<p class="code-caption"><span class="code-number">代码 26.21　</span>基于 OpenTelemetry 的结构化 Agent 追踪</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">opentelemetry</span><span class="w"> </span><span class="kn">import</span> <span class="n">trace</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">opentelemetry.sdk.trace</span><span class="w"> </span><span class="kn">import</span> <span class="n">TracerProvider</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">opentelemetry.sdk.trace.export</span><span class="w"> </span><span class="kn">import</span> <span class="n">BatchSpanProcessor</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">opentelemetry.exporter.otlp.proto.grpc.trace_exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">OTLPSpanExporter</span>

<span class="c1"># 配置追踪</span>
<span class="n">provider</span> <span class="o">=</span> <span class="n">TracerProvider</span><span class="p">()</span>
<span class="n">provider</span><span class="o">.</span><span class="n">add_span_processor</span><span class="p">(</span>
    <span class="n">BatchSpanProcessor</span><span class="p">(</span><span class="n">OTLPSpanExporter</span><span class="p">(</span><span class="n">endpoint</span><span class="o">=</span><span class="s2">"http://collector:4317"</span><span class="p">))</span>
<span class="p">)</span>
<span class="n">trace</span><span class="o">.</span><span class="n">set_tracer_provider</span><span class="p">(</span><span class="n">provider</span><span class="p">)</span>
<span class="n">tracer</span> <span class="o">=</span> <span class="n">trace</span><span class="o">.</span><span class="n">get_tracer</span><span class="p">(</span><span class="s2">"agent.tracer"</span><span class="p">)</span>

<span class="k">class</span><span class="w"> </span><span class="nc">InstrumentedAgent</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">run</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AgentResult</span><span class="p">:</span>
        <span class="k">with</span> <span class="n">tracer</span><span class="o">.</span><span class="n">start_as_current_span</span><span class="p">(</span><span class="s2">"agent.run"</span><span class="p">)</span> <span class="k">as</span> <span class="n">span</span><span class="p">:</span>
            <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"agent.task"</span><span class="p">,</span> <span class="n">task</span><span class="p">)</span>
            <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"agent.model"</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">model</span><span class="p">)</span>

            <span class="n">result</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_execute</span><span class="p">(</span><span class="n">task</span><span class="p">)</span>

            <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"agent.status"</span><span class="p">,</span> <span class="n">result</span><span class="o">.</span><span class="n">status</span><span class="p">)</span>
            <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"agent.tool_calls"</span><span class="p">,</span> <span class="n">result</span><span class="o">.</span><span class="n">tool_call_count</span><span class="p">)</span>
            <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"agent.tokens_used"</span><span class="p">,</span> <span class="n">result</span><span class="o">.</span><span class="n">tokens_used</span><span class="p">)</span>
            <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"agent.cost_usd"</span><span class="p">,</span> <span class="n">result</span><span class="o">.</span><span class="n">cost_usd</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">result</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_call_llm</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">messages</span><span class="p">:</span> <span class="nb">list</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="k">with</span> <span class="n">tracer</span><span class="o">.</span><span class="n">start_as_current_span</span><span class="p">(</span><span class="s2">"llm.call"</span><span class="p">)</span> <span class="k">as</span> <span class="n">span</span><span class="p">:</span>
            <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"llm.model"</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">model</span><span class="p">)</span>
            <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"llm.prompt_tokens"</span><span class="p">,</span> <span class="n">count_tokens</span><span class="p">(</span><span class="n">messages</span><span class="p">))</span>
            <span class="n">response</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">llm</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span><span class="n">messages</span><span class="p">)</span>
            <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"llm.completion_tokens"</span><span class="p">,</span> <span class="n">count_tokens</span><span class="p">([</span><span class="n">response</span><span class="p">]))</span>
            <span class="k">return</span> <span class="n">response</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_call_tool</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">tool_name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">args</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="k">with</span> <span class="n">tracer</span><span class="o">.</span><span class="n">start_as_current_span</span><span class="p">(</span><span class="sa">f</span><span class="s2">"tool.</span><span class="si">{</span><span class="n">tool_name</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span> <span class="k">as</span> <span class="n">span</span><span class="p">:</span>
            <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"tool.name"</span><span class="p">,</span> <span class="n">tool_name</span><span class="p">)</span>
            <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"tool.args"</span><span class="p">,</span> <span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">(</span><span class="n">args</span><span class="p">))</span>
            <span class="k">try</span><span class="p">:</span>
                <span class="n">result</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">tools</span><span class="p">[</span><span class="n">tool_name</span><span class="p">](</span><span class="o">**</span><span class="n">args</span><span class="p">)</span>
                <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"tool.success"</span><span class="p">,</span> <span class="kc">True</span><span class="p">)</span>
                <span class="k">return</span> <span class="n">result</span>
            <span class="k">except</span> <span class="ne">Exception</span> <span class="k">as</span> <span class="n">e</span><span class="p">:</span>
                <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"tool.success"</span><span class="p">,</span> <span class="kc">False</span><span class="p">)</span>
                <span class="n">span</span><span class="o">.</span><span class="n">set_attribute</span><span class="p">(</span><span class="s2">"tool.error"</span><span class="p">,</span> <span class="nb">str</span><span class="p">(</span><span class="n">e</span><span class="p">))</span>
                <span class="n">span</span><span class="o">.</span><span class="n">record_exception</span><span class="p">(</span><span class="n">e</span><span class="p">)</span>
                <span class="k">raise</span>
</code></pre></div>

## 26.6.2　失败分类 {#failure-categorization}

<p>系统化的失败分析需要一套失败模式分类法。如果缺少结构化分类，工程团队就会被即兴调试耗尽——只医表象、不治根因。下面这套分类涵盖了生产 Agent 系统中最常见的六类失败，以及它们的可观察症状、自动化检测机制和经过验证的修复策略。</p>

<p>每种失败类型对系统设计的含义各不相同：<em>Tool 错误</em>属于基础设施故障，需要重试逻辑与断路器；<em>推理错误</em>属于模型层故障，需要迭代 Prompt；<em>幻觉</em>需要引入接地（grounding）机制；<em>死循环</em>则需要硬性架构保护。在实际场景中，用户可见的单次失败往往跨越多个类别（例如：Tool 错误触发 Agent 尝试自救时的推理错误，进一步演变为死循环）。</p>

<p class="table-caption" id="table-26-1"><span class="table-number">表 26.1　</span>Agent 失败分类，附检测与修复策略</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>失败类型</strong></th><th scope="col"><strong>症状</strong></th><th scope="col"><strong>检测</strong></th><th scope="col"><strong>修复策略</strong></th></tr></thead>
<tbody><tr><td>Tool 错误</td><td>Tool 调用抛异常、返回空结果</td><td>错误率监控</td><td>重试逻辑、回退 Tool</td></tr>
<tr><td>推理错误</td><td>选错 Tool、参数错误</td><td>轨迹分析</td><td>改进 Prompt、加入 Few-shot 示例</td></tr>
<tr><td>幻觉</td><td>捏造事实、编造 Tool 结果</td><td>事实核查、接地校验</td><td>检索增强生成（RAG）、强制要求引用</td></tr>
<tr><td>死循环</td><td>反复调用同一 Tool 而无进展</td><td>循环检测、最大迭代次数</td><td>硬性上限、破环 Prompt</td></tr>
<tr><td>上下文溢出</td><td>历史被截断、丢失上下文</td><td>Token 计数</td><td>摘要、上下文管理</td></tr>
<tr><td>拒绝</td><td>对合法任务也予以拒绝</td><td>输出分类</td><td>调整 Prompt、微调 guardrail</td></tr></tbody>
</table>
</div>

## 26.6.3　重放与调试 Workflow {#replay-and-debugging-workflows}

<p>当生产环境出现故障时，能精确重放当时的执行过程是无价的：</p>

<p class="code-caption"><span class="code-number">代码 26.22　</span>用于调试的 Agent 执行重放</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">langsmith</span><span class="w"> </span><span class="kn">import</span> <span class="n">Client</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">datetime</span><span class="w"> </span><span class="kn">import</span> <span class="n">datetime</span><span class="p">,</span> <span class="n">timezone</span>

<span class="n">ls</span> <span class="o">=</span> <span class="n">Client</span><span class="p">()</span>  <span class="c1"># 使用 LANGSMITH_API_KEY 环境变量</span>

<span class="c1"># 通过 run ID 加载失败的执行轨迹</span>
<span class="n">root_run</span> <span class="o">=</span> <span class="n">ls</span><span class="o">.</span><span class="n">read_run</span><span class="p">(</span><span class="s2">"run-abc123-def456"</span><span class="p">)</span>
<span class="n">child_runs</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="n">ls</span><span class="o">.</span><span class="n">list_runs</span><span class="p">(</span>
    <span class="n">project_name</span><span class="o">=</span><span class="s2">"research-agent"</span><span class="p">,</span>
    <span class="nb">filter</span><span class="o">=</span><span class="sa">f</span><span class="s1">'eq(parent_run_id, "</span><span class="si">{</span><span class="n">root_run</span><span class="o">.</span><span class="n">id</span><span class="si">}</span><span class="s1">")'</span><span class="p">,</span>
    <span class="n">order</span><span class="o">=</span><span class="s2">"asc"</span><span class="p">,</span>
<span class="p">))</span>

<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Trace: </span><span class="si">{</span><span class="n">root_run</span><span class="o">.</span><span class="n">id</span><span class="si">}</span><span class="s2"> | Status: </span><span class="si">{</span><span class="n">root_run</span><span class="o">.</span><span class="n">status</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Error: </span><span class="si">{</span><span class="n">root_run</span><span class="o">.</span><span class="n">error</span><span class="si">}</span><span class="s2">"</span> <span class="k">if</span> <span class="n">root_run</span><span class="o">.</span><span class="n">error</span> <span class="k">else</span> <span class="s2">""</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Total tokens: </span><span class="si">{</span><span class="n">root_run</span><span class="o">.</span><span class="n">total_tokens</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span><span class="p">)</span>

<span class="c1"># 逐步遍历每个子 run（LLM 调用、Tool 调用等）</span>
<span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">run</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">child_runs</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Step </span><span class="si">{</span><span class="n">i</span><span class="si">}</span><span class="s2">: [</span><span class="si">{</span><span class="n">run</span><span class="o">.</span><span class="n">run_type</span><span class="si">}</span><span class="s2">] </span><span class="si">{</span><span class="n">run</span><span class="o">.</span><span class="n">name</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"  Input:  </span><span class="si">{</span><span class="nb">str</span><span class="p">(</span><span class="n">run</span><span class="o">.</span><span class="n">inputs</span><span class="p">)[:</span><span class="mi">200</span><span class="p">]</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"  Output: </span><span class="si">{</span><span class="nb">str</span><span class="p">(</span><span class="n">run</span><span class="o">.</span><span class="n">outputs</span><span class="p">)[:</span><span class="mi">200</span><span class="p">]</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">run</span><span class="o">.</span><span class="n">error</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"  ERROR: </span><span class="si">{</span><span class="n">run</span><span class="o">.</span><span class="n">error</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
        <span class="c1"># 检查导致失败的具体 Prompt</span>
        <span class="k">if</span> <span class="n">run</span><span class="o">.</span><span class="n">run_type</span> <span class="o">==</span> <span class="s2">"llm"</span><span class="p">:</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"  Model: </span><span class="si">{</span><span class="n">run</span><span class="o">.</span><span class="n">extra</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s1">'invocation_params'</span><span class="p">,</span><span class="w"> </span><span class="p">{})</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s1">'model'</span><span class="p">)</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"  Messages: </span><span class="si">{</span><span class="n">run</span><span class="o">.</span><span class="n">inputs</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s1">'messages'</span><span class="p">,</span><span class="w"> </span><span class="p">[])[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>

<span class="c1"># 用修改后的 Prompt 或模型重跑失败步骤</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">OpenAI</span>
<span class="n">client</span> <span class="o">=</span> <span class="n">OpenAI</span><span class="p">()</span>
<span class="n">failing_run</span> <span class="o">=</span> <span class="n">child_runs</span><span class="p">[</span><span class="mi">4</span><span class="p">]</span>  <span class="c1"># 例如出错的那一步</span>
<span class="n">response</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">chat</span><span class="o">.</span><span class="n">completions</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">,</span>  <span class="c1"># 换更强的模型试试</span>
    <span class="n">messages</span><span class="o">=</span><span class="n">failing_run</span><span class="o">.</span><span class="n">inputs</span><span class="p">[</span><span class="s2">"messages"</span><span class="p">],</span>
    <span class="n">temperature</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span>
<span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Replay output: </span><span class="si">{</span><span class="n">response</span><span class="o">.</span><span class="n">choices</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">message</span><span class="o">.</span><span class="n">content</span><span class="p">[:</span><span class="mi">300</span><span class="p">]</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
</code></pre></div>
