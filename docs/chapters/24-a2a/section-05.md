# 24.5　消息格式与 Schema {#sec:a2a:messages}

## 24.5.1　结构化与非结构化消息 {#structured-vs.-unstructured-messages}

<p>A2A 支持从完全非结构化（纯文本）到完全结构化（带类型的 JSON Schema）的整个谱系。选择何种格式取决于参与的 Agent：</p>

<p class="table-caption" id="table-24-2"><span class="table-number">表 24.2　</span>结构化与非结构化 A2A 消息的权衡。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>消息类型</strong></th><th scope="col"><strong>优点</strong></th><th scope="col"><strong>缺点</strong></th></tr></thead>
<tbody><tr><td>纯文本</td><td>灵活、可读性强、易于生成</td><td>难以可靠解析，无 Schema 校验</td></tr>
<tr><td>结构化 JSON</td><td>机器可解析、可校验、带类型</td><td>需要事先约定 Schema，灵活性较差</td></tr>
<tr><td>混合（文本 + 数据）</td><td>人类可读的意图 + 机器可解析的负载</td><td>构造和解析更复杂</td></tr></tbody>
</table>
</div>

## 24.5.2　多模态消息 {#multi-modal-messages}

<p>A2A 消息的结构是一个 <strong>role</strong>（<code>user</code> 或 <code>agent</code>）外加一组带类型的 <strong>parts</strong>：</p>

<p class="table-caption" id="table-24-3"><span class="table-number">表 24.3　</span>A2A 消息 part 类型（线上格式使用 <code>&quot;type&quot;: &quot;text&quot;|&quot;file&quot;|&quot;data&quot;</code>）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>Part 类型</strong></th><th scope="col"><strong>字段</strong></th><th scope="col"><strong>应用场景</strong></th></tr></thead>
<tbody><tr><td><code>TextPart</code></td><td><code>text: string</code></td><td>自然语言指令、响应</td></tr>
<tr><td><code>FilePart</code></td><td><code>mimeType</code>、<code>uri</code> 或 <code>bytes</code></td><td>文档、图像、音频、代码文件</td></tr>
<tr><td><code>DataPart</code></td><td><code>data: object</code></td><td>结构化 JSON（Tool 结果、Schema）</td></tr></tbody>
</table>
</div>

<p>现代 Agent 越来越多地处理非文本模态。A2A 的 <code>FilePart</code> 支持任意 MIME 类型，从而可以构建丰富的多模态工作流：</p>

<div class="admonition example"><p class="admonition-title">多模态 A2A 消息：数据分析</p>
<div class="highlight"><pre><span></span><code><span class="c1"># 一条同时包含文本指令、数据负载和文件的消息</span>
<span class="n">message</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>
    <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[</span>
        <span class="p">{</span>
            <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"text"</span><span class="p">,</span>
            <span class="s2">"text"</span><span class="p">:</span> <span class="s2">"Analyze the attached CSV and the schema below. "</span>
                    <span class="s2">"Identify anomalies and produce a summary report."</span>
        <span class="p">},</span>
        <span class="p">{</span>
            <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"file"</span><span class="p">,</span>
            <span class="s2">"mimeType"</span><span class="p">:</span> <span class="s2">"text/csv"</span><span class="p">,</span>
            <span class="s2">"uri"</span><span class="p">:</span> <span class="s2">"https://storage.example.com/data/sales_q3.csv"</span>
        <span class="p">},</span>
        <span class="p">{</span>
            <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"data"</span><span class="p">,</span>
            <span class="s2">"data"</span><span class="p">:</span> <span class="p">{</span>
                <span class="s2">"schema"</span><span class="p">:</span> <span class="p">{</span>
                    <span class="s2">"columns"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"date"</span><span class="p">,</span> <span class="s2">"region"</span><span class="p">,</span> <span class="s2">"product"</span><span class="p">,</span> <span class="s2">"revenue"</span><span class="p">,</span> <span class="s2">"units"</span><span class="p">],</span>
                    <span class="s2">"types"</span><span class="p">:</span>   <span class="p">[</span><span class="s2">"date"</span><span class="p">,</span> <span class="s2">"string"</span><span class="p">,</span> <span class="s2">"string"</span><span class="p">,</span> <span class="s2">"float"</span><span class="p">,</span> <span class="s2">"int"</span><span class="p">]</span>
                <span class="p">},</span>
                <span class="s2">"expectedRowCount"</span><span class="p">:</span> <span class="mi">15000</span><span class="p">,</span>
                <span class="s2">"anomalyThreshold"</span><span class="p">:</span> <span class="mf">3.0</span>  <span class="c1"># z 分数阈值</span>
            <span class="p">}</span>
        <span class="p">}</span>
    <span class="p">]</span>
<span class="p">}</span>
</code></pre></div>
</div>

<div class="admonition example"><p class="admonition-title">多模态 A2A 消息：图像分析</p>
<div class="highlight"><pre><span></span><code><span class="c1"># 多模态消息：文本 + 图像 + 结构化数据</span>
<span class="n">multimodal_message</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>
    <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[</span>
        <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"text"</span><span class="p">,</span>
         <span class="s2">"text"</span><span class="p">:</span> <span class="s2">"Describe what is wrong with this chart and suggest fixes."</span><span class="p">},</span>
        <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"file"</span><span class="p">,</span>
         <span class="s2">"mimeType"</span><span class="p">:</span> <span class="s2">"image/png"</span><span class="p">,</span>
         <span class="s2">"bytes"</span><span class="p">:</span> <span class="n">base64</span><span class="o">.</span><span class="n">b64encode</span><span class="p">(</span><span class="n">chart_image_bytes</span><span class="p">)</span><span class="o">.</span><span class="n">decode</span><span class="p">()},</span>
        <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"data"</span><span class="p">,</span>
         <span class="s2">"data"</span><span class="p">:</span> <span class="p">{</span>
             <span class="s2">"chartType"</span><span class="p">:</span> <span class="s2">"bar"</span><span class="p">,</span>
             <span class="s2">"dataSource"</span><span class="p">:</span> <span class="s2">"Q3 Revenue by Region"</span><span class="p">,</span>
             <span class="s2">"knownIssues"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"y-axis does not start at zero"</span><span class="p">,</span>
                             <span class="s2">"missing error bars"</span><span class="p">]</span>
         <span class="p">}}</span>
    <span class="p">]</span>
<span class="p">}</span>
</code></pre></div>
</div>

## 24.5.3　上下文传递：哪些该共享，哪些应保留 {#context-passing-what-to-share-vs.-what-to-keep-private}

<p>多 Agent 系统中一个关键的设计决策是<em>上下文范围控制</em>：要把多少对话历史和内部状态传递给子 Agent。</p>

<div class="admonition note"><p class="admonition-title">上下文范围控制原则</p>
<p>最小化上下文（Minimal Context）</p>

<p>只传递子 Agent 完成其任务所必需的内容。可降低 Token 用量、延迟以及敏感信息泄露的风险。</p>

<p>摘要化上下文（Summarized Context）</p>

<p>不要传递原始对话历史，而是传递结构化摘要：目标、约束、已做出的决策以及相关事实。</p>

<p>私有状态（Private State）</p>

<p>内部推理、中间草稿以及用户的个人身份信息（PII），除非确有需要，通常<em>不应</em>转发给子 Agent。</p>

<p>关联 ID（Correlation IDs）</p>

<p>始终传递 <code>correlationId</code>，以便子 Agent 的行为能够在日志和审计轨迹中追溯到发起的工作流。</p>
</div>

## 24.5.4　会话串联与关联 ID {#conversation-threading-and-correlation-ids}

<p>在复杂工作流中，许多任务可能同时处于进行中。<strong>关联 ID</strong> 用于跨 Agent 串联相关任务：</p>

<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">uuid</span>

<span class="k">class</span><span class="w"> </span><span class="nc">WorkflowContext</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""在多 Agent 工作流中携带关联元数据。"""</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">workflow_id</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span> <span class="o">=</span> <span class="n">workflow_id</span> <span class="ow">or</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">())</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">span_id</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">())</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">parent_span_id</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">child_context</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="s2">"WorkflowContext"</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""为子任务创建子上下文。"""</span>
        <span class="n">child</span> <span class="o">=</span> <span class="n">WorkflowContext</span><span class="p">(</span><span class="n">workflow_id</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span><span class="p">)</span>
        <span class="n">child</span><span class="o">.</span><span class="n">parent_span_id</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">span_id</span>
        <span class="k">return</span> <span class="n">child</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">to_metadata</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
        <span class="k">return</span> <span class="p">{</span>
            <span class="s2">"x-workflow-id"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span><span class="p">,</span>
            <span class="s2">"x-span-id"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">span_id</span><span class="p">,</span>
            <span class="s2">"x-parent-span-id"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">parent_span_id</span>
        <span class="p">}</span>

<span class="c1"># 使用方式：在每次 A2A 任务提交时附加该元数据</span>
<span class="n">ctx</span> <span class="o">=</span> <span class="n">WorkflowContext</span><span class="p">()</span>
<span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">send_task</span><span class="p">(</span>
    <span class="n">message</span><span class="o">=</span><span class="n">message</span><span class="p">,</span>
    <span class="n">metadata</span><span class="o">=</span><span class="n">ctx</span><span class="o">.</span><span class="n">to_metadata</span><span class="p">()</span>
<span class="p">)</span>
<span class="c1"># 子任务使用子上下文以便追踪</span>
<span class="n">sub_ctx</span> <span class="o">=</span> <span class="n">ctx</span><span class="o">.</span><span class="n">child_context</span><span class="p">()</span>
</code></pre></div>
