# 25.4　通信协议 {#subsec:communication-protocols}

<p>有效的多智能体系统需要定义良好的通信协议：约定的 Agent 间消息格式、语义和模式。（关于标准化的 Agent 间协议，参见第 <a href="../24-a2a/index.md#sec:a2a">24</a> 章。）</p>

## 25.4.1　结构化消息格式 {#structured-message-formats}

<p>LLM Agent 之间的消息应该是结构化的，以支持可靠解析与路由。一个最小的消息 schema：</p>

<p class="code-caption"><span class="code-number">代码 25.3　</span>Agent 消息 schema</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">pydantic</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseModel</span><span class="p">,</span> <span class="n">Field</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Literal</span><span class="p">,</span> <span class="n">Optional</span><span class="p">,</span> <span class="n">Dict</span><span class="p">,</span> <span class="n">Any</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">datetime</span><span class="w"> </span><span class="kn">import</span> <span class="n">datetime</span><span class="p">,</span> <span class="n">timezone</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">uuid</span>

<span class="n">PerformativeType</span> <span class="o">=</span> <span class="n">Literal</span><span class="p">[</span>
    <span class="s2">"inform"</span><span class="p">,</span>    <span class="c1"># 共享信息</span>
    <span class="s2">"request"</span><span class="p">,</span>   <span class="c1"># 请求执行某动作</span>
    <span class="s2">"propose"</span><span class="p">,</span>   <span class="c1"># 提议某种行动方案</span>
    <span class="s2">"accept"</span><span class="p">,</span>    <span class="c1"># 接受提议</span>
    <span class="s2">"reject"</span><span class="p">,</span>    <span class="c1"># 拒绝提议</span>
    <span class="s2">"query"</span><span class="p">,</span>     <span class="c1"># 提出问题</span>
    <span class="s2">"confirm"</span><span class="p">,</span>   <span class="c1"># 确认收到/完成</span>
    <span class="s2">"failure"</span><span class="p">,</span>   <span class="c1"># 报告失败</span>
<span class="p">]</span>

<span class="k">class</span><span class="w"> </span><span class="nc">AgentMessage</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">message_id</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="n">Field</span><span class="p">(</span><span class="n">default_factory</span><span class="o">=</span><span class="k">lambda</span><span class="p">:</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">()))</span>
    <span class="n">conversation_id</span><span class="p">:</span> <span class="nb">str</span>          <span class="c1"># 将相关消息分组</span>
    <span class="n">sender</span><span class="p">:</span> <span class="nb">str</span>                   <span class="c1"># Agent 标识</span>
    <span class="n">receiver</span><span class="p">:</span> <span class="nb">str</span>                 <span class="c1"># 目标 agent（或 "broadcast"）</span>
    <span class="n">performative</span><span class="p">:</span> <span class="n">PerformativeType</span>
    <span class="n">content</span><span class="p">:</span> <span class="nb">str</span>                  <span class="c1"># 自然语言内容</span>
    <span class="n">metadata</span><span class="p">:</span> <span class="n">Dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">Any</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span> <span class="c1"># 结构化载荷</span>
    <span class="n">reply_to</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>  <span class="c1"># 所回复的 message_id</span>
    <span class="n">timestamp</span><span class="p">:</span> <span class="n">datetime</span> <span class="o">=</span> <span class="n">Field</span><span class="p">(</span><span class="n">default_factory</span><span class="o">=</span><span class="k">lambda</span><span class="p">:</span> <span class="n">datetime</span><span class="o">.</span><span class="n">now</span><span class="p">(</span><span class="n">timezone</span><span class="o">.</span><span class="n">utc</span><span class="p">))</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">to_llm_prompt</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="c1"># 将消息渲染为接收方 agent 的 prompt 片段。</span>
<span class="w">        </span><span class="sd">"""Render message as a prompt fragment for the receiving agent."""</span>
        <span class="k">return</span> <span class="p">(</span>
            <span class="sa">f</span><span class="s2">"[MESSAGE from </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">sender</span><span class="si">}</span><span class="s2">]</span><span class="se">\n</span><span class="s2">"</span>
            <span class="sa">f</span><span class="s2">"Type: </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">performative</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
            <span class="sa">f</span><span class="s2">"Content: </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">content</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
            <span class="o">+</span> <span class="p">(</span><span class="sa">f</span><span class="s2">"Metadata: </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">metadata</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span> <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">metadata</span> <span class="k">else</span> <span class="s2">""</span><span class="p">)</span>
        <span class="p">)</span>
</code></pre></div>

## 25.4.2　Performative 类型（受 FIPA-ACL 启发） {#performative-types-fipa-acl-inspired}

<p>借鉴 FIPA Agent 通信语言 <a class="book-citation" id="cite-fipa2002acl-1" href="../../references.md#ref-fipa2002acl" aria-label="参考文献 93">[93]</a>，并针对 LLM Agent 进行现代化改造：</p>

<p class="table-caption" id="table-25-1"><span class="table-number">表 25.1　</span>受 FIPA-ACL 启发的 LLM Agent 消息 performative 类型。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>Performative</strong></th><th scope="col"><strong>语义</strong></th><th scope="col"><strong>示例用途</strong></th></tr></thead>
<tbody><tr><td><code>inform</code></td><td>发送方相信 <span class="arithmatex">\(\phi\)</span> 为真</td><td>共享研究发现</td></tr>
<tr><td><code>request</code></td><td>发送方希望接收方执行 <span class="arithmatex">\(\alpha\)</span></td><td>委派子任务</td></tr>
<tr><td><code>propose</code></td><td>发送方提议计划 <span class="arithmatex">\(\pi\)</span></td><td>建议某种方案</td></tr>
<tr><td><code>accept</code></td><td>接收方同意提议</td><td>确认任务分配</td></tr>
<tr><td><code>reject</code></td><td>接收方拒绝提议</td><td>拒绝不兼容的任务</td></tr>
<tr><td><code>query</code></td><td>发送方希望知道 <span class="arithmatex">\(\phi\)</span></td><td>请求澄清</td></tr>
<tr><td><code>confirm</code></td><td>发送方确认 <span class="arithmatex">\(\phi\)</span> 已发生</td><td>确认完成</td></tr>
<tr><td><code>failure</code></td><td>发送方未能完成 <span class="arithmatex">\(\alpha\)</span></td><td>报告错误</td></tr></tbody>
</table>
</div>

## 25.4.3　上下文共享策略 {#context-sharing-strategies}

<p>多 Agent 通信中的一个关键挑战是<strong>上下文管理</strong>：每个 Agent 需要多少历史？三种策略：</p>

<ul>
<li><p><strong>完整历史</strong>：将整个会话历史传递给每个 Agent。信息最丰富但代价高昂；上下文窗口会迅速被填满。</p></li>
<li><p><strong>摘要</strong>：由一个摘要 Agent 将先前的交流压缩为紧凑的摘要。高效但有损；重要细节可能被丢弃。</p></li>
<li><p><strong>相关摘录</strong>：通过语义搜索仅检索最相关的历史消息。在成本与信息量之间取得平衡；需要检索机制。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">上下文共享经验法则</p>
<p>对短对话（<span class="arithmatex">\(&lt;\)</span>10 轮）使用<strong>完整历史</strong>；对中等长度对话使用<strong>摘要</strong>；对长时运行的 Agent 会话使用<strong>检索增强的摘录</strong>。始终原文保留最近的 <span class="arithmatex">\(k\)</span> 条消息，以保持即时上下文。</p>
</div>
