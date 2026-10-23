# 26.3　主流框架深度剖析 {#subsec:frameworks}

<p>Agent 框架生态快速增长，每个框架都体现了不同的设计哲学与目标场景。我们深入剖析当下采用最广泛的几个框架。</p>

## 26.3.1　LangGraph {#subsubsec:langgraph}

<p>LangGraph <a class="book-citation" id="cite-langchain2024langgraph-1" href="../../references.md#ref-langchain2024langgraph" aria-label="参考文献 154">[154]</a> 由 LangChain Inc. 开发，将 Agent 执行建模为一个<em>有向图</em>：节点表示计算步骤，边表示步骤之间的迁移。这种基于图的抽象提供了对 Agent 流程的显式控制，使复杂多步骤行为更易于推理、测试与调试。</p>

#### 核心概念。 {#core-concepts.}

<ul>
<li><p><strong>State（状态）</strong>：一个带类型的字典（使用 Python 的 <code>TypedDict</code> 或 Pydantic），它在图中流动，并由每个节点更新</p></li>
<li><p><strong>Nodes（节点）</strong>：接收当前状态并返回状态更新的 Python 函数</p></li>
<li><p><strong>Edges（边）</strong>：节点之间的迁移，可以是无条件的，也可以是条件的（按状态路由）</p></li>
<li><p><strong>Checkpointing（检查点）</strong>：内置的图状态持久化，支持暂停／恢复以及人机协同（Human-in-the-Loop）Workflow</p></li>
<li><p><strong>Subgraphs（子图）</strong>：可组合、可嵌套到更大图中的图组件</p></li>
</ul>

#### 状态管理。 {#state-management.}

<p>LangGraph 的状态管理是其最强大的特性之一。状态 schema 充当节点之间的契约，使数据流显式且类型安全：</p>

<p class="code-caption"><span class="code-number">代码 26.1　</span>LangGraph 状态 schema 定义</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">TypedDict</span><span class="p">,</span> <span class="n">Annotated</span><span class="p">,</span> <span class="n">List</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.graph.message</span><span class="w"> </span><span class="kn">import</span> <span class="n">add_messages</span>

<span class="k">class</span><span class="w"> </span><span class="nc">AgentState</span><span class="p">(</span><span class="n">TypedDict</span><span class="p">):</span>
    <span class="c1"># 消息通过 add_messages reducer 累积</span>
    <span class="n">messages</span><span class="p">:</span> <span class="n">Annotated</span><span class="p">[</span><span class="n">List</span><span class="p">[</span><span class="n">BaseMessage</span><span class="p">],</span> <span class="n">add_messages</span><span class="p">]</span>
    <span class="c1"># 简单字段在每次更新时被覆盖</span>
    <span class="n">current_tool</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span>
    <span class="n">iteration_count</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">final_answer</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span>
    <span class="n">error</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span>
</code></pre></div>

#### 检查点与人机协同。 {#checkpointing-and-human-in-the-loop.}

<p>LangGraph 的 checkpointer 在每次节点执行后保存图状态。它支持：</p>

<ul>
<li><p><strong>续跑</strong>：长时间运行的 Agent 可以暂停并恢复而不丢失进度</p></li>
<li><p><strong>人工审批</strong>：图可以在指定节点暂停，等待人工输入后再继续</p></li>
<li><p><strong>时光回溯</strong>：运维人员可以从任意检查点重放执行以进行调试</p></li>
</ul>

<p class="code-caption"><span class="code-number">代码 26.2　</span>LangGraph 检查点与人机协同</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.checkpoint.sqlite</span><span class="w"> </span><span class="kn">import</span> <span class="n">SqliteSaver</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.graph</span><span class="w"> </span><span class="kn">import</span> <span class="n">StateGraph</span><span class="p">,</span> <span class="n">START</span><span class="p">,</span> <span class="n">END</span>

<span class="c1"># 持久化 checkpointer</span>
<span class="n">memory</span> <span class="o">=</span> <span class="n">SqliteSaver</span><span class="o">.</span><span class="n">from_conn_string</span><span class="p">(</span><span class="s2">"agent_state.db"</span><span class="p">)</span>

<span class="c1"># 构建带中断点的图</span>
<span class="n">builder</span> <span class="o">=</span> <span class="n">StateGraph</span><span class="p">(</span><span class="n">AgentState</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"plan"</span><span class="p">,</span> <span class="n">plan_node</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"human_review"</span><span class="p">,</span> <span class="n">human_review_node</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"execute"</span><span class="p">,</span> <span class="n">execute_node</span><span class="p">)</span>

<span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="n">START</span><span class="p">,</span> <span class="s2">"plan"</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"plan"</span><span class="p">,</span> <span class="s2">"human_review"</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"human_review"</span><span class="p">,</span> <span class="s2">"execute"</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"execute"</span><span class="p">,</span> <span class="n">END</span><span class="p">)</span>

<span class="c1"># 编译时绑定 checkpointer，并在 human_review 之前中断</span>
<span class="n">graph</span> <span class="o">=</span> <span class="n">builder</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span>
    <span class="n">checkpointer</span><span class="o">=</span><span class="n">memory</span><span class="p">,</span>
    <span class="n">interrupt_before</span><span class="o">=</span><span class="p">[</span><span class="s2">"human_review"</span><span class="p">]</span>
<span class="p">)</span>

<span class="c1"># 运行直到中断</span>
<span class="n">config</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"configurable"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"thread_id"</span><span class="p">:</span> <span class="s2">"task-001"</span><span class="p">}}</span>
<span class="n">result</span> <span class="o">=</span> <span class="n">graph</span><span class="o">.</span><span class="n">invoke</span><span class="p">({</span><span class="s2">"messages"</span><span class="p">:</span> <span class="p">[</span><span class="n">HumanMessage</span><span class="p">(</span><span class="s2">"Analyze Q3 sales"</span><span class="p">)]},</span> <span class="n">config</span><span class="p">)</span>

<span class="c1"># 在人工输入后恢复</span>
<span class="n">graph</span><span class="o">.</span><span class="n">update_state</span><span class="p">(</span><span class="n">config</span><span class="p">,</span> <span class="p">{</span><span class="s2">"human_feedback"</span><span class="p">:</span> <span class="s2">"Approved, proceed"</span><span class="p">})</span>
<span class="n">result</span> <span class="o">=</span> <span class="n">graph</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span><span class="kc">None</span><span class="p">,</span> <span class="n">config</span><span class="p">)</span>  <span class="c1"># 从检查点恢复</span>
</code></pre></div>

<p>下面两段代码把上述全部要素——状态 schema、Tool 节点、条件路由、检查点与调用——组合成一个完整的研究型 Agent：迭代收集信息并综合输出报告。</p>

<p class="code-caption"><span class="code-number">代码 26.3　</span>研究型 Agent —— 状态、Tool 与节点函数</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">TypedDict</span><span class="p">,</span> <span class="n">Annotated</span><span class="p">,</span> <span class="n">List</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">ChatOpenAI</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_core.tools</span><span class="w"> </span><span class="kn">import</span> <span class="n">tool</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_core.messages</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseMessage</span><span class="p">,</span> <span class="n">HumanMessage</span><span class="p">,</span> <span class="n">AIMessage</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.graph</span><span class="w"> </span><span class="kn">import</span> <span class="n">StateGraph</span><span class="p">,</span> <span class="n">START</span><span class="p">,</span> <span class="n">END</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.prebuilt</span><span class="w"> </span><span class="kn">import</span> <span class="n">ToolNode</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.graph.message</span><span class="w"> </span><span class="kn">import</span> <span class="n">add_messages</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.checkpoint.sqlite</span><span class="w"> </span><span class="kn">import</span> <span class="n">SqliteSaver</span>

<span class="c1"># --- Tool 定义 ---</span>
<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">search_web</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""在网络上搜索某个主题的最新信息。"""</span>
    <span class="k">return</span> <span class="sa">f</span><span class="s2">"Search results for: </span><span class="si">{</span><span class="n">query</span><span class="si">}</span><span class="s2">"</span>  <span class="c1"># 占位实现；实际调用真实 API</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">read_document</span><span class="p">(</span><span class="n">url</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""获取并读取指定 URL 上的文档内容。"""</span>
    <span class="k">return</span> <span class="sa">f</span><span class="s2">"Document content from: </span><span class="si">{</span><span class="n">url</span><span class="si">}</span><span class="s2">"</span>

<span class="n">tools</span> <span class="o">=</span> <span class="p">[</span><span class="n">search_web</span><span class="p">,</span> <span class="n">read_document</span><span class="p">]</span>

<span class="c1"># --- 状态 Schema ---</span>
<span class="k">class</span><span class="w"> </span><span class="nc">ResearchState</span><span class="p">(</span><span class="n">TypedDict</span><span class="p">):</span>
    <span class="n">messages</span><span class="p">:</span> <span class="n">Annotated</span><span class="p">[</span><span class="n">List</span><span class="p">[</span><span class="n">BaseMessage</span><span class="p">],</span> <span class="n">add_messages</span><span class="p">]</span>
    <span class="n">research_topic</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">iteration</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">status</span><span class="p">:</span> <span class="nb">str</span>  <span class="c1"># "researching" | "drafting" | "done" | "error"</span>

<span class="c1"># --- 节点函数 ---</span>
<span class="k">def</span><span class="w"> </span><span class="nf">research_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">ResearchState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""LLM 决定下一步搜索什么或宣告完成。"""</span>
    <span class="n">llm</span> <span class="o">=</span> <span class="n">ChatOpenAI</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">)</span><span class="o">.</span><span class="n">bind_tools</span><span class="p">(</span><span class="n">tools</span><span class="p">)</span>
    <span class="n">response</span> <span class="o">=</span> <span class="n">llm</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span><span class="n">state</span><span class="p">[</span><span class="s2">"messages"</span><span class="p">])</span>
    <span class="k">return</span> <span class="p">{</span><span class="s2">"messages"</span><span class="p">:</span> <span class="p">[</span><span class="n">response</span><span class="p">],</span> <span class="s2">"iteration"</span><span class="p">:</span> <span class="n">state</span><span class="p">[</span><span class="s2">"iteration"</span><span class="p">]</span> <span class="o">+</span> <span class="mi">1</span><span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">should_continue</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">ResearchState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""路由：有 Tool 调用 -&gt; 执行 Tool；无调用 -&gt; 进入综合。"""</span>
    <span class="n">last</span> <span class="o">=</span> <span class="n">state</span><span class="p">[</span><span class="s2">"messages"</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
    <span class="k">if</span> <span class="nb">hasattr</span><span class="p">(</span><span class="n">last</span><span class="p">,</span> <span class="s2">"tool_calls"</span><span class="p">)</span> <span class="ow">and</span> <span class="n">last</span><span class="o">.</span><span class="n">tool_calls</span><span class="p">:</span>
        <span class="k">return</span> <span class="s2">"tools"</span>
    <span class="k">if</span> <span class="n">state</span><span class="p">[</span><span class="s2">"iteration"</span><span class="p">]</span> <span class="o">&gt;=</span> <span class="mi">10</span><span class="p">:</span>
        <span class="k">return</span> <span class="s2">"error"</span>
    <span class="k">return</span> <span class="s2">"synthesize"</span>

<span class="k">def</span><span class="w"> </span><span class="nf">synthesize_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">ResearchState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""根据已累积的研究材料产出最终报告。"""</span>
    <span class="n">llm</span> <span class="o">=</span> <span class="n">ChatOpenAI</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">)</span>
    <span class="n">prompt</span> <span class="o">=</span> <span class="p">(</span>
        <span class="sa">f</span><span class="s2">"Synthesize a comprehensive report on: </span><span class="si">{</span><span class="n">state</span><span class="p">[</span><span class="s1">'research_topic'</span><span class="p">]</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
        <span class="s2">"Use all search results and documents gathered above."</span>
    <span class="p">)</span>
    <span class="n">response</span> <span class="o">=</span> <span class="n">llm</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span>
        <span class="n">state</span><span class="p">[</span><span class="s2">"messages"</span><span class="p">]</span> <span class="o">+</span> <span class="p">[</span><span class="n">HumanMessage</span><span class="p">(</span><span class="n">content</span><span class="o">=</span><span class="n">prompt</span><span class="p">)]</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="p">{</span><span class="s2">"messages"</span><span class="p">:</span> <span class="p">[</span><span class="n">response</span><span class="p">],</span> <span class="s2">"status"</span><span class="p">:</span> <span class="s2">"done"</span><span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">error_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">ResearchState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
    <span class="k">return</span> <span class="p">{</span><span class="s2">"status"</span><span class="p">:</span> <span class="s2">"error"</span><span class="p">,</span> <span class="s2">"messages"</span><span class="p">:</span> <span class="p">[</span>
        <span class="n">AIMessage</span><span class="p">(</span><span class="n">content</span><span class="o">=</span><span class="s2">"Research exceeded maximum iterations."</span><span class="p">)</span>
    <span class="p">]}</span>
</code></pre></div>

<p class="code-caption"><span class="code-number">代码 26.4　</span>研究型 Agent —— 图构建与调用</p>
<div class="highlight"><pre><span></span><code><span class="c1"># --- 图构建 ---</span>
<span class="n">tool_node</span> <span class="o">=</span> <span class="n">ToolNode</span><span class="p">(</span><span class="n">tools</span><span class="p">)</span>
<span class="n">builder</span> <span class="o">=</span> <span class="n">StateGraph</span><span class="p">(</span><span class="n">ResearchState</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"research"</span><span class="p">,</span> <span class="n">research_node</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"tools"</span><span class="p">,</span> <span class="n">tool_node</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"synthesize"</span><span class="p">,</span> <span class="n">synthesize_node</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"error"</span><span class="p">,</span> <span class="n">error_node</span><span class="p">)</span>

<span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="n">START</span><span class="p">,</span> <span class="s2">"research"</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_conditional_edges</span><span class="p">(</span>
    <span class="s2">"research"</span><span class="p">,</span> <span class="n">should_continue</span><span class="p">,</span>
    <span class="p">{</span><span class="s2">"tools"</span><span class="p">:</span> <span class="s2">"tools"</span><span class="p">,</span> <span class="s2">"synthesize"</span><span class="p">:</span> <span class="s2">"synthesize"</span><span class="p">,</span> <span class="s2">"error"</span><span class="p">:</span> <span class="s2">"error"</span><span class="p">}</span>
<span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"tools"</span><span class="p">,</span> <span class="s2">"research"</span><span class="p">)</span>   <span class="c1"># Tool 执行后回到研究节点</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"synthesize"</span><span class="p">,</span> <span class="n">END</span><span class="p">)</span>
<span class="n">builder</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"error"</span><span class="p">,</span> <span class="n">END</span><span class="p">)</span>

<span class="c1"># 编译时启用持久化以保留对话记忆</span>
<span class="k">with</span> <span class="n">SqliteSaver</span><span class="o">.</span><span class="n">from_conn_string</span><span class="p">(</span><span class="s2">":memory:"</span><span class="p">)</span> <span class="k">as</span> <span class="n">checkpointer</span><span class="p">:</span>
    <span class="n">graph</span> <span class="o">=</span> <span class="n">builder</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">checkpointer</span><span class="o">=</span><span class="n">checkpointer</span><span class="p">)</span>

<span class="c1"># --- 调用 ---</span>
<span class="n">result</span> <span class="o">=</span> <span class="n">graph</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span>
    <span class="p">{</span><span class="s2">"messages"</span><span class="p">:</span> <span class="p">[</span><span class="n">HumanMessage</span><span class="p">(</span><span class="n">content</span><span class="o">=</span><span class="s2">"Research recent advances in RLHF"</span><span class="p">)],</span>
     <span class="s2">"research_topic"</span><span class="p">:</span> <span class="s2">"Recent advances in RLHF"</span><span class="p">,</span>
     <span class="s2">"iteration"</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span> <span class="s2">"status"</span><span class="p">:</span> <span class="s2">"researching"</span><span class="p">},</span>
    <span class="n">config</span><span class="o">=</span><span class="p">{</span><span class="s2">"configurable"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"thread_id"</span><span class="p">:</span> <span class="s2">"research-1"</span><span class="p">}}</span>
<span class="p">)</span>
</code></pre></div>

<figure id="fig:langgraph-graph"><a href="../../assets/figures/fig_073_langgraph-graph.png" target="_blank" rel="noopener" aria-label="查看图 26.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_073_langgraph-graph.png" alt="研究型 Agent 的 LangGraph 执行图。条件边实现了 Tool 使用循环与错误处理。" loading="lazy"></a><figcaption><span class="figure-number">图 26.2　</span>研究型 Agent 的 LangGraph 执行图。条件边实现了 Tool 使用循环与错误处理。</figcaption></figure>

## 26.3.2　AutoGen（Microsoft） {#subsubsec:autogen}

<p>AutoGen <a class="book-citation" id="cite-wu2023autogen-1" href="../../references.md#ref-wu2023autogen" aria-label="参考文献 385">[385]</a> 由 Microsoft Research 开发，采用了截然不同的思路：它将 Agent 建模为通过结构化消息传递进行通信的<em>可对话实体（conversable entity）</em>。AutoGen 不局限于单个 Agent 循环，而是支持多个 Agent 在共享会话中协作以求解复杂任务。</p>

#### Conversable Agents。 {#conversable-agents.}

<p>每个 AutoGen Agent 都是一个 <code>ConversableAgent</code>，具备：</p>

<ul>
<li><p>一条 <strong>系统消息</strong>，定义其角色与能力</p></li>
<li><p>一种 <strong>人类输入模式</strong>，控制何时征求人工输入（<code>ALWAYS</code>、<code>NEVER</code>、<code>TERMINATE</code>）</p></li>
<li><p>一份 <strong>代码执行配置</strong>，指定是否以及如何运行代码</p></li>
<li><p>一份可调用 Tool 的 <strong>函数映射表</strong></p></li>
</ul>

#### 群聊（Group Chat）模式。 {#group-chat-patterns.}

<p>AutoGen 的 <code>GroupChat</code> 让多个 Agent 在同一会话中协作。<code>GroupChatManager</code> 负责协调发言轮转，可采用轮询（round-robin）、基于 LLM 的发言者选择或自定义路由逻辑。</p>

<p class="code-caption"><span class="code-number">代码 26.5　</span>AutoGen 多 Agent 群聊</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">autogen</span>

<span class="n">config_list</span> <span class="o">=</span> <span class="p">[{</span><span class="s2">"model"</span><span class="p">:</span> <span class="s2">"gpt-4o"</span><span class="p">,</span> <span class="s2">"api_key"</span><span class="p">:</span> <span class="n">os</span><span class="o">.</span><span class="n">environ</span><span class="p">[</span><span class="s2">"OPENAI_API_KEY"</span><span class="p">]}]</span>
<span class="n">llm_config</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"config_list"</span><span class="p">:</span> <span class="n">config_list</span><span class="p">,</span> <span class="s2">"temperature"</span><span class="p">:</span> <span class="mi">0</span><span class="p">}</span>

<span class="c1"># 专门化的 Agents</span>
<span class="n">planner</span> <span class="o">=</span> <span class="n">autogen</span><span class="o">.</span><span class="n">AssistantAgent</span><span class="p">(</span>
    <span class="n">name</span><span class="o">=</span><span class="s2">"Planner"</span><span class="p">,</span>
    <span class="n">system_message</span><span class="o">=</span><span class="s2">"""You are a strategic planner. Break complex tasks into</span>
<span class="s2">    clear subtasks and assign them to the appropriate specialist agents.</span>
<span class="s2">    Always end your message with a clear action item for another agent."""</span><span class="p">,</span>
    <span class="n">llm_config</span><span class="o">=</span><span class="n">llm_config</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">coder</span> <span class="o">=</span> <span class="n">autogen</span><span class="o">.</span><span class="n">AssistantAgent</span><span class="p">(</span>
    <span class="n">name</span><span class="o">=</span><span class="s2">"Coder"</span><span class="p">,</span>
    <span class="n">system_message</span><span class="o">=</span><span class="s2">"""You are an expert Python programmer. Write clean,</span>
<span class="s2">    well-documented code. Always test your code before presenting it."""</span><span class="p">,</span>
    <span class="n">llm_config</span><span class="o">=</span><span class="n">llm_config</span><span class="p">,</span>
    <span class="n">code_execution_config</span><span class="o">=</span><span class="p">{</span><span class="s2">"work_dir"</span><span class="p">:</span> <span class="s2">"coding"</span><span class="p">,</span> <span class="s2">"use_docker"</span><span class="p">:</span> <span class="kc">True</span><span class="p">},</span>
<span class="p">)</span>

<span class="n">critic</span> <span class="o">=</span> <span class="n">autogen</span><span class="o">.</span><span class="n">AssistantAgent</span><span class="p">(</span>
    <span class="n">name</span><span class="o">=</span><span class="s2">"Critic"</span><span class="p">,</span>
    <span class="n">system_message</span><span class="o">=</span><span class="s2">"""You review code and plans for correctness, efficiency,</span>
<span class="s2">    and security. Provide specific, actionable feedback."""</span><span class="p">,</span>
    <span class="n">llm_config</span><span class="o">=</span><span class="n">llm_config</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">user_proxy</span> <span class="o">=</span> <span class="n">autogen</span><span class="o">.</span><span class="n">UserProxyAgent</span><span class="p">(</span>
    <span class="n">name</span><span class="o">=</span><span class="s2">"UserProxy"</span><span class="p">,</span>
    <span class="n">human_input_mode</span><span class="o">=</span><span class="s2">"TERMINATE"</span><span class="p">,</span>
    <span class="n">max_consecutive_auto_reply</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span>
    <span class="n">is_termination_msg</span><span class="o">=</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="s2">"TASK_COMPLETE"</span> <span class="ow">in</span> <span class="n">x</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"content"</span><span class="p">,</span> <span class="s2">""</span><span class="p">),</span>
    <span class="n">code_execution_config</span><span class="o">=</span><span class="p">{</span><span class="s2">"work_dir"</span><span class="p">:</span> <span class="s2">"output"</span><span class="p">,</span> <span class="s2">"use_docker"</span><span class="p">:</span> <span class="kc">False</span><span class="p">},</span>
<span class="p">)</span>

<span class="c1"># 采用基于 LLM 的发言者选择的群聊</span>
<span class="n">groupchat</span> <span class="o">=</span> <span class="n">autogen</span><span class="o">.</span><span class="n">GroupChat</span><span class="p">(</span>
    <span class="n">agents</span><span class="o">=</span><span class="p">[</span><span class="n">user_proxy</span><span class="p">,</span> <span class="n">planner</span><span class="p">,</span> <span class="n">coder</span><span class="p">,</span> <span class="n">critic</span><span class="p">],</span>
    <span class="n">messages</span><span class="o">=</span><span class="p">[],</span>
    <span class="n">max_round</span><span class="o">=</span><span class="mi">20</span><span class="p">,</span>
    <span class="n">speaker_selection_method</span><span class="o">=</span><span class="s2">"auto"</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">manager</span> <span class="o">=</span> <span class="n">autogen</span><span class="o">.</span><span class="n">GroupChatManager</span><span class="p">(</span><span class="n">groupchat</span><span class="o">=</span><span class="n">groupchat</span><span class="p">,</span> <span class="n">llm_config</span><span class="o">=</span><span class="n">llm_config</span><span class="p">)</span>

<span class="c1"># 发起会话</span>
<span class="n">user_proxy</span><span class="o">.</span><span class="n">initiate_chat</span><span class="p">(</span>
    <span class="n">manager</span><span class="p">,</span>
    <span class="n">message</span><span class="o">=</span><span class="s2">"Analyze the CSV dataset in 'sales_data.csv' and generate a summary report with visualizations."</span>
<span class="p">)</span>
</code></pre></div>

#### 代码执行 Agent。 {#code-execution-agents.}

<p>代码执行能力是 AutoGen 的一大特色。<code>UserProxyAgent</code> 可在沙箱环境（Docker 容器或本地进程）中执行 Python 与 shell 代码，使 Agent 能够迭代地编写、测试和修复代码。</p>

<div class="admonition warning"><p class="admonition-title">AutoGen 安全注意事项</p>
<p>代码执行 Agent 可以运行任意代码。在生产环境中务必使用 Docker 进行隔离。将 <code>code_execution_config</code> 配置为 <code>&quot;use_docker&quot;: True</code> 并限制网络访问。绝不可以高权限运行 AutoGen 的代码执行 Agent。</p>
</div>

## 26.3.3　CrewAI {#subsubsec:crewai}

<p>CrewAI <a class="book-citation" id="cite-moura2023crewai-1" href="../../references.md#ref-moura2023crewai" aria-label="参考文献 261">[261]</a> 为多 Agent 系统引入了一种<em>基于角色（role-based）</em>的范式，灵感来自组织管理学。Agent 通过其职业角色、目标和背景故事来定义——这一设计借力 LLM 对人类组织结构的理解。</p>

#### 核心抽象。 {#core-abstractions.}

<ul>
<li><p><strong>Agent</strong>：由 <code>role</code>（角色）、<code>goal</code>（目标）、<code>backstory</code>（背景故事）和可用 <code>tools</code>（工具）定义</p></li>
<li><p><strong>Task</strong>：一项具体任务，包含 <code>description</code>（描述）、<code>expected_output</code>（预期输出）以及指定的 <code>agent</code></p></li>
<li><p><strong>Crew</strong>：Agent 与任务的集合，附带执行 <code>process</code>（顺序或层级）</p></li>
<li><p><strong>Process</strong>：执行策略——<code>sequential</code>（任务按顺序运行）或 <code>hierarchical</code>（由 manager Agent 进行任务分派）</p></li>
</ul>

<p class="code-caption"><span class="code-number">代码 26.6　</span>CrewAI 基于角色的 Agent 团队</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">crewai</span><span class="w"> </span><span class="kn">import</span> <span class="n">Agent</span><span class="p">,</span> <span class="n">Task</span><span class="p">,</span> <span class="n">Crew</span><span class="p">,</span> <span class="n">Process</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">crewai_tools</span><span class="w"> </span><span class="kn">import</span> <span class="n">SerperDevTool</span><span class="p">,</span> <span class="n">WebsiteSearchTool</span>

<span class="n">search_tool</span> <span class="o">=</span> <span class="n">SerperDevTool</span><span class="p">()</span>
<span class="n">web_tool</span> <span class="o">=</span> <span class="n">WebsiteSearchTool</span><span class="p">()</span>

<span class="c1"># 用丰富的角色描述定义 Agent</span>
<span class="n">researcher</span> <span class="o">=</span> <span class="n">Agent</span><span class="p">(</span>
    <span class="n">role</span><span class="o">=</span><span class="s2">"Senior Research Analyst"</span><span class="p">,</span>
    <span class="n">goal</span><span class="o">=</span><span class="s2">"Uncover cutting-edge developments in AI and provide "</span>
         <span class="s2">"comprehensive, accurate research summaries"</span><span class="p">,</span>
    <span class="n">backstory</span><span class="o">=</span><span class="s2">"""You are a seasoned research analyst with 15 years of</span>
<span class="s2">    experience in technology research. You have a talent for finding</span>
<span class="s2">    obscure but highly relevant information and synthesizing it into</span>
<span class="s2">    clear, actionable insights."""</span><span class="p">,</span>
    <span class="n">tools</span><span class="o">=</span><span class="p">[</span><span class="n">search_tool</span><span class="p">,</span> <span class="n">web_tool</span><span class="p">],</span>
    <span class="n">verbose</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">allow_delegation</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">writer</span> <span class="o">=</span> <span class="n">Agent</span><span class="p">(</span>
    <span class="n">role</span><span class="o">=</span><span class="s2">"Tech Content Strategist"</span><span class="p">,</span>
    <span class="n">goal</span><span class="o">=</span><span class="s2">"Craft compelling, technically accurate content that "</span>
         <span class="s2">"engages both technical and non-technical audiences"</span><span class="p">,</span>
    <span class="n">backstory</span><span class="o">=</span><span class="s2">"""You are a renowned content strategist known for</span>
<span class="s2">    translating complex technical concepts into engaging narratives.</span>
<span class="s2">    Your writing has appeared in major tech publications."""</span><span class="p">,</span>
    <span class="n">tools</span><span class="o">=</span><span class="p">[</span><span class="n">web_tool</span><span class="p">],</span>
    <span class="n">verbose</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">allow_delegation</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
<span class="p">)</span>

<span class="c1"># 定义任务并明确预期输出</span>
<span class="n">research_task</span> <span class="o">=</span> <span class="n">Task</span><span class="p">(</span>
    <span class="n">description</span><span class="o">=</span><span class="s2">"""Conduct comprehensive research on </span><span class="si">{topic}</span><span class="s2">.</span>
<span class="s2">    Identify key trends, major players, recent breakthroughs,</span>
<span class="s2">    and potential future directions. Focus on developments from</span>
<span class="s2">    the past 6 months."""</span><span class="p">,</span>
    <span class="n">expected_output</span><span class="o">=</span><span class="s2">"""A detailed research report with:</span>
<span class="s2">    - Executive summary (200 words)</span>
<span class="s2">    - Key findings (5-7 bullet points)</span>
<span class="s2">    - Detailed analysis (500 words)</span>
<span class="s2">    - Sources and citations"""</span><span class="p">,</span>
    <span class="n">agent</span><span class="o">=</span><span class="n">researcher</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">writing_task</span> <span class="o">=</span> <span class="n">Task</span><span class="p">(</span>
    <span class="n">description</span><span class="o">=</span><span class="s2">"""Using the research provided, write a compelling</span>
<span class="s2">    blog post about </span><span class="si">{topic}</span><span class="s2"> for a technical audience."""</span><span class="p">,</span>
    <span class="n">expected_output</span><span class="o">=</span><span class="s2">"""A polished blog post (800-1000 words) with:</span>
<span class="s2">    - Engaging headline</span>
<span class="s2">    - Introduction hook</span>
<span class="s2">    - 3-4 main sections with subheadings</span>
<span class="s2">    - Conclusion with call to action"""</span><span class="p">,</span>
    <span class="n">agent</span><span class="o">=</span><span class="n">writer</span><span class="p">,</span>
    <span class="n">context</span><span class="o">=</span><span class="p">[</span><span class="n">research_task</span><span class="p">],</span>  <span class="c1"># 依赖研究任务的输出</span>
<span class="p">)</span>

<span class="c1"># 组装 Crew</span>
<span class="n">crew</span> <span class="o">=</span> <span class="n">Crew</span><span class="p">(</span>
    <span class="n">agents</span><span class="o">=</span><span class="p">[</span><span class="n">researcher</span><span class="p">,</span> <span class="n">writer</span><span class="p">],</span>
    <span class="n">tasks</span><span class="o">=</span><span class="p">[</span><span class="n">research_task</span><span class="p">,</span> <span class="n">writing_task</span><span class="p">],</span>
    <span class="n">process</span><span class="o">=</span><span class="n">Process</span><span class="o">.</span><span class="n">sequential</span><span class="p">,</span>
    <span class="n">verbose</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">result</span> <span class="o">=</span> <span class="n">crew</span><span class="o">.</span><span class="n">kickoff</span><span class="p">(</span><span class="n">inputs</span><span class="o">=</span><span class="p">{</span><span class="s2">"topic"</span><span class="p">:</span> <span class="s2">"Reinforcement Learning for LLMs"</span><span class="p">})</span>
</code></pre></div>

#### 层级化执行（Hierarchical Process）。 {#hierarchical-process.}

<p>在层级模式下，CrewAI 会自动创建一个 manager Agent，根据各 worker Agent 的角色与能力把任务分派下去。这种方式贴近真实组织结构，能在无需显式排序的情况下处理复杂且相互依赖的 Workflow。</p>

## 26.3.4　OpenAI Assistants API 与 Agents SDK {#subsubsec:openai-agents}

<p>OpenAI 为 Agent 开发提供了两套互补产品：<strong>Assistants API</strong>，面向有状态 Agent 的托管基础设施；以及 <strong>Agents SDK</strong> <a class="book-citation" id="cite-openai2024agentssdk-1" href="../../references.md#ref-openai2024agentssdk" aria-label="参考文献 277">[277]</a>（前身为 Swarm），一个面向多 Agent 编排的轻量 Python 库。</p>

#### Assistants API 架构。 {#assistants-api-architecture.}

<p>Assistants API 通过三个核心对象在服务端管理 Agent 状态：</p>

<ul>
<li><p><strong>Assistant</strong>：一个已配置的 Agent，包含模型、指令和 Tool</p></li>
<li><p><strong>Thread</strong>：与用户会话关联的持久化对话历史</p></li>
<li><p><strong>Run</strong>：Assistant 在某 Thread 上的一次执行，具有状态生命周期（<code>queued</code> <span class="arithmatex">\(\to\)</span> <code>in_progress</code> <span class="arithmatex">\(\to\)</span> <code>requires_action</code> <span class="arithmatex">\(\to\)</span> <code>completed</code>）</p></li>
</ul>

#### 内置 Tool。 {#built-in-tools.}

<p>Assistants API 提供三种无需外部基础设施的托管 Tool：</p>

<ul>
<li><p><strong>Code Interpreter</strong>：在带文件 I/O 的沙箱环境中执行 Python</p></li>
<li><p><strong>File Search</strong>：基于向量库（vector store）的上传文档检索</p></li>
<li><p><strong>Web Search</strong>：实时网页浏览（在部分模型中可用）</p></li>
</ul>

<p class="code-caption"><span class="code-number">代码 26.7　</span>OpenAI Assistants API 与 Tool 使用</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">OpenAI</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">time</span>

<span class="n">client</span> <span class="o">=</span> <span class="n">OpenAI</span><span class="p">()</span>

<span class="c1"># 创建一个持久化的 assistant</span>
<span class="n">assistant</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">beta</span><span class="o">.</span><span class="n">assistants</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
    <span class="n">name</span><span class="o">=</span><span class="s2">"Data Analysis Assistant"</span><span class="p">,</span>
    <span class="n">instructions</span><span class="o">=</span><span class="s2">"""You are an expert data analyst. When given data files,</span>
<span class="s2">    analyze them thoroughly and provide actionable insights with</span>
<span class="s2">    visualizations where appropriate."""</span><span class="p">,</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">,</span>
    <span class="n">tools</span><span class="o">=</span><span class="p">[</span>
        <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"code_interpreter"</span><span class="p">},</span>
        <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"file_search"</span><span class="p">},</span>
    <span class="p">],</span>
<span class="p">)</span>

<span class="c1"># 为用户会话创建一个 thread</span>
<span class="n">thread</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">beta</span><span class="o">.</span><span class="n">threads</span><span class="o">.</span><span class="n">create</span><span class="p">()</span>

<span class="c1"># 上传数据文件</span>
<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="s2">"sales_data.csv"</span><span class="p">,</span> <span class="s2">"rb"</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">file</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">files</span><span class="o">.</span><span class="n">create</span><span class="p">(</span><span class="n">file</span><span class="o">=</span><span class="n">f</span><span class="p">,</span> <span class="n">purpose</span><span class="o">=</span><span class="s2">"assistants"</span><span class="p">)</span>

<span class="c1"># 添加带文件附件的消息</span>
<span class="n">client</span><span class="o">.</span><span class="n">beta</span><span class="o">.</span><span class="n">threads</span><span class="o">.</span><span class="n">messages</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
    <span class="n">thread_id</span><span class="o">=</span><span class="n">thread</span><span class="o">.</span><span class="n">id</span><span class="p">,</span>
    <span class="n">role</span><span class="o">=</span><span class="s2">"user"</span><span class="p">,</span>
    <span class="n">content</span><span class="o">=</span><span class="s2">"Analyze this sales data and identify the top 3 trends."</span><span class="p">,</span>
    <span class="n">attachments</span><span class="o">=</span><span class="p">[{</span><span class="s2">"file_id"</span><span class="p">:</span> <span class="n">file</span><span class="o">.</span><span class="n">id</span><span class="p">,</span> <span class="s2">"tools"</span><span class="p">:</span> <span class="p">[{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"code_interpreter"</span><span class="p">}]}],</span>
<span class="p">)</span>

<span class="c1"># 创建并轮询 run</span>
<span class="n">run</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">beta</span><span class="o">.</span><span class="n">threads</span><span class="o">.</span><span class="n">runs</span><span class="o">.</span><span class="n">create_and_poll</span><span class="p">(</span>
    <span class="n">thread_id</span><span class="o">=</span><span class="n">thread</span><span class="o">.</span><span class="n">id</span><span class="p">,</span>
    <span class="n">assistant_id</span><span class="o">=</span><span class="n">assistant</span><span class="o">.</span><span class="n">id</span><span class="p">,</span>
<span class="p">)</span>

<span class="k">if</span> <span class="n">run</span><span class="o">.</span><span class="n">status</span> <span class="o">==</span> <span class="s2">"completed"</span><span class="p">:</span>
    <span class="n">messages</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">beta</span><span class="o">.</span><span class="n">threads</span><span class="o">.</span><span class="n">messages</span><span class="o">.</span><span class="n">list</span><span class="p">(</span><span class="n">thread_id</span><span class="o">=</span><span class="n">thread</span><span class="o">.</span><span class="n">id</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">messages</span><span class="o">.</span><span class="n">data</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">content</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">text</span><span class="o">.</span><span class="n">value</span><span class="p">)</span>
<span class="k">elif</span> <span class="n">run</span><span class="o">.</span><span class="n">status</span> <span class="o">==</span> <span class="s2">"requires_action"</span><span class="p">:</span>
    <span class="c1"># 处理函数式 Tool 调用</span>
    <span class="n">tool_calls</span> <span class="o">=</span> <span class="n">run</span><span class="o">.</span><span class="n">required_action</span><span class="o">.</span><span class="n">submit_tool_outputs</span><span class="o">.</span><span class="n">tool_calls</span>
    <span class="n">outputs</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">tc</span> <span class="ow">in</span> <span class="n">tool_calls</span><span class="p">:</span>
        <span class="n">result</span> <span class="o">=</span> <span class="n">dispatch_tool</span><span class="p">(</span><span class="n">tc</span><span class="o">.</span><span class="n">function</span><span class="o">.</span><span class="n">name</span><span class="p">,</span> <span class="n">tc</span><span class="o">.</span><span class="n">function</span><span class="o">.</span><span class="n">arguments</span><span class="p">)</span>
        <span class="n">outputs</span><span class="o">.</span><span class="n">append</span><span class="p">({</span><span class="s2">"tool_call_id"</span><span class="p">:</span> <span class="n">tc</span><span class="o">.</span><span class="n">id</span><span class="p">,</span> <span class="s2">"output"</span><span class="p">:</span> <span class="n">result</span><span class="p">})</span>
    <span class="n">client</span><span class="o">.</span><span class="n">beta</span><span class="o">.</span><span class="n">threads</span><span class="o">.</span><span class="n">runs</span><span class="o">.</span><span class="n">submit_tool_outputs</span><span class="p">(</span>
        <span class="n">thread_id</span><span class="o">=</span><span class="n">thread</span><span class="o">.</span><span class="n">id</span><span class="p">,</span> <span class="n">run_id</span><span class="o">=</span><span class="n">run</span><span class="o">.</span><span class="n">id</span><span class="p">,</span> <span class="n">tool_outputs</span><span class="o">=</span><span class="n">outputs</span>
    <span class="p">)</span>
</code></pre></div>

#### OpenAI Agents SDK：Swarm 模式。 {#openai-agents-sdk-swarm-patterns.}

<p>Agents SDK 提供了一个面向多 Agent 交接（handoff）的轻量框架。核心原语是 <em>handoff</em>：一个 Agent 可以把控制权连同上下文一起移交给另一个 Agent，从而支持由专门化 Agent 处理特定子任务的模块化架构。</p>

<p class="code-caption"><span class="code-number">代码 26.8　</span>OpenAI Agents SDK：handoff 与 guardrail</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">agents</span><span class="w"> </span><span class="kn">import</span> <span class="n">Agent</span><span class="p">,</span> <span class="n">Runner</span><span class="p">,</span> <span class="n">RunConfig</span><span class="p">,</span> <span class="n">handoff</span><span class="p">,</span> <span class="n">InputGuardrail</span><span class="p">,</span> <span class="n">GuardrailFunctionOutput</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pydantic</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseModel</span>

<span class="c1"># 输入校验 guardrail</span>
<span class="k">class</span><span class="w"> </span><span class="nc">SafetyCheck</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">is_safe</span><span class="p">:</span> <span class="nb">bool</span>
    <span class="n">reason</span><span class="p">:</span> <span class="nb">str</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">safety_guardrail</span><span class="p">(</span><span class="n">ctx</span><span class="p">,</span> <span class="n">agent</span><span class="p">,</span> <span class="n">input_data</span><span class="p">):</span>
    <span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="n">Runner</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
        <span class="n">Agent</span><span class="p">(</span>
            <span class="n">name</span><span class="o">=</span><span class="s2">"SafetyChecker"</span><span class="p">,</span>
            <span class="n">instructions</span><span class="o">=</span><span class="s2">"Check if the request is safe and appropriate."</span><span class="p">,</span>
            <span class="n">output_type</span><span class="o">=</span><span class="n">SafetyCheck</span><span class="p">,</span>
        <span class="p">),</span>
        <span class="n">input_data</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="n">GuardrailFunctionOutput</span><span class="p">(</span>
        <span class="n">output_info</span><span class="o">=</span><span class="n">result</span><span class="o">.</span><span class="n">final_output</span><span class="p">,</span>
        <span class="n">tripwire_triggered</span><span class="o">=</span><span class="ow">not</span> <span class="n">result</span><span class="o">.</span><span class="n">final_output</span><span class="o">.</span><span class="n">is_safe</span><span class="p">,</span>
    <span class="p">)</span>

<span class="c1"># 专门化 Agent</span>
<span class="n">billing_agent</span> <span class="o">=</span> <span class="n">Agent</span><span class="p">(</span>
    <span class="n">name</span><span class="o">=</span><span class="s2">"BillingAgent"</span><span class="p">,</span>
    <span class="n">instructions</span><span class="o">=</span><span class="s2">"Handle billing inquiries, refunds, and payment issues."</span><span class="p">,</span>
    <span class="n">tools</span><span class="o">=</span><span class="p">[</span><span class="n">lookup_invoice</span><span class="p">,</span> <span class="n">process_refund</span><span class="p">],</span>
<span class="p">)</span>

<span class="n">technical_agent</span> <span class="o">=</span> <span class="n">Agent</span><span class="p">(</span>
    <span class="n">name</span><span class="o">=</span><span class="s2">"TechnicalAgent"</span><span class="p">,</span>
    <span class="n">instructions</span><span class="o">=</span><span class="s2">"Resolve technical issues and bugs."</span><span class="p">,</span>
    <span class="n">tools</span><span class="o">=</span><span class="p">[</span><span class="n">check_system_status</span><span class="p">,</span> <span class="n">create_ticket</span><span class="p">],</span>
<span class="p">)</span>

<span class="c1"># 带 handoff 的分诊 Agent</span>
<span class="n">triage_agent</span> <span class="o">=</span> <span class="n">Agent</span><span class="p">(</span>
    <span class="n">name</span><span class="o">=</span><span class="s2">"TriageAgent"</span><span class="p">,</span>
    <span class="n">instructions</span><span class="o">=</span><span class="s2">"""Classify customer requests and route to the appropriate</span>
<span class="s2">    specialist. Use handoffs to transfer to billing or technical agents."""</span><span class="p">,</span>
    <span class="n">handoffs</span><span class="o">=</span><span class="p">[</span>
        <span class="n">handoff</span><span class="p">(</span><span class="n">billing_agent</span><span class="p">,</span> <span class="n">tool_name_override</span><span class="o">=</span><span class="s2">"transfer_to_billing"</span><span class="p">),</span>
        <span class="n">handoff</span><span class="p">(</span><span class="n">technical_agent</span><span class="p">,</span> <span class="n">tool_name_override</span><span class="o">=</span><span class="s2">"transfer_to_technical"</span><span class="p">),</span>
    <span class="p">],</span>
    <span class="n">input_guardrails</span><span class="o">=</span><span class="p">[</span><span class="n">InputGuardrail</span><span class="p">(</span><span class="n">guardrail_function</span><span class="o">=</span><span class="n">safety_guardrail</span><span class="p">)],</span>
<span class="p">)</span>

<span class="c1"># 以启用 tracing 的方式运行</span>
<span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="n">Runner</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
    <span class="n">triage_agent</span><span class="p">,</span>
    <span class="s2">"I was charged twice for my subscription last month."</span><span class="p">,</span>
    <span class="n">run_config</span><span class="o">=</span><span class="n">RunConfig</span><span class="p">(</span><span class="n">tracing_disabled</span><span class="o">=</span><span class="kc">False</span><span class="p">),</span>
<span class="p">)</span>
</code></pre></div>

## 26.3.5　DSPy {#subsubsec:dspy}

<p>DSPy <a class="book-citation" id="cite-khattab2023dspy-1" href="../../references.md#ref-khattab2023dspy" aria-label="参考文献 183">[183]</a>（Declarative Self-improving Python，声明式自改进 Python）采用了截然不同的 Agent 开发思路：与其手工调 Prompt，DSPy 通过自动化优化<em>编译</em>高层程序规约（program specification）为优化后的 Prompt。</p>

#### 核心理念。 {#core-philosophy.}

<p>DSPy 把模块<em>应做什么</em>（signature）与<em>应怎样做</em>（Prompt）分离开来。然后由优化器搜索最佳 Prompt 与少样本示例，使其在开发集上的某项指标最大化。这让 DSPy 程序对模型更替更鲁棒，并省去了手工 Prompt 调优的麻烦。</p>

<p class="code-caption"><span class="code-number">代码 26.9　</span>DSPy signature 与 module</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">dspy</span>

<span class="c1"># 配置语言模型</span>
<span class="n">lm</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">LM</span><span class="p">(</span><span class="s2">"openai/gpt-4o"</span><span class="p">,</span> <span class="n">temperature</span><span class="o">=</span><span class="mf">0.0</span><span class="p">)</span>
<span class="n">dspy</span><span class="o">.</span><span class="n">configure</span><span class="p">(</span><span class="n">lm</span><span class="o">=</span><span class="n">lm</span><span class="p">)</span>

<span class="c1"># Signature 定义输入/输出契约</span>
<span class="k">class</span><span class="w"> </span><span class="nc">GenerateAnswer</span><span class="p">(</span><span class="n">dspy</span><span class="o">.</span><span class="n">Signature</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""Answer questions with factual, concise responses."""</span>
    <span class="n">context</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">InputField</span><span class="p">(</span><span class="n">desc</span><span class="o">=</span><span class="s2">"Relevant passages"</span><span class="p">)</span>
    <span class="n">question</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">InputField</span><span class="p">()</span>
    <span class="n">answer</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">OutputField</span><span class="p">(</span><span class="n">desc</span><span class="o">=</span><span class="s2">"Concise factual answer"</span><span class="p">)</span>

<span class="k">class</span><span class="w"> </span><span class="nc">AssessAnswer</span><span class="p">(</span><span class="n">dspy</span><span class="o">.</span><span class="n">Signature</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""Assess whether an answer is faithful to the context."""</span>
    <span class="n">context</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">InputField</span><span class="p">()</span>
    <span class="n">question</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">InputField</span><span class="p">()</span>
    <span class="n">answer</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">InputField</span><span class="p">()</span>
    <span class="n">faithful</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">OutputField</span><span class="p">()</span>
    <span class="n">confidence</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">OutputField</span><span class="p">(</span><span class="n">desc</span><span class="o">=</span><span class="s2">"Confidence score 0-1"</span><span class="p">)</span>

<span class="c1"># Module 把 signature 组合成程序</span>
<span class="k">class</span><span class="w"> </span><span class="nc">RAGAgent</span><span class="p">(</span><span class="n">dspy</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">num_passages</span><span class="o">=</span><span class="mi">3</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">retrieve</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">Retrieve</span><span class="p">(</span><span class="n">k</span><span class="o">=</span><span class="n">num_passages</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">generate</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">ChainOfThought</span><span class="p">(</span><span class="n">GenerateAnswer</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">assess</span> <span class="o">=</span> <span class="n">dspy</span><span class="o">.</span><span class="n">Predict</span><span class="p">(</span><span class="n">AssessAnswer</span><span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">dspy</span><span class="o">.</span><span class="n">Prediction</span><span class="p">:</span>
        <span class="n">context</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">retrieve</span><span class="p">(</span><span class="n">question</span><span class="p">)</span><span class="o">.</span><span class="n">passages</span>
        <span class="n">prediction</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">generate</span><span class="p">(</span><span class="n">context</span><span class="o">=</span><span class="n">context</span><span class="p">,</span> <span class="n">question</span><span class="o">=</span><span class="n">question</span><span class="p">)</span>

        <span class="c1"># 带断言的自评</span>
        <span class="n">assessment</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">assess</span><span class="p">(</span>
            <span class="n">context</span><span class="o">=</span><span class="n">context</span><span class="p">,</span>
            <span class="n">question</span><span class="o">=</span><span class="n">question</span><span class="p">,</span>
            <span class="n">answer</span><span class="o">=</span><span class="n">prediction</span><span class="o">.</span><span class="n">answer</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">dspy</span><span class="o">.</span><span class="n">Assert</span><span class="p">(</span>
            <span class="n">assessment</span><span class="o">.</span><span class="n">faithful</span><span class="p">,</span>
            <span class="s2">"Answer not faithful to context "</span>
            <span class="s2">"(confidence: "</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">assessment</span><span class="o">.</span><span class="n">confidence</span><span class="p">)</span> <span class="o">+</span> <span class="s2">")"</span>
        <span class="p">)</span>
        <span class="k">return</span> <span class="n">prediction</span>
</code></pre></div>

#### 优化器（Optimizer）。 {#optimizers.}

<p>DSPy 的优化器可以自动改进程序性能：</p>

<p class="code-caption"><span class="code-number">代码 26.10　</span>用 MIPRO 进行 DSPy 优化</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">dspy.teleprompt</span><span class="w"> </span><span class="kn">import</span> <span class="n">MIPROv2</span>

<span class="c1"># 定义评测指标</span>
<span class="k">def</span><span class="w"> </span><span class="nf">answer_metric</span><span class="p">(</span><span class="n">example</span><span class="p">,</span> <span class="n">prediction</span><span class="p">,</span> <span class="n">trace</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">example</span><span class="o">.</span><span class="n">answer</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span> <span class="ow">in</span> <span class="n">prediction</span><span class="o">.</span><span class="n">answer</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span>

<span class="c1"># 用 MIPRO 优化器编译</span>
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">MIPROv2</span><span class="p">(</span>
    <span class="n">metric</span><span class="o">=</span><span class="n">answer_metric</span><span class="p">,</span>
    <span class="n">auto</span><span class="o">=</span><span class="s2">"medium"</span><span class="p">,</span>  <span class="c1"># 控制优化预算</span>
<span class="p">)</span>

<span class="n">compiled_agent</span> <span class="o">=</span> <span class="n">optimizer</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span>
    <span class="n">RAGAgent</span><span class="p">(),</span>
    <span class="n">trainset</span><span class="o">=</span><span class="n">train_examples</span><span class="p">,</span>
    <span class="n">num_candidates</span><span class="o">=</span><span class="mi">30</span><span class="p">,</span>
    <span class="n">max_bootstrapped_demos</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">max_labeled_demos</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span>
<span class="p">)</span>

<span class="c1"># 保存优化后的程序</span>
<span class="n">compiled_agent</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="s2">"optimized_rag_agent.json"</span><span class="p">)</span>
</code></pre></div>

<div class="admonition tip"><p class="admonition-title">何时使用 DSPy</p>
<p>DSPy 在以下场景表现尤佳：(1) 有明确的评测指标；(2) 拥有 50 条以上样本的开发集；(3) 需要把 Agent 在不同 LLM 之间迁移；或 (4) 手工 Prompt 工程已陷入瓶颈。它不太适合那种「正确」输出本身就主观的高度创意类任务。</p>
</div>

## 26.3.6　Semantic Kernel（Microsoft） {#subsubsec:semantic-kernel}

<p>Semantic Kernel <a class="book-citation" id="cite-microsoft2023semantickernel-1" href="../../references.md#ref-microsoft2023semantickernel" aria-label="参考文献 256">[256]</a>（SK）是 Microsoft 面向企业的 Agent 框架，专为与已有软件系统及组织级 Workflow 集成而设计。它提供一种<em>插件架构（plugin architecture）</em>，让开发者能把已有业务逻辑暴露为 AI 可调用的函数。</p>

#### 插件架构。 {#plugin-architecture.}

<p>插件是 kernel 可调用的函数集合（即「skill」），可通过以下方式定义：</p>

<ul>
<li><p><strong>原生函数</strong>：使用 <code>@kernel_function</code> 装饰的普通 Python/C# 方法</p></li>
<li><p><strong>Prompt 函数</strong>：以文件形式存放的参数化 Prompt 模板</p></li>
<li><p><strong>OpenAPI 插件</strong>：从 OpenAPI 规范自动生成</p></li>
</ul>

<p class="code-caption"><span class="code-number">代码 26.11　</span>Semantic Kernel 插件与 planner</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">semantic_kernel</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">sk</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">semantic_kernel.functions</span><span class="w"> </span><span class="kn">import</span> <span class="n">kernel_function</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">semantic_kernel.connectors.ai.open_ai</span><span class="w"> </span><span class="kn">import</span> <span class="n">OpenAIChatCompletion</span>

<span class="n">kernel</span> <span class="o">=</span> <span class="n">sk</span><span class="o">.</span><span class="n">Kernel</span><span class="p">()</span>
<span class="n">kernel</span><span class="o">.</span><span class="n">add_service</span><span class="p">(</span><span class="n">OpenAIChatCompletion</span><span class="p">(</span><span class="n">ai_model_id</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">))</span>

<span class="c1"># 定义一个原生插件</span>
<span class="k">class</span><span class="w"> </span><span class="nc">EmailPlugin</span><span class="p">:</span>
    <span class="nd">@kernel_function</span><span class="p">(</span><span class="n">description</span><span class="o">=</span><span class="s2">"Send an email to a recipient"</span><span class="p">)</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">send_email</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">recipient</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">subject</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">body</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="c1"># 与邮件服务集成</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"Email sent to </span><span class="si">{</span><span class="n">recipient</span><span class="si">}</span><span class="s2">: </span><span class="si">{</span><span class="n">subject</span><span class="si">}</span><span class="s2">"</span>

    <span class="nd">@kernel_function</span><span class="p">(</span><span class="n">description</span><span class="o">=</span><span class="s2">"Search emails by keyword"</span><span class="p">)</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">search_emails</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">max_results</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">10</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="c1"># 与邮件搜索 API 集成</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"Found </span><span class="si">{</span><span class="n">max_results</span><span class="si">}</span><span class="s2"> emails matching: </span><span class="si">{</span><span class="n">query</span><span class="si">}</span><span class="s2">"</span>

<span class="k">class</span><span class="w"> </span><span class="nc">CalendarPlugin</span><span class="p">:</span>
    <span class="nd">@kernel_function</span><span class="p">(</span><span class="n">description</span><span class="o">=</span><span class="s2">"Schedule a meeting"</span><span class="p">)</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">schedule_meeting</span><span class="p">(</span>
        <span class="bp">self</span><span class="p">,</span> <span class="n">title</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">attendees</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">datetime_str</span><span class="p">:</span> <span class="nb">str</span>
    <span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"Meeting '</span><span class="si">{</span><span class="n">title</span><span class="si">}</span><span class="s2">' scheduled for </span><span class="si">{</span><span class="n">datetime_str</span><span class="si">}</span><span class="s2">"</span>

<span class="c1"># 注册插件</span>
<span class="n">kernel</span><span class="o">.</span><span class="n">add_plugin</span><span class="p">(</span><span class="n">EmailPlugin</span><span class="p">(),</span> <span class="n">plugin_name</span><span class="o">=</span><span class="s2">"Email"</span><span class="p">)</span>
<span class="n">kernel</span><span class="o">.</span><span class="n">add_plugin</span><span class="p">(</span><span class="n">CalendarPlugin</span><span class="p">(),</span> <span class="n">plugin_name</span><span class="o">=</span><span class="s2">"Calendar"</span><span class="p">)</span>

<span class="c1"># 使用函数调用式 planner</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">semantic_kernel.planners</span><span class="w"> </span><span class="kn">import</span> <span class="n">FunctionCallingStepwisePlanner</span>

<span class="n">planner</span> <span class="o">=</span> <span class="n">FunctionCallingStepwisePlanner</span><span class="p">(</span><span class="n">service_id</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">)</span>
<span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="n">planner</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span>
    <span class="n">kernel</span><span class="p">,</span>
    <span class="s2">"Schedule a meeting with alice@company.com to discuss Q4 planning "</span>
    <span class="s2">"next Tuesday at 2pm, then send her a confirmation email."</span>
<span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">result</span><span class="p">))</span>
</code></pre></div>

#### 记忆与连接器。 {#memory-and-connectors.}

<p>Semantic Kernel 的记忆系统通过统一接口支持多种后端（Azure Cognitive Search、Chroma、Pinecone、Weaviate）。其连接器系统支持与 Microsoft 365、Azure DevOps 以及自定义 REST API 等企业服务的集成。</p>

#### 聚焦企业集成。 {#enterprise-integration-focus.}

<p>SK 特别适合企业部署，原因包括：</p>

<ul>
<li><p>针对 .NET 生态的原生 C# 支持</p></li>
<li><p>与 Azure OpenAI 的集成，支持托管身份（managed identity）认证</p></li>
<li><p>合规友好的架构，自带审计日志</p></li>
<li><p>支持本地（on-premises）模型部署</p></li>
</ul>

## 26.3.7　NVIDIA OO Agents（NOOA） {#subsubsec:nooa}

<p>NOOA <a class="book-citation" id="cite-nvidia_oo_agents_2026-1" href="../../references.md#ref-nvidia_oo_agents_2026" aria-label="参考文献 102">[102]</a> 是 NVIDIA 的模型无关框架，建立在一个激进的前提之上：<strong>Agent 就是一个 Python 对象</strong>。其他框架引入 DSL、图定义或重度装饰器抽象，而 NOOA 把 Agent 概念直接映射到 Python 的原生构造上——字段即状态，方法即能力，docstring 即提示，类型注解即契约。方法体为字面 <code>...</code>（Python 的 <code>Ellipsis</code>）的方法成为<em>智能体方法（agentic method）</em>：在运行时由 LLM 驱动的循环实现。方法体为普通代码的方法保持为确定性 Python，开发者与模型都可以调用。</p>

<div class="admonition note"><p class="admonition-title">NOOA 设计映射</p>
<span id="table-26-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>Agent 概念</strong></th><th scope="col"><strong>Python 构造</strong></th></tr></thead>
<tbody><tr><td>Agent 身份与指令</td><td>类 + 类 docstring</td></tr>
<tr><td>持久状态</td><td>实例字段（带类型）</td></tr>
<tr><td>能力 / 工具</td><td><code>self</code> 上的方法</td></tr>
<tr><td>提示</td><td>方法 docstring</td></tr>
<tr><td>I/O 契约</td><td>类型注解</td></tr>
<tr><td>LLM 驱动行为</td><td>方法体 = <code>...</code></td></tr>
<tr><td>确定性逻辑</td><td>方法体 = 普通代码</td></tr></tbody>
</table>
</div>
</div>

#### 核心编程模型。 {#heading-26-38}

<p>该框架的核心洞见是：Python 早已具备 Agent 所需的抽象。考虑一个总结论文并维护阅读清单的研究 Agent：</p>

<p class="code-caption"><span class="code-number">代码 26.12　</span>NOOA Agent：状态、确定性方法与智能体方法</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">nooa</span><span class="w"> </span><span class="kn">import</span> <span class="n">Agent</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">nooa.unifiedllm</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_llm_client</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pydantic</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseModel</span><span class="p">,</span> <span class="n">Field</span>

<span class="n">llm</span> <span class="o">=</span> <span class="n">get_llm_client</span><span class="p">(</span><span class="s2">"claude-sonnet-4-5-20250514"</span><span class="p">)</span>

<span class="k">class</span><span class="w"> </span><span class="nc">PaperSummary</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">title</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">key_findings</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span>
    <span class="n">relevance_score</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="n">Field</span><span class="p">(</span><span class="n">ge</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">le</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ResearchAgent</span><span class="p">(</span><span class="n">Agent</span><span class="p">,</span> <span class="n">llm</span><span class="o">=</span><span class="n">llm</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""You are a research assistant specializing in ML papers.</span>
<span class="sd">    Summarize concisely. Score relevance to the user's topic."""</span>

    <span class="c1"># 状态：实例上的类型化字段</span>
    <span class="n">topic</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">reading_list</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">PaperSummary</span><span class="p">]</span> <span class="o">=</span> <span class="p">[]</span>

    <span class="c1"># 确定性方法：普通 Python，可由模型或开发者调用</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">top_papers</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">n</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">5</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="n">PaperSummary</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""Return the highest-relevance papers seen so far."""</span>
        <span class="k">return</span> <span class="nb">sorted</span><span class="p">(</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">reading_list</span><span class="p">,</span>
            <span class="n">key</span><span class="o">=</span><span class="k">lambda</span> <span class="n">p</span><span class="p">:</span> <span class="n">p</span><span class="o">.</span><span class="n">relevance_score</span><span class="p">,</span>
            <span class="n">reverse</span><span class="o">=</span><span class="kc">True</span>
        <span class="p">)[:</span><span class="n">n</span><span class="p">]</span>

    <span class="c1"># 智能体方法：由 LLM 在运行时实现</span>
    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">summarize</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">abstract</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">PaperSummary</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""Read the abstract and produce a structured summary.</span>
<span class="sd">        Score relevance against self.topic."""</span>
        <span class="o">...</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">literature_review</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">abstracts</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">])</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""Summarize each abstract, add to reading_list,</span>
<span class="sd">        then synthesize a literature review of the top findings."""</span>
        <span class="o">...</span>
</code></pre></div>

<p>三件事同时发生：(1) <code>top_papers</code> 是普通方法——可单元测试、确定性，并且在智能体执行期间也可作为可调用工具提供给 LLM；(2) <code>summarize</code> 返回带类型的 <code>PaperSummary</code>——框架会依据 Pydantic schema 自动校验 LLM 输出，失败时重试；(3) <code>literature_review</code> 可以在内部同时调用 <code>summarize</code> 与 <code>top_papers</code>，在单一控制流中组合智能体逻辑与确定性逻辑。</p>

#### 代码即动作。 {#heading-26-39}

<p>当模型执行一个智能体方法时，它不输出 JSON 工具调用。相反，它在 Jupyter 风格的 REPL 中编写 Python 代码，可以访问 <code>self</code>、import 与任何辅助方法。这意味着模型的动作空间是整个 Python 语言——它可以循环、分支、捕获异常、组合方法调用，而框架无需把每个可能的动作定义成独立的工具 schema：</p>

<p class="code-caption"><span class="code-number">代码 26.13　</span>LLM 在智能体方法内部生成的内容</p>
<div class="highlight"><pre><span></span><code><span class="c1"># 在 literature_review(...) 中，模型会写：</span>
<span class="k">for</span> <span class="n">abstract</span> <span class="ow">in</span> <span class="n">abstracts</span><span class="p">:</span>
    <span class="n">summary</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">summarize</span><span class="p">(</span><span class="n">abstract</span><span class="p">)</span>
    <span class="bp">self</span><span class="o">.</span><span class="n">reading_list</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">summary</span><span class="p">)</span>

<span class="n">top</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">top_papers</span><span class="p">(</span><span class="n">n</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>
<span class="n">review</span> <span class="o">=</span> <span class="s2">"Key findings from top papers:</span><span class="se">\n</span><span class="s2">"</span>
<span class="k">for</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">top</span><span class="p">:</span>
    <span class="n">review</span> <span class="o">+=</span> <span class="sa">f</span><span class="s2">"- </span><span class="si">{</span><span class="n">p</span><span class="o">.</span><span class="n">title</span><span class="si">}</span><span class="s2">: </span><span class="si">{</span><span class="s1">', '</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">p</span><span class="o">.</span><span class="n">key_findings</span><span class="p">)</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
<span class="k">return</span> <span class="n">review</span>
</code></pre></div>

<p>模型编写的代码会调用 <code>self.summarize</code>——它本身也是一个智能体方法。这就产生了嵌套的 Agent 循环：外层 <code>literature_review</code> 循环把每篇论文委派给内层 <code>summarize</code> 循环，并完整追踪父子 span 层级。</p>

#### 按引用传递与活对象。 {#heading-26-40}

<p>与把工具输入序列化为 JSON 的框架不同，NOOA 按引用传递 Python 对象。当方法收到 <code>Order</code> 或 <code>Database</code> 对象时，模型可以在沙箱化的 REPL 中调用它的方法、检查它的字段、修改它的状态。这消除了 Agent 对象模型与序列化边界之间的阻抗失配。</p>

#### 多 Agent 组合。 {#heading-26-41}

<p>Agent 通过 Python 的对象模型自然组合——一个 Agent 持有对另一个 Agent 的引用：</p>

<p class="code-caption"><span class="code-number">代码 26.14　</span>NOOA 通过对象引用做多 Agent 组合</p>
<div class="highlight"><pre><span></span><code><span class="k">class</span><span class="w"> </span><span class="nc">PlannerAgent</span><span class="p">(</span><span class="n">Agent</span><span class="p">,</span> <span class="n">llm</span><span class="o">=</span><span class="n">llm</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""Decompose complex research tasks into sub-tasks."""</span>

    <span class="n">researcher</span><span class="p">:</span> <span class="n">ResearchAgent</span>  <span class="c1"># 子智能体，从父智能体继承 llm</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">deep_dive</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""Break the question into sub-questions, delegate each</span>
<span class="sd">        to self.researcher, then synthesize a final answer."""</span>
        <span class="o">...</span>
</code></pre></div>

<p>规划者的智能体方法可以直接调用 <code>self.researcher.summarize(...)</code> 或 <code>self.researcher.literature_review(...)</code>——没有消息总线、没有序列化协议，只是恰好会触发嵌套 LLM 循环的 Python 方法调用。</p>

#### 追踪与可观测性。 {#heading-26-42}

<p>默认情况下，每次 LLM 调用、代码执行与方法调用都会被追踪，并保留父子 span 关系。内置的追踪查看器（<code>nooa start-dev</code>）为嵌套的智能体执行提供实时可见性，无需外部可观测性基础设施。</p>

#### 模型无关性。 {#heading-26-43}

<p>NOOA 支持任何通过其 <code>UnifiedLLM</code> 层可访问的模型：Anthropic、OpenAI、通过 Ollama 的本地模型，或通过 vLLM 的自托管端点。同一个 Agent 类可以在不同提供商之间原样工作——模型选择是构造参数，而不是架构决策。</p>

<div class="admonition tip"><p class="admonition-title">何时选择 NOOA</p>
<p>当你有以下需求时，NOOA 是合适的选择：</p>

<ul>
<li><p>极小的抽象开销——懂 Python 就懂这个框架</p></li>
<li><p>带自动校验与重试的类型安全 I/O</p></li>
<li><p>通过标准 <code>pytest</code> 可测试（mock 掉 LLM，正常测试确定性方法）</p></li>
<li><p>代码即动作，而非僵硬的工具 schema</p></li>
<li><p>无序列化边界的嵌套 Agent 组合</p></li>
</ul>

<p>权衡：NOOA 是研究级软件（2026 年 7 月），没有 LangGraph 或 Semantic Kernel 那样的生产加固、托管部署或生态集成。它优先的是编程模型的清晰，而非运维成熟度。</p>
</div>
