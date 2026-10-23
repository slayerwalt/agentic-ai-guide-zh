# 16.7　智能体 RAG {#sec:agentic_rag}

## 16.7.1　动机：静态 RAG 的局限 {#motivation-limits-of-static-rag}

<p>标准 RAG 遵循固定的“先检索后生成”模式。这在以下情况下会失败：</p>

<ul>
<li><p><strong>多跳问题</strong>：“是谁创办了在 2023 年收购 OpenAI 主要竞争对手的那家公司？”这类问题需要串联多次检索</p></li>
<li><p><strong>歧义查询</strong>：正确的检索策略取决于已找到的内容</p></li>
<li><p><strong>异构来源</strong>：不同子问题需要不同的知识库</p></li>
<li><p><strong>迭代精炼</strong>：初次检索可能揭示需要换一种查询方式</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">将 RAG 视为马尔可夫决策过程</p>
<p>智能体 RAG 将检索视为一个序贯决策问题。<em>状态</em>是当前的上下文（查询 + 至今为止检索到的文档）；<em>动作</em>包括检索、推理、生成和停止；<em>Reward</em> 是答案的正确性。Agent 学习一种关于何时检索以及检索什么的 Policy。</p>
</div>

## 16.7.2　智能体 RAG 架构 {#agentic-rag-architecture}

<figure id="fig:agentic_rag"><a href="../../assets/figures/fig_051_agentic_rag.png" target="_blank" rel="noopener" aria-label="查看图 16.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_051_agentic_rag.png" alt="智能体 RAG 控制流。Agent 在返回答案前迭代地进行规划、检索、判断充分性，并自检接地性。" loading="lazy"></a><figcaption><span class="figure-number">图 16.2　</span>智能体 RAG 控制流。Agent 在返回答案前迭代地进行规划、检索、判断充分性，并自检接地性。</figcaption></figure>

## 16.7.3　多源路由 {#multi-source-routing}

<p>智能体 RAG 系统可以将子查询路由到专门化的知识源。其核心洞见是：不同问题类型需要不同的检索后端——没有任何单一索引能在所有方面都表现出色。</p>

#### 为什么要路由？ {#why-route}

<p>设想一个金融分析师助手处理四种查询：</p>

<ul>
<li><p>“我们公司的带薪休假（PTO）政策是什么？” <span class="arithmatex">\(\rightarrow\)</span> <strong>向量数据库</strong>（内部文档）</p></li>
<li><p>“美联储昨天宣布了什么？” <span class="arithmatex">\(\rightarrow\)</span> <strong>网页搜索</strong>（实时）</p></li>
<li><p>“按地区展示 Q3 营收” <span class="arithmatex">\(\rightarrow\)</span> <strong>SQL 数据库</strong>（结构化数据）</p></li>
<li><p>“我们的鉴权中间件如何校验 Token？” <span class="arithmatex">\(\rightarrow\)</span> <strong>代码索引</strong>（代码库）</p></li>
</ul>

<p>“从单一索引扁平化检索”的做法要么错过答案，要么返回无关段落。路由在检索开始前，为<em>合适的子问题选择合适的工具</em>。</p>

#### 路由策略。 {#routing-strategies.}

<p>按复杂度递增的三种主要方法：</p>

<ol>
<li><p><strong>基于规则的路由。</strong>通过关键词触发（例如，SQL 关键字 <span class="arithmatex">\(\rightarrow\)</span> 数据库，URL 模式 <span class="arithmatex">\(\rightarrow\)</span> 网页）。快速且可解释，但对歧义查询非常脆弱。</p></li>
<li><p><strong>基于分类器的路由。</strong>由一个轻量模型（如微调过的 BERT 分类器，或在查询 Embedding 上的逻辑回归）预测最佳来源。延迟低（<span class="arithmatex">\(&lt;\)</span>10 毫秒），可基于路由日志训练，但需要带标签数据。</p></li>
<li><p><strong>基于 LLM 的路由。</strong>LLM 本身在一次结构化输出调用中决定来源（见下面的代码清单）。最为灵活——可以处理新颖的查询类型并解释其推理——但会增加一次 LLM 调用的延迟。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">路由器即可学习的 Policy</p>
<p>多源路由在最简单的形式下是一个<em>分类</em>问题，在最丰富的形式下则是一个<em>规划</em>问题。当被建模为 RL Policy——其中状态是查询加上对话历史，动作是来源的选择（以及可选的查询改写），Reward 是下游答案质量——时，路由器可以通过 Policy Gradient 技术（Chapter <a href="../08-preference-optimization/index.md#ch:po-variants">8</a>）进行端到端优化。</p>
</div>

#### 实践考量。 {#practical-considerations.}

<ul>
<li><p><strong>回退链</strong>：若主要来源返回的结果置信度较低，则尝试次优来源。</p></li>
<li><p><strong>并行扇出</strong>：对于歧义查询，同时从多个来源检索，并通过倒数排名融合（Table <a href="section-03.md#tab:retrieval_methods">16.2</a>）合并结果。</p></li>
<li><p><strong>成本感知</strong>：网页搜索和 API 调用可能存在金钱开销或速率限制；路由器应将其纳入考量。</p></li>
<li><p><strong>可观测性</strong>：记录每个路由决策及其推理过程——对调试和再训练至关重要。</p></li>
</ul>

<p class="code-caption"><span class="code-number">代码 16.9　</span>多源智能体 RAG 路由器</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">enum</span><span class="w"> </span><span class="kn">import</span> <span class="n">Enum</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pydantic</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseModel</span>

<span class="k">class</span><span class="w"> </span><span class="nc">KnowledgeSource</span><span class="p">(</span><span class="nb">str</span><span class="p">,</span> <span class="n">Enum</span><span class="p">):</span>
    <span class="n">VECTOR_DB</span>   <span class="o">=</span> <span class="s2">"vector_db"</span>    <span class="c1"># 内部文档</span>
    <span class="n">WEB_SEARCH</span>  <span class="o">=</span> <span class="s2">"web_search"</span>   <span class="c1"># 实时网页</span>
    <span class="n">SQL_DB</span>      <span class="o">=</span> <span class="s2">"sql_db"</span>       <span class="c1"># 结构化数据</span>
    <span class="n">CODE_INDEX</span>  <span class="o">=</span> <span class="s2">"code_index"</span>   <span class="c1"># 代码库</span>
    <span class="n">API</span>         <span class="o">=</span> <span class="s2">"api"</span>          <span class="c1"># 外部 API</span>

<span class="k">class</span><span class="w"> </span><span class="nc">RouteDecision</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">source</span><span class="p">:</span> <span class="n">KnowledgeSource</span>
    <span class="n">refined_query</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">reasoning</span><span class="p">:</span> <span class="nb">str</span>

<span class="k">def</span><span class="w"> </span><span class="nf">route_query</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">llm</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">RouteDecision</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""使用 LLM 决定该查询哪个知识源。"""</span>
    <span class="n">prompt</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">"""Given the query: "</span><span class="si">{</span><span class="n">query</span><span class="si">}</span><span class="s2">"</span>

<span class="s2">Decide which knowledge source to use:</span>
<span class="s2">- vector_db: for internal documents, policies, past reports</span>
<span class="s2">- web_search: for current events, recent information</span>
<span class="s2">- sql_db: for numerical data, statistics, structured records</span>
<span class="s2">- code_index: for code examples, API documentation</span>
<span class="s2">- api: for real-time data (weather, stock prices, etc.)</span>

<span class="s2">Return a JSON with: source, refined_query, reasoning."""</span>

    <span class="k">return</span> <span class="n">llm</span><span class="o">.</span><span class="n">with_structured_output</span><span class="p">(</span><span class="n">RouteDecision</span><span class="p">)</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span><span class="n">prompt</span><span class="p">)</span>
</code></pre></div>

## 16.7.4　完整的智能体 RAG 实现 {#full-agentic-rag-implementation}

<p>前面几节介绍了各个独立组件——路由、检索、评估。一个完整的智能体 RAG 系统将它们编排为一个<em>有状态节点的图</em>，其中控制流取决于中间结果。下面的实现使用 LangGraph 将四个节点串成一个循环：</p>

<ol>
<li><p><strong>Plan（规划）</strong>：将用户查询分解为子查询（每个信息需求对应一个）。</p></li>
<li><p><strong>Retrieve（检索）</strong>：将每个子查询路由到合适的来源并取回文档。</p></li>
<li><p><strong>Evaluate（评估）</strong>：判断累积的上下文是否足以回答原始查询。</p></li>
<li><p><strong>Generate（生成）</strong>：综合检索到的文档生成带引用的最终答案。</p></li>
</ol>

<p>关键设计模式是<em>条件循环</em>：评估之后，Agent 要么进入生成阶段（如果上下文足够，或迭代预算耗尽），要么带着精炼后的子查询回到检索阶段。这与一个在信息收集动作上运行的 RL Agent 所遵循的“感知—行动—评估”循环如出一辙。</p>

<p class="code-caption"><span class="code-number">代码 16.10　</span>基于 LangGraph 的智能体 RAG</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">TypedDict</span><span class="p">,</span> <span class="n">Annotated</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.graph</span><span class="w"> </span><span class="kn">import</span> <span class="n">StateGraph</span><span class="p">,</span> <span class="n">END</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.prebuilt</span><span class="w"> </span><span class="kn">import</span> <span class="n">ToolNode</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">operator</span>

<span class="k">class</span><span class="w"> </span><span class="nc">AgentState</span><span class="p">(</span><span class="n">TypedDict</span><span class="p">):</span>
    <span class="n">query</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">sub_queries</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span>
    <span class="n">retrieved_docs</span><span class="p">:</span> <span class="n">Annotated</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">],</span> <span class="n">operator</span><span class="o">.</span><span class="n">add</span><span class="p">]</span>
    <span class="n">context_sufficient</span><span class="p">:</span> <span class="nb">bool</span>
    <span class="n">answer</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">iterations</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">max_iterations</span><span class="p">:</span> <span class="nb">int</span>

<span class="k">def</span><span class="w"> </span><span class="nf">plan_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">AgentState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AgentState</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""将查询分解为子查询。"""</span>
    <span class="n">sub_queries</span> <span class="o">=</span> <span class="n">decompose_query</span><span class="p">(</span><span class="n">state</span><span class="p">[</span><span class="s2">"query"</span><span class="p">])</span>
    <span class="k">return</span> <span class="p">{</span><span class="o">**</span><span class="n">state</span><span class="p">,</span> <span class="s2">"sub_queries"</span><span class="p">:</span> <span class="n">sub_queries</span><span class="p">,</span> <span class="s2">"iterations"</span><span class="p">:</span> <span class="mi">0</span><span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">retrieve_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">AgentState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AgentState</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""为当前的子查询检索文档。"""</span>
    <span class="n">new_docs</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">sq</span> <span class="ow">in</span> <span class="n">state</span><span class="p">[</span><span class="s2">"sub_queries"</span><span class="p">]:</span>
        <span class="n">source</span> <span class="o">=</span> <span class="n">route_query</span><span class="p">(</span><span class="n">sq</span><span class="p">)</span>
        <span class="n">docs</span> <span class="o">=</span> <span class="n">retrieve_from_source</span><span class="p">(</span><span class="n">sq</span><span class="p">,</span> <span class="n">source</span><span class="p">)</span>
        <span class="n">new_docs</span><span class="o">.</span><span class="n">extend</span><span class="p">(</span><span class="n">docs</span><span class="p">)</span>
    <span class="k">return</span> <span class="p">{</span><span class="o">**</span><span class="n">state</span><span class="p">,</span> <span class="s2">"retrieved_docs"</span><span class="p">:</span> <span class="n">new_docs</span><span class="p">,</span>
            <span class="s2">"iterations"</span><span class="p">:</span> <span class="n">state</span><span class="p">[</span><span class="s2">"iterations"</span><span class="p">]</span> <span class="o">+</span> <span class="mi">1</span><span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">evaluate_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">AgentState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AgentState</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""评估检索到的上下文是否充分。"""</span>
    <span class="n">sufficient</span> <span class="o">=</span> <span class="n">evaluate_context_sufficiency</span><span class="p">(</span>
        <span class="n">query</span><span class="o">=</span><span class="n">state</span><span class="p">[</span><span class="s2">"query"</span><span class="p">],</span>
        <span class="n">docs</span><span class="o">=</span><span class="n">state</span><span class="p">[</span><span class="s2">"retrieved_docs"</span><span class="p">]</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="p">{</span><span class="o">**</span><span class="n">state</span><span class="p">,</span> <span class="s2">"context_sufficient"</span><span class="p">:</span> <span class="n">sufficient</span><span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">generate_node</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">AgentState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AgentState</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""基于检索到的上下文生成回答。"""</span>
    <span class="n">answer</span> <span class="o">=</span> <span class="n">generate_with_citations</span><span class="p">(</span>
        <span class="n">query</span><span class="o">=</span><span class="n">state</span><span class="p">[</span><span class="s2">"query"</span><span class="p">],</span>
        <span class="n">docs</span><span class="o">=</span><span class="n">state</span><span class="p">[</span><span class="s2">"retrieved_docs"</span><span class="p">]</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="p">{</span><span class="o">**</span><span class="n">state</span><span class="p">,</span> <span class="s2">"answer"</span><span class="p">:</span> <span class="n">answer</span><span class="p">}</span>

<span class="k">def</span><span class="w"> </span><span class="nf">should_retrieve</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">AgentState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="k">if</span> <span class="n">state</span><span class="p">[</span><span class="s2">"context_sufficient"</span><span class="p">]:</span>
        <span class="k">return</span> <span class="s2">"generate"</span>
    <span class="k">if</span> <span class="n">state</span><span class="p">[</span><span class="s2">"iterations"</span><span class="p">]</span> <span class="o">&gt;=</span> <span class="n">state</span><span class="p">[</span><span class="s2">"max_iterations"</span><span class="p">]:</span>
        <span class="k">return</span> <span class="s2">"generate"</span>  <span class="c1"># 放弃继续检索，用现有内容生成</span>
    <span class="k">return</span> <span class="s2">"retrieve"</span>

<span class="c1"># 构建图</span>
<span class="n">workflow</span> <span class="o">=</span> <span class="n">StateGraph</span><span class="p">(</span><span class="n">AgentState</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"plan"</span><span class="p">,</span>     <span class="n">plan_node</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"retrieve"</span><span class="p">,</span> <span class="n">retrieve_node</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"evaluate"</span><span class="p">,</span> <span class="n">evaluate_node</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_node</span><span class="p">(</span><span class="s2">"generate"</span><span class="p">,</span> <span class="n">generate_node</span><span class="p">)</span>

<span class="n">workflow</span><span class="o">.</span><span class="n">set_entry_point</span><span class="p">(</span><span class="s2">"plan"</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"plan"</span><span class="p">,</span>     <span class="s2">"retrieve"</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"retrieve"</span><span class="p">,</span> <span class="s2">"evaluate"</span><span class="p">)</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_conditional_edges</span><span class="p">(</span><span class="s2">"evaluate"</span><span class="p">,</span> <span class="n">should_retrieve</span><span class="p">,</span>
    <span class="p">{</span><span class="s2">"retrieve"</span><span class="p">:</span> <span class="s2">"retrieve"</span><span class="p">,</span> <span class="s2">"generate"</span><span class="p">:</span> <span class="s2">"generate"</span><span class="p">})</span>
<span class="n">workflow</span><span class="o">.</span><span class="n">add_edge</span><span class="p">(</span><span class="s2">"generate"</span><span class="p">,</span> <span class="n">END</span><span class="p">)</span>

<span class="n">agent</span> <span class="o">=</span> <span class="n">workflow</span><span class="o">.</span><span class="n">compile</span><span class="p">()</span>

<span class="c1"># 运行</span>
<span class="n">result</span> <span class="o">=</span> <span class="n">agent</span><span class="o">.</span><span class="n">invoke</span><span class="p">({</span>
    <span class="s2">"query"</span><span class="p">:</span> <span class="s2">"What were the main causes of the 2023 banking crisis?"</span><span class="p">,</span>
    <span class="s2">"max_iterations"</span><span class="p">:</span> <span class="mi">3</span><span class="p">,</span>
    <span class="s2">"retrieved_docs"</span><span class="p">:</span> <span class="p">[],</span>
    <span class="s2">"iterations"</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span>
<span class="p">})</span>
</code></pre></div>

## 16.7.5　工具增强的 RAG {#tool-augmented-rag}

<p>智能体 RAG 可以将检索与计算工具结合：</p>

<p class="code-caption"><span class="code-number">代码 16.11　</span>结合 SQL 与检索的工具增强 RAG</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">langchain.agents</span><span class="w"> </span><span class="kn">import</span> <span class="n">create_tool_calling_agent</span><span class="p">,</span> <span class="n">AgentExecutor</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain.tools</span><span class="w"> </span><span class="kn">import</span> <span class="n">tool</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">search_documents</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""搜索内部文档知识库。"""</span>
    <span class="n">docs</span> <span class="o">=</span> <span class="n">vectorstore</span><span class="o">.</span><span class="n">similarity_search</span><span class="p">(</span><span class="n">query</span><span class="p">,</span> <span class="n">k</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>
    <span class="k">return</span> <span class="s2">"</span><span class="se">\n\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">d</span><span class="o">.</span><span class="n">page_content</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">docs</span><span class="p">)</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">query_database</span><span class="p">(</span><span class="n">sql</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""在分析数据库上执行 SQL 查询。"""</span>
    <span class="k">return</span> <span class="n">db</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">sql</span><span class="p">)</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">web_search</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""在网上搜索最新信息。"""</span>
    <span class="k">return</span> <span class="n">tavily_client</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="n">query</span><span class="p">)</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">execute_python</span><span class="p">(</span><span class="n">code</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""执行 Python 代码进行计算。"""</span>
    <span class="k">return</span> <span class="n">python_repl</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">code</span><span class="p">)</span>

<span class="n">tools</span> <span class="o">=</span> <span class="p">[</span><span class="n">search_documents</span><span class="p">,</span> <span class="n">query_database</span><span class="p">,</span> <span class="n">web_search</span><span class="p">,</span> <span class="n">execute_python</span><span class="p">]</span>
<span class="n">agent</span> <span class="o">=</span> <span class="n">create_tool_calling_agent</span><span class="p">(</span><span class="n">llm</span><span class="p">,</span> <span class="n">tools</span><span class="p">,</span> <span class="n">prompt</span><span class="p">)</span>
<span class="n">executor</span> <span class="o">=</span> <span class="n">AgentExecutor</span><span class="p">(</span><span class="n">agent</span><span class="o">=</span><span class="n">agent</span><span class="p">,</span> <span class="n">tools</span><span class="o">=</span><span class="n">tools</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</code></pre></div>

## 16.7.6　Search-R1：RL 训练的智能体 RAG {#search-r1-rl-trained-agentic-rag}

<p>上述智能体 RAG 方法依赖<em>Prompt 工程</em>的编排——Agent 的搜索行为由指令控制，而非通过训练学习。<strong>Search-R1</strong> <a class="book-citation" id="cite-jin2025searchr1-1" href="../../references.md#ref-jin2025searchr1" aria-label="参考文献 170">[170]</a> 采取了根本不同的方法：通过强化学习训练 LLM，使其<em>学会在推理过程中何时、检索什么、检索多少次</em>。</p>

#### 核心思想。 {#core-idea.}

<p>Search-R1 扩展了 DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 推理框架，将搜索引擎查询视为 RL 训练循环中的<strong>Action</strong>。在思维链（Chain-of-Thought，CoT）生成过程中，模型可以输出特殊 Token <code>&lt;search&gt;query&lt;/search&gt;</code>，触发对搜索引擎的实时检索。检索结果被注入回推理上下文，模型继续生成。</p>

#### 形式化设定。 {#formal-setup.}

<p>模型生成一个与搜索动作交错的推理轨迹：</p>



<div class="arithmatex">\[\underbrace{\text{think}_1}_{\text{reasoning}} \to \underbrace{\texttt{&lt;search&gt;}q_1\texttt{&lt;/search&gt;}}_{\text{action}} \to \underbrace{[\text{results}_1]}_{\text{observation}} \to \text{think}_2 \to \texttt{&lt;search&gt;}q_2\texttt{&lt;/search&gt;} \to \cdots \to \text{answer}\]</div>

<p>整条轨迹（推理 + 搜索 + 最终答案）由一个终端 Reward 打分：最终答案相对于真实标签的正确性。</p>

#### 训练算法。 {#training-algorithm.}

<p>Search-R1 使用组相对策略优化（Group Relative Policy Optimization, GRPO）：</p>

<ol>
<li><p><strong>每个问题采样 <span class="arithmatex">\(N\)</span> 条轨迹</strong>，每条可能包含 0–5 次搜索调用</p></li>
<li><p><strong>实时执行搜索</strong>——环境返回真实的搜索引擎结果</p></li>
<li><p><strong>对终端答案的正确性打分</strong>（与真实答案的精确匹配或 F1）</p></li>
<li><p><strong>计算组相对优势</strong>：<span class="arithmatex">\(\hat{A}_i = (R_i - \mu_G) / \sigma_G\)</span></p></li>
<li><p>使用 GRPO 的裁剪目标<strong>更新 Policy</strong>——强化那些有效搜索的轨迹</p></li>
</ol>

<p>模型学会：</p>

<ul>
<li><p><strong>在不确定时才搜索</strong>——避免对已掌握的知识进行不必要的搜索</p></li>
<li><p><strong>构造有效查询</strong>——学会能返回相关结果的查询措辞</p></li>
<li><p><strong>多次搜索</strong>——基于初次结果迭代精炼查询</p></li>
<li><p><strong>整合检索到的上下文</strong>——利用搜索结果支撑或修正其推理</p></li>
</ul>

#### Search-R1 与基于 Prompt 的智能体 RAG 的差异。 {#how-search-r1-differs-from-prompt-based-agentic-rag.}

<p class="table-caption" id="table-16-4"><span class="table-number">表 16.4　</span>Search-R1（RL 训练）与基于 Prompt 的智能体 RAG 对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>基于 Prompt 的智能体 RAG</strong></th><th scope="col"><strong>Search-R1</strong></th></tr></thead>
<tbody><tr><td>搜索决策</td><td>Prompt / 启发式</td><td>通过 RL 学习</td></tr>
<tr><td>查询构造</td><td>由 Prompt 指引（“改写查询”）</td><td>端到端训练</td></tr>
<tr><td>搜索次数</td><td>固定或推理时由 LLM 决定</td><td>学习到的最优次数</td></tr>
<tr><td>训练信号</td><td>无（冻结模型）</td><td>正确性 Reward</td></tr>
<tr><td>搜索结果整合</td><td>追加到上下文</td><td>交错在 CoT 中</td></tr>
<tr><td>失败恢复</td><td>启发式重试</td><td>学习到的退避 / 重构造</td></tr>
<tr><td>推理开销</td><td>框架开销（LangGraph）</td><td>原生模型行为</td></tr></tbody>
</table>
</div>

#### 结果。 {#results.-1-dup2}

<p>在开放域问答基准（NQ <a class="book-citation" id="cite-kwiatkowski2019natural-1" href="../../references.md#ref-kwiatkowski2019natural" aria-label="参考文献 194">[194]</a>、TriviaQA <a class="book-citation" id="cite-joshi2017triviaqa-1" href="../../references.md#ref-joshi2017triviaqa" aria-label="参考文献 173">[173]</a>、HotpotQA <a class="book-citation" id="cite-yang2018hotpotqa-1" href="../../references.md#ref-yang2018hotpotqa" aria-label="参考文献 405">[405]</a>）上，使用 7B 模型的 Search-R1 优于：</p>

<ul>
<li><p>标准 RAG（单次检索）准确率高出 15–20%</p></li>
<li><p>基于 Prompt 的智能体 RAG（ReAct 风格）准确率高出 8–12%</p></li>
<li><p>接近搭配标准 RAG 的大得多模型（70B）的性能</p></li>
</ul>

<p>关键洞见是：<strong>学习“何时以及如何搜索”比拥有一个知道更多内容的大模型更有价值</strong>。一个善于搜索的小模型可以击败不搜索的大模型。</p>

<div class="admonition tip"><p class="admonition-title">Search-R1：范式转变</p>
<p>传统 RAG 问的是：“对于这个查询，我应该检索什么？”（一种在生成之前做出的流水线决策）。</p>

<p>Search-R1 问的是：“基于我已经推理出的内容，我是否需要更多信息？如果需要，什么具体问题能填补这个空缺？”（一种在生成<em>过程中</em>做出的、可学习的决策）。</p>

<p>这就像考试前先翻教科书的学生，与在做题中途意识到卡住、随即去查阅参考资料的学生之间的差别。后者更高效、更有针对性。</p>
</div>
