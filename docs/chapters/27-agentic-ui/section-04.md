# 27.4　框架与库 {#subsec:ui-frameworks}

<p>日益增长的框架生态加速了 Agent 化 UI 的开发。我们按主要语言与使用场景，考察其中被最广泛采用的框架。</p>

## 27.4.1　Vercel AI SDK {#vercel-ai-sdk}

<p>Vercel AI SDK <a class="book-citation" id="cite-vercel2024aisdk-1" href="../../references.md#ref-vercel2024aisdk" aria-label="参考文献 358">[358]</a> 是一个 TypeScript/JavaScript 库，用于在 React、Next.js、Svelte 与 Vue 中构建流式 AI 界面。它是当前生产级 Web Agent UI 中应用最广泛的框架。</p>

<p><strong>核心抽象：</strong></p>

<ul>
<li><p><code>useChat</code>：管理聊天对话的 React Hook，支持流式响应、消息历史与加载状态。</p></li>
<li><p><code>useCompletion</code>：用于带流式的单轮文本补全的 Hook。</p></li>
<li><p><code>useObject</code>：流式传输结构化 JSON 对象，使复杂输出能够渐进式渲染。</p></li>
<li><p><code>streamText</code> / <code>streamObject</code>：通过 HTTP 流式传输 LLM 响应的服务端函数。</p></li>
</ul>

<p><strong>生成式 UI（AI SDK RSC）：</strong>Vercel AI SDK 最具特色的能力，是通过 React Server Components（RSC）支持<em>生成式 UI</em>（generative UI）。LLM 不再仅返回文本，而是可以调用工具，其结果被渲染为任意 React 组件——天气小部件、股票图表、预订表单——并直接流式注入 UI。进一步讨论参见 <a href="section-05.md#subsec:generative-ui">27.5</a> 节。</p>

## 27.4.2　Chainlit {#chainlit}

<p>Chainlit <a class="book-citation" id="cite-chainlit2024-1" href="../../references.md#ref-chainlit2024" aria-label="参考文献 40">[40]</a> 是一个 Python 框架，以极少的样板代码构建生产可用的 Agent UI。它在 LangChain 与 LlamaIndex 生态中尤其流行。</p>

<p><strong>关键特性：</strong></p>

<ul>
<li><p><strong>步骤可视化</strong>：Chainlit 原生地将 LangChain 与 LlamaIndex 的执行步骤渲染为可折叠的树，显示每次链调用、检索与工具调用。</p></li>
<li><p><strong>多模态支持</strong>：开箱支持文件上传、图像展示、音频播放与 PDF 渲染。</p></li>
<li><p><strong>认证与会话</strong>：内置用户认证、持久化对话历史与多用户支持。</p></li>
<li><p><strong>自定义元素</strong>：可从 Python 注册并渲染 React 组件，实现丰富的自定义可视化。</p></li>
<li><p><strong>反馈采集</strong>：内置点赞/踩反馈，附带可选评论并存入数据库。</p></li>
</ul>

<p class="code-caption"><span class="code-number">代码 27.1　</span>最小化的 Chainlit Agent 及其步骤可视化</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">chainlit</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">cl</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">ChatOpenAI</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_core.tools</span><span class="w"> </span><span class="kn">import</span> <span class="n">tool</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langgraph.prebuilt</span><span class="w"> </span><span class="kn">import</span> <span class="n">create_react_agent</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">search</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="c1"># Search for information.</span>
<span class="w">    </span><span class="sd">"""Search for information."""</span>
    <span class="k">return</span> <span class="sa">f</span><span class="s2">"Results for: </span><span class="si">{</span><span class="n">query</span><span class="si">}</span><span class="s2">"</span>

<span class="n">agent</span> <span class="o">=</span> <span class="n">create_react_agent</span><span class="p">(</span>
    <span class="n">ChatOpenAI</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">),</span> <span class="n">tools</span><span class="o">=</span><span class="p">[</span><span class="n">search</span><span class="p">]</span>
<span class="p">)</span>

<span class="nd">@cl</span><span class="o">.</span><span class="n">on_message</span>
<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">on_message</span><span class="p">(</span><span class="n">message</span><span class="p">:</span> <span class="n">cl</span><span class="o">.</span><span class="n">Message</span><span class="p">):</span>
    <span class="c1"># Chainlit 在使用回调处理器时，会自动将每一步渲染为可折叠的 UI 元素</span>
    <span class="c1"># Chainlit automatically renders each step as a collapsible UI element</span>
    <span class="c1"># when using the callback handler</span>
    <span class="k">async</span> <span class="k">with</span> <span class="n">cl</span><span class="o">.</span><span class="n">Step</span><span class="p">(</span><span class="n">name</span><span class="o">=</span><span class="s2">"Agent"</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="s2">"run"</span><span class="p">)</span> <span class="k">as</span> <span class="n">step</span><span class="p">:</span>
        <span class="n">step</span><span class="o">.</span><span class="n">input</span> <span class="o">=</span> <span class="n">message</span><span class="o">.</span><span class="n">content</span>
        <span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="n">agent</span><span class="o">.</span><span class="n">ainvoke</span><span class="p">(</span>
            <span class="p">{</span><span class="s2">"messages"</span><span class="p">:</span> <span class="p">[{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">message</span><span class="o">.</span><span class="n">content</span><span class="p">}]},</span>
            <span class="n">config</span><span class="o">=</span><span class="p">{</span><span class="s2">"callbacks"</span><span class="p">:</span> <span class="p">[</span><span class="n">cl</span><span class="o">.</span><span class="n">LangchainCallbackHandler</span><span class="p">()]}</span>
        <span class="p">)</span>
        <span class="n">output</span> <span class="o">=</span> <span class="n">result</span><span class="p">[</span><span class="s2">"messages"</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">content</span>
        <span class="n">step</span><span class="o">.</span><span class="n">output</span> <span class="o">=</span> <span class="n">output</span>

    <span class="k">await</span> <span class="n">cl</span><span class="o">.</span><span class="n">Message</span><span class="p">(</span><span class="n">content</span><span class="o">=</span><span class="n">output</span><span class="p">)</span><span class="o">.</span><span class="n">send</span><span class="p">()</span>
</code></pre></div>

## 27.4.3　Gradio {#gradio}

<p>Gradio <a class="book-citation" id="cite-abid2019gradio-1" href="../../references.md#ref-abid2019gradio" aria-label="参考文献 1">[1]</a> 是一个 Python 库，用于快速搭建 ML demo 与 Agent 界面。其 <code>gr.ChatInterface</code> 与 <code>gr.Blocks</code> API 使得用极少代码即可快速原型化对话式 Agent。</p>

<p><strong>在 Agent 化 UI 方面的优势：</strong></p>

<ul>
<li><p><strong>零配置部署</strong>：通过 Hugging Face Spaces 一行代码完成分享。</p></li>
<li><p><strong>自定义组件</strong>：Gradio 自定义组件系统允许构建能与 Python Backend 无缝集成的 React 组件。</p></li>
<li><p><strong>多模态输入</strong>：以极少配置支持文件上传、图像、音频、视频及网络摄像头输入。</p></li>
<li><p><strong>流式</strong>：原生支持基于生成器（generator）的流式响应。</p></li>
</ul>

<p><strong>局限：</strong>Gradio 的布局系统不如完整 React 框架灵活，其状态管理以会话为作用域，使复杂的多 Agent 协调具有挑战性。</p>

## 27.4.4　Streamlit {#streamlit}

<p>Streamlit <a class="book-citation" id="cite-streamlit2024-1" href="../../references.md#ref-streamlit2024" aria-label="参考文献 156">[156]</a> 是一个用于数据应用的 Python 框架，被广泛用于 Agent 仪表盘与监控 UI。其响应式执行模型——每次交互整个脚本重跑——简单直观，但对复杂的 Agent 化工作流可能受限。</p>

<p><strong>Agent 化用例：</strong></p>

<ul>
<li><p><strong>Agent 仪表盘</strong>：使用 <code>st.metric</code>、<code>st.dataframe</code> 与 <code>st.status</code> 提供实时指标、任务队列与状态展示。</p></li>
<li><p><strong>会话状态</strong>：<code>st.session_state</code> 在重跑之间持久化 Agent 状态，从而支持多轮对话。</p></li>
<li><p><strong>流式</strong>：<code>st.write_stream</code> 渐进式渲染生成器输出。</p></li>
<li><p><strong>Fragments</strong>：<code>@st.fragment</code> 装饰器允许局部重跑，提升实时更新仪表盘的性能。</p></li>
</ul>

## 27.4.5　OpenAI Assistants Playground {#openai-assistants-playground}

<p>OpenAI Assistants Playground 可视为 Agent 化 UI 设计的参考实现。它演示了：</p>

<ul>
<li><p>基于线程（thread）、带持久化历史的对话管理。</p></li>
<li><p>文件附件与检索过程的可视化。</p></li>
<li><p>代码解释器执行及其输出展示（stdout、图像、文件）。</p></li>
<li><p>函数调用展示，附带输入/输出检查。</p></li>
<li><p>运行步骤可视化，展示模型调用与工具调用的序列。</p></li>
</ul>

<p>尽管 Playground 并非用于构建自定义 UI 的框架，但其设计模式被广泛仿效。</p>

## 27.4.6　LangGraph Studio {#langgraph-studio}

<p>LangGraph Studio <a class="book-citation" id="cite-langgraph2024studio-1" href="../../references.md#ref-langgraph2024studio" aria-label="参考文献 155">[155]</a> 是一个桌面应用，为 LangGraph Agent 提供可视化 IDE。它是目前最成熟的工具使用与工作流可视化环境。</p>

<p><strong>特性：</strong></p>

<ul>
<li><p><strong>图可视化</strong>：交互式渲染 Agent 的状态机，节点代表 Agent 步骤，边代表状态转移。</p></li>
<li><p><strong>状态检查</strong>：在执行的任意点都可以以结构化 JSON 形式检视完整的 Agent 状态（全部变量、记忆、工具结果）。</p></li>
<li><p><strong>时间旅行调试</strong>：回放任意先前执行步骤、修改状态，并从该点重新运行。</p></li>
<li><p><strong>人机协同集成</strong>：可在任意节点设置断点；执行会暂停并等待人类输入后再继续。</p></li>
<li><p><strong>多 Agent 支持</strong>：可视化 supervisor 与子 Agent 的层级以及 Agent 间消息传递。</p></li>
</ul>

## 27.4.7　框架对比 {#framework-comparison}

<p>表 <a href="#tab:ui-framework-comparison">27.1</a> 总结了上文所讨论框架的关键特征。</p>

<p class="table-caption" id="tab:ui-framework-comparison"><span class="table-number">表 27.1　</span>Agent 化 UI 框架对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>框架</strong></th><th scope="col"><strong>语言</strong></th><th scope="col"><strong>流式</strong></th><th scope="col"><strong>工具可视化</strong></th><th scope="col"><strong>多 Agent</strong></th><th scope="col"><strong>生成式 UI</strong></th><th scope="col"><strong>生产可用</strong></th></tr></thead>
<tbody><tr><td>Vercel AI SDK</td><td>TypeScript</td><td>✓</td><td>部分</td><td>部分</td><td>✓</td><td>✓</td></tr>
<tr><td>Chainlit</td><td>Python</td><td>✓</td><td>✓</td><td>部分</td><td>部分</td><td>✓</td></tr>
<tr><td>Gradio</td><td>Python</td><td>✓</td><td><span class="arithmatex">\(\circ\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\circ\)</span></td><td>✓</td></tr>
<tr><td>Streamlit</td><td>Python</td><td>✓</td><td><span class="arithmatex">\(\circ\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr>
<tr><td>OAI Playground</td><td>N/A（托管）</td><td>✓</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td></tr>
<tr><td>LangGraph Studio</td><td>Python/TS</td><td>✓</td><td>✓</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>部分</td></tr></tbody>
</table>
</div>
