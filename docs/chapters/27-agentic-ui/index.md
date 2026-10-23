# 第 27 章 · Agentic UI 框架 {#sec:agentic-ui}

<p>随着大语言模型从被动的文本生成器演进为能够规划、调用工具与进行多步推理的主动 Agent，人类与之交互的界面也必须同步演化。传统聊天界面——为单轮或短上下文对话而设计——无法满足 Agent 化工作流的要求：长时间运行的任务、分支决策树、并行的工具调用，以及对有效人类监督的需求。本节考察 <em>Agent 化 UI 框架</em>（agentic UI frameworks）的图景：使丰富、透明、可信赖的人机协作成为可能的设计范式、组件库与实现模式。</p>

## 27.1　动机：超越聊天框 {#subsec:ui-motivation}

<div class="admonition tip"><p class="admonition-title">为什么 Agent 需要专门的界面</p>
<p>聊天气泡传达的是<em>结果</em>。Agent 化 UI 传达的则是<em>过程</em>——推理、调用的工具、做出的决策，以及需要人类判断介入的节点。没有这种可见性，用户无法信任、纠正 Agent，也无法从中学习。</p>
</div>

<p>聊天界面与 Agent 化界面之间的差距，犹如自动售货机与熟练协作者之间的差距。当一个 Agent 执行 20 步研究任务、浏览网页、撰写并运行代码、综合成一份报告时，用户需要解答的问题是简单的文本响应所无法提供的：</p>

<ul>
<li><p><strong>Agent 此刻在做什么？</strong> 长时间运行的任务需要进度反馈；沉默会滋生不信任。</p></li>
<li><p><strong>Agent 为何做出这个决策？</strong> 推理过程的透明性使用户能够尽早发现错误。</p></li>
<li><p><strong>使用了哪些工具，输入了什么？</strong> 工具来源（tool provenance）对核实事实性论断与审计行为至关重要。</p></li>
<li><p><strong>我应该在哪里介入？</strong> Agent 必须呈现那些需要人类判断的决策点，同时避免用每一项微决策淹没用户。</p></li>
<li><p><strong>我能撤销吗？</strong> 不可逆操作（发送邮件、修改文件、执行代码）需要明确的确认与回滚路径。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">自动化偏差风险</p>
<p>人机自动化交互的研究一致表明，用户会过度信任自动化系统，尤其是当这些系统自信地呈现输出而不附带不确定性信号时 <a class="book-citation" id="cite-parasuraman1997humans-1" href="../../references.md#ref-parasuraman1997humans" aria-label="参考文献 284">[284]</a>。Agent 化 UI 必须通过呈现不确定性、展示推理过程，以及让质疑或覆盖 Agent 决策变得容易，来主动对抗自动化偏差（automation bias）。</p>
</div>

<p>因此，Agent 化 UI 的设计位于人机交互（Human-Computer Interaction, HCI）、可解释 AI（Explainable AI, XAI）与软件工程的交汇处。核心设计目标包括：</p>

<ol>
<li><p><strong>透明性</strong>：让 Agent 的内部状态对用户清晰可读。</p></li>
<li><p><strong>可控性</strong>：提供有意义的介入节点，而不要求持续监督。</p></li>
<li><p><strong>信任校准</strong>：帮助用户建立关于 Agent 能力与局限的准确心智模型。</p></li>
<li><p><strong>效率</strong>：最小化认知负担；在合适的时间呈现合适的信息。</p></li>
<li><p><strong>可恢复性</strong>：让错误的发现与撤销成本低廉。</p></li>
</ol>

## 27.2　Agent 的 UI 范式 {#subsec:ui-paradigms}

<p>没有任何单一的 UI 范式能适应所有 Agent 化用例。合适的界面取决于任务时长、所需的人类介入程度、输出类型以及用户的专业水平。其谱系从完全对话式的聊天界面，跨越到几乎无需人类交互的完全自主仪表盘。</p>

### 27.2.1　基于聊天的界面 {#chat-based-interfaces}

<p>聊天范式——消息气泡、文本输入框和滚动历史——仍然是与 LLM 交互最为熟悉的入口。其优势在于学习曲线平缓与自然语言的灵活性。在 Agent 化场景下，聊天界面通常会扩展以下能力：</p>

<ul>
<li><p><strong>流式响应</strong>：Token 在生成时即逐步出现，提供即时反馈并降低感知延迟。通过服务器推送事件（Server-Sent Events, SSE）或 WebSockets 实现。</p></li>
<li><p><strong>内联工具指示器</strong>：消息流中的小徽章或可展开区段，用于显示何时调用了工具（例如 “<code>[Searched the web for: climate change 2024]</code>”）。</p></li>
<li><p><strong>输入指示与状态消息</strong>：“Agent is thinking…”、“Running Python code…”、“Fetching results…” 等提示在延迟间隙保持用户的知情感。</p></li>
<li><p><strong>消息线程化</strong>：对于多轮 Agent 任务，可折叠的子线程能容纳中间步骤，而不会让主对话变得杂乱。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">聊天 UI 对 Agent 的局限</p>
<p>聊天界面将本质上并行的过程串行化呈现。当一个 Agent 同时分发任务到五个工具时，线性的消息流会错误地表达实际的执行图。对于复杂的 Agent 化工作流，聊天应被更丰富的范式所扩展——甚至取代。</p>
</div>

### 27.2.2　画布与产物式界面 {#canvas-and-artifact-based-interfaces}

<p>画布范式（canvas paradigm）由 Claude Artifacts1 与 ChatGPT Canvas2 推广，引入了<em>分屏</em>布局：左侧承载对话，右侧（“canvas” 或 “artifact panel”）将生成的内容——代码、文档、图表、电子表格——以可实时编辑的产物（artifact）形式呈现。</p>

<p>关键特征：</p>

<ul>
<li><p><strong>持久化产物</strong>：生成的内容跨轮次持久存在，可以通过自然语言指令迭代优化（“把图表改成蓝色”、“给这个函数加上错误处理”）。</p></li>
<li><p><strong>原位编辑</strong>：用户可以直接编辑产物，Agent 也能观察并响应这些编辑。</p></li>
<li><p><strong>版本历史</strong>：产物维护修订历史，支持回滚到任意先前状态。</p></li>
<li><p><strong>多产物工作区</strong>：进阶实现支持多个并存的产物（例如一个代码文件、其测试套件以及一个文档页面）。</p></li>
</ul>

<p>画布范式特别适合<em>共创</em>（co-creation）任务：写作、编程、数据分析与设计——这些场景的产出是文档或产物，而非对话式的回答。</p>

### 27.2.3　工作流可视化 {#workflow-visualization}

<p>对于执行结构化计划（步骤序列或步骤图）的 Agent，工作流可视化 UI 使计划显式化并可追踪。该范式常见于：</p>

<ul>
<li><p><strong>Agent 化流水线</strong>（LangGraph、AutoGen、CrewAI）：将 Agent 的执行图渲染为有向无环图（Directed Acyclic Graph, DAG）或流程图，节点代表步骤，边代表数据流或控制流。</p></li>
<li><p><strong>任务分解视图</strong>：将 Agent 的高层计划呈现为清单或甘特图式的时间线，每个子任务可展开以显示其自身的步骤。</p></li>
<li><p><strong>实时进度追踪</strong>：节点在执行时改变颜色或显示加载图标；完成的节点展示输出；失败的节点展示错误细节。</p></li>
</ul>

<p>LangGraph Studio3 是该范式的典范，它为 LangGraph Agent 提供基于图的调试器与可视化工具。用户可以检查每个节点的状态、回放执行过程，并注入修改后的状态以测试备选路径。</p>

### 27.2.4　仪表盘与监控界面 {#dashboard-and-monitoring-interfaces}

<p>对于长时间运行或生产环境中的 Agent，仪表盘 UI 提供运维视角：</p>

<ul>
<li><p><strong>实时状态</strong>：哪些 Agent 正在运行、空闲或失败；当前任务与步骤。</p></li>
<li><p><strong>资源指标</strong>：Token 消耗、API 调用计数、延迟直方图、成本估算。</p></li>
<li><p><strong>队列管理</strong>：待处理任务、优先级排序、限流状态。</p></li>
<li><p><strong>告警与异常检测</strong>：异常行为（过多的重试、成本激增、反复失败）以通知形式呈现。</p></li>
<li><p><strong>历史分析</strong>：任务完成率、平均时长、错误频率随时间的变化。</p></li>
</ul>

<p>仪表盘 UI 通常基于 Grafana4、自定义 React 仪表盘或 Streamlit 等工具构建，面向<em>运维人员</em>而非最终用户。</p>

### 27.2.5　协作型界面 {#collaborative-interfaces}

<p>协作型 UI 将 Agent 视为共享工作区（文档、代码库或设计画布）中与人类协作者并列的同侪贡献者。关键特性包括：</p>

<ul>
<li><p><strong>在线指示</strong>：Agent 以带名字的光标或头像出现在共享工作区中。</p></li>
<li><p><strong>变更归属</strong>：Agent 所做的编辑在视觉上与人类编辑相区分（例如用颜色编码的 diff）。</p></li>
<li><p><strong>内联建议</strong>：Agent 以修订标记或评论的形式提出变更，人类可以接受、拒绝或修改。</p></li>
<li><p><strong>冲突解决</strong>：当 Agent 与人类同时编辑同一区域时，UI 会呈现冲突并辅助解决。</p></li>
</ul>

<p>该范式正在 Cursor5（带 AI 的协作式代码编辑）、Notion AI6 以及集成 Gemini 的 Google Docs7 等工具中兴起。</p>

### 27.2.6　带检查点的自主模式 {#autonomous-with-checkpoints}

<p>在自主性谱系的另一端，一些 Agent 几乎完全独立运行——浏览网页、编写代码、执行命令——仅在预定义的、需要人类批准的<em>检查点</em>（checkpoint）处现身。该范式被用于：</p>

<ul>
<li><p><strong>计算机使用 Agent</strong>（Anthropic Computer Use8、OpenAI Operator9）：Agent 控制浏览器或桌面；UI 显示实时屏幕画面，并在不可逆操作前暂停等待批准。</p></li>
<li><p><strong>带门控的自动化流水线</strong>：CI/CD 风格的工作流——Agent 完成一个阶段后等待人类“合并”方可继续。</p></li>
<li><p><strong>定时 Agent</strong>：按计划运行并异步报告结果的 Agent，通过基于通知的 UI 来审阅输出并批准后续操作。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">实践中的检查点 UI</p>
<p>被指派“清理我的邮件收件箱”的 Agent 可能会自主分类并归档 500 封邮件，然后暂停并给出摘要：“我发现 23 封邮件来自您 6 个月未打开的邮件列表。是否需要全部、部分或都不退订？”用户审阅列表、做出选择，Agent 继续执行。这种模式——以人类决策点穿插的自主执行——在效率与可控之间取得平衡。</p>
</div>

## 27.3　Agent 的关键 UI 组件 {#subsec:ui-components}

<p>无论采取何种总体范式，Agent 化 UI 都共享一组反复出现的组件。本节梳理其中最重要的部分，并给出各自的设计指导。</p>

### 27.3.1　思考与推理展示 {#thought-and-reasoning-display}

<p>现代 LLM——尤其是采用思维链（Chain-of-Thought，CoT）或延展思考训练的模型（例如 OpenAI o1/o3、带扩展思考的 Anthropic Claude）——在产出最终响应之前会生成大量内部推理。呈现这些推理是一把双刃剑：它提升了透明度，但冗长的内部独白也可能淹没用户。</p>

<p>最佳实践：</p>

<ul>
<li><p><strong>可折叠的推理区块</strong>：显示摘要（“思考了 12 秒”），并为想看细节的用户提供展开按钮。</p></li>
<li><p><strong>渐进式披露</strong>：默认仅展示最终结论；推理可按需查看。</p></li>
<li><p><strong>结构化推理</strong>：如果模型产出结构化思考（假设、证据、结论），应以视觉层次来呈现，而非堆成大段文字墙。</p></li>
<li><p><strong>推理与响应的区分</strong>：在视觉上清晰区分内部推理（可能包含错误或失败尝试）与最终响应。</p></li>
</ul>

### 27.3.2　工具使用可视化 {#tool-use-visualization}

<p>工具调用是 Agent 与外部世界交互的主要机制。将其可视化对建立信任与进行调试至关重要。</p>

<div class="admonition note"><p class="admonition-title">工具调用的解剖</p>
<p>每次工具调用都有四个值得展示的组成部分：(1) <strong>工具名</strong>与图标；(2) <strong>输入参数</strong>（可能是较大的 JSON）；(3) <strong>输出/结果</strong>（可能很大）；(4) <strong>耗时</strong>（延迟）。UI 必须在完整性与可读性之间取得平衡。</p>
</div>

<p>工具可视化的设计模式：</p>

<ul>
<li><p><strong>内联工具卡片</strong>：消息流中的紧凑卡片，显示工具名、一行输入摘要与状态（运行中/成功/错误）。可展开查看完整细节。</p></li>
<li><p><strong>工具时间线</strong>：横向时间线显示一轮中的全部工具调用及其耗时，便于识别瓶颈。</p></li>
<li><p><strong>输入/输出 diff</strong>：对于修改状态的工具（例如文件编辑），展示前后对比 diff。</p></li>
<li><p><strong>工具图标与品牌</strong>：常用工具（网页搜索、代码执行、文件系统、API）使用可辨识的图标，便于快速扫视。</p></li>
<li><p><strong>错误高亮</strong>：失败的工具调用以红色显示，附带错误消息与任何重试尝试。</p></li>
</ul>

### 27.3.3　进度指示器 {#progress-indicators}

<p>多步 Agent 任务需要丰富的进度反馈：</p>

<ul>
<li><p><strong>步骤级进度</strong>：编号的计划步骤列表，每完成一项即打钩。对于动态计划，步骤可随 Agent 调整而增删。</p></li>
<li><p><strong>Token 流指示器</strong>：生成期间的闪烁光标或动画省略号；为高级用户提供每秒 Token 数计数器。</p></li>
<li><p><strong>预计完成</strong>：在可行时，依据任务复杂度与历史表现给出预计到达时间（ETA），并以合适的不确定性表达（“大约 2–5 分钟”）。</p></li>
<li><p><strong>子任务嵌套</strong>：对于层级化任务，提供带可展开子任务的树状进度视图。</p></li>
<li><p><strong>取消</strong>：清晰可见的“停止”按钮，能优雅地终止 Agent 并总结迄今为止已完成的工作。</p></li>
</ul>

### 27.3.4　审批门控 {#approval-gates}

<p>审批门控（approval gates）是人机协同（Human-in-the-Loop）控制的主要机制。它必须设计得既<em>富含信息</em>（给用户足够的上下文以做出良好决策），又不会令人<em>疲劳</em>（无需对每个琐碎操作都请求批准）。</p>

<div class="admonition warning"><p class="admonition-title">审批门控中的告警疲劳</p>
<p>如果 Agent 请求批准过于频繁，用户将不加阅读地反射式批准——使门控的目的落空。分层审批策略（参见 <a href="#subsec:hitl-design">27.7</a> 节）是维持有效监督的关键。</p>
</div>

<p>审批门控的 UI 元素：</p>

<ul>
<li><p><strong>操作摘要</strong>：用通俗语言描述 Agent 想要做什么（“向 john@example.com 发送附带报告的邮件”）。</p></li>
<li><p><strong>风险指示器</strong>：操作可逆性的视觉信号（绿色 = 易于撤销，黄色 = 难以撤销，红色 = 不可逆）。</p></li>
<li><p><strong>批准 / 拒绝 / 修改</strong>：三选项界面；“修改”会在批准前打开操作参数的编辑器。</p></li>
<li><p><strong>上下文面板</strong>：可展开区域，展示 Agent 想执行该操作的缘由（相关推理、先前步骤）。</p></li>
<li><p><strong>超时行为</strong>：清晰说明用户不响应时会发生什么（Agent 暂停而非继续）。</p></li>
</ul>

### 27.3.5　上下文展示 {#context-display}

<p>Agent 会维护影响其行为的内部状态——记忆、激活的工具、检索到的文档、对话历史。让这些状态可见有助于用户理解并预测 Agent 的行为。</p>

<ul>
<li><p><strong>记忆面板</strong>：显示 Agent 当前对用户、任务以及先前交互“记住”了什么。用户可编辑。</p></li>
<li><p><strong>激活工具列表</strong>：Agent 当前可用的工具，配有启用/禁用开关。</p></li>
<li><p><strong>检索到的上下文</strong>：当前位于 Agent 上下文窗口中的文档或数据块，附带来源引用。</p></li>
<li><p><strong>Token 预算指示器</strong>：上下文窗口已消耗多少，帮助用户判断何时开启新会话。</p></li>
</ul>

### 27.3.6　错误与恢复 UI {#error-and-recovery-ui}

<p>Agent 会失败——工具返回错误、模型出现幻觉、计划变得不可行。UI 必须优雅地处理失败：</p>

<ul>
<li><p><strong>错误卡片</strong>：内联显示失败信息，包括错误类型、消息以及 Agent 的解读。</p></li>
<li><p><strong>重试控件</strong>：手动重试按钮，可选地调整参数。</p></li>
<li><p><strong>备选方案</strong>：当主方案失败时，Agent 提出备选；UI 将其作为可选项呈现。</p></li>
<li><p><strong>部分结果</strong>：若多步任务中途失败，UI 显示已完成的步骤及其输出，保留部分价值。</p></li>
<li><p><strong>升级路径</strong>：当 Agent 无法继续时，提供通往人工支持或手动完成的清晰路径。</p></li>
</ul>

### 27.3.7　置信度指示器 {#confidence-indicators}

<p>LLM 是带有（已校准或未校准的）不确定性的概率系统。呈现置信度有助于用户判断何时该信任、何时该核实：</p>

<ul>
<li><p><strong>措辞犹豫的展示</strong>：高亮诸如“我不确定”或“你或许需要核实”之类的措辞，以引起对低置信度论断的注意。</p></li>
<li><p><strong>来源质量指示</strong>：对检索到的信息显示来源新近度、权威性与相关性评分。</p></li>
<li><p><strong>显式不确定性请求</strong>：“你有多自信？”按钮促使 Agent 自评并解释其不确定性。</p></li>
<li><p><strong>核实建议</strong>：对于高风险输出，Agent 主动建议核实步骤（“我建议独立核对这个计算”）。</p></li>
</ul>

## 27.4　框架与库 {#subsec:ui-frameworks}

<p>日益增长的框架生态加速了 Agent 化 UI 的开发。我们按主要语言与使用场景，考察其中被最广泛采用的框架。</p>

### 27.4.1　Vercel AI SDK {#vercel-ai-sdk}

<p>Vercel AI SDK <a class="book-citation" id="cite-vercel2024aisdk-1" href="../../references.md#ref-vercel2024aisdk" aria-label="参考文献 358">[358]</a> 是一个 TypeScript/JavaScript 库，用于在 React、Next.js、Svelte 与 Vue 中构建流式 AI 界面。它是当前生产级 Web Agent UI 中应用最广泛的框架。</p>

<p><strong>核心抽象：</strong></p>

<ul>
<li><p><code>useChat</code>：管理聊天对话的 React Hook，支持流式响应、消息历史与加载状态。</p></li>
<li><p><code>useCompletion</code>：用于带流式的单轮文本补全的 Hook。</p></li>
<li><p><code>useObject</code>：流式传输结构化 JSON 对象，使复杂输出能够渐进式渲染。</p></li>
<li><p><code>streamText</code> / <code>streamObject</code>：通过 HTTP 流式传输 LLM 响应的服务端函数。</p></li>
</ul>

<p><strong>生成式 UI（AI SDK RSC）：</strong>Vercel AI SDK 最具特色的能力，是通过 React Server Components（RSC）支持<em>生成式 UI</em>（generative UI）。LLM 不再仅返回文本，而是可以调用工具，其结果被渲染为任意 React 组件——天气小部件、股票图表、预订表单——并直接流式注入 UI。进一步讨论参见 <a href="#subsec:generative-ui">27.5</a> 节。</p>

### 27.4.2　Chainlit {#chainlit}

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

### 27.4.3　Gradio {#gradio}

<p>Gradio <a class="book-citation" id="cite-abid2019gradio-1" href="../../references.md#ref-abid2019gradio" aria-label="参考文献 1">[1]</a> 是一个 Python 库，用于快速搭建 ML demo 与 Agent 界面。其 <code>gr.ChatInterface</code> 与 <code>gr.Blocks</code> API 使得用极少代码即可快速原型化对话式 Agent。</p>

<p><strong>在 Agent 化 UI 方面的优势：</strong></p>

<ul>
<li><p><strong>零配置部署</strong>：通过 Hugging Face Spaces 一行代码完成分享。</p></li>
<li><p><strong>自定义组件</strong>：Gradio 自定义组件系统允许构建能与 Python Backend 无缝集成的 React 组件。</p></li>
<li><p><strong>多模态输入</strong>：以极少配置支持文件上传、图像、音频、视频及网络摄像头输入。</p></li>
<li><p><strong>流式</strong>：原生支持基于生成器（generator）的流式响应。</p></li>
</ul>

<p><strong>局限：</strong>Gradio 的布局系统不如完整 React 框架灵活，其状态管理以会话为作用域，使复杂的多 Agent 协调具有挑战性。</p>

### 27.4.4　Streamlit {#streamlit}

<p>Streamlit <a class="book-citation" id="cite-streamlit2024-1" href="../../references.md#ref-streamlit2024" aria-label="参考文献 156">[156]</a> 是一个用于数据应用的 Python 框架，被广泛用于 Agent 仪表盘与监控 UI。其响应式执行模型——每次交互整个脚本重跑——简单直观，但对复杂的 Agent 化工作流可能受限。</p>

<p><strong>Agent 化用例：</strong></p>

<ul>
<li><p><strong>Agent 仪表盘</strong>：使用 <code>st.metric</code>、<code>st.dataframe</code> 与 <code>st.status</code> 提供实时指标、任务队列与状态展示。</p></li>
<li><p><strong>会话状态</strong>：<code>st.session_state</code> 在重跑之间持久化 Agent 状态，从而支持多轮对话。</p></li>
<li><p><strong>流式</strong>：<code>st.write_stream</code> 渐进式渲染生成器输出。</p></li>
<li><p><strong>Fragments</strong>：<code>@st.fragment</code> 装饰器允许局部重跑，提升实时更新仪表盘的性能。</p></li>
</ul>

### 27.4.5　OpenAI Assistants Playground {#openai-assistants-playground}

<p>OpenAI Assistants Playground 可视为 Agent 化 UI 设计的参考实现。它演示了：</p>

<ul>
<li><p>基于线程（thread）、带持久化历史的对话管理。</p></li>
<li><p>文件附件与检索过程的可视化。</p></li>
<li><p>代码解释器执行及其输出展示（stdout、图像、文件）。</p></li>
<li><p>函数调用展示，附带输入/输出检查。</p></li>
<li><p>运行步骤可视化，展示模型调用与工具调用的序列。</p></li>
</ul>

<p>尽管 Playground 并非用于构建自定义 UI 的框架，但其设计模式被广泛仿效。</p>

### 27.4.6　LangGraph Studio {#langgraph-studio}

<p>LangGraph Studio <a class="book-citation" id="cite-langgraph2024studio-1" href="../../references.md#ref-langgraph2024studio" aria-label="参考文献 155">[155]</a> 是一个桌面应用，为 LangGraph Agent 提供可视化 IDE。它是目前最成熟的工具使用与工作流可视化环境。</p>

<p><strong>特性：</strong></p>

<ul>
<li><p><strong>图可视化</strong>：交互式渲染 Agent 的状态机，节点代表 Agent 步骤，边代表状态转移。</p></li>
<li><p><strong>状态检查</strong>：在执行的任意点都可以以结构化 JSON 形式检视完整的 Agent 状态（全部变量、记忆、工具结果）。</p></li>
<li><p><strong>时间旅行调试</strong>：回放任意先前执行步骤、修改状态，并从该点重新运行。</p></li>
<li><p><strong>人机协同集成</strong>：可在任意节点设置断点；执行会暂停并等待人类输入后再继续。</p></li>
<li><p><strong>多 Agent 支持</strong>：可视化 supervisor 与子 Agent 的层级以及 Agent 间消息传递。</p></li>
</ul>

### 27.4.7　框架对比 {#framework-comparison}

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

## 27.5　生成式 UI {#subsec:generative-ui}

<div class="admonition tip"><p class="admonition-title">生成式 UI 概念</p>
<p>传统的 LLM 界面将模型输出渲染为文本或 markdown。<em>生成式 UI</em>（generative UI）颠覆了这一模式：模型的工具调用<em>生成</em> UI 组件。模型不仅决定<em>说什么</em>，还根据内容类型与用户意图决定<em>如何呈现</em>——以图表、表单、地图或日历小部件的形式。</p>
</div>

<p>生成式 UI 标志着 LLM 与界面关系的根本转变。开发者不再预先规定所有可能的 UI 状态，而是由模型动态选择并参数化适合当前上下文的 UI 组件。</p>

### 27.5.1　用 React Server Components 构建动态界面 {#react-server-components-for-dynamic-interfaces}

<p>Vercel AI SDK 的 RSC（React Server Components10）集成是生成式 UI 最成熟的实现。其架构如下：</p>

<ol>
<li><p>用户向 Next.js11 的 server action 发送消息。</p></li>
<li><p>服务端以一组工具调用 LLM，每个工具关联一个 React 组件。</p></li>
<li><p>当 LLM 调用某个工具（例如 <code>show_weather</code>）时，服务端以工具输出作为 props 渲染相应的 React 组件。</p></li>
<li><p>渲染后的组件作为 React Server Component 流式发送给客户端，在聊天中内联出现。</p></li>
</ol>

<p class="code-caption"><span class="code-number">代码 27.2　</span>基于 Vercel AI SDK RSC 的生成式 UI（TypeScript）</p>
<div class="highlight"><pre><span></span><code><span class="c1">// app/actions.tsx (Server Action)</span>
<span class="k">import</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">streamUI</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="kr">from</span><span class="w"> </span><span class="s1">'ai/rsc'</span><span class="p">;</span>
<span class="k">import</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">openai</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="kr">from</span><span class="w"> </span><span class="s1">'@ai-sdk/openai'</span><span class="p">;</span>
<span class="k">import</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">WeatherCard</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="kr">from</span><span class="w"> </span><span class="s1">'@/components/WeatherCard'</span><span class="p">;</span>
<span class="k">import</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">StockChart</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="kr">from</span><span class="w"> </span><span class="s1">'@/components/StockChart'</span><span class="p">;</span>

<span class="k">export</span><span class="w"> </span><span class="k">async</span><span class="w"> </span><span class="kd">function</span><span class="w"> </span><span class="nx">chat</span><span class="p">(</span><span class="nx">userMessage</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="nx">result</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="k">await</span><span class="w"> </span><span class="nx">streamUI</span><span class="p">({</span>
<span class="w">    </span><span class="nx">model</span><span class="o">:</span><span class="w"> </span><span class="kt">openai</span><span class="p">(</span><span class="s1">'gpt-4o'</span><span class="p">),</span>
<span class="w">    </span><span class="nx">messages</span><span class="o">:</span><span class="w"> </span><span class="p">[{</span><span class="w"> </span><span class="nx">role</span><span class="o">:</span><span class="w"> </span><span class="s1">'user'</span><span class="p">,</span><span class="w"> </span><span class="nx">content</span><span class="o">:</span><span class="w"> </span><span class="kt">userMessage</span><span class="w"> </span><span class="p">}],</span>
<span class="w">    </span><span class="nx">tools</span><span class="o">:</span><span class="w"> </span><span class="p">{</span>
<span class="w">      </span><span class="nx">show_weather</span><span class="o">:</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="nx">description</span><span class="o">:</span><span class="w"> </span><span class="s1">'Display current weather for a location'</span><span class="p">,</span>
<span class="w">        </span><span class="nx">parameters</span><span class="o">:</span><span class="w"> </span><span class="kt">z.object</span><span class="p">({</span>
<span class="w">          </span><span class="nx">location</span><span class="o">:</span><span class="w"> </span><span class="kt">z.string</span><span class="p">(),</span>
<span class="w">          </span><span class="nx">unit</span><span class="o">:</span><span class="w"> </span><span class="kt">z.enum</span><span class="p">([</span><span class="s1">'celsius'</span><span class="p">,</span><span class="w"> </span><span class="s1">'fahrenheit'</span><span class="p">]),</span>
<span class="w">        </span><span class="p">}),</span>
<span class="w">        </span><span class="c1">// 工具结果作为 React 组件渲染</span>
<span class="w">        </span><span class="c1">// Tool result rendered as a React component</span>
<span class="w">        </span><span class="nx">generate</span><span class="o">:</span><span class="w"> </span><span class="kt">async</span><span class="w"> </span><span class="p">({</span><span class="w"> </span><span class="nx">location</span><span class="p">,</span><span class="w"> </span><span class="nx">unit</span><span class="w"> </span><span class="p">})</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">          </span><span class="kd">const</span><span class="w"> </span><span class="nx">data</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="k">await</span><span class="w"> </span><span class="nx">fetchWeather</span><span class="p">(</span><span class="nx">location</span><span class="p">,</span><span class="w"> </span><span class="nx">unit</span><span class="p">);</span>
<span class="w">          </span><span class="k">return</span><span class="w"> </span><span class="p">&lt;</span><span class="nt">WeatherCard</span><span class="w"> </span><span class="na">data</span><span class="o">=</span><span class="p">{</span><span class="nx">data</span><span class="p">}</span><span class="w"> </span><span class="p">/&gt;;</span>
<span class="w">        </span><span class="p">},</span>
<span class="w">      </span><span class="p">},</span>
<span class="w">      </span><span class="nx">show_stock</span><span class="o">:</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="nx">description</span><span class="o">:</span><span class="w"> </span><span class="s1">'Display stock price chart'</span><span class="p">,</span>
<span class="w">        </span><span class="nx">parameters</span><span class="o">:</span><span class="w"> </span><span class="kt">z.object</span><span class="p">({</span><span class="w"> </span><span class="nx">ticker</span><span class="o">:</span><span class="w"> </span><span class="kt">z.string</span><span class="p">()</span><span class="w"> </span><span class="p">}),</span>
<span class="w">        </span><span class="nx">generate</span><span class="o">:</span><span class="w"> </span><span class="kt">async</span><span class="w"> </span><span class="p">({</span><span class="w"> </span><span class="nx">ticker</span><span class="w"> </span><span class="p">})</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">          </span><span class="kd">const</span><span class="w"> </span><span class="nx">data</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="k">await</span><span class="w"> </span><span class="nx">fetchStockData</span><span class="p">(</span><span class="nx">ticker</span><span class="p">);</span>
<span class="w">          </span><span class="k">return</span><span class="w"> </span><span class="p">&lt;</span><span class="nt">StockChart</span><span class="w"> </span><span class="na">ticker</span><span class="o">=</span><span class="p">{</span><span class="nx">ticker</span><span class="p">}</span><span class="w"> </span><span class="na">data</span><span class="o">=</span><span class="p">{</span><span class="nx">data</span><span class="p">}</span><span class="w"> </span><span class="p">/&gt;;</span>
<span class="w">        </span><span class="p">},</span>
<span class="w">      </span><span class="p">},</span>
<span class="w">    </span><span class="p">},</span>
<span class="w">  </span><span class="p">});</span>
<span class="w">  </span><span class="k">return</span><span class="w"> </span><span class="nx">result</span><span class="p">.</span><span class="nx">value</span><span class="p">;</span>
<span class="p">}</span>
</code></pre></div>

### 27.5.2　基于内容类型的自适应界面 {#adaptive-interfaces-based-on-content-type}

<p>生成式 UI 使界面能够根据所呈现内容的性质自适应：</p>

<ul>
<li><p><strong>表格数据</strong> <span class="arithmatex">\(\rightarrow\)</span> 可排序、可过滤、带导出选项的数据表。</p></li>
<li><p><strong>地理数据</strong> <span class="arithmatex">\(\rightarrow\)</span> 带标记与图层的交互式地图。</p></li>
<li><p><strong>时间序列</strong> <span class="arithmatex">\(\rightarrow\)</span> 可缩放、带注释的折线图。</p></li>
<li><p><strong>代码</strong> <span class="arithmatex">\(\rightarrow\)</span> 带语法高亮与运行按钮的编辑器。</p></li>
<li><p><strong>文档</strong> <span class="arithmatex">\(\rightarrow\)</span> 带注释工具的格式化文档查看器。</p></li>
<li><p><strong>表单/结构化输入</strong> <span class="arithmatex">\(\rightarrow\)</span> 动态生成的表单字段。</p></li>
</ul>

<p>模型在此扮演<em>UI 编排者</em>（UI orchestrator）的角色，为每条信息选择最合适的呈现方式。这降低了开发者预判所有可能输出类型并预先构建相应组件的需求。</p>

<div class="admonition question"><p class="admonition-title">生成式 UI 的边界</p>
<p>应该将多少 UI 生成委托给模型？完全由模型驱动的 UI 存在不一致性、无障碍访问失败与安全漏洞的风险（例如，模型生成一个把数据提交到非预期端点的表单）。实践中，当模型从一个由预构建、可访问且安全组件组成的<em>精选库</em>中进行选择，而非生成任意 HTML 或 JSX 时，生成式 UI 的效果最佳。</p>
</div>

## 27.6　流式与实时模式 {#subsec:streaming-patterns}

<p>流式（Stream）是 Agent 化 UI 的基础：它把体验从“等待一个结果”转变为“观察 Agent 在工作”。本节涵盖关键的流式模式及其实现考量。</p>

### 27.6.1　Token 流 {#token-streaming}

<p>Token 流在 Token 被生成时增量传输 LLM 输出，而不等待完整响应。常用的两种传输机制：</p>

<ul>
<li><p><strong>服务器推送事件（Server-Sent Events, SSE）</strong>12：从服务器到客户端的单向 HTTP 流。每个事件携带一块 Token。SSE 简单，运行于标准 HTTP/1.1 之上，并由浏览器自动重连。它是 LLM 流式 API 的主导机制（OpenAI、Anthropic、Google 都使用 SSE）。</p></li>
<li><p><strong>WebSockets</strong>：双向持久连接。实现更复杂，但在客户端需要在流中途发送数据的交互式流场景下是必要的（例如打断 Agent、在生成中途提供反馈）。</p></li>
</ul>

<p class="code-caption"><span class="code-number">代码 27.3　</span>基于 FastAPI 的 SSE Token 流</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">fastapi</span><span class="w"> </span><span class="kn">import</span> <span class="n">FastAPI</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">fastapi.responses</span><span class="w"> </span><span class="kn">import</span> <span class="n">StreamingResponse</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">AsyncOpenAI</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>

<span class="n">app</span> <span class="o">=</span> <span class="n">FastAPI</span><span class="p">()</span>
<span class="n">client</span> <span class="o">=</span> <span class="n">AsyncOpenAI</span><span class="p">()</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">token_stream</span><span class="p">(</span><span class="n">prompt</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
    <span class="c1"># 产出 SSE 格式 Token 块的生成器。</span>
<span class="w">    </span><span class="sd">"""Generator that yields SSE-formatted token chunks."""</span>
    <span class="n">stream</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">chat</span><span class="o">.</span><span class="n">completions</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
        <span class="n">model</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">,</span>
        <span class="n">messages</span><span class="o">=</span><span class="p">[{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">prompt</span><span class="p">}],</span>
        <span class="n">stream</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="k">async</span> <span class="k">for</span> <span class="n">chunk</span> <span class="ow">in</span> <span class="n">stream</span><span class="p">:</span>
        <span class="n">delta</span> <span class="o">=</span> <span class="n">chunk</span><span class="o">.</span><span class="n">choices</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">delta</span>
        <span class="k">if</span> <span class="n">delta</span><span class="o">.</span><span class="n">content</span><span class="p">:</span>
            <span class="c1"># SSE 格式："data: &lt;json&gt;\n\n"</span>
            <span class="c1"># SSE format: "data: &lt;json&gt;\n\n"</span>
            <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s1">'token'</span><span class="p">:</span><span class="w"> </span><span class="n">delta</span><span class="o">.</span><span class="n">content</span><span class="p">})</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>
        <span class="k">elif</span> <span class="n">chunk</span><span class="o">.</span><span class="n">choices</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">finish_reason</span><span class="p">:</span>
            <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s1">'done'</span><span class="p">:</span><span class="w"> </span><span class="kc">True</span><span class="p">})</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>

<span class="nd">@app</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"/stream"</span><span class="p">)</span>
<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">stream_endpoint</span><span class="p">(</span><span class="n">prompt</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">StreamingResponse</span><span class="p">(</span>
        <span class="n">token_stream</span><span class="p">(</span><span class="n">prompt</span><span class="p">),</span>
        <span class="n">media_type</span><span class="o">=</span><span class="s2">"text/event-stream"</span><span class="p">,</span>
        <span class="n">headers</span><span class="o">=</span><span class="p">{</span><span class="s2">"Cache-Control"</span><span class="p">:</span> <span class="s2">"no-cache"</span><span class="p">,</span> <span class="s2">"X-Accel-Buffering"</span><span class="p">:</span> <span class="s2">"no"</span><span class="p">},</span>
    <span class="p">)</span>
</code></pre></div>

### 27.6.2　工具调用流 {#tool-call-streaming}

<p>现代 LLM API 支持流式工具调用：工具名与参数被增量地流式传输，使 UI 能够在工具尚未真正被调用之前就显示“Agent 正在调用 <code>search_web</code>，查询：‘climate change 2024’…”。这需要解析不完整的 JSON，可通过流式 JSON 解析器实现。</p>

<p>工具调用流的模式：</p>

<ul>
<li><p><strong>渐进式参数展示</strong>：在参数流入时即展示，即便调用尚未完成。</p></li>
<li><p><strong>并行工具调用指示器</strong>：当模型同时调用多个工具时，将它们全部标记为待处理，并在结果到达时分别更新。</p></li>
<li><p><strong>工具结果流</strong>：有些工具（例如代码执行、网页抓取）本身就能流式输出结果；应将其渐进式管道传输至 UI。</p></li>
</ul>

### 27.6.3　多 Agent 流 {#multi-agent-streaming}

<p>在多 Agent 系统中，可能有多个 Agent 同时产出输出。UI 必须处理并行流：</p>

<ul>
<li><p><strong>带 Agent 标签的流</strong>：每条流都标记 Agent 身份；UI 将其渲染到不同的泳道或面板中。</p></li>
<li><p><strong>流合并</strong>：对于 supervisor-子 Agent 模式，supervisor 的流可能与子 Agent 流交错；UI 必须维护一致的顺序。</p></li>
<li><p><strong>背压（Backpressure）</strong>：如果 UI 渲染速度跟不上流的到达速度（例如多个 Agent 同时生成），背压机制必须防止缓冲区溢出。策略包括：丢弃中间 Token（仅展示最新）、批量更新或暂停较慢的流。</p></li>
</ul>

### 27.6.4　乐观 UI 更新 {#optimistic-ui-updates}

<p>乐观 UI 更新（optimistic UI updates）通过在服务器确认前立即将用户操作反映到 UI 上，提升感知响应性：</p>

<ul>
<li><p>当用户发送消息时，它会（乐观地）立即出现在聊天历史中，而请求仍在传输中。</p></li>
<li><p>当审批门控被接受时，UI 立即将该操作显示为“已批准”，并开始展示 Agent 的后续步骤，甚至先于服务器处理该批准。</p></li>
<li><p>如果服务器返回错误，乐观更新将被回滚并显示错误状态。</p></li>
</ul>

### 27.6.5　背压处理 {#backpressure-handling}

<p>在高吞吐量的 Agent 化场景下，数据到达速率可能超出 UI 的渲染能力。管理背压的策略：</p>

<ul>
<li><p><strong>Token 批处理</strong>：将 Token 缓冲 50–100ms 后批量渲染而非逐个渲染，降低 DOM 更新频率。</p></li>
<li><p><strong>虚拟滚动</strong>：对于长输出，仅渲染可视区域的内容，丢弃屏幕外的 DOM 节点。</p></li>
<li><p><strong>节流更新</strong>：对于指标与状态展示，无论入流速率如何，都以固定速率（例如 10 Hz）更新。</p></li>
<li><p><strong>渐进式细节</strong>：在高吞吐期间显示摘要视图；完整细节按需提供。</p></li>
</ul>

## 27.7　人机协同 UI 设计 {#subsec:hitl-design}

<p>人机协同（Human-in-the-Loop, HITL）交互是 Agent 化 UI 中最关键的设计挑战之一。目标是在维持有效人类监督的同时，避免形成抹杀自动化效率收益的瓶颈。</p>

### 27.7.1　何时打断 Agent {#when-to-interrupt-the-agent}

<p>并非所有 Agent 操作都需要人类审查。一项有原则的打断策略应考虑：</p>

<ul>
<li><p><strong>可逆性</strong>：不可逆操作（删除文件、发送邮件、进行采购）始终需要批准。可逆操作（读取文件、网页搜索）一般不需要。</p></li>
<li><p><strong>影响范围</strong>：影响外部系统或他人的操作比纯本地操作更需谨慎审查。</p></li>
<li><p><strong>置信度</strong>：当 Agent 对其对用户意图的解读置信度较低时，应请求澄清而非径直执行。</p></li>
<li><p><strong>成本</strong>：高成本操作（大规模 API 调用、昂贵计算）需要批准。</p></li>
<li><p><strong>新颖性</strong>：Agent 在此上下文中此前未执行过的操作，比例行操作更需审查。</p></li>
</ul>

### 27.7.2　分层审批工作流 {#tiered-approval-workflows}

<p>分层审批策略可在监督与效率之间取得平衡：</p>

<div class="admonition note"><p class="admonition-title">三层审批模型</p>
<p><strong>Tier 1（自动批准）：</strong>低风险、可逆、例行操作。例如：网页搜索、读取文件、调用只读 API。Agent 无须打断即可执行；操作被记录以便审计。</p>

<p><strong>Tier 2（通知）：</strong>中风险操作。UI 显示非阻塞通知（“Agent 已向您的草稿箱发送一封草稿邮件”），用户可异步审阅。在短窗口期（例如 30 秒）内允许在操作最终生效前取消。</p>

<p><strong>Tier 3（要求批准）：</strong>高风险、不可逆或高成本操作。Agent 暂停并呈现阻塞式审批门控。用户必须显式批准、拒绝或修改，Agent 才会继续。</p>
</div>

<p>各层之间的阈值可由用户配置（“发送邮件前总是询问”），也可从用户行为中学得（如果用户总是批准网页搜索，则将来自动批准它们）。</p>

### 27.7.3　反馈机制 {#feedback-mechanisms}

<p>除了审批门控，Agent 化 UI 还应提供丰富的反馈机制，帮助 Agent 随时间改进：</p>

<ul>
<li><p><strong>点赞/踩</strong>：对响应的简单二元反馈，被存储并用于基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）微调或偏好学习。</p></li>
<li><p><strong>内联纠正</strong>：用户可直接编辑 Agent 的输出；原始输出与纠正输出之间的差异即是一个训练信号。</p></li>
<li><p><strong>偏好选择</strong>：当 Agent 提供多个选项时，用户的选择即偏好信号。</p></li>
<li><p><strong>显式指令</strong>：“下次别这么做”、“X 之前总是询问”、“偏好方法 Y 而非 Z”——更新 Agent 行为策略的自然语言指令。</p></li>
<li><p><strong>带理由的评分</strong>：评分附带可选的自由文本解释，提供比二元反馈更丰富的信号。</p></li>
</ul>

### 27.7.4　通过 UI 交互教会 Agent {#teaching-the-agent-through-ui-interaction}

<p>最成熟的 HITL UI 将每次交互视为教学机会：</p>

<ul>
<li><p><strong>示范</strong>：用户手动执行任务；Agent 观察并学习首选方法。</p></li>
<li><p><strong>带泛化的纠正</strong>：当用户纠正某个 Agent 操作时，UI 询问“我以后是否总是这样改？”，以将纠正泛化。</p></li>
<li><p><strong>偏好引出</strong>：周期性提示用户对比两种 Agent 行为并指出更偏好哪种。</p></li>
<li><p><strong>行为配置文件</strong>：UI 维护一份可见的“偏好”档案，用户可审阅与编辑，使 Agent 习得的行为透明可控。</p></li>
</ul>

## 27.8　无障碍与信任 {#subsec:ui-trust}

<p>信任不是一项功能——它是一个系统在持续按预期行为、清晰自我解释并能从失败中优雅恢复时所涌现出的属性。Agent 化 UI 必须将信任作为一等关切来设计。</p>

### 27.8.1　解释 Agent 的决策 {#explaining-agent-decisions}

<p>Agent 化 UI 中的可解释性超越了展示思维链。它要求：</p>

<ul>
<li><p><strong>决策理由</strong>：对于重要决策，Agent 不仅应解释<em>决定了什么</em>，还应解释<em>为什么</em>——考虑了哪些因素、拒绝了哪些备选、做出了哪些假设。</p></li>
<li><p><strong>来源归属</strong>：论断应链接到其来源；检索到的文档应可引用。</p></li>
<li><p><strong>反事实解释</strong>：“如果你说的是 X 而不是 Y，我会做 Z”——帮助用户理解 Agent 的决策边界。</p></li>
<li><p><strong>不确定性量化</strong>：明确说明置信度，以及驱动不确定性的因素。</p></li>
</ul>

### 27.8.2　展示置信度水平 {#showing-confidence-levels}

<p>置信度指示器必须经过校准且有意义：</p>

<ul>
<li><p><strong>语言化置信度</strong>：对大多数用户而言，自然语言表达（“我相当有信心”、“这点我不太确定”）比数值概率更易解读。</p></li>
<li><p><strong>视觉化置信度</strong>：颜色编码（绿/黄/红）、图标变体或字重可在不增加文字的情况下编码置信度。</p></li>
<li><p><strong>按论断的置信度</strong>：对于含多项论断的响应，逐项的置信度指示（例如内联脚注）比单一的响应级评分更具信息量。</p></li>
</ul>

### 27.8.3　撤销与回滚能力 {#undo-and-rollback-capabilities}

<p>每一项重要的 Agent 操作在技术可行时都应可撤销：</p>

<ul>
<li><p><strong>带撤销的操作日志</strong>：所有 Agent 操作的时序日志，每个可逆操作旁附“撤销”按钮。</p></li>
<li><p><strong>基于快照的回滚</strong>：对于有状态任务（例如代码编辑、文档写作），定期快照支持回滚到任意先前状态。</p></li>
<li><p><strong>Dry-run 模式</strong>：在执行计划之前，Agent 可模拟运行并展示预测的状态变化，允许用户在任何真实操作发生前批准或修改。</p></li>
<li><p><strong>优雅降级</strong>：当撤销不可行时（例如邮件已发出），UI 应清楚说明，并提供最佳可行的替代方案（例如发送跟进邮件）。</p></li>
</ul>

### 27.8.4　UI 中的审计轨迹 {#audit-trails-in-the-ui}

<p>对于企业与受监管的使用场景，审计轨迹（audit trails）必不可少：</p>

<ul>
<li><p><strong>不可变操作日志</strong>：每一次 Agent 操作、工具调用与人类批准都附带时间戳、用户身份与完整参数被记录。</p></li>
<li><p><strong>可导出的历史</strong>：审计轨迹可导出为 JSON、CSV 或 PDF，用于合规报告。</p></li>
<li><p><strong>Diff 视图</strong>：对于文档或代码修改，审计轨迹包含前后对比 diff。</p></li>
<li><p><strong>会话回放</strong>：可逐步回放整个 Agent 会话，用于调试或合规审查。</p></li>
</ul>

### 27.8.5　管理用户期望 {#managing-user-expectations}

<p>失准的期望是用户不信任的主要来源。Agent 化 UI 应主动管理期望：</p>

<ul>
<li><p><strong>能力披露</strong>：清晰、易获取的文档，说明 Agent 能做什么、不能做什么。</p></li>
<li><p><strong>局限承认</strong>：当 Agent 遇到超出其能力的任务时，应明确说明，而不是默默尝试并失败。</p></li>
<li><p><strong>不确定性沟通</strong>：主动沟通不确定性，而非等待用户去发现错误。</p></li>
<li><p><strong>一致的人格</strong>：一致的 Agent 身份与沟通风格能建立熟悉感与可预测性。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">通过透明度建立信任：案例研究</p>
<p>设想一个被指派预订航班的 Agent。低信任 UI 呈现：“我已为您预订航班。确认号：AA1234。”高信任 UI 则呈现：(1) 所使用搜索参数的摘要；(2) 考虑过的备选及选中该航班的原因；(3) 所执行的具体操作（对预订系统的 API 调用）；(4) 含预订链接的确认详情；(5) 在接下来 24 小时内有效的撤销选项；(6) 关于 Agent 无法做之事的说明（例如“我无法修改此预订；您需要直接致电航空公司”）。第二种 UI 占用更多屏幕空间，但建立了用户对 Agent 行为正确性的信心，并给予他们核实与（必要时）恢复所需的信息。</p>
</div>

## 27.9　实现示例：全栈 Agent 化 UI {#subsec:ui-implementation}

<p>现在给出一个具体的实现示例，将流式、工具可视化与审批门控组合于一个 Python/React 技术栈中。Backend 使用 FastAPI 配合 LangGraph；Frontend 使用 React，借鉴 Vercel AI SDK 的模式并适配自定义 Backend。</p>

### 27.9.1　Backend：FastAPI + LangGraph 实现流式与审批门控 {#backend-fastapi-langgraph-with-streaming-and-approval-gates}

<p class="code-caption"><span class="code-number">代码 27.4　</span>带流式与审批门控的 FastAPI 后端</p>
<div class="highlight"><pre><span></span><code><span class="c1"># backend/main.py</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">asyncio</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">AsyncGenerator</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">fastapi</span><span class="w"> </span><span class="kn">import</span> <span class="n">FastAPI</span><span class="p">,</span> <span class="n">HTTPException</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">fastapi.responses</span><span class="w"> </span><span class="kn">import</span> <span class="n">StreamingResponse</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pydantic</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseModel</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">ChatOpenAI</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">langchain_core.tools</span><span class="w"> </span><span class="kn">import</span> <span class="n">tool</span>

<span class="n">app</span> <span class="o">=</span> <span class="n">FastAPI</span><span class="p">()</span>

<span class="c1"># -- 工具定义 ----------------------------------------------------------</span>
<span class="c1"># -- Tool definitions ----------------------------------------------------------</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">web_search</span><span class="p">(</span><span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="c1"># 在网上搜索信息。</span>
<span class="w">    </span><span class="sd">"""Search the web for information."""</span>
    <span class="k">return</span> <span class="sa">f</span><span class="s2">"Search results for '</span><span class="si">{</span><span class="n">query</span><span class="si">}</span><span class="s2">': [simulated results]"</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">send_email</span><span class="p">(</span><span class="n">to</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">subject</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">body</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="c1"># 发送一封邮件。需要人工批准。</span>
<span class="w">    </span><span class="sd">"""Send an email. REQUIRES HUMAN APPROVAL."""</span>
    <span class="k">return</span> <span class="sa">f</span><span class="s2">"Email sent to </span><span class="si">{</span><span class="n">to</span><span class="si">}</span><span class="s2"> with subject '</span><span class="si">{</span><span class="n">subject</span><span class="si">}</span><span class="s2">'"</span>

<span class="nd">@tool</span>
<span class="k">def</span><span class="w"> </span><span class="nf">read_file</span><span class="p">(</span><span class="n">path</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="c1"># 从文件系统读取文件。</span>
<span class="w">    </span><span class="sd">"""Read a file from the filesystem."""</span>
    <span class="k">try</span><span class="p">:</span>
        <span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">path</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
            <span class="k">return</span> <span class="n">f</span><span class="o">.</span><span class="n">read</span><span class="p">()</span>
    <span class="k">except</span> <span class="ne">FileNotFoundError</span><span class="p">:</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"Error: File not found: </span><span class="si">{</span><span class="n">path</span><span class="si">}</span><span class="s2">"</span>

<span class="c1"># 需要批准的工具（Tier 3）</span>
<span class="c1"># Tools requiring approval (Tier 3)</span>
<span class="n">APPROVAL_REQUIRED_TOOLS</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"send_email"</span><span class="p">}</span>

<span class="c1"># -- 审批门控存储（内存版；生产环境请使用 Redis） ------------------</span>
<span class="c1"># -- Approval gate store (in-memory; use Redis in production) ------------------</span>

<span class="n">approval_store</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">Event</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>
<span class="n">approval_results</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">dict</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>

<span class="c1"># -- LLM 设置 -----------------------------------------------------------------</span>
<span class="c1"># -- LLM setup -----------------------------------------------------------------</span>

<span class="n">llm</span> <span class="o">=</span> <span class="n">ChatOpenAI</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">,</span> <span class="n">streaming</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">tools</span> <span class="o">=</span> <span class="p">[</span><span class="n">web_search</span><span class="p">,</span> <span class="n">send_email</span><span class="p">,</span> <span class="n">read_file</span><span class="p">]</span>
<span class="n">llm_with_tools</span> <span class="o">=</span> <span class="n">llm</span><span class="o">.</span><span class="n">bind_tools</span><span class="p">(</span><span class="n">tools</span><span class="p">)</span>

<span class="k">def</span><span class="w"> </span><span class="nf">should_request_approval</span><span class="p">(</span><span class="n">tool_name</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">bool</span><span class="p">:</span>
    <span class="k">return</span> <span class="n">tool_name</span> <span class="ow">in</span> <span class="n">APPROVAL_REQUIRED_TOOLS</span>

<span class="c1"># -- 流式端点 --------------------------------------------------------</span>
<span class="c1"># -- Streaming endpoint --------------------------------------------------------</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">agent_stream</span><span class="p">(</span>
    <span class="n">session_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
    <span class="n">user_message</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
<span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AsyncGenerator</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="kc">None</span><span class="p">]:</span>
    <span class="c1"># 以 SSE 形式流式发送 Agent 事件。</span>
<span class="w">    </span><span class="sd">"""Stream agent events as SSE."""</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">sse</span><span class="p">(</span><span class="n">event_type</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">data</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s1">'type'</span><span class="p">:</span><span class="w"> </span><span class="n">event_type</span><span class="p">,</span><span class="w"> </span><span class="o">**</span><span class="n">data</span><span class="p">})</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>

    <span class="k">yield</span> <span class="n">sse</span><span class="p">(</span><span class="s2">"status"</span><span class="p">,</span> <span class="p">{</span><span class="s2">"message"</span><span class="p">:</span> <span class="s2">"Agent starting..."</span><span class="p">})</span>

    <span class="c1"># 模拟多步 Agent 执行</span>
    <span class="c1"># Simulate multi-step agent execution</span>
    <span class="n">steps</span> <span class="o">=</span> <span class="p">[</span>
        <span class="p">(</span><span class="s2">"thinking"</span><span class="p">,</span> <span class="p">{</span><span class="s2">"content"</span><span class="p">:</span> <span class="s2">"Analyzing the request..."</span><span class="p">}),</span>
        <span class="p">(</span><span class="s2">"tool_call"</span><span class="p">,</span> <span class="p">{</span>
            <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"web_search"</span><span class="p">,</span>
            <span class="s2">"input"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"query"</span><span class="p">:</span> <span class="n">user_message</span><span class="p">},</span>
            <span class="s2">"tier"</span><span class="p">:</span> <span class="mi">1</span><span class="p">,</span>  <span class="c1"># 自动批准 / Auto-approve</span>
        <span class="p">}),</span>
        <span class="p">(</span><span class="s2">"tool_result"</span><span class="p">,</span> <span class="p">{</span>
            <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"web_search"</span><span class="p">,</span>
            <span class="s2">"output"</span><span class="p">:</span> <span class="sa">f</span><span class="s2">"Results for: </span><span class="si">{</span><span class="n">user_message</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span>
            <span class="s2">"duration_ms"</span><span class="p">:</span> <span class="mi">342</span><span class="p">,</span>
        <span class="p">}),</span>
    <span class="p">]</span>

    <span class="k">for</span> <span class="n">event_type</span><span class="p">,</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">steps</span><span class="p">:</span>
        <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="mf">0.5</span><span class="p">)</span>  <span class="c1"># 模拟处理时间 / Simulate processing time</span>
        <span class="k">yield</span> <span class="n">sse</span><span class="p">(</span><span class="n">event_type</span><span class="p">,</span> <span class="n">data</span><span class="p">)</span>

    <span class="c1"># 模拟一项需要批准的 Tier 3 操作</span>
    <span class="c1"># Simulate a Tier 3 action requiring approval</span>
    <span class="n">approval_id</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">session_id</span><span class="si">}</span><span class="s2">_email_001"</span>
    <span class="n">approval_event</span> <span class="o">=</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">Event</span><span class="p">()</span>
    <span class="n">approval_store</span><span class="p">[</span><span class="n">approval_id</span><span class="p">]</span> <span class="o">=</span> <span class="n">approval_event</span>

    <span class="k">yield</span> <span class="n">sse</span><span class="p">(</span><span class="s2">"approval_required"</span><span class="p">,</span> <span class="p">{</span>
        <span class="s2">"approval_id"</span><span class="p">:</span> <span class="n">approval_id</span><span class="p">,</span>
        <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"send_email"</span><span class="p">,</span>
        <span class="s2">"tier"</span><span class="p">:</span> <span class="mi">3</span><span class="p">,</span>
        <span class="s2">"risk"</span><span class="p">:</span> <span class="s2">"irreversible"</span><span class="p">,</span>
        <span class="s2">"action_summary"</span><span class="p">:</span> <span class="s2">"Send summary email to user@example.com"</span><span class="p">,</span>
        <span class="s2">"parameters"</span><span class="p">:</span> <span class="p">{</span>
            <span class="s2">"to"</span><span class="p">:</span> <span class="s2">"user@example.com"</span><span class="p">,</span>
            <span class="s2">"subject"</span><span class="p">:</span> <span class="sa">f</span><span class="s2">"Research results: </span><span class="si">{</span><span class="n">user_message</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span>
            <span class="s2">"body"</span><span class="p">:</span> <span class="s2">"Here are the findings..."</span><span class="p">,</span>
        <span class="p">},</span>
    <span class="p">})</span>

    <span class="c1"># 等待人工批准（5 分钟后超时）</span>
    <span class="c1"># Wait for human approval (timeout after 5 minutes)</span>
    <span class="k">try</span><span class="p">:</span>
        <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">wait_for</span><span class="p">(</span><span class="n">approval_event</span><span class="o">.</span><span class="n">wait</span><span class="p">(),</span> <span class="n">timeout</span><span class="o">=</span><span class="mi">300</span><span class="p">)</span>
        <span class="n">result</span> <span class="o">=</span> <span class="n">approval_results</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="n">approval_id</span><span class="p">,</span> <span class="p">{})</span>

        <span class="k">if</span> <span class="n">result</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"approved"</span><span class="p">):</span>
            <span class="k">yield</span> <span class="n">sse</span><span class="p">(</span><span class="s2">"tool_call"</span><span class="p">,</span> <span class="p">{</span>
                <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"send_email"</span><span class="p">,</span>
                <span class="s2">"input"</span><span class="p">:</span> <span class="n">result</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"parameters"</span><span class="p">,</span> <span class="p">{}),</span>
                <span class="s2">"tier"</span><span class="p">:</span> <span class="mi">3</span><span class="p">,</span>
                <span class="s2">"approved_by"</span><span class="p">:</span> <span class="s2">"human"</span><span class="p">,</span>
            <span class="p">})</span>
            <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="mf">0.3</span><span class="p">)</span>
            <span class="k">yield</span> <span class="n">sse</span><span class="p">(</span><span class="s2">"tool_result"</span><span class="p">,</span> <span class="p">{</span>
                <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"send_email"</span><span class="p">,</span>
                <span class="s2">"output"</span><span class="p">:</span> <span class="s2">"Email sent successfully"</span><span class="p">,</span>
                <span class="s2">"duration_ms"</span><span class="p">:</span> <span class="mi">128</span><span class="p">,</span>
            <span class="p">})</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="k">yield</span> <span class="n">sse</span><span class="p">(</span><span class="s2">"action_rejected"</span><span class="p">,</span> <span class="p">{</span>
                <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"send_email"</span><span class="p">,</span>
                <span class="s2">"reason"</span><span class="p">:</span> <span class="n">result</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"reason"</span><span class="p">,</span> <span class="s2">"User rejected"</span><span class="p">),</span>
            <span class="p">})</span>
    <span class="k">except</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">TimeoutError</span><span class="p">:</span>
        <span class="k">yield</span> <span class="n">sse</span><span class="p">(</span><span class="s2">"approval_timeout"</span><span class="p">,</span> <span class="p">{</span>
            <span class="s2">"approval_id"</span><span class="p">:</span> <span class="n">approval_id</span><span class="p">,</span>
            <span class="s2">"message"</span><span class="p">:</span> <span class="s2">"Approval timed out; action skipped"</span><span class="p">,</span>
        <span class="p">})</span>

    <span class="c1"># 最终响应</span>
    <span class="c1"># Final response</span>
    <span class="k">yield</span> <span class="n">sse</span><span class="p">(</span><span class="s2">"token"</span><span class="p">,</span> <span class="p">{</span><span class="s2">"content"</span><span class="p">:</span> <span class="s2">"I've completed the research. "</span><span class="p">})</span>
    <span class="k">yield</span> <span class="n">sse</span><span class="p">(</span><span class="s2">"token"</span><span class="p">,</span> <span class="p">{</span><span class="s2">"content"</span><span class="p">:</span> <span class="s2">"Here's a summary of what I found..."</span><span class="p">})</span>
    <span class="k">yield</span> <span class="n">sse</span><span class="p">(</span><span class="s2">"done"</span><span class="p">,</span> <span class="p">{</span><span class="s2">"total_tokens"</span><span class="p">:</span> <span class="mi">847</span><span class="p">,</span> <span class="s2">"duration_ms"</span><span class="p">:</span> <span class="mi">2341</span><span class="p">})</span>

<span class="nd">@app</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"/chat/stream"</span><span class="p">)</span>
<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">chat_stream</span><span class="p">(</span><span class="n">session_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">message</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">StreamingResponse</span><span class="p">(</span>
        <span class="n">agent_stream</span><span class="p">(</span><span class="n">session_id</span><span class="p">,</span> <span class="n">message</span><span class="p">),</span>
        <span class="n">media_type</span><span class="o">=</span><span class="s2">"text/event-stream"</span><span class="p">,</span>
        <span class="n">headers</span><span class="o">=</span><span class="p">{</span><span class="s2">"Cache-Control"</span><span class="p">:</span> <span class="s2">"no-cache"</span><span class="p">,</span> <span class="s2">"X-Accel-Buffering"</span><span class="p">:</span> <span class="s2">"no"</span><span class="p">},</span>
    <span class="p">)</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ApprovalRequest</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">approval_id</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">approved</span><span class="p">:</span> <span class="nb">bool</span>
    <span class="n">parameters</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">reason</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>

<span class="nd">@app</span><span class="o">.</span><span class="n">post</span><span class="p">(</span><span class="s2">"/chat/approve"</span><span class="p">)</span>
<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">handle_approval</span><span class="p">(</span><span class="n">req</span><span class="p">:</span> <span class="n">ApprovalRequest</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">req</span><span class="o">.</span><span class="n">approval_id</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">approval_store</span><span class="p">:</span>
        <span class="k">raise</span> <span class="n">HTTPException</span><span class="p">(</span><span class="n">status_code</span><span class="o">=</span><span class="mi">404</span><span class="p">,</span> <span class="n">detail</span><span class="o">=</span><span class="s2">"Approval not found"</span><span class="p">)</span>
    <span class="n">approval_results</span><span class="p">[</span><span class="n">req</span><span class="o">.</span><span class="n">approval_id</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">"approved"</span><span class="p">:</span> <span class="n">req</span><span class="o">.</span><span class="n">approved</span><span class="p">,</span>
        <span class="s2">"parameters"</span><span class="p">:</span> <span class="n">req</span><span class="o">.</span><span class="n">parameters</span><span class="p">,</span>
        <span class="s2">"reason"</span><span class="p">:</span> <span class="n">req</span><span class="o">.</span><span class="n">reason</span><span class="p">,</span>
    <span class="p">}</span>
    <span class="n">approval_store</span><span class="p">[</span><span class="n">req</span><span class="o">.</span><span class="n">approval_id</span><span class="p">]</span><span class="o">.</span><span class="n">set</span><span class="p">()</span>
    <span class="k">return</span> <span class="p">{</span><span class="s2">"status"</span><span class="p">:</span> <span class="s2">"ok"</span><span class="p">}</span>
</code></pre></div>

### 27.9.2　Frontend：React 实现流式与工具可视化 {#frontend-react-with-streaming-and-tool-visualization}

<p class="code-caption"><span class="code-number">代码 27.5　</span>带流式工具可视化与审批门控的 React 前端</p>
<div class="highlight"><pre><span></span><code><span class="c1">// frontend/AgentChat.tsx</span>
<span class="k">import</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">useState</span><span class="p">,</span><span class="w"> </span><span class="nx">useEffect</span><span class="p">,</span><span class="w"> </span><span class="nx">useRef</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="kr">from</span><span class="w"> </span><span class="s1">'react'</span><span class="p">;</span>

<span class="c1">// -- 类型定义 ---------------------------------------------------------------------</span>
<span class="c1">// -- Types ---------------------------------------------------------------------</span>

<span class="kr">type</span><span class="w"> </span><span class="nx">AgentEvent</span><span class="w"> </span><span class="o">=</span>
<span class="w">  </span><span class="o">|</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kr">type</span><span class="o">:</span><span class="w"> </span><span class="s1">'status'</span><span class="p">;</span><span class="w"> </span><span class="nx">message</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="w"> </span><span class="p">}</span>
<span class="w">  </span><span class="o">|</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kr">type</span><span class="o">:</span><span class="w"> </span><span class="s1">'thinking'</span><span class="p">;</span><span class="w"> </span><span class="nx">content</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="w"> </span><span class="p">}</span>
<span class="w">  </span><span class="o">|</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kr">type</span><span class="o">:</span><span class="w"> </span><span class="s1">'token'</span><span class="p">;</span><span class="w"> </span><span class="nx">content</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="w"> </span><span class="p">}</span>
<span class="w">  </span><span class="o">|</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kr">type</span><span class="o">:</span><span class="w"> </span><span class="s1">'tool_call'</span><span class="p">;</span><span class="w"> </span><span class="nx">tool</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="p">;</span><span class="w"> </span><span class="nx">input</span><span class="o">:</span><span class="w"> </span><span class="kt">object</span><span class="p">;</span><span class="w"> </span><span class="nx">tier</span><span class="o">:</span><span class="w"> </span><span class="kt">number</span><span class="w"> </span><span class="p">}</span>
<span class="w">  </span><span class="o">|</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kr">type</span><span class="o">:</span><span class="w"> </span><span class="s1">'tool_result'</span><span class="p">;</span><span class="w"> </span><span class="nx">tool</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="p">;</span><span class="w"> </span><span class="nx">output</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="p">;</span><span class="w"> </span><span class="nx">duration_ms</span><span class="o">:</span><span class="w"> </span><span class="kt">number</span><span class="w"> </span><span class="p">}</span>
<span class="w">  </span><span class="o">|</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kr">type</span><span class="o">:</span><span class="w"> </span><span class="s1">'approval_required'</span><span class="p">;</span><span class="w"> </span><span class="nx">approval_id</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="p">;</span><span class="w"> </span><span class="nx">tool</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="p">;</span>
<span class="w">      </span><span class="nx">tier</span><span class="o">:</span><span class="w"> </span><span class="kt">number</span><span class="p">;</span><span class="w"> </span><span class="nx">risk</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="p">;</span><span class="w"> </span><span class="nx">action_summary</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="p">;</span><span class="w"> </span><span class="nx">parameters</span><span class="o">:</span><span class="w"> </span><span class="kt">object</span><span class="w"> </span><span class="p">}</span>
<span class="w">  </span><span class="o">|</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kr">type</span><span class="o">:</span><span class="w"> </span><span class="s1">'action_rejected'</span><span class="p">;</span><span class="w"> </span><span class="nx">tool</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="p">;</span><span class="w"> </span><span class="nx">reason</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="w"> </span><span class="p">}</span>
<span class="w">  </span><span class="o">|</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kr">type</span><span class="o">:</span><span class="w"> </span><span class="s1">'done'</span><span class="p">;</span><span class="w"> </span><span class="nx">total_tokens</span><span class="o">:</span><span class="w"> </span><span class="kt">number</span><span class="p">;</span><span class="w"> </span><span class="nx">duration_ms</span><span class="o">:</span><span class="w"> </span><span class="kt">number</span><span class="w"> </span><span class="p">};</span>

<span class="c1">// -- 工具卡片组件 -------------------------------------------------------</span>
<span class="c1">// -- Tool Card Component -------------------------------------------------------</span>

<span class="kd">function</span><span class="w"> </span><span class="nx">ToolCard</span><span class="p">({</span><span class="w"> </span><span class="nx">event</span><span class="w"> </span><span class="p">}</span><span class="o">:</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">event</span><span class="o">:</span><span class="w"> </span><span class="kt">AgentEvent</span><span class="w"> </span><span class="o">&amp;</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kr">type</span><span class="o">:</span><span class="w"> </span><span class="s1">'tool_call'</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="p">})</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="p">[</span><span class="nx">expanded</span><span class="p">,</span><span class="w"> </span><span class="nx">setExpanded</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nx">useState</span><span class="p">(</span><span class="kc">false</span><span class="p">);</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="nx">tierColors</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="mf">1</span><span class="o">:</span><span class="w"> </span><span class="s1">'#22c55e'</span><span class="p">,</span><span class="w"> </span><span class="mf">2</span><span class="o">:</span><span class="w"> </span><span class="s1">'#f59e0b'</span><span class="p">,</span><span class="w"> </span><span class="mf">3</span><span class="o">:</span><span class="w"> </span><span class="s1">'#ef4444'</span><span class="w"> </span><span class="p">};</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="nx">color</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nx">tierColors</span><span class="p">[</span><span class="nx">event</span><span class="p">.</span><span class="nx">tier</span><span class="w"> </span><span class="kr">as</span><span class="w"> </span><span class="nx">keyof</span><span class="w"> </span><span class="ow">typeof</span><span class="w"> </span><span class="nx">tierColors</span><span class="p">]</span><span class="w"> </span><span class="o">||</span><span class="w"> </span><span class="s1">'#6b7280'</span><span class="p">;</span>

<span class="w">  </span><span class="k">return</span><span class="w"> </span><span class="p">(</span>
<span class="w">    </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">border</span><span class="o">:</span><span class="w"> </span><span class="sb">`1px solid </span><span class="si">${</span><span class="nx">color</span><span class="si">}</span><span class="sb">`</span><span class="p">,</span><span class="w"> </span><span class="nx">borderRadius</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="p">,</span><span class="w"> </span><span class="nx">padding</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="p">,</span>
<span class="w">                  </span><span class="nx">margin</span><span class="o">:</span><span class="w"> </span><span class="s1">'4px 0'</span><span class="p">,</span><span class="w"> </span><span class="nx">fontSize</span><span class="o">:</span><span class="w"> </span><span class="kt">13</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">      </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">display</span><span class="o">:</span><span class="w"> </span><span class="s1">'flex'</span><span class="p">,</span><span class="w"> </span><span class="nx">alignItems</span><span class="o">:</span><span class="w"> </span><span class="s1">'center'</span><span class="p">,</span><span class="w"> </span><span class="nx">gap</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">        </span><span class="p">&lt;</span><span class="nt">span</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">color</span><span class="p">,</span><span class="w"> </span><span class="nx">fontWeight</span><span class="o">:</span><span class="w"> </span><span class="kt">600</span><span class="w"> </span><span class="p">}}&gt;[</span><span class="nx">gear</span><span class="p">]</span><span class="w"> </span><span class="p">{</span><span class="nx">event</span><span class="p">.</span><span class="nx">tool</span><span class="p">}&lt;/</span><span class="nt">span</span><span class="p">&gt;</span>
<span class="w">        </span><span class="p">&lt;</span><span class="nt">span</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">color</span><span class="o">:</span><span class="w"> </span><span class="s1">'#6b7280'</span><span class="p">,</span><span class="w"> </span><span class="nx">fontSize</span><span class="o">:</span><span class="w"> </span><span class="kt">11</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">          </span><span class="nx">Tier</span><span class="w"> </span><span class="p">{</span><span class="nx">event</span><span class="p">.</span><span class="nx">tier</span><span class="p">}</span><span class="w"> </span><span class="o">|</span><span class="w"> </span><span class="p">{</span><span class="nx">event</span><span class="p">.</span><span class="nx">tier</span><span class="w"> </span><span class="o">===</span><span class="w"> </span><span class="mf">1</span><span class="w"> </span><span class="o">?</span><span class="w"> </span><span class="s1">'Auto'</span><span class="w"> </span><span class="o">:</span><span class="w"> </span><span class="s1">'Approved'</span><span class="p">}</span>
<span class="w">        </span><span class="p">&lt;/</span><span class="nt">span</span><span class="p">&gt;</span>
<span class="w">        </span><span class="p">&lt;</span><span class="nt">button</span><span class="w"> </span><span class="na">onClick</span><span class="o">=</span><span class="p">{()</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="nx">setExpanded</span><span class="p">(</span><span class="o">!</span><span class="nx">expanded</span><span class="p">)}</span>
<span class="w">                </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">marginLeft</span><span class="o">:</span><span class="w"> </span><span class="s1">'auto'</span><span class="p">,</span><span class="w"> </span><span class="nx">fontSize</span><span class="o">:</span><span class="w"> </span><span class="kt">11</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">          </span><span class="p">{</span><span class="nx">expanded</span><span class="w"> </span><span class="o">?</span><span class="w"> </span><span class="s1">'Hide'</span><span class="w"> </span><span class="o">:</span><span class="w"> </span><span class="s1">'Details'</span><span class="p">}</span>
<span class="w">        </span><span class="p">&lt;/</span><span class="nt">button</span><span class="p">&gt;</span>
<span class="w">      </span><span class="p">&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">      </span><span class="p">{</span><span class="nx">expanded</span><span class="w"> </span><span class="o">&amp;&amp;</span><span class="w"> </span><span class="p">(</span>
<span class="w">        </span><span class="p">&lt;</span><span class="nt">pre</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">marginTop</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="p">,</span><span class="w"> </span><span class="nx">fontSize</span><span class="o">:</span><span class="w"> </span><span class="kt">11</span><span class="p">,</span><span class="w"> </span><span class="nx">background</span><span class="o">:</span><span class="w"> </span><span class="s1">'#f3f4f6'</span><span class="p">,</span>
<span class="w">                      </span><span class="nx">padding</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="p">,</span><span class="w"> </span><span class="nx">borderRadius</span><span class="o">:</span><span class="w"> </span><span class="kt">4</span><span class="p">,</span><span class="w"> </span><span class="nx">overflow</span><span class="o">:</span><span class="w"> </span><span class="s1">'auto'</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">          </span><span class="p">{</span><span class="nb">JSON</span><span class="p">.</span><span class="nx">stringify</span><span class="p">(</span><span class="nx">event</span><span class="p">.</span><span class="nx">input</span><span class="p">,</span><span class="w"> </span><span class="kc">null</span><span class="p">,</span><span class="w"> </span><span class="mf">2</span><span class="p">)}</span>
<span class="w">        </span><span class="p">&lt;/</span><span class="nt">pre</span><span class="p">&gt;</span>
<span class="w">      </span><span class="p">)}</span>
<span class="w">    </span><span class="p">&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">  </span><span class="p">);</span>
<span class="p">}</span>

<span class="c1">// -- 审批门控组件 ---------------------------------------------------</span>
<span class="c1">// -- Approval Gate Component ---------------------------------------------------</span>

<span class="kd">function</span><span class="w"> </span><span class="nx">ApprovalGate</span><span class="p">({</span>
<span class="w">  </span><span class="nx">event</span><span class="p">,</span>
<span class="w">  </span><span class="nx">onDecision</span><span class="p">,</span>
<span class="p">}</span><span class="o">:</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="nx">event</span><span class="o">:</span><span class="w"> </span><span class="kt">AgentEvent</span><span class="w"> </span><span class="o">&amp;</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kr">type</span><span class="o">:</span><span class="w"> </span><span class="s1">'approval_required'</span><span class="w"> </span><span class="p">};</span>
<span class="w">  </span><span class="nx">onDecision</span><span class="o">:</span><span class="w"> </span><span class="p">(</span><span class="nx">approved</span><span class="o">:</span><span class="w"> </span><span class="kt">boolean</span><span class="p">,</span><span class="w"> </span><span class="nx">params?</span><span class="o">:</span><span class="w"> </span><span class="kt">object</span><span class="p">)</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="ow">void</span><span class="p">;</span>
<span class="p">})</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="nx">riskColors</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">reversible</span><span class="o">:</span><span class="w"> </span><span class="s1">'#22c55e'</span><span class="p">,</span><span class="w"> </span><span class="s1">'hard-to-undo'</span><span class="o">:</span><span class="w"> </span><span class="s1">'#f59e0b'</span><span class="p">,</span>
<span class="w">                       </span><span class="nx">irreversible</span><span class="o">:</span><span class="w"> </span><span class="s1">'#ef4444'</span><span class="w"> </span><span class="p">};</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="nx">riskColor</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nx">riskColors</span><span class="p">[</span><span class="nx">event</span><span class="p">.</span><span class="nx">risk</span><span class="w"> </span><span class="kr">as</span><span class="w"> </span><span class="nx">keyof</span><span class="w"> </span><span class="ow">typeof</span><span class="w"> </span><span class="nx">riskColors</span><span class="p">]</span><span class="w"> </span><span class="o">||</span><span class="w"> </span><span class="s1">'#6b7280'</span><span class="p">;</span>

<span class="w">  </span><span class="k">return</span><span class="w"> </span><span class="p">(</span>
<span class="w">    </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">border</span><span class="o">:</span><span class="w"> </span><span class="sb">`2px solid </span><span class="si">${</span><span class="nx">riskColor</span><span class="si">}</span><span class="sb">`</span><span class="p">,</span><span class="w"> </span><span class="nx">borderRadius</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="p">,</span>
<span class="w">                  </span><span class="nx">padding</span><span class="o">:</span><span class="w"> </span><span class="kt">16</span><span class="p">,</span><span class="w"> </span><span class="nx">margin</span><span class="o">:</span><span class="w"> </span><span class="s1">'8px 0'</span><span class="p">,</span><span class="w"> </span><span class="nx">background</span><span class="o">:</span><span class="w"> </span><span class="s1">'#fef9f0'</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">      </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">fontWeight</span><span class="o">:</span><span class="w"> </span><span class="kt">700</span><span class="p">,</span><span class="w"> </span><span class="nx">color</span><span class="o">:</span><span class="w"> </span><span class="kt">riskColor</span><span class="p">,</span><span class="w"> </span><span class="nx">marginBottom</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">        </span><span class="p">[</span><span class="o">!</span><span class="p">]</span><span class="w"> </span><span class="nx">Approval</span><span class="w"> </span><span class="nx">Required</span><span class="o">:</span><span class="w"> </span><span class="p">{</span><span class="nx">event</span><span class="p">.</span><span class="nx">tool</span><span class="p">}</span>
<span class="w">      </span><span class="p">&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">      </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">marginBottom</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="w"> </span><span class="p">}}&gt;{</span><span class="nx">event</span><span class="p">.</span><span class="nx">action_summary</span><span class="p">}&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">      </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">fontSize</span><span class="o">:</span><span class="w"> </span><span class="kt">12</span><span class="p">,</span><span class="w"> </span><span class="nx">color</span><span class="o">:</span><span class="w"> </span><span class="s1">'#6b7280'</span><span class="p">,</span><span class="w"> </span><span class="nx">marginBottom</span><span class="o">:</span><span class="w"> </span><span class="kt">12</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">        </span><span class="nx">Risk</span><span class="w"> </span><span class="nx">level</span><span class="o">:</span><span class="w"> </span><span class="p">&lt;</span><span class="nt">span</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">color</span><span class="o">:</span><span class="w"> </span><span class="kt">riskColor</span><span class="w"> </span><span class="p">}}&gt;{</span><span class="nx">event</span><span class="p">.</span><span class="nx">risk</span><span class="p">}&lt;/</span><span class="nt">span</span><span class="p">&gt;</span>
<span class="w">      </span><span class="p">&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">      </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">display</span><span class="o">:</span><span class="w"> </span><span class="s1">'flex'</span><span class="p">,</span><span class="w"> </span><span class="nx">gap</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">        </span><span class="p">&lt;</span><span class="nt">button</span>
<span class="w">          </span><span class="na">onClick</span><span class="o">=</span><span class="p">{()</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="nx">onDecision</span><span class="p">(</span><span class="kc">true</span><span class="p">,</span><span class="w"> </span><span class="nx">event</span><span class="p">.</span><span class="nx">parameters</span><span class="p">)}</span>
<span class="w">          </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">background</span><span class="o">:</span><span class="w"> </span><span class="s1">'#22c55e'</span><span class="p">,</span><span class="w"> </span><span class="nx">color</span><span class="o">:</span><span class="w"> </span><span class="s1">'white'</span><span class="p">,</span><span class="w"> </span><span class="nx">border</span><span class="o">:</span><span class="w"> </span><span class="s1">'none'</span><span class="p">,</span>
<span class="w">                   </span><span class="nx">borderRadius</span><span class="o">:</span><span class="w"> </span><span class="kt">6</span><span class="p">,</span><span class="w"> </span><span class="nx">padding</span><span class="o">:</span><span class="w"> </span><span class="s1">'8px 16px'</span><span class="p">,</span><span class="w"> </span><span class="nx">cursor</span><span class="o">:</span><span class="w"> </span><span class="s1">'pointer'</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">          </span><span class="p">[</span><span class="nx">ok</span><span class="p">]</span><span class="w"> </span><span class="nx">Approve</span>
<span class="w">        </span><span class="p">&lt;/</span><span class="nt">button</span><span class="p">&gt;</span>
<span class="w">        </span><span class="p">&lt;</span><span class="nt">button</span>
<span class="w">          </span><span class="na">onClick</span><span class="o">=</span><span class="p">{()</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="nx">onDecision</span><span class="p">(</span><span class="kc">false</span><span class="p">)}</span>
<span class="w">          </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">background</span><span class="o">:</span><span class="w"> </span><span class="s1">'#ef4444'</span><span class="p">,</span><span class="w"> </span><span class="nx">color</span><span class="o">:</span><span class="w"> </span><span class="s1">'white'</span><span class="p">,</span><span class="w"> </span><span class="nx">border</span><span class="o">:</span><span class="w"> </span><span class="s1">'none'</span><span class="p">,</span>
<span class="w">                   </span><span class="nx">borderRadius</span><span class="o">:</span><span class="w"> </span><span class="kt">6</span><span class="p">,</span><span class="w"> </span><span class="nx">padding</span><span class="o">:</span><span class="w"> </span><span class="s1">'8px 16px'</span><span class="p">,</span><span class="w"> </span><span class="nx">cursor</span><span class="o">:</span><span class="w"> </span><span class="s1">'pointer'</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">          </span><span class="p">[</span><span class="nx">x</span><span class="p">]</span><span class="w"> </span><span class="nx">Reject</span>
<span class="w">        </span><span class="p">&lt;/</span><span class="nt">button</span><span class="p">&gt;</span>
<span class="w">      </span><span class="p">&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">    </span><span class="p">&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">  </span><span class="p">);</span>
<span class="p">}</span>

<span class="c1">// -- 主聊天组件 -------------------------------------------------------</span>
<span class="c1">// -- Main Chat Component -------------------------------------------------------</span>

<span class="k">export</span><span class="w"> </span><span class="kd">function</span><span class="w"> </span><span class="nx">AgentChat</span><span class="p">()</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="p">[</span><span class="nx">events</span><span class="p">,</span><span class="w"> </span><span class="nx">setEvents</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nx">useState</span><span class="p">&lt;</span><span class="nt">AgentEvent</span><span class="err">[]</span><span class="p">&gt;([]);</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="p">[</span><span class="nx">response</span><span class="p">,</span><span class="w"> </span><span class="nx">setResponse</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nx">useState</span><span class="p">(</span><span class="s1">''</span><span class="p">);</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="p">[</span><span class="nx">isStreaming</span><span class="p">,</span><span class="w"> </span><span class="nx">setIsStreaming</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nx">useState</span><span class="p">(</span><span class="kc">false</span><span class="p">);</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="p">[</span><span class="nx">input</span><span class="p">,</span><span class="w"> </span><span class="nx">setInput</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nx">useState</span><span class="p">(</span><span class="s1">''</span><span class="p">);</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="nx">sessionId</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nx">useRef</span><span class="p">(</span><span class="sb">`session_</span><span class="si">${</span><span class="nb">Date</span><span class="p">.</span><span class="nx">now</span><span class="p">()</span><span class="si">}</span><span class="sb">`</span><span class="p">);</span>

<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="nx">sendMessage</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="k">async</span><span class="w"> </span><span class="p">()</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="o">!</span><span class="nx">input</span><span class="p">.</span><span class="nx">trim</span><span class="p">()</span><span class="w"> </span><span class="o">||</span><span class="w"> </span><span class="nx">isStreaming</span><span class="p">)</span><span class="w"> </span><span class="k">return</span><span class="p">;</span>
<span class="w">    </span><span class="nx">setEvents</span><span class="p">([]);</span>
<span class="w">    </span><span class="nx">setResponse</span><span class="p">(</span><span class="s1">''</span><span class="p">);</span>
<span class="w">    </span><span class="nx">setIsStreaming</span><span class="p">(</span><span class="kc">true</span><span class="p">);</span>

<span class="w">    </span><span class="kd">const</span><span class="w"> </span><span class="nx">url</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="sb">`/chat/stream?session_id=</span><span class="si">${</span><span class="nx">sessionId</span><span class="p">.</span><span class="nx">current</span><span class="si">}</span><span class="sb">`</span>
<span class="w">              </span><span class="o">+</span><span class="w"> </span><span class="sb">`&amp;message=</span><span class="si">${</span><span class="nb">encodeURIComponent</span><span class="p">(</span><span class="nx">input</span><span class="p">)</span><span class="si">}</span><span class="sb">`</span><span class="p">;</span>
<span class="w">    </span><span class="kd">const</span><span class="w"> </span><span class="nx">es</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="ow">new</span><span class="w"> </span><span class="nx">EventSource</span><span class="p">(</span><span class="nx">url</span><span class="p">);</span>

<span class="w">    </span><span class="nx">es</span><span class="p">.</span><span class="nx">onmessage</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">(</span><span class="nx">e</span><span class="p">)</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">      </span><span class="kd">const</span><span class="w"> </span><span class="nx">event</span><span class="o">:</span><span class="w"> </span><span class="kt">AgentEvent</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="nb">JSON</span><span class="p">.</span><span class="nx">parse</span><span class="p">(</span><span class="nx">e</span><span class="p">.</span><span class="nx">data</span><span class="p">);</span>
<span class="w">      </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="nx">event</span><span class="p">.</span><span class="kr">type</span><span class="w"> </span><span class="o">===</span><span class="w"> </span><span class="s1">'token'</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="nx">setResponse</span><span class="p">(</span><span class="nx">prev</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="nx">prev</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="nx">event</span><span class="p">.</span><span class="nx">content</span><span class="p">);</span>
<span class="w">      </span><span class="p">}</span><span class="w"> </span><span class="k">else</span><span class="w"> </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="nx">event</span><span class="p">.</span><span class="kr">type</span><span class="w"> </span><span class="o">===</span><span class="w"> </span><span class="s1">'done'</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="nx">setIsStreaming</span><span class="p">(</span><span class="kc">false</span><span class="p">);</span>
<span class="w">        </span><span class="nx">es</span><span class="p">.</span><span class="nx">close</span><span class="p">();</span>
<span class="w">      </span><span class="p">}</span><span class="w"> </span><span class="k">else</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="nx">setEvents</span><span class="p">(</span><span class="nx">prev</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="p">[...</span><span class="nx">prev</span><span class="p">,</span><span class="w"> </span><span class="nx">event</span><span class="p">]);</span>
<span class="w">      </span><span class="p">}</span>
<span class="w">    </span><span class="p">};</span>

<span class="w">    </span><span class="nx">es</span><span class="p">.</span><span class="nx">onerror</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="p">()</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">setIsStreaming</span><span class="p">(</span><span class="kc">false</span><span class="p">);</span><span class="w"> </span><span class="nx">es</span><span class="p">.</span><span class="nx">close</span><span class="p">();</span><span class="w"> </span><span class="p">};</span>
<span class="w">    </span><span class="nx">setInput</span><span class="p">(</span><span class="s1">''</span><span class="p">);</span>
<span class="w">  </span><span class="p">};</span>

<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="nx">handleApproval</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="k">async</span><span class="w"> </span><span class="p">(</span>
<span class="w">    </span><span class="nx">approvalId</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="p">,</span>
<span class="w">    </span><span class="nx">approved</span><span class="o">:</span><span class="w"> </span><span class="kt">boolean</span><span class="p">,</span>
<span class="w">    </span><span class="nx">parameters?</span><span class="o">:</span><span class="w"> </span><span class="kt">object</span><span class="p">,</span>
<span class="w">  </span><span class="p">)</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="k">await</span><span class="w"> </span><span class="nx">fetch</span><span class="p">(</span><span class="s1">'/chat/approve'</span><span class="p">,</span><span class="w"> </span><span class="p">{</span>
<span class="w">      </span><span class="nx">method</span><span class="o">:</span><span class="w"> </span><span class="s1">'POST'</span><span class="p">,</span>
<span class="w">      </span><span class="nx">headers</span><span class="o">:</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="s1">'Content-Type'</span><span class="o">:</span><span class="w"> </span><span class="s1">'application/json'</span><span class="w"> </span><span class="p">},</span>
<span class="w">      </span><span class="nx">body</span><span class="o">:</span><span class="w"> </span><span class="kt">JSON.stringify</span><span class="p">({</span><span class="w"> </span><span class="nx">approval_id</span><span class="o">:</span><span class="w"> </span><span class="kt">approvalId</span><span class="p">,</span><span class="w"> </span><span class="nx">approved</span><span class="p">,</span><span class="w"> </span><span class="nx">parameters</span><span class="w"> </span><span class="p">}),</span>
<span class="w">    </span><span class="p">});</span>
<span class="w">  </span><span class="p">};</span>

<span class="w">  </span><span class="k">return</span><span class="w"> </span><span class="p">(</span>
<span class="w">    </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">maxWidth</span><span class="o">:</span><span class="w"> </span><span class="kt">800</span><span class="p">,</span><span class="w"> </span><span class="nx">margin</span><span class="o">:</span><span class="w"> </span><span class="s1">'0 auto'</span><span class="p">,</span><span class="w"> </span><span class="nx">padding</span><span class="o">:</span><span class="w"> </span><span class="kt">16</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">      </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">minHeight</span><span class="o">:</span><span class="w"> </span><span class="kt">400</span><span class="p">,</span><span class="w"> </span><span class="nx">border</span><span class="o">:</span><span class="w"> </span><span class="s1">'1px solid #e5e7eb'</span><span class="p">,</span>
<span class="w">                    </span><span class="nx">borderRadius</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="p">,</span><span class="w"> </span><span class="nx">padding</span><span class="o">:</span><span class="w"> </span><span class="kt">16</span><span class="p">,</span><span class="w"> </span><span class="nx">marginBottom</span><span class="o">:</span><span class="w"> </span><span class="kt">16</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">        </span><span class="p">{</span><span class="nx">events</span><span class="p">.</span><span class="nx">map</span><span class="p">((</span><span class="nx">event</span><span class="p">,</span><span class="w"> </span><span class="nx">i</span><span class="p">)</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">          </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="nx">event</span><span class="p">.</span><span class="kr">type</span><span class="w"> </span><span class="o">===</span><span class="w"> </span><span class="s1">'tool_call'</span><span class="p">)</span>
<span class="w">            </span><span class="k">return</span><span class="w"> </span><span class="p">&lt;</span><span class="nt">ToolCard</span><span class="w"> </span><span class="na">key</span><span class="o">=</span><span class="p">{</span><span class="nx">i</span><span class="p">}</span><span class="w"> </span><span class="na">event</span><span class="o">=</span><span class="p">{</span><span class="nx">event</span><span class="p">}</span><span class="w"> </span><span class="p">/&gt;;</span>
<span class="w">          </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="nx">event</span><span class="p">.</span><span class="kr">type</span><span class="w"> </span><span class="o">===</span><span class="w"> </span><span class="s1">'approval_required'</span><span class="p">)</span>
<span class="w">            </span><span class="k">return</span><span class="w"> </span><span class="p">(</span>
<span class="w">              </span><span class="p">&lt;</span><span class="nt">ApprovalGate</span><span class="w"> </span><span class="na">key</span><span class="o">=</span><span class="p">{</span><span class="nx">i</span><span class="p">}</span><span class="w"> </span><span class="na">event</span><span class="o">=</span><span class="p">{</span><span class="nx">event</span><span class="p">}</span>
<span class="w">                </span><span class="na">onDecision</span><span class="o">=</span><span class="p">{(</span><span class="nx">approved</span><span class="p">,</span><span class="w"> </span><span class="nx">params</span><span class="p">)</span><span class="w"> </span><span class="p">=&gt;</span>
<span class="w">                  </span><span class="nx">handleApproval</span><span class="p">(</span><span class="nx">event</span><span class="p">.</span><span class="nx">approval_id</span><span class="p">,</span><span class="w"> </span><span class="nx">approved</span><span class="p">,</span><span class="w"> </span><span class="nx">params</span><span class="p">)}</span><span class="w"> </span><span class="p">/&gt;</span>
<span class="w">            </span><span class="p">);</span>
<span class="w">          </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="nx">event</span><span class="p">.</span><span class="kr">type</span><span class="w"> </span><span class="o">===</span><span class="w"> </span><span class="s1">'status'</span><span class="w"> </span><span class="o">||</span><span class="w"> </span><span class="nx">event</span><span class="p">.</span><span class="kr">type</span><span class="w"> </span><span class="o">===</span><span class="w"> </span><span class="s1">'thinking'</span><span class="p">)</span>
<span class="w">            </span><span class="k">return</span><span class="w"> </span><span class="p">(</span>
<span class="w">              </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">key</span><span class="o">=</span><span class="p">{</span><span class="nx">i</span><span class="p">}</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">color</span><span class="o">:</span><span class="w"> </span><span class="s1">'#6b7280'</span><span class="p">,</span><span class="w"> </span><span class="nx">fontSize</span><span class="o">:</span><span class="w"> </span><span class="kt">12</span><span class="p">,</span>
<span class="w">                                    </span><span class="nx">fontStyle</span><span class="o">:</span><span class="w"> </span><span class="s1">'italic'</span><span class="p">,</span><span class="w"> </span><span class="nx">margin</span><span class="o">:</span><span class="w"> </span><span class="s1">'4px 0'</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">                </span><span class="p">{</span><span class="nx">event</span><span class="p">.</span><span class="kr">type</span><span class="w"> </span><span class="o">===</span><span class="w"> </span><span class="s1">'thinking'</span><span class="w"> </span><span class="o">?</span><span class="w"> </span><span class="nx">event.content</span><span class="w"> </span><span class="o">:</span><span class="w"> </span><span class="kt">event.message</span><span class="p">}</span>
<span class="w">              </span><span class="p">&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">            </span><span class="p">);</span>
<span class="w">          </span><span class="k">return</span><span class="w"> </span><span class="kc">null</span><span class="p">;</span>
<span class="w">        </span><span class="p">})}</span>
<span class="w">        </span><span class="p">{</span><span class="nx">response</span><span class="w"> </span><span class="o">&amp;&amp;</span><span class="w"> </span><span class="p">(</span>
<span class="w">          </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">marginTop</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="p">,</span><span class="w"> </span><span class="nx">lineHeight</span><span class="o">:</span><span class="w"> </span><span class="kt">1.6</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">            </span><span class="p">{</span><span class="nx">response</span><span class="p">}</span>
<span class="w">            </span><span class="p">{</span><span class="nx">isStreaming</span><span class="w"> </span><span class="o">&amp;&amp;</span><span class="w"> </span><span class="p">&lt;</span><span class="nt">span</span><span class="w"> </span><span class="na">className</span><span class="o">=</span><span class="s">"cursor-blink"</span><span class="p">&gt;</span><span class="o">|</span><span class="p">&lt;/</span><span class="nt">span</span><span class="p">&gt;}</span>
<span class="w">          </span><span class="p">&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">        </span><span class="p">)}</span>
<span class="w">      </span><span class="p">&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">      </span><span class="p">&lt;</span><span class="nt">div</span><span class="w"> </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">display</span><span class="o">:</span><span class="w"> </span><span class="s1">'flex'</span><span class="p">,</span><span class="w"> </span><span class="nx">gap</span><span class="o">:</span><span class="w"> </span><span class="kt">8</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">        </span><span class="p">&lt;</span><span class="nt">input</span>
<span class="w">          </span><span class="na">value</span><span class="o">=</span><span class="p">{</span><span class="nx">input</span><span class="p">}</span>
<span class="w">          </span><span class="na">onChange</span><span class="o">=</span><span class="p">{</span><span class="nx">e</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="nx">setInput</span><span class="p">(</span><span class="nx">e</span><span class="p">.</span><span class="nx">target</span><span class="p">.</span><span class="nx">value</span><span class="p">)}</span>
<span class="w">          </span><span class="na">onKeyDown</span><span class="o">=</span><span class="p">{</span><span class="nx">e</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="nx">e</span><span class="p">.</span><span class="nx">key</span><span class="w"> </span><span class="o">===</span><span class="w"> </span><span class="s1">'Enter'</span><span class="w"> </span><span class="o">&amp;&amp;</span><span class="w"> </span><span class="nx">sendMessage</span><span class="p">()}</span>
<span class="w">          </span><span class="na">placeholder</span><span class="o">=</span><span class="s">"Ask the agent..."</span>
<span class="w">          </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">flex</span><span class="o">:</span><span class="w"> </span><span class="kt">1</span><span class="p">,</span><span class="w"> </span><span class="nx">padding</span><span class="o">:</span><span class="w"> </span><span class="s1">'8px 12px'</span><span class="p">,</span><span class="w"> </span><span class="nx">borderRadius</span><span class="o">:</span><span class="w"> </span><span class="kt">6</span><span class="p">,</span>
<span class="w">                   </span><span class="nx">border</span><span class="o">:</span><span class="w"> </span><span class="s1">'1px solid #d1d5db'</span><span class="p">,</span><span class="w"> </span><span class="nx">fontSize</span><span class="o">:</span><span class="w"> </span><span class="kt">14</span><span class="w"> </span><span class="p">}}</span>
<span class="w">        </span><span class="p">/&gt;</span>
<span class="w">        </span><span class="p">&lt;</span><span class="nt">button</span><span class="w"> </span><span class="na">onClick</span><span class="o">=</span><span class="p">{</span><span class="nx">sendMessage</span><span class="p">}</span><span class="w"> </span><span class="na">disabled</span><span class="o">=</span><span class="p">{</span><span class="nx">isStreaming</span><span class="p">}</span>
<span class="w">                </span><span class="na">style</span><span class="o">=</span><span class="p">{{</span><span class="w"> </span><span class="nx">padding</span><span class="o">:</span><span class="w"> </span><span class="s1">'8px 16px'</span><span class="p">,</span><span class="w"> </span><span class="nx">background</span><span class="o">:</span><span class="w"> </span><span class="s1">'#3b82f6'</span><span class="p">,</span>
<span class="w">                         </span><span class="nx">color</span><span class="o">:</span><span class="w"> </span><span class="s1">'white'</span><span class="p">,</span><span class="w"> </span><span class="nx">border</span><span class="o">:</span><span class="w"> </span><span class="s1">'none'</span><span class="p">,</span><span class="w"> </span><span class="nx">borderRadius</span><span class="o">:</span><span class="w"> </span><span class="kt">6</span><span class="p">,</span>
<span class="w">                         </span><span class="nx">cursor</span><span class="o">:</span><span class="w"> </span><span class="kt">isStreaming</span><span class="w"> </span><span class="o">?</span><span class="w"> </span><span class="s1">'not-allowed'</span><span class="w"> </span><span class="o">:</span><span class="w"> </span><span class="s1">'pointer'</span><span class="w"> </span><span class="p">}}&gt;</span>
<span class="w">          </span><span class="p">{</span><span class="nx">isStreaming</span><span class="w"> </span><span class="o">?</span><span class="w"> </span><span class="s1">'Running...'</span><span class="w"> </span><span class="o">:</span><span class="w"> </span><span class="s1">'Send'</span><span class="p">}</span>
<span class="w">        </span><span class="p">&lt;/</span><span class="nt">button</span><span class="p">&gt;</span>
<span class="w">      </span><span class="p">&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">    </span><span class="p">&lt;/</span><span class="nt">div</span><span class="p">&gt;</span>
<span class="w">  </span><span class="p">);</span>
<span class="p">}</span>
</code></pre></div>

<div class="admonition example"><p class="admonition-title">该实现展示了什么</p>
<p>上述代码展示了若干关键的 Agent 化 UI 模式协同工作：</p>

<ul>
<li><p><strong>SSE 流式</strong>：Backend 通过单条 HTTP 连接流式传输不同类型的事件（状态、思考、工具调用、Token）。</p></li>
<li><p><strong>类型化事件协议</strong>：事件类型的可辨识联合（discriminated union）使 Frontend 能够对每种事件分别渲染。</p></li>
<li><p><strong>工具可视化</strong>：<code>ToolCard</code> 渲染工具调用，附带层级指示与可展开的输入细节。</p></li>
<li><p><strong>审批门控</strong>：<code>ApprovalGate</code> 阻断流并等待人工输入，Agent 才会继续执行不可逆操作。</p></li>
<li><p><strong>异步审批</strong>：Backend 使用 <code>asyncio.Event</code> 在等待 Frontend 的批准 POST 请求期间暂停流，从而干净地解耦审批 UI 与流式逻辑。</p></li>
</ul>
</div>

## 27.10　小结 {#subsec:ui-summary}

<p>Agent 化 UI 框架代表了人机交互的新前沿，要求我们从第一性原理出发重新思考界面设计。本节的关键洞见包括：</p>

<ol>
<li><p><strong>范式选择至关重要</strong>：合适的 UI 范式（聊天、画布、工作流、仪表盘、协作、自主）取决于任务结构、所需的人类介入与输出类型。大多数生产系统会组合多种范式。</p></li>
<li><p><strong>透明性不可妥协</strong>：用户无法信任他们看不见的东西。思考展示、工具可视化与上下文面板并非可选功能——它们是可信赖 Agent 化系统的基础。</p></li>
<li><p><strong>流式是基线</strong>：用户期望实时看到 Agent 在工作。Token 流、工具调用流与多 Agent 流是入门级必备能力。</p></li>
<li><p><strong>审批门控必须分层</strong>：平坦的审批策略（全批准或全不批准）在实践中会失败。对安全操作自动批准、对危险操作门控的分层策略，能在维持监督的同时避免瓶颈。</p></li>
<li><p><strong>生成式 UI 是前沿</strong>：LLM 不仅能生成文本，还能生成 UI 组件——图表、表单、地图、小部件——这使得界面可以适配内容，而不必把内容硬塞进固定模板。</p></li>
<li><p><strong>信任靠一致性与可恢复性赢得</strong>：在建立用户信任方面，撤销能力、审计轨迹与已校准的置信度指示器与原始能力同等重要。</p></li>
</ol>

<div class="admonition note"><p class="admonition-title">设计原则：作为透明协作者的 Agent</p>
<p>Agent 化 UI 设计的北极星是<em>透明协作者</em>：一个其操作始终可见、推理始终可访问、错误始终可恢复、能力与局限始终明确的 Agent。每一项 UI 决策都应以此为标准来评估。</p>
</div>

<p>本节所描述的框架与模式——Vercel AI SDK、Chainlit、Gradio、Streamlit、LangGraph Studio——提供了构建模块。实践者面临的挑战是依据自身用户的具体需求与所在领域的具体风险，深思熟虑地将它们组合起来。</p>
