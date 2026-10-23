# 第 21 章 · Agent 环境与基准 {#sec:agentic-environments}

## 21.1　动机：为什么 Agent 需要环境 {#sec:env-motivation}

<p>对话式语言模型的评估在原则上是直截了当的：给出一个 Prompt、收集一次回复，并依据参考答案或人工评判进行打分。Agent 评估则根本不同：Agent 必须在世界中<em>行动</em>、观察后果，并在一连串步骤中调整其行为。任何单次回复都无法刻画这一点；只有结构化的<em>环境</em>才能做到。</p>

<p><strong>范围说明。</strong> 我们在强化学习意义上使用<em>环境</em>一词：Agent 为训练或评估而与之交互的世界——而非在线服务时承载 Agent 的生产基础设施（harness、编排器）。执行沙箱出现在此处，是因为它们<em>使</em>这种环境成为可能，但 Agent harness 本身在第 18 章讨论。</p>

<div class="admonition note"><p class="admonition-title">Chatbot 与 Agent 评估的鸿沟</p>
<p><strong>Chatbot 评估</strong>衡量单次生成的质量：流畅度、事实性、有用性。<strong>Agent 评估</strong>衡量 <em>Policy</em> 的质量：Agent 能否在多样化的长时程任务上稳定达成目标？这一鸿沟不仅是量上的——它需要完全不同的基础设施。</p>
</div>

<p>三股力量驱动了对专用 Agent 环境的需求：</p>

##### 安全探索。 {#safe-exploration.}

<p>真实世界系统——生产数据库、在线网站、金融 API——无法承受训练中 Agent 的探索行为。沙箱环境提供一个忠实副本，Agent 可在其中失败、恢复、学习，且不造成不可逆的损害。安全隔离（如 Docker 容器、受限网络的虚拟机）不是可选项，而是一等设计要求。</p>

##### 可复现的评估。 {#reproducible-evaluation.}

<p>基准测试要求每个 Agent 在相同条件下面对相同任务。环境必须按需具备确定性、受版本控制并可分发，以便一个实验室报告的结果能在另一个实验室复现。该性质的缺失历来使 Agent 基准难以比较。</p>

##### 课程学习。 {#curriculum-learning.}

<p>从零开始在困难任务上训练 Agent 在样本效率上很低。提供<em>难度课程</em>的环境——随着 Agent 的进步逐步增加任务复杂度——可显著减少达到目标性能所需的环境交互次数。这与人类学习的方式相似：子技能的掌握先于整体的掌握。</p>

<div class="admonition tip"><p class="admonition-title">把环境视为 LLM 的 RL &quot;Gym&quot;</p>
<p>正如 OpenAI Gym <a class="book-citation" id="cite-brockman2016openai-1" href="../../references.md#ref-brockman2016openai" aria-label="参考文献 31">[31]</a> 标准化了 RL 算法与模拟控制任务之间的接口，Agent 环境标准化了基于 LLM 的 Agent 与其必须解决的各类任务之间的接口。这一类比相当贴切：<code>reset()</code> 初始化新的 Episode，<code>step(action)</code> 推进世界状态并返回观察和奖励，<code>render()</code> 产生当前状态的人类可读视图。</p>
</div>

## 21.2　环境设计原则 {#sec:env-design}

<p>一个设计良好的 Agent 环境暴露出四个正交的设计轴：<em>观察空间</em>、<em>动作空间</em>、<em>奖励信号</em>和<em>Episode 结构</em>。让每一项都正确是必要的；让四者同时正确则是环境工程的手艺所在。</p>

### 21.2.1　观察空间设计 {#observation-space-design}

<p>观察是 Agent 在每一步<em>看到</em>的内容。对于基于 LLM 的 Agent，观察几乎总以文本形式呈现，但其来源材料差异极大：</p>

<ul>
<li><p><strong>纯文本</strong>：终端输出、文件内容、API 响应、错误信息。与任意 LLM 兼容度最高，但丢失空间和视觉结构。</p></li>
<li><p><strong>结构化（JSON/XML）</strong>：机器可读的状态表示。可实现精确的指代落地，但要求 Agent 解析结构而非阅读散文。</p></li>
<li><p><strong>多模态</strong>：截图、可访问性树（accessibility tree）、渲染后的 HTML。GUI 和 Web 任务必备；需要具视觉能力的模型或独立的感知模块。</p></li>
<li><p><strong>混合</strong>：截图配合可访问性树（OSWorld 和 VisualWebArena 中使用），同时提供视觉上下文和结构化元素标识符，融合两种模态的优势。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">观察泄露（Observation Leakage）</p>
<p>一个常见的设计错误是在观察中包含 Agent 本不应访问的信息——例如真值答案、奖励值或未来的任务步骤。观察泄露会虚高基准分数，并产出在真实环境（缺乏这些信息）中部署时灾难性失败的 Agent。</p>
</div>

### 21.2.2　动作空间设计 {#action-space-design-dup2}

<p>动作空间定义了 Agent 可以<em>做</em>什么。对 LLM Agent 而言，动作通常是一段文本字符串，由环境解析并执行。常见动作类型包括：</p>

<ul>
<li><p><strong>工具调用</strong>：对外部函数（搜索、计算器、日历）的结构化调用。常以 JSON 或 XML 函数调用语法格式化。</p></li>
<li><p><strong>代码执行</strong>：Agent 写出的代码在沙箱中运行；stdout/stderr 作为下一次观察返回。这是最具表达力的动作类型。</p></li>
<li><p><strong>API 交互</strong>：对 Web 服务的 HTTP 请求、数据库查询、Shell 命令。</p></li>
<li><p><strong>GUI 动作</strong>：<code>click(x,y)</code>、<code>type(&quot;text&quot;)</code>、<code>scroll(direction)</code>、<code>key(&quot;Enter&quot;)</code>。用于计算机使用类环境。</p></li>
<li><p><strong>自然语言</strong>：发送给另一个 Agent、人类或子任务规划器的自由文本。</p></li>
</ul>

### 21.2.3　奖励信号设计 {#reward-signal-design}

<p>奖励设计是环境工程中最难的部分。奖励必须：</p>

<ol>
<li><p><strong>对齐</strong>：高奖励应对应真正的任务完成，而非表面代理指标。</p></li>
<li><p><strong>可学习</strong>：信号必须足够稠密，以使 Agent 能够取得进展；长时程任务上的纯稀疏奖励若无额外塑形（shaping）通常无法学习。</p></li>
<li><p><strong>抗作弊</strong>：Agent 不应能够在未真正完成任务的情况下获得高奖励（reward hacking，奖励欺骗）。</p></li>
</ol>

<p class="table-caption" id="table-21-1"><span class="table-number">表 21.1　</span>Agent 环境中各类奖励信号及其权衡。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>奖励类型</strong></th><th scope="col"><strong>优点</strong></th><th scope="col"><strong>缺点</strong></th></tr></thead>
<tbody><tr><td>稀疏（结束时 0/1）</td><td>对齐性好，难以作弊</td><td>难以学习</td></tr>
<tr><td>稠密（步级）</td><td>易于学习</td><td>易产生塑形伪影</td></tr>
<tr><td>内在（好奇心）</td><td>驱动探索</td><td>可能偏离任务</td></tr>
<tr><td>LLM-as-judge</td><td>灵活、细腻</td><td>昂贵、不一致</td></tr>
<tr><td>基于执行</td><td>真值依据</td><td>仅适用于可验证任务</td></tr></tbody>
</table>
</div>

### 21.2.4　Episode 结构 {#episode-structure}

<p>Episode 可以采用若干结构：</p>

<ul>
<li><p><strong>定长</strong>：Agent 恰好走 <span class="arithmatex">\(T\)</span> 步。实现简单；在已解决的任务上浪费算力。</p></li>
<li><p><strong>提前终止</strong>：当 Agent 标示完成或达到终止状态时 Episode 结束。更高效，但需要可靠的终止检测器。</p></li>
<li><p><strong>开放式</strong>：无固定时程；Agent 持续运行直到资源预算（Token、API 调用次数、墙钟时间）耗尽。最接近真实部署，但最难评估。</p></li>
</ul>

##### 自适应 Episode 长度与提前终止。 {#adaptive-episode-length-and-early-termination.}

<p>近期工作挑战了“Episode 长度必须在训练前固定”的假设：</p>

<ul>
<li><p><strong>时程课程。</strong> AELA <a class="book-citation" id="cite-yoo2025aela-1" href="../../references.md#ref-yoo2025aela" aria-label="参考文献 411">[411]</a> 从短 Episode 开始，并随 Agent 能力的提升（以 Policy 熵的收敛来度量）逐渐扩展时程。前期的短 Episode 每个训练样本能暴露更多样化的初始状态。</p></li>
<li><p><strong>截断作为 RL 惩罚。</strong> DLER <a class="book-citation" id="cite-liu2025dler-1" href="../../references.md#ref-liu2025dler" aria-label="参考文献 235">[235]</a> 表明，对于推理模型，最简单的长度控制——硬截断——只要配合批级别的奖励归一化和动态采样，就能很好地工作，从而避免被截断的 rollout 丢失奖励信号。</p></li>
<li><p><strong>学习停止。</strong> 模型本身可以学习何时停止推理，而非依赖固定预算。<a class="book-citation" id="cite-liu2025answerstop-1" href="../../references.md#ref-liu2025answerstop" aria-label="参考文献 227">[227]</a> 提出三种策略：当连续推理步骤收敛到同一答案时停止；提升“思考结束” Token 的概率；或在隐状态激活上训练一个轻量分类器以预测最优停止点。</p></li>
<li><p><strong>部分 rollout 回收。</strong> APRIL <a class="book-citation" id="cite-april2025-1" href="../../references.md#ref-april2025" aria-label="参考文献 251">[251]</a> 超额发起 rollout 请求，一旦达到目标 Batch 数即终止；未完成的回复被回收作为后续步骤的热启动前缀，从而消除了少数慢样本阻塞整个 Batch 的长尾停顿（吞吐提升 20–35%）。TLT <a class="book-citation" id="cite-hu2025tlt-1" href="../../references.md#ref-hu2025tlt" aria-label="参考文献 149">[149]</a> 针对同一瓶颈，通过即时训练一个自适应草稿模型来对掉队样本进行投机解码（端到端加速 1.7<span class="arithmatex">\(\times\)</span>，无损）。</p></li>
</ul>

### 21.2.5　难度课程与自适应环境 {#difficulty-curriculum-and-adaptive-environments}

<p>静态基准测量 Agent 能力的一个固定快照。自适应环境更进一步：在线监控 Agent 表现并调整任务难度，使 Agent 保持在“最近发展区”——难到足以从中学习，易到偶尔能成功。相关技术包括：</p>

<ul>
<li><p><strong>程序化生成</strong>：从参数化分布中采样任务；根据近期成功率调整难度参数。Prioritized Level Replay <a class="book-citation" id="cite-jiang2021plr-1" href="../../references.md#ref-jiang2021plr" aria-label="参考文献 167">[167]</a> 通过估计的学习潜力（如 GAE 幅值）对每个生成关卡打分，并更频繁地回放高价值关卡。</p></li>
<li><p><strong>自博弈 / 对抗式环境设计</strong>：PAIRED <a class="book-citation" id="cite-dennis2020paired-1" href="../../references.md#ref-dennis2020paired" aria-label="参考文献 74">[74]</a> 训练一个对手提出能最大化“主角”与“反派” Agent 之间<em>悔值（regret）</em>的环境，从而无需手工设计难度时间表即可产生复杂度逐步上升的自然课程。</p></li>
<li><p><strong>后见之明重标注</strong>：用 Agent <em>实际</em>达到的目标重新标注失败轨迹，使失败也能提供学习信号（Hindsight Experience Replay, HER） <a class="book-citation" id="cite-andrychowicz2017hindsight-1" href="../../references.md#ref-andrychowicz2017hindsight" aria-label="参考文献 8">[8]</a>。</p></li>
<li><p><strong>面向 LLM 的难度定向数据筛选</strong>：在 RLVR 训练中，并非所有问题都提供等量信号。近期工作优先选择中等难度的问题——即模型成功率大致在 30–70% 之间的题目——因为它们能提供最高的梯度信息量 <a class="book-citation" id="cite-wang2025dataefficiency-1" href="../../references.md#ref-wang2025dataefficiency" aria-label="参考文献 370">[370]</a>。ADCL <a class="book-citation" id="cite-liu2025adcl-1" href="../../references.md#ref-liu2025adcl" aria-label="参考文献 230">[230]</a> 随着模型提升而周期性地重新估计难度，避免课程过时。</p></li>
</ul>

## 21.3　Agent 环境的类型 {#sec:env-types}

### 21.3.1　代码执行沙箱 {#code-execution-sandboxes}

<p>对 LLM 而言，最基础的 Agent 环境就是代码执行沙箱：Agent 写代码，沙箱运行，返回输出。这一简单循环支撑了出人意料比例的真实 Agent 部署。</p>

<p><strong>基于 Docker 的隔离</strong>是最常见的做法。每个 Episode 从已知镜像启动一个全新容器，在其中执行 Agent 代码，并在 Episode 结束时销毁该容器。网络访问、文件系统写入、进程派生均可在容器级别加以控制。</p>

<p><strong>E2B</strong>（Environments to Benchmarks）提供一个托管云沙箱 API：Agent 通过 HTTP 发送代码，E2B 在隔离的 Firecracker microVM 中执行（启动时间不到 200 毫秒）并返回 stdout/stderr。E2B 处理容器生命周期管理的基础设施复杂性，便于集成进 Agent 训练循环。</p>

<p><strong>Modal</strong> 提供类似的托管执行模型，但 GPU 支持更强，适合需要在任务中运行 ML 工作负载的 Agent。</p>

<div class="admonition warning"><p class="admonition-title">沙箱逃逸与安全</p>
<p>代码执行沙箱是主要的攻击面。一个能力足够强的 Agent（或被 Prompt 注入的恶意载荷）可能尝试通过内核漏洞、网络外渗或资源耗尽来逃逸沙箱。纵深防御不可或缺：结合使用容器隔离、seccomp 配置、只读根文件系统、网络出站过滤以及 CPU/内存 cgroups。绝不要以宿主级别权限运行 Agent 生成的代码。</p>
</div>

### 21.3.2　Web 环境 {#web-environments}

<p>Web 环境向 Agent 提供一个浏览器，并要求其在真实或模拟网站上完成任务。</p>

<p><strong>WebArena</strong> <a class="book-citation" id="cite-zhou2024webarena-1" href="../../references.md#ref-zhou2024webarena" aria-label="参考文献 440">[440]</a> 提供一个自托管测试平台，包含四个功能性 Web 应用——电商商城、社交论坛、GitLab 实例和 CMS——外加一个地图服务，共计 812 个长时程任务。Agent 通过浏览器自动化 API 进行交互；任务涉及多步导航、表单填写和信息检索。人类性能约 78%；最先进的 LLM Agent 仅在 35–45% 左右。</p>

<p><strong>VisualWebArena</strong> <a class="book-citation" id="cite-koh2024visualwebarena-1" href="../../references.md#ref-koh2024visualwebarena" aria-label="参考文献 188">[188]</a> 在 WebArena 基础上扩展，加入需要解读网页图像的视觉落地任务。观察为截图配合可访问性树；Agent 必须在两种模态中落地其动作。</p>

<p><strong>Mind2Web</strong> <a class="book-citation" id="cite-deng2024mind2web-1" href="../../references.md#ref-deng2024mind2web" aria-label="参考文献 73">[73]</a> 是一个大规模数据集，涵盖 137 个真实网站上的 2,000 个任务，通过人类示范收集。与 WebArena 不同，Mind2Web 聚焦于对未见网站的泛化，是更困难的分布外测试。</p>

<div class="admonition example"><p class="admonition-title">WebArena 任务示例</p>
<p><strong>任务</strong>：“在电商网站上找到价格低于 50 美元的最便宜的红色连衣裙并将其加入购物车。”</p>

<p><strong>Agent 轨迹</strong>：</p>

<ol>
<li><p>导航至服装类目。</p></li>
<li><p>应用颜色筛选：红色。</p></li>
<li><p>按价格升序排序。</p></li>
<li><p>识别第一个低于 50 美元的商品。</p></li>
<li><p>点击“加入购物车”。</p></li>
<li><p>核实购物车内容。</p></li>
</ol>

<p>环境对照真值商品检查最终购物车状态；若正确奖励为 1，否则为 0。</p>
</div>

### 21.3.3　计算机使用环境 {#computer-use-environments}

<p>计算机使用环境（Computer Use）让 Agent 控制一个完整的桌面操作系统，通过截图和/或可访问性 API 进行观察。</p>

<p><strong>OSWorld</strong> <a class="book-citation" id="cite-xie2024osworld-1" href="../../references.md#ref-xie2024osworld" aria-label="参考文献 397">[397]</a> 跨三种操作系统（Ubuntu、Windows、macOS）测试桌面自动化，涵盖 369 个任务，覆盖各类生产力应用（LibreOffice、VS Code、Chrome、GIMP 等）。Agent 通过截图观察，并以 <code>pyautogui</code> 风格的鼠标键盘命令行动。人-Agent 差距相当悬殊：标注者在约 72% 的任务上成功，而最强 LLM Agent 仅能达到 <span class="arithmatex">\(\sim\)</span>18%，凸显了像素级 GUI 控制的难度。</p>

<p><strong>WindowsAgentArena</strong> <a class="book-citation" id="cite-bonatti2024windows-1" href="../../references.md#ref-bonatti2024windows" aria-label="参考文献 27">[27]</a> 专门聚焦于 Windows 11，包含 19 个应用上的 154 个任务，强调企业工作流：Excel 公式、PowerPoint 编辑、Outlook 邮件管理。</p>

<div class="admonition note"><p class="admonition-title">截图瓶颈</p>
<p>计算机使用类 Agent 面临一个根本挑战：截图维度极高（通常 <span class="arithmatex">\(1920 \times 1080 \times 3\)</span> 像素），但其中大部分信息与当前动作无关。高效的 Agent 学会关注屏幕的小区域、使用可访问性树通过名称（而非像素坐标）识别可交互元素，并对此前访问过的 UI 状态维护一个紧凑的工作记忆。</p>
</div>

### 21.3.4　软件工程环境 {#software-engineering-environments}

<p>软件工程（Software Engineering, SWE）环境要求 Agent 解决真实世界的编程任务：修 bug、实现功能、写测试。</p>

<p><strong>SWE-bench</strong> <a class="book-citation" id="cite-jimenez2024swebench-1" href="../../references.md#ref-jimenez2024swebench" aria-label="参考文献 169">[169]</a> 取材自 12 个广泛使用的 Python 项目（Django、Flask、scikit-learn 等）的 2,294 个真实 pull request。每个实例将一段 issue 描述与一份预留的测试套件配对，仅在应用正确补丁后测试才通过。Agent 必须理解仓库结构、定位相关代码、实施修复并用测试套件加以验证。<strong>SWE-bench Verified</strong> 子集（500 个 issue）经过人工正确性校验，是标准评估目标。</p>

<p><strong>SWE-agent</strong> <a class="book-citation" id="cite-yang2024sweagent-1" href="../../references.md#ref-yang2024sweagent" aria-label="参考文献 404">[404]</a> 既是基准环境也是 Agent 框架。它引入了<em>Agent-计算机接口（Agent-Computer Interface, ACI）</em>：一组为 LLM Agent 优化的 Shell 命令（如 <code>search_file</code>、<code>open</code>、<code>edit</code>），相比原始 bash 降低了动作空间复杂度。</p>

<div class="admonition example"><p class="admonition-title">SWE-bench 工作流</p>
<p><strong>输入</strong>：一段 GitHub issue 描述，以及该 issue 提交时刻的完整仓库快照。</p>

<p><strong>Agent 动作</strong>：<code>find_file</code>、<code>view</code>、<code>edit</code>、<code>python -m pytest tests/</code>。</p>

<p><strong>奖励</strong>：Agent 补丁应用后若所有目标测试通过则为 1，否则为 0。不计部分得分。</p>
</div>

### 21.3.5　科研环境 {#scientific-research-environments}

<p>科研环境推动 Agent 走向自主知识生成：阅读论文、形成假设、设计实验、解读结果。</p>

<p><strong>PaperQA2</strong> <a class="book-citation" id="cite-lala2023paperqa-1" href="../../references.md#ref-lala2023paperqa" aria-label="参考文献 245">[245]</a> 是一个检索增强 Agent，通过搜索 PDF 语料库、抽取相关段落、综合出带引用的答案来回答科学问题。它既是文献依据型推理的工具，也是其基准。</p>

<p><strong>AI Scientist</strong> <a class="book-citation" id="cite-lu2024aiscientist-1" href="../../references.md#ref-lu2024aiscientist" aria-label="参考文献 240">[240]</a> 是一个端到端的科研自动化系统：给定一个研究方向，Agent 即生成假设、撰写并运行实验、解读结果、产出论文初稿。该环境包含一个 Python 执行沙箱、文献搜索 API 和 LaTeX 编译器。</p>

<p><strong>MLAgentBench</strong> <a class="book-citation" id="cite-huang2024mlagentbench-1" href="../../references.md#ref-huang2024mlagentbench" aria-label="参考文献 151">[151]</a> 在机器学习工程任务上评估 Agent：在算力预算内提升给定数据集上的模型精度。Agent 可读取数据、编写训练脚本、运行实验并迭代。</p>

### 21.3.6　游戏与仿真环境 {#game-and-simulation-environments}

<p>游戏提供了丰富的长时程环境，具有定义良好的奖励信号且无真实世界后果。</p>

<p><strong>NetHack</strong> <a class="book-citation" id="cite-kuttler2020nethack-1" href="../../references.md#ref-kuttler2020nethack" aria-label="参考文献 196">[196]</a> 是一款程序化生成的 roguelike 游戏，状态空间巨大，要求长期规划、物品管理以及对意外事件的适应。NetHack Learning Environment（NLE）提供了 Gym 兼容接口。</p>

<p><strong>Voyager / Minecraft</strong> <a class="book-citation" id="cite-wang2023voyager-1" href="../../references.md#ref-wang2023voyager" aria-label="参考文献 362">[362]</a> 把 Minecraft 引擎用作一个开放式环境。Voyager 引入了难度逐级上升的任务课程（采集木材 <span class="arithmatex">\(\to\)</span> 制造工具 <span class="arithmatex">\(\to\)</span> 建造庇护所 <span class="arithmatex">\(\to\)</span> 探索下界）以及一个跨 Episode 累积可复用代码片段的技能库。</p>

<p><strong>GAIA</strong> <a class="book-citation" id="cite-mialon2023gaia-1" href="../../references.md#ref-mialon2023gaia" aria-label="参考文献 254">[254]</a> 提出 466 个问题，要求链式工具使用——Web 搜索、代码执行、文件解析——并按所需推理步数划分为三个难度等级。该基准鲜明地暴露出人类能力（约 92% 准确率）与当前 LLM Agent（GPT-4 配合插件发布初期约 15%，后续系统约 30%）之间的鸿沟。</p>

### 21.3.7　多 Agent 环境 {#multi-agent-environments}

<p>多 Agent 环境涉及两个或更多 LLM Agent 之间以及与共享世界的交互。</p>

<ul>
<li><p><strong>协商（Negotiation）</strong>：具有私有效用函数的 Agent 必须通过对话达成交易。经典环境包括 DealOrNoDeal <a class="book-citation" id="cite-lewis2017dealornodeal-1" href="../../references.md#ref-lewis2017dealornodeal" aria-label="参考文献 206">[206]</a> 和 CaSiNo <a class="book-citation" id="cite-chawla2021casino-1" href="../../references.md#ref-chawla2021casino" aria-label="参考文献 42">[42]</a>。</p></li>
<li><p><strong>辩论（Debate）</strong>：两个 Agent 持相对立场进行辩论；由裁判 Agent（或人类）评判论证质量。用于通过对抗压力引出真实的推理。</p></li>
<li><p><strong>协作式任务完成</strong>：具有互补能力（规划者、执行者、评论者）的 Agent 必须协作完成任一单独 Agent 无法解决的任务。相关框架包括 AutoGen <a class="book-citation" id="cite-wu2023autogen-1" href="../../references.md#ref-wu2023autogen" aria-label="参考文献 385">[385]</a>、CrewAI <a class="book-citation" id="cite-moura2023crewai-1" href="../../references.md#ref-moura2023crewai" aria-label="参考文献 261">[261]</a> 和 MetaGPT <a class="book-citation" id="cite-hong2023metagpt-1" href="../../references.md#ref-hong2023metagpt" aria-label="参考文献 142">[142]</a>。</p></li>
<li><p><strong>竞争性游戏</strong>：Agent 在零和博弈（国际象棋、围棋、扑克）中对弈，对手本身就是另一个 LLM Agent。在这类环境中的自博弈已经在狭窄领域内产生了超人类的表现。</p></li>
</ul>

## 21.4　OpenEnv：标准化的 Agent 环境接口 {#sec:openenv}

<p>Agent 环境的激增带来了碎片化问题：每个环境都暴露不同的 API、使用不同的观察格式、需要不同的脚手架。<strong>OpenEnv</strong> <a class="book-citation" id="cite-huggingface2025openenv-1" href="../../references.md#ref-huggingface2025openenv" aria-label="参考文献 88">[88]</a> 是 Hugging Face 最近推出的开源框架，直接针对这一问题：它为 Agent 执行环境提供 Gymnasium 风格 <a class="book-citation" id="cite-towers2024gymnasium-1" href="../../references.md#ref-towers2024gymnasium" aria-label="参考文献 355">[355]</a> 的接口（<code>step()</code>、<code>reset()</code>、<code>state()</code>），以基于 Docker 的隔离部署通过 WebSocket 通信。OpenEnv 与更广泛的标准化努力互补，如 AgentGym <a class="book-citation" id="cite-xi2024agentgym-1" href="../../references.md#ref-xi2024agentgym" aria-label="参考文献 390">[390]</a>（为 LLM Agent 跨多种环境提供统一格式平台）和 BrowserGym <a class="book-citation" id="cite-drouin2024browsergym-1" href="../../references.md#ref-drouin2024browsergym" aria-label="参考文献 201">[201]</a>（标准化 Web Agent 基准的观察和动作空间）。下文的设计原则概括了这些项目共同收敛出的最佳实践。</p>

<figure id="fig:openenv-arch"><a href="../../assets/figures/fig_064_openenv-arch.png" target="_blank" rel="noopener" aria-label="查看图 21.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_064_openenv-arch.png" alt="OpenEnv 架构与一个 LLM Agent。Agent 通过 harness 循环进行推理，并调用类型化的 EnvClient。客户端通过 WebSocket 与运行在 Docker 容器内的 HTTPEnvServer 通信。RL 训练器（虚线）可选地包裹该循环，以采集 rollout 和奖励信号用于 Policy 优化。" loading="lazy"></a><figcaption><span class="figure-number">图 21.1　</span>OpenEnv 架构与一个 LLM Agent。Agent 通过 harness 循环进行推理，并调用类型化的 <code>EnvClient</code>。客户端通过 WebSocket 与运行在 Docker 容器内的 <code>HTTPEnvServer</code> 通信。RL 训练器（虚线）可选地包裹该循环，以采集 rollout 和奖励信号用于 Policy 优化。</figcaption></figure>

### 21.4.1　标准化的 Agent–环境接口 {#standardized-agentenvironment-interface}

<p>OpenEnv 为 Agent 执行环境定义了一个类型化接口。其设计沿袭了 Gymnasium 的简洁，但面向通过 HTTP/WebSocket 与工具交互的 LLM Agent：</p>

<ul>
<li><p><code>env.reset()</code> <span class="arithmatex">\(\to\)</span> <code>StepResult</code>：开始一个新 Episode；返回初始观察。</p></li>
<li><p><code>env.step(action)</code> <span class="arithmatex">\(\to\)</span> <code>StepResult(observation, reward, done)</code>：执行一个动作并返回结果观察、标量奖励和终止标志。</p></li>
<li><p><code>env.state()</code> <span class="arithmatex">\(\to\)</span> 当前环境状态（Episode ID、步数、特定环境的字段）。</p></li>
<li><p><code>env.close()</code>：释放资源（停止容器、关闭连接）。</p></li>
</ul>

<p>动作和观察是强类型的 Python dataclass，针对每个环境特定。例如，编码环境定义 <code>CodeAction(code=...)</code> 并返回带 <code>stdout</code>、<code>stderr</code> 和 <code>exit_code</code> 的观察；游戏环境则定义其自己的动作/观察类型。这种逐环境的类型化使 Agent 获得结构化、可预测的接口，同时保持三个核心方法（<code>reset</code>、<code>step</code>、<code>state</code>）的通用性。</p>

##### 架构。 {#architecture.-dup3}

<p>每个环境是一个继承自 <code>Environment</code> 的 Python 类（实现 <code>reset()</code> 和 <code>step()</code>）。它在 Docker 容器内通过 <code>HTTPEnvServer</code> 提供服务，后者暴露一个 FastAPI/WebSocket 端点。客户端使用 <code>EnvClient</code> 的特定环境子类，处理序列化和连接生命周期。容器可通过 <code>from_docker_image()</code> 本地启动，或通过 base URL 远程连接：</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">coding_env</span><span class="w"> </span><span class="kn">import</span> <span class="n">CodeAction</span><span class="p">,</span> <span class="n">CodingEnv</span>

<span class="c1"># 选项 1：启动本地 Docker 容器</span>
<span class="n">client</span> <span class="o">=</span> <span class="n">CodingEnv</span><span class="o">.</span><span class="n">from_docker_image</span><span class="p">(</span><span class="s2">"coding-env:latest"</span><span class="p">)</span>

<span class="c1"># 选项 2：连接到远端部署</span>
<span class="c1"># client = CodingEnv(base_url="http://localhost:8000")</span>

<span class="c1"># 与环境交互</span>
<span class="n">result</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">reset</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">observation</span><span class="o">.</span><span class="n">stdout</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">observation</span><span class="o">.</span><span class="n">stderr</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">observation</span><span class="o">.</span><span class="n">exit_code</span><span class="p">)</span>

<span class="n">result</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">CodeAction</span><span class="p">(</span><span class="n">code</span><span class="o">=</span><span class="s2">"print(2 + 2)"</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">observation</span><span class="o">.</span><span class="n">stdout</span><span class="p">)</span>       <span class="c1"># "4\n"</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">observation</span><span class="o">.</span><span class="n">exit_code</span><span class="p">)</span>    <span class="c1"># 0</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">reward</span><span class="p">,</span> <span class="n">result</span><span class="o">.</span><span class="n">done</span><span class="p">)</span>

<span class="c1"># 查看状态</span>
<span class="n">state</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">state</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">state</span><span class="o">.</span><span class="n">episode_id</span><span class="p">,</span> <span class="n">state</span><span class="o">.</span><span class="n">step_count</span><span class="p">)</span>

<span class="n">client</span><span class="o">.</span><span class="n">close</span><span class="p">()</span>
</code></pre></div>

##### 环境即服务器。 {#environment-as-a-server.}

<p>创建新环境只需实现 <code>Environment</code> 基类：</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">openenv.core.env_server</span><span class="w"> </span><span class="kn">import</span> <span class="n">Environment</span><span class="p">,</span> <span class="n">create_app</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">dataclasses</span><span class="w"> </span><span class="kn">import</span> <span class="n">dataclass</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">MyAction</span><span class="p">:</span>
    <span class="n">text</span><span class="p">:</span> <span class="nb">str</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">MyObservation</span><span class="p">:</span>
    <span class="n">response</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">reward</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.0</span>
    <span class="n">done</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>

<span class="k">class</span><span class="w"> </span><span class="nc">MyEnvironment</span><span class="p">(</span><span class="n">Environment</span><span class="p">):</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">reset</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">MyObservation</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">MyObservation</span><span class="p">(</span><span class="n">response</span><span class="o">=</span><span class="s2">"Ready"</span><span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">step</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">action</span><span class="p">:</span> <span class="n">MyAction</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">MyObservation</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">MyObservation</span><span class="p">(</span><span class="n">response</span><span class="o">=</span><span class="sa">f</span><span class="s2">"Echo: </span><span class="si">{</span><span class="n">action</span><span class="o">.</span><span class="n">text</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span>
                             <span class="n">reward</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">done</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>

<span class="n">app</span> <span class="o">=</span> <span class="n">create_app</span><span class="p">(</span><span class="n">MyEnvironment</span><span class="p">(),</span> <span class="n">MyAction</span><span class="p">,</span> <span class="n">MyObservation</span><span class="p">)</span>
<span class="c1"># 运行：uvicorn module:app --host 0.0.0.0 --port 8000</span>
</code></pre></div>

##### harness 集成（实验性）。 {#harness-integration-experimental.}

<p>RFC 0054 引入了一个面向 harness 的层，RL 训练框架通过 MCP 风格的工具调用与环境交互。<code>build_harness_rollout_func()</code> 辅助函数可产生一个 TRL 兼容的 rollout 函数，将 OpenEnv 直接桥接进 TorchForge <a class="book-citation" id="cite-meta2025torchforge-1" href="../../references.md#ref-meta2025torchforge" aria-label="参考文献 348">[348]</a> 等现有训练流水线。</p>

##### 治理。 {#governance.}

<p>OpenEnv 由一个包含 Meta-PyTorch、NVIDIA、Unsloth、Modal、Prime Intellect、Reflection 和 Hugging Face 的技术委员会公开治理——确保该标准在广泛的产业输入下演进，而非任由单一厂商的议程主导。</p>

### 21.4.2　环境注册与发现 {#environment-registries-and-discovery}

<p>OpenEnv 环境可部署为 Hugging Face Spaces 或本地 Docker 镜像，无需手动安装即可发现和使用。无论部署目标如何，客户端接口都保持一致：</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">echo_env</span><span class="w"> </span><span class="kn">import</span> <span class="n">EchoAction</span><span class="p">,</span> <span class="n">EchoEnv</span>

<span class="c1"># 连接到远端 HF Space 部署</span>
<span class="n">client</span> <span class="o">=</span> <span class="n">EchoEnv</span><span class="p">(</span><span class="n">base_url</span><span class="o">=</span><span class="s2">"https://openenv-echo-env.hf.space"</span><span class="p">)</span>
<span class="n">result</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">reset</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">observation</span><span class="o">.</span><span class="n">echoed_message</span><span class="p">)</span>  <span class="c1"># "Echo environment ready!"</span>

<span class="n">result</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">EchoAction</span><span class="p">(</span><span class="n">message</span><span class="o">=</span><span class="s2">"Hello!"</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">observation</span><span class="o">.</span><span class="n">echoed_message</span><span class="p">)</span>  <span class="c1"># "Hello!"</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">reward</span><span class="p">)</span>
<span class="n">client</span><span class="o">.</span><span class="n">close</span><span class="p">()</span>
</code></pre></div>

<p>OpenEnv 生态系统已覆盖 70 多个环境（OpenSpiel 游戏、Atari、BrowserGym、编码沙箱、金融 RL、交通仿真等）。RFC 0025 提出一个正式的<em>工具发现</em>协议，使 Agent 能在运行时查询陌生环境接受哪些动作。</p>

### 21.4.3　组合式环境 {#compositional-environments}

<p>真实的 Agent 部署很少只使用单一工具。OpenEnv 支持通过类型化动作暴露多种能力的丰富环境。例如，编码环境在单一沙箱会话内支持代码执行、文件 I/O 和 Shell 命令：</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">coding_env</span><span class="w"> </span><span class="kn">import</span> <span class="n">CodeAction</span><span class="p">,</span> <span class="n">CodingEnv</span>

<span class="n">client</span> <span class="o">=</span> <span class="n">CodingEnv</span><span class="o">.</span><span class="n">from_docker_image</span><span class="p">(</span><span class="s2">"coding-env:latest"</span><span class="p">)</span>
<span class="n">result</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">reset</span><span class="p">()</span>

<span class="c1"># 执行代码</span>
<span class="n">result</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">CodeAction</span><span class="p">(</span><span class="n">code</span><span class="o">=</span><span class="s2">"x = 42</span><span class="se">\n</span><span class="s2">print(x)"</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">observation</span><span class="o">.</span><span class="n">stdout</span><span class="p">)</span>   <span class="c1"># "42"</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">observation</span><span class="o">.</span><span class="n">exit_code</span><span class="p">)</span>  <span class="c1"># 0</span>

<span class="c1"># 状态在同一 Episode 内跨步骤保持</span>
<span class="n">result</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">CodeAction</span><span class="p">(</span><span class="n">code</span><span class="o">=</span><span class="s2">"print(x + 1)"</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">observation</span><span class="o">.</span><span class="n">stdout</span><span class="p">)</span>   <span class="c1"># "43"</span>

<span class="n">state</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">state</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">state</span><span class="o">.</span><span class="n">step_count</span><span class="p">)</span>  <span class="c1"># 2</span>

<span class="n">client</span><span class="o">.</span><span class="n">close</span><span class="p">()</span>
</code></pre></div>

<p>对于需要多样化工具访问（代码 + Web + 文件）的 Agent，OpenEnv 的 RFC 0036 提出 MCP（模型上下文协议，Model Context Protocol）集成，允许任何兼容 MCP 的工具服务器被封装为一个 OpenEnv 环境。此外，<code>openenv</code> CLI 可通过单条命令脚手架化、构建并部署新环境到 Hugging Face Spaces。</p>

### 21.4.4　环境版本控制与可复现性 {#environment-versioning-and-reproducibility}

<p>基准的可信度要求环境行为在评估时被冻结。最佳实践包括：</p>

<ul>
<li><p><strong>语义化版本</strong>：<code>WebArena-v1.2.0</code> 保证在同一次要版本号内向后兼容。</p></li>
<li><p><strong>Docker 镜像固定</strong>：环境运行时被打包为带内容寻址哈希的 Docker 镜像。</p></li>
<li><p><strong>基于种子的确定性</strong>：所有随机元素（程序化生成、网络响应）都被设种子并记录，使任意轨迹都能精确重放。</p></li>
<li><p><strong>排行榜快照</strong>：公开排行榜在分数旁记录环境版本，防止基准悄然漂移。</p></li>
</ul>

## 21.5　构建自定义环境 {#sec:custom-env}

### 21.5.1　面向 LLM Agent 的 Gymnasium 风格 API {#gymnasium-style-api-for-llm-agents}

<p>Gymnasium API <a class="book-citation" id="cite-towers2024gymnasium-1--section-05" href="../../references.md#ref-towers2024gymnasium" aria-label="参考文献 355">[355]</a>（OpenAI Gym 的继任者）是 RL 环境的事实标准。将其适配到 LLM Agent 需要两处修改：(1) 观察和动作是字符串（或包含字符串的字典）而非数值数组；(2) <code>step</code> 方法必须处理异步工具执行。</p>

### 21.5.2　奖励函数工程 {#reward-function-engineering}

<p>LLM Agent 环境的奖励函数通常是<em>基于执行</em>的：环境在每个 Episode 后运行一个验证器，若任务被解决则返回 1，否则返回 0。对于没有明确验证器的任务，选项包括：</p>

<ul>
<li><p><strong>LLM-as-judge</strong>：由一个独立的 LLM 根据任务描述对 Agent 的最终状态打分。</p></li>
<li><p><strong>基于评分量表（Rubric）</strong>：结构化的评分量表将任务分解为若干子标准，每项独立打分。</p></li>
<li><p><strong>人工标注</strong>：由人类评估者对一组随机抽样的轨迹打分；这些分数用于校准一个自动化代理指标。</p></li>
</ul>

### 21.5.3　状态管理与检查点 {#state-management-and-checkpointing}

<p>长时程任务可能需要数小时的墙钟时间。环境应支持：</p>

<ul>
<li><p><strong>状态序列化</strong>：完整的环境状态（文件系统、浏览器 cookies、数据库内容）可序列化到磁盘并恢复。</p></li>
<li><p><strong>Episode 中检查点</strong>：Agent 可在任意步骤保存检查点并从中恢复，从而支持树搜索式探索。</p></li>
<li><p><strong>轨迹日志</strong>：每个观察、动作和奖励都被记录到结构化文件中，供离线分析和奖励模型训练使用。</p></li>
</ul>

### 21.5.4　用于训练数据采集的并行化 {#parallelization-for-training-data-collection}

<p>通过 RL 训练 LLM Agent 需要数百万次环境交互。并行化策略包括：</p>

<ul>
<li><p><strong>进程级并行</strong>：派生 <span class="arithmatex">\(N\)</span> 个独立的环境进程；并行采集轨迹。</p></li>
<li><p><strong>异步 rollout worker</strong>：使用异步事件循环（如 <code>asyncio</code>）将 LLM 推理延迟与环境执行重叠。</p></li>
<li><p><strong>向量化环境</strong>：将多个环境批处理到单次 <code>step</code> 调用中，摊销 Python 开销。</p></li>
<li><p><strong>云原生扩展</strong>：使用作业调度器（Ray、SLURM）在集群中分发环境 worker，由中心化的回放缓冲（replay buffer）聚合轨迹。</p></li>
</ul>

## 21.6　环境–Agent 接口模式 {#sec:interface-patterns}

<p>图 <a href="#fig:env-agent-interface">21.2</a> 展示了实践中使用的四种主要接口模式。</p>

<figure id="fig:env-agent-interface"><a href="../../assets/figures/fig_065_env-agent-interface.png" target="_blank" rel="noopener" aria-label="查看图 21.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_065_env-agent-interface.png" alt="四种 Agent–环境接口模式。(a) 基于文本是 LLM 最常用的方式；(b) 结构化 JSON 可实现精确解析；(c) 多模态结合截图与可访问性树用于 GUI 任务；(d) 流式接口支持没有离散回合边界的实时交互。" loading="lazy"></a><figcaption><span class="figure-number">图 21.2　</span>四种 Agent–环境接口模式。(a) 基于文本是 LLM 最常用的方式；(b) 结构化 JSON 可实现精确解析；(c) 多模态结合截图与可访问性树用于 GUI 任务；(d) 流式接口支持没有离散回合边界的实时交互。</figcaption></figure>

##### 基于文本的观察/动作。 {#text-based-observationaction.}

<p>Agent 接收字符串观察并产出字符串动作。环境解析该动作（如从 <code>&lt;tool&gt;...&lt;/tool&gt;</code> 块中抽取工具调用）并以字符串形式返回结果。这是兼容性最佳的模式：任何 LLM 无需特殊架构即可参与。</p>

##### 结构化 JSON 观察/动作。 {#structured-json-observationaction.}

<p>观察和动作是符合既定 Schema 的 JSON 对象。这支持严格校验（在执行前拒绝畸形动作）、结构化日志，以及更便利的程序化轨迹分析。代价是 Agent 必须可靠地产出有效 JSON，这要求微调或受约束解码。</p>

##### 多模态（截图 + 可访问性树）。 {#multimodal-screenshot-accessibility-tree.}

<p>用于计算机使用类和 Web 环境。观察为元组 <code>(screenshot: PIL.Image, a11y_tree: dict)</code>。截图提供视觉上下文；可访问性树提供元素标识符，使得动作可不依赖像素级坐标。该混合方式比纯截图控制更鲁棒。</p>

##### 流式 vs. 回合式交互。 {#streaming-vs.-turn-based-interaction.}

<p>当前大多数环境采用回合式模型：Agent 产出完整动作，环境执行后返回下一个观察。流式环境允许 Agent 在观察到达时即接收部分观察（如长时间运行命令的输出），并在流过程中中断或重定向执行。这更接近人类与计算机交互的方式，但对 Agent 架构要求更复杂。</p>

## 21.7　评估 harness 设计 {#sec:eval-harness}

<p>评估 harness 是在一组基准上运行 Agent、采集结果、产出汇总统计的基础设施。优秀的 harness 设计与优秀的环境设计同等重要。</p>

### 21.7.1　确定性 vs. 随机性环境 {#deterministic-vs.-stochastic-environments}

<ul>
<li><p><strong>确定性环境</strong>对相同动作序列产出相同观察序列。易于调试和复现，但可能无法反映真实世界的变异性。</p></li>
<li><p><strong>随机性环境</strong>引入随机性（程序化生成、网络延迟、用户模拟）。每个任务需要多次运行以估计平均性能和置信区间。</p></li>
</ul>

<div class="admonition question"><p class="admonition-title">多少次运行才够？</p>
<p>对于含 <span class="arithmatex">\(N\)</span> 个任务、二元奖励的基准，平均成功率的标准误差为 <span class="arithmatex">\(\sqrt{p(1-p)/N}\)</span>。当 <span class="arithmatex">\(N=500\)</span> 且 <span class="arithmatex">\(p=0.4\)</span> 时，95% 置信区间约为 <span class="arithmatex">\(\pm 4.3\%\)</span>。对随机性环境，需乘以 <span class="arithmatex">\(\sqrt{k}\)</span>，其中 <span class="arithmatex">\(k\)</span> 是每个任务的独立运行次数。随机性基准的常见做法是每任务 3–5 次运行。</p>
</div>

### 21.7.2　留出测试环境 {#held-out-test-environments}

<p>基准的可信度要求在<em>环境</em>级别（而不仅是任务级别）做严格的训练/测试划分。一个在 WebArena 任务上训练过的 Agent 应在训练时未使用的留出任务集合上评估。理想情况下，留出集合应覆盖与训练集不同的网站、任务类型和难度等级。</p>

### 21.7.3　跨环境泛化 {#cross-environment-generalization}

<p>对 Agent 的终极考验是其在某一环境中学到的技能能否迁移到另一环境。跨环境评估协议测量：</p>

<ul>
<li><p><strong>零样本迁移</strong>：在环境 A 上训练，无微调地在环境 B 上测试。</p></li>
<li><p><strong>少样本适配</strong>：评估前提供来自环境 B 的 <span class="arithmatex">\(k\)</span> 个示范。</p></li>
<li><p><strong>持续学习</strong>：依次在环境 A、B、C 上训练；在 C 上训练完成后测量三者上的性能。</p></li>
</ul>

### 21.7.4　人类基线采集 {#human-baseline-collection}

<p>每个基准都应包含人类性能作为参考点。人类基线有三个用途：</p>

<ol>
<li><p>它们为任务难度建立上界。</p></li>
<li><p>它们揭示任务是否根本可解（某些基准任务被证实是模糊的或不可解的）。</p></li>
<li><p>它们为解读 Agent 分数提供校准点（“该 Agent 达到了人类性能的 40%”）。</p></li>
</ol>

<p>人类基线应从具备领域专长的工作者采集（如 SWE-bench 应使用软件工程师而非众包工人），并应包含任务用时测量以便进行效率比较。</p>

## 21.8　代码示例：最小的自定义 LLM Agent 环境 {#sec:code-example}

<div class="admonition example"><p class="admonition-title">用于 LLM Agent 训练的最小自定义环境</p>
<p>以下 Python 类实现了一个文件编辑环境，Agent 必须修改一个 Python 文件以使一个失败的测试通过。它遵循为 LLM Agent 适配过的 Gymnasium API。</p>
</div>

<p class="code-caption"><span class="code-number">代码 21.1　</span>遵循 Gymnasium API 的最小 LLM Agent 环境。</p>
<div class="highlight"><pre><span></span><code><span class="sd">"""</span>
<span class="sd">minimal_env.py  --  用于 LLM Agent 的最小文件编辑环境。</span>

<span class="sd">Agent 接收一个含 bug 的 Python 文件以及一个失败的测试。</span>
<span class="sd">它必须不断编辑该文件直到测试通过。</span>
<span class="sd">奖励：所有测试通过则为 1.0，否则为 0.0。</span>
<span class="sd">"""</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">__future__</span><span class="w"> </span><span class="kn">import</span> <span class="n">annotations</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">subprocess</span><span class="o">,</span><span class="w"> </span><span class="nn">shutil</span><span class="o">,</span><span class="w"> </span><span class="nn">tempfile</span><span class="o">,</span><span class="w"> </span><span class="nn">textwrap</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pathlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">Path</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">dataclasses</span><span class="w"> </span><span class="kn">import</span> <span class="n">dataclass</span><span class="p">,</span> <span class="n">field</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Any</span>

<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># 数据结构</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">StepResult</span><span class="p">:</span>
    <span class="n">observation</span><span class="p">:</span> <span class="nb">str</span>          <span class="c1"># 输入给 LLM 的文本</span>
    <span class="n">reward</span><span class="p">:</span> <span class="nb">float</span>             <span class="c1"># 0.0 或 1.0</span>
    <span class="n">terminated</span><span class="p">:</span> <span class="nb">bool</span>          <span class="c1"># Episode 结束（任务解决或达到最大步数）</span>
    <span class="n">truncated</span><span class="p">:</span> <span class="nb">bool</span>           <span class="c1"># Episode 被截断（预算超出）</span>
    <span class="n">info</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">Any</span><span class="p">]</span> <span class="o">=</span> <span class="n">field</span><span class="p">(</span><span class="n">default_factory</span><span class="o">=</span><span class="nb">dict</span><span class="p">)</span>

<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># 环境</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">FileEditEnv</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    一个 Gymnasium 风格的 LLM 代码修复环境。</span>

<span class="sd">    观察空间 : str  （文件内容 + 测试输出）</span>
<span class="sd">    动作空间 : str  （以下之一：view、edit、run_tests、submit）</span>
<span class="sd">    奖励     : 所有测试通过为 1.0，否则为 0.0</span>
<span class="sd">    """</span>

    <span class="n">MAX_STEPS</span> <span class="o">=</span> <span class="mi">20</span>          <span class="c1"># Episode 硬上限</span>
    <span class="n">TIMEOUT</span>   <span class="o">=</span> <span class="mi">30</span>          <span class="c1"># 每次测试运行的秒数</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">buggy_code</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">test_code</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                 <span class="n">task_description</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">buggy_code</span>       <span class="o">=</span> <span class="n">buggy_code</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">test_code</span>        <span class="o">=</span> <span class="n">test_code</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">task_description</span> <span class="o">=</span> <span class="n">task_description</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="p">:</span> <span class="n">Path</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span> <span class="o">=</span> <span class="mi">0</span>

    <span class="c1"># ------------------------------------------------------------------</span>
    <span class="c1"># 核心 API</span>
    <span class="c1"># ------------------------------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">reset</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">seed</span><span class="p">:</span> <span class="nb">int</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">dict</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""初始化一个新 Episode；返回 (observation, info)。"""</span>
        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="ow">and</span> <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="o">.</span><span class="n">exists</span><span class="p">():</span>
            <span class="n">shutil</span><span class="o">.</span><span class="n">rmtree</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="p">)</span>

        <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span>    <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="n">tempfile</span><span class="o">.</span><span class="n">mkdtemp</span><span class="p">(</span><span class="n">prefix</span><span class="o">=</span><span class="s2">"fileenv_"</span><span class="p">))</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span> <span class="o">=</span> <span class="mi">0</span>

        <span class="c1"># 写入初始文件</span>
        <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">write_text</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">buggy_code</span><span class="p">)</span>
        <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"test_solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">write_text</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">test_code</span><span class="p">)</span>

        <span class="n">obs</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_build_observation</span><span class="p">(</span>
            <span class="n">action_taken</span><span class="o">=</span><span class="s2">"[Episode start]"</span><span class="p">,</span>
            <span class="n">test_output</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">_run_tests</span><span class="p">()</span>
        <span class="p">)</span>
        <span class="k">return</span> <span class="n">obs</span><span class="p">,</span> <span class="p">{</span><span class="s2">"step"</span><span class="p">:</span> <span class="mi">0</span><span class="p">}</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">step</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">action</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">StepResult</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""执行一个 Agent 动作；返回 StepResult。"""</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span> <span class="o">+=</span> <span class="mi">1</span>
        <span class="n">action</span> <span class="o">=</span> <span class="n">action</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span>

        <span class="c1"># --- 解析并分派动作 ---</span>
        <span class="k">if</span> <span class="n">action</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"view"</span><span class="p">):</span>
            <span class="n">result_text</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_action_view</span><span class="p">()</span>
        <span class="k">elif</span> <span class="n">action</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"edit"</span><span class="p">):</span>
            <span class="n">result_text</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_action_edit</span><span class="p">(</span><span class="n">action</span><span class="p">)</span>
        <span class="k">elif</span> <span class="n">action</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"run_tests"</span><span class="p">):</span>
            <span class="n">result_text</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_run_tests</span><span class="p">()</span>
        <span class="k">elif</span> <span class="n">action</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"submit"</span><span class="p">):</span>
            <span class="n">result_text</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_run_tests</span><span class="p">()</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">result_text</span> <span class="o">=</span> <span class="p">(</span>
                <span class="sa">f</span><span class="s2">"Unknown action: </span><span class="si">{</span><span class="n">action</span><span class="si">!r}</span><span class="se">\n</span><span class="s2">"</span>
                <span class="s2">"Valid actions: view | edit &lt;new_content&gt; | "</span>
                <span class="s2">"run_tests | submit"</span>
            <span class="p">)</span>

        <span class="n">test_output</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_run_tests</span><span class="p">()</span>
        <span class="n">passed</span>      <span class="o">=</span> <span class="s2">"passed"</span> <span class="ow">in</span> <span class="n">test_output</span> <span class="ow">and</span> <span class="s2">"failed"</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">test_output</span>
        <span class="n">reward</span>      <span class="o">=</span> <span class="mf">1.0</span> <span class="k">if</span> <span class="n">passed</span> <span class="k">else</span> <span class="mf">0.0</span>
        <span class="n">terminated</span>  <span class="o">=</span> <span class="n">passed</span> <span class="ow">or</span> <span class="n">action</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"submit"</span><span class="p">)</span>
        <span class="n">truncated</span>   <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span> <span class="o">&gt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">MAX_STEPS</span>

        <span class="n">obs</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_build_observation</span><span class="p">(</span><span class="n">action</span><span class="p">,</span> <span class="n">test_output</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">StepResult</span><span class="p">(</span><span class="n">obs</span><span class="p">,</span> <span class="n">reward</span><span class="p">,</span> <span class="n">terminated</span><span class="p">,</span> <span class="n">truncated</span><span class="p">,</span>
                          <span class="p">{</span><span class="s2">"step"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span><span class="p">,</span>
                           <span class="s2">"passed"</span><span class="p">:</span> <span class="n">passed</span><span class="p">})</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">render</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""返回当前状态的人类可读摘要。"""</span>
        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="ow">is</span> <span class="kc">None</span><span class="p">:</span>
            <span class="k">return</span> <span class="s2">"[Environment not initialised]"</span>
        <span class="n">code</span> <span class="o">=</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">read_text</span><span class="p">()</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"=== solution.py ===</span><span class="se">\n</span><span class="si">{</span><span class="n">code</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">close</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="kc">None</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""释放资源。"""</span>
        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="ow">and</span> <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="o">.</span><span class="n">exists</span><span class="p">():</span>
            <span class="n">shutil</span><span class="o">.</span><span class="n">rmtree</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="p">)</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">=</span> <span class="kc">None</span>

    <span class="c1"># ------------------------------------------------------------------</span>
    <span class="c1"># 私有辅助方法</span>
    <span class="c1"># ------------------------------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_action_view</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="n">code</span> <span class="o">=</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">read_text</span><span class="p">()</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"Current solution.py:</span><span class="se">\n</span><span class="s2">```python</span><span class="se">\n</span><span class="si">{</span><span class="n">code</span><span class="si">}</span><span class="se">\n</span><span class="s2">```"</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_action_edit</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">action</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="c1"># 期望格式：edit\n```python\n&lt;code&gt;\n```</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">new_code</span> <span class="o">=</span> <span class="n">action</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">"```python"</span><span class="p">)[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">"```"</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
            <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">write_text</span><span class="p">(</span><span class="n">new_code</span><span class="p">)</span>
            <span class="k">return</span> <span class="s2">"File updated successfully."</span>
        <span class="k">except</span> <span class="ne">IndexError</span><span class="p">:</span>
            <span class="k">return</span> <span class="s2">"Edit failed: wrap new code in ```python ... ```"</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_run_tests</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="n">result</span> <span class="o">=</span> <span class="n">subprocess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
            <span class="p">[</span><span class="s2">"python"</span><span class="p">,</span> <span class="s2">"-m"</span><span class="p">,</span> <span class="s2">"pytest"</span><span class="p">,</span> <span class="s2">"test_solution.py"</span><span class="p">,</span>
             <span class="s2">"-v"</span><span class="p">,</span> <span class="s2">"--tb=short"</span><span class="p">,</span> <span class="s2">"--no-header"</span><span class="p">],</span>
            <span class="n">cwd</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="p">,</span>
            <span class="n">capture_output</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">text</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
            <span class="n">timeout</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">TIMEOUT</span>
        <span class="p">)</span>
        <span class="k">return</span> <span class="n">result</span><span class="o">.</span><span class="n">stdout</span> <span class="o">+</span> <span class="n">result</span><span class="o">.</span><span class="n">stderr</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_build_observation</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">action_taken</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                           <span class="n">test_output</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="n">code</span> <span class="o">=</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">read_text</span><span class="p">()</span>
        <span class="k">return</span> <span class="n">textwrap</span><span class="o">.</span><span class="n">dedent</span><span class="p">(</span><span class="sa">f</span><span class="s2">"""</span>
<span class="s2">            TASK: </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">task_description</span><span class="si">}</span>
<span class="s2">            STEP: </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span><span class="si">}</span><span class="s2">/</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">MAX_STEPS</span><span class="si">}</span>

<span class="s2">            --- Last action ---</span>
<span class="s2">            </span><span class="si">{</span><span class="n">action_taken</span><span class="si">}</span>

<span class="s2">            --- Current solution.py ---</span>
<span class="s2">            </span><span class="si">{</span><span class="n">code</span><span class="si">}</span>

<span class="s2">            --- Test output ---</span>
<span class="s2">            </span><span class="si">{</span><span class="n">test_output</span><span class="si">}</span>

<span class="s2">            --- Available actions ---</span>
<span class="s2">            view                          # show current file</span>
<span class="s2">            edit</span><span class="se">\n</span><span class="s2">```python</span><span class="se">\n</span><span class="s2">&lt;code&gt;</span><span class="se">\n</span><span class="s2">```  # replace file contents</span>
<span class="s2">            run_tests                     # run pytest</span>
<span class="s2">            submit                        # finalise and end episode</span>
<span class="s2">        """</span><span class="p">)</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span>

<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># 使用示例</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">"__main__"</span><span class="p">:</span>
    <span class="n">BUGGY</span> <span class="o">=</span> <span class="s2">"def add(a, b):</span><span class="se">\n</span><span class="s2">    return a - b</span><span class="se">\n</span><span class="s2">"</span>   <span class="c1"># bug：本应是加号</span>
    <span class="n">TESTS</span> <span class="o">=</span> <span class="p">(</span>
        <span class="s2">"from solution import add</span><span class="se">\n</span><span class="s2">"</span>
        <span class="s2">"def test_add(): assert add(2, 3) == 5</span><span class="se">\n</span><span class="s2">"</span>
    <span class="p">)</span>

    <span class="n">env</span> <span class="o">=</span> <span class="n">FileEditEnv</span><span class="p">(</span><span class="n">BUGGY</span><span class="p">,</span> <span class="n">TESTS</span><span class="p">,</span> <span class="s2">"Fix the add() function."</span><span class="p">)</span>
    <span class="n">obs</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">env</span><span class="o">.</span><span class="n">reset</span><span class="p">(</span><span class="n">seed</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">obs</span><span class="p">)</span>

    <span class="c1"># 模拟一次正确的编辑</span>
    <span class="n">fix</span> <span class="o">=</span> <span class="s2">"edit</span><span class="se">\n</span><span class="s2">```python</span><span class="se">\n</span><span class="s2">def add(a, b):</span><span class="se">\n</span><span class="s2">    return a + b</span><span class="se">\n</span><span class="s2">```"</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">env</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">fix</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"</span><span class="se">\n</span><span class="s2">Reward: </span><span class="si">{</span><span class="n">result</span><span class="o">.</span><span class="n">reward</span><span class="si">}</span><span class="s2">  |  Terminated: </span><span class="si">{</span><span class="n">result</span><span class="o">.</span><span class="n">terminated</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="n">env</span><span class="o">.</span><span class="n">close</span><span class="p">()</span>
</code></pre></div>

<div class="admonition note"><p class="admonition-title">示例环境中的设计决策</p>
<ul>
<li><p><strong>纯文本接口</strong>：观察和动作均为纯字符串，兼容任意 LLM。</p></li>
<li><p><strong>基于执行的奖励</strong>：奖励来自实际运行测试套件，而非 LLM 裁判。这使其抗作弊且完美对齐。</p></li>
<li><p><strong>隔离子进程</strong>：测试在带超时的独立进程中运行，防止死循环拖垮训练循环。</p></li>
<li><p><strong>兼容 Gymnasium</strong>：<code>reset</code>/<code>step</code>/<code>render</code>/<code>close</code> 遵循标准 API，可直接用于 RL 训练框架。</p></li>
</ul>
</div>

## 21.9　主要 Agent 环境对比 {#sec:env-comparison}

<p>表 <a href="#tab:env-comparison">21.2</a> 总结了本节讨论的主要 Agent 环境的关键属性。</p>

<p class="table-caption" id="tab:env-comparison"><span class="table-number">表 21.2　</span>面向 LLM Agent 的主要 Agent 环境对比。&quot;SoTA&quot;指撰写时已发表的最佳 LLM Agent 结果。给出人类性能（若有）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>环境</strong></th><th scope="col"><strong>观察类型</strong></th><th scope="col"><strong>动作空间</strong></th><th scope="col"><strong>领域</strong></th><th scope="col"><strong>任务数</strong></th><th scope="col"><strong>人类</strong></th><th scope="col"><strong>SoTA LLM</strong></th></tr></thead>
<tbody><tr><td>WebArena</td><td>文本 + DOM</td><td>浏览器 API</td><td>Web 导航</td><td>812</td><td>78%</td><td><span class="arithmatex">\(\sim\)</span>45%</td></tr>
<tr><td>VisualWebArena</td><td>截图 + DOM</td><td>浏览器 API</td><td>视觉 Web</td><td>910</td><td>88%</td><td><span class="arithmatex">\(\sim\)</span>35%</td></tr>
<tr><td>Mind2Web</td><td>截图 + DOM</td><td>浏览器 API</td><td>真实网站</td><td>2,000</td><td>—</td><td><span class="arithmatex">\(\sim\)</span>30%</td></tr>
<tr><td>OSWorld</td><td>截图</td><td>鼠标 + 键盘</td><td>桌面 OS</td><td>369</td><td>72%</td><td><span class="arithmatex">\(\sim\)</span>18%</td></tr>
<tr><td>WindowsAgentArena</td><td>截图</td><td>鼠标 + 键盘</td><td>Windows 应用</td><td>154</td><td>75%</td><td><span class="arithmatex">\(\sim\)</span>20%</td></tr>
<tr><td>SWE-bench Verified</td><td>文本（仓库）</td><td>Shell + 编辑器</td><td>代码修复</td><td>500</td><td>100%</td><td><span class="arithmatex">\(\sim\)</span>50%</td></tr>
<tr><td>GAIA (Level 1)</td><td>文本 + 文件</td><td>工具调用</td><td>通用 QA</td><td>165</td><td>92%</td><td><span class="arithmatex">\(\sim\)</span>55%</td></tr>
<tr><td>GAIA (Level 3)</td><td>文本 + 文件</td><td>工具调用</td><td>困难 QA</td><td>42</td><td>92%</td><td><span class="arithmatex">\(\sim\)</span>10%</td></tr>
<tr><td>NetHack (NLE)</td><td>文本 + 字符图</td><td>离散动作</td><td>Roguelike 游戏</td><td>—</td><td><span class="arithmatex">\(&gt;\)</span>10k 分</td><td><span class="arithmatex">\(\sim\)</span>5k 分</td></tr>
<tr><td>Voyager (Minecraft)</td><td>文本 + 代码</td><td>代码执行</td><td>开放世界游戏</td><td>课程</td><td>—</td><td>15+ 科技树</td></tr>
<tr><td>MLAgentBench</td><td>文本 + 代码</td><td>Shell + 编辑器</td><td>ML 工程</td><td>13</td><td>—</td><td><span class="arithmatex">\(\sim\)</span>40%</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">解读对比表</p>
<p>人类性能与 SoTA LLM 性能之间的差距，在<em>计算机使用</em>类任务上最大（OSWorld：72% vs. 18%），在<em>代码修复</em>上最小（SWE-bench：100% vs. 50%）。这一模式反映了动作空间的成熟度：LLM 已在海量代码上训练，但基于截图的交互数据相对较少。随着计算机使用类训练数据的积累，差距有望缩小。</p>
</div>

## 21.10　小结 {#sec:env-summary}

<p>Agent 环境是训练和评估 LLM Agent 的基底。本节的关键要点是：</p>

<ol>
<li><p><strong>环境不是可选项。</strong> 安全探索、可复现评估和课程学习都需要结构化环境。若没有环境，Chatbot 评估与 Agent 评估之间的鸿沟无法跨越。</p></li>
<li><p><strong>四个维度都需谨慎设计。</strong> 观察空间、动作空间、奖励信号和 Episode 结构各有可能让整个基准失效的失败模式。</p></li>
<li><p><strong>生态丰富但碎片化。</strong> 代码沙箱、Web 环境、计算机使用类环境、SWE 环境、科研环境、游戏和多 Agent 竞技场各测试不同能力。任何单一环境都不充分。</p></li>
<li><p><strong>标准化很重要。</strong> OpenEnv <a class="book-citation" id="cite-huggingface2025openenv-1--section-10" href="../../references.md#ref-huggingface2025openenv" aria-label="参考文献 88">[88]</a> 提供 Gymnasium 风格的 API，配合 Docker 隔离以及把 Hugging Face Spaces 作为注册中心——降低了构建新环境以及在不同环境间比较 Agent 的成本。</p></li>
<li><p><strong>人类差距真实存在但正在收窄。</strong> 当前 LLM Agent 在大多数基准上达到人类性能的 20–50%。进展最快的是训练数据充裕的领域（代码），最慢的是需要细粒度感知的领域（GUI 控制）。</p></li>
</ol>

<div class="admonition question"><p class="admonition-title">Agent 环境中的开放研究问题</p>
<ul>
<li><p>如何为那些正确性带有主观性或上下文依赖的任务设计奖励函数？</p></li>
<li><p>单一 Agent 架构能否在无需任务特定微调的前提下跨基于文本和多模态环境泛化？</p></li>
<li><p>训练所需的环境保真度应当达到何种程度？在简化的仿真器上训练能迁移到真实部署吗？</p></li>
<li><p>当 LLM 在可能包含基准答案的越来越大的 Web 语料上训练时，如何防止基准污染？</p></li>
</ul>
</div>

### 21.10.1　新兴基准（2026） {#sec:emerging-benchmarks-2026}

##### UniClawBench。 {#heading-21-49}

<p>UniClawBench <a class="book-citation" id="cite-uniclawbench2026-1" href="../../references.md#ref-uniclawbench2026" aria-label="参考文献 214">[214]</a> 在实时、隔离的 Docker 容器中运行 400 个双语真实世界任务——跨五个功能领域测试 Agent，包括跨平台协调。其标志性特征是多 Agent 反馈回路：一个隐藏的监督者与一个模拟用户注入真实世界的摩擦（含混的请求、不断变化的需求、部分失败），衡量的是<em>交互韧性</em>，而非一次性任务完成率。截至 2026 年年中，它是评估生产就绪 Agent 的最严格基准。</p>

##### Long-Horizon-Terminal-Bench。 {#heading-21-50}

<p>Long-Horizon-Terminal-Bench <a class="book-citation" id="cite-terminalbench2026-1" href="../../references.md#ref-terminalbench2026" aria-label="参考文献 386">[386]</a> 覆盖 21 个领域的 46 个复杂任务，测试 Agent 能否驾驭以小时而非分钟计的终端操作。其关键创新是<strong>基于密集奖励的评分</strong>：在每一步给中间进展打分，而不是依赖终点线的二元成功。这暴露了一个残酷的现实：即使是最强的前沿模型，也无法连续执行超过 <span class="arithmatex">\(\sim\)</span>15 条终端命令而不跑偏——揭示长时程顺序执行仍是一个根本性的未解挑战。</p>
