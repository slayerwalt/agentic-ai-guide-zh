# 21.3　Agent 环境的类型 {#sec:env-types}

## 21.3.1　代码执行沙箱 {#code-execution-sandboxes}

<p>对 LLM 而言，最基础的 Agent 环境就是代码执行沙箱：Agent 写代码，沙箱运行，返回输出。这一简单循环支撑了出人意料比例的真实 Agent 部署。</p>

<p><strong>基于 Docker 的隔离</strong>是最常见的做法。每个 Episode 从已知镜像启动一个全新容器，在其中执行 Agent 代码，并在 Episode 结束时销毁该容器。网络访问、文件系统写入、进程派生均可在容器级别加以控制。</p>

<p><strong>E2B</strong>（Environments to Benchmarks）提供一个托管云沙箱 API：Agent 通过 HTTP 发送代码，E2B 在隔离的 Firecracker microVM 中执行（启动时间不到 200 毫秒）并返回 stdout/stderr。E2B 处理容器生命周期管理的基础设施复杂性，便于集成进 Agent 训练循环。</p>

<p><strong>Modal</strong> 提供类似的托管执行模型，但 GPU 支持更强，适合需要在任务中运行 ML 工作负载的 Agent。</p>

<div class="admonition warning"><p class="admonition-title">沙箱逃逸与安全</p>
<p>代码执行沙箱是主要的攻击面。一个能力足够强的 Agent（或被 Prompt 注入的恶意载荷）可能尝试通过内核漏洞、网络外渗或资源耗尽来逃逸沙箱。纵深防御不可或缺：结合使用容器隔离、seccomp 配置、只读根文件系统、网络出站过滤以及 CPU/内存 cgroups。绝不要以宿主级别权限运行 Agent 生成的代码。</p>
</div>

## 21.3.2　Web 环境 {#web-environments}

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

## 21.3.3　计算机使用环境 {#computer-use-environments}

<p>计算机使用环境（Computer Use）让 Agent 控制一个完整的桌面操作系统，通过截图和/或可访问性 API 进行观察。</p>

<p><strong>OSWorld</strong> <a class="book-citation" id="cite-xie2024osworld-1" href="../../references.md#ref-xie2024osworld" aria-label="参考文献 397">[397]</a> 跨三种操作系统（Ubuntu、Windows、macOS）测试桌面自动化，涵盖 369 个任务，覆盖各类生产力应用（LibreOffice、VS Code、Chrome、GIMP 等）。Agent 通过截图观察，并以 <code>pyautogui</code> 风格的鼠标键盘命令行动。人-Agent 差距相当悬殊：标注者在约 72% 的任务上成功，而最强 LLM Agent 仅能达到 <span class="arithmatex">\(\sim\)</span>18%，凸显了像素级 GUI 控制的难度。</p>

<p><strong>WindowsAgentArena</strong> <a class="book-citation" id="cite-bonatti2024windows-1" href="../../references.md#ref-bonatti2024windows" aria-label="参考文献 27">[27]</a> 专门聚焦于 Windows 11，包含 19 个应用上的 154 个任务，强调企业工作流：Excel 公式、PowerPoint 编辑、Outlook 邮件管理。</p>

<div class="admonition note"><p class="admonition-title">截图瓶颈</p>
<p>计算机使用类 Agent 面临一个根本挑战：截图维度极高（通常 <span class="arithmatex">\(1920 \times 1080 \times 3\)</span> 像素），但其中大部分信息与当前动作无关。高效的 Agent 学会关注屏幕的小区域、使用可访问性树通过名称（而非像素坐标）识别可交互元素，并对此前访问过的 UI 状态维护一个紧凑的工作记忆。</p>
</div>

## 21.3.4　软件工程环境 {#software-engineering-environments}

<p>软件工程（Software Engineering, SWE）环境要求 Agent 解决真实世界的编程任务：修 bug、实现功能、写测试。</p>

<p><strong>SWE-bench</strong> <a class="book-citation" id="cite-jimenez2024swebench-1" href="../../references.md#ref-jimenez2024swebench" aria-label="参考文献 169">[169]</a> 取材自 12 个广泛使用的 Python 项目（Django、Flask、scikit-learn 等）的 2,294 个真实 pull request。每个实例将一段 issue 描述与一份预留的测试套件配对，仅在应用正确补丁后测试才通过。Agent 必须理解仓库结构、定位相关代码、实施修复并用测试套件加以验证。<strong>SWE-bench Verified</strong> 子集（500 个 issue）经过人工正确性校验，是标准评估目标。</p>

<p><strong>SWE-agent</strong> <a class="book-citation" id="cite-yang2024sweagent-1" href="../../references.md#ref-yang2024sweagent" aria-label="参考文献 404">[404]</a> 既是基准环境也是 Agent 框架。它引入了<em>Agent-计算机接口（Agent-Computer Interface, ACI）</em>：一组为 LLM Agent 优化的 Shell 命令（如 <code>search_file</code>、<code>open</code>、<code>edit</code>），相比原始 bash 降低了动作空间复杂度。</p>

<div class="admonition example"><p class="admonition-title">SWE-bench 工作流</p>
<p><strong>输入</strong>：一段 GitHub issue 描述，以及该 issue 提交时刻的完整仓库快照。</p>

<p><strong>Agent 动作</strong>：<code>find_file</code>、<code>view</code>、<code>edit</code>、<code>python -m pytest tests/</code>。</p>

<p><strong>奖励</strong>：Agent 补丁应用后若所有目标测试通过则为 1，否则为 0。不计部分得分。</p>
</div>

## 21.3.5　科研环境 {#scientific-research-environments}

<p>科研环境推动 Agent 走向自主知识生成：阅读论文、形成假设、设计实验、解读结果。</p>

<p><strong>PaperQA2</strong> <a class="book-citation" id="cite-lala2023paperqa-1" href="../../references.md#ref-lala2023paperqa" aria-label="参考文献 245">[245]</a> 是一个检索增强 Agent，通过搜索 PDF 语料库、抽取相关段落、综合出带引用的答案来回答科学问题。它既是文献依据型推理的工具，也是其基准。</p>

<p><strong>AI Scientist</strong> <a class="book-citation" id="cite-lu2024aiscientist-1" href="../../references.md#ref-lu2024aiscientist" aria-label="参考文献 240">[240]</a> 是一个端到端的科研自动化系统：给定一个研究方向，Agent 即生成假设、撰写并运行实验、解读结果、产出论文初稿。该环境包含一个 Python 执行沙箱、文献搜索 API 和 LaTeX 编译器。</p>

<p><strong>MLAgentBench</strong> <a class="book-citation" id="cite-huang2024mlagentbench-1" href="../../references.md#ref-huang2024mlagentbench" aria-label="参考文献 151">[151]</a> 在机器学习工程任务上评估 Agent：在算力预算内提升给定数据集上的模型精度。Agent 可读取数据、编写训练脚本、运行实验并迭代。</p>

## 21.3.6　游戏与仿真环境 {#game-and-simulation-environments}

<p>游戏提供了丰富的长时程环境，具有定义良好的奖励信号且无真实世界后果。</p>

<p><strong>NetHack</strong> <a class="book-citation" id="cite-kuttler2020nethack-1" href="../../references.md#ref-kuttler2020nethack" aria-label="参考文献 196">[196]</a> 是一款程序化生成的 roguelike 游戏，状态空间巨大，要求长期规划、物品管理以及对意外事件的适应。NetHack Learning Environment（NLE）提供了 Gym 兼容接口。</p>

<p><strong>Voyager / Minecraft</strong> <a class="book-citation" id="cite-wang2023voyager-1" href="../../references.md#ref-wang2023voyager" aria-label="参考文献 362">[362]</a> 把 Minecraft 引擎用作一个开放式环境。Voyager 引入了难度逐级上升的任务课程（采集木材 <span class="arithmatex">\(\to\)</span> 制造工具 <span class="arithmatex">\(\to\)</span> 建造庇护所 <span class="arithmatex">\(\to\)</span> 探索下界）以及一个跨 Episode 累积可复用代码片段的技能库。</p>

<p><strong>GAIA</strong> <a class="book-citation" id="cite-mialon2023gaia-1" href="../../references.md#ref-mialon2023gaia" aria-label="参考文献 254">[254]</a> 提出 466 个问题，要求链式工具使用——Web 搜索、代码执行、文件解析——并按所需推理步数划分为三个难度等级。该基准鲜明地暴露出人类能力（约 92% 准确率）与当前 LLM Agent（GPT-4 配合插件发布初期约 15%，后续系统约 30%）之间的鸿沟。</p>

## 21.3.7　多 Agent 环境 {#multi-agent-environments}

<p>多 Agent 环境涉及两个或更多 LLM Agent 之间以及与共享世界的交互。</p>

<ul>
<li><p><strong>协商（Negotiation）</strong>：具有私有效用函数的 Agent 必须通过对话达成交易。经典环境包括 DealOrNoDeal <a class="book-citation" id="cite-lewis2017dealornodeal-1" href="../../references.md#ref-lewis2017dealornodeal" aria-label="参考文献 206">[206]</a> 和 CaSiNo <a class="book-citation" id="cite-chawla2021casino-1" href="../../references.md#ref-chawla2021casino" aria-label="参考文献 42">[42]</a>。</p></li>
<li><p><strong>辩论（Debate）</strong>：两个 Agent 持相对立场进行辩论；由裁判 Agent（或人类）评判论证质量。用于通过对抗压力引出真实的推理。</p></li>
<li><p><strong>协作式任务完成</strong>：具有互补能力（规划者、执行者、评论者）的 Agent 必须协作完成任一单独 Agent 无法解决的任务。相关框架包括 AutoGen <a class="book-citation" id="cite-wu2023autogen-1" href="../../references.md#ref-wu2023autogen" aria-label="参考文献 385">[385]</a>、CrewAI <a class="book-citation" id="cite-moura2023crewai-1" href="../../references.md#ref-moura2023crewai" aria-label="参考文献 261">[261]</a> 和 MetaGPT <a class="book-citation" id="cite-hong2023metagpt-1" href="../../references.md#ref-hong2023metagpt" aria-label="参考文献 142">[142]</a>。</p></li>
<li><p><strong>竞争性游戏</strong>：Agent 在零和博弈（国际象棋、围棋、扑克）中对弈，对手本身就是另一个 LLM Agent。在这类环境中的自博弈已经在狭窄领域内产生了超人类的表现。</p></li>
</ul>
