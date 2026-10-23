# 28.28　Agent 环境问题 {#agentic-environments-questions}

<section class="quiz-question" id="q-source-101"><h2 class="question-title">Q：为网页浏览 Agent 环境设计一个 Reward 函数</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：对于 WebArena 风格任务（例如“找到 12 月 15 日从 NYC 到 SF 的最便宜航班”）：</p>

<p><strong>稀疏 Reward</strong>（简单但难以学习）：</p>



<div class="arithmatex">\[r = \begin{cases} 1 &amp; \text{if final page/state matches ground truth} \\ 0 &amp; \text{otherwise} \end{cases}\]</div>

<p><strong>密集 Reward</strong>（更利于训练；设计更难）：</p>

<ol>
<li><p><strong>进度 Reward</strong>：每个让 Agent 更接近目标的页面 <span class="arithmatex">\(+0.1\)</span>（用与目标状态的文本相似度衡量）</p></li>
<li><p><strong>效率惩罚</strong>：每个动作 <span class="arithmatex">\(-0.01\)</span>（鼓励更短轨迹）</p></li>
<li><p><strong>里程碑 Reward</strong>：到达中间目标（如导航到航班搜索页）<span class="arithmatex">\(+0.3\)</span></p></li>
<li><p><strong>无效动作惩罚</strong>：产生错误的动作（404；表单校验失败）<span class="arithmatex">\(-0.05\)</span></p></li>
</ol>

<p><strong>基于势能的 reward shaping</strong>（保留最优 Policy）：</p>



<div class="arithmatex">\[r_{\text{shaped}}(s, a, s&#x27;) = r(s, a, s&#x27;) + \gamma \Phi(s&#x27;) - \Phi(s)\]</div>

<p>其中 <span class="arithmatex">\(\Phi(s) = -\text{min\_steps\_to\_goal}(s)\)</span>（由启发式或学习到的 Value 函数估计）。</p>

<p><strong>挑战</strong>：部分可观察（无法总是判断是否更接近目标）；随机环境（页面内容会变）；Reward hacking（Agent 找到满足 Reward 但不满足用户意图的捷径）。</p>

<p><em><strong>复习：</strong>第 <a href="../12-agentic-training/index.md">12</a> 章与第 <a href="../21-environments/index.md">21</a> 章（LLM Agent 训练；Agent 环境）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-102"><h2 class="question-title">Q：是什么让 SWE-bench 成为一个特别具有挑战性的 Agent 基准？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：SWE-bench 在来自流行 Python 仓库的真实 GitHub issue 上测试 Agent：</p>

<p><strong>为什么难</strong>：</p>

<ol>
<li><p><strong>仓库级上下文</strong>：Agent 必须理解 10 万+ 行的代码库。无法装入上下文窗口——必须探索、搜索和导航。</p></li>
<li><p><strong>规范不充分的任务</strong>：Issue 由人类带着隐含上下文写就。Agent 必须推断真正需要什么。</p></li>
<li><p><strong>跨文件编辑</strong>：解决方案通常跨越多个文件，并有级联依赖。</p></li>
<li><p><strong>测试验证</strong>：必须通过现有测试 <em>以及</em>验证修复的新测试。</p></li>
<li><p><strong>无人辅助</strong>：不像 HumanEval（单函数），SWE-bench 需要完整的软件工程工作流：读 issue <span class="arithmatex">\(\to\)</span> 探索代码 <span class="arithmatex">\(\to\)</span> 定位 bug <span class="arithmatex">\(\to\)</span> 实现修复 <span class="arithmatex">\(\to\)</span> 验证。</p></li>
</ol>

<p><strong>当前 SOTA</strong>（2024–2025）：最佳 Agent 解决 SWE-bench Verified（精选子集）的约 <span class="arithmatex">\(\sim\)</span>50%。完整 SWE-bench：约 <span class="arithmatex">\(\sim\)</span>30%。</p>

<p><strong>对训练的关键洞见</strong>：SWE-bench 揭示了“编程能力”（编写正确函数）与“软件工程能力”（理解系统；导航代码库；做最小改动）之间的差距。在 SWE-bench 风格环境上的 RL 训练教会 Agent 探索和规划策略，而不仅是代码生成。</p>

<p><em><strong>复习：</strong>第 <a href="../21-environments/index.md">21</a> 章（Agent 环境与基准）。</em></p>
</details></section>
