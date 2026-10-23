# 19.9　生产级循环架构 {#sec:production-loops}

## 19.9.1　夜间维护循环 {#sec:nightly-maintenance}

<p>最立即可用的循环模式：一个按计划在夜间运行的自动化，趁团队睡觉时处理累积的工作。</p>

<div class="admonition example"><p class="admonition-title">架构：夜间 CI 分类与修复循环</p>
<p><strong>触发</strong>：Cron 计划（例如每天 02:00 UTC）。</p>

<p><strong>阶段 1 —— 分类</strong>（单 Agent，只读）：</p>

<ul>
<li><p>读取昨天的 CI 失败、新开的 issue 与最近的提交</p></li>
<li><p>对每条发现分类：可自动修复 / 需人工 / 误报</p></li>
<li><p>把发现写入状态文件；归档误报</p></li>
</ul>

<p><strong>阶段 2 —— 修复</strong>（并行子 Agent，隔离工作树）：</p>

<ul>
<li><p>对每个可自动修复的 issue，在隔离工作树中生成一个子 Agent</p></li>
<li><p>每个子 Agent 运行一个验证循环（修复 <span class="arithmatex">\(\to\)</span> 测试 <span class="arithmatex">\(\to\)</span> 重试）</p></li>
<li><p>预算：每个 issue 20 步或 15 分钟</p></li>
</ul>

<p><strong>阶段 3 —— 审查</strong>（独立验证 Agent）：</p>

<ul>
<li><p>审查子 Agent 依据项目编码标准评估每个提议的修复</p></li>
<li><p>批准的修复：开 draft PR、链接到 issue、通过连接器通知频道</p></li>
<li><p>被拒的修复：记录拒绝原因，升级到人工分类收件箱</p></li>
</ul>

<p><strong>持久化</strong>：状态文件（提交到仓库）记录所有已采取的行动，让第二天的运行跳过已解决的 issue，并从失败处接着干。</p>
</div>

## 19.9.2　持续实验循环 {#sec:experimentation-loop}

<p>受 AutoResearch <a class="book-citation" id="cite-karpathy2026autoresearch-1" href="../../references.md#ref-karpathy2026autoresearch" aria-label="参考文献 179">[179]</a> 启发，这一模式把循环应用于科学发现：修改训练脚本，运行限时实验，评估结果，决定提交还是回滚。</p>

<div class="admonition note"><p class="admonition-title">AutoResearch 循环结构</p>
<ol>
<li><p><strong>假设生成</strong>：Agent 提出一项修改（超参数、架构改动、数据增强）</p></li>
<li><p><strong>实现</strong>：Agent 修改训练脚本</p></li>
<li><p><strong>限时执行</strong>：恰好运行 <span class="arithmatex">\(T\)</span> 分钟（硬性墙钟上限）</p></li>
<li><p><strong>评估</strong>：从输出读取验证指标</p></li>
<li><p><strong>决策</strong>：指标提升 <span class="arithmatex">\(\to\)</span> <code>git commit</code>；指标下降 <span class="arithmatex">\(\to\)</span> <code>git rollback</code></p></li>
<li><p><strong>记录</strong>：把假设、结果与推理记入实验日志</p></li>
<li><p><strong>重复</strong>：依据累积的实验历史生成下一个假设</p></li>
</ol>

<p><strong>关键约束</strong>：限时执行防止任何一次失败的实验消耗过多算力。基于 git 的提交/回滚确保代码库只会向前。</p>
</div>

<div class="admonition warning"><p class="admonition-title">研究循环中的局部最小值问题</p>
<p>研究循环容易陷入保守探索。面对困难的优化景观时，Agent 倾向于提出最小改动——把学习率调 0.001——只带来名义上的改进而非真正的进展。这是 RL 探索问题在循环层面的显现。对策包括：定期的“大胆假设”提示、为尝试新颖方法设置多样性奖励，以及在两次运行之间由人类注入研究方向。</p>
</div>

## 19.9.3　常驻编排器 {#sec:always-on-orchestrator}

<p>OpenClaw <a class="book-citation" id="cite-steinberger2026openclaw-1" href="../../references.md#ref-steinberger2026openclaw" aria-label="参考文献 338">[338]</a> 这类系统实现了一个持久的“心跳”机制：一个按固定周期运行的元 Agent，评估仓库状态、监督子循环并分派工作。与每天触发一次的夜间模式不同，常驻编排器保持持续感知，在几分钟内响应事件（新提交、CI 失败、新 issue）。</p>

<p>该架构引入了一个带崩溃恢复的持久状态数据库——如果编排器在循环中途失败，它会从最后一个检查点状态恢复，而不是从头重启。子循环被作为拥有各自终止条件的独立任务来跟踪，编排器监控它们的停滞、预算耗尽或冲突。</p>
