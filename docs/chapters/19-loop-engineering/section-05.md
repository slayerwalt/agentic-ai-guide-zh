# 19.5　循环模式 {#sec:loop-patterns}

<p>在第 <a href="../20-agent-patterns/index.md#sec:agent-design-patterns">20</a> 章介绍的 Agent 设计模式基础之上，循环工程识别出一组自主性逐级递增的模式层级：</p>

## 19.5.1　验证循环 {#sec:validation-loop}

<p>最简单也最常见的模式：生成，对照确定性检查验证，失败则重试。</p>

<div class="admonition example"><p class="admonition-title">验证循环：修复直到测试通过</p>
<div class="highlight"><pre><span></span><code><span class="n">failure_log</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">attempt</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">MAX_ATTEMPTS</span><span class="p">):</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">run_tests</span><span class="p">()</span>
    <span class="k">if</span> <span class="n">result</span><span class="o">.</span><span class="n">passed</span><span class="p">:</span>
        <span class="k">break</span>  <span class="c1"># 成功——退出循环</span>
    <span class="n">fix</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">generate</span><span class="p">(</span>
        <span class="sa">f</span><span class="s2">"Fix this test failure:</span><span class="se">\n</span><span class="si">{</span><span class="n">result</span><span class="o">.</span><span class="n">errors</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>
        <span class="sa">f</span><span class="s2">"Previous attempts that failed: </span><span class="si">{</span><span class="n">failure_log</span><span class="si">}</span><span class="s2">"</span>
    <span class="p">)</span>
    <span class="n">apply_patch</span><span class="p">(</span><span class="n">fix</span><span class="p">)</span>
    <span class="n">failure_log</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">errors</span><span class="p">)</span>
</code></pre></div>

<p><strong>终止条件</strong>：测试通过（成功）或重试预算耗尽（升级）。</p>

<p><strong>关键属性</strong>：验证信号是<em>确定性的</em>——测试运行器产生客观的通过/失败，模型无法争辩。</p>
</div>

<p>这一模式之所以成功，是因为它的奖励信号干净利落。测试运行器充当不可腐蚀的批评者——与 LLM-as-judge 评估不同，它无法被说服或愚弄。</p>

## 19.5.2　Reflexion 循环 {#sec:reflexion-loop}

<p>在验证循环之上加入尝试之间的显式自我批评 <a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a>。每次失败后，Agent 把一段自然语言反思（“我失败是因为修改了错误的函数——错误追溯到 import，而不是实现”）写入情景记忆缓冲区。后续尝试会读取这个缓冲区，从而在<em>单个回合内</em>实现学习，无需权重更新。</p>

<div class="admonition note"><p class="admonition-title">Reflexion：从失败中做上下文内学习</p>
<p>Reflexion 循环在基本重试之外增加了三个组件：</p>

<ol>
<li><p><strong>Actor</strong>：执行任务</p></li>
<li><p><strong>Evaluator</strong>：提供奖励信号（测试、指标或 LLM judge）</p></li>
<li><p><strong>自我反思</strong>：把一条口头化的“学到的教训”写入情景记忆</p></li>
</ol>

<p>反思记忆跨尝试持久化，为 Agent 提供避免重复相同失败的机制。实证上，Reflexion 在 HumanEval 上达到 91% 的 pass@1（基线为 67%），证明口头自我批评在迭代场景中可以替代权重更新 <a class="book-citation" id="cite-shinn2023reflexion-2" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a>。</p>
</div>

## 19.5.3　评估器—优化器循环 {#sec:eval-opt-loop}

<p>一种双模型架构 <a class="book-citation" id="cite-madaan2023selfrefine-1" href="../../references.md#ref-madaan2023selfrefine" aria-label="参考文献 246">[246]</a><a class="book-citation" id="cite-anthropic2024buildingagents-1" href="../../references.md#ref-anthropic2024buildingagents" aria-label="参考文献 10">[10]</a>：一个模型生成候选，另一个独立模型依据显式标准评估它并返回结构化反馈。生成器吸收反馈，产出改进版本。循环重复，直到评估器评分超过阈值或迭代预算耗尽。</p>

<p>关键的设计选择是评估器是<em>确定性的</em>（编译器、测试套件、linter）还是<em>概率性的</em>（另一个 LLM）。确定性评估器带来可靠的收敛；概率性评估器对主观标准是必要的，但引入了评估器—生成器合谋的风险。</p>

## 19.5.4　层级循环 {#sec:hierarchical-loop}

<p>一个元循环生成并监控多个子循环。编排 Agent 把高层目标分解为子任务，把每个子任务指派给运行自己循环的专门化子 Agent，监控它们的进度并综合结果。这对应分层 RL <a class="book-citation" id="cite-barto2003recent-1" href="../../references.md#ref-barto2003recent" aria-label="参考文献 21">[21]</a>：高层策略选择子目标，低层策略执行它们。</p>

<div class="admonition example"><p class="admonition-title">层级循环：通宵仓库维护</p>
<div class="highlight"><pre><span></span><code><span class="c1"># 外层循环：通过自动化每晚运行</span>
<span class="k">for</span> <span class="n">issue</span> <span class="ow">in</span> <span class="n">triage_agent</span><span class="o">.</span><span class="n">identify_actionable_issues</span><span class="p">():</span>
    <span class="c1"># 内层循环：每个问题一个子智能体，使用隔离工作树</span>
    <span class="n">worktree</span> <span class="o">=</span> <span class="n">create_isolated_worktree</span><span class="p">(</span><span class="n">issue</span><span class="o">.</span><span class="n">branch_name</span><span class="p">)</span>

    <span class="n">result</span> <span class="o">=</span> <span class="n">fix_loop</span><span class="p">(</span>
        <span class="n">goal</span><span class="o">=</span><span class="n">issue</span><span class="o">.</span><span class="n">description</span><span class="p">,</span>
        <span class="n">worktree</span><span class="o">=</span><span class="n">worktree</span><span class="p">,</span>
        <span class="n">verifier</span><span class="o">=</span><span class="n">run_tests_and_lint</span><span class="p">,</span>
        <span class="n">max_steps</span><span class="o">=</span><span class="mi">20</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">result</span><span class="o">.</span><span class="n">success</span><span class="p">:</span>
        <span class="n">review</span> <span class="o">=</span> <span class="n">review_agent</span><span class="o">.</span><span class="n">evaluate</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">patch</span><span class="p">)</span>  <span class="c1"># 检查器</span>
        <span class="k">if</span> <span class="n">review</span><span class="o">.</span><span class="n">approved</span><span class="p">:</span>
            <span class="n">open_pull_request</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">patch</span><span class="p">,</span> <span class="n">issue</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">state_db</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">issue</span><span class="p">,</span> <span class="s2">"fix_rejected"</span><span class="p">,</span> <span class="n">review</span><span class="o">.</span><span class="n">feedback</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">state_db</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">issue</span><span class="p">,</span> <span class="s2">"fix_failed"</span><span class="p">,</span> <span class="n">result</span><span class="o">.</span><span class="n">state</span><span class="p">)</span>
</code></pre></div>

<p>外层循环（分类）按计划运行。内层循环（修复）是临时的、任务范围的。审查 Agent 提供制造者—检查者分离。状态跨夜间运行持久化。</p>
</div>

## 19.5.5　自主研究循环 {#sec:research-loop}

<p>Karpathy 的 AutoResearch <a class="book-citation" id="cite-karpathy2026autoresearch-1" href="../../references.md#ref-karpathy2026autoresearch" aria-label="参考文献 179">[179]</a> 展示了一个受严格约束的研究循环：Agent 修改训练脚本，运行一个限时实验（例如在 GPU 上跑 5 分钟），读取验证指标，并决定是提交改动（指标提升）还是回滚（指标下降）。循环无限持续，通过迭代式实验探索超参数与架构空间。</p>

<p>这一模式之所以有效，是因为奖励信号<em>标量且无歧义</em>：验证损失要么下降了要么没有。每轮迭代的严格时间上限防止任何一次失败的实验消耗过多资源。</p>
