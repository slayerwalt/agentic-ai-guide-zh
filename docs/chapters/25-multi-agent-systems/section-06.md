# 25.6　面向 LLM 的多 Agent 模式 {#subsec:mas-patterns}

<p>除了架构拓扑之外，还有若干<strong>交互模式</strong>已被证明对基于 LLM 的多智能体系统特别有效。（这些模式与第 <a href="../20-agent-patterns/index.md#sec:agent-design-patterns">20</a> 章中的单 Agent 设计模式互为补充。）</p>

## 25.6.1　辩论模式（Debate Pattern） {#debate-pattern}

<p>多个 Agent 为不同立场辩论；一个裁判 Agent 评估论点并做出决定。已有研究表明辩论能提升事实准确性并减少幻觉 <a class="book-citation" id="cite-du2023improving-1" href="../../references.md#ref-du2023improving" aria-label="参考文献 78">[78]</a>。</p>

<p class="code-caption"><span class="code-number">代码 25.4　</span>辩论模式的实现</p>
<div class="highlight"><pre><span></span><code><span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">debate_round</span><span class="p">(</span><span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">agents</span><span class="p">:</span> <span class="nb">list</span><span class="p">,</span> <span class="n">judge</span><span class="p">:</span> <span class="n">Agent</span><span class="p">,</span>
                       <span class="n">rounds</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="c1"># 运行多 agent 辩论，并返回裁判的判决。</span>
<span class="w">    </span><span class="sd">"""Run a multi-agent debate and return the judge's verdict."""</span>
    <span class="n">positions</span> <span class="o">=</span> <span class="p">{</span><span class="n">a</span><span class="o">.</span><span class="n">name</span><span class="p">:</span> <span class="k">await</span> <span class="n">a</span><span class="o">.</span><span class="n">generate_position</span><span class="p">(</span><span class="n">question</span><span class="p">)</span>
                 <span class="k">for</span> <span class="n">a</span> <span class="ow">in</span> <span class="n">agents</span><span class="p">}</span>

    <span class="k">for</span> <span class="n">round_num</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">rounds</span><span class="p">):</span>
        <span class="c1"># 每个 agent 看到他人立场并可反驳</span>
        <span class="n">rebuttals</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="k">for</span> <span class="n">agent</span> <span class="ow">in</span> <span class="n">agents</span><span class="p">:</span>
            <span class="n">others</span> <span class="o">=</span> <span class="p">{</span><span class="n">k</span><span class="p">:</span> <span class="n">v</span> <span class="k">for</span> <span class="n">k</span><span class="p">,</span> <span class="n">v</span> <span class="ow">in</span> <span class="n">positions</span><span class="o">.</span><span class="n">items</span><span class="p">()</span>
                      <span class="k">if</span> <span class="n">k</span> <span class="o">!=</span> <span class="n">agent</span><span class="o">.</span><span class="n">name</span><span class="p">}</span>
            <span class="n">rebuttals</span><span class="p">[</span><span class="n">agent</span><span class="o">.</span><span class="n">name</span><span class="p">]</span> <span class="o">=</span> <span class="k">await</span> <span class="n">agent</span><span class="o">.</span><span class="n">rebut</span><span class="p">(</span>
                <span class="n">question</span><span class="p">,</span> <span class="n">positions</span><span class="p">[</span><span class="n">agent</span><span class="o">.</span><span class="n">name</span><span class="p">],</span> <span class="n">others</span>
            <span class="p">)</span>
        <span class="n">positions</span> <span class="o">=</span> <span class="n">rebuttals</span>

    <span class="c1"># 裁判评估所有最终立场</span>
    <span class="n">verdict</span> <span class="o">=</span> <span class="k">await</span> <span class="n">judge</span><span class="o">.</span><span class="n">evaluate</span><span class="p">(</span><span class="n">question</span><span class="p">,</span> <span class="n">positions</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">verdict</span>
</code></pre></div>

## 25.6.2　反思模式（Reflection Pattern） {#reflection-pattern}

<p>一个 Agent 生成输出；第二个 Agent 对其进行批判；第一个 Agent 根据批判进行修订。这实现了一个“生成—批判—修订”循环，迭代地提升质量。</p>

<p class="code-caption"><span class="code-number">代码 25.5　</span>反思模式</p>
<div class="highlight"><pre><span></span><code><span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">reflection_loop</span><span class="p">(</span><span class="n">task</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">generator</span><span class="p">:</span> <span class="n">Agent</span><span class="p">,</span>
                          <span class="n">critic</span><span class="p">:</span> <span class="n">Agent</span><span class="p">,</span> <span class="n">max_rounds</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">3</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
    <span class="n">draft</span> <span class="o">=</span> <span class="k">await</span> <span class="n">generator</span><span class="o">.</span><span class="n">generate</span><span class="p">(</span><span class="n">task</span><span class="p">)</span>

    <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">max_rounds</span><span class="p">):</span>
        <span class="n">critique</span> <span class="o">=</span> <span class="k">await</span> <span class="n">critic</span><span class="o">.</span><span class="n">critique</span><span class="p">(</span><span class="n">task</span><span class="p">,</span> <span class="n">draft</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">critique</span><span class="o">.</span><span class="n">is_satisfactory</span><span class="p">:</span>
            <span class="k">break</span>
        <span class="n">draft</span> <span class="o">=</span> <span class="k">await</span> <span class="n">generator</span><span class="o">.</span><span class="n">revise</span><span class="p">(</span><span class="n">task</span><span class="p">,</span> <span class="n">draft</span><span class="p">,</span> <span class="n">critique</span><span class="o">.</span><span class="n">feedback</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">draft</span>
</code></pre></div>

## 25.6.3　分工模式（Division of Labor Pattern） {#division-of-labor-pattern}

<p>任务被分解为可并行执行的独立子任务。结果由一个综合 Agent 聚合。该模式可在“高度并行（embarrassingly parallel）”任务上最大化吞吐量。</p>

## 25.6.4　流水线模式（Pipeline Pattern） {#pipeline-pattern}

<p>Agent 构成一条顺序处理链：每个 Agent 转换前一个 Agent 的输出。类似于 Unix 管道。适用于具有明确顺序依赖的任务（例如：研究 <span class="arithmatex">\(\to\)</span> 大纲 <span class="arithmatex">\(\to\)</span> 草稿 <span class="arithmatex">\(\to\)</span> 编辑 <span class="arithmatex">\(\to\)</span> 排版）。</p>

## 25.6.5　集成模式（Ensemble Pattern） {#ensemble-pattern}

<p>多个 Agent 独立求解同一问题；由某种选择机制挑出最佳答案（best-of-<span class="arithmatex">\(N\)</span>），或聚合答案（mixture-of-experts 风格）。以算力为代价提升可靠性。</p>



<div class="arithmatex">\[o^* = \arg\max_{o \in \{o_1,\ldots,o_N\}} \text{score}(o, \text{task})
\tag{25.3}\]</div>

<p>其中 <span class="arithmatex">\(\text{score}\)</span> 可以是一个 Reward 模型、一个裁判 LLM 或一个验证器。</p>

## 25.6.6　师生模式（Teacher-Student Pattern） {#teacher-student-pattern}

<p>能力更强的 Agent（teacher）引导能力较弱的 Agent（student）完成任务，提供提示、纠正和解释。该模式在推理阶段实现知识蒸馏，也可用于微调 student Agent。</p>

## 25.6.7　红队模式（Red Team Pattern） {#red-team-pattern}

<p>一个对抗性 Agent（red team）主动尝试在其他 Agent 的输出中寻找弱点、错误或安全违规。Red team Agent 被 Prompt 引导以尽可能挑剔与富有创造性的方式发起攻击。该模式对安全关键应用至关重要。</p>

<div class="admonition example"><p class="admonition-title">Red Team Agent 的 Prompt</p>
<div class="highlight"><pre><span></span><code><span class="n">RED_TEAM_PROMPT</span> <span class="o">=</span> <span class="s2">"""You are a red team agent. Your job is to find</span>
<span class="s2">flaws, errors, biases, safety violations, and failure modes in the</span>
<span class="s2">following output. Be adversarial, creative, and thorough.</span>

<span class="s2">Consider:</span>
<span class="s2">1. Factual errors or hallucinations</span>
<span class="s2">2. Logical inconsistencies</span>
<span class="s2">3. Safety and ethical concerns</span>
<span class="s2">4. Edge cases the solution doesn't handle</span>
<span class="s2">5. Ways a malicious user could exploit this output</span>
<span class="s2">6. Unintended consequences</span>

<span class="s2">Output: </span><span class="si">{agent_output}</span>

<span class="s2">Provide a detailed critique with specific examples of each flaw found."""</span>
</code></pre></div>
</div>
