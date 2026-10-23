# 19.4　循环的伪代码 {#sec:loop-pseudocode}

<p>剥去所有表象，每个 Agent 循环都是一种控制结构，更像恒温器或 REPL，而不是一段对话：</p>

<div class="highlight"><pre><span></span><code><span class="k">def</span><span class="w"> </span><span class="nf">agent_loop</span><span class="p">(</span><span class="n">goal</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">max_steps</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">50</span><span class="p">,</span> <span class="n">budget</span><span class="p">:</span> <span class="n">TokenBudget</span> <span class="o">=</span> <span class="kc">None</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""The canonical agent loop structure."""</span>
    <span class="n">state</span> <span class="o">=</span> <span class="n">initialize_state</span><span class="p">(</span><span class="n">goal</span><span class="p">)</span>        <span class="c1"># 加载外部状态和目标</span>

    <span class="k">for</span> <span class="n">step</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">max_steps</span><span class="p">):</span>
        <span class="c1"># 1. 推理当前状态（ReAct 风格）</span>
        <span class="n">thought</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">reason</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>

        <span class="c1"># 2. 选择并执行动作</span>
        <span class="n">action</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">choose_action</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>
        <span class="n">observation</span> <span class="o">=</span> <span class="n">environment</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="n">action</span><span class="p">)</span>

        <span class="c1"># 3. 用新信息更新状态</span>
        <span class="n">state</span> <span class="o">=</span> <span class="n">update_state</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">thought</span><span class="p">,</span> <span class="n">action</span><span class="p">,</span> <span class="n">observation</span><span class="p">)</span>

        <span class="c1"># 4. 接近上下文窗口上限时压缩上下文</span>
        <span class="k">if</span> <span class="n">state</span><span class="o">.</span><span class="n">token_count</span> <span class="o">&gt;</span> <span class="n">CONTEXT_BUDGET</span> <span class="o">*</span> <span class="mf">0.8</span><span class="p">:</span>
            <span class="n">state</span> <span class="o">=</span> <span class="n">compact</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>        <span class="c1"># 汇总旧步骤</span>

        <span class="c1"># 5. 检查终止条件</span>
        <span class="k">if</span> <span class="n">verifier</span><span class="o">.</span><span class="n">passes</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">goal</span><span class="p">):</span>  <span class="c1"># 确定性检查</span>
            <span class="n">persist_state</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">status</span><span class="o">=</span><span class="s2">"success"</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">Success</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>

        <span class="k">if</span> <span class="n">no_progress_detected</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">window</span><span class="o">=</span><span class="mi">3</span><span class="p">):</span>
            <span class="n">persist_state</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">status</span><span class="o">=</span><span class="s2">"stuck"</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">escalate_to_human</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>

        <span class="k">if</span> <span class="n">budget</span> <span class="ow">and</span> <span class="n">budget</span><span class="o">.</span><span class="n">exhausted</span><span class="p">():</span>
            <span class="n">persist_state</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">status</span><span class="o">=</span><span class="s2">"budget_exceeded"</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">escalate_to_human</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>

    <span class="c1"># 达到硬性上限</span>
    <span class="n">persist_state</span><span class="p">(</span><span class="n">state</span><span class="p">,</span> <span class="n">status</span><span class="o">=</span><span class="s2">"max_steps"</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">escalate_to_human</span><span class="p">(</span><span class="n">state</span><span class="p">)</span>
</code></pre></div>

<p>循环工程中所有有趣的地方，都是关于这些行之一的决策：什么构成有效的 <code>goal</code>，<code>verifier.passes</code> 如何实现，<code>compact</code> 如何在丢弃噪声的同时保留相关历史，以及 <code>no_progress_detected</code> 如何既避免过早终止又避免无限循环。</p>
