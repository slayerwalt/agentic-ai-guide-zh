# 9.5　RLVR 中基于规则的 Reward {#sec:rule-based-rewards}

<p>可验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards, RLVR）使用确定性、基于规则的 Reward 函数，而非习得的奖励模型。这能显著减少奖励黑客行为（尽管模型仍可能利用格式技巧、边界情况或测试记忆来作弊），DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 即采用此方法。</p>

<div class="admonition example"><p class="admonition-title">TRL 中基于规则的 Reward 函数</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">re</span>

<span class="k">def</span><span class="w"> </span><span class="nf">format_reward</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""对使用 &lt;think&gt;...&lt;/think&gt;&lt;answer&gt;...&lt;/answer&gt; 格式给予 Reward。"""</span>
    <span class="n">rewards</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">pattern</span> <span class="o">=</span> <span class="sa">r</span><span class="s2">"&lt;think&gt;.*?&lt;/think&gt;\s*&lt;answer&gt;.*?&lt;/answer&gt;"</span>
    <span class="k">for</span> <span class="n">completion</span> <span class="ow">in</span> <span class="n">completions</span><span class="p">:</span>
        <span class="n">text</span> <span class="o">=</span> <span class="n">completion</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s2">"content"</span><span class="p">]</span>
        <span class="n">rewards</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mf">1.0</span> <span class="k">if</span> <span class="n">re</span><span class="o">.</span><span class="n">fullmatch</span><span class="p">(</span><span class="n">pattern</span><span class="p">,</span> <span class="n">text</span><span class="p">,</span> <span class="n">re</span><span class="o">.</span><span class="n">DOTALL</span><span class="p">)</span> <span class="k">else</span> <span class="mf">0.0</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">rewards</span>

<span class="k">def</span><span class="w"> </span><span class="nf">correctness_reward</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="n">ground_truth</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""对最终答案正确给予 Reward。"""</span>
    <span class="n">rewards</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">completion</span><span class="p">,</span> <span class="n">gt</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="n">ground_truth</span><span class="p">):</span>
        <span class="n">text</span> <span class="o">=</span> <span class="n">completion</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s2">"content"</span><span class="p">]</span>
        <span class="n">match</span> <span class="o">=</span> <span class="n">re</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="sa">r</span><span class="s2">"&lt;answer&gt;(.*?)&lt;/answer&gt;"</span><span class="p">,</span> <span class="n">text</span><span class="p">,</span> <span class="n">re</span><span class="o">.</span><span class="n">DOTALL</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">match</span><span class="p">:</span>
            <span class="n">answer</span> <span class="o">=</span> <span class="n">match</span><span class="o">.</span><span class="n">group</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span>
            <span class="n">rewards</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mf">1.0</span> <span class="k">if</span> <span class="n">answer</span> <span class="o">==</span> <span class="n">gt</span> <span class="k">else</span> <span class="mf">0.0</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">rewards</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mf">0.0</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">rewards</span>

<span class="k">def</span><span class="w"> </span><span class="nf">code_execution_reward</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="n">test_cases</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""对通过测试用例的代码给予 Reward。"""</span>
    <span class="kn">import</span><span class="w"> </span><span class="nn">subprocess</span><span class="o">,</span><span class="w"> </span><span class="nn">tempfile</span><span class="o">,</span><span class="w"> </span><span class="nn">os</span>
    <span class="n">rewards</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">completion</span><span class="p">,</span> <span class="n">tests</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="n">test_cases</span><span class="p">):</span>
        <span class="n">code</span> <span class="o">=</span> <span class="n">completion</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s2">"content"</span><span class="p">]</span>
        <span class="n">passed</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="k">for</span> <span class="n">test</span> <span class="ow">in</span> <span class="n">tests</span><span class="p">:</span>
            <span class="k">with</span> <span class="n">tempfile</span><span class="o">.</span><span class="n">NamedTemporaryFile</span><span class="p">(</span>
                <span class="n">mode</span><span class="o">=</span><span class="s2">"w"</span><span class="p">,</span> <span class="n">suffix</span><span class="o">=</span><span class="s2">".py"</span><span class="p">,</span> <span class="n">delete</span><span class="o">=</span><span class="kc">False</span>
            <span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
                <span class="n">f</span><span class="o">.</span><span class="n">write</span><span class="p">(</span><span class="n">code</span> <span class="o">+</span> <span class="s2">"</span><span class="se">\n</span><span class="s2">"</span> <span class="o">+</span> <span class="n">test</span><span class="p">)</span>
                <span class="n">fname</span> <span class="o">=</span> <span class="n">f</span><span class="o">.</span><span class="n">name</span>
            <span class="k">try</span><span class="p">:</span>
                <span class="n">result</span> <span class="o">=</span> <span class="n">subprocess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
                    <span class="p">[</span><span class="s2">"python"</span><span class="p">,</span> <span class="n">fname</span><span class="p">],</span> <span class="n">capture_output</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                    <span class="n">timeout</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">text</span><span class="o">=</span><span class="kc">True</span>
                <span class="p">)</span>
                <span class="n">passed</span> <span class="o">+=</span> <span class="nb">int</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">returncode</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span>
            <span class="k">except</span> <span class="ne">Exception</span><span class="p">:</span>
                <span class="k">pass</span>
            <span class="k">finally</span><span class="p">:</span>
                <span class="n">os</span><span class="o">.</span><span class="n">unlink</span><span class="p">(</span><span class="n">fname</span><span class="p">)</span>
        <span class="n">rewards</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">passed</span> <span class="o">/</span> <span class="nb">len</span><span class="p">(</span><span class="n">tests</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">rewards</span>
</code></pre></div>
</div>

<div class="admonition warning"><p class="admonition-title">基于规则 Reward 的陷阱</p>
<ul>
<li><p><strong>格式作弊（Format gaming）</strong>：模型学会生成正确格式但内容错误。务必将格式 Reward 与正确性 Reward 结合使用。</p></li>
<li><p><strong>测试用例泄漏</strong>：如果测试用例存在于训练数据中，模型会将其记住。</p></li>
<li><p><strong>超时利用</strong>：模型可能生成会超时的代码（以回避失败）。请使用严格的超时并显式惩罚超时。</p></li>
<li><p><strong>Reward 稀疏性</strong>：二元 Reward(0/1)对复杂任务而言可能过于稀疏。考虑使用部分得分或中间 Reward。</p></li>
</ul>
</div>
