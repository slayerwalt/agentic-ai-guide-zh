# 25.9　现实世界中的多智能体应用 {#subsec:mas-applications}

## 25.9.1　软件开发团队 {#software-development-team}

<p>一支多 Agent 软件开发团队映射了真实的工程组织：</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">dataclasses</span><span class="w"> </span><span class="kn">import</span> <span class="n">dataclass</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Optional</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">asyncio</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">SoftwareTeamState</span><span class="p">:</span>
    <span class="n">requirements</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">architecture</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">code</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">tests</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">review_feedback</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">final_code</span><span class="p">:</span> <span class="n">Optional</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">approved</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>

<span class="k">class</span><span class="w"> </span><span class="nc">SoftwareDevelopmentTeam</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    多 agent 软件团队：</span>
<span class="sd">      Architect -&gt; Coder -&gt; Tester -&gt; Reviewer -&gt; （迭代或交付）</span>
<span class="sd">    """</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">llm_factory</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">architect</span> <span class="o">=</span> <span class="n">llm_factory</span><span class="p">(</span>
            <span class="n">system_prompt</span><span class="o">=</span><span class="s2">"""You are a software architect. Given requirements,</span>
<span class="s2">            produce a clear technical design: components, interfaces, data</span>
<span class="s2">            structures, and implementation plan."""</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">coder</span> <span class="o">=</span> <span class="n">llm_factory</span><span class="p">(</span>
            <span class="n">system_prompt</span><span class="o">=</span><span class="s2">"""You are an expert software engineer. Given a</span>
<span class="s2">            technical design, write clean, well-documented, production-ready</span>
<span class="s2">            code. Follow best practices for the language."""</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tester</span> <span class="o">=</span> <span class="n">llm_factory</span><span class="p">(</span>
            <span class="n">system_prompt</span><span class="o">=</span><span class="s2">"""You are a QA engineer. Given code, write</span>
<span class="s2">            comprehensive tests: unit tests, edge cases, integration tests.</span>
<span class="s2">            Identify potential bugs and failure modes."""</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">reviewer</span> <span class="o">=</span> <span class="n">llm_factory</span><span class="p">(</span>
            <span class="n">system_prompt</span><span class="o">=</span><span class="s2">"""You are a senior code reviewer. Evaluate code</span>
<span class="s2">            for correctness, security, performance, and maintainability.</span>
<span class="s2">            Provide specific, actionable feedback. Approve only if excellent."""</span>
        <span class="p">)</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">build</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">requirements</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                    <span class="n">max_iterations</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">3</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">SoftwareTeamState</span><span class="p">:</span>
        <span class="n">state</span> <span class="o">=</span> <span class="n">SoftwareTeamState</span><span class="p">(</span><span class="n">requirements</span><span class="o">=</span><span class="n">requirements</span><span class="p">)</span>

        <span class="c1"># 阶段 1：架构</span>
        <span class="n">state</span><span class="o">.</span><span class="n">architecture</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">architect</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">"Requirements:</span><span class="se">\n</span><span class="si">{</span><span class="n">requirements</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">Produce technical design."</span>
        <span class="p">)</span>

        <span class="k">for</span> <span class="n">iteration</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">max_iterations</span><span class="p">):</span>
            <span class="c1"># 阶段 2：实现</span>
            <span class="n">prompt</span> <span class="o">=</span> <span class="p">(</span><span class="sa">f</span><span class="s2">"Design:</span><span class="se">\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">architecture</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>
                      <span class="o">+</span> <span class="p">(</span><span class="sa">f</span><span class="s2">"Previous feedback:</span><span class="se">\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">review_feedback</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>
                         <span class="k">if</span> <span class="n">state</span><span class="o">.</span><span class="n">review_feedback</span> <span class="k">else</span> <span class="s2">""</span><span class="p">)</span>
                      <span class="o">+</span> <span class="s2">"Write the implementation."</span><span class="p">)</span>
            <span class="n">state</span><span class="o">.</span><span class="n">code</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">coder</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span><span class="n">prompt</span><span class="p">)</span>

            <span class="c1"># 阶段 3：测试</span>
            <span class="n">state</span><span class="o">.</span><span class="n">tests</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">tester</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"Code:</span><span class="se">\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">code</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">Write comprehensive tests."</span>
            <span class="p">)</span>

            <span class="c1"># 阶段 4：评审</span>
            <span class="n">review</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">reviewer</span><span class="o">.</span><span class="n">invoke</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"Code:</span><span class="se">\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">code</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">Tests:</span><span class="se">\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">tests</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>
                <span class="s2">"Review this code. End with APPROVED or NEEDS_REVISION."</span>
            <span class="p">)</span>

            <span class="k">if</span> <span class="s2">"APPROVED"</span> <span class="ow">in</span> <span class="n">review</span><span class="p">:</span>
                <span class="n">state</span><span class="o">.</span><span class="n">final_code</span> <span class="o">=</span> <span class="n">state</span><span class="o">.</span><span class="n">code</span>
                <span class="n">state</span><span class="o">.</span><span class="n">approved</span> <span class="o">=</span> <span class="kc">True</span>
                <span class="k">break</span>
            <span class="k">else</span><span class="p">:</span>
                <span class="n">state</span><span class="o">.</span><span class="n">review_feedback</span> <span class="o">=</span> <span class="n">review</span>

        <span class="k">return</span> <span class="n">state</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">run</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">requirements</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="n">state</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">build</span><span class="p">(</span><span class="n">requirements</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">state</span><span class="o">.</span><span class="n">approved</span><span class="p">:</span>
            <span class="k">return</span> <span class="sa">f</span><span class="s2">"# Final Implementation</span><span class="se">\n\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">final_code</span><span class="si">}</span><span class="s2">"</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="k">return</span> <span class="sa">f</span><span class="s2">"# Best Attempt (not approved)</span><span class="se">\n\n</span><span class="si">{</span><span class="n">state</span><span class="o">.</span><span class="n">code</span><span class="si">}</span><span class="s2">"</span>
</code></pre></div>

## 25.9.2　研究团队 {#research-team}

<p>一个研究团队 Agent 社会映射了学术合作：</p>

<ul>
<li><p><strong>Literature Reviewer</strong>：检索并综合现有工作</p></li>
<li><p><strong>Hypothesis Generator</strong>：提出新的研究方向</p></li>
<li><p><strong>Experimentalist</strong>：通过执行代码设计并运行实验</p></li>
<li><p><strong>Statistician</strong>：分析结果并评估显著性</p></li>
<li><p><strong>Writer</strong>：将发现综合成连贯的报告</p></li>
</ul>

## 25.9.3　客户服务系统 {#customer-service-system}

<p>一个分级客户服务系统：</p>

<ul>
<li><p><strong>Router</strong>：分类进入的请求并路由到专员</p></li>
<li><p><strong>Billing Specialist</strong>：处理支付与账户问题</p></li>
<li><p><strong>Technical Specialist</strong>：解决产品/服务问题</p></li>
<li><p><strong>Escalation Agent</strong>：处理需要人类判断的复杂情形</p></li>
</ul>

## 25.9.4　创意团队 {#creative-team}

<p>一条创意生产流水线：</p>

<ul>
<li><p><strong>Brainstormer</strong>：无自我审查地产生多样化想法</p></li>
<li><p><strong>Drafter</strong>：将最有前景的想法发展成完整草稿</p></li>
<li><p><strong>Editor</strong>：在清晰度、风格与连贯性上精修草稿</p></li>
<li><p><strong>Critic</strong>：提供对抗式反馈以加强作品</p></li>
</ul>
