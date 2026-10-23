# 9.6　多目标 Reward —— 组合策略 {#sec:multi-objective-rewards}

<p>当使用多个 Reward 信号进行训练时，组合策略会显著影响最终 Policy。</p>

<div class="admonition note"><p class="admonition-title">多 Reward 组合策略</p>
<ol>
<li><p><strong>加权求和（Weighted sum）</strong>：<span class="arithmatex">\(r = \sum_n w_n r_n\)</span>。简单但对尺度敏感。</p></li>
<li><p><strong>先归一化再求和（GDPO）</strong>：在组内将每个 Reward 归一化为零均值、单位方差，再加权求和。尺度无关。</p></li>
<li><p><strong>字典序（Lexicographic）</strong>：按优先级顺序优化 Reward；仅当高优先级 Reward 平局时才考虑低优先级的 Reward。</p></li>
<li><p><strong>约束法（Constrained）</strong>：在次要 Reward 的约束下最大化主 Reward。</p></li>
<li><p><strong>Pareto</strong>：维护一组 Policy 的 Pareto 前沿，并根据偏好进行选择。</p></li>
</ol>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的多 Reward 训练</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="c1"># GDPO:对每个 Reward 独立归一化</span>
    <span class="n">multi_objective_aggregation</span><span class="o">=</span><span class="s2">"normalize_then_sum"</span><span class="p">,</span>
    <span class="n">reward_weights</span><span class="o">=</span><span class="p">[</span><span class="mf">1.0</span><span class="p">,</span> <span class="mf">0.3</span><span class="p">,</span> <span class="mf">0.1</span><span class="p">],</span>  <span class="c1"># 正确性、格式、长度</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">GRPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">reward_funcs</span><span class="o">=</span><span class="p">[</span>
        <span class="n">correctness_reward</span><span class="p">,</span>
        <span class="n">format_reward</span><span class="p">,</span>
        <span class="n">length_penalty_reward</span><span class="p">,</span>
    <span class="p">],</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>
