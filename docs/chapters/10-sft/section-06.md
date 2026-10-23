# 10.6　与 RL 的关系 —— SFT 质量决定 RL 上限 {#sec:sft-rl-connection}

<div class="admonition note"><p class="admonition-title">SFT-RL 关系</p>
<p>SFT 模型是 RL 训练的起点。RL 可以：</p>

<ul>
<li><p><strong>放大</strong>SFT 模型中已存在但较弱的行为。</p></li>
<li><p><strong>抑制</strong>已存在但不理想的行为。</p></li>
<li><p><strong>精炼</strong>回答的风格与格式。</p></li>
</ul>

<p>RL <em>无法</em>：</p>

<ul>
<li><p>引入 SFT 模型中完全不存在的能力。</p></li>
<li><p>从 SFT 阶段严重的灾难性遗忘中恢复。</p></li>
<li><p>弥补系统性有偏的奖励模型。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">SFT 中的探索-利用权衡</p>
<p>为使 RL 起作用，SFT 模型必须偶尔产生正确响应（从而 Reward 信号非零）。如果 SFT 模型对某 prompt <em>从不</em>产生正确响应，RL 无法学会产生正确响应 —— 没有可放大的正向信号。这就是为什么 SFT 质量是 RL 性能的上限。</p>

<p>具体而言：若 SFT 模型能正确解答 10% 的数学题，RL 有可能将其推至 80%。若 SFT 模型解题率为 0%,RL 将毫无进展（所有 Reward 为零，所有优势为零，无 Gradient）。</p>
</div>

### 实践启示 {#practical-implications}

<ol>
<li><p><strong>SFT 数据质量</strong>：使用高质量、多样化的数据。少量高质量数据胜过大量低质量数据。</p></li>
<li><p><strong>SFT 数据覆盖度</strong>：确保 SFT 数据覆盖你想通过 RL 改进的任务。如果某任务不在 SFT 数据中，RL 将难以推进。</p></li>
<li><p><strong>SFT 训练时长</strong>：不要过度训练 SFT 模型。过度训练降低多样性，让 RL 探索更困难。</p></li>
<li><p><strong>Warm-up</strong>：即使 base 模型已经过指令微调，在 RL 之前也可考虑在特定任务数据上进行简短的 SFT warm-up。</p></li>
</ol>

<div class="admonition example"><p class="admonition-title">在 RL 之前检查 SFT 质量</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">tqdm</span><span class="w"> </span><span class="kn">import</span> <span class="n">tqdm</span>

<span class="k">def</span><span class="w"> </span><span class="nf">estimate_pass_at_k</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">tokenizer</span><span class="p">,</span> <span class="n">dataset</span><span class="p">,</span> <span class="n">k</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span> <span class="n">n_samples</span><span class="o">=</span><span class="mi">100</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    估计 SFT 模型的 pass@k。</span>
<span class="sd">    若 pass@1 &lt; 5%,RL 大概率会失败。</span>
<span class="sd">    若 pass@k &lt; 20%,RL 会很吃力。</span>
<span class="sd">    """</span>
    <span class="n">pass_at_1_scores</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">pass_at_k_scores</span> <span class="o">=</span> <span class="p">[]</span>

    <span class="k">for</span> <span class="n">example</span> <span class="ow">in</span> <span class="n">tqdm</span><span class="p">(</span><span class="n">dataset</span><span class="o">.</span><span class="n">select</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">n_samples</span><span class="p">))):</span>
        <span class="n">prompt</span> <span class="o">=</span> <span class="n">example</span><span class="p">[</span><span class="s2">"prompt"</span><span class="p">]</span>
        <span class="n">ground_truth</span> <span class="o">=</span> <span class="n">example</span><span class="p">[</span><span class="s2">"answer"</span><span class="p">]</span>

        <span class="c1"># 采样 k 个补全</span>
        <span class="n">inputs</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="p">(</span><span class="n">prompt</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span><span class="p">)</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">device</span><span class="p">)</span>
        <span class="n">outputs</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">generate</span><span class="p">(</span>
            <span class="o">**</span><span class="n">inputs</span><span class="p">,</span>
            <span class="n">max_new_tokens</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span>
            <span class="n">do_sample</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
            <span class="n">temperature</span><span class="o">=</span><span class="mf">0.8</span><span class="p">,</span>
            <span class="n">num_return_sequences</span><span class="o">=</span><span class="n">k</span><span class="p">,</span>
        <span class="p">)</span>

        <span class="n">correct</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="k">for</span> <span class="n">output</span> <span class="ow">in</span> <span class="n">outputs</span><span class="p">:</span>
            <span class="n">response</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">output</span><span class="p">,</span> <span class="n">skip_special_tokens</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
            <span class="k">if</span> <span class="n">ground_truth</span> <span class="ow">in</span> <span class="n">response</span><span class="p">:</span>
                <span class="n">correct</span> <span class="o">+=</span> <span class="mi">1</span>

        <span class="c1"># pass@1:正确样本占比(估计的成功率)</span>
        <span class="n">pass_at_1_scores</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">correct</span> <span class="o">/</span> <span class="n">k</span><span class="p">)</span>
        <span class="c1"># pass@k:k 个样本中至少一个正确</span>
        <span class="n">pass_at_k_scores</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">correct</span> <span class="o">&gt;=</span> <span class="mi">1</span><span class="p">)</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Pass@1 (estimated): </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">pass_at_1_scores</span><span class="p">)</span><span class="si">:</span><span class="s2">.2%</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Pass@</span><span class="si">{</span><span class="n">k</span><span class="si">}</span><span class="s2">: </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">pass_at_k_scores</span><span class="p">)</span><span class="si">:</span><span class="s2">.2%</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"RL viability: </span><span class="si">{</span><span class="s1">'Good'</span><span class="w"> </span><span class="k">if</span><span class="w"> </span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">pass_at_1_scores</span><span class="p">)</span><span class="w"> </span><span class="o">&gt;</span><span class="w"> </span><span class="mf">0.05</span><span class="w"> </span><span class="k">else</span><span class="w"> </span><span class="s1">'Poor'</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

<span class="n">estimate_pass_at_k</span><span class="p">(</span><span class="n">sft_model</span><span class="p">,</span> <span class="n">tokenizer</span><span class="p">,</span> <span class="n">eval_dataset</span><span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">SFT 最佳实践总结</p>
<ol>
<li><p>使用序列打包（Sequence Packing）以最大化 GPU 利用率。</p></li>
<li><p>使用 completion-only masking 将 Gradient 聚焦于 assistant 响应。</p></li>
<li><p>为你的模型系列使用正确的 chat 模板。</p></li>
<li><p>多任务 SFT 时使用温度缩放（<span class="arithmatex">\(T \approx 2\)</span>）按比例混合数据。</p></li>
<li><p>使用 LoRA 防止灾难性遗忘。</p></li>
<li><p>在开始 RL 前评估 pass@k，以确保 SFT 模型是可行的起点。</p></li>
<li><p>不要过度训练：指令微调通常 1–3 个 Epoch 就足够。</p></li>
<li><p>监控多样性指标（熵、n-gram 多样性）以检测 mode collapse。</p></li>
</ol>
</div>
