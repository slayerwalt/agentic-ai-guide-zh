# 8.5　Best-of-N 采样（拒绝采样） {#best-of-n-sampling-rejection-sampling}

## 8.5.1　动机 {#motivation-4}

<p>有时候最简单的方法获胜。Best-of-N 采样（Rejection Sampling） <a class="book-citation" id="cite-nakano2021webgpt-1" href="../../references.md#ref-nakano2021webgpt" aria-label="参考文献 262">[262]</a> 在 RL 阶段 <em>完全不需要训练</em>——只需生成多个候选并挑选最佳。</p>

## 8.5.2　算法 {#algorithm-1}

<ol>
<li><p>对每个 prompt，从 policy 生成 <span class="arithmatex">\(N\)</span> 条响应（通常 <span class="arithmatex">\(N = 4\)</span>–<span class="arithmatex">\(64\)</span>）</p></li>
<li><p>用 reward model 给所有响应打分</p></li>
<li><p>选择最高分响应</p></li>
<li><p>（可选）将选中的响应作为下一轮的 SFT 数据</p></li>
</ol>



<div class="arithmatex">\[\boxed{\text{Best-of-N response}: \quad y^* = \arg\max_{y_i \sim \pi_\theta(\cdot|x)} r_\phi(x, y_i)}
\tag{8.4}\]</div>

<div class="admonition tip"><p class="admonition-title">为什么 Best-of-N 是合法的&quot;RL&quot;方法</p>
<p><strong>推理时</strong>：Best-of-N 在不改动模型权重的情况下提升输出质量。<span class="arithmatex">\(N=64\)</span> 时胜率比 greedy 提升 10–20%——有时匹配甚至超过 PPO。</p>

<p><strong>作为训练方法</strong>（拒绝采样微调 / RFT）：</p>

<ol>
<li><p>生成大量响应，选出最佳</p></li>
<li><p>在选出的响应上做 SFT</p></li>
<li><p>重复（迭代精炼）</p></li>
</ol>

<p>许多生产模型就是这样训练的：比 PPO 简单、几乎同样有效、完全稳定。</p>

<p><strong>理论联系</strong> <a class="book-citation" id="cite-gao2023scaling-1" href="../../references.md#ref-gao2023scaling" aria-label="参考文献 105">[105]</a>：Best-of-N 实现了隐式的 KL 约束 policy：<span class="arithmatex">\(\pi_\text{BoN}(y|x) \propto \pi_\theta(y|x)^{1-1/N} \cdot r(x,y)^{1/N}\)</span>。</p>
</div>

## 8.5.3　TRL 实现 {#trl-implementation-4}

<p>下面是基于 HuggingFace TRL 的最小可运行示例。</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">pipeline</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>

<span class="c1"># 推理时 Best-of-N（手工实现）</span>
<span class="n">gen_pipeline</span> <span class="o">=</span> <span class="n">pipeline</span><span class="p">(</span><span class="s2">"text-generation"</span><span class="p">,</span> <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span> <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">)</span>

<span class="k">def</span><span class="w"> </span><span class="nf">best_of_n</span><span class="p">(</span><span class="n">prompt</span><span class="p">,</span> <span class="n">n</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span> <span class="n">temperature</span><span class="o">=</span><span class="mf">0.8</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""生成 N 个候选并返回最高 reward 者。"""</span>
    <span class="n">candidates</span> <span class="o">=</span> <span class="n">gen_pipeline</span><span class="p">(</span>
        <span class="n">prompt</span><span class="p">,</span> <span class="n">num_return_sequences</span><span class="o">=</span><span class="n">n</span><span class="p">,</span>
        <span class="n">temperature</span><span class="o">=</span><span class="n">temperature</span><span class="p">,</span> <span class="n">do_sample</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">max_new_tokens</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">scores</span> <span class="o">=</span> <span class="p">[</span><span class="n">reward_model</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">prompt</span><span class="p">,</span> <span class="n">c</span><span class="p">[</span><span class="s2">"generated_text"</span><span class="p">])</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">candidates</span><span class="p">]</span>
    <span class="k">return</span> <span class="n">candidates</span><span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">scores</span><span class="p">)][</span><span class="s2">"generated_text"</span><span class="p">]</span>

<span class="n">best_response</span> <span class="o">=</span> <span class="n">best_of_n</span><span class="p">(</span><span class="n">prompt</span><span class="p">,</span> <span class="n">n</span><span class="o">=</span><span class="mi">16</span><span class="p">)</span>

<span class="c1"># 训练：拒绝采样微调（RFT）</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">SFTConfig</span><span class="p">,</span> <span class="n">SFTTrainer</span>

<span class="c1"># 步骤 1：生成并筛选</span>
<span class="n">all_responses</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">prompt</span> <span class="ow">in</span> <span class="n">prompts</span><span class="p">:</span>
    <span class="n">candidates</span> <span class="o">=</span> <span class="p">[</span><span class="n">generate</span><span class="p">(</span><span class="n">prompt</span><span class="p">,</span> <span class="n">temp</span><span class="o">=</span><span class="mf">0.9</span><span class="p">)</span> <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">16</span><span class="p">)]</span>
    <span class="n">scores</span> <span class="o">=</span> <span class="p">[</span><span class="n">reward_model</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">prompt</span><span class="p">,</span> <span class="n">c</span><span class="p">)</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">candidates</span><span class="p">]</span>
    <span class="n">best_idx</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">scores</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">scores</span><span class="p">[</span><span class="n">best_idx</span><span class="p">]</span> <span class="o">&gt;</span> <span class="n">threshold</span><span class="p">:</span>  <span class="c1"># 质量门槛</span>
        <span class="n">all_responses</span><span class="o">.</span><span class="n">append</span><span class="p">({</span><span class="s2">"prompt"</span><span class="p">:</span> <span class="n">prompt</span><span class="p">,</span> <span class="s2">"completion"</span><span class="p">:</span> <span class="n">candidates</span><span class="p">[</span><span class="n">best_idx</span><span class="p">]})</span>

<span class="c1"># 步骤 2：在最佳响应上做 SFT</span>
<span class="n">sft_config</span> <span class="o">=</span> <span class="n">SFTConfig</span><span class="p">(</span><span class="n">output_dir</span><span class="o">=</span><span class="s2">"./rft_output"</span><span class="p">,</span> <span class="n">learning_rate</span><span class="o">=</span><span class="mf">2e-5</span><span class="p">,</span> <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">max_seq_length</span><span class="o">=</span><span class="mi">2048</span><span class="p">)</span>
<span class="n">trainer</span> <span class="o">=</span> <span class="n">SFTTrainer</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span> <span class="n">args</span><span class="o">=</span><span class="n">sft_config</span><span class="p">,</span> <span class="n">train_dataset</span><span class="o">=</span><span class="n">all_responses</span><span class="p">,</span> <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
<span class="c1"># 步骤 3：用更新后的模型从步骤 1 重复（迭代 RFT）</span>
</code></pre></div>

## 8.5.4　Best-of-N 的扩展律 {#scaling-laws-for-best-of-n}

<span id="table-8-inline-2"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><span class="arithmatex">\(N\)</span></th><th scope="col"><strong>质量增益</strong></th><th scope="col"><strong>成本</strong></th><th scope="col"><strong>说明</strong></th><th scope="col"></th></tr></thead>
<tbody><tr><td>1</td><td>基线</td><td><span class="arithmatex">\(1\times\)</span></td><td>标准采样</td><td></td></tr>
<tr><td>4</td><td>+5–8% 胜率</td><td><span class="arithmatex">\(4\times\)</span></td><td>最低有用值。良好的成本/质量比</td><td></td></tr>
<tr><td>16</td><td>+10–15% 胜率</td><td><span class="arithmatex">\(16\times\)</span></td><td>强力。常匹配 PPO 质量</td><td></td></tr>
<tr><td>64</td><td>+15–20% 胜率</td><td><span class="arithmatex">\(64\times\)</span></td><td>开始边际递减</td><td></td></tr>
<tr><td>256</td><td>+18–22% 胜率</td><td><span class="arithmatex">\(256\times\)</span></td><td>仅用于关键应用</td><td></td></tr></tbody>
</table>
</div>

<div class="admonition note"><p class="admonition-title">Best-of-N 作为基线</p>
<p>始终在相同算力预算下将你的 RL 方法与 Best-of-N 比较。若 64 GPU-小时的 PPO 打不过 64 GPU-小时生成的 Best-of-N，你的 PPO 有 bug。</p>
</div>
