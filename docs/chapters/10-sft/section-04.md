# 10.4　多任务 SFT 的数据混合策略 {#sec:data-mixing}

<div class="admonition tip"><p class="admonition-title">多任务挑战</p>
<p>同时在多任务上训练能提升泛化能力，但也会引发<em>任务干扰</em>：不同任务的 Gradient 相互冲突，降低单一任务的性能。数据混合策略用于控制各任务对训练信号的相对贡献。</p>
</div>

### 比例混合 {#proportional-mixing}

<p>按各数据集的规模成比例地采样：</p>



<div class="arithmatex">\[p_k = \frac{N_k}{\sum_{j=1}^K N_j},\]</div>

<p>其中 <span class="arithmatex">\(N_k\)</span> 是数据集 <span class="arithmatex">\(k\)</span> 中的样本数。这是大多数框架的默认方式，在数据集质量相近时效果良好。</p>

### 温度混合 {#temperature-mixing}

<p>应用温度 <span class="arithmatex">\(T\)</span> 来平滑比例：</p>



<div class="arithmatex">\[p_k \propto N_k^{1/T}.\]</div>

<p><span class="arithmatex">\(T=1\)</span>：比例混合。<span class="arithmatex">\(T \to \infty\)</span>：均匀混合。<span class="arithmatex">\(T &lt; 1\)</span>：对大数据集过采样。<span class="arithmatex">\(T &gt; 1\)</span>：对小数据集过采样。</p>

### 基于质量加权的混合 {#quality-weighted-mixing}

<p>按估计的质量（例如在参考模型下的困惑度、人工质量评分）对数据集加权：</p>



<div class="arithmatex">\[p_k \propto N_k \cdot q_k,\]</div>

<p>其中 <span class="arithmatex">\(q_k\)</span> 是数据集 <span class="arithmatex">\(k\)</span> 的质量分数。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的数据混合</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">datasets</span><span class="w"> </span><span class="kn">import</span> <span class="n">concatenate_datasets</span><span class="p">,</span> <span class="n">interleave_datasets</span>

<span class="c1"># 比例混合(默认)</span>
<span class="n">mixed_dataset</span> <span class="o">=</span> <span class="n">concatenate_datasets</span><span class="p">([</span>
    <span class="n">dataset_math</span><span class="p">,</span>
    <span class="n">dataset_code</span><span class="p">,</span>
    <span class="n">dataset_general</span><span class="p">,</span>
<span class="p">])</span><span class="o">.</span><span class="n">shuffle</span><span class="p">(</span><span class="n">seed</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>

<span class="c1"># 温度混合(T=2:对小数据集过采样)</span>
<span class="n">mixed_dataset</span> <span class="o">=</span> <span class="n">interleave_datasets</span><span class="p">(</span>
    <span class="p">[</span><span class="n">dataset_math</span><span class="p">,</span> <span class="n">dataset_code</span><span class="p">,</span> <span class="n">dataset_general</span><span class="p">],</span>
    <span class="n">probabilities</span><span class="o">=</span><span class="p">[</span><span class="mf">0.4</span><span class="p">,</span> <span class="mf">0.4</span><span class="p">,</span> <span class="mf">0.2</span><span class="p">],</span>   <span class="c1"># 经温度缩放后手动设定</span>
    <span class="n">seed</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">SFTConfig</span><span class="p">(</span><span class="n">output_dir</span><span class="o">=</span><span class="s2">"sft_model"</span><span class="p">)</span>
<span class="n">trainer</span> <span class="o">=</span> <span class="n">SFTTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">mixed_dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>
