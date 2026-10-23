# 9.2　奖励模型架构 {#sec:rm-architectures}

<div class="admonition tip"><p class="admonition-title">LLM 上的分类头</p>
<p>标准的奖励模型架构是：取一个预训练 LLM，将其语言建模头（将隐藏状态映射到词表 logits）替换为一个标量回归头（将最终隐藏状态映射到单一 Reward 值）。</p>
</div>

<p>该架构由以下部分组成：</p>

<ol>
<li><p><strong>骨干网络（Backbone）</strong>：预训练 LLM（例如 Llama、Mistral），将 prompt-response 对编码为一系列隐藏状态。</p></li>
<li><p><strong>池化（Pooling）</strong>：提取最后一个 Token 位置（decoder-only 模型）或 <code>[CLS]</code> Token（encoder 模型）处的隐藏状态。</p></li>
<li><p><strong>回归头（Regression head）</strong>：线性层 <span class="arithmatex">\(W \in \mathbb{R}^{d \times 1}\)</span>，将池化后的隐藏状态映射为标量 Reward。</p></li>
</ol>

<div class="admonition example"><p class="admonition-title">TRL 中的奖励模型训练</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">RewardConfig</span><span class="p">,</span> <span class="n">RewardTrainer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoModelForSequenceClassification</span>

<span class="c1"># 加载带标量头(num_labels=1)的模型</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">AutoModelForSequenceClassification</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span>
    <span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">,</span>
    <span class="n">num_labels</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">RewardConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"reward_model"</span><span class="p">,</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">1e-5</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
    <span class="c1"># Margin loss</span>
    <span class="n">center_rewards_coefficient</span><span class="o">=</span><span class="mf">0.01</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">RewardTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>  <span class="c1"># 必须包含 chosen/rejected 列</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>
</div>
