# 10.3　仅对补全部分进行 mask {#sec:completion-masking}

<div class="admonition tip"><p class="admonition-title">为何要 mask prompt?</p>
<p>在指令微调中，模型应该学习生成 assistant 的回应，而非预测用户的问题或 system prompt。在 prompt Token 上计算 Loss 会浪费 Gradient 信号，并可能导致模型“记忆”prompt 而非学会回应它们。Completion-only masking 将所有非 assistant Token 的 Loss 置零。</p>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的 Completion-Only Masking</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">SFTConfig</span><span class="p">,</span> <span class="n">SFTTrainer</span><span class="p">,</span> <span class="n">DataCollatorForCompletionOnlyLM</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoTokenizer</span>

<span class="n">tokenizer</span> <span class="o">=</span> <span class="n">AutoTokenizer</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">)</span>

<span class="c1"># 定义响应模板(此 Token 之后开始计算 Loss)</span>
<span class="n">response_template</span> <span class="o">=</span> <span class="s2">"&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;"</span>
<span class="n">collator</span> <span class="o">=</span> <span class="n">DataCollatorForCompletionOnlyLM</span><span class="p">(</span>
    <span class="n">response_template</span><span class="o">=</span><span class="n">response_template</span><span class="p">,</span>
    <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">SFTConfig</span><span class="p">(</span>
    <span class="n">max_seq_length</span><span class="o">=</span><span class="mi">2048</span><span class="p">,</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"sft_model"</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">SFTTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
    <span class="n">data_collator</span><span class="o">=</span><span class="n">collator</span><span class="p">,</span>   <span class="c1"># 仅对 completion 部分计算 Loss</span>
    <span class="n">formatting_func</span><span class="o">=</span><span class="n">formatting_func</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition warning"><p class="admonition-title">Completion Masking 陷阱</p>
<ul>
<li><p>响应模板必须与 token 化后的形式精确匹配。token 化中的 off-by-one 错误会导致 mask 应用错误。</p></li>
<li><p>对非常短的回答，mask 掉 prompt 后剩下的 Token 可能不足以提供有意义的 Gradient 信号。可考虑设置最小响应长度阈值。</p></li>
<li><p>多轮对话（Multi-Turn）需要 mask 所有非 assistant 轮次，而不仅是第一轮。</p></li>
</ul>
</div>
