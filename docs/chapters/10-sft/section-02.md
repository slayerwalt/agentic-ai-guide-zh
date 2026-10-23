# 10.2　Chat 模板与格式化 {#sec:chat-templates}

<div class="admonition tip"><p class="admonition-title">为何 Chat 模板重要</p>
<p>语言模型在原始文本上训练，但指令跟随模型需要区分 system prompt、用户消息与 assistant 回应。Chat 模板将这种结构编码进 Token 序列。推理时使用错误模板（或不使用模板）会导致性能显著下降。</p>
</div>

### ChatML 格式 {#chatml-format}

<p>ChatML 是使用最广泛的 chat 模板：</p>

<div class="highlight"><pre><span></span><code><span class="c1"># ChatML 格式</span>
<span class="n">template</span> <span class="o">=</span> <span class="s2">"""&lt;|im_start|&gt;system</span>
<span class="si">{system_message}</span><span class="s2">&lt;|im_end|&gt;</span>
<span class="s2">&lt;|im_start|&gt;user</span>
<span class="si">{user_message}</span><span class="s2">&lt;|im_end|&gt;</span>
<span class="s2">&lt;|im_start|&gt;assistant</span>
<span class="si">{assistant_message}</span><span class="s2">&lt;|im_end|&gt;"""</span>
</code></pre></div>

### Llama 格式 {#llama-format}

<p>Llama 3 使用一种带特殊 Token 的不同模板：</p>

<div class="highlight"><pre><span></span><code><span class="c1"># Llama 3 格式</span>
<span class="n">template</span> <span class="o">=</span> <span class="s2">"""&lt;|begin_of_text|&gt;&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;</span>
<span class="si">{system_message}</span><span class="s2">&lt;|eot_id|&gt;&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;</span>
<span class="si">{user_message}</span><span class="s2">&lt;|eot_id|&gt;&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;</span>
<span class="si">{assistant_message}</span><span class="s2">&lt;|eot_id|&gt;"""</span>
</code></pre></div>

<div class="admonition example"><p class="admonition-title">在 TRL 中应用 Chat 模板</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoTokenizer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">SFTConfig</span><span class="p">,</span> <span class="n">SFTTrainer</span>

<span class="n">tokenizer</span> <span class="o">=</span> <span class="n">AutoTokenizer</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">)</span>

<span class="k">def</span><span class="w"> </span><span class="nf">formatting_func</span><span class="p">(</span><span class="n">example</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""对数据集样本应用 chat 模板。"""</span>
    <span class="n">messages</span> <span class="o">=</span> <span class="p">[</span>
        <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"system"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"You are a helpful assistant."</span><span class="p">},</span>
        <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">example</span><span class="p">[</span><span class="s2">"instruction"</span><span class="p">]},</span>
        <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"assistant"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">example</span><span class="p">[</span><span class="s2">"response"</span><span class="p">]},</span>
    <span class="p">]</span>
    <span class="k">return</span> <span class="n">tokenizer</span><span class="o">.</span><span class="n">apply_chat_template</span><span class="p">(</span>
        <span class="n">messages</span><span class="p">,</span>
        <span class="n">tokenize</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span>
        <span class="n">add_generation_prompt</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span>
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
    <span class="n">formatting_func</span><span class="o">=</span><span class="n">formatting_func</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>
