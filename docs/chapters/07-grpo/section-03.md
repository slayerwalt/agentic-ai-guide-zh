# 7.3　TRL 实现 {#trl-implementation-dup3}

<p>下面是基于 HuggingFace TRL 的最小可运行示例。</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoModelForCausalLM</span><span class="p">,</span> <span class="n">AutoTokenizer</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">AutoModelForCausalLM</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"Qwen/Qwen2.5-7B-Instruct"</span><span class="p">,</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span> <span class="n">attn_implementation</span><span class="o">=</span><span class="s2">"flash_attention_2"</span><span class="p">)</span>
<span class="n">tokenizer</span> <span class="o">=</span> <span class="n">AutoTokenizer</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"Qwen/Qwen2.5-7B-Instruct"</span><span class="p">)</span>

<span class="n">grpo_config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./grpo_output"</span><span class="p">,</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>           <span class="c1"># G = 组大小</span>
    <span class="n">temperature</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>             <span class="c1"># 高温度以保证组内多样性</span>
    <span class="n">max_completion_length</span><span class="o">=</span><span class="mi">2048</span><span class="p">,</span>  <span class="c1"># 最大响应长度</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.04</span><span class="p">,</span>                   <span class="c1"># KL 惩罚系数</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">1e-6</span><span class="p">,</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>  <span class="c1"># 每设备 prompt 数（×8 生成 = 16 响应）</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">gradient_checkpointing</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">max_grad_norm</span><span class="o">=</span><span class="mf">0.5</span><span class="p">,</span>
    <span class="n">logging_steps</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span>
    <span class="c1"># vLLM 生成以加速（GRPO 由于 8 倍生成，速度至关重要）</span>
    <span class="n">use_vllm</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">vllm_gpu_memory_utilization</span><span class="o">=</span><span class="mf">0.7</span><span class="p">,</span>
<span class="p">)</span>

<span class="c1"># Reward 函数：数学任务的二值正确性</span>
<span class="k">def</span><span class="w"> </span><span class="nf">reward_fn</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="n">prompts</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""返回 float 列表：正确为 1.0，错误为 0.0。"""</span>
    <span class="n">rewards</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">completion</span><span class="p">,</span> <span class="n">prompt</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="n">prompts</span><span class="p">):</span>
        <span class="n">answer</span> <span class="o">=</span> <span class="n">extract_answer</span><span class="p">(</span><span class="n">completion</span><span class="p">)</span>
        <span class="n">expected</span> <span class="o">=</span> <span class="n">get_ground_truth</span><span class="p">(</span><span class="n">prompt</span><span class="p">)</span>
        <span class="n">rewards</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mf">1.0</span> <span class="k">if</span> <span class="n">answer</span> <span class="o">==</span> <span class="n">expected</span> <span class="k">else</span> <span class="mf">0.0</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">rewards</span>

<span class="c1"># 可以组合多个 reward 函数！</span>
<span class="k">def</span><span class="w"> </span><span class="nf">format_reward_fn</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""对使用正确 LaTeX 格式的额外奖励。"""</span>
    <span class="k">return</span> <span class="p">[</span><span class="mf">0.5</span> <span class="k">if</span> <span class="s2">"</span><span class="se">\\</span><span class="s2">boxed{"</span> <span class="ow">in</span> <span class="n">c</span> <span class="k">else</span> <span class="mf">0.0</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">completions</span><span class="p">]</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">GRPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">grpo_config</span><span class="p">,</span>
    <span class="n">reward_funcs</span><span class="o">=</span><span class="p">[</span><span class="n">reward_fn</span><span class="p">,</span> <span class="n">format_reward_fn</span><span class="p">],</span>  <span class="c1"># 多目标！</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">math_dataset</span><span class="p">,</span>
    <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>
