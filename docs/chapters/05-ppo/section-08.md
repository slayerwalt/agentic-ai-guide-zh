# 5.8　TRL 实现 {#trl-implementation}

<p>HuggingFace 的 TRL 库 <a class="book-citation" id="cite-vonwerra2022trl-1" href="../../references.md#ref-vonwerra2022trl" aria-label="参考文献 376">[376]</a> 提供了面向大语言模型的所有主流 RL 方法的生产级实现。</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">PPOConfig</span><span class="p">,</span> <span class="n">PPOTrainer</span><span class="p">,</span> <span class="n">AutoModelForCausalLMWithValueHead</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoTokenizer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">peft</span><span class="w"> </span><span class="kn">import</span> <span class="n">LoraConfig</span>

<span class="c1"># 模型初始化</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">AutoModelForCausalLMWithValueHead</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span>
    <span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">,</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span> <span class="n">device_map</span><span class="o">=</span><span class="s2">"auto"</span><span class="p">,</span>
    <span class="n">peft_config</span><span class="o">=</span><span class="n">LoraConfig</span><span class="p">(</span><span class="n">r</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span> <span class="n">lora_alpha</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span> <span class="n">target_modules</span><span class="o">=</span><span class="p">[</span><span class="s2">"q_proj"</span><span class="p">,</span><span class="s2">"v_proj"</span><span class="p">,</span><span class="s2">"k_proj"</span><span class="p">,</span><span class="s2">"o_proj"</span><span class="p">])</span>
<span class="p">)</span>
<span class="n">tokenizer</span> <span class="o">=</span> <span class="n">AutoTokenizer</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">)</span>

<span class="c1"># PPO 配置，包含全部关键超参</span>
<span class="n">ppo_config</span> <span class="o">=</span> <span class="n">PPOConfig</span><span class="p">(</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">1.5e-6</span><span class="p">,</span>        <span class="c1"># 低学习率，利于稳定</span>
    <span class="n">batch_size</span><span class="o">=</span><span class="mi">128</span><span class="p">,</span>              <span class="c1"># 每步的 prompt 数</span>
    <span class="n">mini_batch_size</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span>          <span class="c1"># 梯度累计的单位</span>
    <span class="n">ppo_epochs</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>                <span class="c1"># 每个 batch 的 epoch 数（复用数据）</span>
    <span class="n">gamma</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>                   <span class="c1"># 不做折扣（单轮）</span>
    <span class="n">lam</span><span class="o">=</span><span class="mf">0.95</span><span class="p">,</span>                    <span class="c1"># GAE lambda</span>
    <span class="n">cliprange</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span>               <span class="c1"># PPO epsilon</span>
    <span class="n">cliprange_value</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span>         <span class="c1"># value function 的 clip 范围</span>
    <span class="n">vf_coef</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>                 <span class="c1"># value loss 系数</span>
    <span class="n">init_kl_coef</span><span class="o">=</span><span class="mf">0.05</span><span class="p">,</span>           <span class="c1"># 初始 KL 惩罚</span>
    <span class="n">target_kl</span><span class="o">=</span><span class="mf">6.0</span><span class="p">,</span>               <span class="c1"># 自适应 KL 目标</span>
    <span class="n">whiten_rewards</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>         <span class="c1"># 对 advantage 做归一化</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">max_grad_norm</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">ppo_trainer</span> <span class="o">=</span> <span class="n">PPOTrainer</span><span class="p">(</span><span class="n">config</span><span class="o">=</span><span class="n">ppo_config</span><span class="p">,</span> <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span> <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
    <span class="n">dataset</span><span class="o">=</span><span class="n">prompt_dataset</span><span class="p">,</span> <span class="n">data_collator</span><span class="o">=</span><span class="n">collator</span><span class="p">)</span>

<span class="c1"># 训练循环</span>
<span class="k">for</span> <span class="n">batch</span> <span class="ow">in</span> <span class="n">ppo_trainer</span><span class="o">.</span><span class="n">dataloader</span><span class="p">:</span>
    <span class="c1"># 1. 生成回答</span>
    <span class="n">query_tensors</span> <span class="o">=</span> <span class="n">batch</span><span class="p">[</span><span class="s2">"input_ids"</span><span class="p">]</span>
    <span class="n">response_tensors</span> <span class="o">=</span> <span class="n">ppo_trainer</span><span class="o">.</span><span class="n">generate</span><span class="p">(</span>
        <span class="n">query_tensors</span><span class="p">,</span> <span class="n">max_new_tokens</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span> <span class="n">temperature</span><span class="o">=</span><span class="mf">0.7</span><span class="p">,</span> <span class="n">top_p</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span> <span class="n">do_sample</span><span class="o">=</span><span class="kc">True</span>
    <span class="p">)</span>
    <span class="c1"># 2. 用 reward 模型打分</span>
    <span class="n">texts</span> <span class="o">=</span> <span class="p">[</span><span class="n">tokenizer</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">r</span><span class="p">,</span> <span class="n">skip_special_tokens</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">response_tensors</span><span class="p">]</span>
    <span class="n">rewards</span> <span class="o">=</span> <span class="p">[</span><span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">(</span><span class="n">reward_model</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">q</span><span class="p">,</span> <span class="n">r</span><span class="p">))</span> <span class="k">for</span> <span class="n">q</span><span class="p">,</span> <span class="n">r</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">batch</span><span class="p">[</span><span class="s2">"query"</span><span class="p">],</span> <span class="n">texts</span><span class="p">)]</span>
    <span class="c1"># 3. PPO 更新（内部完成 KL、GAE、clipping）</span>
    <span class="n">stats</span> <span class="o">=</span> <span class="n">ppo_trainer</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">query_tensors</span><span class="p">,</span> <span class="n">response_tensors</span><span class="p">,</span> <span class="n">rewards</span><span class="p">)</span>
    <span class="c1"># 监控指标：stats["ppo/mean_scores"]、stats["ppo/policy/approx_kl"]</span>
</code></pre></div>
