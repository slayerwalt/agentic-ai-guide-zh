# 8.1　Online DPO {#sec:online-dpo}

## 8.1.1　动机 {#motivation-dup3}

<p>标准 DPO 的主要局限：偏好数据由一个 <em>不同的</em> 模型生成（通常是较旧的 checkpoint，甚至是不同的模型族）。随着训练推进，policy 生成的文本与训练 pair 已完全不同 <span class="arithmatex">\(\rightarrow\)</span> loss 在一个无关分布上进行优化。</p>

<p><strong>Online DPO 方案</strong> <a class="book-citation" id="cite-guo2024direct-1" href="../../references.md#ref-guo2024direct" aria-label="参考文献 127">[127]</a>：每一步都从 <em>当前</em> policy 生成新鲜的偏好 pair，用 reward model 判定，然后应用 DPO loss。</p>

## 8.1.2　算法 {#algorithm-dup2}

<ol>
<li><p>从当前 <span class="arithmatex">\(\pi_\theta\)</span> 为每个 prompt 生成 <span class="arithmatex">\(K\)</span> 条响应</p></li>
<li><p>用 reward model <span class="arithmatex">\(r_\phi\)</span> 给所有响应打分</p></li>
<li><p>构造 pair：最高分 = chosen，最低分 = rejected</p></li>
<li><p>在这些新鲜 pair 上应用 DPO loss</p></li>
<li><p>重复（每步重新生成）</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">Online DPO = 两全其美</p>
<ul>
<li><p>来自 DPO：简单的监督式 loss，无 value function，无 GAE，优化稳定</p></li>
<li><p>来自 PPO：on-policy 数据，超越数据集的自我提升，无分布偏移</p></li>
<li><p>与 GRPO 的关键差异：使用 DPO loss（基于 pair）而非 PPO loss（逐样本 advantage）</p></li>
</ul>

<p><strong>权衡</strong>：需要 reward model（DPO 不需要），但不需要 value head（PPO 需要）。复杂度居中。</p>
</div>

## 8.1.3　TRL 实现 {#trl-implementation-dup4}

<p>下面是基于 HuggingFace TRL 的最小可运行示例。</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">OnlineDPOConfig</span><span class="p">,</span> <span class="n">OnlineDPOTrainer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoModelForCausalLM</span><span class="p">,</span> <span class="n">AutoModelForSequenceClassification</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">AutoModelForCausalLM</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">,</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>
<span class="n">reward_model</span> <span class="o">=</span> <span class="n">AutoModelForSequenceClassification</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span>
    <span class="s2">"RLHFlow/ArmoRM-Llama3-8B-v0.1"</span><span class="p">,</span> <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">)</span>

<span class="n">online_dpo_config</span> <span class="o">=</span> <span class="n">OnlineDPOConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./online_dpo_output"</span><span class="p">,</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">5e-7</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>                    <span class="c1"># DPO beta（与标准 DPO 同义）</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>           <span class="c1"># 每个 prompt 的 K 条响应</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">max_new_tokens</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span>
    <span class="n">temperature</span><span class="o">=</span><span class="mf">0.7</span><span class="p">,</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
    <span class="n">logging_steps</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">OnlineDPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">reward_model</span><span class="o">=</span><span class="n">reward_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">online_dpo_config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">prompt_dataset</span><span class="p">,</span>
    <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>

## 8.1.4　Online DPO vs Offline DPO vs PPO {#online-dpo-vs-offline-dpo-vs-ppo}

<span id="table-8-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"></th><th scope="col"><strong>数据</strong></th><th scope="col"><strong>模型</strong></th><th scope="col"><strong>Loss</strong></th><th scope="col"><strong>最适合</strong></th></tr></thead>
<tbody><tr><td>Offline DPO</td><td>静态 pair</td><td>2 (policy + reference)</td><td>DPO</td><td>快速对齐、算力有限</td></tr>
<tr><td>Online DPO</td><td>从 <span class="arithmatex">\(\pi_\theta\)</span> 新鲜采样</td><td>3 (policy + reference + reward model)</td><td>DPO</td><td>当 DPO 停滞、需要探索时</td></tr>
<tr><td>PPO</td><td>从 <span class="arithmatex">\(\pi_\theta\)</span> 新鲜采样</td><td>4 (policy + reference + reward model + value head)</td><td>PPO clip</td><td>极致质量、复杂推理</td></tr></tbody>
</table>
</div>
