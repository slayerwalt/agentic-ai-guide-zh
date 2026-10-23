# 8.3　IPO —— Identity Preference Optimization {#ipo-identity-preference-optimization}

## 8.3.1　动机 {#motivation-2}

<p>DPO 有一个退化解：通过让 chosen 与 rejected 之间的 margin <em>无限大</em> 来达到零 loss。实际上这意味着 DPO 会过拟合——把 chosen 概率推到 1、rejected 概率推到 0，记住训练数据。</p>

<p><strong>IPO 的修复</strong> <a class="book-citation" id="cite-azar2024general-1" href="../../references.md#ref-azar2024general" aria-label="参考文献 14">[14]</a>：与其使用会饱和的 log-sigmoid，不如使用一个针对 <em>特定</em> margin 的平方 loss。loss 在有限差距处最小，而非无穷远。</p>

## 8.3.2　Loss 函数 {#loss-function-1}



<div class="arithmatex">\[\boxed{\mathcal{L}_\text{IPO} = \mathbb{E}\left[\left(\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]}
\tag{8.2}\]</div>

<div class="admonition tip"><p class="admonition-title">IPO vs DPO：通过目标 Margin 实现正则化</p>
<p>DPO：<span class="arithmatex">\(\sigma(\text{margin}) \to 1\)</span> 为最优。Margin <span class="arithmatex">\(\to \infty\)</span>，没有自然停止点。</p>

<p>IPO：Margin <span class="arithmatex">\(\to \frac{1}{2\beta}\)</span> 为最优。平方 loss 同时惩罚 <strong>过小</strong> 与 <strong>过大</strong> 的 margin。</p>

<p>结果：IPO 对噪声标签更鲁棒（错标的 pair 影响有界），且因为不死记硬背而泛化更好。</p>
</div>

## 8.3.3　TRL 实现 {#trl-implementation-2}

<p>下面是基于 HuggingFace TRL 的最小可运行示例。</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="c1"># IPO 在 TRL 中作为 DPO 的 loss_type 变体实现</span>
<span class="n">ipo_config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./ipo_output"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"ipo"</span><span class="p">,</span>             <span class="c1"># 关键差异！</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">5e-7</span><span class="p">,</span>
    <span class="n">max_length</span><span class="o">=</span><span class="mi">2048</span><span class="p">,</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span> <span class="n">ref_model</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span> <span class="n">args</span><span class="o">=</span><span class="n">ipo_config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">pref_dataset</span><span class="p">,</span> <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span> <span class="n">peft_config</span><span class="o">=</span><span class="n">lora_config</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>

## 8.3.4　何时选择 IPO 而非 DPO {#when-to-choose-ipo-over-dpo}

<ul>
<li><p>噪声偏好数据（众包、AI 判定带错误）</p></li>
<li><p>观察到 DPO 过拟合（训练 loss <span class="arithmatex">\(\to\)</span> 0 但评估退化）</p></li>
<li><p>想要更保守、更鲁棒的对齐</p></li>
<li><p>需要多个 epoch（DPO 在第一个 epoch 后退化；IPO 更稳定）</p></li>
</ul>
