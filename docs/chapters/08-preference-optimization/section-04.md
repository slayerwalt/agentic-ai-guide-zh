# 8.4　ORPO —— Odds Ratio Preference Optimization {#sec:orpo}

## 8.4.1　动机 {#motivation-3}

<p>迄今所有方法都需要参考模型——要么是独立副本（显存翻倍），要么通过 LoRA 隐式存在。ORPO <a class="book-citation" id="cite-hong2024orpo-1" href="../../references.md#ref-hong2024orpo" aria-label="参考文献 141">[141]</a> 通过将监督微调（Supervised Fine-Tuning, SFT）与偏好对齐合并到单一 loss 中，彻底消除了参考模型。</p>

<p><strong>关键洞见</strong>：用生成 chosen 与生成 rejected 的 <em>odds ratio</em>（几率比）作为偏好信号。SFT 部分自然防止坍塌（无需 KL 正则）。</p>

## 8.4.2　Loss 函数 {#loss-function-2}



<div class="arithmatex">\[\boxed{\mathcal{L}_\text{ORPO} = \underbrace{\mathcal{L}_\text{SFT}(y_w)}_{\text{standard NLL on chosen}} - \lambda \cdot \underbrace{\log\sigma\left(\log\frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)}_{\text{preference alignment via odds ratio}}}
\tag{8.3}\]</div>

<p>其中 <span class="arithmatex">\(\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}\)</span>。</p>

<div class="admonition tip"><p class="admonition-title">ORPO：一次完成 SFT + 对齐</p>
<p><strong>SFT 项</strong>：训练模型把 chosen 响应生成好（标准语言建模）。</p>

<p><strong>Odds ratio 项</strong>：额外推动模型偏好 chosen 而非 rejected。Odds ratio 是天然的对比，不需要参考模型。</p>

<p><strong>为什么不需要参考？</strong>SFT loss 已将模型锚定在合理文本上，扮演了其他方法中 KL-to-reference 的角色。一个模型、一次前向、一个 loss。显存减少 50%！</p>
</div>

## 8.4.3　TRL 实现 {#trl-implementation-3}

<p>下面是基于 HuggingFace TRL 的最小可运行示例。</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">ORPOConfig</span><span class="p">,</span> <span class="n">ORPOTrainer</span>

<span class="n">orpo_config</span> <span class="o">=</span> <span class="n">ORPOConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./orpo_output"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>                    <span class="c1"># Odds ratio 权重（lambda）</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">5e-7</span><span class="p">,</span>
    <span class="n">max_length</span><span class="o">=</span><span class="mi">2048</span><span class="p">,</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
    <span class="n">gradient_checkpointing</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">ORPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>                 <span class="c1"># 无需 ref_model！</span>
    <span class="n">args</span><span class="o">=</span><span class="n">orpo_config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">pref_dataset</span><span class="p">,</span>  <span class="c1"># 与 DPO 同格式：prompt/chosen/rejected</span>
    <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
    <span class="n">peft_config</span><span class="o">=</span><span class="n">lora_config</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>

## 8.4.4　何时选择 ORPO {#when-to-choose-orpo}

<ul>
<li><p>显存受限：负担不起参考模型副本（70B 可节省 70–140GB）</p></li>
<li><p>从 base 模型起步（尚未 SFT）—— ORPO 同时做 SFT</p></li>
<li><p>想要最简单的 pipeline：一个模型、一个 loss、一次训练</p></li>
<li><p>一开始就有良好的偏好数据</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">ORPO 的局限</p>
<ul>
<li><p>研究不如 DPO/PPO 充分——70B 及以上规模的成熟配方较少</p></li>
<li><p>SFT 部分意味着它需要高质量的 chosen 响应（不仅是相对偏好）</p></li>
<li><p>调试更难：两个 loss 分量可能冲突</p></li>
</ul>
</div>

<div class="admonition note"><p class="admonition-title">另见：SimPO</p>
<p><strong>SimPO</strong> <a class="book-citation" id="cite-meng2024simpo-1" href="../../references.md#ref-meng2024simpo" aria-label="参考文献 252">[252]</a> 是另一种无参考的偏好方法，使用长度归一化的 log-probability 作为隐式 reward，彻底消除参考模型。它在 Section <a href="../06-dpo/section-09.md#sec:simpo">6.9.8</a> 与其他 DPO 扩展一同介绍，因为它们都共享无参考的理念。</p>
</div>
