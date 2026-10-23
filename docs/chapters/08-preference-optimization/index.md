# 第 8 章 · 偏好优化变体 {#ch:po-variants}

<p>本章涵盖一类用不同目标、数据假设或架构权衡来扩展或替换 DPO 的方法。每种方法都解决标准 offline DPO 的某个具体局限：分布偏移（Online DPO）、需要配对数据（KTO）、对噪声标签过拟合（IPO）、参考模型显存开销（ORPO）或训练复杂度（Best-of-N）。</p>

## 8.1　Online DPO {#sec:online-dpo}

### 8.1.1　动机 {#motivation-dup3}

<p>标准 DPO 的主要局限：偏好数据由一个 <em>不同的</em> 模型生成（通常是较旧的 checkpoint，甚至是不同的模型族）。随着训练推进，policy 生成的文本与训练 pair 已完全不同 <span class="arithmatex">\(\rightarrow\)</span> loss 在一个无关分布上进行优化。</p>

<p><strong>Online DPO 方案</strong> <a class="book-citation" id="cite-guo2024direct-1" href="../../references.md#ref-guo2024direct" aria-label="参考文献 127">[127]</a>：每一步都从 <em>当前</em> policy 生成新鲜的偏好 pair，用 reward model 判定，然后应用 DPO loss。</p>

### 8.1.2　算法 {#algorithm-dup2}

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

### 8.1.3　TRL 实现 {#trl-implementation-dup4}

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

### 8.1.4　Online DPO vs Offline DPO vs PPO {#online-dpo-vs-offline-dpo-vs-ppo}

<span id="table-8-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"></th><th scope="col"><strong>数据</strong></th><th scope="col"><strong>模型</strong></th><th scope="col"><strong>Loss</strong></th><th scope="col"><strong>最适合</strong></th></tr></thead>
<tbody><tr><td>Offline DPO</td><td>静态 pair</td><td>2 (policy + reference)</td><td>DPO</td><td>快速对齐、算力有限</td></tr>
<tr><td>Online DPO</td><td>从 <span class="arithmatex">\(\pi_\theta\)</span> 新鲜采样</td><td>3 (policy + reference + reward model)</td><td>DPO</td><td>当 DPO 停滞、需要探索时</td></tr>
<tr><td>PPO</td><td>从 <span class="arithmatex">\(\pi_\theta\)</span> 新鲜采样</td><td>4 (policy + reference + reward model + value head)</td><td>PPO clip</td><td>极致质量、复杂推理</td></tr></tbody>
</table>
</div>

## 8.2　KTO —— Kahneman-Tversky Optimization {#sec:kto}

### 8.2.1　动机 {#motivation-1}

<p>DPO 需要 <em>成对</em> 偏好：对同一 prompt，你既需要好的也需要坏的响应。实际上大多数反馈是 <em>未配对</em> 的：用户对单个响应给出点赞/点踩，没有匹配的 pair。</p>

<p><strong>KTO 的洞见</strong> <a class="book-citation" id="cite-ethayarajh2024kto-1" href="../../references.md#ref-ethayarajh2024kto" aria-label="参考文献 87">[87]</a>：使用前景理论（来自行为经济学）。人类对损失的感受比对收益更强烈。“点踩”应当产生比“点赞”更强的 gradient。</p>

### 8.2.2　Loss 函数 {#loss-function}



<div class="arithmatex">\[\boxed{\mathcal{L}_\text{KTO} = \mathbb{E}_{y_w}\left[\lambda_w (1 - v(x, y_w))\right] + \mathbb{E}_{y_l}\left[\lambda_l \cdot v(x, y_l)\right]}
\tag{8.1}\]</div>

<p>其中 <span class="arithmatex">\(v(x,y) = \sigma\left(\beta \log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} - z_\text{ref}\right)\)</span>，<span class="arithmatex">\(z_\text{ref}\)</span> 是期望 KL 散度（一个滑动 baseline）。</p>

<div class="admonition tip"><p class="admonition-title">基于前景理论的 KTO 直觉</p>
<p><strong>好响应</strong>（<span class="arithmatex">\(y_w\)</span>）：模型通过提高它们的概率获得“效用”。但收益递减——一旦它已相当可能，就不再用力推。</p>

<p><strong>坏响应</strong>（<span class="arithmatex">\(y_l\)</span>）：损失厌恶意味着生成坏文本的惩罚被加权得比生成好文本的奖励更强。默认 <span class="arithmatex">\(\lambda_l = 1.0\)</span>，<span class="arithmatex">\(\lambda_w = 1.0\)</span>，但你可以设 <span class="arithmatex">\(\lambda_l &gt; \lambda_w\)</span>。</p>

<p><strong>关键优势</strong>：每个训练样本都是独立的！无需匹配 pair。可直接使用点赞/点踩数据。</p>
</div>

<div class="admonition example"><p class="admonition-title">KTO 数据格式</p>
<p>与 DPO 需要 <code>{&quot;prompt&quot;: ..., &quot;chosen&quot;: ..., &quot;rejected&quot;: ...}</code> 不同</p>

<p>KTO 只需要：<code>{&quot;prompt&quot;: ..., &quot;completion&quot;: ..., &quot;label&quot;: true/false}</code></p>

<p>这意味着你可以使用：</p>

<ul>
<li><p>生产流量中的点赞/点踩</p></li>
<li><p>论坛上的赞/踩</p></li>
<li><p>二值化的人类评分（4–5 星 = 好，1–2 星 = 坏）</p></li>
<li><p>任何单条响应的质量信号</p></li>
</ul>
</div>

### 8.2.3　TRL 实现 {#trl-implementation-1}

<p>下面是基于 HuggingFace TRL 的最小可运行示例。</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">KTOConfig</span><span class="p">,</span> <span class="n">KTOTrainer</span>

<span class="c1"># 数据格式：{"prompt": str, "completion": str, "label": bool}</span>
<span class="c1"># label=True 表示好的，label=False 表示坏的</span>
<span class="n">kto_dataset</span> <span class="o">=</span> <span class="p">[</span>
    <span class="p">{</span><span class="s2">"prompt"</span><span class="p">:</span> <span class="s2">"What's 2+2?"</span><span class="p">,</span> <span class="s2">"completion"</span><span class="p">:</span> <span class="s2">"The answer is 4."</span><span class="p">,</span> <span class="s2">"label"</span><span class="p">:</span> <span class="kc">True</span><span class="p">},</span>
    <span class="p">{</span><span class="s2">"prompt"</span><span class="p">:</span> <span class="s2">"What's 2+2?"</span><span class="p">,</span> <span class="s2">"completion"</span><span class="p">:</span> <span class="s2">"It might be 5."</span><span class="p">,</span> <span class="s2">"label"</span><span class="p">:</span> <span class="kc">False</span><span class="p">},</span>
<span class="p">]</span>

<span class="n">kto_config</span> <span class="o">=</span> <span class="n">KTOConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./kto_output"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
    <span class="n">desirable_weight</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>        <span class="c1"># 好样本的权重</span>
    <span class="n">undesirable_weight</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>      <span class="c1"># 坏样本的权重（损失厌恶时调高）</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">5e-7</span><span class="p">,</span>
    <span class="n">max_length</span><span class="o">=</span><span class="mi">2048</span><span class="p">,</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">KTOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>  <span class="c1"># LoRA 时可为 None</span>
    <span class="n">args</span><span class="o">=</span><span class="n">kto_config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">kto_dataset</span><span class="p">,</span>
    <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>

### 8.2.4　何时选择 KTO {#when-to-choose-kto}

<ul>
<li><p>你有二值反馈但 <em>没有</em> 匹配 pair</p></li>
<li><p>大规模的生产点赞/点踩数据</p></li>
<li><p>一类占主导（例如 90% 好、10% 坏）—— KTO 对不平衡更友好</p></li>
<li><p>在噪声标签下快速迭代（对噪声比 DPO 更鲁棒）</p></li>
</ul>

## 8.3　IPO —— Identity Preference Optimization {#ipo-identity-preference-optimization}

### 8.3.1　动机 {#motivation-2}

<p>DPO 有一个退化解：通过让 chosen 与 rejected 之间的 margin <em>无限大</em> 来达到零 loss。实际上这意味着 DPO 会过拟合——把 chosen 概率推到 1、rejected 概率推到 0，记住训练数据。</p>

<p><strong>IPO 的修复</strong> <a class="book-citation" id="cite-azar2024general-1" href="../../references.md#ref-azar2024general" aria-label="参考文献 14">[14]</a>：与其使用会饱和的 log-sigmoid，不如使用一个针对 <em>特定</em> margin 的平方 loss。loss 在有限差距处最小，而非无穷远。</p>

### 8.3.2　Loss 函数 {#loss-function-1}



<div class="arithmatex">\[\boxed{\mathcal{L}_\text{IPO} = \mathbb{E}\left[\left(\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]}
\tag{8.2}\]</div>

<div class="admonition tip"><p class="admonition-title">IPO vs DPO：通过目标 Margin 实现正则化</p>
<p>DPO：<span class="arithmatex">\(\sigma(\text{margin}) \to 1\)</span> 为最优。Margin <span class="arithmatex">\(\to \infty\)</span>，没有自然停止点。</p>

<p>IPO：Margin <span class="arithmatex">\(\to \frac{1}{2\beta}\)</span> 为最优。平方 loss 同时惩罚 <strong>过小</strong> 与 <strong>过大</strong> 的 margin。</p>

<p>结果：IPO 对噪声标签更鲁棒（错标的 pair 影响有界），且因为不死记硬背而泛化更好。</p>
</div>

### 8.3.3　TRL 实现 {#trl-implementation-2}

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

### 8.3.4　何时选择 IPO 而非 DPO {#when-to-choose-ipo-over-dpo}

<ul>
<li><p>噪声偏好数据（众包、AI 判定带错误）</p></li>
<li><p>观察到 DPO 过拟合（训练 loss <span class="arithmatex">\(\to\)</span> 0 但评估退化）</p></li>
<li><p>想要更保守、更鲁棒的对齐</p></li>
<li><p>需要多个 epoch（DPO 在第一个 epoch 后退化；IPO 更稳定）</p></li>
</ul>

## 8.4　ORPO —— Odds Ratio Preference Optimization {#sec:orpo}

### 8.4.1　动机 {#motivation-3}

<p>迄今所有方法都需要参考模型——要么是独立副本（显存翻倍），要么通过 LoRA 隐式存在。ORPO <a class="book-citation" id="cite-hong2024orpo-1" href="../../references.md#ref-hong2024orpo" aria-label="参考文献 141">[141]</a> 通过将监督微调（Supervised Fine-Tuning, SFT）与偏好对齐合并到单一 loss 中，彻底消除了参考模型。</p>

<p><strong>关键洞见</strong>：用生成 chosen 与生成 rejected 的 <em>odds ratio</em>（几率比）作为偏好信号。SFT 部分自然防止坍塌（无需 KL 正则）。</p>

### 8.4.2　Loss 函数 {#loss-function-2}



<div class="arithmatex">\[\boxed{\mathcal{L}_\text{ORPO} = \underbrace{\mathcal{L}_\text{SFT}(y_w)}_{\text{standard NLL on chosen}} - \lambda \cdot \underbrace{\log\sigma\left(\log\frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)}_{\text{preference alignment via odds ratio}}}
\tag{8.3}\]</div>

<p>其中 <span class="arithmatex">\(\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}\)</span>。</p>

<div class="admonition tip"><p class="admonition-title">ORPO：一次完成 SFT + 对齐</p>
<p><strong>SFT 项</strong>：训练模型把 chosen 响应生成好（标准语言建模）。</p>

<p><strong>Odds ratio 项</strong>：额外推动模型偏好 chosen 而非 rejected。Odds ratio 是天然的对比，不需要参考模型。</p>

<p><strong>为什么不需要参考？</strong>SFT loss 已将模型锚定在合理文本上，扮演了其他方法中 KL-to-reference 的角色。一个模型、一次前向、一个 loss。显存减少 50%！</p>
</div>

### 8.4.3　TRL 实现 {#trl-implementation-3}

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

### 8.4.4　何时选择 ORPO {#when-to-choose-orpo}

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
<p><strong>SimPO</strong> <a class="book-citation" id="cite-meng2024simpo-1" href="../../references.md#ref-meng2024simpo" aria-label="参考文献 252">[252]</a> 是另一种无参考的偏好方法，使用长度归一化的 log-probability 作为隐式 reward，彻底消除参考模型。它在 Section <a href="../06-dpo/index.md#sec:simpo">6.9.8</a> 与其他 DPO 扩展一同介绍，因为它们都共享无参考的理念。</p>
</div>

## 8.5　Best-of-N 采样（拒绝采样） {#best-of-n-sampling-rejection-sampling}

### 8.5.1　动机 {#motivation-4}

<p>有时候最简单的方法获胜。Best-of-N 采样（Rejection Sampling） <a class="book-citation" id="cite-nakano2021webgpt-1" href="../../references.md#ref-nakano2021webgpt" aria-label="参考文献 262">[262]</a> 在 RL 阶段 <em>完全不需要训练</em>——只需生成多个候选并挑选最佳。</p>

### 8.5.2　算法 {#algorithm-1}

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

### 8.5.3　TRL 实现 {#trl-implementation-4}

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

### 8.5.4　Best-of-N 的扩展律 {#scaling-laws-for-best-of-n}

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

## 8.6　总结：如何选择对齐方法 {#sec:method-comparison}

<p>我们已经纵览了偏好优化与基于 RL 的对齐方法的完整全景。本节将关键权衡整合为一份参考，帮助实践者根据自身约束选择合适方案。</p>

<p class="table-caption" id="table-8-1"><span class="table-number">表 8.1　</span>各对齐方法的横向对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>模型</strong></th><th scope="col"><strong>数据</strong></th><th scope="col"><strong>算力</strong></th><th scope="col"><strong>稳定性</strong></th><th scope="col"><strong>最适合</strong></th></tr></thead>
<tbody><tr><td>PPO</td><td>4</td><td>在线（生成）</td><td>非常高</td><td>低</td><td>极致质量、复杂推理</td></tr>
<tr><td>GRPO</td><td>2（无 critic）</td><td>在线（生成）</td><td>高</td><td>中</td><td>数学/代码（可验证 reward）</td></tr>
<tr><td>DPO</td><td>2</td><td>离线 pair</td><td>低</td><td>高</td><td>风格/安全、算力有限</td></tr>
<tr><td>Online DPO</td><td>3</td><td>在线（生成）</td><td>中</td><td>中-高</td><td>无分布偏移的 DPO</td></tr>
<tr><td>KTO</td><td>2</td><td>未配对二值</td><td>低</td><td>高</td><td>生产反馈、点赞/点踩</td></tr>
<tr><td>IPO</td><td>2</td><td>离线 pair</td><td>低</td><td>非常高</td><td>噪声标签、抗过拟合</td></tr>
<tr><td>ORPO</td><td>1</td><td>离线 pair</td><td>非常低</td><td>高</td><td>显存受限、SFT+对齐合并</td></tr>
<tr><td>Best-of-N</td><td>1+RM</td><td>在线（生成）</td><td>中</td><td>完美</td><td>强基线、数据生成</td></tr></tbody>
</table>
</div>

<figure id="figure-8-1"><a href="../../assets/figures/fig_029_fig29.png" target="_blank" rel="noopener" aria-label="查看图 8.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_029_fig29.png" alt="近似的质量 vs. 算力前沿。位于 SFT 上限线之上的方法超越了单纯监督微调所能达到的水平。位置仅为示意，与模型有关。" loading="lazy"></a><figcaption><span class="figure-number">图 8.1　</span>近似的质量 vs. 算力前沿。位于 SFT 上限线之上的方法超越了单纯监督微调所能达到的水平。位置仅为示意，与模型有关。</figcaption></figure>

<div class="admonition note"><p class="admonition-title">决策树：该用哪种方法？</p>
<ol>
<li><p><strong>有可验证 reward 吗？</strong>（数学/代码）<span class="arithmatex">\(\rightarrow\)</span> <strong>GRPO</strong></p></li>
<li><p><strong>在复杂任务上需要极致质量吗？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>PPO</strong></p></li>
<li><p><strong>有成对偏好吗？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>DPO</strong>（若噪声大则用 IPO）</p></li>
<li><p><strong>仅有未配对的二值反馈？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>KTO</strong></p></li>
<li><p><strong>显存受限、从 base 模型起步？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>ORPO</strong></p></li>
<li><p><strong>DPO 停滞、想要 on-policy？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>Online DPO</strong></p></li>
<li><p><strong>需要快速建立强基线？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>Best-of-N / RFT</strong></p></li>
</ol>
</div>
