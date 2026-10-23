# 第 6 章 · DPO——直接偏好优化 {#dpo-direct-preference-optimization}

## 6.1　动机 {#motivation}

<p>PPO 需要在显存中维持 4 个模型（policy、reference、reward 模型、value head）、复杂的 RL 基础设施，并且以不稳定著称。DPO <a class="book-citation" id="cite-rafailov2023direct-1" href="../../references.md#ref-rafailov2023direct" aria-label="参考文献 302">[302]</a> 提出的问题是：<em>我们能否跳过 RL，直接从偏好中学习？</em></p>

<p><strong>关键洞察</strong>：在 RLHF 目标（reward 最大化 + KL 惩罚）下的最优 policy 拥有<strong>解析解</strong>。由此我们可以推导出一个有监督 loss，隐式优化同一个目标。</p>

## 6.2　数学推导 {#mathematical-derivation}

<p><strong>第 1 步</strong>：RLHF 目标：<span class="arithmatex">\(\max_\pi \mathbb{E}_{x,y\sim\pi}[r(x,y)] - \beta D_\text{KL}[\pi\|\pi_\text{ref}]\)</span></p>

<p><strong>第 2 步</strong>：最优解为：<span class="arithmatex">\(\pi^*(y|x) = \frac{1}{Z(x)} \pi_\text{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)\)</span></p>

<p><strong>第 3 步</strong>：重排式子，用 policy 表达 reward：<span class="arithmatex">\(r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)\)</span></p>

<p><strong>第 4 步</strong>：代入 Bradley-Terry 偏好模型 <span class="arithmatex">\(P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))\)</span>。<span class="arithmatex">\(Z(x)\)</span> 项相互抵消！</p>



<div class="arithmatex">\[\boxed{\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]}
\tag{6.1}\]</div>

<div class="admonition tip"><p class="admonition-title">DPO 实际在做什么</p>
<p>将<strong>隐式 reward</strong> 定义为 <span class="arithmatex">\(\hat{r}(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\)</span>。</p>

<p>DPO 在最小化一个交叉熵 loss，其中“标签”是：被选回答的隐式 reward 应高于被拒回答。Margin 由 <span class="arithmatex">\(\beta\)</span> 控制：</p>

<ul>
<li><p><span class="arithmatex">\(\beta\)</span> 较大：需要较大的 margin <span class="arithmatex">\(\rightarrow\)</span> policy 移动得更激进 <span class="arithmatex">\(\rightarrow\)</span> 有遗忘风险</p></li>
<li><p><span class="arithmatex">\(\beta\)</span> 较小：较小的 margin 就够了 <span class="arithmatex">\(\rightarrow\)</span> policy 更靠近 reference <span class="arithmatex">\(\rightarrow\)</span> 更保守</p></li>
</ul>

<p>Reference 模型起到正则化作用：policy 必须用偏好对齐来“证明”自己偏离 reference 的合理性。</p>
</div>

## 6.3　Gradient 分析 {#gradient-analysis}

<p>DPO 的 gradient 可分解为：</p>



<div class="arithmatex">\[\nabla_\theta \mathcal{L} = -\beta \cdot \underbrace{\sigma(-\hat{r}_w + \hat{r}_l)}_{\text{weight: higher when model is wrong}} \cdot \left[\nabla_\theta \log\pi_\theta(y_w|x) - \nabla_\theta \log\pi_\theta(y_l|x)\right]
\tag{6.2}\]</div>

<p><strong>解读</strong>：Gradient 会提升“被选”项的概率，降低“被拒”项的概率。当模型当前更偏好错误答案时，权重最大——也就是说，它把学习重点放在“易混淆”的成对样本上。</p>

<div class="admonition example"><p class="admonition-title">一个具体的 DPO 例子</p>
<p><strong>Prompt</strong>：“向一个 10 岁的小孩解释量子纠缠。”</p>

<p><strong>被选</strong>（<span class="arithmatex">\(y_w\)</span>）：“想象你有两枚魔法硬币。你抛其中一枚，如果它是正面，另一枚不管离多远都会立刻变成反面！”<br><span class="arithmatex">\(\log\pi_\theta(y_w|x) = -15.3\)</span>，<span class="arithmatex">\(\log\pi_\text{ref}(y_w|x) = -16.1\)</span></p>

<p><strong>被拒</strong>（<span class="arithmatex">\(y_l\)</span>）：“量子纠缠是一种现象，其中两个粒子相互关联，以致于一个粒子的量子态无法被独立描述。”<br><span class="arithmatex">\(\log\pi_\theta(y_l|x) = -12.8\)</span>，<span class="arithmatex">\(\log\pi_\text{ref}(y_l|x) = -12.5\)</span></p>

<p><strong>隐式 reward</strong>：<span class="arithmatex">\(\hat{r}_w = 0.1 \times ((-15.3) - (-16.1)) = 0.08\)</span>，<span class="arithmatex">\(\hat{r}_l = 0.1 \times ((-12.8) - (-12.5)) = -0.03\)</span></p>

<p><strong>Loss 输入</strong>：<span class="arithmatex">\(\sigma(0.08 - (-0.03)) = \sigma(0.11) = 0.527\)</span></p>

<p><strong>Loss</strong>：<span class="arithmatex">\(-\log(0.527) = 0.64\)</span>——模型仅略微偏好被选回答。Gradient 会用力推动。</p>

<p>训练之后：被选回答的概率提升、被拒回答的概率下降，直到 margin 稳定在约 <span class="arithmatex">\(1/(2\beta)\)</span>。</p>
</div>

## 6.4　TRL 实现 {#trl-implementation-dup2}

<p>下面给出一个使用 HuggingFace TRL 的最小可运行示例。</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoModelForCausalLM</span><span class="p">,</span> <span class="n">AutoTokenizer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">peft</span><span class="w"> </span><span class="kn">import</span> <span class="n">LoraConfig</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">datasets</span><span class="w"> </span><span class="kn">import</span> <span class="n">load_dataset</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">AutoModelForCausalLM</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">,</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span> <span class="n">attn_implementation</span><span class="o">=</span><span class="s2">"flash_attention_2"</span><span class="p">)</span>
<span class="n">tokenizer</span> <span class="o">=</span> <span class="n">AutoTokenizer</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">)</span>

<span class="c1"># 数据集格式：{"prompt": str, "chosen": str, "rejected": str}</span>
<span class="n">dataset</span> <span class="o">=</span> <span class="n">load_dataset</span><span class="p">(</span><span class="s2">"argilla/ultrafeedback-binarized-preferences"</span><span class="p">)</span>

<span class="n">lora_config</span> <span class="o">=</span> <span class="n">LoraConfig</span><span class="p">(</span><span class="n">r</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span> <span class="n">lora_alpha</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span> <span class="n">lora_dropout</span><span class="o">=</span><span class="mf">0.05</span><span class="p">,</span>
    <span class="n">target_modules</span><span class="o">=</span><span class="p">[</span><span class="s2">"q_proj"</span><span class="p">,</span><span class="s2">"k_proj"</span><span class="p">,</span><span class="s2">"v_proj"</span><span class="p">,</span><span class="s2">"o_proj"</span><span class="p">,</span><span class="s2">"gate_proj"</span><span class="p">,</span><span class="s2">"up_proj"</span><span class="p">,</span><span class="s2">"down_proj"</span><span class="p">])</span>

<span class="n">dpo_config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./dpo_output"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>                    <span class="c1"># KL 正则强度</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">5e-7</span><span class="p">,</span>          <span class="c1"># 学习率非常低，利于稳定</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"sigmoid"</span><span class="p">,</span>         <span class="c1"># 标准 DPO loss</span>
    <span class="n">max_length</span><span class="o">=</span><span class="mi">2048</span><span class="p">,</span>             <span class="c1"># 最大序列长度</span>
    <span class="n">max_prompt_length</span><span class="o">=</span><span class="mi">1024</span><span class="p">,</span>      <span class="c1"># prompt 截断长度</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>  <span class="c1"># 有效 batch = 16</span>
    <span class="n">gradient_checkpointing</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>          <span class="c1"># DPO 易过拟合——只跑 1 个 epoch！</span>
    <span class="n">warmup_ratio</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
    <span class="n">logging_steps</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span>
    <span class="n">eval_strategy</span><span class="o">=</span><span class="s2">"steps"</span><span class="p">,</span>
    <span class="n">eval_steps</span><span class="o">=</span><span class="mi">200</span><span class="p">,</span>
    <span class="n">save_strategy</span><span class="o">=</span><span class="s2">"steps"</span><span class="p">,</span>
    <span class="n">save_steps</span><span class="o">=</span><span class="mi">500</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>             <span class="c1"># 配合 LoRA 时，ref = base 模型（无需另外拷贝！）</span>
    <span class="n">args</span><span class="o">=</span><span class="n">dpo_config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">[</span><span class="s2">"train"</span><span class="p">],</span>
    <span class="n">eval_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">[</span><span class="s2">"test"</span><span class="p">],</span>
    <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
    <span class="n">peft_config</span><span class="o">=</span><span class="n">lora_config</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
<span class="c1"># 需要监控的关键指标：train/rewards/chosen、train/rewards/rejected、train/rewards/margins</span>
</code></pre></div>

## 6.5　DPO 的完整机制 {#how-dpo-works-full-mechanics}

<p>本节给出 DPO 的完整计算细节——训练时在 token 级别究竟发生了什么。</p>

### 6.5.1　序列级对数概率 {#sequence-level-log-probabilities}

<p>DPO 中的关键量是：在给定 prompt <span class="arithmatex">\(x\)</span> 下，<strong>整段序列</strong> <span class="arithmatex">\(y = (y_1, y_2, \ldots, y_T)\)</span> 的对数概率。它是<strong>逐 token 对数概率之和</strong>：</p>



<div class="arithmatex">\[\boxed{\log \pi_\theta(y|x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{&lt;t})}
\tag{6.3}\]</div>

<p>其中每一项 <span class="arithmatex">\(\log \pi_\theta(y_t | x, y_{&lt;t})\)</span> 都是位置 <span class="arithmatex">\(t\)</span> 处的 log-softmax 输出，对应序列中<em>实际</em>出现的 token <span class="arithmatex">\(y_t\)</span>。这与标准语言建模中的交叉熵 loss 完全一致——只是这里我们<strong>求和</strong>而非求均值。</p>

<p><strong>关键细节</strong>：Gradient 会流过 <span class="arithmatex">\(y_w\)</span> 和 <span class="arithmatex">\(y_l\)</span> 中的<strong>每一个 token 位置</strong>。中间 token 不做掩码——每个 token 都对序列级对数概率有贡献。</p>

### 6.5.2　DPO Loss 的分解 {#the-dpo-loss-decomposed}

<p>从 loss 开始：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\beta \cdot h_\theta(x, y_w, y_l)\right)\right]
\tag{6.4}\]</div>

<p>其中“隐式 reward margin” <span class="arithmatex">\(h_\theta\)</span> 为：</p>



<div class="arithmatex">\[h_\theta(x, y_w, y_l) = \underbrace{\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}}_{\text{chosen reward proxy}} - \underbrace{\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}}_{\text{rejected reward proxy}}
\tag{6.5}\]</div>

<p>展开为 token 级各项：</p>



<div class="arithmatex">\[\boxed{h_\theta = \sum_{t=1}^{|y_w|}\!\left[\log\pi_\theta(y_w^t | x, y_w^{&lt;t}) - \log\pi_{\text{ref}}(y_w^t | x, y_w^{&lt;t})\right] - \sum_{t=1}^{|y_l|}\!\left[\log\pi_\theta(y_l^t | x, y_l^{&lt;t}) - \log\pi_{\text{ref}}(y_l^t | x, y_l^{&lt;t})\right]}
\tag{6.6}\]</div>

### 6.5.3　Forward Pass：逐步详解 {#forward-pass-step-by-step}

<p>对一个训练样本 <span class="arithmatex">\((x, y_w, y_l)\)</span>：</p>

<ol>
<li><p><strong>拼接</strong>：组成两条序列：<span class="arithmatex">\([x; y_w]\)</span> 与 <span class="arithmatex">\([x; y_l]\)</span>。在 batch 内 pad 到相同长度。</p></li>
<li><p><strong>Forward pass（policy <span class="arithmatex">\(\pi_\theta\)</span>）</strong>：把两条序列都喂进模型；在每个回答位置收集 logits。</p></li>
<li><p><strong>提取对数概率</strong>：在回答中每个位置 <span class="arithmatex">\(t\)</span> 处，取 <span class="arithmatex">\(\log\text{softmax}(\text{logits}_t)[y_t]\)</span>——即实际 token 的对数概率。</p></li>
<li><p><strong>对 token 求和</strong>：</p>



<div class="arithmatex">\[\begin{align*}
\text{logp\_chosen} &amp;= \sum_{t \in \text{response positions}} \log\pi_\theta(y_w^t | x, y_w^{&lt;t}) \tag{6.7}\\

\text{logp\_rejected} &amp;= \sum_{t \in \text{response positions}} \log\pi_\theta(y_l^t | x, y_l^{&lt;t})
\tag{6.8}
\end{align*}\]</div></li>
<li><p><strong>减去参考</strong>（预计算或来自第二次 forward pass）：</p>



<div class="arithmatex">\[\begin{align*}
\text{ratio\_w} &amp;= \text{logp\_chosen} - \text{ref\_logp\_chosen} \tag{6.9}\\

\text{ratio\_l} &amp;= \text{logp\_rejected} - \text{ref\_logp\_rejected}
\tag{6.10}
\end{align*}\]</div></li>
<li><p><strong>计算 loss</strong>：<span class="arithmatex">\(\mathcal{L} = -\log\sigma(\beta \cdot (\text{ratio\_w} - \text{ratio\_l}))\)</span></p></li>
<li><p><strong>Backward pass</strong>：Gradient 沿步骤 5 <span class="arithmatex">\(\rightarrow\)</span> 4 <span class="arithmatex">\(\rightarrow\)</span> 3 <span class="arithmatex">\(\rightarrow\)</span> 2 回传，更新 <span class="arithmatex">\(\theta\)</span>。</p></li>
</ol>

### 6.5.4　Token 级 Gradient 分析 {#token-level-gradient-analysis}

<p><strong>是不是每个 token 都收到 gradient？</strong> 是。在被选序列中，位置 <span class="arithmatex">\(t\)</span> 处 logits 上的 gradient 为：</p>



<div class="arithmatex">\[\frac{\partial \mathcal{L}}{\partial \text{logits}_t^{(w)}} = -\underbrace{\sigma(-\beta \cdot h_\theta)}_{\text{scaling factor}} \cdot \beta \cdot \frac{\partial \log\pi_\theta(y_w^t | \cdot)}{\partial \text{logits}_t^{(w)}}
\tag{6.11}\]</div>

<p><strong>关键洞察</strong>：缩放因子 <span class="arithmatex">\(\sigma(-\beta \cdot h_\theta)\)</span> 在两条序列的<strong>所有 token 间共享</strong>，相当于一个自适应学习率：</p>

<ul>
<li><p>当 <span class="arithmatex">\(h_\theta\)</span> 较小（模型分不清被选与被拒）：缩放 <span class="arithmatex">\(\approx 0.5\)</span>——gradient 强，激进地学。</p></li>
<li><p>当 <span class="arithmatex">\(h_\theta\)</span> 较大（模型已经偏好被选）：缩放 <span class="arithmatex">\(\approx 0\)</span>——gradient 可忽略，避免过拟合。</p></li>
</ul>

<p><strong>对被选 token 的影响</strong>：概率被<em>提升</em>（对数概率被推高）。<br><strong>对被拒 token 的影响</strong>：概率被<em>降低</em>（对数概率被压低）。<br><strong>相对 reference</strong>：只有相对 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 的<em>差值</em>才重要。如果模型本就给被选回答高概率（与 reference 一致），那么 gradient 就几乎为零。</p>

### 6.5.5　逐 Token vs. 序列级：长度归一化 {#per-token-vs.-sequence-level-length-normalization}

<p>一个微妙问题：越长的序列对数概率天然越低（求和的项更多，且每项 <span class="arithmatex">\(\leq 0\)</span>）。若 <span class="arithmatex">\(|y_w| \gg |y_l|\)</span>，loss 会偏向于偏好更短的回答。</p>

<p><strong>解决方案</strong>：</p>

<ul>
<li><p><strong>长度归一化的 DPO</strong>：用 <span class="arithmatex">\(\frac{1}{|y|}\sum_t \log\pi_\theta(y_t|\cdot)\)</span> 替换 <span class="arithmatex">\(\log\pi_\theta(y|x)\)</span>。一些实现采用此做法（SimPO 即如此）。</p></li>
<li><p><strong>标准 DPO</strong>：使用原始求和（不归一化）。这会<em>隐式</em>惩罚冗长——模型必须对被选回答中的每个 token 都赋予高概率。</p></li>
<li><p><strong>实际影响</strong>：在 benchmark 上，长度归一化的 DPO 能减少长度博弈，但可能损害指令遵循质量。生产中更常用未归一化的标准版本。</p></li>
</ul>

### 6.5.6　标签掩码：哪些 token 会收到 Gradient {#label-masking-what-gets-gradients}

<div class="admonition note"><p class="admonition-title">DPO 中哪些 token 收到 Gradient</p>
<ul>
<li><p><strong>Prompt token</strong>（<span class="arithmatex">\(x\)</span>）：<strong>无 gradient</strong>。Loss 仅在回答位置上计算。Prompt token 提供上下文，但其 logits 不参与 <span class="arithmatex">\(\log\pi(y|x)\)</span>。</p></li>
<li><p><strong>被选回答 token</strong>（<span class="arithmatex">\(y_w\)</span>）：<strong>所有 token 都收到 gradient</strong>。每个 <span class="arithmatex">\(y_w^t\)</span> 都贡献到求和；gradient 推高它们的概率。</p></li>
<li><p><strong>被拒回答 token</strong>（<span class="arithmatex">\(y_l\)</span>）：<strong>所有 token 都收到 gradient</strong>。每个 <span class="arithmatex">\(y_l^t\)</span> 都贡献到求和；gradient 压低它们的概率。</p></li>
<li><p><strong>Padding token</strong>：<strong>无 gradient</strong>。通过 attention mask 屏蔽掉。</p></li>
</ul>
</div>

### 6.5.7　伪代码：DPO 训练步 {#pseudocode-dpo-training-step}

<div class="admonition example"><p class="admonition-title">DPO Forward + Backward（PyTorch 风格）</p>
<div class="highlight"><pre><span></span><code><span class="k">def</span><span class="w"> </span><span class="nf">dpo_loss</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">ref_model</span><span class="p">,</span> <span class="n">batch</span><span class="p">,</span> <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""一个 DPO 训练步。"""</span>
    <span class="c1"># batch 包含：input_ids_chosen、input_ids_rejected、</span>
    <span class="c1">#            labels_chosen、labels_rejected（prompt 被掩为 -100）</span>

    <span class="c1"># 1. Forward pass：得到逐 token 对数概率</span>
    <span class="n">logps_chosen</span> <span class="o">=</span> <span class="n">get_sequence_logprob</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">batch</span><span class="p">[</span><span class="s2">"chosen"</span><span class="p">])</span>
    <span class="n">logps_rejected</span> <span class="o">=</span> <span class="n">get_sequence_logprob</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">batch</span><span class="p">[</span><span class="s2">"rejected"</span><span class="p">])</span>

    <span class="c1"># 2. 参考模型对数概率（预计算或在此处计算）</span>
    <span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">no_grad</span><span class="p">():</span>
        <span class="n">ref_logps_chosen</span> <span class="o">=</span> <span class="n">get_sequence_logprob</span><span class="p">(</span><span class="n">ref_model</span><span class="p">,</span> <span class="n">batch</span><span class="p">[</span><span class="s2">"chosen"</span><span class="p">])</span>
        <span class="n">ref_logps_rejected</span> <span class="o">=</span> <span class="n">get_sequence_logprob</span><span class="p">(</span><span class="n">ref_model</span><span class="p">,</span> <span class="n">batch</span><span class="p">[</span><span class="s2">"rejected"</span><span class="p">])</span>

    <span class="c1"># 3. 计算隐式 reward margin</span>
    <span class="n">chosen_rewards</span> <span class="o">=</span> <span class="n">beta</span> <span class="o">*</span> <span class="p">(</span><span class="n">logps_chosen</span> <span class="o">-</span> <span class="n">ref_logps_chosen</span><span class="p">)</span>
    <span class="n">rejected_rewards</span> <span class="o">=</span> <span class="n">beta</span> <span class="o">*</span> <span class="p">(</span><span class="n">logps_rejected</span> <span class="o">-</span> <span class="n">ref_logps_rejected</span><span class="p">)</span>

    <span class="c1"># 4. DPO loss = -log(sigmoid(被选 reward - 被拒 reward))</span>
    <span class="n">loss</span> <span class="o">=</span> <span class="o">-</span><span class="n">F</span><span class="o">.</span><span class="n">logsigmoid</span><span class="p">(</span><span class="n">chosen_rewards</span> <span class="o">-</span> <span class="n">rejected_rewards</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>
    <span class="k">return</span> <span class="n">loss</span>

<span class="k">def</span><span class="w"> </span><span class="nf">get_sequence_logprob</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">sequences</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""仅在回答 token 上求和的对数概率。"""</span>
    <span class="n">outputs</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">sequences</span><span class="p">[</span><span class="s2">"input_ids"</span><span class="p">],</span> <span class="n">attention_mask</span><span class="o">=</span><span class="n">sequences</span><span class="p">[</span><span class="s2">"mask"</span><span class="p">])</span>
    <span class="n">logits</span> <span class="o">=</span> <span class="n">outputs</span><span class="o">.</span><span class="n">logits</span><span class="p">[:,</span> <span class="p">:</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="p">:]</span>  <span class="c1"># 为 next-token 预测做位移</span>

    <span class="c1"># 收集实际 token 的对数概率</span>
    <span class="n">labels</span> <span class="o">=</span> <span class="n">sequences</span><span class="p">[</span><span class="s2">"labels"</span><span class="p">][:,</span> <span class="mi">1</span><span class="p">:]</span>  <span class="c1"># 位移后的标签</span>
    <span class="n">log_probs</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">log_softmax</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>
    <span class="n">token_logps</span> <span class="o">=</span> <span class="n">log_probs</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="n">labels</span><span class="o">.</span><span class="n">unsqueeze</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">))</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span>

    <span class="c1"># 掩码：只对回答 token 求和（labels != -100）</span>
    <span class="n">mask</span> <span class="o">=</span> <span class="p">(</span><span class="n">labels</span> <span class="o">!=</span> <span class="o">-</span><span class="mi">100</span><span class="p">)</span><span class="o">.</span><span class="n">float</span><span class="p">()</span>
    <span class="k">return</span> <span class="p">(</span><span class="n">token_logps</span> <span class="o">*</span> <span class="n">mask</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>  <span class="c1"># 形状：[batch_size]</span>
</code></pre></div>
</div>

### 6.5.8　常见陷阱 {#common-pitfalls}

<div class="admonition warning"><p class="admonition-title">DPO 实现中的陷阱</p>
<ul>
<li><p><strong>忘记屏蔽 prompt</strong>：如果 prompt token 被纳入对数概率求和，模型就会优化 prompt 的似然（无意义），且有效 <span class="arithmatex">\(\beta\)</span> 也会出错。</p></li>
<li><p><strong>用均值替代求和</strong>：<span class="arithmatex">\(\frac{1}{T}\sum_t \log\pi\)</span> 与 <span class="arithmatex">\(\sum_t \log\pi\)</span> 会带来不同的隐式长度惩罚。<span class="arithmatex">\(\pi_\theta\)</span> 与 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 之间必须保持一致。</p></li>
<li><p><strong>过期的参考模型</strong>：若 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 与 <span class="arithmatex">\(\pi_\theta\)</span> 相距过远（如 base 模型 vs. 微调后的模型），KL 项会占主导，gradient 消失。解决办法：使用 SFT checkpoint（而非 base）作为 reference。</p></li>
<li><p><strong><span class="arithmatex">\(\beta\)</span> 过大</strong>：放大对数概率差值 <span class="arithmatex">\(\rightarrow\)</span> sigmoid 饱和 <span class="arithmatex">\(\rightarrow\)</span> gradient 归零。从 <span class="arithmatex">\(\beta = 0.1\)</span> 起步，在 <span class="arithmatex">\([0.05, 0.5]\)</span> 区间内调参。</p></li>
<li><p><strong><span class="arithmatex">\(\beta\)</span> 过小</strong>：理论上允许 policy 更大程度地偏离 reference（KL 约束更弱），但 gradient <span class="arithmatex">\(\propto \beta \cdot \sigma(-\beta h)\)</span> 会变得极小 <span class="arithmatex">\(\rightarrow\)</span> loss 地形平坦 <span class="arithmatex">\(\rightarrow\)</span> 收敛极慢。模型“被允许”走得很远，却几乎收不到任何告诉它<em>往哪走</em>的信号。</p></li>
</ul>
</div>

## 6.6　DPO 变体及其各自的失败场景 {#dpo-variants-and-when-each-fails}

<div class="admonition warning"><p class="admonition-title">DPO 何时会失败</p>
<p><strong>1. 分布漂移</strong>：偏好数据来自旧模型。当前 policy 生成的文本截然不同 <span class="arithmatex">\(\rightarrow\)</span> loss 在无关样本上做优化。</p>

<p><strong>2. 无探索</strong>：无法发现数据集之外的行为。陷入局部最优。</p>

<p><strong>3. Reference 崩塌</strong>：Reference 太强，policy 动弹不得；太弱则失去正则化作用。</p>

<p><strong>4. 数据质量</strong>：噪声标签会毒化训练。与对大量样本平均的 PPO 不同，DPO 会“记住”单个的样本对。</p>

<p><strong>5. 偏好数据多样性</strong>：要确保被选/被拒成对样本覆盖质量差异的完整谱系（而不仅是“好 vs 极差”）。在<em>解法思路</em>上有差异的样本对（而不仅是质量差异）能教出更丰富的 policy 区分能力。</p>
</div>

## 6.7　<span class="arithmatex">\(\beta\)</span> 选择指南 {#beta-selection-guide}

<span id="table-6-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><span class="arithmatex">\(\beta\)</span></th><th scope="col"><strong>风格</strong></th><th scope="col"><strong>使用场景</strong></th></tr></thead>
<tbody><tr><td>0.01</td><td>非常激进</td><td>仅当数据极其干净、且你需要大幅度分布迁移时使用</td></tr>
<tr><td>0.05</td><td>激进</td><td>数据较好，希望相对 SFT 取得明显改进</td></tr>
<tr><td>0.1</td><td>标准</td><td>默认起点。质量与稳定性的良好平衡</td></tr>
<tr><td>0.2</td><td>保守</td><td>噪声较多，或模型已接近期望行为</td></tr>
<tr><td>0.5</td><td>非常保守</td><td>安全微调场景，必须保住已有能力</td></tr></tbody>
</table>
</div>

## 6.8　DPO 的 Batch Size 配置与扩展 {#dpo-batch-size-configuration-and-scaling}

<p>与在单序列 token 预测上工作的标准 SFT 不同，DPO 使用一个比较“被偏好序列与被拒序列”的<strong>成对 loss</strong>。这从根本上改变了显存利用与优化稳定性。</p>

### 6.8.1　全局 Batch Size 目标 {#global-batch-size-target}

<p>跨多种 DPO 实现的经验证据给出了一个最佳全局 batch size 范围：</p>



<div class="arithmatex">\[\boxed{B_{\text{global}} \in [32, 128]}
\tag{6.12}\]</div>

<ul>
<li><p><span class="arithmatex">\(B_{\text{global}} &lt; 32\)</span>：隐式 reward 估计中的 gradient 噪声严重 <span class="arithmatex">\(\rightarrow\)</span> policy 在多个对齐目标之间（如有用 vs 安全）破坏性振荡。</p></li>
<li><p><span class="arithmatex">\(B_{\text{global}} &gt; 128\)</span>：收敛速度边际收益递减；分布式算力间通信开销巨大。</p></li>
</ul>

### 6.8.2　数学分解 {#mathematical-decomposition}

<p>由于 DPO 同时加载<strong>两份</strong>模型副本（活动 policy <span class="arithmatex">\(\pi_\theta\)</span> + 冻结 reference <span class="arithmatex">\(\pi_{\text{ref}}\)</span>），每条序列的显存翻倍。全局 batch size 可分解为：</p>



<div class="arithmatex">\[\boxed{B_{\text{global}} = B_{\text{micro}} \times N_{\text{GPUs}} \times K_{\text{accum}}}
\tag{6.13}\]</div>

<ul>
<li><p><span class="arithmatex">\(B_{\text{micro}}\)</span>：每设备 micro-batch 大小（每次 forward pass 的偏好对数）。</p></li>
<li><p><span class="arithmatex">\(N_{\text{GPUs}}\)</span>：并行处理数据的设备数量。</p></li>
<li><p><span class="arithmatex">\(K_{\text{accum}}\)</span>：在权重更新前累计 gradient 的步数。</p></li>
</ul>

<p><strong>成对倍数因子</strong>：单个 DPO 数据样本包含 prompt（<span class="arithmatex">\(x\)</span>）、被选（<span class="arithmatex">\(y_w\)</span>）和被拒（<span class="arithmatex">\(y_l\)</span>）。每个 micro-batch 的实际 tensor 负载为：</p>



<div class="arithmatex">\[T_{\text{sequences}} = 2 \times B_{\text{micro}}
\tag{6.14}\]</div>

<p>对于在 80GB GPU 上、上下文长度 4096–8192 token 的 <span class="arithmatex">\(&gt;\)</span>7B 参数模型，物理上限被刚性约束在 <span class="arithmatex">\(B_{\text{micro}} \in [1, 2]\)</span>。</p>

### 6.8.3　分布式扩展配置 {#distributed-scaling-configurations}

<p class="table-caption" id="table-6-1"><span class="table-number">表 6.1　</span>DPO 训练的分布式扩展配置（目标 <span class="arithmatex">\(B_{\text{global}} = 64\)</span>）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>配置</strong></th><th scope="col"><strong>单 GPU</strong></th><th scope="col"><strong>8 卡节点</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(B_{\text{global}}\)</span></td><td>64</td><td>64</td></tr>
<tr><td><span class="arithmatex">\(B_{\text{micro}}\)</span></td><td>2（4 条序列）</td><td>2（4 条序列）</td></tr>
<tr><td><span class="arithmatex">\(N_{\text{GPUs}}\)</span></td><td>1</td><td>8</td></tr>
<tr><td><span class="arithmatex">\(K_{\text{accum}}\)</span></td><td>32 步</td><td>4 步</td></tr>
<tr><td>吞吐</td><td>串行/慢</td><td>高并行吞吐</td></tr></tbody>
</table>
</div>

### 6.8.4　显存优化：预计算 Reference 对数概率 {#vram-optimization-pre-computing-reference-log-probabilities}

<p>DPO loss：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
\tag{6.15}\]</div>

<p>由于 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 在整个训练过程中<strong>完全静态</strong>，其输出可被预先计算：</p>

<div class="admonition note"><p class="admonition-title">Reference 模型驱逐策略</p>
<ol>
<li><p>训练开始前，仅用 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 对数据集 <span class="arithmatex">\(\mathcal{D}\)</span> 执行一次 forward pass。</p></li>
<li><p>将标量 <span class="arithmatex">\(\log \pi_{\text{ref}}(y_w|x)\)</span> 和 <span class="arithmatex">\(\log \pi_{\text{ref}}(y_l|x)\)</span> 缓存到磁盘。</p></li>
<li><p><strong>把 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 完全从 GPU 显存中驱逐。</strong></p></li>
</ol>

<p><strong>效果</strong>：可用 GPU 显存翻倍 <span class="arithmatex">\(\rightarrow\)</span> <span class="arithmatex">\(B_{\text{micro}}\)</span> 可从 1–2 提升到 4–8，从而最大化硬件利用率与训练吞吐。</p>

<p><em>实现</em>：在 TRL 中，于 <code>DPOConfig</code> 设置 <code>precompute_ref_log_probs=True</code>。对 70B 模型，这能在整个集群上节省约 140GB 的 GPU 显存。</p>
</div>

## 6.9　DPO 扩展与变体 {#sec:dpo-variants}

<p>直接偏好优化（Direct Preference Optimization, DPO）通过推导 reward 函数与最优 policy 之间的解析映射，把 RLHF 重新表述为一个有监督学习问题。标准 DPO loss 为：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(q,y_w,y_l)}\!\left[
    \log \sigma\!\left(
      \beta \log \frac{\pi_\theta(y_w|q)}{\pi_{\text{ref}}(y_w|q)}
      - \beta \log \frac{\pi_\theta(y_l|q)}{\pi_{\text{ref}}(y_l|q)}
    \right)
  \right],\]</div>

<p>其中 <span class="arithmatex">\(y_w\)</span> 为被偏好（获胜）的回答，<span class="arithmatex">\(y_l\)</span> 为被拒（失败）的回答，<span class="arithmatex">\(\beta\)</span> 控制 KL 惩罚强度。下面的小节涵盖最重要的扩展与变体。</p>

### 6.9.1　f-DPO——广义 f-Divergence DPO {#sec:fdpo}

<div class="admonition tip"><p class="admonition-title">超越反向 KL</p>
<p>标准 DPO 使用反向 KL 散度作为 policy 与 reference 之间的正则项。反向 KL 是<em>寻峰</em>的：它倾向于把概率质量集中在少数高 reward 回答上。前向 KL 则是<em>覆盖</em>的：它把概率质量分散去覆盖所有合理回答。f-DPO <a class="book-citation" id="cite-wang2023fdpo-1" href="../../references.md#ref-wang2023fdpo" aria-label="参考文献 363">[363]</a> 推广到任意 f-散度，允许实践者在这些行为间做权衡。</p>
</div>

<p>f-DPO loss 用 f-散度生成器的导数替换对数比：</p>



<div class="arithmatex">\[\mathcal{L}_{f\text{-DPO}} = -\mathbb{E}\!\left[
    f&#x27;\!\left(\frac{\pi_\theta(y_w|q)}{\pi_{\text{ref}}(y_w|q)}\right)
    - f&#x27;\!\left(\frac{\pi_\theta(y_l|q)}{\pi_{\text{ref}}(y_l|q)}\right)
  \right],\]</div>

<p>其中 <span class="arithmatex">\(f&#x27;\)</span> 是 f-散度生成器函数的导数。</p>

<div class="admonition note"><p class="admonition-title">TRL 中的 f-Divergence 选项</p>
<ul>
<li><p><strong>reverse_kl</strong>：<span class="arithmatex">\(f&#x27;(u) = \log u\)</span>。即标准 DPO。寻峰。</p></li>
<li><p><strong>forward_kl</strong>：<span class="arithmatex">\(f&#x27;(u) = -1/u\)</span>。覆盖式。多样性更好。</p></li>
<li><p><strong>js_divergence</strong>：<span class="arithmatex">\(f&#x27;(u) = \log(2u/(u+1))\)</span>。寻峰/覆盖之间的平衡。</p></li>
<li><p><strong>alpha_divergence</strong>：<span class="arithmatex">\(f&#x27;(u) = u^{\alpha-1}\)</span>。在前向 KL 与反向 KL 之间插值。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的 f-DPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="c1"># Jensen-Shannon 散度（平衡）</span>
<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">f_divergence_type</span><span class="o">=</span><span class="s2">"js_divergence"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="c1"># Alpha 散度（alpha=0：前向 KL；alpha=1：反向 KL）</span>
<span class="n">config_alpha</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">f_divergence_type</span><span class="o">=</span><span class="s2">"alpha_divergence"</span><span class="p">,</span>
    <span class="n">f_alpha_divergence_coef</span><span class="o">=</span><span class="mf">0.5</span><span class="p">,</span>   <span class="c1"># alpha 参数</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">何时使用 f-DPO</p>
<ul>
<li><p>当你想在多样性与质量之间取得平衡时，使用 <strong>JS 散度</strong>。</p></li>
<li><p>对多样性至关重要的创意任务，使用<strong>前向 KL</strong>。</p></li>
<li><p>对有唯一正确答案的任务，使用<strong>反向 KL</strong>（即标准 DPO）。</p></li>
<li><p>使用<strong>alpha 散度</strong>可连续插值并精调这一权衡。</p></li>
</ul>
</div>

### 6.9.2　Robust DPO {#sec:robust-dpo}

<div class="admonition tip"><p class="admonition-title">偏好数据中的噪声标签</p>
<p>人类偏好标注是有噪声的。标注者会意见分歧、会出错，有时会把“被偏好/被拒”的标签搞反。标准 DPO 把所有标签视为 ground truth，可能导致模型对噪声过拟合。Robust DPO <a class="book-citation" id="cite-chowdhury2024robustdpo-1" href="../../references.md#ref-chowdhury2024robustdpo" aria-label="参考文献 56">[56]</a> 在已知噪声模型下解析性地对 loss 去偏。</p>
</div>

<p>假设每个标签以概率 <span class="arithmatex">\(\epsilon\)</span>（噪声率）被翻转。去偏后的 loss 为：</p>



<div class="arithmatex">\[\boxed{
    \mathcal{L}_{\text{robust}} =
    \frac{(1-\epsilon)\,\mathcal{L}_{\text{DPO}}(y_w, y_l)
          - \epsilon\,\mathcal{L}_{\text{DPO}}(y_l, y_w)}{1 - 2\epsilon},
  }\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{L}_{\text{DPO}}(y_w, y_l)\)</span> 是把 <span class="arithmatex">\(y_w\)</span> 视为被偏好的标准 DPO loss，而 <span class="arithmatex">\(\mathcal{L}_{\text{DPO}}(y_l, y_w)\)</span> 是把标签翻转后的 loss。该修正去除了标签噪声引入的偏置。</p>

<div class="admonition note"><p class="admonition-title">Robust DPO 的直觉</p>
<p>该公式是一个线性组合，把翻转标签的贡献“减掉”。当 <span class="arithmatex">\(\epsilon=0\)</span> 时，它退化为标准 DPO；当 <span class="arithmatex">\(\epsilon=0.5\)</span> 时，分母趋于零——标签纯属噪声，无法学习。实际中，<span class="arithmatex">\(\epsilon \in [0.05, 0.2]\)</span> 覆盖了大多数真实标注噪声水平。</p>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的 Robust DPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"robust"</span><span class="p">,</span>
    <span class="n">label_smoothing</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>   <span class="c1"># 对应 epsilon = 0.1</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

### 6.9.3　TR-DPO——信任域 DPO {#sec:tr-dpo}

<div class="admonition tip"><p class="admonition-title">过期 Reference 模型问题</p>
<p>标准 DPO 在整个训练过程中使用固定的 reference 模型 <span class="arithmatex">\(\pi_{\text{ref}}\)</span>。随着 policy <span class="arithmatex">\(\pi_\theta\)</span> 改进，KL 惩罚 <span class="arithmatex">\(\beta \log(\pi_\theta/\pi_{\text{ref}})\)</span> 不断增长，最终会主导 loss 并阻止进一步改进。TR-DPO <a class="book-citation" id="cite-gorbatenko2024trdpo-1" href="../../references.md#ref-gorbatenko2024trdpo" aria-label="参考文献 116">[116]</a> 周期性地更新 reference 模型，使其跟随当前 policy。</p>
</div>

<p>TR-DPO 使用指数滑动平均（EMA）更新 reference 模型：</p>



<div class="arithmatex">\[\pi_{\text{ref}}^{(t+1)} \leftarrow
    \alpha \cdot \pi_\theta^{(t)} + (1-\alpha) \cdot \pi_{\text{ref}}^{(t)},\]</div>

<p>其中 <span class="arithmatex">\(\alpha \in (0,1)\)</span> 是混合系数。每 <span class="arithmatex">\(T_{\text{sync}}\)</span> 个 gradient 步执行一次。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 TR-DPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"sigmoid"</span><span class="p">,</span>        <span class="c1"># 标准 DPO loss</span>
    <span class="n">sync_ref_model</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>        <span class="c1"># 启用 TR-DPO</span>
    <span class="n">ref_model_mixup_alpha</span><span class="o">=</span><span class="mf">0.6</span><span class="p">,</span>  <span class="c1"># alpha：混入当前 policy 的比例</span>
    <span class="n">ref_model_sync_steps</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span>   <span class="c1"># T_sync：每 512 步同步一次</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">何时使用 TR-DPO</p>
<ul>
<li><p>Policy 已偏离初始 reference 较远的长训练任务。</p></li>
<li><p>当观察到 DPO loss 因 KL 惩罚主导而过早平台化时。</p></li>
<li><p>从当前 policy 收集新偏好数据的迭代 DPO 流水线。</p></li>
<li><p><span class="arithmatex">\(\alpha\)</span> 接近 1 表示快更新 reference；接近 0 表示慢更新。</p></li>
</ul>
</div>

### 6.9.4　EXO——精确优化 {#sec:exo}

<div class="admonition tip"><p class="admonition-title">DPO 的 KL 方向问题</p>
<p>DPO 是在反向 KL 约束下求解最优 policy 而推导出的。然而，由此得到的 loss 实际上在 reward 空间中优化的是一个<em>前向</em> KL 目标，这个方向是错的。EXO <a class="book-citation" id="cite-ji2024exo-1" href="../../references.md#ref-ji2024exo" aria-label="参考文献 163">[163]</a> 通过使用反向 KL 概率匹配修正了这一点，这正是对齐在理论上正确的目标。</p>
</div>

<p>EXO 最小化模型分布与目标（reward 最优）分布之间的反向 KL：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{EXO}} = \mathbb{E}_{y \sim \pi_\theta}\!\left[
    \log \frac{\pi_\theta(y|q)}{p^*(y|q)}
  \right],\]</div>

<p>其中 <span class="arithmatex">\(p^*(y|q) \propto \pi_{\text{ref}}(y|q) \exp(r(y,q)/\beta)\)</span> 即最优 policy。实践中，EXO 利用可用的偏好对来近似该量：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{EXO}} \approx -\mathbb{E}\!\left[
    \log \sigma\!\left(
      \beta \log \frac{\pi_{\text{ref}}(y_w|q)}{\pi_\theta(y_w|q)}
      - \beta \log \frac{\pi_{\text{ref}}(y_l|q)}{\pi_\theta(y_l|q)}
    \right)
  \right].\]</div>

<p>注意：相对于 DPO，<span class="arithmatex">\(\pi_\theta\)</span> 与 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 的角色被<em>对调</em>了。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 EXO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"exo_pair"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

### 6.9.5　NCA——噪声对比对齐 {#sec:nca}

<div class="admonition tip"><p class="admonition-title">DPO 的似然崩塌</p>
<p>DPO 已知的失败模式之一是<em>似然崩塌</em>：模型学到了降低被拒回答的概率，但同时也降低了被选回答的概率（因为 loss 只在意它们的<em>差值</em>）。NCA <a class="book-citation" id="cite-chen2024nca-1" href="../../references.md#ref-chen2024nca" aria-label="参考文献 43">[43]</a> 增加了一个绝对似然项以防止这一点。</p>
</div>

<p>NCA 把对齐重新表述为噪声对比估计。Loss 包含三项：</p>



<div class="arithmatex">\[\boxed{
    \mathcal{L}_{\text{NCA}} =
      -\log \sigma(r_w)
      - \tfrac{1}{2}\log \sigma(-r_w)
      - \tfrac{1}{2}\log \sigma(-r_l),
  }\]</div>

<p>其中 <span class="arithmatex">\(r_y = \beta \log(\pi_\theta(y|q)/\pi_{\text{ref}}(y|q))\)</span> 是隐式 reward。第一项鼓励 <span class="arithmatex">\(y_w\)</span> 获得高 reward；第二、三项则同时惩罚 <span class="arithmatex">\(y_w\)</span> 与 <span class="arithmatex">\(y_l\)</span> 上的高 reward（防止崩塌）。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 NCA</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"nca_pair"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.01</span><span class="p">,</span>   <span class="c1"># beta 较小：绝对似然项占主导</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">何时使用 NCA</p>
<ul>
<li><p>当观察到 DPO 训练过程中被选回答的概率在下降时。</p></li>
<li><p>对于看重绝对回答质量、而不仅仅是相对排序的任务。</p></li>
<li><p>使用较小的 <span class="arithmatex">\(\beta\)</span>（如 0.01）让绝对似然项获得更高权重。</p></li>
</ul>
</div>

### 6.9.6　SLiC-HF——序列似然校准 {#sec:slic}

<div class="admonition tip"><p class="admonition-title">作为简化替代的 Hinge Loss</p>
<p>DPO 中的 log-sigmoid loss 虽然平滑，但当 margin 较大时收敛缓慢。SLiC-HF <a class="book-citation" id="cite-zhao2023slichf-1" href="../../references.md#ref-zhao2023slichf" aria-label="参考文献 432">[432]</a> 使用 hinge loss：当 margin 超过阈值时为零，否则为线性。它更简单、更快，且效果出乎意料地有竞争力。</p>
</div>

<p>SLiC-HF loss 为：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{SLiC}} = \max\!\left(0,\;
    \delta - \beta\log\frac{\pi_\theta(y_w|q)}{\pi_{\text{ref}}(y_w|q)}
    + \beta\log\frac{\pi_\theta(y_l|q)}{\pi_{\text{ref}}(y_l|q)}
  \right),\]</div>

<p>其中 <span class="arithmatex">\(\delta\)</span> 是 margin 阈值。当模型在被选与被拒回答之间已经赋予 <span class="arithmatex">\(\delta\)</span> 大小的 margin 时，loss 为零。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 SLiC-HF</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"hinge"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

### 6.9.7　Iterative RPO——推理偏好优化 {#sec:rpo}

<div class="admonition tip"><p class="admonition-title">DPO 会“忘记”如何生成</p>
<p>标准 DPO 训练模型去<em>判别</em>被选与被拒回答的优劣。但对于推理任务，模型还需要<em>生成</em>正确的推理轨迹。一个能判别却不能生成的模型在推理时毫无用处。RPO 在被选回答上加入一个 NLL（负对数似然）项，确保模型学会生成它。</p>
</div>

<p>RPO loss 将 DPO 与 SFT 结合：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{RPO}} =
    \lambda_1 \mathcal{L}_{\text{DPO}}(y_w, y_l)
    + \lambda_2 \mathcal{L}_{\text{NLL}}(y_w),\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{L}_{\text{NLL}}(y_w) = -\log \pi_\theta(y_w|q)\)</span> 是作用于被选回答的标准语言建模 loss。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 Iterative RPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"sigmoid"</span><span class="p">,</span>            <span class="c1"># 标准 DPO loss</span>
    <span class="n">rpo_alpha</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>                  <span class="c1"># NLL 正则权重（RPO）</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">何时使用 RPO</p>
<ul>
<li><p>模型必须生成逐步解答的推理任务（数学、代码、逻辑）。</p></li>
<li><p>当 DPO 训练导致模型流畅度或生成质量下降时。</p></li>
<li><p>迭代式流水线：生成 rollout、打标签、用 RPO 训练、再重复。</p></li>
<li><p>NLL 项起到正则化作用，防止 policy 崩塌。</p></li>
</ul>
</div>

### 6.9.8　SimPO——简化版偏好优化 {#sec:simpo}

<div class="admonition tip"><p class="admonition-title">无参考模型的偏好学习</p>
<p>DPO 需要一个 reference 模型来计算隐式 reward，这会让显存翻倍并增加复杂度。SimPO <a class="book-citation" id="cite-meng2024simpo-1" href="../../references.md#ref-meng2024simpo" aria-label="参考文献 252">[252]</a> 取消了 reference 模型，转而用回答的<em>平均对数概率</em>作为隐式 reward，并通过长度归一化项防止模型偏好短回答。</p>
</div>

<p>SimPO 将隐式 reward 定义为：</p>



<div class="arithmatex">\[r_{\text{SimPO}}(y|q) = \frac{\beta}{|y|} \log \pi_\theta(y|q),\]</div>

<p>而 loss 为：</p>



<div class="arithmatex">\[\boxed{
    \mathcal{L}_{\text{SimPO}} = -\mathbb{E}\!\left[
      \log \sigma\!\left(
        \frac{\beta}{|y_w|}\log\pi_\theta(y_w|q)
        - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|q)
        - \gamma
      \right)
    \right],
  }\]</div>

<p>其中 <span class="arithmatex">\(\gamma &gt; 0\)</span> 是目标 reward margin，确保被选回答的 reward 至少比被拒回答高出 <span class="arithmatex">\(\gamma\)</span>。</p>

<div class="admonition note"><p class="admonition-title">SimPO vs DPO vs ORPO</p>
<ul>
<li><p><strong>DPO</strong>：使用 reference 模型；基于比率的隐式 reward。</p></li>
<li><p><strong>ORPO</strong>：无 reference；在 SFT loss 上加入 odds-ratio 项。</p></li>
<li><p><strong>SimPO</strong>：无 reference；长度归一化的对数概率 reward + margin。</p></li>
<li><p>SimPO 比 DPO 简单（无 reference 模型），且比 ORPO 更具理论性。</p></li>
<li><p>SimPO 中的长度归一化至关重要：若没有它，模型会偏好长回答。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的 SimPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"simpo"</span><span class="p">,</span>
    <span class="n">simpo_gamma</span><span class="o">=</span><span class="mf">0.5</span><span class="p">,</span>   <span class="c1"># 目标 reward margin gamma</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">2.5</span><span class="p">,</span>          <span class="c1"># 长度归一化系数</span>
    <span class="c1"># 无需 ref_model！</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>    <span class="c1"># SimPO 是无 reference 的</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>
