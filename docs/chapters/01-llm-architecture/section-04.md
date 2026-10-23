# 1.4　预测头：Transformer 的输出 {#prediction-heads-what-transformers-output}

<p>Transformer 主干网络为每个位置产生上下文隐藏状态 <span class="arithmatex">\(\mathbf{h}_t \in \mathbb{R}^d\)</span>。我们<em>如何处理</em>这些隐藏状态——即<strong>预测头（Prediction Head）</strong>——定义了任务本身。同一个 Transformer 主干网络只需更换预测头，就能服务于完全不同的目的。</p>

<figure id="fig:prediction-heads"><a href="../../assets/figures/fig_008_prediction-heads.png" target="_blank" rel="noopener" aria-label="查看图 1.7 原图" title="点击查看原图"><img src="../../assets/figures/fig_008_prediction-heads.png" alt="同一个 Transformer 主干网络通过更换预测头即可支持不同任务。本文使用的全部三种预测头在最终投影层之下具有完全相同的架构。" loading="lazy"></a><figcaption><span class="figure-number">图 1.7　</span>同一个 Transformer 主干网络通过更换预测头即可支持不同任务。本文使用的全部三种预测头在最终投影层之下具有完全相同的架构。</figcaption></figure>

## 1.4.1　语言建模头（预训练阶段） {#language-modeling-head-pretraining}

<p>标准的语言建模头（Language Modeling Head, LM Head）将最终隐藏状态投影到词表 logits，并以下一个 token 的交叉熵损失进行训练：</p>



<div class="arithmatex">\[P(x_{t+1} | x_{\leq t}) = \text{softmax}(\mathbf{W}_{\text{head}} \cdot \mathbf{h}_t + \mathbf{b})
\tag{1.7}\]</div>

<p>其中 <span class="arithmatex">\(\mathbf{W}_{\text{head}} \in \mathbb{R}^{|\mathcal{V}| \times d}\)</span>（通常与 embedding 矩阵权重共享：<span class="arithmatex">\(\mathbf{W}_{\text{head}} = \mathbf{E}^T\)</span>）。</p>

<div class="admonition note"><p class="admonition-title">LM 头的特性</p>
<ul>
<li><p><strong>训练目标</strong>：因果语言建模（对每个位置预测下一个 token）</p></li>
<li><p><strong>损失</strong>：<span class="arithmatex">\(\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{&lt;t})\)</span></p></li>
<li><p><strong>标签</strong>：每个 token 同时作为输入（右移一位）和目标（左移一位）</p></li>
<li><p><strong>使用阶段</strong>：在大规模语料上的预训练（万亿级 token）</p></li>
<li><p><strong>关键洞察</strong>：模型在下一 token 预测的副产品中学到通用语言理解能力</p></li>
</ul>
</div>

## 1.4.2　条件生成头（SFT / 指令跟随） {#conditional-generation-head-sft-instruction-following}

<p>对于监督微调（Supervised Fine-Tuning, SFT），其架构与 LM 头<em>完全相同</em>——都是将隐藏状态线性投影到词表 logits。区别仅在于<em>在哪些 token 上计算损失</em>：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|}\sum_{t=1}^{|y|} \log P(y_t | x_{\text{prompt}}, y_{&lt;t})
\tag{1.8}\]</div>

<div class="admonition note"><p class="admonition-title">条件生成头 – 与 LM 头的关键区别</p>
<ul>
<li><p><strong>损失掩码</strong>：只在<em>回复</em>的 token 上计算损失，不计算 prompt/指令部分。Prompt 仅提供上下文，不提供梯度信号。</p></li>
<li><p><strong>条件化</strong>：模型学会在特定指令格式（system prompt、用户查询、工具调用）的<em>条件下</em>生成回复。</p></li>
<li><p><strong>格式 token</strong>：特殊 token（<code>&lt;|user|&gt;</code>、<code>&lt;|assistant|&gt;</code>）引导模型产生结构化输出。</p></li>
<li><p><strong>使用阶段</strong>：在精心策划的指令-回复对上进行 SFT；也在强化学习的策略生成阶段使用（即产出动作/回复的策略头）。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">同一个头 – 不同的训练信号</p>
<p>LM 头和 SFT 头在架构上完全相同（同一个 <span class="arithmatex">\(\mathbf{W}_{\text{head}}\)</span>）。唯一的区别是 SFT 阶段会掩码掉 prompt token 的损失。这个细微的改动就把一个通用文本预测器转变为一个指令跟随助手。预测头学会根据上下文条件「激活」不同的生成模式。</p>
</div>

## 1.4.3　价值头（用于 RL 的回归头） {#value-head-regression-for-rl}

<p>在强化学习（PPO、GRPO）中，我们需要估计某个状态<em>有多好</em>——这需要一个标量输出，而不是词表 logits。<strong>价值头（Value Head）</strong>用一个简单的回归层替换了 LM 投影：</p>



<div class="arithmatex">\[V(s_t) = \mathbf{w}_{\text{value}}^T \cdot \mathbf{h}_t + b \in \mathbb{R}
\tag{1.9}\]</div>

<p>其中 <span class="arithmatex">\(\mathbf{w}_{\text{value}} \in \mathbb{R}^d\)</span>，<span class="arithmatex">\(b \in \mathbb{R}\)</span>。</p>

<div class="admonition note"><p class="admonition-title">价值头的特性</p>
<ul>
<li><p><strong>输出</strong>：单个标量（从该状态出发的期望累计奖励）</p></li>
<li><p><strong>损失</strong>：预测回报与真实回报之间的均方误差：<span class="arithmatex">\(\mathcal{L}_V = \frac{1}{T}\sum_t (V(s_t) - R_t)^2\)</span></p></li>
<li><p><strong>架构</strong>：线性层 <span class="arithmatex">\(\mathbb{R}^d \to \mathbb{R}^1\)</span>（有时会用一个小型 MLP：<span class="arithmatex">\(d \to 256 \to 1\)</span>）</p></li>
<li><p><strong>主干共享</strong>：通常与策略共享 Transformer 主干（但有独立的价值头），也可能使用完全独立的 critic 网络</p></li>
<li><p><strong>使用阶段</strong>：PPO 优势估计（GAE）、奖励模型评分</p></li>
</ul>
</div>

## 1.4.4　预测头选择总览 {#head-selection-summary}

<p class="table-caption" id="table-1-9"><span class="table-number">表 1.9　</span>本文使用的各类预测头及其训练场景。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>预测头</strong></th><th scope="col"><strong>输出</strong></th><th scope="col"><strong>损失</strong></th><th scope="col"><strong>阶段</strong></th><th scope="col"><strong>用途</strong></th></tr></thead>
<tbody><tr><td>LM Head</td><td><span class="arithmatex">\(\mathbb{R}^{|\mathcal{V}|}\)</span></td><td>交叉熵（所有 token）</td><td>预训练</td><td>从原始文本学习语言</td></tr>
<tr><td>Conditional Head</td><td><span class="arithmatex">\(\mathbb{R}^{|\mathcal{V}|}\)</span></td><td>交叉熵（仅回复部分）</td><td>SFT</td><td>学习跟随指令</td></tr>
<tr><td>Value Head</td><td><span class="arithmatex">\(\mathbb{R}^1\)</span></td><td>MSE</td><td>RL (PPO)</td><td>估计状态价值以计算优势</td></tr>
<tr><td>Reward Head</td><td><span class="arithmatex">\(\mathbb{R}^1\)</span></td><td>成对排序</td><td>奖励模型训练</td><td>给回复质量打分</td></tr></tbody>
</table>
</div>

<div class="admonition warning"><p class="admonition-title">预测头初始化很重要</p>
<p>为预训练 LM 添加价值头时，应将其初始化为接近零的小随机权重。若初始化值过大，初始价值估计会严重偏离，导致优势值过大、PPO 更新不稳定。常见做法：将最后一层线性层用 <span class="arithmatex">\(\mathcal{N}(0, 1/\sqrt{d})\)</span> 初始化，或直接置零。</p>
</div>

## 1.4.5　HuggingFace 实现 {#huggingface-implementation}

<p class="code-caption"><span class="code-number">代码 1.3　</span>使用 HuggingFace 加载与使用不同的预测头。</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">AutoModelForCausalLM</span><span class="p">,</span>          <span class="c1"># LM 头（预训练 + SFT）</span>
    <span class="n">AutoModelForSequenceClassification</span><span class="p">,</span>  <span class="c1"># 奖励头</span>
    <span class="n">AutoTokenizer</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoModelForCausalLMWithValueHead</span>  <span class="c1"># 价值头（PPO）</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>

<span class="n">model_name</span> <span class="o">=</span> <span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span>
<span class="n">tokenizer</span> <span class="o">=</span> <span class="n">AutoTokenizer</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="n">model_name</span><span class="p">)</span>

<span class="c1"># === 1. LM 头（预训练 / SFT）===</span>
<span class="c1"># 默认的 CausalLM 模型——将隐藏状态投影到词表 logits</span>
<span class="n">lm_model</span> <span class="o">=</span> <span class="n">AutoModelForCausalLM</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span>
    <span class="n">model_name</span><span class="p">,</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span>
    <span class="n">device_map</span><span class="o">=</span><span class="s2">"auto"</span><span class="p">,</span>
<span class="p">)</span>
<span class="c1"># lm_model.lm_head: Linear(hidden_size -&gt; vocab_size)</span>
<span class="c1"># 输出：形状为 (batch, seq_len, vocab_size) 的 logits</span>

<span class="n">inputs</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="p">(</span><span class="s2">"The capital of France is"</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span><span class="p">)</span>
<span class="n">outputs</span> <span class="o">=</span> <span class="n">lm_model</span><span class="p">(</span><span class="o">**</span><span class="n">inputs</span><span class="p">)</span>
<span class="n">next_token_logits</span> <span class="o">=</span> <span class="n">outputs</span><span class="o">.</span><span class="n">logits</span><span class="p">[:,</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="p">:]</span>  <span class="c1"># (batch, vocab_size)</span>
<span class="n">probs</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">softmax</span><span class="p">(</span><span class="n">next_token_logits</span><span class="p">,</span> <span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>

<span class="c1"># === 2. 条件生成头（SFT）===</span>
<span class="c1"># 架构上与 LM 头完全相同——区别在于损失掩码</span>
<span class="c1"># SFT 时只在回复 token 上计算损失：</span>
<span class="n">messages</span> <span class="o">=</span> <span class="p">[</span>
    <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"What is 2+2?"</span><span class="p">},</span>
    <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"assistant"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"4"</span><span class="p">},</span>
<span class="p">]</span>
<span class="n">formatted</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="o">.</span><span class="n">apply_chat_template</span><span class="p">(</span><span class="n">messages</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span><span class="p">)</span>
<span class="n">labels</span> <span class="o">=</span> <span class="n">formatted</span><span class="o">.</span><span class="n">clone</span><span class="p">()</span>
<span class="c1"># 掩码掉 prompt token（设为 -100，交叉熵会忽略它们）</span>
<span class="n">prompt_len</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">tokenizer</span><span class="o">.</span><span class="n">apply_chat_template</span><span class="p">(</span><span class="n">messages</span><span class="p">[:</span><span class="mi">1</span><span class="p">]))</span>
<span class="n">labels</span><span class="p">[:,</span> <span class="p">:</span><span class="n">prompt_len</span><span class="p">]</span> <span class="o">=</span> <span class="o">-</span><span class="mi">100</span>
<span class="n">loss</span> <span class="o">=</span> <span class="n">lm_model</span><span class="p">(</span><span class="n">input_ids</span><span class="o">=</span><span class="n">formatted</span><span class="p">,</span> <span class="n">labels</span><span class="o">=</span><span class="n">labels</span><span class="p">)</span><span class="o">.</span><span class="n">loss</span>

<span class="c1"># === 3. 价值头（PPO Critic）===</span>
<span class="c1"># 在 LM 主干之上加一个 Linear(hidden_size -&gt; 1)</span>
<span class="n">value_model</span> <span class="o">=</span> <span class="n">AutoModelForCausalLMWithValueHead</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span>
    <span class="n">model_name</span><span class="p">,</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span>
    <span class="n">device_map</span><span class="o">=</span><span class="s2">"auto"</span><span class="p">,</span>
<span class="p">)</span>
<span class="c1"># value_model.v_head: Linear(hidden_size -&gt; 1)</span>
<span class="c1"># 同时返回 LM logits 和每个 token 的价值估计</span>

<span class="n">inputs</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="p">(</span><span class="s2">"Explain quantum computing"</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span><span class="p">)</span>
<span class="n">lm_logits</span><span class="p">,</span> <span class="n">loss</span><span class="p">,</span> <span class="n">values</span> <span class="o">=</span> <span class="n">value_model</span><span class="p">(</span>
    <span class="o">**</span><span class="n">inputs</span><span class="p">,</span> <span class="n">return_dict</span><span class="o">=</span><span class="kc">False</span>
<span class="p">)</span>
<span class="c1"># values 形状：(batch, seq_len, 1)——每个 token 一个标量估计</span>

<span class="c1"># === 4. 奖励头（奖励模型）===</span>
<span class="c1"># 分类头：在最后一个 token 上接 Linear(hidden_size -&gt; 1)</span>
<span class="n">reward_model</span> <span class="o">=</span> <span class="n">AutoModelForSequenceClassification</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span>
    <span class="n">model_name</span><span class="p">,</span>
    <span class="n">num_labels</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>              <span class="c1"># 单标量输出</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span>
    <span class="n">device_map</span><span class="o">=</span><span class="s2">"auto"</span><span class="p">,</span>
<span class="p">)</span>
<span class="c1"># 通过池化最后一个 token 的隐藏状态对整个序列打分</span>
<span class="n">inputs</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="p">(</span><span class="s2">"Good response here"</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span><span class="p">)</span>
<span class="n">reward_score</span> <span class="o">=</span> <span class="n">reward_model</span><span class="p">(</span><span class="o">**</span><span class="n">inputs</span><span class="p">)</span><span class="o">.</span><span class="n">logits</span>  <span class="c1"># 形状：(batch, 1)</span>
</code></pre></div>

<div class="admonition tip"><p class="admonition-title">权重共享：LM 头 = embedding 矩阵的转置</p>
<p>大多数现代 LLM 会将 LM 头的权重与输入 embedding 矩阵<em>共享（tie）</em>：<code>lm_head.weight = model.embed_tokens.weight</code>。这意味着 LM 头<em>并非</em>独立学习的一层——它直接复用了 embedding 表。好处包括：参数更少（节省 <span class="arithmatex">\(|\mathcal{V}| \times d\)</span>）、泛化更好，并且 embedding 空间的几何结构直接决定了 token 概率。你可以在 HuggingFace 中验证这一点：对大多数模型而言，<code>model.lm_head.weight is model.model.embed_tokens.weight</code> 会返回 <code>True</code>。</p>
</div>
