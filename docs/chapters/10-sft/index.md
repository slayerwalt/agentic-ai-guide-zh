# 第 10 章 · SFT 最佳实践与技巧 {#sec:sft-best-practices}

<p>监督微调（Supervised Fine-Tuning, SFT）是 RLHF 流水线的基础。SFT 模型的质量决定了 RL 能达到的上限：RL 可以精炼和提升某种行为，但无法可靠地引入 SFT 模型中完全不存在的行为。本章涵盖高效 SFT 的关键技术。</p>

## 10.1　序列打包以提升效率 {#sec:sequence-packing}

<div class="admonition tip"><p class="admonition-title">Padding 问题</p>
<p>标准 SFT 的 Batch 会将所有序列填充（pad）到该 Batch 中最长序列的长度。对长度方差较大的数据集（例如短指令与长文档混合），这会浪费 50–80% 的计算量在 padding Token 上。序列打包（Sequence Packing）消除了这种浪费。</p>
</div>

<p>序列打包将多个短样本拼接为长度为 <code>max_seq_length</code> 的单一序列，以 EOS Token 分隔。Attention mask 确保不同样本的 Token 之间不会相互 attend：</p>

<ol>
<li><p>按长度对样本排序（可选，可提升打包效率）。</p></li>
<li><p>贪心地将样本打包进大小为 <code>max_seq_length</code> 的 bin。</p></li>
<li><p>使用块对角（block-diagonal）Attention mask，防止跨样本 attention。</p></li>
<li><p>只在非 padding Token 上计算 Loss。</p></li>
</ol>

<div class="admonition note"><p class="admonition-title">打包效率</p>
<ul>
<li><p>典型打包效率：85–95%(相比之下，padding 仅 20–50%)。</p></li>
<li><p>加速比：对长度方差较大的数据集可达 2–4<span class="arithmatex">\(\times\)</span>。</p></li>
<li><p>显存：与 padding 相近（每个 Batch 的总 Token 数相同）。</p></li>
<li><p>注意事项：需谨慎设计 attention masking 以避免交叉污染。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的序列打包</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">SFTConfig</span><span class="p">,</span> <span class="n">SFTTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">SFTConfig</span><span class="p">(</span>
    <span class="n">max_seq_length</span><span class="o">=</span><span class="mi">4096</span><span class="p">,</span>
    <span class="n">packing</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>           <span class="c1"># 启用序列打包</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"sft_model"</span><span class="p">,</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">2e-5</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">SFTTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
    <span class="c1"># dataset_text_field="text",  # 或使用 formatting_func</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>
</div>

## 10.2　Chat 模板与格式化 {#sec:chat-templates}

<div class="admonition tip"><p class="admonition-title">为何 Chat 模板重要</p>
<p>语言模型在原始文本上训练，但指令跟随模型需要区分 system prompt、用户消息与 assistant 回应。Chat 模板将这种结构编码进 Token 序列。推理时使用错误模板（或不使用模板）会导致性能显著下降。</p>
</div>

#### ChatML 格式 {#chatml-format}

<p>ChatML 是使用最广泛的 chat 模板：</p>

<div class="highlight"><pre><span></span><code><span class="c1"># ChatML 格式</span>
<span class="n">template</span> <span class="o">=</span> <span class="s2">"""&lt;|im_start|&gt;system</span>
<span class="si">{system_message}</span><span class="s2">&lt;|im_end|&gt;</span>
<span class="s2">&lt;|im_start|&gt;user</span>
<span class="si">{user_message}</span><span class="s2">&lt;|im_end|&gt;</span>
<span class="s2">&lt;|im_start|&gt;assistant</span>
<span class="si">{assistant_message}</span><span class="s2">&lt;|im_end|&gt;"""</span>
</code></pre></div>

#### Llama 格式 {#llama-format}

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

## 10.3　仅对补全部分进行 mask {#sec:completion-masking}

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

## 10.4　多任务 SFT 的数据混合策略 {#sec:data-mixing}

<div class="admonition tip"><p class="admonition-title">多任务挑战</p>
<p>同时在多任务上训练能提升泛化能力，但也会引发<em>任务干扰</em>：不同任务的 Gradient 相互冲突，降低单一任务的性能。数据混合策略用于控制各任务对训练信号的相对贡献。</p>
</div>

#### 比例混合 {#proportional-mixing}

<p>按各数据集的规模成比例地采样：</p>



<div class="arithmatex">\[p_k = \frac{N_k}{\sum_{j=1}^K N_j},\]</div>

<p>其中 <span class="arithmatex">\(N_k\)</span> 是数据集 <span class="arithmatex">\(k\)</span> 中的样本数。这是大多数框架的默认方式，在数据集质量相近时效果良好。</p>

#### 温度混合 {#temperature-mixing}

<p>应用温度 <span class="arithmatex">\(T\)</span> 来平滑比例：</p>



<div class="arithmatex">\[p_k \propto N_k^{1/T}.\]</div>

<p><span class="arithmatex">\(T=1\)</span>：比例混合。<span class="arithmatex">\(T \to \infty\)</span>：均匀混合。<span class="arithmatex">\(T &lt; 1\)</span>：对大数据集过采样。<span class="arithmatex">\(T &gt; 1\)</span>：对小数据集过采样。</p>

#### 基于质量加权的混合 {#quality-weighted-mixing}

<p>按估计的质量（例如在参考模型下的困惑度、人工质量评分）对数据集加权：</p>



<div class="arithmatex">\[p_k \propto N_k \cdot q_k,\]</div>

<p>其中 <span class="arithmatex">\(q_k\)</span> 是数据集 <span class="arithmatex">\(k\)</span> 的质量分数。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的数据混合</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">datasets</span><span class="w"> </span><span class="kn">import</span> <span class="n">concatenate_datasets</span><span class="p">,</span> <span class="n">interleave_datasets</span>

<span class="c1"># 比例混合(默认)</span>
<span class="n">mixed_dataset</span> <span class="o">=</span> <span class="n">concatenate_datasets</span><span class="p">([</span>
    <span class="n">dataset_math</span><span class="p">,</span>
    <span class="n">dataset_code</span><span class="p">,</span>
    <span class="n">dataset_general</span><span class="p">,</span>
<span class="p">])</span><span class="o">.</span><span class="n">shuffle</span><span class="p">(</span><span class="n">seed</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>

<span class="c1"># 温度混合(T=2:对小数据集过采样)</span>
<span class="n">mixed_dataset</span> <span class="o">=</span> <span class="n">interleave_datasets</span><span class="p">(</span>
    <span class="p">[</span><span class="n">dataset_math</span><span class="p">,</span> <span class="n">dataset_code</span><span class="p">,</span> <span class="n">dataset_general</span><span class="p">],</span>
    <span class="n">probabilities</span><span class="o">=</span><span class="p">[</span><span class="mf">0.4</span><span class="p">,</span> <span class="mf">0.4</span><span class="p">,</span> <span class="mf">0.2</span><span class="p">],</span>   <span class="c1"># 经温度缩放后手动设定</span>
    <span class="n">seed</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">SFTConfig</span><span class="p">(</span><span class="n">output_dir</span><span class="o">=</span><span class="s2">"sft_model"</span><span class="p">)</span>
<span class="n">trainer</span> <span class="o">=</span> <span class="n">SFTTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">mixed_dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

## 10.5　当 SFT 反而有害 —— 灾难性遗忘与对齐税 {#sec:sft-pitfalls}

<p>当 LLM 依次经历各训练阶段 —— 预训练 <span class="arithmatex">\(\rightarrow\)</span> 继续预训练 <span class="arithmatex">\(\rightarrow\)</span> SFT <span class="arithmatex">\(\rightarrow\)</span> RLHF/DPO —— 标准基准上的性能下降时常出现。驱动这些回退的是两种<strong>本质上截然不同</strong>的现象，将其混淆将导致错误的缓解策略。</p>

### 10.5.1　灾难性遗忘（结构性擦除） {#catastrophic-forgetting-structural-erasure}

<div class="admonition warning"><p class="admonition-title">灾难性遗忘（Catastrophic Forgetting）</p>
<p>灾难性遗忘是一种<strong>非故意的优化失败</strong>：当一个在分布 <span class="arithmatex">\(\mathcal{D}_A\)</span> 上优化过的网络随后在与之不相交的分布 <span class="arithmatex">\(\mathcal{D}_B\)</span> 上训练时，为 <span class="arithmatex">\(\mathcal{D}_B\)</span> 所需的权重更新<em>物理上覆盖</em>了编码 <span class="arithmatex">\(\mathcal{D}_A\)</span> 的参数结构：</p>



<div class="arithmatex">\[\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_B(\theta_t) \quad \implies \quad \mathcal{L}_A(\theta_{t+1}) \gg \mathcal{L}_A(\theta_t)
\tag{10.1}\]</div>

<p>知识被<strong>摧毁</strong> —— 编码任务 A 的权重不再存在。不重新训练就无法恢复。</p>
</div>

<p><strong>症状</strong>：</p>

<ul>
<li><p>在微调数据之外的任务上完全崩溃（例如模型在 chat 数据上 SFT 后忘记如何做数学）</p></li>
<li><p>语言多样性丧失 —— 模型只能以微调分布的狭窄风格生成</p></li>
<li><p>在微调阶段未被强化的知识上，事实准确性下降</p></li>
<li><p>仅在英语数据上 SFT 后多语言能力下降</p></li>
</ul>

<p><strong>机制根源 —— Fisher 信息视角</strong>：任务 A 的 Fisher 信息矩阵 <span class="arithmatex">\(F\)</span> 标识了哪些参数对 <span class="arithmatex">\(\mathcal{D}_A\)</span> “重要”:</p>



<div class="arithmatex">\[F = \mathbb{E}_{x \sim \mathcal{D}_A}\!\left[\nabla_\theta \log \pi_\theta(x)\, \nabla_\theta \log \pi_\theta(x)^T\right]
\tag{10.2}\]</div>

<p>Fisher 特征值高的参数对任务 A 至关重要。无约束的任务 B 梯度下降完全忽略这些特征值 —— <span class="arithmatex">\(\Delta\theta\)</span> 沿 <span class="arithmatex">\(\nabla\mathcal{L}_B\)</span> 方向移动，而不管是否摧毁了 <span class="arithmatex">\(\mathcal{L}_A\)</span> 的高 Fisher 方向。</p>

### 10.5.2　对齐税（行为约束） {#alignment-tax-behavioral-constraint}

<p>对齐税（Alignment Tax）是<strong>一种有意为之、可预期的权衡</strong>：模型的原始能力（无约束生成、最大化推理带宽）下降，是因为 Policy 被约束去生成安全、格式良好、与偏好对齐的输出。</p>

<p><strong>机制</strong>：在 DPO/PPO 期间，Policy <span class="arithmatex">\(\pi_\theta\)</span> 通过 KL 散度被惩罚以防偏离参考 <span class="arithmatex">\(\pi_{\text{ref}}\)</span>:</p>



<div class="arithmatex">\[r_{\text{implicit}}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
\tag{10.3}\]</div>

<p>这条“链条”约束着模型的<strong>输出分布</strong> —— 它无法探索偏离参考太远的高方差推理路径。知识并未被<strong>擦除</strong>；而是被<em>抑制</em>了。模型仍“知道”答案，但其分布被压平，偏向安全、通用的回应。</p>

<p><strong>症状</strong>：</p>

<ul>
<li><p>过度拒答（对良性查询回复“I can’t help with that”）</p></li>
<li><p>风格僵硬 —— 模糊措辞、过多警示语、冗长的安全免责声明</p></li>
<li><p>在原始能力基准（MMLU、HumanEval）上分数下降，而在偏好基准（MT-Bench、AlpacaEval）上提升</p></li>
<li><p>生成复杂、高熵输出（创意写作、新颖算法）的能力下降</p></li>
</ul>

### 10.5.3　对比分类 {#comparative-taxonomy}

<p class="table-caption" id="table-10-1"><span class="table-number">表 10.1　</span>灾难性遗忘 vs 对齐税 —— 完整对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>灾难性遗忘</strong></th><th scope="col"><strong>对齐税</strong></th></tr></thead>
<tbody><tr><td><strong>意图性</strong></td><td>非故意（优化产物）</td><td>可预期权衡（为安全/有用而有意承担）</td></tr>
<tr><td><strong>参数状态</strong></td><td>先验知识被物理覆盖</td><td>潜在分布被约束/截断</td></tr>
<tr><td><strong>信息</strong></td><td><strong>被摧毁</strong>：权重不再编码该能力</td><td><strong>被抑制</strong>：知识仍存在但更难触发</td></tr>
<tr><td><strong>主导阶段</strong></td><td>顺序 SFT、领域继续预训练</td><td>偏好优化（PPO、DPO、KTO、RLHF）</td></tr>
<tr><td><strong>主要症状</strong></td><td>基线能力完全崩溃</td><td>过度拒答、风格僵硬、原始基准分数下降</td></tr>
<tr><td><strong>可逆性</strong></td><td>不重新从 checkpoint 训练则不可逆</td><td>部分可逆：调整 <span class="arithmatex">\(\beta\)</span>、system prompt 或微调</td></tr>
<tr><td><strong>检测</strong></td><td>预训练评估集上的困惑度飙升</td><td>困惑度稳定但能力基准上的胜率下降</td></tr>
<tr><td><strong>随模型规模</strong></td><td>各尺度相近</td><td>小模型支付更大的对齐税</td></tr></tbody>
</table>
</div>

### 10.5.4　缓解策略 {#mitigation-strategies}

<p><strong>针对灾难性遗忘</strong>：</p>

<ol>
<li><p><strong>数据回放（Data replay）</strong>：将 5–10% 的预训练数据混入 SFT 数据集。确保 Gradient 更新不会完全忽视预训练分布。</p></li>
<li><p><strong>弹性权重整合（Elastic Weight Consolidation, EWC）</strong> <a class="book-citation" id="cite-kirkpatrick2017overcoming-1" href="../../references.md#ref-kirkpatrick2017overcoming" aria-label="参考文献 186">[186]</a>：添加正则项 <span class="arithmatex">\(\Omega(\theta) = \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^*)^2\)</span>，惩罚对原任务而言 Fisher 信息高的参数的变化。</p></li>
<li><p><strong>LoRA / 参数高效微调</strong>：只训练低秩 adapter（参数量 <span class="arithmatex">\(&lt;1\%\)</span>），基础权重完全冻结。这避免了预训练知识的<em>永久性摧毁</em> —— 你随时可以移除 adapter 来恢复原模型。然而，<strong>当 adapter 处于激活状态时</strong>，组合系统 <span class="arithmatex">\((W_0 + BA)\)</span> 仍可能表现出遗忘：adapter 可能将模型的有效行为推离旧技能。LoRA 保护的是 checkpoint，而非激活时的推理行为。</p></li>
<li><p><strong>保守的学习率</strong>：使用 <span class="arithmatex">\(1\)</span>–<span class="arithmatex">\(5 \times 10^{-6}\)</span> 配合较少的 Epoch(1–3)。更大的学习率会加速遗忘。</p></li>
<li><p><strong>渐进式训练</strong>：逐步混合分布，随时间增加 SFT 数据比例，而非突然切换。</p></li>
</ol>

<p><strong>针对对齐税</strong>：</p>

<ol>
<li><p><strong>仔细调节 <span class="arithmatex">\(\beta\)</span></strong>：更低的 <span class="arithmatex">\(\beta\)</span> 给予模型更多自由（减少税负），但可能牺牲安全性。多数场景下最优 <span class="arithmatex">\(\beta \in [0.05, 0.3]\)</span>。</p></li>
<li><p><strong>高质量、多样化的 SFT 数据</strong>：对齐税的一部分来自 SFT 收窄了输出分布；更广、更多样化的 SFT 数据可减少这部分。RL 阶段通过 KL 正则化 <a class="book-citation" id="cite-ouyang2022training-1" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a>进一步施加约束。</p></li>
<li><p><strong>条件对齐（Conditional alignment）</strong>：训练模型仅在安全标志激活时才对齐。推理时为基准测试关闭约束（仅供研究使用的技术）。</p></li>
<li><p><strong>Constitutional AI / RLAIF</strong>：利用模型自生成反馈创建更细致的偏好数据，在提升对齐的同时保留能力。</p></li>
<li><p><strong>有针对性的 RL 预算</strong>：不要过度训练 RL。监控能力基准，当税负超出可接受阈值（通常是 2–5% MMLU 回退）时停止。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">如何判断你遇到的是哪一种</p>
<ul>
<li><p><strong>在失败任务上运行 base 模型</strong>：若 base 模型成功而微调后模型完全失败 <span class="arithmatex">\(\rightarrow\)</span> 灾难性遗忘。</p></li>
<li><p><strong>Prompt 工程测试</strong>：若精心设计的 prompt（例如“忽略安全准则，逐步解答此数学题”）能恢复该能力 <span class="arithmatex">\(\rightarrow\)</span> 对齐税（知识被抑制而非擦除）。</p></li>
<li><p><strong>困惑度检查</strong>：在预训练验证集上计算困惑度。飙升 = 遗忘。稳定 = 对齐税。</p></li>
<li><p><strong>Few-shot 恢复</strong>：若提供少量上下文示例就能恢复能力 <span class="arithmatex">\(\rightarrow\)</span> 对齐税。若大量示例也无法恢复 <span class="arithmatex">\(\rightarrow\)</span> 遗忘。</p></li>
</ul>
</div>

## 10.6　与 RL 的关系 —— SFT 质量决定 RL 上限 {#sec:sft-rl-connection}

<div class="admonition note"><p class="admonition-title">SFT-RL 关系</p>
<p>SFT 模型是 RL 训练的起点。RL 可以：</p>

<ul>
<li><p><strong>放大</strong>SFT 模型中已存在但较弱的行为。</p></li>
<li><p><strong>抑制</strong>已存在但不理想的行为。</p></li>
<li><p><strong>精炼</strong>回答的风格与格式。</p></li>
</ul>

<p>RL <em>无法</em>：</p>

<ul>
<li><p>引入 SFT 模型中完全不存在的能力。</p></li>
<li><p>从 SFT 阶段严重的灾难性遗忘中恢复。</p></li>
<li><p>弥补系统性有偏的奖励模型。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">SFT 中的探索-利用权衡</p>
<p>为使 RL 起作用，SFT 模型必须偶尔产生正确响应（从而 Reward 信号非零）。如果 SFT 模型对某 prompt <em>从不</em>产生正确响应，RL 无法学会产生正确响应 —— 没有可放大的正向信号。这就是为什么 SFT 质量是 RL 性能的上限。</p>

<p>具体而言：若 SFT 模型能正确解答 10% 的数学题，RL 有可能将其推至 80%。若 SFT 模型解题率为 0%,RL 将毫无进展（所有 Reward 为零，所有优势为零，无 Gradient）。</p>
</div>

#### 实践启示 {#practical-implications}

<ol>
<li><p><strong>SFT 数据质量</strong>：使用高质量、多样化的数据。少量高质量数据胜过大量低质量数据。</p></li>
<li><p><strong>SFT 数据覆盖度</strong>：确保 SFT 数据覆盖你想通过 RL 改进的任务。如果某任务不在 SFT 数据中，RL 将难以推进。</p></li>
<li><p><strong>SFT 训练时长</strong>：不要过度训练 SFT 模型。过度训练降低多样性，让 RL 探索更困难。</p></li>
<li><p><strong>Warm-up</strong>：即使 base 模型已经过指令微调，在 RL 之前也可考虑在特定任务数据上进行简短的 SFT warm-up。</p></li>
</ol>

<div class="admonition example"><p class="admonition-title">在 RL 之前检查 SFT 质量</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">tqdm</span><span class="w"> </span><span class="kn">import</span> <span class="n">tqdm</span>

<span class="k">def</span><span class="w"> </span><span class="nf">estimate_pass_at_k</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">tokenizer</span><span class="p">,</span> <span class="n">dataset</span><span class="p">,</span> <span class="n">k</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span> <span class="n">n_samples</span><span class="o">=</span><span class="mi">100</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    估计 SFT 模型的 pass@k。</span>
<span class="sd">    若 pass@1 &lt; 5%,RL 大概率会失败。</span>
<span class="sd">    若 pass@k &lt; 20%,RL 会很吃力。</span>
<span class="sd">    """</span>
    <span class="n">pass_at_1_scores</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">pass_at_k_scores</span> <span class="o">=</span> <span class="p">[]</span>

    <span class="k">for</span> <span class="n">example</span> <span class="ow">in</span> <span class="n">tqdm</span><span class="p">(</span><span class="n">dataset</span><span class="o">.</span><span class="n">select</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">n_samples</span><span class="p">))):</span>
        <span class="n">prompt</span> <span class="o">=</span> <span class="n">example</span><span class="p">[</span><span class="s2">"prompt"</span><span class="p">]</span>
        <span class="n">ground_truth</span> <span class="o">=</span> <span class="n">example</span><span class="p">[</span><span class="s2">"answer"</span><span class="p">]</span>

        <span class="c1"># 采样 k 个补全</span>
        <span class="n">inputs</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="p">(</span><span class="n">prompt</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span><span class="p">)</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">device</span><span class="p">)</span>
        <span class="n">outputs</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">generate</span><span class="p">(</span>
            <span class="o">**</span><span class="n">inputs</span><span class="p">,</span>
            <span class="n">max_new_tokens</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span>
            <span class="n">do_sample</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
            <span class="n">temperature</span><span class="o">=</span><span class="mf">0.8</span><span class="p">,</span>
            <span class="n">num_return_sequences</span><span class="o">=</span><span class="n">k</span><span class="p">,</span>
        <span class="p">)</span>

        <span class="n">correct</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="k">for</span> <span class="n">output</span> <span class="ow">in</span> <span class="n">outputs</span><span class="p">:</span>
            <span class="n">response</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">output</span><span class="p">,</span> <span class="n">skip_special_tokens</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
            <span class="k">if</span> <span class="n">ground_truth</span> <span class="ow">in</span> <span class="n">response</span><span class="p">:</span>
                <span class="n">correct</span> <span class="o">+=</span> <span class="mi">1</span>

        <span class="c1"># pass@1:正确样本占比(估计的成功率)</span>
        <span class="n">pass_at_1_scores</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">correct</span> <span class="o">/</span> <span class="n">k</span><span class="p">)</span>
        <span class="c1"># pass@k:k 个样本中至少一个正确</span>
        <span class="n">pass_at_k_scores</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">correct</span> <span class="o">&gt;=</span> <span class="mi">1</span><span class="p">)</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Pass@1 (estimated): </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">pass_at_1_scores</span><span class="p">)</span><span class="si">:</span><span class="s2">.2%</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Pass@</span><span class="si">{</span><span class="n">k</span><span class="si">}</span><span class="s2">: </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">pass_at_k_scores</span><span class="p">)</span><span class="si">:</span><span class="s2">.2%</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"RL viability: </span><span class="si">{</span><span class="s1">'Good'</span><span class="w"> </span><span class="k">if</span><span class="w"> </span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">pass_at_1_scores</span><span class="p">)</span><span class="w"> </span><span class="o">&gt;</span><span class="w"> </span><span class="mf">0.05</span><span class="w"> </span><span class="k">else</span><span class="w"> </span><span class="s1">'Poor'</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

<span class="n">estimate_pass_at_k</span><span class="p">(</span><span class="n">sft_model</span><span class="p">,</span> <span class="n">tokenizer</span><span class="p">,</span> <span class="n">eval_dataset</span><span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">SFT 最佳实践总结</p>
<ol>
<li><p>使用序列打包（Sequence Packing）以最大化 GPU 利用率。</p></li>
<li><p>使用 completion-only masking 将 Gradient 聚焦于 assistant 响应。</p></li>
<li><p>为你的模型系列使用正确的 chat 模板。</p></li>
<li><p>多任务 SFT 时使用温度缩放（<span class="arithmatex">\(T \approx 2\)</span>）按比例混合数据。</p></li>
<li><p>使用 LoRA 防止灾难性遗忘。</p></li>
<li><p>在开始 RL 前评估 pass@k，以确保 SFT 模型是可行的起点。</p></li>
<li><p>不要过度训练：指令微调通常 1–3 个 Epoch 就足够。</p></li>
<li><p>监控多样性指标（熵、n-gram 多样性）以检测 mode collapse。</p></li>
</ol>
</div>
