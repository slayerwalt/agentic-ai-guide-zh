# 第 7 章 · GRPO —— 组相对策略优化 {#grpo-group-relative-policy-optimization}

<p>组相对策略优化（Group Relative Policy Optimization, GRPO） <a class="book-citation" id="cite-shao2024deepseekmath-1" href="../../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a> 是一种专为语言模型设计的强化学习算法，它消除了对独立 value network（critic）的需求。GRPO 由 DeepSeek 在 DeepSeekMath 工作中提出，随后在 DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 中被扩展到更大规模，已经迅速成为 LLM 训练中的主流 RL 方法——被大多数开源对齐框架（TRL、OpenRLHF、veRL）作为默认算法采用。</p>

<p>其核心思想看似简单却出人意料地有效：与其训练一个神经网络去预测期望 reward（即 PPO 中的 critic），GRPO 通过对同一 prompt 生成多条响应、利用该组的 reward 统计量作为 baseline，从经验上 <em>估计</em> 这一基线。这从显存中去掉了一整个模型，将工程复杂度减半，而且——令人意外的是——往往优于 PPO，因为经验基线比训练不充分的 value function 更准确。</p>

<p>GRPO 在下列场景中尤其有效：</p>

<ul>
<li><p><strong>推理任务</strong>：具有可验证 reward（数学、代码），二值正确性提供干净信号。</p></li>
<li><p><strong>大模型</strong>（70B 及以上）：去掉 critic 带来的显存节省至关重要。</p></li>
<li><p><strong>多轮对话（Multi-Turn）与 Agent 场景</strong>：跨工具调用（Tool Calling）的价值估计本就难以处理。</p></li>
</ul>

<p>本章涵盖 GRPO 的动机、算法、关键变体（Dr. GRPO、DAPO、2-GRPO、GDPO）以及基于 TRL 的实战实现。</p>

## 7.1　动机 {#motivation-dup2}

<p>PPO 的 value model（critic）在语言任务上存在三大问题：</p>

<ol>
<li><p><strong>显存</strong>：value head 与 policy 共享主干（70B 模型占 140GB）。若分离则显存翻倍。</p></li>
<li><p><strong>精度</strong>：对部分序列预测期望 reward 极其困难。value function 经常出错 <span class="arithmatex">\(\rightarrow\)</span> advantage 出错 <span class="arithmatex">\(\rightarrow\)</span> gradient 方向出错。</p></li>
<li><p><strong>训练</strong>：value head 需要大量样本才能收敛。RL 早期它给出嘈杂的预测，破坏 policy 学习的稳定性。</p></li>
</ol>

<p><strong>GRPO 的关键洞见</strong> <a class="book-citation" id="cite-shao2024deepseekmath-1--section-01" href="../../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a>：与其学习 <span class="arithmatex">\(V(s)\)</span>，不如从一组样本中经验地 <em>估计</em> 它。对同一 prompt 生成 <span class="arithmatex">\(G\)</span> 条响应，计算它们的 reward，并用组统计量作为 baseline。</p>

## 7.2　算法 {#algorithm}

<ol>
<li><p>对每个 prompt <span class="arithmatex">\(x\)</span>，采样 <span class="arithmatex">\(G\)</span> 条 completion：<span class="arithmatex">\(\{y_1, \ldots, y_G\} \sim \pi_\theta(\cdot|x)\)</span></p></li>
<li><p>对每条打分：<span class="arithmatex">\(r_i = R(x, y_i)\)</span></p></li>
<li><p>组内归一化：<span class="arithmatex">\(\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}\)</span>，其中 <span class="arithmatex">\(\mu_G = \frac{1}{G}\sum_j r_j\)</span>，<span class="arithmatex">\(\sigma_G = \text{std}(\{r_j\})\)</span></p></li>
<li><p>用这些 advantage 应用 PPO 风格的 clipped 更新</p></li>
</ol>



<div class="arithmatex">\[\boxed{\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}, \qquad L = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_i,\; \text{clip}(r_t(\theta), 1{\pm}\epsilon)\hat{A}_i\right)\right] - \beta D_\text{KL}[\pi_\theta\|\pi_\text{ref}]}
\tag{7.1}\]</div>

<div class="admonition tip"><p class="admonition-title">为什么组内归一化有效</p>
<p><strong>组均值近似 <span class="arithmatex">\(V(s)\)</span></strong>：若对同一 prompt 采样足够多的响应，其平均 reward 就是期望 reward 的蒙特卡洛估计，即 value function。</p>

<p><strong>高于均值 = 好动作</strong>：<span class="arithmatex">\(\hat{A}_i &gt; 0\)</span> 表示该响应对该 prompt 比平均更好，应强化。</p>

<p><strong>低于均值 = 坏动作</strong>：<span class="arithmatex">\(\hat{A}_i &lt; 0\)</span> 表示比平均更差，应抑制。</p>

<p><strong>归一化</strong>：除以 <span class="arithmatex">\(\sigma_G\)</span> 确保 advantage 在不同 reward 量级的 prompt 之间具有尺度不变性。</p>

<p><strong>DeepSeek-R1 突破</strong> <a class="book-citation" id="cite-deepseek2025r1-1--section-02" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a>：纯 GRPO 配合二值正确性 reward（答对 <span class="arithmatex">\(r=1\)</span>，答错 <span class="arithmatex">\(r=0\)</span>）在数学/代码上训练时，模型自发涌现出思维链（Chain-of-Thought，CoT）推理、自我验证和错误纠正——完全没有被显式指示这么做。</p>
</div>

<figure id="figure-7-1"><a href="../../assets/figures/fig_029_fig29.png" target="_blank" rel="noopener" aria-label="查看图 7.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_029_fig29.png" alt="GRPO 实战：对一个数学 prompt 采样 \(G{=}5\) 条响应。三条正确（\(r{=}1\)），两条错误（\(r{=}0\)）。组均值 \(\mu_G{=}0.6\) 充当 baseline；正确响应获得正 advantage（强化），错误响应获得负 advantage（抑制）。" loading="lazy"></a><figcaption><span class="figure-number">图 7.1　</span>GRPO 实战：对一个数学 prompt 采样 <span class="arithmatex">\(G{=}5\)</span> 条响应。三条正确（<span class="arithmatex">\(r{=}1\)</span>），两条错误（<span class="arithmatex">\(r{=}0\)</span>）。组均值 <span class="arithmatex">\(\mu_G{=}0.6\)</span> 充当 baseline；正确响应获得正 advantage（强化），错误响应获得负 advantage（抑制）。</figcaption></figure>

## 7.3　TRL 实现 {#trl-implementation-dup3}

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

## 7.4　组大小分析 {#group-size-analysis}

<span id="table-7-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><span class="arithmatex">\(G\)</span></th><th scope="col"><strong>信号质量</strong></th><th scope="col"><strong>算力</strong></th><th scope="col"><strong>使用场景</strong></th></tr></thead>
<tbody><tr><td>2</td><td>极嘈杂（掷硬币）</td><td>低</td><td>不推荐 —— 太嘈杂，无法稳定学习</td></tr>
<tr><td>4</td><td>中等</td><td>中等</td><td>快速实验、简单任务（通过率 <span class="arithmatex">\(&gt;\)</span> 50%）</td></tr>
<tr><td>8</td><td>良好（标准）</td><td>高</td><td>默认。适合大多数任务的良好平衡</td></tr>
<tr><td>16</td><td>极佳</td><td>非常高</td><td>困难任务（通过率 <span class="arithmatex">\(&lt;\)</span> 20%），需要多次尝试才能拿到正样本</td></tr>
<tr><td>32</td><td>近乎完美</td><td>极高</td><td>仅在算力充裕且任务非常困难时使用</td></tr></tbody>
</table>
</div>

<div class="admonition note"><p class="admonition-title">关键：组内必须同时包含成功与失败</p>
<p>若 <span class="arithmatex">\(G\)</span> 条响应全对（<span class="arithmatex">\(r_i = 1 \;\forall i\)</span>）：所有 advantage = 0，无学习信号！<br>若全错：同样的问题。Prompt 难度必须与模型能力匹配。<br><strong>Goldilocks 原则</strong>：将 prompt 过滤至当前模型 20–80% 通过率。随着模型改进，每 500 步重新筛选。</p>
</div>

## 7.5　GRPO 变体与扩展 {#grpo-variants-and-extensions}

### 7.5.1　GRPO 组中的多样性 {#diversity-in-grpo-groups}

<div class="admonition tip"><p class="admonition-title">RL 训练中的模式坍塌</p>
<p>若没有多样性压力，RL 训练的 LLM 会坍塌到狭窄的高 reward 响应集合上：</p>

<ul>
<li><p>模型对每种题型学到一个“模板”答案</p></li>
<li><p>熵急剧下降；模型变得确定性</p></li>
<li><p>Reward hacking 变得更容易（狭窄输出更容易被利用）</p></li>
<li><p>泛化能力受损：模型记住的是 reward 模式而非推理</p></li>
</ul>

<p>KL 惩罚 <span class="arithmatex">\(\beta D_\text{KL}[\pi_\theta \| \pi_\text{ref}]\)</span> 是主要的多样性机制，但单靠它并不够。</p>
</div>

<div class="admonition note"><p class="admonition-title">GRPO 组多样性</p>
<p>GRPO 对每个 prompt 生成 <span class="arithmatex">\(N\)</span> 条响应并使用组内排序。组内多样性至关重要：</p>

<ul>
<li><p><strong>高温度</strong>（<span class="arithmatex">\(\tau=0.8\)</span>–<span class="arithmatex">\(1.0\)</span>）：确保响应多样，便于有意义的比较</p></li>
<li><p><strong>较大的 <span class="arithmatex">\(N\)</span></strong>（8–16）：样本越多，越可能同时包含好的和差的方案</p></li>
<li><p><strong>DAPO 的“不重复”惩罚</strong>：拒绝组内重复响应，强制探索</p></li>
<li><p>若 <span class="arithmatex">\(N\)</span> 条响应完全相同：advantage 为零，无学习信号</p></li>
<li><p>若响应过于多样（随机）：reward 信号嘈杂，学习缓慢</p></li>
</ul>

<p><strong>甜点</strong>：温度选取应让方案有差异，同时仍然切题。</p>
</div>

<p class="table-caption" id="table-7-1"><span class="table-number">表 7.1　</span>RL 训练中促进多样性的方法。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>如何促进多样性</strong></th></tr></thead>
<tbody><tr><td>熵奖励（Entropy bonus）</td><td>在 reward 中加入 <span class="arithmatex">\(\alpha H(\pi_\theta)\)</span>。直接惩罚低熵（确定性）policy。</td></tr>
<tr><td>KL 惩罚</td><td><span class="arithmatex">\(-\beta D_\text{KL}[\pi_\theta \| \pi_\text{ref}]\)</span> 防止坍塌到单一模式。</td></tr>
<tr><td>拒绝采样（Rejection Sampling）</td><td>生成大量候选，按 reward 保留 top-<span class="arithmatex">\(k\)</span>。自然筛选出多样的高质量响应。</td></tr>
<tr><td>Best-of-N</td><td>推理时：生成 <span class="arithmatex">\(N\)</span> 条响应，全部评分，返回最佳。多样性来自采样。</td></tr>
<tr><td>带多样性 pair 的 DPO</td><td>在 chosen/rejected 不仅质量不同、<em>方法</em>也不同的 pair 上训练。</td></tr>
<tr><td>多 reward</td><td>使用多个 reward model（安全、有用性、代码质量）。防止坍塌到单一维度。</td></tr></tbody>
</table>
</div>

<div class="admonition warning"><p class="admonition-title">多样性 vs. 质量的权衡</p>
<p>更多样性并非总是更好：</p>

<ul>
<li><p>过度多样（高熵）= 随机、无用的响应</p></li>
<li><p>多样性不足（低熵）= 重复的、被 reward hack 的响应</p></li>
<li><p><strong>监控</strong>：训练中跟踪响应熵、唯一 n-gram 比例和 reward 分布宽度。若三者同时下降，你就遇到了坍塌问题。</p></li>
</ul>
</div>

#### 用于 RL 数据收集的 Verbalized Sampling {#verbalized-sampling-for-rl-data-collection}

<p>后训练对齐（基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）、直接偏好优化（Direct Preference Optimization, DPO））经常因为 <em>典型性偏差</em>（typicality bias）而降低输出多样性：人类标注者系统性地偏好熟悉的、“典型的”文本而非新颖的替代方案。这种模式坍塌是数据层面的现象，并非纯算法问题。</p>

<p>Verbalized Sampling（VS） <a class="book-citation" id="cite-zhang2025verbalized-1" href="../../references.md#ref-zhang2025verbalized" aria-label="参考文献 422">[422]</a> 是一种无需训练的 prompt 策略，通过要求模型在单次生成中 <strong>显式言语化多条响应上的概率分布</strong> 来规避这种坍塌。</p>

<div class="admonition note"><p class="admonition-title">Verbalized Sampling —— 核心思想</p>
<p>与其采样单一响应（这会坍塌到众数），不如 prompt 模型输出 <em>多个候选响应及其概率</em>：</p>

<p><code>‘‘Generate 5 jokes about coffee and their corresponding probabilities.’’</code></p>

<p>模型会产出如下列表：</p>

<ol>
<li><p>笑话 A（概率：0.35）</p></li>
<li><p>笑话 B（概率：0.25）</p></li>
<li><p>笑话 C（概率：0.20）</p></li>
<li><p>笑话 D（概率：0.12)</p></li>
<li><p>笑话 E（概率：0.08）</p></li>
</ol>

<p>然后从这个言语化分布中采样。由于模型显式表征了整个分布（而不仅是 argmax），那些概率较低但富有创意/多样的响应就变得可达。</p>
</div>

<div class="highlight"><pre><span></span><code><span class="c1"># Verbalized Sampling：prompt 模型输出分布</span>
<span class="k">def</span><span class="w"> </span><span class="nf">verbalized_sample</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">tokenizer</span><span class="p">,</span> <span class="n">task</span><span class="p">,</span> <span class="n">n</span><span class="o">=</span><span class="mi">5</span><span class="p">):</span>
    <span class="n">prompt</span> <span class="o">=</span> <span class="p">(</span>
        <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">task</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>
        <span class="sa">f</span><span class="s2">"Generate </span><span class="si">{</span><span class="n">n</span><span class="si">}</span><span class="s2"> different responses and assign a probability "</span>
        <span class="sa">f</span><span class="s2">"to each (probabilities should sum to 1.0). "</span>
        <span class="sa">f</span><span class="s2">"Format: [response] (probability: X.XX)"</span>
    <span class="p">)</span>
    <span class="n">output</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">generate</span><span class="p">(</span>
        <span class="n">tokenizer</span><span class="p">(</span><span class="n">prompt</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span><span class="p">)</span><span class="o">.</span><span class="n">input_ids</span><span class="p">,</span>
        <span class="n">max_new_tokens</span><span class="o">=</span><span class="mi">1024</span><span class="p">,</span>
        <span class="n">temperature</span><span class="o">=</span><span class="mf">0.7</span><span class="p">,</span>
        <span class="n">do_sample</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="c1"># 从输出中解析响应及概率</span>
    <span class="n">responses</span><span class="p">,</span> <span class="n">probs</span> <span class="o">=</span> <span class="n">parse_verbalized_distribution</span><span class="p">(</span>
        <span class="n">tokenizer</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">output</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
    <span class="p">)</span>
    <span class="c1"># 从言语化分布中采样</span>
    <span class="kn">import</span><span class="w"> </span><span class="nn">random</span>
    <span class="n">chosen</span> <span class="o">=</span> <span class="n">random</span><span class="o">.</span><span class="n">choices</span><span class="p">(</span><span class="n">responses</span><span class="p">,</span> <span class="n">weights</span><span class="o">=</span><span class="n">probs</span><span class="p">,</span> <span class="n">k</span><span class="o">=</span><span class="mi">1</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
    <span class="k">return</span> <span class="n">chosen</span>
</code></pre></div>

<div class="admonition tip"><p class="admonition-title">为什么 Verbalized Sampling 有效</p>
<ul>
<li><p><strong>绕过模式坍塌</strong>：从对齐模型的标准采样高度集中在一两个“安全”响应。VS 强迫模型说出那些它 <em>知道</em> 但通常不会浮现出来的替代方案。</p></li>
<li><p><strong>多样性是语义的</strong>：与温度缩放（词法噪声）不同，VS 产出真正不同的方案——模型会就不同选项进行推理。</p></li>
<li><p><strong>随能力扩展</strong>：能力更强的模型产出校准更好的言语化分布——它们从 VS 中受益 <em>更多</em>（创意写作上获得 1.6–2.1<span class="arithmatex">\(\times\)</span> 多样性增益）。</p></li>
<li><p><strong>无需训练</strong>：不需要微调或修改解码；在推理时与任何遵循指令的模型协同工作。</p></li>
<li><p><strong>用于 GRPO</strong>：用 VS 为每个 prompt 生成 <span class="arithmatex">\(G\)</span> 个响应候选——确保组内包含语义多样的方案，而非仅表层变化。</p></li>
</ul>
</div>

<p>在深入各种扩展之前，让我们简要回顾前几节建立的基础 GRPO 算法。其核心机制——采样一组 completion、归一化它们的 reward、并应用 clipped policy gradient——简洁优雅。然而，实践者很快发现了一些具体的失败模式：预训练偏差稀释 gradient（Dr. GRPO）、对称 clipping 限制探索（DAPO）、过大的组大小造成浪费（2-GRPO）、以及多目标场景下 reward 尺度失衡（GDPO）。下面各节依次解决这些问题。</p>

<div class="admonition note"><p class="admonition-title">GRPO 基线回顾</p>
<p>给定 prompt <span class="arithmatex">\(q\)</span>，从当前 policy <span class="arithmatex">\(\pi_\theta\)</span> 采样 <span class="arithmatex">\(G\)</span> 条 completion <span class="arithmatex">\(\{o_1,\dots,o_G\}\)</span>。计算 reward <span class="arithmatex">\(\{r_1,\dots,r_G\}\)</span> 并归一化：</p>



<div class="arithmatex">\[\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}, \qquad
  \mu_r = \frac{1}{G}\sum_{i=1}^G r_i, \quad
  \sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i-\mu_r)^2}.\]</div>

<p>逐 token 的 clipped surrogate loss 为：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}
    \sum_{t=1}^{|o_i|}
    \min\!\Bigl(
      \rho_{i,t}\,\hat{A}_i,\;
      \mathrm{clip}(\rho_{i,t},1{-}\epsilon,1{+}\epsilon)\,\hat{A}_i
    \Bigr),\]</div>

<p>其中 <span class="arithmatex">\(\rho_{i,t} = \pi_\theta(o_{i,t}|q,o_{i,&lt;t})\,/\,\pi_{\text{old}}(o_{i,t}|q,o_{i,&lt;t})\)</span>。</p>
</div>

### 7.5.2　DAPO —— Dynamic Adaptive Policy Optimization {#sec:dapo}

<div class="admonition tip"><p class="admonition-title">为什么需要 DAPO？</p>
<p>基础 GRPO 使用 <em>对称</em> clipping：无论 policy 想增加还是减小某 token 的概率，约束都是相同的。但探索与利用具有不同的风险特征。增大一个好 token 的概率通常是安全的；而抑制一个恰好出现在坏 completion 中的 token，若该 token 本身是中性的，则可能是灾难性的错误。DAPO <a class="book-citation" id="cite-yu2025dapo-1" href="../../references.md#ref-yu2025dapo" aria-label="参考文献 413">[413]</a> 引入五项针对性修复，共同显著改进训练稳定性与最终性能。</p>
</div>

#### 组件 1 —— 非对称 Clipping（Clip-Higher） {#component-1-asymmetric-clipping-clip-higher}

<p>标准 PPO/GRPO 在 <span class="arithmatex">\([1-\epsilon, 1+\epsilon]\)</span> 处对称地 clip importance ratio。DAPO 将其替换为非对称区间：</p>



<div class="arithmatex">\[\boxed{
    \mathrm{clip}_{\text{DAPO}}(\rho, A) =
    \begin{cases}
      \mathrm{clip}(\rho,\, 1-\epsilon,\, 1+\epsilon_{\text{high}}) &amp; \text{if } A &gt; 0 \\
      \mathrm{clip}(\rho,\, 1-\epsilon,\, 1+\epsilon) &amp; \text{if } A \le 0
    \end{cases}
  }\]</div>

<p>其中 <span class="arithmatex">\(\epsilon_{\text{high}} &gt; \epsilon\)</span>（典型取值：<span class="arithmatex">\(\epsilon=0.2\)</span>，<span class="arithmatex">\(\epsilon_{\text{high}}=0.28\)</span>）。当 advantage 为正时，允许 policy 更进一步偏向好 token；当 advantage 为负时，应用常规的保守 clipping 以避免过度抑制。</p>

#### 组件 2 —— Token 级 Loss 聚合 {#component-2-token-level-loss-aggregation}

<p>基础 GRPO 将 loss 除以 <em>序列数</em> <span class="arithmatex">\(G\)</span>。DAPO 则除以所有序列的 <em>token 总数</em>：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{token}} =
    -\frac{1}{\sum_{i=1}^G |o_i|}
    \sum_{i=1}^G \sum_{t=1}^{|o_i|}
    \min\!\bigl(\rho_{i,t}\hat{A}_i,\;
                \mathrm{clip}_{\text{DAPO}}(\rho_{i,t},\hat{A}_i)\,\hat{A}_i\bigr).\]</div>

<p>这样可以防止长 completion 仅因为 token 数更多而主导 gradient 信号。</p>

#### 组件 3 —— 过长过滤（Overlong Filtering） {#component-3-overlong-filtering}

<p>当一条 completion 被截断（在最大长度预算内没有 EOS token），它会提供 <em>误导性</em> 信号：模型因为那些被正确生成、只是恰好出现在截断边界之前的 token 而被惩罚。DAPO 将这些 completion 整条 mask 掉：</p>



<div class="arithmatex">\[m_i = \mathbf{1}[\text{EOS} \in o_i], \qquad
  \mathcal{L}_{\text{filtered}} =
    -\frac{\sum_{i=1}^G m_i \sum_t (\cdots)}{\sum_{i=1}^G m_i |o_i|}.\]</div>

#### 组件 4 —— 软过长惩罚（Soft Overlong Punishment） {#component-4-soft-overlong-punishment}

<p>比起硬 mask，更软的变体应用一个长度惩罚，随着 completion 接近最大长度 <span class="arithmatex">\(L_{\max}\)</span> 平滑增长：</p>



<div class="arithmatex">\[r_i \leftarrow r_i - \lambda \cdot \max\!\left(0,\, \frac{|o_i| - L_{\text{cache}}}{L_{\max} - L_{\text{cache}}}\right),\]</div>

<p>其中 <span class="arithmatex">\(L_{\text{cache}}\)</span> 是一个“安全”长度阈值。</p>

#### 组件 5 —— 动态采样（Dynamic Sampling） {#component-5-dynamic-sampling}

<p>DAPO 会重新采样那些整组 completion 获得相同 reward（全对或全错）的 prompt，因为此类组在归一化后贡献零 gradient。这能在整个训练过程中保持有效 batch size 稳定。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 DAPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="c1"># 非对称 clipping</span>
    <span class="n">epsilon</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span>
    <span class="n">epsilon_high</span><span class="o">=</span><span class="mf">0.28</span><span class="p">,</span>          <span class="c1"># Clip-Higher</span>
    <span class="c1"># Token 级 loss</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"dapo"</span><span class="p">,</span>           <span class="c1"># 启用 token 级聚合</span>
    <span class="c1"># 过长过滤</span>
    <span class="n">mask_truncated_completions</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="c1"># 生成预算</span>
    <span class="n">max_completion_length</span><span class="o">=</span><span class="mi">1024</span><span class="p">,</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="c1"># 注意：DAPO loss 内部已处理零方差组过滤</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">GRPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">reward_funcs</span><span class="o">=</span><span class="p">[</span><span class="n">reward_fn</span><span class="p">],</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">何时使用 DAPO</p>
<ul>
<li><p>completion 经常触及长度上限的长篇推理任务。</p></li>
<li><p>任何观察到 reward 方差在训练中途坍塌的场景。</p></li>
<li><p>当基础 GRPO 表现不稳定（loss 尖峰、熵坍塌）时。</p></li>
<li><p>在大多数任务上推荐作为基础 GRPO 的 <em>即插即用改进</em>。</p></li>
</ul>
</div>

### 7.5.3　GSPO —— Group Sequence Policy Optimization {#sec:gspo}

<div class="admonition tip"><p class="admonition-title">Off-Policy 问题</p>
<p>GRPO <em>逐 token</em> 地 clip importance ratio。但一条 500 token 的序列，即使每个单独的 ratio 都在 <span class="arithmatex">\([1-\epsilon, 1+\epsilon]\)</span> 内，逐 token ratio 的乘积可能大或小到天文数字。当在同一 batch 上进行多次 gradient 步（off-policy）时，这种不匹配迅速放大，clipping 界限在序列级别上变得毫无意义。</p>
</div>

<p>GSPO <a class="book-citation" id="cite-chen2025gspo-1" href="../../references.md#ref-chen2025gspo" aria-label="参考文献 51">[51]</a> 将 <em>序列级</em> importance weight 定义为逐 token ratio 的几何平均，等价于完整序列概率比的 <span class="arithmatex">\(|o_i|\)</span> 次方根：</p>



<div class="arithmatex">\[\boxed{
    s_i(\theta) = \left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)}\right)^{1/|o_i|}
    = \exp\!\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log \frac{\pi_\theta(o_{i,t}|q,o_{i,&lt;t})}{\pi_{\text{old}}(o_{i,t}|q,o_{i,&lt;t})}\right).
  }\]</div>

<p>这是 <em>长度归一化</em> 的序列概率比。GSPO loss 对每条序列 clip 这个单一标量：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{GSPO}} = -\frac{1}{G}\sum_{i=1}^G
    \min\!\Bigl(s_i(\theta)\,\hat{A}_i,\;
               \mathrm{clip}(s_i(\theta),1{-}\epsilon,1{+}\epsilon)\,\hat{A}_i\Bigr).\]</div>

<div class="admonition note"><p class="admonition-title">GSPO 与 GRPO 的 Clipping 对比</p>
<ul>
<li><p><strong>GRPO</strong>：独立 clip <span class="arithmatex">\(|o_i|\)</span> 个逐 token ratio。一条序列可以所有 ratio 都在界内、却拥有 <span class="arithmatex">\(10^{50}\)</span> 的乘积 ratio。</p></li>
<li><p><strong>GSPO</strong>：对每条序列 clip 一次几何平均。保证 <em>序列级</em> policy 变化有界。</p></li>
<li><p>GSPO 在 off-policy 重要性采样上理论正确；GRPO 只是近似。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的 GSPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="c1"># 序列级重要性采样</span>
    <span class="n">importance_sampling_level</span><span class="o">=</span><span class="s2">"sequence"</span><span class="p">,</span>   <span class="c1"># GSPO 模式</span>
    <span class="c1"># Off-policy：每个 batch 复用多次 gradient 步</span>
    <span class="n">steps_per_generation</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="n">epsilon</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">GRPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">reward_funcs</span><span class="o">=</span><span class="p">[</span><span class="n">reward_fn</span><span class="p">],</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition warning"><p class="admonition-title">何时使用 GSPO</p>
<p>GSPO 在 <code>steps_per_generation &gt; 1</code>（off-policy 训练）时最有用。对于纯 on-policy 训练（<span class="arithmatex">\(\text{steps\_per\_generation}=1\)</span>），它与 GRPO 的差异可忽略。Off-policy 训练能极大降低生成成本（最昂贵的步骤），使得 GSPO + off-policy 成为强力的效率选择。</p>
</div>

### 7.5.4　Dr. GRPO —— Debiased Reward GRPO {#sec:dr-grpo}

<div class="admonition tip"><p class="admonition-title">预训练偏差问题</p>
<p>标准 GRPO 在组内归一化 advantage，但 <em>预训练分布</em> 会引入系统性偏差：在预训练数据中常见的 token 即便不携带任务相关信息也会获得较大 gradient。Dr. GRPO <a class="book-citation" id="cite-liu2024drgrpo-1" href="../../references.md#ref-liu2024drgrpo" aria-label="参考文献 232">[232]</a> 识别并纠正这种偏差，将 gradient 信号聚焦于 <em>有信息量的</em> token。</p>
</div>

<p>Dr. GRPO 修改逐 token gradient 权重，以考虑该 token 对 reward 信号的边际贡献。模型本就赋予高概率的 token（无论 reward 如何）会被降权：</p>



<div class="arithmatex">\[w_{i,t} = \hat{A}_i \cdot \bigl(1 - \pi_{\text{ref}}(o_{i,t}|q,o_{i,&lt;t})\bigr),\]</div>

<p>其中 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 是参考（预训练）模型。这是一种 <em>token 效率</em> 形式：gradient 被集中到 policy 真正需要改变的 token 上。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 Dr. GRPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"dr_grpo"</span><span class="p">,</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.04</span><span class="p">,</span>   <span class="c1"># KL 惩罚系数</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">GRPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>   <span class="c1"># token 加权需要参考模型</span>
    <span class="n">reward_funcs</span><span class="o">=</span><span class="p">[</span><span class="n">reward_fn</span><span class="p">],</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">何时使用 Dr. GRPO</p>
<ul>
<li><p>当训练任务在预训练与 RL 之间词表分布严重不匹配时。</p></li>
<li><p>当观察到常见的填充 token 主导 gradient 时。</p></li>
<li><p>与接近初始 policy 的参考模型配合良好。</p></li>
</ul>
</div>

### 7.5.5　2-GRPO —— 最简的两次 Rollout GRPO {#sec:2grpo}

<div class="admonition tip"><p class="admonition-title">&quot;It Takes Two&quot; 洞见</p>
<p>&quot;It Takes Two&quot; 论文 <a class="book-citation" id="cite-xu2025twograpo-1" href="../../references.md#ref-xu2025twograpo" aria-label="参考文献 399">[399]</a> 从经验和理论上证明：<span class="arithmatex">\(G=2\)</span> 的 GRPO（每个 prompt 仅两条 completion）在大多数推理基准上能匹配甚至超过 <span class="arithmatex">\(G=16\)</span> 的 GRPO。这令人意外——为什么更少样本反而足够？</p>
</div>

<p>关键洞见是：GRPO 的有效性并 <em>不</em> 主要来自准确的 advantage 估计（那需要较大的 <span class="arithmatex">\(G\)</span>），而是来自一个结构上类似 DPO 的隐式 <em>对比目标</em>：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{2-GRPO}} \approx
  -\mathbb{E}_{(o^+, o^-) \sim \pi_\theta}\!\left[
    \log \sigma\!\left(
      \beta \log \frac{\pi_\theta(o^+|q)}{\pi_{\text{old}}(o^+|q)}
      - \beta \log \frac{\pi_\theta(o^-|q)}{\pi_{\text{old}}(o^-|q)}
    \right)
  \right],\]</div>

<p>其中 <span class="arithmatex">\(o^+\)</span> 是较高 reward 的 completion，<span class="arithmatex">\(o^-\)</span> 是较低 reward 的。<span class="arithmatex">\(G=2\)</span> 时这一对比结构是显式的；<span class="arithmatex">\(G=16\)</span> 时同样的信号存在，但被冗余对稀释。</p>

<div class="admonition note"><p class="admonition-title">2-GRPO 的算力节省</p>
<ul>
<li><p><span class="arithmatex">\(G=2\)</span> vs <span class="arithmatex">\(G=16\)</span>：<strong>生成算力减少 8<span class="arithmatex">\(\times\)</span></strong>。</p></li>
<li><p>生成通常是瓶颈（占 60–80% 的 wall-clock 时间）。</p></li>
<li><p>端到端总训练加速：约 4–6<span class="arithmatex">\(\times\)</span>。</p></li>
<li><p>在 GSM8K、MATH 和代码基准上无精度损失。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的 2-GRPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>      <span class="c1"># 关键改动 —— 只做两次 rollout</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"grpo"</span><span class="p">,</span>       <span class="c1"># 标准 GRPO loss 即可</span>
    <span class="n">epsilon</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span>
    <span class="c1"># G=2 时 batch size 至少为 2 * num_prompts_per_step</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">GRPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">reward_funcs</span><span class="o">=</span><span class="p">[</span><span class="n">reward_fn</span><span class="p">],</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition warning"><p class="admonition-title">2-GRPO 的注意事项</p>
<p><span class="arithmatex">\(G=2\)</span> 时 advantage 归一化只对两个值进行，因此归一化后的 advantage 总是 <span class="arithmatex">\(\{+1, -1\}\)</span>（若两 reward 相等则为 <span class="arithmatex">\(\{0, 0\}\)</span>）。这意味着 gradient 幅度与 reward 差距无关。对于 reward 差异的 <em>幅度</em> 很重要的任务（例如部分得分），更大的 <span class="arithmatex">\(G\)</span> 仍然有利。</p>
</div>

### 7.5.6　SAPO —— Soft Adaptive Policy Optimization {#sec:sapo}

<div class="admonition tip"><p class="admonition-title">硬 Clipping 的脆弱性</p>
<p>PPO 风格的 clipping 产生不连续 gradient：clip 带外 gradient 为零，带内非零。这种“悬崖效应”会在边界附近造成不稳定，并使信赖域对 <span class="arithmatex">\(\epsilon\)</span> 的选择敏感。SAPO <a class="book-citation" id="cite-han2025sapo-1" href="../../references.md#ref-han2025sapo" aria-label="参考文献 131">[131]</a> 用一个平滑、温度可控的 gate 函数替换硬 clip。</p>
</div>

<p>SAPO 用一个平滑替代品替换 <span class="arithmatex">\(\min(\rho A, \mathrm{clip}(\rho,\cdot)\,A)\)</span> 目标：</p>



<div class="arithmatex">\[\boxed{
    \mathcal{L}_{\text{SAPO}}(\rho, A) =
    \begin{cases}
      -A \cdot \sigma\!\left(\dfrac{\rho - 1}{\tau_+}\right) \cdot \rho
        &amp; \text{if } A &gt; 0 \\[8pt]
      -A \cdot \sigma\!\left(\dfrac{1 - \rho}{\tau_-}\right) \cdot \rho
        &amp; \text{if } A \le 0
    \end{cases}
  }\]</div>

<p>其中 <span class="arithmatex">\(\sigma\)</span> 是 sigmoid 函数，<span class="arithmatex">\(\tau_+, \tau_-\)</span> 是非对称温度参数。温度越高 gate 越软（更多探索）；温度越低越接近硬 clipping。</p>

<div class="admonition note"><p class="admonition-title">SAPO 温度直觉</p>
<ul>
<li><p><span class="arithmatex">\(\tau_+ = 1.0\)</span>：正 advantage 的中等 gate（允许探索）。</p></li>
<li><p><span class="arithmatex">\(\tau_- = 1.05\)</span>：负 advantage 略软的 gate（避免过度抑制）。</p></li>
<li><p><span class="arithmatex">\(\tau \to 0\)</span>：恢复硬 PPO clipping。</p></li>
<li><p><span class="arithmatex">\(\tau \to \infty\)</span>：恢复未 clip 的 policy gradient。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的 SAPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"sapo"</span><span class="p">,</span>
    <span class="n">sapo_temperature_pos</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>    <span class="c1"># 正 advantage 的 tau_+</span>
    <span class="n">sapo_temperature_neg</span><span class="o">=</span><span class="mf">1.05</span><span class="p">,</span>   <span class="c1"># 负 advantage 的 tau_-</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">GRPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">reward_funcs</span><span class="o">=</span><span class="p">[</span><span class="n">reward_fn</span><span class="p">],</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

### 7.5.7　TIS 与 MIS —— Truncated 与 Masked Importance Sampling {#sec:tis-mis}

<div class="admonition warning"><p class="admonition-title">隐蔽的 vLLM 概率不匹配</p>
<p>当使用 vLLM 进行快速生成时，vLLM 返回的 log-probability 与训练前向传播中计算的不同 <a class="book-citation" id="cite-zhong2025tismis-1" href="../../references.md#ref-zhong2025tismis" aria-label="参考文献 436">[436]</a>。这 <em>不是</em> bug —— 它源于不同的 CUDA 内核、不同的浮点精度以及不同的 Attention 实现（如 FlashAttention vs PagedAttention）。这种不匹配悄悄破坏了 on-policy 假设：用于计算 importance ratio 的“旧 policy”概率是错的，导致 gradient 估计出现偏差。</p>
</div>

#### Truncated Importance Sampling（TIS） {#truncated-importance-sampling-tis}

<p>TIS 通过将 gradient 乘以一个截断的修正因子来纠正偏差：</p>



<div class="arithmatex">\[\boxed{
    w_{\text{TIS}}(o_i) = \min\!\left(C,\; \frac{\pi_{\text{train}}(o_i|q)}{\pi_{\text{vllm}}(o_i|q)}\right),
  }\]</div>

<p>其中 <span class="arithmatex">\(\pi_{\text{train}}\)</span> 是训练前向传播给出的概率，<span class="arithmatex">\(\pi_{\text{vllm}}\)</span> 是 vLLM 报告的概率。在 <span class="arithmatex">\(C\)</span> 处截断可防止极端修正破坏训练稳定性。</p>

#### Masked Importance Sampling（MIS） {#masked-importance-sampling-mis}

<p>MIS 采取更激进的策略：对任何修正 ratio 超过阈值 <span class="arithmatex">\(C\)</span> 的序列，将其 gradient 置零：</p>



<div class="arithmatex">\[w_{\text{MIS}}(o_i) = \mathbf{1}\!\left[\frac{\pi_{\text{train}}(o_i|q)}{\pi_{\text{vllm}}(o_i|q)} \le C\right].\]</div>

<p>这更保守，但避免了大（即使被截断）修正权重的风险。</p>

#### 序列级 vs Token 级 IS {#sequence-level-vs-token-level-is}

<p>TIS 和 MIS 都既可在 token 级、也可在序列级应用：</p>

<ul>
<li><p><strong>序列级</strong>：将 ratio 计算为所有 token 的几何平均（如 GSPO 那样）。理论正确但方差更高。</p></li>
<li><p><strong>Token 级</strong>：为每个 token 单独计算 ratio。有偏（逐 token 修正的乘积不等于序列修正），但方差更低。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">TRL 中的 TIS 和 MIS</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="c1"># 针对 vLLM 概率不匹配的 Truncated IS 修正</span>
<span class="n">config_tis</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="n">use_vllm</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">vllm_importance_sampling_correction</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">vllm_importance_sampling_mode</span><span class="o">=</span><span class="s2">"sequence_truncate"</span><span class="p">,</span>  <span class="c1"># TIS</span>
    <span class="n">vllm_importance_sampling_cap</span><span class="o">=</span><span class="mf">5.0</span><span class="p">,</span>                   <span class="c1"># C 阈值</span>
<span class="p">)</span>

<span class="c1"># Masked IS 修正</span>
<span class="n">config_mis</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="n">use_vllm</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">vllm_importance_sampling_correction</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">vllm_importance_sampling_mode</span><span class="o">=</span><span class="s2">"sequence_mask"</span><span class="p">,</span>      <span class="c1"># MIS</span>
    <span class="n">vllm_importance_sampling_cap</span><span class="o">=</span><span class="mf">3.0</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">何时使用 TIS/MIS</p>
<ul>
<li><p>使用 vLLM 生成时 <strong>始终</strong> 考虑开启。</p></li>
<li><p>当不匹配较小时（同模型、不同精度）首选 TIS。</p></li>
<li><p>当不匹配较大或不可预测时首选 MIS。</p></li>
<li><p>序列级 IS 在理论上更受推荐；token 级是实用上的折中。</p></li>
</ul>
</div>

### 7.5.8　VESPO —— Variational Sequence-Level Soft Policy Optimization {#sec:vespo}

<div class="admonition tip"><p class="admonition-title">原理性的 Reward 重塑</p>
<p>大多数 GRPO 变体只是启发式地修改 clipping 机制。VESPO 从变分推断框架推导出一个原理性的 reward 重塑核，将 policy optimization 视为近似后验推断。VESPO <a class="book-citation" id="cite-luo2025vespo-1" href="../../references.md#ref-luo2025vespo" aria-label="参考文献 243">[243]</a> 推导出的核是平滑、非对称的，并天然地处理异步或 off-policy 训练中的陈旧性问题。</p>
</div>

<p>VESPO 从变分目标推导出每条 trajectory <span class="arithmatex">\(\tau\)</span> 的加权函数 <span class="arithmatex">\(W(\tau)\)</span>。最终的 gradient 权重形式为：</p>



<div class="arithmatex">\[\boxed{
    g(\tau) = W(\tau)^k \cdot \exp\!\bigl(\lambda(1 - W(\tau))\bigr),
  }\]</div>

<p>其中 <span class="arithmatex">\(W(\tau) = \pi_\theta(\tau)/\pi_{\text{old}}(\tau)\)</span> 是序列级 importance weight，<span class="arithmatex">\(k\)</span> 控制加权的锐度，<span class="arithmatex">\(\lambda\)</span> 控制对陈旧（低权重）trajectory 的指数衰减。该核：</p>

<ul>
<li><p>处处平滑（在 clip 边界没有不连续 gradient）。</p></li>
<li><p>通过指数项天然地降权陈旧 trajectory（<span class="arithmatex">\(W \ll 1\)</span>）。</p></li>
<li><p>是非对称的：高权重 trajectory（<span class="arithmatex">\(W &gt; 1\)</span>）与低权重 trajectory 区别对待。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">TRL 中的 VESPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"vespo"</span><span class="p">,</span>
    <span class="n">vespo_k_pos</span><span class="o">=</span><span class="mf">2.0</span><span class="p">,</span>         <span class="c1"># 锐度指数（正 advantage）</span>
    <span class="n">vespo_lambda_pos</span><span class="o">=</span><span class="mf">3.0</span><span class="p">,</span>    <span class="c1"># 陈旧性衰减（正 advantage）</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="n">steps_per_generation</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>  <span class="c1"># off-policy；VESPO 处理陈旧性</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">GRPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">reward_funcs</span><span class="o">=</span><span class="p">[</span><span class="n">reward_fn</span><span class="p">],</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

### 7.5.9　DPPO —— Direct Policy Divergence Policy Optimization {#sec:dppo}

<div class="admonition tip"><p class="admonition-title">Ratio Clipping 的问题</p>
<p>PPO 的 ratio clipping 是约束新旧 policy 间 KL 散度的代理。但这个代理并不完美：clipping 对低概率 token 过度惩罚（这里概率的小绝对变化对应较大的 ratio 变化），而对高概率 token 惩罚不足（这里大的绝对变化对应较小的 ratio 变化）。DPPO <a class="book-citation" id="cite-an2025dppo-1" href="../../references.md#ref-an2025dppo" aria-label="参考文献 7">[7]</a> 用 <em>直接散度估计</em> 替换 ratio clipping。</p>
</div>

<p>DPPO 直接使用新旧 policy 分布间的 Total Variation（TV）或 KL 散度来计算信赖域约束：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{DPPO}} = -\mathbb{E}\!\left[
    \hat{A} \cdot \pi_\theta(o|q) \cdot \mathbf{1}[D(\pi_\theta \| \pi_{\text{old}}) \le \delta]
  \right],\]</div>

<p>其中 <span class="arithmatex">\(D\)</span> 是所选的散度度量。在实践中，DPPO 用 token 级二值或 top-<span class="arithmatex">\(k\)</span> mask 来近似：</p>

<ul>
<li><p><strong>binary_tv</strong>：mask 掉 <span class="arithmatex">\(|\pi_\theta - \pi_{\text{old}}| &gt; \delta\)</span> 的 token。</p></li>
<li><p><strong>binary_kl</strong>：mask 掉 <span class="arithmatex">\(\pi_\theta \log(\pi_\theta/\pi_{\text{old}}) &gt; \delta\)</span> 的 token。</p></li>
<li><p><strong>topk_tv</strong>：仅保留按 TV 贡献排序的 top-<span class="arithmatex">\(k\)</span> token。</p></li>
<li><p><strong>topk_kl</strong>：仅保留按 KL 贡献排序的 top-<span class="arithmatex">\(k\)</span> token。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">DPPO —— 概念性实现</p>
<p>DPPO 目前尚未作为内置 TRL trainer 提供。一种自定义实现可基于 GRPOTrainer，修改 loss 以基于分布散度（TV 或 KL）而非标准概率 ratio 进行 clip：</p>

<div class="highlight"><pre><span></span><code><span class="c1"># 伪代码：DPPO 需要自定义 trainer 子类</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.04</span><span class="p">,</span>
<span class="p">)</span>
<span class="c1"># 重写 loss 计算以使用分布式 clipping：</span>
<span class="c1"># 当 TV(pi_new || pi_old) &gt; delta 时 clip，而非</span>
<span class="c1"># 当 pi_new/pi_old 超过 [1-eps, 1+eps] 时</span>
</code></pre></div>
</div>

<div class="admonition warning"><p class="admonition-title">DPPO 处于研究阶段</p>
<p>DPPO 是较新的研究贡献，尚未集成到主流 RL 库中。它在你观察到标准 ratio clipping 失效时（例如在 token 概率分布高度偏斜的任务上）最有用。</p>
</div>

### 7.5.10　ScaleRL 和 CISPO {#sec:scalerl-cispo}

<div class="admonition tip"><p class="admonition-title">RL 的扩展律</p>
<p>ScaleRL 论文 <a class="book-citation" id="cite-luo2025scalerl-1" href="../../references.md#ref-luo2025scalerl" aria-label="参考文献 242">[242]</a> 系统研究了什么因素能让 LLM 的 RL 训练有效扩展。关键发现是：两项修改——batch 级 reward 缩放与 DAPO 风格的 token 级 loss——共同解锁了规模化下的强性能，单独任一项都不够。CISPO（Clipped IS Policy Optimization）就是由此得到的算法。</p>
</div>

#### Batch 级 Reward 缩放 {#batch-level-reward-scaling}

<p>标准 GRPO 在单个 prompt 的 <span class="arithmatex">\(G\)</span> 条 completion 组内归一化 reward。CISPO 则在 <em>整个 batch</em> 上归一化 reward：</p>



<div class="arithmatex">\[\hat{A}_i = \frac{r_i - \mu_{\text{batch}}}{\sigma_{\text{batch}} + \epsilon},\]</div>

<p>其中 <span class="arithmatex">\(\mu_{\text{batch}}\)</span> 和 <span class="arithmatex">\(\sigma_{\text{batch}}\)</span> 在当前训练 batch 的所有 reward 上计算。这提供更稳定的 baseline，并防止任何单一 prompt 主导 gradient。</p>

#### CISPO Loss {#cispo-loss}

<p>CISPO 结合了 batch 级缩放、DAPO 的 token 级 loss 聚合以及非对称 clipping：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{CISPO}} =
    -\frac{1}{\sum_{i,t} m_{i,t}}
    \sum_{i=1}^G \sum_{t=1}^{|o_i|} m_{i,t} \cdot
    \min\!\bigl(\rho_{i,t}\hat{A}_i,\;
               \mathrm{clip}_{\text{DAPO}}(\rho_{i,t},\hat{A}_i)\,\hat{A}_i\bigr),\]</div>

<p>其中 <span class="arithmatex">\(m_{i,t}\)</span> 是过长过滤 mask。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 CISPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"cispo"</span><span class="p">,</span>
    <span class="n">scale_rewards</span><span class="o">=</span><span class="s2">"batch"</span><span class="p">,</span>          <span class="c1"># batch 级 reward 归一化</span>
    <span class="n">mask_truncated_completions</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">epsilon</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span>
    <span class="n">epsilon_high</span><span class="o">=</span><span class="mf">5.0</span><span class="p">,</span>               <span class="c1"># CISPO 的 epsilon_max（ScaleRL 论文）</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">GRPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">reward_funcs</span><span class="o">=</span><span class="p">[</span><span class="n">reward_fn</span><span class="p">],</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">ScaleRL 关键发现</p>
<ol>
<li><p>单独的 batch 级 reward 缩放：改进有限。</p></li>
<li><p>单独的 token 级 loss：改进有限。</p></li>
<li><p>两者结合：<strong>协同效应</strong> —— 显著优于单独任一项。</p></li>
<li><p>batch 越大，从 batch 级缩放中受益越多。</p></li>
<li><p>CISPO 是大规模 RL 训练的推荐默认方案。</p></li>
</ol>
</div>

### 7.5.11　GDPO —— Group Reward-Decoupled Policy Optimization {#sec:gdpo}

<div class="admonition tip"><p class="admonition-title">多 Reward 坍塌问题</p>
<p>在多目标 RL 中（例如同时优化正确性与格式），标准 GRPO 归一化 <em>组合后</em> 的 reward。若某一项 reward 方差远高于另一项，它会主导归一化 advantage，相当于忽略其他 reward。这就是 <em>advantage 坍塌</em>：低方差 reward 贡献近乎为零的 gradient。GDPO <a class="book-citation" id="cite-zhong2025gdpo-1" href="../../references.md#ref-zhong2025gdpo" aria-label="参考文献 437">[437]</a> 在聚合前 <em>独立</em> 归一化每项 reward。</p>
</div>

<p>其核心机制在聚合前 <em>独立</em> 归一化每项 reward：</p>



<div class="arithmatex">\[\boxed{
    \hat{A}_n^{(i)} = \frac{r_n^{(i)} - \mu_n}{\sigma_n + \epsilon}, \qquad
    \hat{A}^{(i)} = \sum_{n=1}^N w_n \hat{A}_n^{(i)},
  }\]</div>

<p>其中 <span class="arithmatex">\(r_n^{(i)}\)</span> 是 completion <span class="arithmatex">\(i\)</span> 的第 <span class="arithmatex">\(n\)</span> 项 reward，<span class="arithmatex">\(\mu_n\)</span> 和 <span class="arithmatex">\(\sigma_n\)</span> 是组内第 <span class="arithmatex">\(n\)</span> 项 reward 的均值与标准差，<span class="arithmatex">\(w_n\)</span> 是用户指定权重。</p>

<div class="admonition note"><p class="admonition-title">GDPO 与标准多 reward GRPO 的对比</p>
<ul>
<li><p><strong>标准</strong>：<span class="arithmatex">\(\hat{A}^{(i)} = \frac{\sum_n w_n r_n^{(i)} - \mu_{\text{combined}}}{\sigma_{\text{combined}}}\)</span>。高方差 reward 占主导。</p></li>
<li><p><strong>GDPO</strong>：单独归一化每项 reward 后再组合。每项 reward 按其权重 <span class="arithmatex">\(w_n\)</span> 按比例贡献。</p></li>
<li><p>当 reward 量级或方差差异极大时，GDPO 不可或缺。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的 GDPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="n">multi_objective_aggregation</span><span class="o">=</span><span class="s2">"normalize_then_sum"</span><span class="p">,</span>
    <span class="n">reward_weights</span><span class="o">=</span><span class="p">[</span><span class="mf">1.0</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">],</span>   <span class="c1"># [正确性, 格式] 的权重</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
<span class="p">)</span>

<span class="k">def</span><span class="w"> </span><span class="nf">correctness_reward</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
    <span class="k">return</span> <span class="p">[</span><span class="mf">1.0</span> <span class="k">if</span> <span class="n">is_correct</span><span class="p">(</span><span class="n">c</span><span class="p">)</span> <span class="k">else</span> <span class="mf">0.0</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">completions</span><span class="p">]</span>

<span class="k">def</span><span class="w"> </span><span class="nf">format_reward</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
    <span class="k">return</span> <span class="p">[</span><span class="mf">0.1</span> <span class="k">if</span> <span class="n">has_good_format</span><span class="p">(</span><span class="n">c</span><span class="p">)</span> <span class="k">else</span> <span class="mf">0.0</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">completions</span><span class="p">]</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">GRPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">reward_funcs</span><span class="o">=</span><span class="p">[</span><span class="n">correctness_reward</span><span class="p">,</span> <span class="n">format_reward</span><span class="p">],</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

### 7.5.12　GOPO —— Group Ordinal Policy Optimization {#gopo-group-ordinal-policy-optimization}

<p>GOPO <a class="book-citation" id="cite-choi2025gopo-1" href="../../references.md#ref-choi2025gopo" aria-label="参考文献 54">[54]</a> 始于一个简单观察：reward model 是用成对比较（“A 是否优于 B？”）训练的，因此只有其输出的 <strong>排序</strong> 是可信的——原始数值分数本身没有内在意义。然而 GRPO 直接把这些原始幅度喂入 advantage 计算。对于不可验证 reward 的任务——摘要、开放式聊天、指令跟随——这种不匹配引入了噪声，因为 0.6 reward 分的差距可能在输出空间某个区域反映真实质量，而在另一个区域毫无意义。</p>

<p><strong>关键洞见</strong>：完全丢弃 reward 幅度。仅使用组内 reward 的 <strong>序数排名</strong>。</p>

<p><strong>算法</strong>：给定一组 <span class="arithmatex">\(N\)</span> 条响应 <span class="arithmatex">\(\{o_1, \ldots, o_N\}\)</span> 及其 reward <span class="arithmatex">\(\{r_1, \ldots, r_N\}\)</span>：</p>

<ol>
<li><p>按 reward 对响应排名：赋予 rank <span class="arithmatex">\(\text{rank}(o_i) \in \{1, \ldots, N\}\)</span>（1 = 最差，<span class="arithmatex">\(N\)</span> = 最佳）。</p></li>
<li><p>用基于 rank 的分数替换原始 advantage：</p>



<div class="arithmatex">\[\boxed{\hat{A}_i^{\text{GOPO}} = f\!\left(\frac{\text{rank}(o_i)}{N}\right)}
\tag{7.2}\]</div>

<p>其中 <span class="arithmatex">\(f\)</span> 是单调变换（例如线性映射到 <span class="arithmatex">\([-1, 1]\)</span> 或分位数归一化）。</p></li>
<li><p>用基于 rank 的 advantage 应用 PPO 风格的 clipped 目标。</p></li>
</ol>

<p><strong>与 GRPO 的对比</strong>：</p>

<span id="table-7-inline-3"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方面</strong></th><th scope="col"><strong>GRPO</strong></th><th scope="col"><strong>GOPO</strong></th></tr></thead>
<tbody><tr><td>Advantage 信号</td><td><span class="arithmatex">\(\hat{A}_i = (r_i - \mu)/\sigma\)</span>（使用幅度）</td><td><span class="arithmatex">\(\hat{A}_i = f(\text{rank}_i / N)\)</span>（仅使用序数 rank）</td></tr>
<tr><td>对 reward 尺度的敏感性</td><td>高 —— 校准不良的 RM 分数会扭曲 advantage</td><td>无 —— 对单调 reward 变换不变</td></tr>
<tr><td>最适合</td><td>可验证 reward（二值、校准良好）</td><td>不可验证 reward（基于 RM、幅度嘈杂）</td></tr></tbody>
</table>
</div>

<p><strong>经验性增益</strong>（在不可验证任务上相对 GRPO）：</p>

<ul>
<li><p>整个优化过程中，reward 曲线（训练和留出集）都位于 GRPO 之上</p></li>
<li><p>由独立 LLM 评估器判定的胜率在大多数训练 checkpoint 上有所改进</p></li>
<li><p>收敛显著更快——以更少的 gradient 步数达到 GRPO 的最终质量</p></li>
<li><p>优势随 reward model 变得更嘈杂或校准更差而扩大</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">GOPO vs. GRPO 的使用选择</p>
<ul>
<li><p><strong>使用 GRPO</strong>：当 reward 可验证且精确时（数学正确性、代码测试通过/失败、二值信号）。幅度携带有意义的信息。</p></li>
<li><p><strong>使用 GOPO</strong>：当 reward 来自学习的 reward model、面向主观任务（有用性、风格、安全）时。RM 的相对排序可信，但其绝对分数是任意的。</p></li>
</ul>
</div>
