# 第 9 章 · 奖励模型训练 {#sec:reward-models}

<p>奖励模型是连接人类偏好与 RL 训练信号的桥梁。训练良好的奖励模型对成功的 RLHF 至关重要；训练不佳的奖励模型会导致奖励黑客攻击（reward hacking）和行为失准。本章涵盖奖励模型的理论基础、实用训练技术以及架构选择。</p>

## 9.1　Bradley-Terry 模型 —— 完整推导 {#sec:bradley-terry}

<p>Bradley-Terry 模型 <a class="book-citation" id="cite-bradley1952rank-1" href="../../references.md#ref-bradley1952rank" aria-label="参考文献 29">[29]</a> 是成对偏好学习（pairwise preference learning）的标准概率框架。给定对 prompt <span class="arithmatex">\(q\)</span> 的两个回答 <span class="arithmatex">\(y_1\)</span> 和 <span class="arithmatex">\(y_2\)</span>，该模型假设：</p>



<div class="arithmatex">\[P(y_1 \succ y_2 \mid q) = \sigma(r(y_1, q) - r(y_2, q))
    = \frac{e^{r(y_1,q)}}{e^{r(y_1,q)} + e^{r(y_2,q)}},\]</div>

<p>其中 <span class="arithmatex">\(r: \mathcal{Y} \times \mathcal{Q} \to \mathbb{R}\)</span> 是标量 Reward 函数，<span class="arithmatex">\(\sigma\)</span> 是 sigmoid 函数。</p>

#### 最大似然估计 {#maximum-likelihood-estimation}

<p>给定偏好对数据集 <span class="arithmatex">\(\mathcal{D} = \{(q^{(k)}, y_w^{(k)}, y_l^{(k)})\}_{k=1}^N\)</span>，最大似然估计（MLE）目标为：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{BT}}(\phi) =
    -\frac{1}{N}\sum_{k=1}^N \log \sigma\!\bigl(r_\phi(y_w^{(k)}, q^{(k)}) - r_\phi(y_l^{(k)}, q^{(k)})\bigr),\]</div>

<p>其中 <span class="arithmatex">\(r_\phi\)</span> 是由 <span class="arithmatex">\(\phi\)</span> 参数化的神经网络。这是一个二元交叉熵 Loss，其中“正”类别对应偏好的回答。</p>

<div class="admonition note"><p class="admonition-title">Bradley-Terry 假设</p>
<ol>
<li><p>偏好具有<em>传递性</em>：若 <span class="arithmatex">\(y_1 \succ y_2\)</span> 且 <span class="arithmatex">\(y_2 \succ y_3\)</span>，则 <span class="arithmatex">\(y_1 \succ y_3\)</span>。</p></li>
<li><p>偏好由<em>标量</em>Reward 决定（不存在多维偏好）。</p></li>
<li><p>偏好概率仅依赖于 Reward 的<em>差值</em>。</p></li>
<li><p>各偏好对之间相互<em>独立</em>（不存在标注者效应）。</p></li>
</ol>

<p>这些假设在实践中常被违反，这催生了一些扩展模型，例如用于排序的 Plackett-Luce 模型和多维奖励模型。</p>
</div>

#### Margin Loss 扩展 {#margin-loss-extension}

<p>一种常见的扩展是引入 margin <span class="arithmatex">\(m\)</span>，以确保胜出与落败 Reward 之间存在最小间隔：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{margin}} =
    -\frac{1}{N}\sum_{k=1}^N \log \sigma\!\bigl(r_\phi(y_w^{(k)}, q^{(k)}) - r_\phi(y_l^{(k)}, q^{(k)}) - m\bigr).\]</div>

## 9.2　奖励模型架构 {#sec:rm-architectures}

<div class="admonition tip"><p class="admonition-title">LLM 上的分类头</p>
<p>标准的奖励模型架构是：取一个预训练 LLM，将其语言建模头（将隐藏状态映射到词表 logits）替换为一个标量回归头（将最终隐藏状态映射到单一 Reward 值）。</p>
</div>

<p>该架构由以下部分组成：</p>

<ol>
<li><p><strong>骨干网络（Backbone）</strong>：预训练 LLM（例如 Llama、Mistral），将 prompt-response 对编码为一系列隐藏状态。</p></li>
<li><p><strong>池化（Pooling）</strong>：提取最后一个 Token 位置（decoder-only 模型）或 <code>[CLS]</code> Token（encoder 模型）处的隐藏状态。</p></li>
<li><p><strong>回归头（Regression head）</strong>：线性层 <span class="arithmatex">\(W \in \mathbb{R}^{d \times 1}\)</span>，将池化后的隐藏状态映射为标量 Reward。</p></li>
</ol>

<div class="admonition example"><p class="admonition-title">TRL 中的奖励模型训练</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">RewardConfig</span><span class="p">,</span> <span class="n">RewardTrainer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoModelForSequenceClassification</span>

<span class="c1"># 加载带标量头(num_labels=1)的模型</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">AutoModelForSequenceClassification</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span>
    <span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">,</span>
    <span class="n">num_labels</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">RewardConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"reward_model"</span><span class="p">,</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">1e-5</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
    <span class="c1"># Margin loss</span>
    <span class="n">center_rewards_coefficient</span><span class="o">=</span><span class="mf">0.01</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">RewardTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>  <span class="c1"># 必须包含 chosen/rejected 列</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>
</div>

## 9.3　奖励模型训练技巧 {#sec:rm-tricks}

#### Reward 中心化 {#reward-centering}

<p>原始的奖励模型输出可能存在任意尺度和偏移。对 Reward 进行中心化（减去均值）可以稳定 RL 训练：</p>



<div class="arithmatex">\[r_{\text{centered}}(y, q) = r_\phi(y, q) - \mathbb{E}_{y&#x27; \sim \pi_\theta}[r_\phi(y&#x27;, q)].\]</div>

<p>在 TRL 中，这通过 <code>center_rewards_coefficient</code> 参数实现，它会向奖励模型 Loss 中添加一个正则项，惩罚均值非零的 Reward。</p>

#### 长度偏差校正 {#length-bias-correction}

<p>奖励模型往往存在<em>长度偏差</em>：无论质量如何，都倾向于给较长的回答分配更高的 Reward。可通过以下方式校正：</p>

<ol>
<li><p><strong>长度归一化（Length normalisation）</strong>：将 Reward 除以回答长度。</p></li>
<li><p><strong>长度受控训练（Length-controlled training）</strong>：将长度作为特征，训练模型使其对长度不变。</p></li>
<li><p><strong>校准（Calibration）</strong>：事后回归以剔除长度效应。</p></li>
</ol>

#### Margin Loss {#margin-losses}

<p>在 Bradley-Terry Loss 中加入 margin <span class="arithmatex">\(m\)</span>，可确保奖励模型对偏好回答与非偏好回答打出具有显著区分度的分数：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{margin}} = \max\!\bigl(0,\; m - (r_w - r_l)\bigr).\]</div>

## 9.4　过程奖励模型 vs 结果奖励模型 {#sec:prm-orm}

<div class="admonition note"><p class="admonition-title">PRM 与 ORM 对比</p>
<span id="table-9-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>属性</strong></th><th scope="col"><strong>ORM</strong></th><th scope="col"><strong>PRM</strong></th></tr></thead>
<tbody><tr><td>Reward 信号</td><td>仅最终答案</td><td>每一步推理</td></tr>
<tr><td>训练数据</td><td>(prompt, answer, correct?)</td><td>(prompt, steps, step labels)</td></tr>
<tr><td>标注成本</td><td>低</td><td>高</td></tr>
<tr><td>信用分配</td><td>稀疏</td><td>稠密</td></tr>
<tr><td>Reward 黑客</td><td>较易被攻击</td><td>较难被攻击</td></tr>
<tr><td>最适合</td><td>简单任务</td><td>多步推理</td></tr>
<tr><td>推理成本</td><td>低</td><td>高（需对每步打分）</td></tr></tbody>
</table>
</div>
</div>

<div class="admonition tip"><p class="admonition-title">何时使用 PRM</p>
<p>过程奖励模型（PRM）在以下场景中最有价值：</p>

<ul>
<li><p>任务需要多步推理（数学、代码、逻辑）。</p></li>
<li><p>最终答案是二元的（正确/错误），但中间步骤质量参差不齐。</p></li>
<li><p>你希望将奖励模型用于<em>搜索</em>（例如带步骤分数的 beam search）。</p></li>
<li><p>你能获得步骤级标注（或可自动生成）。</p></li>
</ul>

<p>对于简单任务（情感、毒性、事实性），ORM 已经足够，且成本低得多。</p>
</div>

<div class="admonition example"><p class="admonition-title">LLM RLHF 中的 PBRS</p>
<p><strong>原始 Reward</strong>：二元正确性（最终答案正确为 1，否则为 0）—— 对多步推理而言极其稀疏。</p>

<p><strong>势函数（Potential function）</strong>：<span class="arithmatex">\(\Phi(s) =\)</span> 来自验证器的部分得分（例如，逻辑上有效的中间推理步骤的比例）。</p>

<p><strong>塑形后的 Reward</strong>：Agent 对每个有效推理步骤都获得增量信号，同时保留“最优 Policy 仍然最大化最终答案正确率”的理论保证。</p>

<p><strong>实际实现</strong>：</p>

<ul>
<li><p>对思维链（Chain-of-Thought, CoT）中每一步打分的过程奖励模型（PRM）</p></li>
<li><p>代码生成中的中间编译检查</p></li>
<li><p>多部分答案的部分匹配分数</p></li>
</ul>

<p>这是基于势函数的 Reward 塑形（Potential-Based Reward Shaping, PBRS） <a class="book-citation" id="cite-ng1999policy-1" href="../../references.md#ref-ng1999policy" aria-label="参考文献 265">[265]</a> 在 LLM 场景下的直接应用 —— “塑形 Reward 保留最优 Policy”这一理论保证使得 PRM 成为在推理任务中获取稠密 Reward 的有原则方法。</p>
</div>

#### 自动 PRM 标注 {#automatic-prm-annotation}

<p>步骤级标注可通过以下方法自动生成：</p>

<ol>
<li><p><strong>蒙特卡洛 rollout</strong>：对每个中间步骤，采样多个补全（completion），将到达正确答案的比例作为该步的 Reward。</p></li>
<li><p><strong>LLM-as-judge</strong>：使用一个强大的 LLM 来评估每一步。</p></li>
<li><p><strong>形式化验证</strong>：对数学/代码任务，使用验证器逐步检查。</p></li>
</ol>

## 9.5　RLVR 中基于规则的 Reward {#sec:rule-based-rewards}

<p>可验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards, RLVR）使用确定性、基于规则的 Reward 函数，而非习得的奖励模型。这能显著减少奖励黑客行为（尽管模型仍可能利用格式技巧、边界情况或测试记忆来作弊），DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 即采用此方法。</p>

<div class="admonition example"><p class="admonition-title">TRL 中基于规则的 Reward 函数</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">re</span>

<span class="k">def</span><span class="w"> </span><span class="nf">format_reward</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""对使用 &lt;think&gt;...&lt;/think&gt;&lt;answer&gt;...&lt;/answer&gt; 格式给予 Reward。"""</span>
    <span class="n">rewards</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">pattern</span> <span class="o">=</span> <span class="sa">r</span><span class="s2">"&lt;think&gt;.*?&lt;/think&gt;\s*&lt;answer&gt;.*?&lt;/answer&gt;"</span>
    <span class="k">for</span> <span class="n">completion</span> <span class="ow">in</span> <span class="n">completions</span><span class="p">:</span>
        <span class="n">text</span> <span class="o">=</span> <span class="n">completion</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s2">"content"</span><span class="p">]</span>
        <span class="n">rewards</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mf">1.0</span> <span class="k">if</span> <span class="n">re</span><span class="o">.</span><span class="n">fullmatch</span><span class="p">(</span><span class="n">pattern</span><span class="p">,</span> <span class="n">text</span><span class="p">,</span> <span class="n">re</span><span class="o">.</span><span class="n">DOTALL</span><span class="p">)</span> <span class="k">else</span> <span class="mf">0.0</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">rewards</span>

<span class="k">def</span><span class="w"> </span><span class="nf">correctness_reward</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="n">ground_truth</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""对最终答案正确给予 Reward。"""</span>
    <span class="n">rewards</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">completion</span><span class="p">,</span> <span class="n">gt</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="n">ground_truth</span><span class="p">):</span>
        <span class="n">text</span> <span class="o">=</span> <span class="n">completion</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s2">"content"</span><span class="p">]</span>
        <span class="n">match</span> <span class="o">=</span> <span class="n">re</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="sa">r</span><span class="s2">"&lt;answer&gt;(.*?)&lt;/answer&gt;"</span><span class="p">,</span> <span class="n">text</span><span class="p">,</span> <span class="n">re</span><span class="o">.</span><span class="n">DOTALL</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">match</span><span class="p">:</span>
            <span class="n">answer</span> <span class="o">=</span> <span class="n">match</span><span class="o">.</span><span class="n">group</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span>
            <span class="n">rewards</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mf">1.0</span> <span class="k">if</span> <span class="n">answer</span> <span class="o">==</span> <span class="n">gt</span> <span class="k">else</span> <span class="mf">0.0</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">rewards</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mf">0.0</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">rewards</span>

<span class="k">def</span><span class="w"> </span><span class="nf">code_execution_reward</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="n">test_cases</span><span class="p">,</span> <span class="o">**</span><span class="n">kwargs</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""对通过测试用例的代码给予 Reward。"""</span>
    <span class="kn">import</span><span class="w"> </span><span class="nn">subprocess</span><span class="o">,</span><span class="w"> </span><span class="nn">tempfile</span><span class="o">,</span><span class="w"> </span><span class="nn">os</span>
    <span class="n">rewards</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">completion</span><span class="p">,</span> <span class="n">tests</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">completions</span><span class="p">,</span> <span class="n">test_cases</span><span class="p">):</span>
        <span class="n">code</span> <span class="o">=</span> <span class="n">completion</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s2">"content"</span><span class="p">]</span>
        <span class="n">passed</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="k">for</span> <span class="n">test</span> <span class="ow">in</span> <span class="n">tests</span><span class="p">:</span>
            <span class="k">with</span> <span class="n">tempfile</span><span class="o">.</span><span class="n">NamedTemporaryFile</span><span class="p">(</span>
                <span class="n">mode</span><span class="o">=</span><span class="s2">"w"</span><span class="p">,</span> <span class="n">suffix</span><span class="o">=</span><span class="s2">".py"</span><span class="p">,</span> <span class="n">delete</span><span class="o">=</span><span class="kc">False</span>
            <span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
                <span class="n">f</span><span class="o">.</span><span class="n">write</span><span class="p">(</span><span class="n">code</span> <span class="o">+</span> <span class="s2">"</span><span class="se">\n</span><span class="s2">"</span> <span class="o">+</span> <span class="n">test</span><span class="p">)</span>
                <span class="n">fname</span> <span class="o">=</span> <span class="n">f</span><span class="o">.</span><span class="n">name</span>
            <span class="k">try</span><span class="p">:</span>
                <span class="n">result</span> <span class="o">=</span> <span class="n">subprocess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
                    <span class="p">[</span><span class="s2">"python"</span><span class="p">,</span> <span class="n">fname</span><span class="p">],</span> <span class="n">capture_output</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                    <span class="n">timeout</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">text</span><span class="o">=</span><span class="kc">True</span>
                <span class="p">)</span>
                <span class="n">passed</span> <span class="o">+=</span> <span class="nb">int</span><span class="p">(</span><span class="n">result</span><span class="o">.</span><span class="n">returncode</span> <span class="o">==</span> <span class="mi">0</span><span class="p">)</span>
            <span class="k">except</span> <span class="ne">Exception</span><span class="p">:</span>
                <span class="k">pass</span>
            <span class="k">finally</span><span class="p">:</span>
                <span class="n">os</span><span class="o">.</span><span class="n">unlink</span><span class="p">(</span><span class="n">fname</span><span class="p">)</span>
        <span class="n">rewards</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">passed</span> <span class="o">/</span> <span class="nb">len</span><span class="p">(</span><span class="n">tests</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">rewards</span>
</code></pre></div>
</div>

<div class="admonition warning"><p class="admonition-title">基于规则 Reward 的陷阱</p>
<ul>
<li><p><strong>格式作弊（Format gaming）</strong>：模型学会生成正确格式但内容错误。务必将格式 Reward 与正确性 Reward 结合使用。</p></li>
<li><p><strong>测试用例泄漏</strong>：如果测试用例存在于训练数据中，模型会将其记住。</p></li>
<li><p><strong>超时利用</strong>：模型可能生成会超时的代码（以回避失败）。请使用严格的超时并显式惩罚超时。</p></li>
<li><p><strong>Reward 稀疏性</strong>：二元 Reward(0/1)对复杂任务而言可能过于稀疏。考虑使用部分得分或中间 Reward。</p></li>
</ul>
</div>

## 9.6　多目标 Reward —— 组合策略 {#sec:multi-objective-rewards}

<p>当使用多个 Reward 信号进行训练时，组合策略会显著影响最终 Policy。</p>

<div class="admonition note"><p class="admonition-title">多 Reward 组合策略</p>
<ol>
<li><p><strong>加权求和（Weighted sum）</strong>：<span class="arithmatex">\(r = \sum_n w_n r_n\)</span>。简单但对尺度敏感。</p></li>
<li><p><strong>先归一化再求和（GDPO）</strong>：在组内将每个 Reward 归一化为零均值、单位方差，再加权求和。尺度无关。</p></li>
<li><p><strong>字典序（Lexicographic）</strong>：按优先级顺序优化 Reward；仅当高优先级 Reward 平局时才考虑低优先级的 Reward。</p></li>
<li><p><strong>约束法（Constrained）</strong>：在次要 Reward 的约束下最大化主 Reward。</p></li>
<li><p><strong>Pareto</strong>：维护一组 Policy 的 Pareto 前沿，并根据偏好进行选择。</p></li>
</ol>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的多 Reward 训练</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">GRPOConfig</span><span class="p">,</span> <span class="n">GRPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">GRPOConfig</span><span class="p">(</span>
    <span class="c1"># GDPO:对每个 Reward 独立归一化</span>
    <span class="n">multi_objective_aggregation</span><span class="o">=</span><span class="s2">"normalize_then_sum"</span><span class="p">,</span>
    <span class="n">reward_weights</span><span class="o">=</span><span class="p">[</span><span class="mf">1.0</span><span class="p">,</span> <span class="mf">0.3</span><span class="p">,</span> <span class="mf">0.1</span><span class="p">],</span>  <span class="c1"># 正确性、格式、长度</span>
    <span class="n">num_generations</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">GRPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">reward_funcs</span><span class="o">=</span><span class="p">[</span>
        <span class="n">correctness_reward</span><span class="p">,</span>
        <span class="n">format_reward</span><span class="p">,</span>
        <span class="n">length_penalty_reward</span><span class="p">,</span>
    <span class="p">],</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

## 9.7　基于列表排序的 Reward {#listwise-rank-based-rewards}

<p>虽然 Bradley-Terry 模型处理的是<em>成对</em>偏好（<span class="arithmatex">\(y_w \succ y_l\)</span>），但许多实际场景需要同时对多个回答进行排序。列表式（listwise）奖励模型从完整排序中学习，提供更丰富的训练信号，并实现更好的校准。</p>

#### 动机：超越成对比较 {#motivation-beyond-pairwise}

<div class="admonition tip"><p class="admonition-title">为何要用 listwise?</p>
<ul>
<li><p><strong>更丰富的信号</strong>：对 <span class="arithmatex">\(K\)</span> 个回答的排序包含 <span class="arithmatex">\(\binom{K}{2}\)</span> 个隐式成对比较，同时还能捕捉<em>相对 margin</em>（例如排名第 1 比第 3 好多少）。</p></li>
<li><p><strong>更好的校准</strong>：成对 BT 模型只学习 Reward 的<em>差值</em>；listwise 模型学习的是绝对 Reward 尺度。</p></li>
<li><p><strong>天然契合 GRPO</strong>：GRPO 对每个 prompt 生成 <span class="arithmatex">\(N\)</span> 个回答并排序 —— listwise Reward 与该工作流直接对齐。</p></li>
<li><p><strong>标注效率</strong>：对 5 个回答排序比独立标注所有 10 个可能的成对组合更快。</p></li>
</ul>
</div>

#### Plackett-Luce 模型 {#plackett-luce-model}

<p>Plackett-Luce(PL)模型 <a class="book-citation" id="cite-plackett1975analysis-1" href="../../references.md#ref-plackett1975analysis" aria-label="参考文献 291">[291]</a> 是将 Bradley-Terry 推广至完整排序的标准扩展。给定 <span class="arithmatex">\(K\)</span> 个回答 <span class="arithmatex">\(y_1, \ldots, y_K\)</span> 及其排序 <span class="arithmatex">\(\pi\)</span>(其中 <span class="arithmatex">\(\pi(1)\)</span> 为最佳):</p>

<div class="admonition note"><p class="admonition-title">Plackett-Luce 似然</p>


<div class="arithmatex">\[P(\pi \mid q) = \prod_{i=1}^{K} \frac{e^{r_\phi(y_{\pi(i)}, q)}}{\sum_{j=i}^{K} e^{r_\phi(y_{\pi(j)}, q)}}\]</div>

<p><strong>直觉</strong>：顺序地从剩余项中选出最佳。每一步选中 <span class="arithmatex">\(\pi(i)\)</span> 的概率是对剩余项做 Softmax。</p>

<p><strong>Loss 函数</strong>：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{PL}}(\phi) = -\frac{1}{|\mathcal{D}|} \sum_{(q, \pi) \in \mathcal{D}} \sum_{i=1}^{K-1} \left[ r_\phi(y_{\pi(i)}, q) - \log \sum_{j=i}^{K} e^{r_\phi(y_{\pi(j)}, q)} \right]\]</div>
</div>

<div class="admonition tip"><p class="admonition-title">Plackett-Luce 退化为 Bradley-Terry</p>
<p>当 <span class="arithmatex">\(K=2\)</span> 时，PL 模型给出：<span class="arithmatex">\(P(y_1 \succ y_2) = \frac{e^{r(y_1)}}{e^{r(y_1)} + e^{r(y_2)}} = \sigma(r(y_1) - r(y_2))\)</span> —— 这正是 Bradley-Terry 模型。PL 是其严格的推广。</p>
</div>

#### ListMLE 与基于排序的 Loss {#listmle-and-rank-based-losses}

<div class="admonition note"><p class="admonition-title">列表式 Loss 函数</p>
<ul>
<li><p><strong>ListMLE</strong> <a class="book-citation" id="cite-xia2008listwise-1" href="../../references.md#ref-xia2008listwise" aria-label="参考文献 391">[391]</a>：直接最大化真实排序的 PL 似然。简洁有效。</p></li>
<li><p><strong>ListNet</strong> <a class="book-citation" id="cite-cao2007listnet-1" href="../../references.md#ref-cao2007listnet" aria-label="参考文献 38">[38]</a>：最小化模型的 top-1 概率分布与真实分布之间的 KL 散度：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{ListNet}} = -\sum_{i=1}^{K} P_{\text{true}}(y_i \text{ is best}) \cdot \log P_{\text{model}}(y_i \text{ is best})\]</div>

<p>其中 <span class="arithmatex">\(P_{\text{model}}(y_i \text{ is best}) = \frac{e^{r_\phi(y_i)}}{\sum_j e^{r_\phi(y_j)}}\)</span>。</p></li>
<li><p><strong>LambdaRank</strong> <a class="book-citation" id="cite-burges2006lambdarank-1" href="../../references.md#ref-burges2006lambdarank" aria-label="参考文献 34">[34]</a>：用排序指标（例如 NDCG）的变化对成对 Gradient 加权。当排名靠前的质量更重要时尤为有用。</p></li>
<li><p><strong>RankNet</strong> <a class="book-citation" id="cite-burges2005ranknet-1" href="../../references.md#ref-burges2005ranknet" aria-label="参考文献 33">[33]</a>：对所有成对组合求和的成对交叉熵 —— 等价于在从排序中抽取的所有 <span class="arithmatex">\(\binom{K}{2}\)</span> 个对上应用 BT。</p></li>
</ul>
</div>

#### GRPO 与拒绝采样中的 listwise Reward {#listwise-rewards-for-grpo-and-rejection-sampling}

<div class="admonition note"><p class="admonition-title">与 GRPO 的集成</p>
<p>GRPO 天然产生有序的分组：对每个 prompt,<span class="arithmatex">\(N\)</span> 个回答被打分并排序。可以直接在这些排序上训练 listwise 奖励模型：</p>

<ol>
<li><p><strong>生成</strong>：从 Policy 中对每个 prompt 采样 <span class="arithmatex">\(N=8\)</span> 个回答。</p></li>
<li><p><strong>排序</strong>：使用已有的奖励模型（或人工标注者）产生完整排序 <span class="arithmatex">\(\pi\)</span>。</p></li>
<li><p><strong>训练 listwise RM</strong>：在 <span class="arithmatex">\((q, \pi)\)</span> 元组上优化 PL Loss。</p></li>
<li><p><strong>在 GRPO 中使用</strong>：listwise RM 对每个回答分配标量 Reward <span class="arithmatex">\(r(y_i, q)\)</span>；GRPO 将优势计算为 <span class="arithmatex">\(\hat{A}_i = (r_i - \mu) / \sigma\)</span>。</p></li>
</ol>

<p><strong>相对成对方法的优势</strong>：listwise RM 同时看到全部 <span class="arithmatex">\(N\)</span> 个回答，可学到排名第 1 应当比排名第 <span class="arithmatex">\(N\)</span> 拥有高得多的 Reward（而非仅仅“比另一个回答略好”）。</p>
</div>

#### 实用考量 {#practical-considerations}

<div class="admonition warning"><p class="admonition-title">Listwise 训练挑战</p>
<ul>
<li><p><strong>标注成本</strong>：完整排序昂贵。部分排序（8 个中取 top-3）能在质量损失很小的前提下降低成本。</p></li>
<li><p><strong>并列</strong>：真实排序常有并列。使用 PL 的并列扩展：对并列项分配相等的概率质量。</p></li>
<li><p><strong>位置偏差</strong>：标注者倾向偏好排在前面的项。随机化展示顺序并训练去偏。</p></li>
<li><p><strong>列表长度</strong>：训练时 <span class="arithmatex">\(K=4\)</span>–8 较为常见。更长的列表（<span class="arithmatex">\(K&gt;16\)</span>）只会增加噪声，收益有限。</p></li>
<li><p><strong>一致性</strong>：不同标注者的排序可能不一致。使用标注者间一致性（<span class="arithmatex">\(\kappa &gt; 0.6\)</span>）作为质量过滤。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">Plackett-Luce 训练代码</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch.nn.functional</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">F</span>

<span class="k">def</span><span class="w"> </span><span class="nf">plackett_luce_loss</span><span class="p">(</span><span class="n">rewards</span><span class="p">,</span> <span class="n">rankings</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    Args:</span>
<span class="sd">        rewards: (batch, K) - 对 K 个回答预测的标量 Reward</span>
<span class="sd">        rankings: (batch, K) - 真实排序索引(0 = 最佳)</span>
<span class="sd">    Returns:</span>
<span class="sd">        标量 Loss</span>
<span class="sd">    """</span>
    <span class="n">batch_size</span><span class="p">,</span> <span class="n">K</span> <span class="o">=</span> <span class="n">rewards</span><span class="o">.</span><span class="n">shape</span>
    <span class="c1"># 按真实排序顺序对 Reward 排序</span>
    <span class="n">sorted_rewards</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="n">rewards</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">rankings</span><span class="p">)</span>  <span class="c1"># (batch, K)</span>

    <span class="c1"># PL 对数似然:对各位置求和</span>
    <span class="n">loss</span> <span class="o">=</span> <span class="mf">0.0</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">K</span> <span class="o">-</span> <span class="mi">1</span><span class="p">):</span>
        <span class="c1"># 对剩余项(位置 i 到 K)做 log-softmax</span>
        <span class="n">remaining</span> <span class="o">=</span> <span class="n">sorted_rewards</span><span class="p">[:,</span> <span class="n">i</span><span class="p">:]</span>           <span class="c1"># (batch, K-i)</span>
        <span class="n">log_probs</span> <span class="o">=</span> <span class="n">remaining</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">-</span> <span class="n">torch</span><span class="o">.</span><span class="n">logsumexp</span><span class="p">(</span><span class="n">remaining</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
        <span class="n">loss</span> <span class="o">-=</span> <span class="n">log_probs</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>

    <span class="k">return</span> <span class="n">loss</span> <span class="o">/</span> <span class="p">(</span><span class="n">K</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span>

<span class="c1"># 示例:每个 prompt 8 个回答,由标注者排序</span>
<span class="n">rewards</span> <span class="o">=</span> <span class="n">reward_model</span><span class="p">(</span><span class="n">responses</span><span class="p">)</span>          <span class="c1"># (batch, 8)</span>
<span class="n">rankings</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">argsort</span><span class="p">(</span><span class="n">human_scores</span><span class="p">,</span> <span class="n">descending</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>  <span class="c1"># 最佳在前</span>
<span class="n">loss</span> <span class="o">=</span> <span class="n">plackett_luce_loss</span><span class="p">(</span><span class="n">rewards</span><span class="p">,</span> <span class="n">rankings</span><span class="p">)</span>
<span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
</code></pre></div>
</div>
