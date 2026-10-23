# 13.5　QwQ 与 Qwen 推理模型 {#subsec:qwq_qwen}

<p>阿里巴巴 Qwen 团队开发了一系列推理模型（QwQ-32B <a class="book-citation" id="cite-qwen2024qwq-1" href="../../references.md#ref-qwen2024qwq" aria-label="参考文献 351">[351]</a>、Qwen3 <a class="book-citation" id="cite-qwen2025qwen3-1" href="../../references.md#ref-qwen2025qwen3" aria-label="参考文献 352">[352]</a>），与 DeepSeek-R1 一同代表了开源前沿。其方法在若干关键方面有所不同。</p>

## 13.5.1　多阶段 RL 流水线 {#multi-stage-rl-pipeline}

<p>Qwen 的推理流水线采用了更精细的多阶段方法：</p>

<ol>
<li><p><strong>基础预训练</strong>：具备强大数学与编码能力的 Qwen2.5 基础模型</p></li>
<li><p><strong>多样推理任务上的 SFT</strong>：在涵盖广泛推理任务（数学、代码、科学、逻辑）的混合数据上进行监督微调（Supervised Fine-Tuning, SFT）</p></li>
<li><p><strong>拒绝采样微调（Rejection Sampling Fine-Tuning, RFT）</strong>：对每个问题生成 <span class="arithmatex">\(N\)</span> 个解，保留正确的解，再进行微调</p></li>
<li><p><strong>RL 第一阶段</strong>：在数学和代码上使用可验证奖励的组相对策略优化（Group Relative Policy Optimization, GRPO）</p></li>
<li><p><strong>RL 第二阶段</strong>：更广泛的 RL，包括指令遵循与安全</p></li>
</ol>

## 13.5.2　拒绝采样与 RL 的结合 {#rejection-sampling-rl-combination}

<p>Qwen 方法的一项关键创新是<strong>拒绝采样与 RL 的迭代结合</strong>：</p>

<ol>
<li><p><strong>初始化</strong>：以 SFT 模型作为 Policy <span class="arithmatex">\(\pi_0\)</span>。</p></li>
<li><p><strong>拒绝采样（Rejection Sampling）</strong>：采样 <span class="arithmatex">\(N\)</span> 个解 <span class="arithmatex">\(\{y_i\}_{i=1}^N \sim \pi_{k-1}(\cdot \mid q)\)</span>，保留正确的解 <span class="arithmatex">\(\mathcal{Y}^+(q) = \{y_i : r(y_i, y^*) = 1\}\)</span>。</p></li>
<li><p><strong>SFT 更新</strong>：<span class="arithmatex">\(\pi_k^{\text{SFT}} \leftarrow \text{SFT}(\pi_{k-1}, \bigcup_q \mathcal{Y}^+(q))\)</span></p></li>
<li><p><strong>RL 更新</strong>：<span class="arithmatex">\(\pi_k \leftarrow \text{GRPO}(\pi_k^{\text{SFT}}, \mathcal{D})\)</span></p></li>
<li><p><strong>重复</strong>步骤 2–4 共 <span class="arithmatex">\(K\)</span> 次迭代，得到最终 Policy <span class="arithmatex">\(\pi_K\)</span>。</p></li>
</ol>

<p>拒绝采样步骤提供高质量的正样本来锚定 Policy，而 RL 则在当前分布之外进行探索。这种组合比纯 RL 更稳定，又比纯 SFT 更有能力。</p>

## 13.5.3　工具集成推理 {#tool-integrated-reasoning}

<p>QwQ-32B 与 Qwen3 模型支持<strong>工具集成推理（tool-integrated reasoning）</strong>：模型可以在其推理链中调用外部工具（Python 解释器、搜索引擎、计算器）。这通过特殊 Token 实现：</p>

<p class="code-caption"><span class="code-number">代码 13.1　</span>QwQ 中的工具集成推理格式</p>
<div class="highlight"><pre><span></span><code><span class="o">&lt;</span><span class="n">think</span><span class="o">&gt;</span>
<span class="n">Let</span> <span class="n">me</span> <span class="n">solve</span> <span class="n">this</span> <span class="n">step</span> <span class="n">by</span> <span class="n">step</span><span class="o">.</span>
<span class="n">First</span><span class="p">,</span> <span class="n">I</span><span class="s1">'ll compute the eigenvalues of the matrix.</span>

<span class="o">&lt;</span><span class="n">tool_call</span><span class="o">&gt;</span>
<span class="p">{</span><span class="s2">"name"</span><span class="p">:</span> <span class="s2">"python"</span><span class="p">,</span> <span class="s2">"arguments"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"code"</span><span class="p">:</span> <span class="s2">"import numpy as np</span><span class="se">\n</span><span class="s2">A = np.array([[2,1],[1,3]])</span><span class="se">\n</span><span class="s2">eigenvalues = np.linalg.eigvals(A)</span><span class="se">\n</span><span class="s2">print(eigenvalues)"</span><span class="p">}}</span>
<span class="o">&lt;/</span><span class="n">tool_call</span><span class="o">&gt;</span>

<span class="o">&lt;</span><span class="n">tool_response</span><span class="o">&gt;</span>
<span class="p">[</span><span class="mf">1.38196601</span> <span class="mf">3.61803399</span><span class="p">]</span>
<span class="o">&lt;/</span><span class="n">tool_response</span><span class="o">&gt;</span>

<span class="n">The</span> <span class="n">eigenvalues</span> <span class="n">are</span> <span class="n">approximately</span> <span class="mf">1.382</span> <span class="ow">and</span> <span class="mf">3.618</span><span class="o">.</span>
<span class="n">These</span> <span class="n">are</span> <span class="p">(</span><span class="mi">5</span> <span class="o">+/-</span> <span class="n">sqrt</span><span class="p">(</span><span class="mi">5</span><span class="p">))</span><span class="o">/</span><span class="mi">2</span><span class="p">,</span> <span class="n">which</span> <span class="n">are</span> <span class="n">the</span> <span class="n">golden</span> <span class="n">ratio</span> <span class="ow">and</span> <span class="n">its</span> <span class="n">conjugate</span><span class="o">...</span>
<span class="o">&lt;/</span><span class="n">think</span><span class="o">&gt;</span>
<span class="o">&lt;</span><span class="n">answer</span><span class="o">&gt;</span><span class="n">The</span> <span class="n">eigenvalues</span> <span class="n">are</span> <span class="p">(</span><span class="mi">5</span> <span class="o">+/-</span> <span class="n">sqrt</span><span class="p">(</span><span class="mi">5</span><span class="p">))</span><span class="o">/</span><span class="mi">2</span><span class="o">&lt;/</span><span class="n">answer</span><span class="o">&gt;</span>
</code></pre></div>

<p>RL 训练奖励基于最终答案计算，但模型会学会策略性地使用工具，因为工具调用能提升得到正确答案的概率。</p>
