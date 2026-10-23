# 6.5　DPO 的完整机制 {#how-dpo-works-full-mechanics}

<p>本节给出 DPO 的完整计算细节——训练时在 token 级别究竟发生了什么。</p>

## 6.5.1　序列级对数概率 {#sequence-level-log-probabilities}

<p>DPO 中的关键量是：在给定 prompt <span class="arithmatex">\(x\)</span> 下，<strong>整段序列</strong> <span class="arithmatex">\(y = (y_1, y_2, \ldots, y_T)\)</span> 的对数概率。它是<strong>逐 token 对数概率之和</strong>：</p>



<div class="arithmatex">\[\boxed{\log \pi_\theta(y|x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{&lt;t})}
\tag{6.3}\]</div>

<p>其中每一项 <span class="arithmatex">\(\log \pi_\theta(y_t | x, y_{&lt;t})\)</span> 都是位置 <span class="arithmatex">\(t\)</span> 处的 log-softmax 输出，对应序列中<em>实际</em>出现的 token <span class="arithmatex">\(y_t\)</span>。这与标准语言建模中的交叉熵 loss 完全一致——只是这里我们<strong>求和</strong>而非求均值。</p>

<p><strong>关键细节</strong>：Gradient 会流过 <span class="arithmatex">\(y_w\)</span> 和 <span class="arithmatex">\(y_l\)</span> 中的<strong>每一个 token 位置</strong>。中间 token 不做掩码——每个 token 都对序列级对数概率有贡献。</p>

## 6.5.2　DPO Loss 的分解 {#the-dpo-loss-decomposed}

<p>从 loss 开始：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\beta \cdot h_\theta(x, y_w, y_l)\right)\right]
\tag{6.4}\]</div>

<p>其中“隐式 reward margin” <span class="arithmatex">\(h_\theta\)</span> 为：</p>



<div class="arithmatex">\[h_\theta(x, y_w, y_l) = \underbrace{\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}}_{\text{chosen reward proxy}} - \underbrace{\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}}_{\text{rejected reward proxy}}
\tag{6.5}\]</div>

<p>展开为 token 级各项：</p>



<div class="arithmatex">\[\boxed{h_\theta = \sum_{t=1}^{|y_w|}\!\left[\log\pi_\theta(y_w^t | x, y_w^{&lt;t}) - \log\pi_{\text{ref}}(y_w^t | x, y_w^{&lt;t})\right] - \sum_{t=1}^{|y_l|}\!\left[\log\pi_\theta(y_l^t | x, y_l^{&lt;t}) - \log\pi_{\text{ref}}(y_l^t | x, y_l^{&lt;t})\right]}
\tag{6.6}\]</div>

## 6.5.3　Forward Pass：逐步详解 {#forward-pass-step-by-step}

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

## 6.5.4　Token 级 Gradient 分析 {#token-level-gradient-analysis}

<p><strong>是不是每个 token 都收到 gradient？</strong> 是。在被选序列中，位置 <span class="arithmatex">\(t\)</span> 处 logits 上的 gradient 为：</p>



<div class="arithmatex">\[\frac{\partial \mathcal{L}}{\partial \text{logits}_t^{(w)}} = -\underbrace{\sigma(-\beta \cdot h_\theta)}_{\text{scaling factor}} \cdot \beta \cdot \frac{\partial \log\pi_\theta(y_w^t | \cdot)}{\partial \text{logits}_t^{(w)}}
\tag{6.11}\]</div>

<p><strong>关键洞察</strong>：缩放因子 <span class="arithmatex">\(\sigma(-\beta \cdot h_\theta)\)</span> 在两条序列的<strong>所有 token 间共享</strong>，相当于一个自适应学习率：</p>

<ul>
<li><p>当 <span class="arithmatex">\(h_\theta\)</span> 较小（模型分不清被选与被拒）：缩放 <span class="arithmatex">\(\approx 0.5\)</span>——gradient 强，激进地学。</p></li>
<li><p>当 <span class="arithmatex">\(h_\theta\)</span> 较大（模型已经偏好被选）：缩放 <span class="arithmatex">\(\approx 0\)</span>——gradient 可忽略，避免过拟合。</p></li>
</ul>

<p><strong>对被选 token 的影响</strong>：概率被<em>提升</em>（对数概率被推高）。<br><strong>对被拒 token 的影响</strong>：概率被<em>降低</em>（对数概率被压低）。<br><strong>相对 reference</strong>：只有相对 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 的<em>差值</em>才重要。如果模型本就给被选回答高概率（与 reference 一致），那么 gradient 就几乎为零。</p>

## 6.5.5　逐 Token vs. 序列级：长度归一化 {#per-token-vs.-sequence-level-length-normalization}

<p>一个微妙问题：越长的序列对数概率天然越低（求和的项更多，且每项 <span class="arithmatex">\(\leq 0\)</span>）。若 <span class="arithmatex">\(|y_w| \gg |y_l|\)</span>，loss 会偏向于偏好更短的回答。</p>

<p><strong>解决方案</strong>：</p>

<ul>
<li><p><strong>长度归一化的 DPO</strong>：用 <span class="arithmatex">\(\frac{1}{|y|}\sum_t \log\pi_\theta(y_t|\cdot)\)</span> 替换 <span class="arithmatex">\(\log\pi_\theta(y|x)\)</span>。一些实现采用此做法（SimPO 即如此）。</p></li>
<li><p><strong>标准 DPO</strong>：使用原始求和（不归一化）。这会<em>隐式</em>惩罚冗长——模型必须对被选回答中的每个 token 都赋予高概率。</p></li>
<li><p><strong>实际影响</strong>：在 benchmark 上，长度归一化的 DPO 能减少长度博弈，但可能损害指令遵循质量。生产中更常用未归一化的标准版本。</p></li>
</ul>

## 6.5.6　标签掩码：哪些 token 会收到 Gradient {#label-masking-what-gets-gradients}

<div class="admonition note"><p class="admonition-title">DPO 中哪些 token 收到 Gradient</p>
<ul>
<li><p><strong>Prompt token</strong>（<span class="arithmatex">\(x\)</span>）：<strong>无 gradient</strong>。Loss 仅在回答位置上计算。Prompt token 提供上下文，但其 logits 不参与 <span class="arithmatex">\(\log\pi(y|x)\)</span>。</p></li>
<li><p><strong>被选回答 token</strong>（<span class="arithmatex">\(y_w\)</span>）：<strong>所有 token 都收到 gradient</strong>。每个 <span class="arithmatex">\(y_w^t\)</span> 都贡献到求和；gradient 推高它们的概率。</p></li>
<li><p><strong>被拒回答 token</strong>（<span class="arithmatex">\(y_l\)</span>）：<strong>所有 token 都收到 gradient</strong>。每个 <span class="arithmatex">\(y_l^t\)</span> 都贡献到求和；gradient 压低它们的概率。</p></li>
<li><p><strong>Padding token</strong>：<strong>无 gradient</strong>。通过 attention mask 屏蔽掉。</p></li>
</ul>
</div>

## 6.5.7　伪代码：DPO 训练步 {#pseudocode-dpo-training-step}

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

## 6.5.8　常见陷阱 {#common-pitfalls}

<div class="admonition warning"><p class="admonition-title">DPO 实现中的陷阱</p>
<ul>
<li><p><strong>忘记屏蔽 prompt</strong>：如果 prompt token 被纳入对数概率求和，模型就会优化 prompt 的似然（无意义），且有效 <span class="arithmatex">\(\beta\)</span> 也会出错。</p></li>
<li><p><strong>用均值替代求和</strong>：<span class="arithmatex">\(\frac{1}{T}\sum_t \log\pi\)</span> 与 <span class="arithmatex">\(\sum_t \log\pi\)</span> 会带来不同的隐式长度惩罚。<span class="arithmatex">\(\pi_\theta\)</span> 与 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 之间必须保持一致。</p></li>
<li><p><strong>过期的参考模型</strong>：若 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 与 <span class="arithmatex">\(\pi_\theta\)</span> 相距过远（如 base 模型 vs. 微调后的模型），KL 项会占主导，gradient 消失。解决办法：使用 SFT checkpoint（而非 base）作为 reference。</p></li>
<li><p><strong><span class="arithmatex">\(\beta\)</span> 过大</strong>：放大对数概率差值 <span class="arithmatex">\(\rightarrow\)</span> sigmoid 饱和 <span class="arithmatex">\(\rightarrow\)</span> gradient 归零。从 <span class="arithmatex">\(\beta = 0.1\)</span> 起步，在 <span class="arithmatex">\([0.05, 0.5]\)</span> 区间内调参。</p></li>
<li><p><strong><span class="arithmatex">\(\beta\)</span> 过小</strong>：理论上允许 policy 更大程度地偏离 reference（KL 约束更弱），但 gradient <span class="arithmatex">\(\propto \beta \cdot \sigma(-\beta h)\)</span> 会变得极小 <span class="arithmatex">\(\rightarrow\)</span> loss 地形平坦 <span class="arithmatex">\(\rightarrow\)</span> 收敛极慢。模型“被允许”走得很远，却几乎收不到任何告诉它<em>往哪走</em>的信号。</p></li>
</ul>
</div>
