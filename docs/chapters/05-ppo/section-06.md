# 5.6　用于 RLHF 的 PPO：完整循环 {#ppo-for-rlhf-the-full-loop}

<figure id="figure-5-1"><a href="../../assets/figures/fig_025_fig25.png" target="_blank" rel="noopener" aria-label="查看图 5.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_025_fig25.png" alt="" loading="lazy"></a><figcaption><span class="figure-number">图 5.1　</span></figcaption></figure>

<div class="admonition example"><p class="admonition-title">70B Chat 模型的具体 PPO 步骤</p>
<p><strong>设置</strong>：128 条 prompt 一个 batch，policy 为 Llama-3-70B，最大生成 512 个 token。</p>

<p><strong>第 1 步——生成</strong>：采样 128 条回答（temperature=0.7，top-p=0.9）。耗时占 60%。</p>

<p><strong>第 2 步——打分</strong>：Reward 模型对每个 (prompt, response) 对打分。取值范围：0.2–0.95。</p>

<p><strong>第 3 步——KL</strong>：计算逐 token 的 KL：<span class="arithmatex">\(\text{KL}_t = \log\pi_\theta(y_t|y_{&lt;t}) - \log\pi_\text{ref}(y_t|y_{&lt;t})\)</span>。跨 token 的均值 KL：通常 3–8。</p>

<p><strong>第 4 步——最终 reward</strong>：<span class="arithmatex">\(R = r_\text{RM} - 0.05 \times \text{mean\_KL}\)</span>（只在最后一个 token 给）。</p>

<p><strong>第 5 步——GAE</strong>：使用 value head 的预测，为每个 token 位置计算 <span class="arithmatex">\(\hat{A}_t\)</span>。对 advantage 做 whitening（零均值、单位方差）。</p>

<p><strong>第 6 步——更新</strong>：在大小为 16 的 mini-batch 上跑 4 个 epoch 的 SGD。Clip 比率 <span class="arithmatex">\(\epsilon = 0.2\)</span>，梯度范数 clip 至 1.0。</p>

<p><strong>结果</strong>：Policy 每步约提升 0.005 的胜率。10K 步后：相对于 SFT 绝对提升 5–10%。</p>
</div>

<div class="admonition warning"><p class="admonition-title">大语言模型 RL 中的 Tokenization 陷阱</p>
<p>在计算逐 token KL 惩罚与 advantage 时，请记住：tokenization 决定了什么算一“步”。单个概念性 action（例如输出“2024”）依据 tokenizer 的不同可能跨越 1–4 个 token。这带来若干微妙问题：</p>

<ul>
<li><p><strong>KL 记账</strong>：相同的语义内容若被切成不同数量的 token，逐 token KL 加总会得出不同总额（例如，被切成更多子词的稀有词会获得更高的总 KL 惩罚）。</p></li>
<li><p><strong>Credit 分配</strong>：GAE 按 token 位置分配 advantage——但语义“决策”常跨越多个 token。模型其实只在词首 token 真正“决定”；后续子词 token 基本是确定性的。</p></li>
<li><p><strong>Reward 放置</strong>：只在最后一个 token 给 reward 意味着所有先前 token 必须通过 GAE 向后回传 credit——越长的回答信号被稀释得越厉害。</p></li>
</ul>

<p><strong>缓解措施</strong>：一些系统按序列长度对 KL 做归一化，使用词级 reward shaping，或在语义边界（而非最末 token）处施加 reward。</p>
</div>
