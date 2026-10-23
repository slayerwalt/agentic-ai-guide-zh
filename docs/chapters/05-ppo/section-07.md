# 5.7　细节机制：Logits 与 Policy 更新 {#detailed-mechanics-logits-and-policy-updates}

<figure id="figure-5-2"><a href="../../assets/figures/fig_026_fig26.png" target="_blank" rel="noopener" aria-label="查看图 5.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_026_fig26.png" alt="PPO 端到端流程：从 prompt batch 出发，经过生成、reward 打分、KL 计算、advantage 估计，到 clipped policy 更新。反馈环显示更新后的 policy 又被用于下一次生成。" loading="lazy"></a><figcaption><span class="figure-number">图 5.2　</span>PPO 端到端流程：从 prompt batch 出发，经过生成、reward 打分、KL 计算、advantage 估计，到 clipped policy 更新。反馈环显示更新后的 policy 又被用于下一次生成。</figcaption></figure>

<p>PPO 在内存中维护两份不同的参数状态，它们共享同一神经网络拓扑，但在优化过程中持有不同的权重值：</p>

<div class="admonition note"><p class="admonition-title">核心架构：两个网络</p>
<ol>
<li><p><strong>Policy 网络（<span class="arithmatex">\(\pi_\theta\)</span>）：</strong>由权重 <span class="arithmatex">\(\theta\)</span> 参数化的、处于活动状态的实时网络。优化过程中通过反向传播持续更新。</p></li>
<li><p><strong>旧 Policy 网络（<span class="arithmatex">\(\pi_{\theta_{\text{old}}}\)</span>）：</strong>由权重 <span class="arithmatex">\(\theta_{\text{old}}\)</span> 参数化的冻结快照。在单个优化周期内充当静态锚，防止 policy 漂移过快。</p></li>
</ol>
</div>

## 5.7.1　阶段 1：Rollout（数据收集） {#phase-1-rollout-data-collection}

<p>在数据收集期间，Agent 与环境交互 <span class="arithmatex">\(T\)</span> 步。在每个时间步 <span class="arithmatex">\(t\)</span>：</p>

<ol>
<li><p>环境给出当前 state/观测 <span class="arithmatex">\(s_t\)</span>（对大语言模型而言：prompt + 到目前为止已生成的 token）。</p></li>
<li><p><span class="arithmatex">\(s_t\)</span> 被送入当前网络快照（<span class="arithmatex">\(\theta_{\text{old}}\)</span>）。</p></li>
<li><p>网络输出原始未归一化的值——<strong>logits</strong> <span class="arithmatex">\(z_{\text{old}}\)</span>——一个长度为 <span class="arithmatex">\(|V|\)</span>（词表大小 32K–128K）的向量。</p></li>
<li><p>通过 Softmax 计算概率：</p>



<div class="arithmatex">\[\boxed{P(a \mid s_t) = \text{Softmax}(z_{\text{old}}) = \frac{\exp(z_{\text{old}, a})}{\sum_{j=1}^{|V|} \exp(z_{\text{old}, j})}}
\tag{5.14}\]</div></li>
<li><p>从 <span class="arithmatex">\(P(a \mid s_t)\)</span> 中采样一个 action <span class="arithmatex">\(a_t\)</span>（下一个 token），并将 transition 元组 <span class="arithmatex">\(\langle s_t, a_t, r_t, s_{t+1} \rangle\)</span> 连同 <span class="arithmatex">\(\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\)</span> 一起存入 rollout buffer。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">为何要存对数概率？</p>
<p>在 rollout 阶段把 <span class="arithmatex">\(\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\)</span> 作为标量保存，可以避免在优化期间重新运行冻结的网络。每个 mini-batch 因此可省下一次完整 forward pass——对 70B 模型而言意义重大。</p>
</div>

## 5.7.2　阶段 2：优化循环（Mini-Batch 更新） {#phase-2-optimization-loop-mini-batch-updates}

<p>一旦 rollout buffer 填满，PPO 就在 mini-batch 上跑 <span class="arithmatex">\(K\)</span> 个 epoch（通常 3–10）。在每个 gradient 步中，使用所存 state <span class="arithmatex">\(s_t\)</span> 为两个 policy 同时生成 logits：</p>

<p><strong>旧 Policy 评估</strong>（冻结）：</p>



<div class="arithmatex">\[z_{\text{old}} = f(s_t; \theta_{\text{old}}) \quad \longrightarrow \quad \log \pi_{\theta_{\text{old}}}(a_t \mid s_t) = \text{LogSoftmax}(z_{\text{old}})[a_t]
\tag{5.15}\]</div>

<p><em>实现捷径：复用 rollout 时保存的标量，而非重新计算。</em></p>

<p><strong>实时 Policy 评估</strong>（更新中）：</p>



<div class="arithmatex">\[z_{\text{new}} = f(s_t; \theta) \quad \longrightarrow \quad \log \pi_\theta(a_t \mid s_t) = \text{LogSoftmax}(z_{\text{new}})[a_t]
\tag{5.16}\]</div>

<p>由于每次 mini-batch gradient 步后 <span class="arithmatex">\(\theta\)</span> 都会更新，<span class="arithmatex">\(z_{\text{new}}\)</span> 会在整个优化循环中持续变化，而 <span class="arithmatex">\(z_{\text{old}}\)</span> 则始终保持不变。</p>

## 5.7.3　从 Logits 到概率比 {#from-logits-to-probability-ratio}

<p>PPO 的核心比率衡量某个 action 在新 policy 下相对旧 policy 的概率变化：</p>



<div class="arithmatex">\[\boxed{r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}}
\tag{5.17}\]</div>

<p>为了避免直接除原始概率带来的数值下溢/上溢灾难，计算在<strong>对数空间</strong>中进行：</p>



<div class="arithmatex">\[\begin{align*}
\log \pi_\theta(a_t \mid s_t) &amp;= \text{LogSoftmax}(z_{\text{new}})[a_t] \tag{5.18}\\

\log \pi_{\theta_{\text{old}}}(a_t \mid s_t) &amp;= \text{LogSoftmax}(z_{\text{old}})[a_t]
\tag{5.19}
\end{align*}\]</div>

<p>比率由差值的指数恢复：</p>



<div class="arithmatex">\[\boxed{r_t(\theta) = \exp\!\left(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\right)}
\tag{5.20}\]</div>

<p>该比率随后被代入 PPO 的 clipping 目标：</p>



<div class="arithmatex">\[\boxed{\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\!\left(r_t(\theta)\hat{A}_t, \;\text{clip}(r_t(\theta),\, 1{-}\epsilon,\, 1{+}\epsilon)\,\hat{A}_t\right) \right]}
\tag{5.21}\]</div>

<div class="admonition tip"><p class="admonition-title">Clipping 的工作方式</p>
<ul>
<li><p>若 <span class="arithmatex">\(\hat{A}_t &gt; 0\)</span>（好 action）：比率被 clip 至 <span class="arithmatex">\(1+\epsilon\)</span>——不会对好 action 过度利用。</p></li>
<li><p>若 <span class="arithmatex">\(\hat{A}_t &lt; 0\)</span>（坏 action）：比率被 clip 至 <span class="arithmatex">\(1-\epsilon\)</span>——不会对坏 action 过度惩罚。</p></li>
<li><p><span class="arithmatex">\(\min(\cdot)\)</span> 保证我们始终取更保守的估计。</p></li>
</ul>

<p>结果：在一个信任域内实现单调改进——不会出现灾难性崩溃。</p>
</div>

## 5.7.4　PPO 权重生命周期 {#the-ppo-weight-lifecycle}

<p class="table-caption" id="table-5-1"><span class="table-number">表 5.1　</span><span class="arithmatex">\(\theta\)</span> 与 <span class="arithmatex">\(\theta_{\text{old}}\)</span> 在 PPO 训练各阶段的演化。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>阶段</strong></th><th scope="col"><strong>实时 <span class="arithmatex">\(\theta\)</span></strong></th><th scope="col"><strong>旧 <span class="arithmatex">\(\theta_{\text{old}}\)</span></strong></th><th scope="col"><strong>比率 <span class="arithmatex">\(r_t(\theta)\)</span></strong></th></tr></thead>
<tbody><tr><td>1. Rollout 起始</td><td>活动副本</td><td>同一活动副本</td><td>恒为 <span class="arithmatex">\(1.0\)</span>（按定义）</td></tr>
<tr><td>2. Batch 第 1 步</td><td>计算 gradient</td><td>冻结</td><td><span class="arithmatex">\(1.0\)</span>（初始步）</td></tr>
<tr><td>3. Batch 第 <span class="arithmatex">\(N\)</span> 步</td><td>正在修改（<span class="arithmatex">\(\theta \neq \theta_{\text{old}}\)</span>）</td><td>冻结</td><td>偏离 <span class="arithmatex">\(1.0\)</span>（如 <span class="arithmatex">\(1.06\)</span>、<span class="arithmatex">\(0.94\)</span>）</td></tr>
<tr><td>4. Clipping 生效</td><td>受 <span class="arithmatex">\(\epsilon\)</span> 限制</td><td>冻结</td><td>被锁在边界（<span class="arithmatex">\(1 \pm \epsilon\)</span>）</td></tr>
<tr><td>5. 优化结束</td><td>高度优化后</td><td>被丢弃</td><td>不适用</td></tr>
<tr><td>6. 下一周期</td><td><span class="arithmatex">\(\theta \rightarrow \theta_{\text{old}}\)</span></td><td>接收最新的 <span class="arithmatex">\(\theta\)</span></td><td>重置回 <span class="arithmatex">\(1.0\)</span></td></tr></tbody>
</table>
</div>

## 5.7.5　连续 Action 空间扩展 {#continuous-action-spaces-extension}

<p>对于连续 action 空间（在大语言模型中不常见，但对机器人 RL 很重要），网络输出的是分布参数而非离散 logits：</p>

<ul>
<li><p>预测的均值向量 <span class="arithmatex">\(\mu\)</span></p></li>
<li><p>预测的标准差向量 <span class="arithmatex">\(\sigma\)</span></p></li>
</ul>

<p>对数概率由高斯对数 PDF 计算：</p>



<div class="arithmatex">\[\boxed{\log \pi(a_t \mid s_t) = -\frac{1}{2}\left(\frac{a_t - \mu}{\sigma}\right)^{\!2} - \log(\sigma) - \frac{1}{2}\log(2\pi)}
\tag{5.22}\]</div>

<p>比率 <span class="arithmatex">\(r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})\)</span> 的计算方式完全相同，并被代入同一个 clipping 目标。</p>
