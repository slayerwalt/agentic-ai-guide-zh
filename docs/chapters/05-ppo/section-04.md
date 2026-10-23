# 5.4　PPO Gradient 与更新规则的推导 {#derivation-of-the-ppo-gradient-and-update-rule}

<p>本节追溯从 RL 目标到 PPO 更新规则的数学路径，说明 clipped surrogate <em>为何</em>有效。</p>

## 5.4.1　第 1 步：RL 目标 {#step-1-the-rl-objective}

<p>目标是在 policy 下最大化期望累计 reward：</p>



<div class="arithmatex">\[J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T r_t\right]
\tag{5.3}\]</div>

## 5.4.2　第 2 步：Policy Gradient 定理 {#step-2-policy-gradient-theorem}

<p><span class="arithmatex">\(J(\theta)\)</span> 关于 policy 参数的 gradient：</p>



<div class="arithmatex">\[\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right]}
\tag{5.4}\]</div>

<p>其中 <span class="arithmatex">\(\hat{A}_t\)</span> 是优势函数（即在 state <span class="arithmatex">\(s_t\)</span> 下 action <span class="arithmatex">\(a_t\)</span> 相对平均 action 的好坏程度）。用 advantage 替代完整 return 是为了降低方差。</p>

## 5.4.3　第 3 步：离策略数据的重要性采样 {#step-3-importance-sampling-for-off-policy-data}

<p>PPO 使用 <span class="arithmatex">\(\pi_{\theta_{\text{old}}}\)</span> 收集数据，却更新 <span class="arithmatex">\(\pi_\theta\)</span>。为修正这种分布失配，需应用重要性采样：</p>



<div class="arithmatex">\[\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right]
\tag{5.5}\]</div>

<p>定义概率比 <span class="arithmatex">\(r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\)</span>。利用恒等式 <span class="arithmatex">\(\nabla_\theta \log f = \frac{\nabla_\theta f}{f}\)</span>，可得：</p>



<div class="arithmatex">\[\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[\nabla_\theta\, r_t(\theta) \cdot \hat{A}_t\right]
\tag{5.6}\]</div>

<p>这意味着最大化 <strong>surrogate 目标</strong>：</p>



<div class="arithmatex">\[L^{\text{CPI}}(\theta) = \mathbb{E}_t\left[r_t(\theta) \cdot \hat{A}_t\right]
\tag{5.7}\]</div>

## 5.4.4　第 4 步：无约束 Surrogate 的问题 {#step-4-the-problem-with-unconstrained-surrogates}

<p><span class="arithmatex">\(L^{\text{CPI}}\)</span> 是一个合法的目标，但若无约束，单次 gradient 步就可能让 <span class="arithmatex">\(r_t(\theta)\)</span> 远离 1.0，导致：</p>

<ul>
<li><p>重要性权重变得极端 <span class="arithmatex">\(\rightarrow\)</span> 方差升高</p></li>
<li><p>Policy 进入未经检验的区域 <span class="arithmatex">\(\rightarrow\)</span> reward 模型给出不可靠的分数</p></li>
<li><p>灾难性崩溃：policy 生成垃圾，且无法恢复</p></li>
</ul>

<p><strong>TRPO 方案</strong>：约束 <span class="arithmatex">\(D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \leq \delta\)</span>。需要二阶方法（成本高）。</p>

## 5.4.5　第 5 步：PPO 的 Clipped Surrogate（一阶近似） {#step-5-ppos-clipped-surrogate-first-order-approximation}

<p>PPO 用一个 <strong>clipped 目标</strong>取代硬性 KL 约束，仅使用一阶 gradient 即可达到类似行为：</p>



<div class="arithmatex">\[\boxed{L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\hat{A}_t\right)\right]}
\tag{5.8}\]</div>

<p><strong>Gradient 的推导</strong>：</p>

<p>令 <span class="arithmatex">\(L_t = \min(r_t \hat{A}_t,\; \bar{r}_t \hat{A}_t)\)</span>，其中 <span class="arithmatex">\(\bar{r}_t = \text{clip}(r_t, 1{-}\epsilon, 1{+}\epsilon)\)</span>。</p>



<div class="arithmatex">\[\nabla_\theta L_t = \begin{cases}
\nabla_\theta r_t(\theta) \cdot \hat{A}_t &amp; \text{if } r_t \hat{A}_t &lt; \bar{r}_t \hat{A}_t \text{ (unclipped term is smaller)} \\
0 &amp; \text{if } r_t \hat{A}_t \geq \bar{r}_t \hat{A}_t \text{ (clipped term is smaller, gradient = 0)}
\end{cases}
\tag{5.9}\]</div>

<p>展开各条件：</p>

<ul>
<li><p><strong>当 <span class="arithmatex">\(\hat{A}_t &gt; 0\)</span> 且 <span class="arithmatex">\(r_t &lt; 1+\epsilon\)</span></strong>：Gradient 正常传播——policy 被鼓励提升 <span class="arithmatex">\(\pi_\theta(a_t|s_t)\)</span>。</p></li>
<li><p><strong>当 <span class="arithmatex">\(\hat{A}_t &gt; 0\)</span> 且 <span class="arithmatex">\(r_t \geq 1+\epsilon\)</span></strong>：Gradient 为<strong>零</strong>——policy 提升已足够，停止继续推。</p></li>
<li><p><strong>当 <span class="arithmatex">\(\hat{A}_t &lt; 0\)</span> 且 <span class="arithmatex">\(r_t &gt; 1-\epsilon\)</span></strong>：Gradient 正常传播——policy 被鼓励降低 <span class="arithmatex">\(\pi_\theta(a_t|s_t)\)</span>。</p></li>
<li><p><strong>当 <span class="arithmatex">\(\hat{A}_t &lt; 0\)</span> 且 <span class="arithmatex">\(r_t \leq 1-\epsilon\)</span></strong>：Gradient 为<strong>零</strong>——policy 降低已足够，停止继续推。</p></li>
</ul>

## 5.4.6　第 6 步：完整的 PPO 更新规则 {#step-6-the-complete-ppo-update-rule}

<p>将 clipped policy loss、value loss 与熵奖励组合起来：</p>



<div class="arithmatex">\[\boxed{\theta_{k+1} = \theta_k + \alpha \cdot \nabla_\theta \left[L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 H[\pi_\theta]\right]}
\tag{5.10}\]</div>

<p>其中：</p>



<div class="arithmatex">\[\begin{align*}
L^{\text{VF}}(\theta) &amp;= \left(V_\theta(s_t) - V_t^{\text{target}}\right)^2 &amp; &amp;\text{(value function regression loss)} \tag{5.11}\\

H[\pi_\theta] &amp;= -\sum_a \pi_\theta(a|s_t)\log\pi_\theta(a|s_t) &amp; &amp;\text{(entropy of the policy)}
\tag{5.12}
\end{align*}\]</div>

<div class="admonition tip"><p class="admonition-title">小结：它为何有效</p>
<ol>
<li><p><strong>Policy gradient 定理</strong>给出了改进 policy 的方向。</p></li>
<li><p><strong>重要性采样</strong>让我们能在多个 epoch 中复用来自 <span class="arithmatex">\(\pi_{\theta_{\text{old}}}\)</span> 的数据。</p></li>
<li><p><strong>Clipping</strong> 防止重要性权重变得极端，保持更新安全。</p></li>
<li><p><strong><span class="arithmatex">\(\min\)</span> 算子</strong>保证我们始终在 (clipped, unclipped) 中取更保守的一项——对改进设置悲观下界。</p></li>
<li><p><strong>结果</strong>：以概率 1 实现单调改进，且只用一阶 gradient。无需 Hessian、共轭梯度或线搜索。</p></li>
</ol>
</div>
