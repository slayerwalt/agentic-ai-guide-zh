# 3.6　策略梯度方法——REINFORCE {#policy-gradient-methods-reinforce}

<p>与其学习价值函数再推导策略，不如直接优化策略参数 <span class="arithmatex">\(\theta\)</span> 以最大化期望回报 <a class="book-citation" id="cite-williams1992simple-1" href="../../references.md#ref-williams1992simple" aria-label="参考文献 379">[379]</a>。</p>

<p><strong>目标</strong>：<span class="arithmatex">\(J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T r_t\right]\)</span></p>

<p><strong>策略梯度定理（Policy Gradient Theorem）</strong>：</p>



<div class="arithmatex">\[\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]}
\tag{3.19}\]</div>

<div class="admonition note"><p class="admonition-title">策略梯度定理——形式化推导（5 步）</p>
<p><strong>第 1 步</strong>：定义目标。我们希望最大化期望回报：</p>



<div class="arithmatex">\[J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^T r_t\right] = \sum_\tau P(\tau|\theta) R(\tau)\]</div>

<p>其中 <span class="arithmatex">\(P(\tau|\theta) = p(s_0)\prod_{t=0}^T \pi_\theta(a_t|s_t)\, p(s_{t+1}|s_t, a_t)\)</span> 是轨迹概率。</p>

<p><strong>第 2 步</strong>：求梯度。只有 <span class="arithmatex">\(\pi_\theta\)</span> 项依赖于 <span class="arithmatex">\(\theta\)</span>（状态转移 <span class="arithmatex">\(p\)</span> 与 <span class="arithmatex">\(\theta\)</span> 无关）：</p>



<div class="arithmatex">\[\nabla_\theta J = \sum_\tau \nabla_\theta P(\tau|\theta)\, R(\tau)\]</div>

<p><strong>第 3 步</strong>：应用<strong>对数求导技巧（log-derivative trick）</strong>：<span class="arithmatex">\(\nabla_\theta P(\tau|\theta) = P(\tau|\theta)\, \nabla_\theta \log P(\tau|\theta)\)</span>：</p>



<div class="arithmatex">\[\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\nabla_\theta \log P(\tau|\theta)\, R(\tau)\right]\]</div>

<p><strong>第 4 步</strong>：展开 <span class="arithmatex">\(\log P(\tau|\theta)\)</span>。<span class="arithmatex">\(\log p(s_0)\)</span> 与 <span class="arithmatex">\(\log p(s_{t+1}|s_t,a_t)\)</span> 在 <span class="arithmatex">\(\nabla_\theta\)</span> 下消失：</p>



<div class="arithmatex">\[\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)\]</div>

<p><strong>第 5 步</strong>：合并。未来奖励不依赖于过去动作（因果性），因此每个 <span class="arithmatex">\(\nabla\log\pi\)</span> 仅与未来回报 <span class="arithmatex">\(G_t = \sum_{t&#x27;=t}^T r_{t&#x27;}\)</span> 配对：</p>



<div class="arithmatex">\[\boxed{\nabla_\theta J = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]}\]</div>
</div>

<div class="admonition tip"><p class="admonition-title">这一结果为何优美</p>
<p>该梯度<strong>不需要对环境的状态转移</strong> <span class="arithmatex">\(p(s&#x27;|s,a)\)</span> <strong>求导</strong>。对数求导技巧将其转化为一个期望，只需<em>运行策略并观察奖励</em>就能估计。把 <span class="arithmatex">\(G_t\)</span> 替换为优势 <span class="arithmatex">\(\hat{A}_t = G_t - V(s_t)\)</span> 可以在不引入偏差的前提下降低方差（因为对任何仅依赖状态的基线 <span class="arithmatex">\(b(s)\)</span>，都有 <span class="arithmatex">\(\mathbb{E}[\nabla\log\pi \cdot b(s)] = 0\)</span>）。</p>
</div>

<p><strong>REINFORCE 算法（REINFORCE）</strong> <a class="book-citation" id="cite-williams1992simple-2" href="../../references.md#ref-williams1992simple" aria-label="参考文献 379">[379]</a>（Williams, 1992）：</p>

<ol>
<li><p>在 <span class="arithmatex">\(\pi_\theta\)</span> 下采样完整轨迹 <span class="arithmatex">\(\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)\)</span></p></li>
<li><p>为每个时间步计算回报 <span class="arithmatex">\(G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}\)</span></p></li>
<li><p>更新：<span class="arithmatex">\(\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\)</span></p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">REINFORCE 直觉——「奖励加权的最大似然」</p>
<p><span class="arithmatex">\(\nabla_\theta \log \pi_\theta(a_t|s_t)\)</span> 是提升动作 <span class="arithmatex">\(a_t\)</span> 概率的方向。乘以 <span class="arithmatex">\(G_t\)</span> 意味着：</p>

<ul>
<li><p>高奖励轨迹：提升所有所采取动作的概率（<span class="arithmatex">\(G_t\)</span> 为正）</p></li>
<li><p>低奖励轨迹：降低所采取动作的概率（扣除基线后 <span class="arithmatex">\(G_t\)</span> 为负）</p></li>
</ul>

<p>这是一种监督学习，「标签」是你采取的动作，并按结果好坏加权。</p>
</div>

<p><strong>基线带来的方差缩减</strong>：</p>



<div class="arithmatex">\[\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]
\tag{3.20}\]</div>

<p>任何与 <span class="arithmatex">\(a_t\)</span> 无关的基线 <span class="arithmatex">\(b(s_t)\)</span> 都能在保持梯度无偏的同时降低方差。最佳选择：<span class="arithmatex">\(b(s_t) = V^\pi(s_t)\)</span>。此时 <span class="arithmatex">\(G_t - V(s_t) \approx A^\pi(s_t, a_t)\)</span> = 优势。</p>

<div class="admonition warning"><p class="admonition-title">REINFORCE 的局限</p>
<ul>
<li><p><strong>方差高</strong>：每次梯度仅使用一条轨迹，需要数千样本才能保证更新稳定。</p></li>
<li><p><strong>无自举</strong>：必须等待完整 episode（没有部分回报信号）。</p></li>
<li><p><strong>样本效率低</strong>：数据使用一次即丢弃（同策略）。</p></li>
<li><p><strong>缺乏步长控制</strong>：策略更新步可能灾难性地过大。</p></li>
</ul>

<p>这些局限催生了如下演进路径：REINFORCE <span class="arithmatex">\(\to\)</span> Actor-Critic <span class="arithmatex">\(\to\)</span> TRPO <span class="arithmatex">\(\to\)</span> <strong>PPO</strong>。</p>
</div>
