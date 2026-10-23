# 3.8　广义优势估计（Generalized Advantage Estimation，GAE） {#generalized-advantage-estimation-gae}

<p><strong>动机</strong>：Actor-Critic 框架需要对优势 <span class="arithmatex">\(A(s,a) = Q(s,a) - V(s)\)</span> 进行良好估计——这个动作比平均水平好多少？但这里存在根本性的张力：</p>

<ul>
<li><p><strong>1 步 TD 优势</strong>（<span class="arithmatex">\(r_t + \gamma V(s_{t+1}) - V(s_t)\)</span>）：方差低（仅含一步随机性），但<strong>有偏</strong>——若价值函数 <span class="arithmatex">\(V\)</span> 不准确，优势估计就会系统性偏离。</p></li>
<li><p><strong>Monte Carlo 优势</strong>（<span class="arithmatex">\(G_t - V(s_t)\)</span>）：无偏（使用实际回报），但<strong>方差高</strong>——许多随机奖励之和在不同 episode 间波动剧烈。</p></li>
</ul>

<p>GAE <a class="book-citation" id="cite-schulman2016high-1" href="../../references.md#ref-schulman2016high" aria-label="参考文献 318">[318]</a>（Schulman 等，2016）通过单一参数 <span class="arithmatex">\(\lambda \in [0, 1]\)</span> 在两种极端之间提供<strong>平滑插值</strong>。它对所有 <span class="arithmatex">\(n\)</span> 的 <span class="arithmatex">\(n\)</span> 步优势估计进行指数加权平均，给出一种在偏差与方差之间权衡的原则化方式。</p>

<p><strong>核心思想</strong>：在每个时间步计算 1 步 TD 误差 <span class="arithmatex">\(\delta_t\)</span>，然后用指数衰减权重 <span class="arithmatex">\((\gamma\lambda)^l\)</span> 进行混合——近期 TD 误差权重完整，远期的被降权：</p>



<div class="arithmatex">\[\boxed{\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)}
\tag{3.23}\]</div>

<figure id="figure-3-3"><a href="../../assets/figures/fig_022_fig22.png" target="_blank" rel="noopener" aria-label="查看图 3.3 原图" title="点击查看原图"><img src="../../assets/figures/fig_022_fig22.png" alt="GAE 数据流：每个 TD 残差 \(\delta_{t+l}^V\) 在求和前被乘以权重 \((\gamma\lambda)^l\)。\(\lambda\) 越大，纳入的未来残差越多（偏差更低，方差更高）。" loading="lazy"></a><figcaption><span class="figure-number">图 3.3　</span>GAE 数据流：每个 TD 残差 <span class="arithmatex">\(\delta_{t+l}^V\)</span> 在求和前被乘以权重 <span class="arithmatex">\((\gamma\lambda)^l\)</span>。<span class="arithmatex">\(\lambda\)</span> 越大，纳入的未来残差越多（偏差更低，方差更高）。</figcaption></figure>

<div class="admonition tip"><p class="admonition-title"><span class="arithmatex">\(\lambda\)</span> 控制什么——偏差-方差权衡</p>
<ul>
<li><p><span class="arithmatex">\(\lambda = 0\)</span>：<span class="arithmatex">\(\hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\)</span>。完全信任价值函数。方差低，但若 <span class="arithmatex">\(V\)</span> 不准确则有偏。</p></li>
<li><p><span class="arithmatex">\(\lambda = 1\)</span>：<span class="arithmatex">\(\hat{A}_t = \sum_l \gamma^l r_{t+l} - V(s_t)\)</span>。完整 Monte Carlo 回报减去基线。无偏但方差极高。</p></li>
<li><p><span class="arithmatex">\(\lambda = 0.95\)</span>（标准取值）：黄金平衡点。主要信任 <span class="arithmatex">\(V\)</span>，但在远期效应上用实际回报修正。之所以可行，是因为价值头在初始训练后会变得准确。</p></li>
</ul>

<p>针对 LLM 的常用取值：<span class="arithmatex">\(\gamma = 1.0\)</span>（无时间折扣——单轮对话中所有 token 权重相等），<span class="arithmatex">\(\lambda = 0.95\)</span>。</p>
</div>

## 3.8.1　GAE 中偏差与方差的直观映射 {#intuitive-mapping-of-bias-and-variance-in-gae}

<p>在监督学习中，偏差与方差源于模型的结构性假设。而在通过 GAE 进行的强化学习中，它们源于<strong>你信任有缺陷的模型有多深 vs 你信任混乱的环境有多深</strong>：</p>

<ul>
<li><p><strong>偏差（系统性偏离）</strong>：当估计器依赖于价值网络 <span class="arithmatex">\(V_\theta\)</span> 的结构性假设与不完美预测时产生。如果 <span class="arithmatex">\(\theta\)</span> 训练不足或容量不够，基线猜测就会系统性出错。</p></li>
<li><p><strong>方差（样本抖动）</strong>：当估计器依赖于长且无约束的环境轨迹时产生。随机转移、随机种子与策略执行噪声在长时间步上累积，导致不同 rollout 之间的实际样本奖励剧烈波动。</p></li>
</ul>

## 3.8.2　架构光谱：边界情形分析 {#the-architectural-spectrum-boundary-analysis}

<figure id="figure-3-4"><a href="../../assets/figures/fig_023_fig23.png" target="_blank" rel="noopener" aria-label="查看图 3.4 原图" title="点击查看原图"><img src="../../assets/figures/fig_023_fig23.png" alt="GAE 中的偏差 vs 方差：\(\lambda\) 控制这一权衡。较小的 \(\lambda\)（左）通过自举得到高偏差/低方差；较大的 \(\lambda\)（右）使用完整 Monte Carlo 回报得到低偏差/高方差。最优选择（\(\lambda \in [0.9, 0.95]\)）在稳定训练与准确的长程信用分配之间取得平衡。" loading="lazy"></a><figcaption><span class="figure-number">图 3.4　</span>GAE 中的偏差 vs 方差：<span class="arithmatex">\(\lambda\)</span> 控制这一权衡。较小的 <span class="arithmatex">\(\lambda\)</span>（左）通过自举得到高偏差/低方差；较大的 <span class="arithmatex">\(\lambda\)</span>（右）使用完整 Monte Carlo 回报得到低偏差/高方差。最优选择（<span class="arithmatex">\(\lambda \in [0.9, 0.95]\)</span>）在稳定训练与准确的长程信用分配之间取得平衡。</figcaption></figure>

<p>超参数 <span class="arithmatex">\(\lambda\)</span> 充当了两种基本估计范式之间的「滑尺」。</p>

<div class="admonition note"><p class="admonition-title">高偏差/低方差的限制（<span class="arithmatex">\(\lambda = 0\)</span>）</p>


<div class="arithmatex">\[\hat{A}_t^{\text{GAE}(\gamma, 0)} = \delta_t^V = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)
\tag{3.24}\]</div>

<ul>
<li><p><strong>行为</strong>：优势在很大程度上由参数 <span class="arithmatex">\(\theta\)</span> 的当前状态决定。</p></li>
<li><p><strong>直觉</strong>：高度<strong>有偏</strong>，因为网络在 1 步窗口内为自己打分；若 <span class="arithmatex">\(V_\theta\)</span> 不准确，梯度步就会被破坏。<strong>方差低</strong>，因为它忽略 <span class="arithmatex">\(t+1\)</span> 步之后的未来随机事件，从而产生平滑、稳定的参数更新。</p></li>
<li><p><strong>风险</strong>：策略陷入次优局部极小——永远发现不了复杂的延迟奖励序列。</p></li>
</ul>
</div>

<div class="admonition note"><p class="admonition-title">低偏差/高方差的限制（<span class="arithmatex">\(\lambda = 1\)</span>）</p>
<p>当 <span class="arithmatex">\(\lambda = 1\)</span> 时，中间的价值项相互裂项相消，GAE 退化为 Monte Carlo 回报减去基线：</p>



<div class="arithmatex">\[\hat{A}_t^{\text{GAE}(\gamma, 1)} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V_\theta(s_t)
\tag{3.25}\]</div>

<ul>
<li><p><strong>行为</strong>：舍弃自举式前瞻，将整个 episode 的实际现实进行累加。</p></li>
<li><p><strong>直觉</strong>：对真实的环境转移完全<strong>无偏</strong>——衡量的是实际奖励而非神经网络近似。然而表现出极端的<strong>高方差</strong>：episode 早期的微小扰动可能导致完全不同的总回报，使策略更新变得不稳定。</p></li>
<li><p><strong>风险</strong>：破坏性梯度更新；训练爆炸。</p></li>
</ul>
</div>

## 3.8.3　权衡矩阵 {#the-trade-off-matrix}

<p>通过选择 <span class="arithmatex">\(\lambda \in [0.95, 0.99]\)</span>，GAE 可以最小化优势估计的总均方误差：</p>

<p class="table-caption" id="table-3-1"><span class="table-number">表 3.1　</span>GAE 参数选择的操作性对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>配置</strong></th><th scope="col"><strong>统计特性</strong></th><th scope="col"><strong>核心依赖</strong></th><th scope="col"><strong>实际风险</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(\lambda = 0\)</span></td><td>高偏差、低方差</td><td>模型参数（<span class="arithmatex">\(\theta\)</span>）</td><td>策略陷入次优局部极小</td></tr>
<tr><td><span class="arithmatex">\(\lambda \in [0.95, 0.99]\)</span></td><td>均衡（最优 MSE）</td><td>混合融合</td><td>需要根据环境随机性进行调参</td></tr>
<tr><td><span class="arithmatex">\(\lambda = 1\)</span></td><td>低偏差、高方差</td><td>经验性环境 rollout</td><td>破坏性梯度更新；训练爆炸</td></tr></tbody>
</table>
</div>

## 3.8.4　<span class="arithmatex">\(\lambda\)</span> 调参诊断 {#diagnostics-for-tuning-lambda}

<p>监控训练曲线能直接揭示当前主导因素是偏差还是方差：</p>

<ol>
<li><p><strong>高方差指标</strong>：策略熵骤降，同时价值函数的解释方差变得高度为负或剧烈波动 <span class="arithmatex">\(\rightarrow\)</span> 策略更新噪声大。<strong>对策</strong>：降低 <span class="arithmatex">\(\lambda\)</span> 以平滑目标更新。</p></li>
<li><p><strong>高偏差指标</strong>：智能体在早期获得稳定训练，但完全无法发现复杂的延迟奖励序列 <span class="arithmatex">\(\rightarrow\)</span> 由于自举导致对长程依赖的低估。<strong>对策</strong>：将 <span class="arithmatex">\(\lambda\)</span> 提高至接近 <span class="arithmatex">\(1.0\)</span>，让策略接触真实的下游轨迹信号。</p></li>
</ol>
