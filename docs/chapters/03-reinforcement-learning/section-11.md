# 3.11　奖励塑形（Reward Shaping） {#reward-shaping}

<p><strong>奖励塑形（Reward Shaping）</strong> <a class="book-citation" id="cite-ng1999policy-1" href="../../references.md#ref-ng1999policy" aria-label="参考文献 265">[265]</a>是一种由开发者修改或补充环境原始奖励函数的技术。其主要目的是将<strong>稀疏奖励</strong>场景（智能体仅在任务最终完成时才获得反馈）转换为带有中间反馈信号的<strong>稠密奖励</strong>场景，以加速收敛。</p>

## 3.11.1　数学框架 {#the-mathematical-framework}

<p>设时间步 <span class="arithmatex">\(t\)</span> 的原始奖励为 <span class="arithmatex">\(R_t(s, a, s&#x27;)\)</span>。重塑后的奖励添加了一个辅助塑形函数 <span class="arithmatex">\(F\)</span>：</p>



<div class="arithmatex">\[\boxed{R&#x27;_t(s, a, s&#x27;) = R_t(s, a, s&#x27;) + F(s, a, s&#x27;)}
\tag{3.26}\]</div>

<div class="admonition warning"><p class="admonition-title">朴素塑形的风险：奖励黑客（Reward Hacking）</p>
<p>如果 <span class="arithmatex">\(F(s, a, s&#x27;)\)</span> 被随意设计，智能体会找到结构性漏洞来最大化辅助信号，同时忽略全局目标。</p>

<p><strong>示例</strong>：一个因到达中间地标而获得奖励的导航智能体可能学会无限绕着某个检查点循环以累积无限奖励——而永远不会到达目的地。</p>

<p>在 LLM 中：若模型因「听起来自信」而获得奖励，它可能学会无论准确与否都以「Absolutely!」开头。</p>
</div>

## 3.11.2　基于势函数的奖励塑形（Potential-Based Reward Shaping，PBRS） {#potential-based-reward-shaping-pbrs}

<p>为了从数学上保证塑形<strong>不会</strong>改变最优策略，可使用<strong>基于势函数的奖励塑形（PBRS）</strong>。塑形函数 <span class="arithmatex">\(F\)</span> 被约束为跨状态的标量势函数 <span class="arithmatex">\(\Phi\)</span> 之差：</p>



<div class="arithmatex">\[\boxed{F(s, a, s&#x27;) = \gamma\, \Phi(s&#x27;) - \Phi(s)}
\tag{3.27}\]</div>

<p>其中 <span class="arithmatex">\(\Phi: \mathcal{S} \to \mathbb{R}\)</span> 是一个实值势函数，衡量某状态相对目标的接近程度，<span class="arithmatex">\(\gamma\)</span> 是折扣因子。</p>

<p>完整的 PBRS 奖励：</p>



<div class="arithmatex">\[R&#x27;(s, a, s&#x27;) = R(s, a, s&#x27;) + \gamma\, \Phi(s&#x27;) - \Phi(s)
\tag{3.28}\]</div>

## 3.11.3　理论保证 {#theoretical-guarantees}

<div class="admonition note"><p class="admonition-title">PBRS 策略不变性定理</p>
<ul>
<li><p><strong>策略不变性</strong>：在重塑奖励 <span class="arithmatex">\(R&#x27;\)</span> 下的最优策略 <span class="arithmatex">\(\pi^*\)</span> 与原始奖励 <span class="arithmatex">\(R\)</span> 下的最优策略<strong>完全相同</strong>。塑形不会引入次优行为。</p></li>
<li><p><strong>回路免疫</strong>：任何从某状态出发又回到同一状态的循环轨迹，其净势能变化恰好为零（<span class="arithmatex">\(\Phi(s) - \Phi(s) = 0\)</span>）。智能体无法利用回路来作弊式刷奖励。</p></li>
<li><p><strong>收敛加速</strong>：尽管最优策略不变，塑形后的奖励提供了更稠密的梯度信号，使智能体在稀疏奖励环境下收敛速度提升 5–50 倍。</p></li>
</ul>
</div>
