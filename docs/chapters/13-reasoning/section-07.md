# 13.7　推理的扩展律 {#subsec:reasoning_scaling}

<p>近期工作 <a class="book-citation" id="cite-snell2024scaling-1" href="../../references.md#ref-snell2024scaling" aria-label="参考文献 335">[335]</a><a class="book-citation" id="cite-wu2024empirical-1" href="../../references.md#ref-wu2024empirical" aria-label="参考文献 388">[388]</a> 已经证实测试时计算量与推理性能之间存在可预测的扩展关系，将经典扩展律 <a class="book-citation" id="cite-kaplan2020scaling-1" href="../../references.md#ref-kaplan2020scaling" aria-label="参考文献 177">[177]</a> 延伸到了推理阶段。</p>

## 13.7.1　训练计算量与测试时计算量的权衡 {#training-compute-vs.-test-time-compute-tradeoff}

<p>推理模型的根本性扩展问题是：<strong>给定固定的总计算预算 <span class="arithmatex">\(C_{\text{total}} = C_{\text{train}} + N \cdot C_{\text{test}}\)</span>（其中 <span class="arithmatex">\(N\)</span> 为查询数量），应如何分配计算？</strong></p>

<p>设 <span class="arithmatex">\(\mathcal{A}(C_{\text{train}}, C_{\text{test}})\)</span> 表示用 <span class="arithmatex">\(C_{\text{train}}\)</span> FLOPs 训练、每个查询给定 <span class="arithmatex">\(C_{\text{test}}\)</span> 推理 FLOPs 的模型准确率。经验上：</p>



<div class="arithmatex">\[\mathcal{A}(C_{\text{train}}, C_{\text{test}}) \approx 1 - \exp\!\left(-a \cdot C_{\text{train}}^{\alpha} \cdot C_{\text{test}}^{\beta}\right)
\tag{13.46}\]</div>

<p>其中 <span class="arithmatex">\(a, \alpha, \beta &gt; 0\)</span> 为常数。对于固定总预算 <span class="arithmatex">\(C_{\text{total}}\)</span>，最优分配满足训练与推理之间每 FLOP 的边际回报相等的条件：</p>



<div class="arithmatex">\[\frac{\partial \mathcal{A}}{\partial C_{\text{train}}} = \frac{1}{N} \cdot \frac{\partial \mathcal{A}}{\partial C_{\text{test}}}
\tag{13.47}\]</div>

<p>直觉上：一个 FLOP 的训练计算惠及全部 <span class="arithmatex">\(N\)</span> 个查询，而一个 FLOP 的测试时计算只惠及单个查询。在最优点，测试时计算的单查询边际价值是训练计算的 <span class="arithmatex">\(N\)</span> 倍（因为训练成本被摊销）。将此应用于公式 [eq:reasoning_scaling_law] 可得到最优训练计算占比：</p>



<div class="arithmatex">\[\frac{C_{\text{train}}^*}{C_{\text{total}}} = \frac{\alpha}{\alpha + \beta}
\tag{13.48}\]</div>

<p>对于特定预算结构 <span class="arithmatex">\(C_{\text{total}} = C_{\text{train}} + N \cdot C_{\text{test}}\)</span>，在乘性准确率模型下该占比与 <span class="arithmatex">\(N\)</span> 无关。然而在实际中 <span class="arithmatex">\(\alpha\)</span> 和 <span class="arithmatex">\(\beta\)</span> 是问题相关的：对于高流量部署（大 <span class="arithmatex">\(N\)</span>），即便基础模型的微小提升也会占主导，倾向于训练投入；对于低流量、高价值的查询（小 <span class="arithmatex">\(N\)</span>），测试时计算则更具成本效益。</p>

## 13.7.2　何时投资于更长推理链 vs. 更好的基础模型 {#when-to-invest-in-longer-chains-vs.-better-base-models}

<div class="admonition note"><p class="admonition-title">推理链长度与模型容量</p>
<p>对于容量为 <span class="arithmatex">\(C\)</span> 的模型在难度为 <span class="arithmatex">\(D\)</span> 的问题上，最优推理链长度 <span class="arithmatex">\(L^*\)</span> 满足：</p>



<div class="arithmatex">\[L^* \propto \frac{D}{C^{\gamma}}
\tag{13.49}\]</div>

<p>其中 <span class="arithmatex">\(\gamma &gt; 0\)</span>。这意味着：</p>

<ul>
<li><p><strong>困难问题</strong>无论模型规模如何都需要更长的推理链</p></li>
<li><p>在相同问题难度下，<strong>更大的模型</strong>需要更短的推理链</p></li>
<li><p><strong>收益递减</strong>：超过 <span class="arithmatex">\(L^*\)</span> 后，额外的 Token 不再带来收益，反而可能有害（过度思考）</p></li>
</ul>
</div>

<p>“过度思考（overthinking）”现象——拥有极长推理链的模型反而比拥有适中推理链的模型表现<em>更差</em>——已被经验观察到，归因于：</p>

<ul>
<li><p>长推理链中错误的累积（误差传播）</p></li>
<li><p>偏离主解路径的干扰</p></li>
<li><p>对错误中间结论的过度自信</p></li>
</ul>

## 13.7.3　最优 Token 预算分配 {#optimal-token-budget-allocation}

<p>对于给定 Token 预算 <span class="arithmatex">\(B\)</span> 的模型，“思考” Token <span class="arithmatex">\(T_{\text{think}}\)</span> 与“作答” Token <span class="arithmatex">\(T_{\text{answer}}\)</span> 之间的分配应满足：</p>



<div class="arithmatex">\[T_{\text{think}}^* = \arg\max_{T} \mathcal{A}(T, B - T)
\tag{13.50}\]</div>

<p>经验上，最优划分是问题相关的：</p>

<ul>
<li><p><strong>简单问题</strong>：<span class="arithmatex">\(T_{\text{think}}^* / B \approx 0.3\)</span>（30% 思考）</p></li>
<li><p><strong>困难问题</strong>：<span class="arithmatex">\(T_{\text{think}}^* / B \approx 0.8\)</span>（80% 思考）</p></li>
<li><p><strong>极困难问题</strong>：<span class="arithmatex">\(T_{\text{think}}^* / B \approx 0.95\)</span>（95% 思考，极简作答）</p></li>
</ul>

<p>这促成了<strong>自适应思考预算（adaptive thinking budgets）</strong>：为更难的问题分配更多 Token，难度可由模型在初次解题尝试中的不确定度来估计。</p>
