# 19.12　循环的经济学 {#sec:loop-economics}

<p>循环用算力（token、墙钟时间）换取质量。理解这一权衡对生产部署至关重要。</p>

<div class="admonition note"><p class="admonition-title">循环成本模型</p>
<p>对于一个平均 <span class="arithmatex">\(N\)</span> 轮迭代、每轮消耗 <span class="arithmatex">\(T\)</span> 个 token、单价为每 token <span class="arithmatex">\(p\)</span> 的循环：</p>



<div class="arithmatex">\[\text{Cost}_{\text{loop}} = N \cdot T \cdot p + \text{Cost}_{\text{tools}} + \text{Cost}_{\text{verification}}\]</div>

<p>使用提示缓存（系统提示与早期上下文被缓存，按较低的 <span class="arithmatex">\(p_c &lt; p\)</span> 计费）时：</p>



<div class="arithmatex">\[\text{Cost}_{\text{cached}} = T_{\text{new}} \cdot p + T_{\text{cached}} \cdot p_c + \text{(tools + verification)}\]</div>

<p>提示缓存对循环尤其有价值，因为目标定义、技能内容与早期历史跨迭代保持不变——每次调用通常有 60–80% 的上下文可缓存。</p>
</div>

<p>经济决策：循环产出的价值（节省的开发者时间、通宵生产力、质量提升）是否大于它的 token 成本？一个运行 20 轮、每轮 $0.02 的循环总成本是 $0.40——如果能省下开发者 30 分钟，这微不足道。但没有预算上限的失控循环，一夜之间就能烧掉数百美元。</p>
