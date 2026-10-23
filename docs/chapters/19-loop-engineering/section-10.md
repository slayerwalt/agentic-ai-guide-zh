# 19.10　长时程循环中的上下文管理 {#sec:loop-context-management}

<p>长时程循环面临一个根本张力：上下文窗口是 Agent 的工作记忆，但容量固定。每一步都会追加新信息（思考、观测、错误），没有主动管理，窗口就会填满、质量下降。本节介绍跨大量迭代维持上下文健康的工程技术。</p>

## 19.10.1　压缩策略 {#sec:compaction-strategies}

<p class="table-caption" id="table-19-3"><span class="table-number">表 19.3　</span>长时程循环的上下文压缩策略。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>策略</strong></th><th scope="col"><strong>机制</strong></th><th scope="col"><strong>权衡</strong></th></tr></thead>
<tbody><tr><td>周期摘要</td><td>每 <span class="arithmatex">\(k\)</span> 步用摘要替换原始历史</td><td>有损；可能丢弃之后需要的细节</td></tr>
<tr><td>滑动窗口</td><td>逐字保留最近 <span class="arithmatex">\(w\)</span> 步；更早的摘要化</td><td>偏重新近；可能丢失早期上下文</td></tr>
<tr><td>重要性加权</td><td>保留改变状态的步骤；丢弃无操作</td><td>需要定义“重要性”</td></tr>
<tr><td>外部记忆</td><td>细节写入文件；上下文只留引用</td><td>需要显式回读；增加延迟</td></tr>
<tr><td>子 Agent 隔离</td><td>子任务在新上下文中运行；只返回结论</td><td>生成开销；协调成本</td></tr></tbody>
</table>
</div>

<p>最优策略取决于任务。对调试循环，滑动窗口效果很好——最近的错误最相关。对研究循环，重要性加权保留能跨大量迭代保存关键实验结果。对复杂的多文件重构，外部记忆（写一个总结迄今所有改动的进度文件）能防止 Agent 丢失对自己修改的追踪。</p>

## 19.10.2　上下文预算 {#sec:context-budget}

<p>一个实用的启发式：为当前步骤的推理与生成预留上下文窗口的 20–30%。这意味着有效历史不应超过容量的 70–80%。当接近这个阈值时，在下一轮迭代<em>之前</em>触发压缩——而不是等模型已经产出劣化输出之后。</p>

<div class="highlight"><pre><span></span><code><span class="n">CONTEXT_CAPACITY</span> <span class="o">=</span> <span class="mi">128_000</span>  <span class="c1"># token 总数</span>
<span class="n">RESERVED_FOR_GENERATION</span> <span class="o">=</span> <span class="mf">0.25</span> <span class="o">*</span> <span class="n">CONTEXT_CAPACITY</span>
<span class="n">ACTIVE_BUDGET</span> <span class="o">=</span> <span class="n">CONTEXT_CAPACITY</span> <span class="o">-</span> <span class="n">RESERVED_FOR_GENERATION</span>

<span class="k">def</span><span class="w"> </span><span class="nf">should_compact</span><span class="p">(</span><span class="n">state</span><span class="p">:</span> <span class="n">LoopState</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">bool</span><span class="p">:</span>
    <span class="k">return</span> <span class="n">state</span><span class="o">.</span><span class="n">token_count</span> <span class="o">&gt;</span> <span class="n">ACTIVE_BUDGET</span> <span class="o">*</span> <span class="mf">0.85</span>
</code></pre></div>
