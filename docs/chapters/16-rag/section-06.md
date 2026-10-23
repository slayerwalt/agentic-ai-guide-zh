# 16.6　高效 RAG 解码：REFRAG {#efficient-rag-decoding-refrag}

<p>RAG 的一个实际瓶颈是<em>解码延迟</em>：拼接到 LLM 上下文中的检索段落往往很长但相关性稀疏，从而拉高首 Token 时延（Time-to-First-Token, TTFT）和 KV-cache 内存。REFRAG <a class="book-citation" id="cite-lin2025refrag-1" href="../../references.md#ref-lin2025refrag" aria-label="参考文献 220">[220]</a> 观察到，由于检索到的段落是独立来源的（通过重排序中的多样性或去重），它们的 Attention 模式呈<em>块对角</em>——大多数跨段落 Attention 接近于零。这种稀疏性意味着解码阶段对 RAG 上下文的大部分计算其实是不必要的。</p>

#### 压缩—感知—展开（Compress–Sense–Expand）框架。 {#compresssenseexpand-framework.}

<p>REFRAG 通过三阶段解码策略利用这种结构：</p>

<ol>
<li><p><strong>压缩</strong>：用紧凑的摘要（如对每个段落块的 Key/Value 进行均值池化）替换检索段落的完整 KV 表示，大幅降低内存占用。</p></li>
<li><p><strong>感知</strong>：在每个解码步骤上，对压缩后的表示执行轻量级 Attention，识别哪些段落块与当前 Token 相关。</p></li>
<li><p><strong>展开</strong>：仅为被选中的块重建完整的 KV 项，对稀疏的活跃集合执行精确 Attention。</p></li>
</ol>

#### 结果。 {#results.-dup3}

<p>在基于 LLaMA 的模型上，REFRAG 实现了最高 <span class="arithmatex">\(30.85\times\)</span> 的 TTFT 加速（比此前的稀疏 Attention 基线提升 <span class="arithmatex">\(3.75\times\)</span>），且困惑度无损。在固定内存预算下，它还将有效上下文长度扩展了 <span class="arithmatex">\(16\times\)</span>。这些收益在 RAG、多轮对话以及长文档摘要任务中均成立。</p>

<div class="admonition tip"><p class="admonition-title">REFRAG 为何对智能体 RAG 至关重要</p>
<p>智能体 RAG（Section <a href="section-07.md#sec:agentic_rag">16.7</a>）需要为每个查询进行<em>多轮</em>检索，延迟会累积。REFRAG 这类高效解码方法是必备的基础设施：通过确保每一轮的解码代价相对上下文长度是次线性的，它们使迭代式的检索—推理—生成循环在大规模下具备可行性。</p>
</div>
