# 16.11　各种 RAG 方法的综合对比 {#comprehensive-rag-approach-comparison}

<p class="table-caption" id="table-16-7"><span class="table-number">表 16.7　</span>各种 RAG 方法在关键维度上的对比</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>准确率</strong></th><th scope="col"><strong>延迟</strong></th><th scope="col"><strong>复杂度</strong></th><th scope="col"><strong>成本</strong></th><th scope="col"><strong>最佳适用场景</strong></th></tr></thead>
<tbody><tr><td>朴素 RAG <a class="book-citation" id="cite-lewis2020retrieval-1" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a></td><td>中等</td><td>低</td><td>低</td><td>低</td><td>原型、简单问答</td></tr>
<tr><td>RAG + 重排序 <a class="book-citation" id="cite-nogueira2019passage-1" href="../../references.md#ref-nogueira2019passage" aria-label="参考文献 268">[268]</a></td><td>高</td><td>中等</td><td>中等</td><td>中等</td><td>生产级问答系统</td></tr>
<tr><td>HyDE <a class="book-citation" id="cite-gao2022precise-1" href="../../references.md#ref-gao2022precise" aria-label="参考文献 106">[106]</a></td><td>高</td><td>中等</td><td>低</td><td>中等</td><td>语义不匹配的领域</td></tr>
<tr><td>多查询 RAG</td><td>高</td><td>中等</td><td>中等</td><td>中等</td><td>歧义查询</td></tr>
<tr><td>RAG-Fusion <a class="book-citation" id="cite-rackauckas2023ragfusion-1" href="../../references.md#ref-rackauckas2023ragfusion" aria-label="参考文献 300">[300]</a></td><td>高</td><td>中等</td><td>中等</td><td>中等</td><td>多样化的查询类型</td></tr>
<tr><td>Self-RAG <a class="book-citation" id="cite-asai2023selfrag-1" href="../../references.md#ref-asai2023selfrag" aria-label="参考文献 13">[13]</a></td><td>高</td><td>中等</td><td>高</td><td>中等</td><td>选择性检索</td></tr>
<tr><td>CRAG <a class="book-citation" id="cite-yan2024crag-1" href="../../references.md#ref-yan2024crag" aria-label="参考文献 401">[401]</a></td><td>高</td><td>中等</td><td>高</td><td>高</td><td>不可靠的语料库</td></tr>
<tr><td>自适应 RAG <a class="book-citation" id="cite-jeong2024adaptive-1" href="../../references.md#ref-jeong2024adaptive" aria-label="参考文献 162">[162]</a></td><td>高</td><td>低–高</td><td>高</td><td>中等</td><td>查询复杂度混合</td></tr>
<tr><td>Graph RAG <a class="book-citation" id="cite-edge2024local-1" href="../../references.md#ref-edge2024local" aria-label="参考文献 82">[82]</a></td><td>极高</td><td>高</td><td>极高</td><td>高</td><td>全局综合查询</td></tr>
<tr><td>智能体 RAG</td><td>极高</td><td>高</td><td>极高</td><td>高</td><td>多跳推理</td></tr>
<tr><td>RAFT <a class="book-citation" id="cite-zhang2024raft-1" href="../../references.md#ref-zhang2024raft" aria-label="参考文献 425">[425]</a></td><td>极高</td><td>低</td><td>极高</td><td>极高</td><td>特定领域部署</td></tr></tbody>
</table>
</div>

<div class="admonition question"><p class="admonition-title">RAG 系统的关键设计问题</p>
<p>在为生产环境设计 RAG 系统时，请考虑以下问题：</p>

<ol>
<li><p><strong>查询分布是怎样的？</strong>事实型 vs. 分析型 vs. 多跳查询需要不同的检索策略。</p></li>
<li><p><strong>语料库有多大、变化多频繁？</strong>数百万文档且频繁更新更适合具备增量索引的托管向量数据库。</p></li>
<li><p><strong>延迟需求如何？</strong>低于 100ms 的响应排除了重排序和智能体循环；批处理或异步场景则可以承受。</p></li>
<li><p><strong>接地有多关键？</strong>高风险领域（医疗、法律、金融）需要忠实度评估与引用验证。</p></li>
<li><p><strong>词汇是否专业化？</strong>特定领域术语可能需要混合检索或经过领域适配的 Embedding 模型。</p></li>
</ol>
</div>

<div class="admonition note"><p class="admonition-title">RAG 最佳实践总结</p>
<ul>
<li><p><strong>从简单开始</strong>：分块得当的朴素 RAG 往往优于分块糟糕的复杂系统</p></li>
<li><p><strong>独立评估检索</strong>：先修好检索，再优化生成</p></li>
<li><p><strong>使用混合检索</strong>：BM25 + 稠密检索 + RRF 是一个强大的默认方案</p></li>
<li><p><strong>加上重排序</strong>：在 top-20 候选上使用交叉编码器重排序器具有很高的 ROI</p></li>
<li><p><strong>监控忠实度</strong>：在生产中使用 LLM 判官跟踪幻觉率</p></li>
<li><p><strong>积极缓存</strong>：文档只 Embedding 一次；缓存频繁查询的 Embedding</p></li>
<li><p><strong>带重叠地分块</strong>：10–15% 的重叠可防止在边界处丢失信息</p></li>
<li><p><strong>存储丰富的元数据</strong>：来源、日期、章节和文档类型能支持强大的预过滤，显著提升精度</p></li>
</ul>
</div>
