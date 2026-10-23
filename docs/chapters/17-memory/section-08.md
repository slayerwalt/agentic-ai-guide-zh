# 17.8　记忆方案的对比 {#sec:memory-comparison}

<p class="table-caption" id="table-17-2"><span class="table-number">表 17.2　</span>在关键维度上对智能体记忆架构的对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>架构</strong></th><th scope="col"><strong>容量</strong></th><th scope="col"><strong>检索</strong></th><th scope="col"><strong>更新成本</strong></th><th scope="col"><strong>可训练</strong></th><th scope="col"><strong>最适合</strong></th></tr></thead>
<tbody><tr><td>上下文内（工作记忆）</td><td><span class="arithmatex">\(O(L)\)</span> Token</td><td>0 ms</td><td>免费</td><td>通过微调</td><td>短任务、主动推理</td></tr>
<tr><td>稠密 RAG <a class="book-citation" id="cite-lewis2020retrieval-1" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a></td><td><span class="arithmatex">\(O(10^7)\)</span> 文档</td><td>10–50 ms</td><td><span class="arithmatex">\(O(1)\)</span> Embedding</td><td>仅编码器</td><td>语义检索、问答</td></tr>
<tr><td>稀疏（BM25） <a class="book-citation" id="cite-robertson2009probabilistic-1" href="../../references.md#ref-robertson2009probabilistic" aria-label="参考文献 309">[309]</a></td><td><span class="arithmatex">\(O(10^8)\)</span> 文档</td><td>1–5 ms</td><td><span class="arithmatex">\(O(|d|)\)</span> 索引</td><td>否</td><td>关键词检索、法律/医疗</td></tr>
<tr><td>混合 RAG <a class="book-citation" id="cite-chen2022hybrid-1" href="../../references.md#ref-chen2022hybrid" aria-label="参考文献 44">[44]</a></td><td><span class="arithmatex">\(O(10^7)\)</span> 文档</td><td>15–60 ms</td><td><span class="arithmatex">\(O(1)\)</span> Embedding</td><td>仅编码器</td><td>通用检索</td></tr>
<tr><td>摘要</td><td>无上限</td><td>0 ms（在上下文内）</td><td><span class="arithmatex">\(O(|e|)\)</span> 次 LLM 调用</td><td>通过微调</td><td>长对话、叙事</td></tr>
<tr><td>知识图谱 <a class="book-citation" id="cite-lacroix2020tensor-1" href="../../references.md#ref-lacroix2020tensor" aria-label="参考文献 197">[197]</a></td><td><span class="arithmatex">\(O(10^9)\)</span> 三元组</td><td>5–100 ms</td><td><span class="arithmatex">\(O(1)\)</span> 插入</td><td>Embedding 层</td><td>结构化事实、多跳</td></tr>
<tr><td>键值记忆网络 <a class="book-citation" id="cite-sukhbaatar2015end-1" href="../../references.md#ref-sukhbaatar2015end" aria-label="参考文献 342">[342]</a></td><td><span class="arithmatex">\(O(M)\)</span> 槽位</td><td><span class="arithmatex">\(O(M)\)</span> Attention</td><td>一次 Gradient 更新</td><td>完全可训</td><td>端到端可微任务</td></tr>
<tr><td>MemGPT 分层 <a class="book-citation" id="cite-packer2023memgpt-1" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a></td><td>无上限</td><td>0–100 ms</td><td>混合</td><td>通过 RL</td><td>长周期 Agent、助手</td></tr>
<tr><td>Graph RAG <a class="book-citation" id="cite-edge2024local-1" href="../../references.md#ref-edge2024local" aria-label="参考文献 82">[82]</a></td><td><span class="arithmatex">\(O(10^7)\)</span> 节点</td><td>20–200 ms</td><td><span class="arithmatex">\(O(1)\)</span> 插入</td><td>仅编码器</td><td>复杂推理、社区结构</td></tr></tbody>
</table>
</div>
