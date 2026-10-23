# 2.2.5　vLLM 中的投机解码 {#speculative-decoding-integration}

<p>投机解码（Speculative Decoding）使用一个小型<em>草稿模型</em>（draft model，例如 1B 参数）快速提出 <span class="arithmatex">\(k\)</span> 个候选 token，再由大型<em>目标模型</em>（target model）在一次前向传播中验证。从第一个被拒绝处之前的所有 token 都会被接受（预期接受数：每次验证 3–5 个 token）。这为延迟敏感的单序列生成带来 2–3<span class="arithmatex">\(\times\)</span> 提速，且无质量损失。</p>

<p>vLLM 将投机解码与 PagedAttention 集成：</p>

<ul>
<li><p>草稿 token 被分配投机性 KV 块</p></li>
<li><p>拒绝时，投机块被释放（在分页分配下成本极低）</p></li>
<li><p>接受时，投机块被提升为主序列的一部分</p></li>
<li><p>块表更新为 <span class="arithmatex">\(O(k)\)</span>——只需更新少量表项</p></li>
</ul>
