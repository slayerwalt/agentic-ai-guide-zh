# 28.15　专家混合（Mixture of Experts）问题 {#mixture-of-experts-questions}

<section class="quiz-question" id="q-source-65"><h2 class="question-title">Q：Mixtral 8x7B 总共有 47B 参数但每个 token 只激活 13B。请解释它如何工作，以及为何高效。</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：Mixtral 将每个 FFN 层替换为 8 个并行的专家 FFN（每个专家的 FFN 部分约 <span class="arithmatex">\(\sim\)</span>7B 参数）。一个 router 网络为每个 token 选择 Top-2 专家。</p>

<p><strong>为什么总共 47B</strong>：Attention 层是共享的（不复制）= <span class="arithmatex">\(\sim\)</span>5B。FFN 专家：8 <span class="arithmatex">\(\times\)</span> <span class="arithmatex">\(\sim\)</span>5.25B = 42B。总计：<span class="arithmatex">\(\sim\)</span>47B。</p>

<p><strong>为什么 13B 激活</strong>：每个 token 只有 2 个专家被激活。激活参数 = attention（<span class="arithmatex">\(\sim\)</span>5B）+ 2 个 FFN 专家（<span class="arithmatex">\(\sim\)</span>2 <span class="arithmatex">\(\times\)</span> 5.25B）<span class="arithmatex">\(\approx\)</span> 13B。</p>

<p><strong>为什么高效</strong>：计算成本随<em>激活</em>参数（13B）扩展，与 13B 稠密模型相当。但容量（存储的知识）随<em>总</em>参数（47B）扩展，与更大的模型相当。结果：Mixtral 以 13B 的计算成本匹配 Llama-2 70B 的质量。</p>

<p><strong>显存成本</strong>：仍需把全部 47B 参数装进显存（所有专家都加载），所以显存 = 47B 模型，但计算 = 13B 模型。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-66"><h2 class="question-title">Q：MoE 中的负载均衡问题是什么，如何解决？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：不加约束时，router 倾向于将大部分 token 路由到 1–2 个“热门”专家（强者愈强的正反馈）。这导致：</p>

<ul>
<li><p>容量浪费：8 个专家中有 6 个未被使用，模型实际上缩小到 2 个专家的规模。</p></li>
<li><p>计算不均衡：如果每个专家在不同 GPU 上，热门专家会成为瓶颈，其他专家闲置。</p></li>
</ul>

<p><strong>解决方案</strong>：辅助负载均衡 Loss：<span class="arithmatex">\(\mathcal{L}_{\text{bal}} = \alpha \cdot N \sum_{i=1}^N f_i \cdot p_i\)</span>，其中 <span class="arithmatex">\(f_i\)</span> = 路由到专家 <span class="arithmatex">\(i\)</span> 的 token 比例，<span class="arithmatex">\(p_i\)</span> = 专家 <span class="arithmatex">\(i\)</span> 的平均 router 概率。这会惩罚不均匀的分布。</p>

<p><strong>替代方案</strong>：专家容量因子——对每个 Batch 中每个专家的最大 token 数做硬上限。溢出的 token 被丢弃或重新路由。</p>

<p>典型 <span class="arithmatex">\(\alpha\)</span>：0.01–0.1（足够小以不损害主 Loss，足够大以防止坍缩）。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>
