# 28.30　RAG 与 Agent 化 RAG 问题 {#rag-and-agentic-rag-questions}

<section class="quiz-question" id="q-source-105"><h2 class="question-title">Q：解释互逆排名融合（Reciprocal Rank Fusion, RRF），以及它为何对混合检索有效</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：RRF 合并多个检索系统的排名，无需对分数做校准：</p>



<div class="arithmatex">\[\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}\]</div>

<p>其中 <span class="arithmatex">\(r(d)\)</span> 是文档 <span class="arithmatex">\(d\)</span> 在检索器 <span class="arithmatex">\(r\)</span> 中的排名，<span class="arithmatex">\(k=60\)</span> 是防止高排名文档主导的常数。</p>

<p><strong>为何有效</strong>：</p>

<ol>
<li><p><strong>无需分数归一化</strong>：BM25 分数无界；稠密相似度在 <span class="arithmatex">\([-1, 1]\)</span>。RRF 只使用排名，使二者可直接比较。</p></li>
<li><p><strong>对离群值鲁棒</strong>：单个检索器给出异常高分也不会主导，因为即便是排名 1，<span class="arithmatex">\(1/(k+1) \approx 0.016\)</span>。</p></li>
<li><p><strong>互补信号</strong>：BM25 捕捉精确关键词匹配；稠密检索捕捉语义相似度。两者都排名高的文档得到加成。</p></li>
</ol>

<p><strong>示例</strong>：文档 <span class="arithmatex">\(d\)</span> 在 BM25 中排名 3，在稠密中排名 7。RRF 分数 <span class="arithmatex">\(= 1/(60+3) + 1/(60+7) = 0.0159 + 0.0149 = 0.0308\)</span>。在某一边排名 1 但在另一边排名 100 的文档得到 <span class="arithmatex">\(1/61 + 1/160 = 0.0226\)</span>——尽管有排名 1，但更低。</p>

<p><strong>实践中</strong>：混合（BM25 + 稠密 + RRF）在 85%+ 的基准上优于单独使用任一种。</p>

<p><em><strong>复习：</strong>第 <a href="../16-rag/index.md">16</a> 章（检索增强生成）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-106"><h2 class="question-title">Q：什么是 Agent 化 RAG，它与标准 RAG 有何不同？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>标准 RAG</strong> 遵循固定流水线：查询 <span class="arithmatex">\(\to\)</span> 检索 <span class="arithmatex">\(\to\)</span> 生成。它没有能力：</p>

<ul>
<li><p>判断是否根本需要检索</p></li>
<li><p>评估检索到的文档是否足够</p></li>
<li><p>在检索失败时重新表述查询</p></li>
<li><p>整合多步检索的信息</p></li>
</ul>

<p><strong>Agent 化 RAG</strong> 将检索视为 Agent MDP 中的一个<em>动作</em>：</p>

<ul>
<li><p><strong>是否检索决策</strong>：Agent 评估它是否已经知道答案（对训练数据内的事实性问题跳过检索）</p></li>
<li><p><strong>查询规划</strong>：把复杂问题分解为子查询（“X 发生在哪一年？” + “那时谁是总统？”）</p></li>
<li><p><strong>自我评估</strong>：检索后评估相关性。若不足，则重新表述查询或尝试不同来源。</p></li>
<li><p><strong>多跳推理</strong>：检索 <span class="arithmatex">\(\to\)</span> 推理 <span class="arithmatex">\(\to\)</span> 识别知识缺口 <span class="arithmatex">\(\to\)</span> 再次检索</p></li>
<li><p><strong>来源路由</strong>：将查询路由到合适的知识库（时事查网络；公司信息查内部文档；编程查代码搜索）</p></li>
</ul>

<p><strong>关键架构差异</strong>：标准 RAG = 确定性流水线。Agent 化 RAG = 带条件转移的状态机（LangGraph 的 retrieve/grade/rewrite/generate 节点模式）。</p>

<p><strong>权衡</strong>：Agent 化 RAG 在复杂查询上更准确，但增加延迟（多次 LLM 调用用于路由/评分）。简单事实查询用标准 RAG；多跳或歧义查询用 Agent 化 RAG。</p>

<p><em><strong>复习：</strong>第 <a href="../16-rag/index.md">16</a> 章与第 <a href="../18-agent-harness/index.md">18</a> 章（RAG；Agent Harness）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-107"><h2 class="question-title">Q：比较 Self-RAG 与 CRAG 两种改进检索质量的方法</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>Self-RAG</strong>（Asai 等，2023）：</p>

<ul>
<li><p>将特殊的<em>反思 token</em>训练进 LLM 词表</p></li>
<li><p>推理时，模型输出 [Retrieve]、[IsRel]、[IsSup]、[IsUse] 等 token</p></li>
<li><p>模型决定<em>何时</em>检索（并非每个查询都需要）</p></li>
<li><p>检索后，模型自我评分：检索段落是否相关？我的答案是否从中得出？</p></li>
<li><p><strong>训练</strong>：在用 GPT-4 反思标签增强的数据上做 SFT</p></li>
<li><p><strong>优点</strong>：单一模型处理一切。<strong>缺点</strong>：需要定制训练。</p></li>
</ul>

<p><strong>CRAG</strong>（纠正式 RAG，Corrective RAG，Yan 等，2024）：</p>

<ul>
<li><p>使用一个轻量的<em>检索评估器</em>（独立模型）来给检索到的文档评分</p></li>
<li><p>根据置信度有三种动作：<code>Correct</code>（按原样使用）、<code>Ambiguous</code>（用网络搜索增强）、<code>Incorrect</code>（丢弃；回退到网络）</p></li>
<li><p>增加<em>知识精炼</em>步骤：仅从检索文档中抽取相关句子</p></li>
<li><p><strong>优点</strong>：可与任何冻结 LLM 配合。<strong>缺点</strong>：需要额外的评估模型；增加延迟。</p></li>
</ul>

<p><strong>关键差异</strong>：Self-RAG 把检索决策嵌入 LLM 自身（需要训练）。CRAG 是包裹在任意 LLM 周围的流水线方法（无需训练）。Self-RAG 更优雅；CRAG 在现有模型的生产环境中更实用。</p>

<p><em><strong>复习：</strong>第 <a href="../16-rag/index.md">16</a> 章（检索增强生成）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-108"><h2 class="question-title">Q：什么是“中间迷失”（lost-in-the-middle）问题，如何缓解？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：当检索到的上下文很长（很多段落）时，LLM 不成比例地关注上下文的<em>开头</em>和<em>结尾</em>，而忽略中间的信息。如果答案在 10 段中的第 5 段，模型可能漏掉。</p>

<p><strong>经验证据</strong>：Liu 等（2023）表明，对于 20 文档检索，相关文档位于位置 5–15 时，相比位置 1–3，准确率下降 15%–20%。</p>

<p><strong>缓解策略</strong>：</p>

<ol>
<li><p><strong>重排序并截断</strong>：用 cross-encoder 重排序，然后只保留 top-3 最相关段落（更少 = 更少中间迷失）。</p></li>
<li><p><strong>策略性排序</strong>：把最相关的段落同时放在上下文的开头和结尾，低相关的放中间。</p></li>
<li><p><strong>上下文压缩</strong>：插入前将每段摘要到 1–2 句。文本越少 = 位置偏倚越小。</p></li>
<li><p><strong>Map-Reduce</strong>：独立处理每段（map），再合并答案（reduce）。完全消除位置效应。</p></li>
<li><p><strong>引用 Prompt</strong>：要求模型引用其使用的段落。这迫使 Attention 关注所有段落。</p></li>
<li><p><strong>减小块大小</strong>：更小的块意味着覆盖答案所需的总块数更少。</p></li>
</ol>

<p><strong>最佳实践</strong>：检索很多（20+），重排序到 top 3–5，按相关性排序（最佳在前）。这在大多数用例中完全绕开此问题。</p>

<p><em><strong>复习：</strong>第 <a href="../16-rag/index.md">16</a> 章（检索增强生成）。</em></p>
</details></section>
