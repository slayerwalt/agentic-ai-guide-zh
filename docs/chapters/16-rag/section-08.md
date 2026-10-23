# 16.8　评测 {#evaluation-dup3}

<p>评测一个 RAG 系统比单独评测检索或生成更难，因为错误可能源自流水线的<em>任何阶段</em>——而且会逐级累积。完美的生成器无法弥补无关的检索结果，而当生成器产生幻觉或忽略上下文时，完美的检索器也会被浪费。</p>

<p>因此，有效的 RAG 评测在<strong>三个层面</strong>进行：</p>

<ol>
<li><p><strong>检索质量</strong>：检索器是否找出了正确的段落？（Recall、Precision、MRR、NDCG）</p></li>
<li><p><strong>生成质量</strong>：答案是否正确、忠实于检索上下文且完整？（正确性、忠实度、答案相关性）</p></li>
<li><p><strong>端到端质量</strong>：整个系统是否令用户满意？（人类偏好、任务成功率、按延迟调整的效用）</p></li>
</ol>

<p>一种常见的失败模式是只优化某一层——例如，通过大 <span class="arithmatex">\(K\)</span> 最大化 Recall@<span class="arithmatex">\(K\)</span> 会让上下文塞满边缘相关的段落，反而<em>降低</em>生成质量。下文的指标涵盖检索与生成两端，便于从业者诊断瓶颈所在阶段。</p>

## 16.8.1　检索指标 {#retrieval-metrics}

<p>令 <span class="arithmatex">\(\mathcal{R}_k\)</span> 为前 <span class="arithmatex">\(k\)</span> 名的检索文档集合，<span class="arithmatex">\(\mathcal{R}^*\)</span> 为相关文档集合。</p>

#### Recall@K。 {#recallk.}



<div class="arithmatex">\[\text{Recall@}K = \frac{|\mathcal{R}_K \cap \mathcal{R}^*|}{|\mathcal{R}^*|}
\tag{16.22}\]</div>

#### Precision@K。 {#precisionk.}



<div class="arithmatex">\[\text{Precision@}K = \frac{|\mathcal{R}_K \cap \mathcal{R}^*|}{K}
\tag{16.23}\]</div>

#### 平均倒数排名（Mean Reciprocal Rank, MRR）。 {#mean-reciprocal-rank-mrr.}



<div class="arithmatex">\[\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
\tag{16.24}\]</div>

<p>其中 <span class="arithmatex">\(\text{rank}_i\)</span> 是查询 <span class="arithmatex">\(i\)</span> 第一个相关文档的排名。</p>

#### 归一化折损累计增益（NDCG@K）。 {#normalized-discounted-cumulative-gain-ndcgk.}



<div class="arithmatex">\[\text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}, \quad
  \text{DCG@}K = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}
\tag{16.25}\]</div>

<p>其中 <span class="arithmatex">\(\text{rel}_i \in \{0, 1, 2, \ldots\}\)</span> 是第 <span class="arithmatex">\(i\)</span> 个结果的分级相关性，IDCG 是理想（完美）的 DCG。</p>

## 16.8.2　生成指标 {#generation-metrics}

#### 忠实度（Faithfulness）。 {#faithfulness.}

<p>衡量生成的答案是否<em>接地</em>于检索到的上下文——即答案中的每一项主张都能归因到某一篇检索到的文档。由 LLM 判官评估：</p>



<div class="arithmatex">\[\text{Faithfulness} = \frac{\text{\# claims supported by context}}{\text{\# total claims in answer}}
\tag{16.26}\]</div>

#### 答案相关性（Answer Relevance）。 {#answer-relevance.}

<p>衡量答案是否针对问题作答。计算方式是从答案中生成问题，并衡量其与原始查询的相似度：</p>



<div class="arithmatex">\[\text{AnswerRelevance} = \frac{1}{N} \sum_{i=1}^{N} \cos\!\left(E(q), E(\hat{q}_i)\right)
\tag{16.27}\]</div>

<p>其中 <span class="arithmatex">\(\hat{q}_i\)</span> 是从答案中生成的问题。</p>

#### 上下文精度与召回率。 {#context-precision-and-recall.}



<div class="arithmatex">\[\begin{align*}
\text{ContextPrecision@}K &amp;= \frac{1}{K} \sum_{k=1}^{K} \text{Precision@}k \cdot \mathbf{1}[\text{doc}_k \text{ is relevant}] \tag{16.28}\\

  \text{ContextRecall} &amp;= \frac{\text{\# ground-truth claims attributable to context}}{\text{\# total ground-truth claims}}
\tag{16.29}
\end{align*}\]</div>

## 16.8.3　RAGAs 框架 {#ragas-framework}

<p>RAGAs（Retrieval Augmented Generation Assessment） <a class="book-citation" id="cite-es2023ragas-1" href="../../references.md#ref-es2023ragas" aria-label="参考文献 85">[85]</a> 提供了一个使用 LLM 判官的无参考评测框架：</p>

<p class="code-caption"><span class="code-number">代码 16.12　</span>RAGAs 评测（v0.1 API；v0.2+ 使用 <code>user_input</code>、<code>response</code>、<code>retrieved_contexts</code>、<code>reference</code>）</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">ragas</span><span class="w"> </span><span class="kn">import</span> <span class="n">evaluate</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">ragas.metrics</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">faithfulness</span><span class="p">,</span>
    <span class="n">answer_relevancy</span><span class="p">,</span>
    <span class="n">context_precision</span><span class="p">,</span>
    <span class="n">context_recall</span><span class="p">,</span>
    <span class="n">answer_correctness</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">datasets</span><span class="w"> </span><span class="kn">import</span> <span class="n">Dataset</span>

<span class="n">eval_dataset</span> <span class="o">=</span> <span class="n">Dataset</span><span class="o">.</span><span class="n">from_dict</span><span class="p">({</span>
    <span class="s2">"question"</span><span class="p">:</span>  <span class="n">questions</span><span class="p">,</span>
    <span class="s2">"answer"</span><span class="p">:</span>    <span class="n">generated_answers</span><span class="p">,</span>
    <span class="s2">"contexts"</span><span class="p">:</span>  <span class="n">retrieved_contexts</span><span class="p">,</span>   <span class="c1"># 嵌套列表（每条样本对应一个上下文列表）</span>
    <span class="s2">"ground_truth"</span><span class="p">:</span> <span class="n">reference_answers</span><span class="p">,</span>
<span class="p">})</span>

<span class="n">results</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span>
    <span class="n">dataset</span><span class="o">=</span><span class="n">eval_dataset</span><span class="p">,</span>
    <span class="n">metrics</span><span class="o">=</span><span class="p">[</span>
        <span class="n">faithfulness</span><span class="p">,</span>
        <span class="n">answer_relevancy</span><span class="p">,</span>
        <span class="n">context_precision</span><span class="p">,</span>
        <span class="n">context_recall</span><span class="p">,</span>
        <span class="n">answer_correctness</span><span class="p">,</span>
    <span class="p">],</span>
<span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">results</span><span class="o">.</span><span class="n">to_pandas</span><span class="p">())</span>
</code></pre></div>

## 16.8.4　常见失败模式 {#common-failure-modes-dup2}

<div class="admonition warning"><p class="admonition-title">需监控的 RAG 失败模式</p>
<ol>
<li><p><strong>检索遗漏</strong>：相关文档存在于语料库中却未被检索到。原因：分块不佳、Embedding 模型不匹配、查询—文档词汇差距。</p></li>
<li><p><strong>上下文污染</strong>：检索到的文档包含误导性或矛盾的信息，导致模型生成错误答案。</p></li>
<li><p><strong>中部迷失（Lost-in-the-Middle）</strong>：LLM 对长上下文的开头和结尾关注更强；中间部分的相关信息可能被忽略 <a class="book-citation" id="cite-liu2023lost-1" href="../../references.md#ref-liu2023lost" aria-label="参考文献 225">[225]</a>。</p></li>
<li><p><strong>过度检索</strong>：检索块过多会稀释相关信号，并增加延迟与成本。</p></li>
<li><p><strong>即便检索仍幻觉</strong>：模型忽略检索到的上下文，基于参数化记忆生成内容，尤其当上下文与训练数据相冲突时。</p></li>
<li><p><strong>引用伪造</strong>：模型将主张归因到并不支持这些主张的文档。</p></li>
</ol>
</div>
