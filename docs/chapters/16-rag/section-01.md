# 16.1　动机与问题陈述 {#motivation-and-problem-statement}

<div class="admonition note"><p class="admonition-title">为什么 LLM 需要外部知识</p>
<p>大型语言模型以<em>参数化</em>方式存储知识——在训练过程中压缩进数十亿个权重之中。这带来了三个根本性局限：</p>

<ol>
<li><p><strong>幻觉</strong>：当查询超出模型可靠知识边界时，模型会自信地生成听起来合理但事实错误的陈述。</p></li>
<li><p><strong>知识过时</strong>：训练数据存在截止日期；模型无法了解训练之后发生的事件、论文或产品更新。</p></li>
<li><p><strong>领域特异性</strong>：通用模型缺乏对专有代码库、内部文档、专业法规或企业数据的深入了解。</p></li>
</ol>
</div>

## 16.1.1　参数化知识 vs. 非参数化知识 {#parametric-vs.-non-parametric-knowledge}

<p>我们可以将两种知识来源之间的区别形式化。令 <span class="arithmatex">\(\mathcal{M}_\theta\)</span> 表示参数为 <span class="arithmatex">\(\theta\)</span> 的语言模型，<span class="arithmatex">\(\mathcal{D} = \{d_1, d_2, \ldots, d_N\}\)</span> 为外部文档语料库。在每种范式下，给定查询 <span class="arithmatex">\(q\)</span> 生成回答 <span class="arithmatex">\(a\)</span> 的概率为：</p>



<div class="arithmatex">\[\begin{align*}
P_{\text{parametric}}(a \mid q) &amp;= P_{\mathcal{M}_\theta}(a \mid q) \tag{16.1}\\[6pt]

  P_{\text{RAG}}(a \mid q, \mathcal{D}) &amp;= \sum_{d \in \mathcal{D}} P_{\mathcal{M}_\theta}(a \mid q, d)\,
    P_{\text{ret}}(d \mid q, \mathcal{D})
\tag{16.2}
\end{align*}\]</div>

<p>其中 <span class="arithmatex">\(P_{\text{ret}}(d \mid q, \mathcal{D})\)</span> 是关于文档的检索分布。RAG 对检索到的证据进行边缘化，使生成过程接地于非参数化知识。</p>

<div class="admonition tip"><p class="admonition-title">图书馆类比</p>
<p>可以把参数化 LLM 想象成一位记住了一整座庞大图书馆但已毕业多年的学者。RAG 为这位学者发放了一张图书证——他们可以实时查阅资料、引用来源，并在需要查证文献时坦然承认，而不是凭记忆瞎猜。</p>
</div>

## 16.1.2　何时选择 RAG vs. 微调 vs. 长上下文 {#when-to-use-rag-vs.-fine-tuning-vs.-long-context}

<p class="table-caption" id="table-16-1"><span class="table-number">表 16.1　</span>决策指南：RAG vs. 微调 vs. 长上下文</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>标准</strong></th><th scope="col"><strong>RAG</strong></th><th scope="col"><strong>微调</strong></th><th scope="col"><strong>长上下文</strong></th><th scope="col"><strong>RAG + 微调</strong></th></tr></thead>
<tbody><tr><td>知识频繁更新</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr>
<tr><td>需要引用 / 接地</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td>✓</td></tr>
<tr><td>专有大型语料库</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr>
<tr><td>适配风格 / 格式</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr>
<tr><td>教授新推理技能</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr>
<tr><td>语料库可放入上下文窗口</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td></tr>
<tr><td>要求低延迟</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td></tr></tbody>
</table>
</div>

<div class="admonition warning"><p class="admonition-title">常见误解</p>
<p>RAG <em>不是</em>微调的替代品。微调教会模型<em>如何</em>推理和响应；RAG 则提供<em>推理的内容</em>。两者是互补的。一个经过良好指令跟随微调的模型比基础模型更能有效利用检索到的上下文。</p>
</div>
