# 14.5　生成任务的指标 {#subsec:generation-metrics}

<p>生成指标量化了在具备参考答案或明确正确性标准的任务上模型输出的质量。</p>

## 14.5.1　BLEU {#bleu}

<p>BLEU（Bilingual Evaluation Understudy） <a class="book-citation" id="cite-papineni2002bleu-1" href="../../references.md#ref-papineni2002bleu" aria-label="参考文献 283">[283]</a> 度量假设译文 <span class="arithmatex">\(h\)</span> 与一个或多个参考 <span class="arithmatex">\(\mathcal{R}\)</span> 之间的 <span class="arithmatex">\(n\)</span>-gram 精度：</p>



<div class="arithmatex">\[\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
\tag{14.17}\]</div>

<p>其中 <span class="arithmatex">\(p_n\)</span> 是修正后的 <span class="arithmatex">\(n\)</span>-gram 精度，<span class="arithmatex">\(w_n = 1/N\)</span> 为均匀权重，BP 为简短惩罚（brevity penalty）：</p>



<div class="arithmatex">\[\text{BP} = \begin{cases} 1 &amp; \text{if } |h| &gt; |r| \\ e^{1 - |r|/|h|} &amp; \text{if } |h| \leq |r| \end{cases}
\tag{14.18}\]</div>

<p>其中 <span class="arithmatex">\(|r|\)</span> 为长度最接近的参考的长度。修正后的 <span class="arithmatex">\(n\)</span>-gram 精度将每个 <span class="arithmatex">\(n\)</span>-gram 的计数截断到其在任一参考中的最大出现次数：</p>



<div class="arithmatex">\[p_n = \frac{\sum_{\text{ngram} \in h} \min\!\left(\text{count}(\text{ngram}, h),\, \max_{r \in \mathcal{R}} \text{count}(\text{ngram}, r)\right)}{\sum_{\text{ngram} \in h} \text{count}(\text{ngram}, h)}
\tag{14.19}\]</div>

<div class="admonition warning"><p class="admonition-title">BLEU 的局限</p>
<p>BLEU 是为带多个参考的机器翻译而设计的。对于只有单一参考的开放式生成，即使输出质量很高，BLEU 分数也常常接近零。BLEU 无法捕捉语义相似性，会惩罚合理的复述，并且对分词敏感。仅在存在多个多样化参考且任务输出多样性较低时再使用 BLEU。</p>
</div>

## 14.5.2　ROUGE {#rouge}

<p>ROUGE（Recall-Oriented Understudy for Gisting Evaluation） <a class="book-citation" id="cite-lin2004rouge-1" href="../../references.md#ref-lin2004rouge" aria-label="参考文献 216">[216]</a> 是一类面向召回的指标，专为摘要任务设计：</p>



<div class="arithmatex">\[\begin{align*}
\text{ROUGE-N} &amp;= \frac{\sum_{r \in \mathcal{R}} \sum_{\text{ngram} \in r} \min(\text{count}(\text{ngram}, h), \text{count}(\text{ngram}, r))}{\sum_{r \in \mathcal{R}} \sum_{\text{ngram} \in r} \text{count}(\text{ngram}, r)} \tag{14.20}\\[6pt]

    \text{ROUGE-L} &amp;= \frac{\text{LCS}(h, r)}{|r|}
\tag{14.21}
\end{align*}\]</div>

<p>其中 LCS 表示最长公共子序列（Longest Common Subsequence）。ROUGE-1 与 ROUGE-2 度量 unigram 与 bigram 的召回；ROUGE-L 则捕捉句子级结构。F-measure 变体在精度与召回之间取得平衡：</p>



<div class="arithmatex">\[\text{ROUGE-N}_F = \frac{(1+\beta^2) \cdot P \cdot R}{\beta^2 P + R}
\tag{14.22}\]</div>

<p>当 <span class="arithmatex">\(\beta = 1\)</span> 时为等权。</p>

## 14.5.3　BERTScore {#bertscore}

<p>BERTScore <a class="book-citation" id="cite-zhang2020bertscore-1" href="../../references.md#ref-zhang2020bertscore" aria-label="参考文献 426">[426]</a> 使用预训练 BERT 模型的上下文 Embedding 来计算 Token 级相似度。给定假设 Token <span class="arithmatex">\(\hat{\mathbf{x}} = \langle \hat{x}_1, \ldots, \hat{x}_m \rangle\)</span> 与参考 Token <span class="arithmatex">\(\mathbf{x} = \langle x_1, \ldots, x_n \rangle\)</span>，及其 Embedding <span class="arithmatex">\(\hat{\mathbf{e}}_i\)</span> 与 <span class="arithmatex">\(\mathbf{e}_j\)</span>：</p>



<div class="arithmatex">\[\begin{align*}
R_{\text{BERT}} &amp;= \frac{1}{|x|} \sum_{x_j \in \mathbf{x}} \max_{\hat{x}_i \in \hat{\mathbf{x}}} \frac{\hat{\mathbf{e}}_i^\top \mathbf{e}_j}{\|\hat{\mathbf{e}}_i\| \|\mathbf{e}_j\|} \tag{14.23}\\[4pt]

    P_{\text{BERT}} &amp;= \frac{1}{|\hat{x}|} \sum_{\hat{x}_i \in \hat{\mathbf{x}}} \max_{x_j \in \mathbf{x}} \frac{\hat{\mathbf{e}}_i^\top \mathbf{e}_j}{\|\hat{\mathbf{e}}_i\| \|\mathbf{e}_j\|} \tag{14.24}\\[4pt]

    F_{\text{BERT}} &amp;= 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}
\tag{14.25}
\end{align*}\]</div>

<p>BERTScore 与人类判断的相关性优于 BLEU 与 ROUGE，尤其在处理复述以及语义等价但词面不同的输出时。使用逆文档频率（IDF）进行重要性加权可进一步提升相关性：</p>



<div class="arithmatex">\[R_{\text{BERT}}^{\text{idf}} = \frac{\sum_{x_j \in \mathbf{x}} \text{idf}(x_j) \max_{\hat{x}_i} \cos(\hat{\mathbf{e}}_i, \mathbf{e}_j)}{\sum_{x_j \in \mathbf{x}} \text{idf}(x_j)}
\tag{14.26}\]</div>

## 14.5.4　METEOR {#meteor}

<p>METEOR <a class="book-citation" id="cite-banerjee2005meteor-1" href="../../references.md#ref-banerjee2005meteor" aria-label="参考文献 19">[19]</a> 通过在 unigram 匹配上计算 F-score，并配合词干化与同义词匹配模块，解决了 BLEU 对召回的盲视：</p>



<div class="arithmatex">\[\text{METEOR} = F_{\text{mean}} \cdot (1 - \text{Pen})
\tag{14.27}\]</div>

<p>其中 <span class="arithmatex">\(F_{\text{mean}} = \frac{10PR}{R + 9P}\)</span>（召回加权的调和平均），碎片化惩罚 <span class="arithmatex">\(\text{Pen} = 0.5 \cdot (c/u_m)^3\)</span> 对非连续匹配进行惩罚（<span class="arithmatex">\(c\)</span> 为块数，<span class="arithmatex">\(u_m\)</span> 为匹配的 unigram 数）。</p>

## 14.5.5　困惑度（Perplexity） {#perplexity}

<p>困惑度（Perplexity）衡量语言模型对留出文本序列 <span class="arithmatex">\(w_1, w_2, \ldots, w_T\)</span> 的预测能力：</p>



<div class="arithmatex">\[\text{PPL}(w_{1:T}) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(w_t \mid w_{1:t-1})\right)
\tag{14.28}\]</div>

<p>困惑度越低，预测性能越好。困惑度适合在相同分词与测试集上比较模型，但在词表或分词器不同的模型间不能直接比较。在评估中，困惑度最实用的用途是作为一种合理性检查（sanity check）以及用于检测分布漂移。</p>

## 14.5.6　代码任务的 Pass@k {#passk-for-code}

<p>对于代码生成，功能正确性通过在测试用例上执行生成的代码来衡量。pass@<span class="arithmatex">\(k\)</span> 指标 <a class="book-citation" id="cite-chen2021evaluating-1" href="../../references.md#ref-chen2021evaluating" aria-label="参考文献 47">[47]</a> 估计在 <span class="arithmatex">\(k\)</span> 个生成样本中至少有一个通过全部测试的概率：</p>



<div class="arithmatex">\[\text{pass@}k = \mathbb{E}_{\text{problems}}\!\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
\tag{14.29}\]</div>

<p>其中 <span class="arithmatex">\(n\)</span> 是每个问题生成的样本总数，<span class="arithmatex">\(c\)</span> 是其中通过测试的样本数。该无偏估计量避免了朴素估计量（精确采样 <span class="arithmatex">\(k\)</span> 个解并检查是否通过）的高方差问题。在实践中通常生成 <span class="arithmatex">\(n = 200\)</span> 个样本，并报告 pass@1、pass@10、pass@100。</p>

<div class="admonition example"><p class="admonition-title">Pass@k 计算</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">scipy.special</span><span class="w"> </span><span class="kn">import</span> <span class="n">comb</span>

<span class="k">def</span><span class="w"> </span><span class="nf">pass_at_k</span><span class="p">(</span><span class="n">n</span><span class="p">:</span> <span class="nb">int</span><span class="p">,</span> <span class="n">c</span><span class="p">:</span> <span class="nb">int</span><span class="p">,</span> <span class="n">k</span><span class="p">:</span> <span class="nb">int</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""pass@k 的无偏估计量。</span>

<span class="sd">    Args:</span>
<span class="sd">        n: 每个问题生成的样本总数</span>
<span class="sd">        c: 通过全部测试的样本数</span>
<span class="sd">        k: 考虑的样本数</span>
<span class="sd">    """</span>
    <span class="k">if</span> <span class="n">n</span> <span class="o">-</span> <span class="n">c</span> <span class="o">&lt;</span> <span class="n">k</span><span class="p">:</span>
        <span class="k">return</span> <span class="mf">1.0</span>
    <span class="k">return</span> <span class="mf">1.0</span> <span class="o">-</span> <span class="n">comb</span><span class="p">(</span><span class="n">n</span> <span class="o">-</span> <span class="n">c</span><span class="p">,</span> <span class="n">k</span><span class="p">,</span> <span class="n">exact</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span> <span class="o">/</span> <span class="n">comb</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">k</span><span class="p">,</span> <span class="n">exact</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="c1"># 示例：200 个样本，15 个通过，计算 pass@1、pass@10、pass@100</span>
<span class="k">for</span> <span class="n">k</span> <span class="ow">in</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">100</span><span class="p">]:</span>
    <span class="n">score</span> <span class="o">=</span> <span class="n">pass_at_k</span><span class="p">(</span><span class="n">n</span><span class="o">=</span><span class="mi">200</span><span class="p">,</span> <span class="n">c</span><span class="o">=</span><span class="mi">15</span><span class="p">,</span> <span class="n">k</span><span class="o">=</span><span class="n">k</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"pass@</span><span class="si">{</span><span class="n">k</span><span class="si">}</span><span class="s2">: </span><span class="si">{</span><span class="n">score</span><span class="si">:</span><span class="s2">.4f</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
<span class="c1"># pass@1:   0.0750</span>
<span class="c1"># pass@10:  0.5391</span>
<span class="c1"># pass@100: 0.9999</span>
</code></pre></div>
</div>

## 14.5.7　精确匹配与 F1 {#exact-match-and-f1}

<p>对于抽取式问答（例如 SQuAD），有两个标准指标：</p>

<ul>
<li><p><strong>精确匹配（Exact Match, EM）：</strong>二元指示量，判断归一化（小写化、去除冠词与标点）后的预测答案字符串是否与任一金标答案完全相同。</p></li>
<li><p><strong>Token 级 F1：</strong>将预测与金标答案视为 Token 的多重集，并计算 F1 分数：</p>



<div class="arithmatex">\[F1 = \frac{2 \cdot |\text{pred} \cap \text{gold}|}{|\text{pred}| + |\text{gold}|}
\tag{14.30}\]</div></li>
</ul>

<p>对于多答案设置，报告对所有金标答案取最大值后的 F1。</p>

<p class="table-caption" id="table-14-3"><span class="table-number">表 14.3　</span>生成指标汇总：适用性与关键性质。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>指标</strong></th><th scope="col"><strong>任务</strong></th><th scope="col"><strong>是否无参考？</strong></th><th scope="col"><strong>与人类判断的相关性</strong></th></tr></thead>
<tbody><tr><td>BLEU</td><td>翻译</td><td>否</td><td>低–中</td></tr>
<tr><td>ROUGE</td><td>摘要</td><td>否</td><td>中</td></tr>
<tr><td>BERTScore</td><td>通用 NLG</td><td>否</td><td>高</td></tr>
<tr><td>METEOR</td><td>翻译</td><td>否</td><td>中–高</td></tr>
<tr><td>Perplexity</td><td>语言模型质量</td><td>是</td><td>低</td></tr>
<tr><td>Pass@k</td><td>代码生成</td><td>否（依赖测试）</td><td>极高</td></tr>
<tr><td>Exact Match</td><td>抽取式问答</td><td>否</td><td>极高</td></tr>
<tr><td>Token F1</td><td>抽取式问答</td><td>否</td><td>高</td></tr></tbody>
</table>
</div>
