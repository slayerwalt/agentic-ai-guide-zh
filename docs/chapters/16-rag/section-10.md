# 16.10　RAG 与微调的协同 {#rag-fine-tuning-synergy}

## 16.10.1　何时将 RAG 与微调结合 {#when-to-combine-rag-with-fine-tuning}

<p>微调与 RAG 解决的是互补的弱点：</p>

<ul>
<li><p><strong>仅微调</strong>：模型学到风格与格式，但可能在事实上产生幻觉</p></li>
<li><p><strong>仅 RAG</strong>：模型能访问事实，但可能不知道如何最优地使用它们</p></li>
<li><p><strong>组合</strong>：对模型进行微调，使其<em>善于使用检索到的上下文</em>——引用来源、承认不确定性、忽略无关上下文</p></li>
</ul>

## 16.10.2　RAFT：检索增强微调（Retrieval-Augmented Fine-Tuning） {#raft-retrieval-augmented-fine-tuning}

<p>RAFT <a class="book-citation" id="cite-zhang2024raft-1" href="../../references.md#ref-zhang2024raft" aria-label="参考文献 425">[425]</a> 在混合了相关文档与<em>干扰</em>文档的设置下训练模型回答问题，教会模型识别并只使用相关上下文：</p>

<ol>
<li><p>对每个训练样本 <span class="arithmatex">\((q, a, d^*)\)</span>，采样 <span class="arithmatex">\(k-1\)</span> 个干扰文档 <span class="arithmatex">\(\{d_i^-\}\)</span></p></li>
<li><p>在 <code>[q, </code><span class="arithmatex">\(d^*\)</span><code>, </code><span class="arithmatex">\(d_1^-\)</span><code>, …, </code><span class="arithmatex">\(d_{k-1}^-\)</span><code>]</code> <span class="arithmatex">\(\to\)</span> <code>[chain-of-thought + a]</code> 上微调</p></li>
<li><p>思维链显式引用 <span class="arithmatex">\(d^*\)</span> 中的内容，教模型对答案进行接地</p></li>
</ol>



<div class="arithmatex">\[\mathcal{L}_{\text{RAFT}} = -\mathbb{E}_{(q,a,d^*,\{d_i^-\})} \left[
    \log P_\theta\!\left(\text{CoT}(d^*) \oplus a \;\middle|\; q, d^*, \{d_i^-\}\right)
  \right]
\tag{16.30}\]</div>

## 16.10.3　检索器—生成器联合训练 {#joint-retriever-generator-training}

<p>为获得极致性能，可以对检索器与生成器进行联合训练。REALM <a class="book-citation" id="cite-guu2020realm-1" href="../../references.md#ref-guu2020realm" aria-label="参考文献 128">[128]</a> 与 RAG <a class="book-citation" id="cite-lewis2020retrieval-1" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a> 论文提出了端到端训练方法，让 Gradient 流经检索步骤：</p>



<div class="arithmatex">\[\nabla_\theta \mathcal{L} = \nabla_\theta \left[
    -\log \sum_{d \in \mathcal{D}} P_\theta(a \mid q, d) \cdot P_\phi(d \mid q)
  \right]
\tag{16.31}\]</div>

<p>检索器参数 <span class="arithmatex">\(\phi\)</span> 通过 REINFORCE 估计器更新，或将 <span class="arithmatex">\(P_\phi(d \mid q)\)</span> 视为关于文档的可微 Attention 进行更新。</p>

<div class="admonition warning"><p class="admonition-title">联合训练的挑战</p>
<p>检索器—生成器联合训练强大但复杂：(1) 随着 <span class="arithmatex">\(\phi\)</span> 变化，文档索引必须定期刷新（异步索引刷新），(2) 训练信号稀疏（仅 top-<span class="arithmatex">\(k\)</span> 文档参与贡献），(3) 在未从预训练检索器仔细初始化的情况下，训练不稳定。</p>
</div>
