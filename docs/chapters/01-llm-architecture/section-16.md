# 1.16　幻觉检测 {#sec:hallucination}

<p>LLM 会生成流畅但可能事实错误的文本——这种现象称为<strong>幻觉</strong>（hallucination） <a class="book-citation" id="cite-ji2023hallucination-1" href="../../references.md#ref-ji2023hallucination" aria-label="参考文献 164">[164]</a>。本节介绍在模型层面（不依赖外部检索或多智能体校验）的基本检测方法。</p>

## 1.16.1　幻觉的类型 {#types-of-hallucination}

<div class="admonition note"><p class="admonition-title">幻觉分类法</p>
<ul>
<li><p><strong>内在型（Intrinsic）</strong>：与所提供的输入/上下文矛盾（例如摘要与原文相反）</p></li>
<li><p><strong>外在型（Extrinsic）</strong>：生成无法从输入验证的陈述，且事实上错误</p></li>
<li><p><strong>忠实性（Faithfulness）</strong>：输出偏离指令或所指定的约束</p></li>
</ul>
</div>

## 1.16.2　检测方法（模型层级） {#detection-methods-model-level}

<p class="table-caption" id="table-1-20"><span class="table-number">表 1.20　</span>在模型层面运作的基础幻觉检测方法。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>机制</strong></th><th scope="col"><strong>信号</strong></th></tr></thead>
<tbody><tr><td>Token 级熵</td><td>生成时的高熵表示不确定 <a class="book-citation" id="cite-kadavath2022language-1" href="../../references.md#ref-kadavath2022language" aria-label="参考文献 175">[175]</a></td><td><span class="arithmatex">\(H(P(x_t)) &gt; \tau\)</span></td></tr>
<tr><td>序列对数概率</td><td>输出的平均对数概率较低提示存在虚构</td><td><span class="arithmatex">\(\frac{1}{T}\sum_t \log P(x_t)\)</span></td></tr>
<tr><td>一致性采样</td><td>生成 <span class="arithmatex">\(N\)</span> 个回复；一致性低 <span class="arithmatex">\(=\)</span> 可能幻觉 <a class="book-citation" id="cite-manakul2023selfcheckgpt-1" href="../../references.md#ref-manakul2023selfcheckgpt" aria-label="参考文献 249">[249]</a></td><td>矛盾率</td></tr>
<tr><td>语义熵（Semantic Entropy）</td><td>对语义（而非字符串）聚类；语义熵高 <span class="arithmatex">\(=\)</span> 不确定 <a class="book-citation" id="cite-kuhn2023semantic-1" href="../../references.md#ref-kuhn2023semantic" aria-label="参考文献 193">[193]</a></td><td>聚类多样性</td></tr>
<tr><td>DoLA</td><td>对比后层与前层的 logits；放大事实知识 <a class="book-citation" id="cite-chuang2024dola-1" href="../../references.md#ref-chuang2024dola" aria-label="参考文献 58">[58]</a></td><td>层间差异</td></tr></tbody>
</table>
</div>

#### 语义熵（Semantic Entropy）。 {#semantic-entropy.}

<p>Kuhn 等人 <a class="book-citation" id="cite-kuhn2023semantic-2" href="../../references.md#ref-kuhn2023semantic" aria-label="参考文献 193">[193]</a> 观察到 token 级熵并不可靠（同义改写包含不同 token 但意义相同）。他们改为生成多个回复，按语义等价（通过 NLI）聚类，并在语义聚类上计算熵：</p>



<div class="arithmatex">\[SE = -\sum_{c \in \text{clusters}} P(c) \log P(c)\]</div>

<p>高 SE 意味着模型产生<em>语义不同</em>的答案——这是强烈的幻觉信号。</p>

#### SelfCheckGPT。 {#selfcheckgpt.}

<p>Manakul 等人 <a class="book-citation" id="cite-manakul2023selfcheckgpt-2" href="../../references.md#ref-manakul2023selfcheckgpt" aria-label="参考文献 249">[249]</a> 通过检查自一致性来检测幻觉：生成多个回复并验证主回复中的陈述是否被其他回复支持。若模型 “自相矛盾”，则该陈述很可能是幻觉。无需外部知识。</p>

#### 按层对比解码（Decoding by Contrasting Layers，DoLA）。 {#dola-decoding-by-contrasting-layers.}

<p>Chuang 等人 <a class="book-citation" id="cite-chuang2024dola-2" href="../../references.md#ref-chuang2024dola" aria-label="参考文献 58">[58]</a> 观察到事实知识在 Transformer 较深的层中浮现，而较早的层保留更多通用/不确定的表示。DoLA 在每个解码步上对比较深（“成熟”）层与较早（“未成熟”）层的 logits 分布：</p>



<div class="arithmatex">\[\text{DoLA}(x_t) = \text{softmax}\!\bigl(\log P_{\text{late}}(x_t) - \log P_{\text{early}}(x_t)\bigr)\]</div>

<p>通过放大深层中事实知识所编码的信号，DoLA 在推理时降低幻觉，<em>无需重新训练</em>——只需对对比层进行一次额外的前向传递。它与基于采样的方法互补，可以与之结合使用。</p>

<div class="admonition warning"><p class="admonition-title">模型层级检测的局限</p>
<p>这些方法检测的是<em>不确定性</em>，而非<em>错误性</em>。模型可能自信地犯错（低熵、一致的回复——但事实上错误）。要实现可靠检测，应结合基于检索的验证（RAG）或外部事实核查工具。</p>
</div>
