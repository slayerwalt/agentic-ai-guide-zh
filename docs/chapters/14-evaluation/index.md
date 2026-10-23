# 第 14 章 · LLM 评估 {#sec:llm-evaluation}

<p>评估是任何严谨机器学习流水线的支柱，但它或许是大语言模型开发中最被低估的组成部分。与经典的监督学习不同——在那里，一个带有真实标签的留出测试集就能提供干净的信号——评估 LLM 需要应对开放式生成、主观质量判断、多步推理链，以及无处不在的基准污染风险。本节系统性地梳理评估领域的全景：从评估类型的分类法、人工标注的机制，到排序指标的数学原理、LLM-as-Judge 的实操要点，再到那些悄无声息地腐蚀评估流水线的陷阱。</p>

<div class="admonition note"><p class="admonition-title">为何 LLM 评估如此困难</p>
<p><strong>三个根本性挑战</strong>使 LLM 评估有别于经典 ML 评估：</p>

<ol>
<li><p><strong>输出空间无界。</strong>语言模型可以生成任意字符串；很少存在唯一正确答案。</p></li>
<li><p><strong>质量是多维的。</strong>有用性、事实性、安全性、连贯性与风格是相互独立的维度，彼此之间可能存在权衡。</p></li>
<li><p><strong>评估本身就是一个语言任务。</strong>判断回答是否优良需要理解能力，这意味着评估本身会暴露在与生成相同的失败模式之下。</p></li>
</ol>
</div>

## 本章内容

<ul class="chapter-outline">
<li id="subsec:eval-scheme"><a href="section-01.md">14.1　评估方案设计</a></li>
<li id="subsec:data-collection"><a href="section-02.md">14.2　评估数据采集</a></li>
<li id="subsec:synthetic-data"><a href="section-03.md">14.3　评估用合成数据生成</a></li>
<li id="subsec:ranking-metrics"><a href="section-04.md">14.4　排序任务的指标</a></li>
<li id="subsec:generation-metrics"><a href="section-05.md">14.5　生成任务的指标</a></li>
<li id="subsec:agentic-metrics"><a href="section-06.md">14.6　Agent 任务的指标</a></li>
<li id="subsec:llm-as-judge"><a href="section-07.md">14.7　LLM-as-Judge</a></li>
<li id="subsec:eval-pitfalls"><a href="section-08.md">14.8　评估陷阱</a></li>
</ul>
