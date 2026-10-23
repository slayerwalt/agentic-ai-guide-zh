# 29.14　评估指标 {#evaluation-metrics}

<span id="table-29-inline-9"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>指标</strong></th><th scope="col"><strong>范围</strong></th><th scope="col"><strong>衡量内容</strong></th></tr></thead>
<tbody><tr><td>Perplexity</td><td><span class="arithmatex">\([1, \infty)\)</span></td><td>模型的“困惑度”；越低语言建模越好</td></tr>
<tr><td>Win Rate（对比基线）</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>评审/人类偏好的输出比例</td></tr>
<tr><td>BLEU</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>与参考文本的 <span class="arithmatex">\(n\)</span>-gram 重叠（偏精确率）</td></tr>
<tr><td>ROUGE-L</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>与参考文本的最长公共子序列</td></tr>
<tr><td>Pass@<span class="arithmatex">\(k\)</span></td><td><span class="arithmatex">\([0, 1]\)</span></td><td><span class="arithmatex">\(k\)</span> 个代码样本中至少 1 个通过测试的概率</td></tr>
<tr><td>MMLU / GPQA</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>知识/推理基准上的多选准确率</td></tr>
<tr><td>HumanEval</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>生成代码的功能正确率</td></tr>
<tr><td>Faithfulness (RAG)</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>受检索上下文支持的论断比例</td></tr>
<tr><td>Context Relevancy</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>检索内容中与查询相关的比例</td></tr>
<tr><td>Answer Relevancy</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>答案对问题的回应程度</td></tr></tbody>
</table>
</div>
