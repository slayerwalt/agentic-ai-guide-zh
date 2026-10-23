# 19.6　验证工程 {#sec:verification-engineering}

<p>验证是循环工程的奖励函数。它的质量决定了循环是收敛到正确答案、收敛到错误答案，还是根本不收敛。本节建立一张按可靠性排序的验证策略层级。</p>

## 19.6.1　验证层级 {#sec:verification-hierarchy}

<p class="table-caption" id="table-19-2"><span class="table-number">表 19.2　</span>按可靠性排序的验证策略。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>策略</strong></th><th scope="col"><strong>信号来源</strong></th><th scope="col"><strong>可靠性</strong></th><th scope="col"><strong>适用场景</strong></th></tr></thead>
<tbody><tr><td>编译</td><td>语言工具链</td><td>确定性</td><td>代码必须能解析/构建</td></tr>
<tr><td>类型检查</td><td>静态分析器</td><td>确定性</td><td>类型化语言</td></tr>
<tr><td>单元/集成测试</td><td>测试运行器</td><td>确定性</td><td>测试存在且正确</td></tr>
<tr><td>Lint 与格式化</td><td>风格工具</td><td>确定性</td><td>约定已代码化</td></tr>
<tr><td>指标对比</td><td>训练/评估脚本</td><td>数值型</td><td>ML 实验</td></tr>
<tr><td>LLM-as-judge</td><td>第二个模型</td><td>概率性</td><td>主观质量标准</td></tr>
<tr><td>人工审查</td><td>领域专家</td><td>黄金标准</td><td>高风险的决策</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">验证原则</p>
<p>循环的可靠程度不会超过它最不可靠的那一步验证。只要存在确定性检查，就用它。把 LLM-as-judge 留给那些确实无法机械验证的标准。绝不让生成模型评判自己的输出——制造者与检查者必须分离。</p>
</div>

## 19.6.2　确定性 vs. 概率性验证 {#sec:det-vs-prob-verification}

<p>确定性验证器（编译器、测试套件、linter）是黄金标准，因为它们产生客观的通过/失败，模型无法靠说服钻空子。测试要么通过要么不通过——再雄辩的推理也改变不了裁决。</p>

<p>概率性验证器（LLM-as-judge）对没有机械检查的任务是必要的——代码质量、文档清晰度、UX 改进——但它们引入了失败模式：</p>

<ul>
<li><p><strong>自我评估偏差</strong>：如果同一个模型既生成又评估，它会对自己的输出手下留情。</p></li>
<li><p><strong>评估器博弈</strong>：经过多次迭代，生成器可能学会产出满足评估器表面模式的输出，却没有实现真正的质量。</p></li>
<li><p><strong>评估器不一致</strong>：由于采样方差，同一个 LLM-judge 可能在不同运行中对相同输出给出不同分数。</p></li>
</ul>

<p>缓解策略包括：用另一个（通常更强）模型作评估器、提供带二元标准的显式评分细则，以及定期用人类判断校准 LLM-judge。</p>
