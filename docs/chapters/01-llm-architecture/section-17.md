# 1.17　LLM 安全与负责任 AI {#sec:safety}

<p>安全并非事后补救——它是 LLM 训练流水线不可或缺的一部分。本节涵盖 LLM 安全的关键维度以及用于强制负责任行为的机制。</p>

## 1.17.1　威胁分类 {#threat-taxonomy}

<p class="table-caption" id="table-1-21"><span class="table-number">表 1.21　</span>LLM 安全威胁类别。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>类别</strong></th><th scope="col"><strong>描述与示例</strong></th></tr></thead>
<tbody><tr><td><strong>有害内容</strong></td><td>生成有毒、暴力或非法的指令（生物武器、CSAM）</td></tr>
<tr><td><strong>偏见与歧视</strong></td><td>延续刻板印象；在不同人口群体间的不公正对待 <a class="book-citation" id="cite-gallegos2024bias-1" href="../../references.md#ref-gallegos2024bias" aria-label="参考文献 104">[104]</a></td></tr>
<tr><td><strong>隐私侵犯</strong></td><td>泄露训练数据中的 PII；记忆攻击 <a class="book-citation" id="cite-carlini2021extracting-1" href="../../references.md#ref-carlini2021extracting" aria-label="参考文献 39">[39]</a></td></tr>
<tr><td><strong>越狱（Jailbreaking）</strong></td><td>绕过安全护栏的对抗性 Prompt <a class="book-citation" id="cite-zou2023universal-1" href="../../references.md#ref-zou2023universal" aria-label="参考文献 443">[443]</a></td></tr>
<tr><td><strong>虚假信息</strong></td><td>生成令人信服但虚假的陈述（规模化的幻觉）</td></tr>
<tr><td><strong>双重用途</strong></td><td>合法能力（编程、化学）被武器化用于伤害</td></tr></tbody>
</table>
</div>

## 1.17.2　安全训练流水线 {#safety-training-pipeline}

<figure id="figure-1-14"><a href="../../assets/figures/fig_015_fig15.png" target="_blank" rel="noopener" aria-label="查看图 1.14 原图" title="点击查看原图"><img src="../../assets/figures/fig_015_fig15.png" alt="安全贯穿每个阶段：预训练中的数据过滤、SFT 中的拒答示例、RLHF 中专门的安全奖励模型，以及迭代式红队测试。" loading="lazy"></a><figcaption><span class="figure-number">图 1.14　</span>安全贯穿每个阶段：预训练中的数据过滤、SFT 中的拒答示例、RLHF 中专门的安全奖励模型，以及迭代式红队测试。</figcaption></figure>

## 1.17.3　关键安全机制 {#key-safety-mechanisms}

<div class="admonition note"><p class="admonition-title">安全技术</p>
<ul>
<li><p><strong>数据过滤</strong>：从预训练语料中剔除有毒、有偏见以及含 PII 的文本</p></li>
<li><p><strong>安全 SFT</strong>：在恰当拒答的示例上训练（“我无法帮你做这件事，因为……”）</p></li>
<li><p><strong>宪法 AI（Constitutional AI）</strong> <a class="book-citation" id="cite-bai2022constitutional-1" href="../../references.md#ref-bai2022constitutional" aria-label="参考文献 16">[16]</a>：基于原则进行自我批评（Self-Critique）；模型依照一部规则宪法修订自身输出</p></li>
<li><p><strong>安全奖励模型</strong>：基于安全标注配对训练独立的 RM；在 RLHF 中通过加权和与 helpfulness RM 结合</p></li>
<li><p><strong>护栏（Guardrails）</strong>：在服务时拦截有害请求/回复的输入/输出分类器</p></li>
<li><p><strong>红队测试（Red teaming）</strong> <a class="book-citation" id="cite-perez2022red-1" href="../../references.md#ref-perez2022red" aria-label="参考文献 290">[290]</a>：系统性的对抗评估，在部署前发现失效模式</p></li>
</ul>
</div>

## 1.17.4　有用性与安全性的权衡 {#the-helpfulnesssafety-tradeoff}

<div class="admonition tip"><p class="admonition-title">在有用性与安全性之间平衡</p>
<p>对安全的过度优化会带来<em>过度拒答</em>问题：模型拒绝无害的请求（例如在教育语境中拒绝讨论历史暴力）。目标是一个帕累托最优策略，在安全约束<em>之内</em>最大化有用性：</p>



<div class="arithmatex">\[\max_\theta \; \mathbb{E}[R_\text{helpful}] \quad \text{subject to} \quad \mathbb{E}[R_\text{safety}] \geq \tau\]</div>

<p>在实践中，这通过加权奖励实现：<span class="arithmatex">\(R = \alpha R_\text{helpful} + (1-\alpha) R_\text{safety}\)</span>，并仔细调整 <span class="arithmatex">\(\alpha\)</span>（通常 0.6–0.8）。Meta 的 Llama-3 报告称使用独立的安全与有用性奖励模型，并采用基于 margin 的加权 <a class="book-citation" id="cite-grattafiori2024llama3-1" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a>。</p>
</div>

## 1.17.5　评测 {#evaluation}

<ul>
<li><p><strong>安全基准</strong>：ToxiGen、RealToxicityPrompts、BBQ（偏见）、CrowS-Pairs</p></li>
<li><p><strong>越狱鲁棒性</strong>：GCG 攻击 <a class="book-citation" id="cite-zou2023universal-2" href="../../references.md#ref-zou2023universal" aria-label="参考文献 443">[443]</a>、多轮越狱、编码型 Prompt</p></li>
<li><p><strong>过度拒答率</strong>：度量无害 Prompt 上的假阳性拒答（目标 <span class="arithmatex">\(&lt;\)</span>5%）</p></li>
<li><p><strong>红队评估</strong>：由领域专家（生物安全、网络安全）执行的人类对抗性测试</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">安全永无完结</p>
<p>任何技术组合都无法提供绝对的安全。新的攻击向量不断被发现（多模态越狱、剥除安全训练的微调攻击、many-shot prompting）。安全需要持续监控、对新威胁的快速响应，以及纵深防御（多层独立防线）。</p>
</div>
