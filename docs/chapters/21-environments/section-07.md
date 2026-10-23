# 21.7　评估 harness 设计 {#sec:eval-harness}

<p>评估 harness 是在一组基准上运行 Agent、采集结果、产出汇总统计的基础设施。优秀的 harness 设计与优秀的环境设计同等重要。</p>

## 21.7.1　确定性 vs. 随机性环境 {#deterministic-vs.-stochastic-environments}

<ul>
<li><p><strong>确定性环境</strong>对相同动作序列产出相同观察序列。易于调试和复现，但可能无法反映真实世界的变异性。</p></li>
<li><p><strong>随机性环境</strong>引入随机性（程序化生成、网络延迟、用户模拟）。每个任务需要多次运行以估计平均性能和置信区间。</p></li>
</ul>

<div class="admonition question"><p class="admonition-title">多少次运行才够？</p>
<p>对于含 <span class="arithmatex">\(N\)</span> 个任务、二元奖励的基准，平均成功率的标准误差为 <span class="arithmatex">\(\sqrt{p(1-p)/N}\)</span>。当 <span class="arithmatex">\(N=500\)</span> 且 <span class="arithmatex">\(p=0.4\)</span> 时，95% 置信区间约为 <span class="arithmatex">\(\pm 4.3\%\)</span>。对随机性环境，需乘以 <span class="arithmatex">\(\sqrt{k}\)</span>，其中 <span class="arithmatex">\(k\)</span> 是每个任务的独立运行次数。随机性基准的常见做法是每任务 3–5 次运行。</p>
</div>

## 21.7.2　留出测试环境 {#held-out-test-environments}

<p>基准的可信度要求在<em>环境</em>级别（而不仅是任务级别）做严格的训练/测试划分。一个在 WebArena 任务上训练过的 Agent 应在训练时未使用的留出任务集合上评估。理想情况下，留出集合应覆盖与训练集不同的网站、任务类型和难度等级。</p>

## 21.7.3　跨环境泛化 {#cross-environment-generalization}

<p>对 Agent 的终极考验是其在某一环境中学到的技能能否迁移到另一环境。跨环境评估协议测量：</p>

<ul>
<li><p><strong>零样本迁移</strong>：在环境 A 上训练，无微调地在环境 B 上测试。</p></li>
<li><p><strong>少样本适配</strong>：评估前提供来自环境 B 的 <span class="arithmatex">\(k\)</span> 个示范。</p></li>
<li><p><strong>持续学习</strong>：依次在环境 A、B、C 上训练；在 C 上训练完成后测量三者上的性能。</p></li>
</ul>

## 21.7.4　人类基线采集 {#human-baseline-collection}

<p>每个基准都应包含人类性能作为参考点。人类基线有三个用途：</p>

<ol>
<li><p>它们为任务难度建立上界。</p></li>
<li><p>它们揭示任务是否根本可解（某些基准任务被证实是模糊的或不可解的）。</p></li>
<li><p>它们为解读 Agent 分数提供校准点（“该 Agent 达到了人类性能的 40%”）。</p></li>
</ol>

<p>人类基线应从具备领域专长的工作者采集（如 SWE-bench 应使用软件工程师而非众包工人），并应包含任务用时测量以便进行效率比较。</p>
