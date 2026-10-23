# 12.7　用例：从零构建一个研究 Agent {#use-case-building-a-research-agent-from-scratch}

<p>本用例展示了如何使用本文讨论的技术，构建一个完全自主的<strong>研究 Agent</strong>——一个能够提出假设、检索文献、分析数据、编写代码、运行实验并产出最终报告的 LLM。</p>

## 12.7.1　问题定义 {#problem-definition}

<div class="admonition note"><p class="admonition-title">研究 Agent 需求</p>
<p><strong>输入</strong>：一个研究问题（例如，“Learning rate warmup 时长对 7B 模型 GRPO 收敛的影响是什么？”）</p>

<p><strong>输出</strong>：一份包含方法论、实验、结果和结论的完整研究报告。</p>

<p><strong>所需能力</strong>：</p>

<ol>
<li><p><strong>文献搜索</strong>：查询 arXiv、Semantic Scholar，找出相关论文</p></li>
<li><p><strong>假设生成</strong>：从背景知识中提出可检验的假设</p></li>
<li><p><strong>实验设计</strong>：编写带恰当对照的训练脚本</p></li>
<li><p><strong>代码执行</strong>：运行实验，收集指标</p></li>
<li><p><strong>数据分析</strong>：解析日志、计算统计量、生成图表</p></li>
<li><p><strong>科学写作</strong>：将发现综合成一份连贯的报告</p></li>
<li><p><strong>自我纠正</strong>：检测失败实验并用修改后的参数重试</p></li>
</ol>
</div>

## 12.7.2　MDP 形式化 {#mdp-formulation}

<div class="admonition note"><p class="admonition-title">研究 Agent MDP</p>
<ul>
<li><p><strong>状态</strong> <span class="arithmatex">\(s_t\)</span>：系统 Prompt + 研究问题 + 完整的动作/观测历史（工具输出、代码结果、搜索结果）。上下文窗口：128K Token。</p></li>
<li><p><strong>动作</strong> <span class="arithmatex">\(a_t\)</span>：来自动作空间的结构化工具调用（见下文）+ 推理轨迹（CoT）。</p></li>
<li><p><strong>转移</strong> <span class="arithmatex">\(T(s_{t+1}|s_t, a_t)\)</span>：确定性——将动作 + 工具输出附加到上下文。</p></li>
<li><p><strong>Reward</strong> <span class="arithmatex">\(R\)</span>：基于报告质量的稀疏终端 reward（见下方 Reward 设计）。</p></li>
<li><p><strong>视野</strong>：20–100 步（典型研究轨迹）。</p></li>
<li><p><strong>折扣</strong> <span class="arithmatex">\(\gamma = 1.0\)</span>（Episode 式；有限任务不折扣）。</p></li>
</ul>
</div>

## 12.7.3　动作空间 {#action-space}

<p class="table-caption" id="table-12-7"><span class="table-number">表 12.7　</span>研究 Agent 的工具/动作空间。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>工具</strong></th><th scope="col"><strong>类别</strong></th><th scope="col"><strong>描述</strong></th></tr></thead>
<tbody><tr><td><code>search_papers</code></td><td>文献</td><td>查询 Semantic Scholar/arXiv。返回标题、摘要、引用。</td></tr>
<tr><td><code>read_paper</code></td><td>文献</td><td>获取论文全文或指定章节。</td></tr>
<tr><td><code>write_code</code></td><td>实验</td><td>向工作区写入 Python/训练脚本。</td></tr>
<tr><td><code>execute_code</code></td><td>实验</td><td>在沙盒环境中运行脚本。返回 stdout/stderr。</td></tr>
<tr><td><code>read_file</code></td><td>分析</td><td>读取日志、CSV 或中间结果。</td></tr>
<tr><td><code>plot_data</code></td><td>分析</td><td>生成 matplotlib/seaborn 可视化。</td></tr>
<tr><td><code>compute_stats</code></td><td>分析</td><td>运行统计检验（t 检验、置信区间）。</td></tr>
<tr><td><code>write_report</code></td><td>输出</td><td>撰写最终研究报告的各个章节（LaTeX/Markdown）。</td></tr>
<tr><td><code>think</code></td><td>推理</td><td>内部推理步骤（不进行外部工具调用）。</td></tr>
<tr><td><code>submit</code></td><td>终止</td><td>提交最终报告。结束 Episode。</td></tr></tbody>
</table>
</div>

## 12.7.4　架构：模型与基础设施选择 {#architecture-model-and-infrastructure-choices}

<div class="admonition note"><p class="admonition-title">架构决策——应用论文中的概念</p>
<ul>
<li><p><strong>基础模型</strong>：Qwen-2.5 72B（强推理 + 代码能力）。QLoRA 微调（<span class="arithmatex">\(r=32\)</span>，所有线性层）——见 LoRA 一节。</p></li>
<li><p><strong>推理</strong>：vLLM，TP=4，启用前缀缓存（系统 Prompt 在 rollout 间共享）——见 vLLM 一节。</p></li>
<li><p><strong>训练</strong>：每个研究问题用 <span class="arithmatex">\(N=4\)</span> 条轨迹的 GRPO——无需 Value 模型（见 GRPO 一节）。</p></li>
<li><p><strong>硬件</strong>：8<span class="arithmatex">\(\times\)</span>H100 节点。QLoRA 适配器装入 48 GB；vLLM 生成使用剩余容量。</p></li>
<li><p><strong>上下文管理</strong>：128K 上下文配合 FlashAttention（见 FlashAttention 一节）。超出上下文的轨迹采用滑动窗口摘要。</p></li>
<li><p><strong>推测式解码</strong>：长研究轨迹期间用 Eagle 头进行快速生成（见推测式解码一节）。</p></li>
</ul>
</div>

## 12.7.5　Reward 设计 {#reward-design}

<div class="admonition note"><p class="admonition-title">多组件研究 Reward</p>
<p>当 Agent 调用 <code>submit</code> 时计算终端 reward：</p>



<div class="arithmatex">\[R = w_1 R_{\text{quality}} + w_2 R_{\text{correctness}} + w_3 R_{\text{novelty}} + w_4 R_{\text{efficiency}} + w_5 R_{\text{format}}\]</div>

<span id="table-12-inline-12"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>组件</strong></th><th scope="col"><strong>权重</strong></th><th scope="col"><strong>度量方式</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(R_{\text{quality}}\)</span></td><td>0.30</td><td>LLM-as-judge（GPT-4 在清晰度、深度、严谨性上对报告打 1–10 分）</td></tr>
<tr><td><span class="arithmatex">\(R_{\text{correctness}}\)</span></td><td>0.30</td><td>代码无错误执行 + 结果可复现</td></tr>
<tr><td><span class="arithmatex">\(R_{\text{novelty}}\)</span></td><td>0.15</td><td>LLM-judge：报告是否提供了超越论文综述的洞见？</td></tr>
<tr><td><span class="arithmatex">\(R_{\text{efficiency}}\)</span></td><td>0.15</td><td>步骤越少奖励越多：<span class="arithmatex">\(R_{\text{eff}} = \max(0, 1 - \text{steps}/100)\)</span></td></tr>
<tr><td><span class="arithmatex">\(R_{\text{format}}\)</span></td><td>0.10</td><td>报告包含所有所需章节（引言、方法、结果、结论）</td></tr></tbody>
</table>
</div>

<p><strong>中间塑形</strong>：每次成功代码执行 +0.1；每次运行时错误 <span class="arithmatex">\(-\)</span>0.05（鼓励一次写对代码）。</p>
</div>

<div class="admonition warning"><p class="admonition-title">Reward Hacking 风险</p>
<ul>
<li><p><strong>伪造结果</strong>：Agent 捏造实验输出。<em>修复</em>：通过对照执行日志与报告数字，验证代码确实运行过。</p></li>
<li><p><strong>浅薄报告</strong>：Agent 逐字复制论文摘要。<em>修复</em>：新颖性 reward + 抄袭检测。</p></li>
<li><p><strong>长度博弈</strong>：长报告得分更高。<em>修复</em>：效率 reward + 长度惩罚。</p></li>
<li><p><strong>回避难题</strong>：Agent 避开困难研究问题。<em>修复</em>：带难度级别的课程。</p></li>
</ul>
</div>

## 12.7.6　训练流水线 {#training-pipeline}

<ol>
<li><p><strong>阶段 1——SFT 预热</strong>（500 步）：</p>

<ul>
<li><p>收集 200 条专家研究轨迹（人类研究者使用工具）</p></li>
<li><p>在成功轨迹上用仅 completion 掩码做 SFT（掩盖工具输出）</p></li>
<li><p>这教 Agent 工具使用语法和基本研究工作流</p></li>
</ul></li>
<li><p><strong>阶段 2——GRPO 训练</strong>（3000 步）：</p>

<ul>
<li><p>Prompt 池：跨 10 个领域（ML、NLP、CV、系统等）的 500 个研究问题</p></li>
<li><p>每个问题：生成 <span class="arithmatex">\(N=4\)</span> 条完整研究轨迹</p></li>
<li><p>用多组件 reward 给每条轨迹打分</p></li>
<li><p>GRPO 优势：<span class="arithmatex">\(\hat{A}_i = (R_i - \mu_G) / \sigma_G\)</span></p></li>
<li><p>用裁剪目标更新 Policy（clip <span class="arithmatex">\(\epsilon=0.2\)</span>，KL <span class="arithmatex">\(\beta=0.05\)</span>）</p></li>
<li><p>课程：从简单的“总结关于 X 的发现”任务开始，进展到“设计并运行关于 X 的实验”</p></li>
</ul></li>
<li><p><strong>阶段 3——拒绝采样微调</strong>（200 步）：</p>

<ul>
<li><p>每个难问题生成 16 条轨迹，按 reward 保留 top-2</p></li>
<li><p>在这些高质量轨迹上做 SFT</p></li>
<li><p>稳定在最难研究任务上的表现</p></li>
</ul></li>
</ol>

## 12.7.7　示例轨迹：完整 MDP 追踪 {#example-trajectory-full-mdp-trace}

<p>为了说明所有 MDP 组件在实践中如何协同工作，我们追踪一个从问题到提交的完整研究 Episode，并为每一步标注其形式化 MDP 元素。</p>

<div class="admonition example"><p class="admonition-title">研究 Agent 轨迹——完整 MDP 追踪</p>
<p><strong>研究问题</strong>：“将 GRPO 组大小增加到 <span class="arithmatex">\(N=16\)</span> 之上能否改善数学推理表现？”</p>

<p>下面是一条完整轨迹，每一步都有显式 MDP 符号：状态 <span class="arithmatex">\(s_t\)</span>、动作 <span class="arithmatex">\(a_t\)</span>、转移 <span class="arithmatex">\(T\)</span>、观测 <span class="arithmatex">\(o_t\)</span> 和累积 reward。</p>

<span id="table-12-inline-13"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><span class="arithmatex">\(t\)</span></th><th scope="col"><strong>元素</strong></th><th scope="col"><strong>描述</strong></th></tr></thead>
<tbody><tr><td></td><td><span class="arithmatex">\(a_0\)</span></td><td><code>think</code>：“我需要关于 GRPO 组大小消融的论文，然后进行一次受控实验。”</td></tr>
<tr><td></td><td><span class="arithmatex">\(o_0\)</span></td><td>内部推理（无工具输出）。塑形 reward：<span class="arithmatex">\(r_0 = 0\)</span>。</td></tr>
<tr><td>1</td><td><span class="arithmatex">\(s_1\)</span></td><td><span class="arithmatex">\(s_0 \oplus (a_0, o_0)\)</span>——上下文现在包含推理计划。</td></tr>
<tr><td></td><td><span class="arithmatex">\(a_1\)</span></td><td><code>search_papers(query=&quot;GRPO group size ablation math&quot;)</code></td></tr>
<tr><td></td><td><span class="arithmatex">\(o_1\)</span></td><td>返回 5 篇论文：[DAPO, DeepSeek-Math, Dr. GRPO, ...]。<span class="arithmatex">\(r_1 = 0\)</span>。</td></tr>
<tr><td>2</td><td><span class="arithmatex">\(s_2\)</span></td><td><span class="arithmatex">\(s_1 \oplus (a_1, o_1)\)</span>——搜索结果进入上下文。</td></tr>
<tr><td></td><td><span class="arithmatex">\(a_2\)</span></td><td><code>read_paper(id=&quot;DAPO&quot;, sections=[&quot;experiments&quot;, &quot;ablations&quot;])</code></td></tr>
<tr><td></td><td><span class="arithmatex">\(o_2\)</span></td><td>关键发现：“在代码任务上超过 <span class="arithmatex">\(N=16\)</span> 收益递减；数学未测试。”<span class="arithmatex">\(r_2 = 0\)</span>。</td></tr>
<tr><td>3</td><td><span class="arithmatex">\(s_3\)</span></td><td><span class="arithmatex">\(s_2 \oplus (a_2, o_2)\)</span>——识别出文献空白。</td></tr>
<tr><td></td><td><span class="arithmatex">\(a_3\)</span></td><td><code>think</code>：“DAPO 只测了代码。我将在 GSM8K 上用 Llama-3 8B 测试 <span class="arithmatex">\(N \in \{4, 8, 16, 32, 64\}\)</span>。对照：相同总训练 Token 数。”</td></tr>
<tr><td></td><td><span class="arithmatex">\(o_3\)</span></td><td>假设形成。<span class="arithmatex">\(r_3 = 0\)</span>。</td></tr>
<tr><td>4</td><td><span class="arithmatex">\(s_4\)</span></td><td><span class="arithmatex">\(s_3 \oplus (a_3, o_3)\)</span>——实验方案进入上下文。</td></tr>
<tr><td></td><td><span class="arithmatex">\(a_4\)</span></td><td><code>write_code(&quot;train_grpo_gsm8k.py&quot;)</code>——85 行脚本，含 group_size 的 argparse、正确的随机种子、在 GSM8K 测试集上评估。</td></tr>
<tr><td></td><td><span class="arithmatex">\(o_4\)</span></td><td>文件写入成功。<span class="arithmatex">\(r_4 = 0\)</span>。</td></tr>
<tr><td>5</td><td><span class="arithmatex">\(s_5\)</span></td><td><span class="arithmatex">\(s_4 \oplus (a_4, o_4)\)</span>——代码在工作区可用。</td></tr>
<tr><td></td><td><span class="arithmatex">\(a_5\)</span></td><td><code>execute_code(&quot;python train_grpo_gsm8k.py –N 4 –seed 42&quot;)</code></td></tr>
<tr><td></td><td><span class="arithmatex">\(o_5\)</span></td><td><code>Accuracy: 67.2% (CI: 65.1–69.3)</code>。<span class="arithmatex">\(r_5 = +0.1\)</span>（成功执行）。</td></tr>
<tr><td>6–8</td><td><span class="arithmatex">\(a_{6\text{--}8}\)</span></td><td>用 <span class="arithmatex">\(N=8, 16, 32\)</span> 执行 <span class="arithmatex">\(\to\)</span> 结果：71.4%、74.8%、75.9%。每个 <span class="arithmatex">\(r_t = +0.1\)</span>。</td></tr>
<tr><td>9</td><td><span class="arithmatex">\(s_9\)</span></td><td>上下文包含 4 个数据点。</td></tr>
<tr><td></td><td><span class="arithmatex">\(a_9\)</span></td><td><code>execute_code(&quot;python train_grpo_gsm8k.py –N 64 –seed 42&quot;)</code></td></tr>
<tr><td></td><td><span class="arithmatex">\(o_9\)</span></td><td><code>Accuracy: 75.6% (CI: 73.4–77.8)</code>——比 <span class="arithmatex">\(N=32\)</span> 略<em>下降</em>！<span class="arithmatex">\(r_9 = +0.1\)</span>。</td></tr>
<tr><td>10</td><td><span class="arithmatex">\(s_{10}\)</span></td><td>5 个结果全部可用。</td></tr>
<tr><td></td><td><span class="arithmatex">\(a_{10}\)</span></td><td><code>compute_stats(test=&quot;paired_t_test&quot;, data=[67.2, 71.4, 74.8, 75.9, 75.6], groups=[4,8,16,32,64])</code></td></tr>
<tr><td></td><td><span class="arithmatex">\(o_{10}\)</span></td><td><span class="arithmatex">\(N=4\)</span> 对比 <span class="arithmatex">\(N=16\)</span> 时 <span class="arithmatex">\(p &lt; 0.01\)</span>；<span class="arithmatex">\(N=32\)</span> 对比 <span class="arithmatex">\(N=64\)</span> 时 <span class="arithmatex">\(p = 0.72\)</span>（不显著）。<span class="arithmatex">\(r_{10} = 0\)</span>。</td></tr>
<tr><td>11</td><td><span class="arithmatex">\(s_{11}\)</span></td><td>收益递减的统计证据。</td></tr>
<tr><td></td><td><span class="arithmatex">\(a_{11}\)</span></td><td><code>plot_data(x=[4,8,16,32,64], y=[67.2,71.4,74.8,75.9,75.6], xlabel=&quot;Group Size N&quot;, ylabel=&quot;GSM8K Accuracy (%)&quot;, title=&quot;GRPO Group Size vs Math Reasoning&quot;)</code></td></tr>
<tr><td></td><td><span class="arithmatex">\(o_{11}\)</span></td><td>图保存到 <code>results/group_size_curve.png</code>。<span class="arithmatex">\(r_{11} = 0\)</span>。</td></tr>
<tr><td>12</td><td><span class="arithmatex">\(s_{12}\)</span></td><td>完整证据组装完成。</td></tr>
<tr><td></td><td><span class="arithmatex">\(a_{12}\)</span></td><td><code>write_report</code>：4 节报告（引言、方法、带图的结果、结论：“<span class="arithmatex">\(N=16\)</span>–32 是最优；<span class="arithmatex">\(N=64\)</span> 无增益且算力增加 4<span class="arithmatex">\(\times\)</span>”）。</td></tr>
<tr><td></td><td><span class="arithmatex">\(o_{12}\)</span></td><td>报告写入完成（2,100 词）。<span class="arithmatex">\(r_{12} = 0\)</span>。</td></tr>
<tr><td>13</td><td><span class="arithmatex">\(a_{13}\)</span></td><td><code>submit</code>——Episode 终止。</td></tr>
<tr><td></td><td><span class="arithmatex">\(R_{\text{terminal}}\)</span></td><td>LLM-judge 打分：质量 = 8/10，代码正确，新颖（把 DAPO 扩展到数学），13 步，所有章节齐全。</td></tr></tbody>
</table>
</div>

<p><strong>终端 reward 计算</strong>：</p>



<div class="arithmatex">\[\begin{multline*}
R = \underbrace{0.30 \times \tfrac{8}{10}}_{\text{质量}} + \underbrace{0.30 \times 1.0}_{\text{正确}} + \underbrace{0.15 \times \tfrac{7}{10}}_{\text{新颖}} + \underbrace{0.15 \times (1 - \tfrac{13}{100})}_{\text{效率}} + \underbrace{0.10 \times 1.0}_{\text{格式}} \\
= 0.24 + 0.30 + 0.105 + 0.13 + 0.10 = \mathbf{0.875}
\end{multline*}\]</div>

<p><strong>中间塑形总额</strong>：<span class="arithmatex">\(5 \times (+0.1) = +0.5\)</span>（5 次成功代码执行）。</p>

<p><strong>GRPO 上下文</strong>：此轨迹在 <span class="arithmatex">\(N=4\)</span> 组中得分最高（其他得分为 0.61、0.72、0.53）。GRPO 优势：</p>



<div class="arithmatex">\[\hat{A} = \frac{0.875 - \bar{R}}{\sigma_R} = \frac{0.875 - 0.684}{0.129} = +1.48 \quad \text{（强化强烈）}\]</div>

<p><strong>所体现的关键 MDP 性质</strong>：</p>

<ul>
<li><p><strong>确定性 <span class="arithmatex">\(T\)</span></strong>：每次工具调用产生可预测的状态扩展（<span class="arithmatex">\(s_{t+1} = s_t \oplus (a_t, o_t)\)</span>）。</p></li>
<li><p><strong>稀疏终端 reward</strong>：真正的质量信号只在 <code>submit</code> 时给出；中间塑形很小。</p></li>
<li><p><strong>长视野</strong>：13 步且 <span class="arithmatex">\(\gamma = 1.0\)</span>（Episode 式任务不折扣）。</p></li>
<li><p><strong>自我纠正机会</strong>：在第 9 步，Agent 观察到 <span class="arithmatex">\(N=64\)</span> 并未改进——据此调整其结论，而非挑选数据。</p></li>
<li><p><strong>动作多样性</strong>：推理（<code>think</code>）、信息收集（<code>search</code>、<code>read</code>）、执行（<code>write_code</code>、<code>execute</code>）、分析（<code>compute_stats</code>、<code>plot</code>）和输出（<code>write_report</code>、<code>submit</code>）的混合。</p></li>
</ul>
</div>

## 12.7.8　关键设计决策与权衡 {#key-design-decisions-and-tradeoffs}

<p class="table-caption" id="table-12-8"><span class="table-number">表 12.8　</span>研究 Agent 的设计决策，映射到论文各章节。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>决策</strong></th><th scope="col"><strong>论文章节</strong></th><th scope="col"><strong>理由</strong></th></tr></thead>
<tbody><tr><td>QLoRA（<span class="arithmatex">\(r=32\)</span>）</td><td>LoRA 一节</td><td>72B 模型；全量微调太昂贵。<span class="arithmatex">\(r=32\)</span> 用于复杂推理。</td></tr>
<tr><td>GRPO（而非 PPO）</td><td>GRPO 一节</td><td>无需 Value 模型；研究质量难以中途预测。</td></tr>
<tr><td>稀疏终端 reward</td><td>Reward Shaping</td><td>研究质量只能在完成时度量；中间塑形最小。</td></tr>
<tr><td><span class="arithmatex">\(N=4\)</span> 轨迹</td><td>GRPO 组大小</td><td>平衡：足够的多样性用于排序，又不至过于昂贵（100 步轨迹）。</td></tr>
<tr><td>128K 上下文</td><td>FlashAttention</td><td>长轨迹包含论文内容 + 代码 + 结果。</td></tr>
<tr><td>vLLM + 前缀缓存</td><td>vLLM 一节</td><td>系统 Prompt + 研究问题在 4 条 rollout 间共享。</td></tr>
<tr><td>课程训练</td><td>Agent RL</td><td>从简单（文献综述）<span class="arithmatex">\(\to\)</span> 困难（设计 + 执行实验）。</td></tr>
<tr><td>LLM-as-judge reward</td><td>Reward Model</td><td>研究质量是主观的；LLM judge 比基于规则更灵活。</td></tr></tbody>
</table>
</div>

## 12.7.9　评估 {#evaluation-dup2}

<div class="admonition note"><p class="admonition-title">研究 Agent 评估框架</p>
<ul>
<li><p><strong>保留问题</strong>（50 个）：训练期间未见的研究问题，覆盖多种领域。</p></li>
<li><p><strong>人类评估</strong>：领域专家在 1–5 分制（质量、正确性、可操作性）上对报告评分。</p></li>
<li><p><strong>可复现性</strong>：重新运行报告中的 Agent 代码；验证结果一致。</p></li>
<li><p><strong>对比基线</strong>：(1) 零样本 GPT-4 + 工具（无 RL 训练），(2) 仅 SFT Agent，(3) 人类研究者。</p></li>
<li><p><strong>效率指标</strong>：按任务难度归一化的完成步数。</p></li>
</ul>

<p><strong>预期结果</strong>（基于类似的 Agent RL 工作）：</p>

<span id="table-12-inline-15"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>Agent</strong></th><th scope="col"><strong>报告质量（1–5）</strong></th><th scope="col"><strong>平均步数</strong></th></tr></thead>
<tbody><tr><td>零样本 GPT-4 + 工具</td><td>2.8</td><td>25</td></tr>
<tr><td>仅 SFT</td><td>3.4</td><td>18</td></tr>
<tr><td>GRPO 训练（我们的）</td><td>4.1</td><td>14</td></tr>
<tr><td>人类研究者</td><td>4.5</td><td>N/A</td></tr></tbody>
</table>
</div>
</div>

## 12.7.10　经验教训与失败模式 {#lessons-and-failure-modes}

<div class="admonition warning"><p class="admonition-title">研究 Agent 训练中的常见失败</p>
<ul>
<li><p><strong>无限循环</strong>：Agent 反复搜索论文却不前进。<em>修复</em>：步数预算 + 对参数相同的重复工具调用的惩罚。</p></li>
<li><p><strong>代码调试螺旋</strong>：Agent 花 20+ 步修一个 Bug。<em>修复</em>：重试次数上限为 3；若代码失败 3 次，放弃当前方法尝试替代方案。</p></li>
<li><p><strong>幻觉引用</strong>：Agent 编造论文标题/结果。<em>修复</em>：通过工具输出验证所有引用存在；惩罚不可验证的断言。</p></li>
<li><p><strong>过早提交</strong>：Agent 提交不完整报告以避免长轨迹的惩罚。<em>修复</em>：设定最低质量阈值（<span class="arithmatex">\(R &gt; 0.4\)</span>）作为有效提交；低于阈值视为失败。</p></li>
<li><p><strong>对 judge 的 reward hacking</strong>：Agent 学会产生在 LLM judge 处得高分但科学上肤浅的文本。<em>修复</em>：轮换 judge 模型；定期在 reward 中纳入人类评估。</p></li>
</ul>
</div>
