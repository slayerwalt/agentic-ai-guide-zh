# 30.2　前路：开放挑战 {#the-road-ahead-open-challenges}

## 30.2.1　从交互中学习 {#learning-from-interaction}

<p>当前的 RLHF 流水线 <a class="book-citation" id="cite-ouyang2022training-1" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a> 将对齐视为一次性的训练阶段。未来则指向<strong>从部署中持续学习</strong>：Agent 能够从每一次用户交互、工具失败与环境观察中改进——同时避免灾难性遗忘（Catastrophic Forgetting） <a class="book-citation" id="cite-kirkpatrick2017overcoming-1" href="../../references.md#ref-kirkpatrick2017overcoming" aria-label="参考文献 186">[186]</a> 与奖励漂移。关键开放问题：</p>

<ul>
<li><p>非平稳奖励分布下的在线学习。</p></li>
<li><p>生产环境中的安全探索 <a class="book-citation" id="cite-garcia2015comprehensive-1" href="../../references.md#ref-garcia2015comprehensive" aria-label="参考文献 107">[107]</a>（在学习过程中避免有害动作）。</p></li>
<li><p>在长 Agent 轨迹（数百次工具调用）上的高效信用分配。</p></li>
</ul>

## 30.2.2　可扩展监督 {#scalable-oversight}

<p>当 Agent 能力越来越强时，人类监督本身成为瓶颈。当前方法（RLHF <a class="book-citation" id="cite-ouyang2022training-2" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a>、Constitutional AI <a class="book-citation" id="cite-bai2022constitutional-1" href="../../references.md#ref-bai2022constitutional" aria-label="参考文献 16">[16]</a>）依赖人类评估模型输出——但当模型输出超出人类理解能力时会发生什么？</p>

<ul>
<li><p><strong>递归奖励建模（Recursive Reward Modeling）</strong> <a class="book-citation" id="cite-christiano2017deep-1" href="../../references.md#ref-christiano2017deep" aria-label="参考文献 57">[57]</a>：用 AI 协助人类评估 AI。</p></li>
<li><p><strong>辩论与放大（Debate and Amplification）</strong> <a class="book-citation" id="cite-irving2018debate-1" href="../../references.md#ref-irving2018debate" aria-label="参考文献 157">[157]</a>：两个模型相互辩论，人类裁判哪一方论证更令人信服。</p></li>
<li><p><strong>基于过程的监督（Process-based Supervision）</strong> <a class="book-citation" id="cite-lightman2023lets-1" href="../../references.md#ref-lightman2023lets" aria-label="参考文献 215">[215]</a>：奖励正确的推理步骤，而不仅是最终答案。</p></li>
<li><p><strong>机制可解释性（Mechanistic Interpretability）</strong> <a class="book-citation" id="cite-olsson2022context-1" href="../../references.md#ref-olsson2022context" aria-label="参考文献 273">[273]</a>：理解模型内部<em>在做什么</em>，而不仅是它输出了什么。</p></li>
</ul>

## 30.2.3　世界模型与规划 {#world-models-and-planning}

<p>当前 Agent 是反应式的——它们一次只观察并响应一步。未来 Agent 将需要<strong>内部世界模型（World Models）</strong> <a class="book-citation" id="cite-hafner2020dream-1" href="../../references.md#ref-hafner2020dream" aria-label="参考文献 130">[130]</a> 以支持前瞻式规划：</p>

<ul>
<li><p>在执行动作之前预测其后果。</p></li>
<li><p>对可能动作序列进行树搜索（类似 AlphaGo <a class="book-citation" id="cite-silver2016mastering-1" href="../../references.md#ref-silver2016mastering" aria-label="参考文献 331">[331]</a> 与 MuZero <a class="book-citation" id="cite-schrittwieser2020mastering-1" href="../../references.md#ref-schrittwieser2020mastering" aria-label="参考文献 316">[316]</a>，但面向开放任务）。</p></li>
<li><p>从交互轨迹中学习环境的状态转移。</p></li>
</ul>

## 30.2.4　多智能体生态 {#multi-agent-ecosystems}

<p>A2A 协议 <a class="book-citation" id="cite-google-a2a-2025-1" href="../../references.md#ref-google-a2a-2025" aria-label="参考文献 113">[113]</a> 与多智能体框架预示着这样一种未来：成百上千的专用 Agent 协作、协商并相互委派——形成所谓的“智能体经济（Economy of Agents）” <a class="book-citation" id="cite-nisan2007algorithmic-1" href="../../references.md#ref-nisan2007algorithmic" aria-label="参考文献 267">[267]</a>。开放挑战包括：</p>

<ul>
<li><p>服务于不同委托方的 Agent 之间的信任与验证。</p></li>
<li><p>竞争性场景中涌现的合作与涌现的欺骗的对比 <a class="book-citation" id="cite-hubinger2024sleeper-1" href="../../references.md#ref-hubinger2024sleeper" aria-label="参考文献 153">[153]</a>。</p></li>
<li><p>资源分配（算力、工具访问、优先级）的市场机制。</p></li>
<li><p>治理：当一条由 10 个 Agent 组成的链路产生有害结果时，由谁负责？ <a class="book-citation" id="cite-amodei2016concrete-1" href="../../references.md#ref-amodei2016concrete" aria-label="参考文献 6">[6]</a></p></li>
</ul>

## 30.2.5　智能体安全与信任 {#agent-security-and-trust}

<p>自主 Agent 继承了其底层 LLM 的全部安全漏洞，再加上工具访问、多智能体委派与持久化记忆所带来的新攻击面（第 19–21 章）。关键的未解问题：</p>

<ul>
<li><p><strong>规模化的 Prompt 注入</strong> <a class="book-citation" id="cite-greshake2023indirect-1" href="../../references.md#ref-greshake2023indirect" aria-label="参考文献 120">[120]</a>：随着 Agent 消费不可信内容（网页、邮件、API 响应），间接 prompt 注入成为系统性问题。目前尚无稳健的防御手段。</p></li>
<li><p><strong>混淆代理（Confused Deputy）攻击</strong>：持有合法凭证的 Agent 可能被骗，代表潜伏在数据流中的攻击者滥用其凭证 <a class="book-citation" id="cite-anthropic-mcp-2024-1" href="../../references.md#ref-anthropic-mcp-2024" aria-label="参考文献 9">[9]</a>。</p></li>
<li><p><strong>不削弱能力的沙箱化</strong>：最小权限执行限制了 Agent 能做的事，但过度严格的沙箱会抹杀智能体的价值。如何找到合适边界仍是开放设计问题。</p></li>
<li><p><strong>审计与归因</strong>：当 Agent 链跨越多个组织（通过 A2A <a class="book-citation" id="cite-google-a2a-2025-2" href="../../references.md#ref-google-a2a-2025" aria-label="参考文献 113">[113]</a>）时，追溯<em>谁</em>授权了<em>什么</em>动作在架构上仍未解决。</p></li>
<li><p><strong>信任校准</strong>：Agent 必须学会何时<em>不要</em>信任——工具响应是否真实、其他 Agent 的声明是否经过验证。</p></li>
</ul>

## 30.2.6　超越基准的评估 {#evaluation-beyond-benchmarks}

<p>第 14 章揭示了基准塑造了我们所构建的系统——然而当前评估仍存在关键缺口：</p>

<ul>
<li><p><strong>真实部署指标</strong>：SWE-bench <a class="book-citation" id="cite-jimenez2024swebench-1" href="../../references.md#ref-jimenez2024swebench" aria-label="参考文献 169">[169]</a> 与 GAIA <a class="book-citation" id="cite-mialon2023gaia-1" href="../../references.md#ref-mialon2023gaia" aria-label="参考文献 254">[254]</a> 等基准衡量的是孤立任务；而生产 Agent 面对的是模糊目标、变动的需求与多轮恢复。</p></li>
<li><p><strong>奖励模型有效性</strong>：RLHF 假设 RM 能捕捉人类偏好，但奖励黑客 <a class="book-citation" id="cite-skalse2022defining-1" href="../../references.md#ref-skalse2022defining" aria-label="参考文献 333">[333]</a> 与分布漂移会在规模化时侵蚀这一假设。</p></li>
<li><p><strong>成本—质量前沿</strong>：两个 Agent 可能取得相同准确率，但其中一个的 token 成本高出 10<span class="arithmatex">\(\times\)</span>。评估必须开始关注成本。</p></li>
<li><p><strong>分布漂移下的安全性</strong>：在测试中安全的 Agent 在新输入下可能行为不安全。智能体规模上的对抗评估 <a class="book-citation" id="cite-perez2022red-1" href="../../references.md#ref-perez2022red" aria-label="参考文献 290">[290]</a> 与红队测试仍不成熟。</p></li>
</ul>

## 30.2.7　效率与可获得性 {#efficiency-and-accessibility}

<p>对 70B 模型做 RLHF 训练成本在 $10K–$100K。运行一个自主 Agent 在每个复杂任务上的成本是 $1–$50。要让智能体 AI 产生广泛影响，需要：</p>

<ul>
<li><p>将智能体能力从大模型蒸馏到小模型 <a class="book-citation" id="cite-hinton2015distilling-1" href="../../references.md#ref-hinton2015distilling" aria-label="参考文献 139">[139]</a><a class="book-citation" id="cite-kim2024distillation-1" href="../../references.md#ref-kim2024distillation" aria-label="参考文献 184">[184]</a>。</p></li>
<li><p>更高效的 RL 算法（更少样本、更低方差） <a class="book-citation" id="cite-schulman2017proximal-1" href="../../references.md#ref-schulman2017proximal" aria-label="参考文献 319">[319]</a>。</p></li>
<li><p>无需云端往返即可运行的端侧 Agent。</p></li>
<li><p>在智能体任务上达到闭源质量的开放权重模型 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a>。</p></li>
</ul>
