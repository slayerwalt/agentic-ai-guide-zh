# 21.10　小结 {#sec:env-summary}

<p>Agent 环境是训练和评估 LLM Agent 的基底。本节的关键要点是：</p>

<ol>
<li><p><strong>环境不是可选项。</strong> 安全探索、可复现评估和课程学习都需要结构化环境。若没有环境，Chatbot 评估与 Agent 评估之间的鸿沟无法跨越。</p></li>
<li><p><strong>四个维度都需谨慎设计。</strong> 观察空间、动作空间、奖励信号和 Episode 结构各有可能让整个基准失效的失败模式。</p></li>
<li><p><strong>生态丰富但碎片化。</strong> 代码沙箱、Web 环境、计算机使用类环境、SWE 环境、科研环境、游戏和多 Agent 竞技场各测试不同能力。任何单一环境都不充分。</p></li>
<li><p><strong>标准化很重要。</strong> OpenEnv <a class="book-citation" id="cite-huggingface2025openenv-1" href="../../references.md#ref-huggingface2025openenv" aria-label="参考文献 88">[88]</a> 提供 Gymnasium 风格的 API，配合 Docker 隔离以及把 Hugging Face Spaces 作为注册中心——降低了构建新环境以及在不同环境间比较 Agent 的成本。</p></li>
<li><p><strong>人类差距真实存在但正在收窄。</strong> 当前 LLM Agent 在大多数基准上达到人类性能的 20–50%。进展最快的是训练数据充裕的领域（代码），最慢的是需要细粒度感知的领域（GUI 控制）。</p></li>
</ol>

<div class="admonition question"><p class="admonition-title">Agent 环境中的开放研究问题</p>
<ul>
<li><p>如何为那些正确性带有主观性或上下文依赖的任务设计奖励函数？</p></li>
<li><p>单一 Agent 架构能否在无需任务特定微调的前提下跨基于文本和多模态环境泛化？</p></li>
<li><p>训练所需的环境保真度应当达到何种程度？在简化的仿真器上训练能迁移到真实部署吗？</p></li>
<li><p>当 LLM 在可能包含基准答案的越来越大的 Web 语料上训练时，如何防止基准污染？</p></li>
</ul>
</div>

## 21.10.1　新兴基准（2026） {#sec:emerging-benchmarks-2026}

#### UniClawBench。 {#heading-21-49}

<p>UniClawBench <a class="book-citation" id="cite-uniclawbench2026-1" href="../../references.md#ref-uniclawbench2026" aria-label="参考文献 214">[214]</a> 在实时、隔离的 Docker 容器中运行 400 个双语真实世界任务——跨五个功能领域测试 Agent，包括跨平台协调。其标志性特征是多 Agent 反馈回路：一个隐藏的监督者与一个模拟用户注入真实世界的摩擦（含混的请求、不断变化的需求、部分失败），衡量的是<em>交互韧性</em>，而非一次性任务完成率。截至 2026 年年中，它是评估生产就绪 Agent 的最严格基准。</p>

#### Long-Horizon-Terminal-Bench。 {#heading-21-50}

<p>Long-Horizon-Terminal-Bench <a class="book-citation" id="cite-terminalbench2026-1" href="../../references.md#ref-terminalbench2026" aria-label="参考文献 386">[386]</a> 覆盖 21 个领域的 46 个复杂任务，测试 Agent 能否驾驭以小时而非分钟计的终端操作。其关键创新是<strong>基于密集奖励的评分</strong>：在每一步给中间进展打分，而不是依赖终点线的二元成功。这暴露了一个残酷的现实：即使是最强的前沿模型，也无法连续执行超过 <span class="arithmatex">\(\sim\)</span>15 条终端命令而不跑偏——揭示长时程顺序执行仍是一个根本性的未解挑战。</p>
