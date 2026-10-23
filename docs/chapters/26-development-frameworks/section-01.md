# 26.1　动机：工程能力的鸿沟 {#subsec:engineering-gap}

<div class="admonition note"><p class="admonition-title">为什么 Agent 工程化如此困难</p>
<p>在 Jupyter notebook 中搭一个能干活的 Agent 并不难；但要让它在生产环境中稳定运行——处理边界情况、从故障中恢复、随负载扩展并持续改进——则需要完全不同的工程纪律。</p>
</div>

<p>研究原型通常假设一个友善的环境：格式良好的输入、可用的 Tool、响应及时的 API，以及一位耐心的人类观察者随时准备在出错时重启流程。生产 Agent 享受不到这些奢侈条件。原型与生产之间的工程鸿沟体现在多个维度：</p>

#### 可靠性。 {#reliability.}

<p>生产级 Agent 必须优雅处理 Tool 失败、从部分状态损坏中恢复，并避免死循环或失控的 API 调用。错误处理必须是系统化的，而不能临时拼凑。</p>

#### 可观测性。 {#observability.}

<p>当 Agent 给出错误答案或采取意外行动时，运维人员需要弄清楚<em>原因</em>。这要求对每一次 LLM 调用、Tool 调用和状态迁移都进行结构化日志记录，而不仅仅记录最终输出。</p>

#### 可测试性。 {#testability.}

<p>Agent 行为是非确定性且依赖上下文的，传统的单元测试远不足以覆盖。完整的 Agent 测试需要专门的评测框架、黄金轨迹（golden trajectory）对比，以及行为测试套件。</p>

#### 部署。 {#deployment.}

<p>Agent 是有状态、长时间运行的进程，单次执行可能跨越数分钟乃至数小时。服务基础设施必须支持异步执行、检查点、故障后续跑（resumption），以及多租户隔离。</p>

#### 迭代。 {#iteration.}

<p>随着外部世界变化、API 演进以及用户行为漂移，生产 Agent 的表现会随时间退化。持续改进需要系统化的失败分析、Prompt 版本管理和微调流水线。</p>

<div class="admonition tip"><p class="admonition-title">Agent 开发成熟度模型</p>
<p>Agent 开发遵循如下成熟度演进路径：</p>

<ol>
<li><p><strong>原型阶段（Prototype）</strong>：单文件脚本、硬编码 Prompt、人工测试</p></li>
<li><p><strong>Alpha 阶段</strong>：模块化代码、基本错误处理、人工评测</p></li>
<li><p><strong>Beta 阶段</strong>：基于框架、自动化测试、预发布环境</p></li>
<li><p><strong>生产阶段（Production）</strong>：完整可观测性、CI/CD、自动扩缩容、服务等级协议（SLA）</p></li>
<li><p><strong>成熟阶段（Mature）</strong>：持续学习、A/B 测试、自我改进闭环</p></li>
</ol>

<p>大多数团队都低估了从第 2 阶段到第 3 阶段的跨度。</p>
</div>
