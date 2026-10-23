# 21.5　构建自定义环境 {#sec:custom-env}

## 21.5.1　面向 LLM Agent 的 Gymnasium 风格 API {#gymnasium-style-api-for-llm-agents}

<p>Gymnasium API <a class="book-citation" id="cite-towers2024gymnasium-1" href="../../references.md#ref-towers2024gymnasium" aria-label="参考文献 355">[355]</a>（OpenAI Gym 的继任者）是 RL 环境的事实标准。将其适配到 LLM Agent 需要两处修改：(1) 观察和动作是字符串（或包含字符串的字典）而非数值数组；(2) <code>step</code> 方法必须处理异步工具执行。</p>

## 21.5.2　奖励函数工程 {#reward-function-engineering}

<p>LLM Agent 环境的奖励函数通常是<em>基于执行</em>的：环境在每个 Episode 后运行一个验证器，若任务被解决则返回 1，否则返回 0。对于没有明确验证器的任务，选项包括：</p>

<ul>
<li><p><strong>LLM-as-judge</strong>：由一个独立的 LLM 根据任务描述对 Agent 的最终状态打分。</p></li>
<li><p><strong>基于评分量表（Rubric）</strong>：结构化的评分量表将任务分解为若干子标准，每项独立打分。</p></li>
<li><p><strong>人工标注</strong>：由人类评估者对一组随机抽样的轨迹打分；这些分数用于校准一个自动化代理指标。</p></li>
</ul>

## 21.5.3　状态管理与检查点 {#state-management-and-checkpointing}

<p>长时程任务可能需要数小时的墙钟时间。环境应支持：</p>

<ul>
<li><p><strong>状态序列化</strong>：完整的环境状态（文件系统、浏览器 cookies、数据库内容）可序列化到磁盘并恢复。</p></li>
<li><p><strong>Episode 中检查点</strong>：Agent 可在任意步骤保存检查点并从中恢复，从而支持树搜索式探索。</p></li>
<li><p><strong>轨迹日志</strong>：每个观察、动作和奖励都被记录到结构化文件中，供离线分析和奖励模型训练使用。</p></li>
</ul>

## 21.5.4　用于训练数据采集的并行化 {#parallelization-for-training-data-collection}

<p>通过 RL 训练 LLM Agent 需要数百万次环境交互。并行化策略包括：</p>

<ul>
<li><p><strong>进程级并行</strong>：派生 <span class="arithmatex">\(N\)</span> 个独立的环境进程；并行采集轨迹。</p></li>
<li><p><strong>异步 rollout worker</strong>：使用异步事件循环（如 <code>asyncio</code>）将 LLM 推理延迟与环境执行重叠。</p></li>
<li><p><strong>向量化环境</strong>：将多个环境批处理到单次 <code>step</code> 调用中，摊销 Python 开销。</p></li>
<li><p><strong>云原生扩展</strong>：使用作业调度器（Ray、SLURM）在集群中分发环境 worker，由中心化的回放缓冲（replay buffer）聚合轨迹。</p></li>
</ul>
