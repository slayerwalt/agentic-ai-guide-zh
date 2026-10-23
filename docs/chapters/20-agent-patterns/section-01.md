# 20.1　工作流模式 {#workflow-patterns}

<p>以下模式改编自 Anthropic 对 Agent 构建模块的分类 <a class="book-citation" id="cite-anthropic2024buildingagents-1" href="../../references.md#ref-anthropic2024buildingagents" aria-label="参考文献 10">[10]</a>，在<em>预定义</em>的控制流中使用 LLM。由系统（而非模型）决定执行顺序。</p>

## 20.1.1　Prompt 链（Prompt Chaining） {#prompt-chaining}

<p>最简单的模式：将复杂任务拆分为固定序列的 LLM 调用，将一个调用的结果作为上下文传入下一个调用。步骤之间的校验门可在错误向下游传播之前及早捕获它们。</p>

<figure id="figure-20-1"><a href="../../assets/figures/fig_060_fig60.png" target="_blank" rel="noopener" aria-label="查看图 20.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_060_fig60.png" alt="带质量门的 Prompt 链。每一步都是独立的 LLM 调用。质量门可以基于 LLM，也可以基于程序逻辑。" loading="lazy"></a><figcaption><span class="figure-number">图 20.1　</span>带质量门的 Prompt 链。每一步都是独立的 LLM 调用。质量门可以基于 LLM，也可以基于程序逻辑。</figcaption></figure>

<p><strong>适用场景</strong>：天然顺序化的任务——内容生成、数据转换、多阶段分析。</p>

<p><strong>关键优势</strong>：每一步可以使用不同的 Prompt、模型或温度。中间结果可检查、可调试。</p>

## 20.1.2　路由（Routing） {#routing}

<p>由分类器（LLM 或传统方法）检查输入，并分派到专门的处理器。</p>

<figure id="figure-20-2"><a href="../../assets/figures/fig_061_fig61.png" target="_blank" rel="noopener" aria-label="查看图 20.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_061_fig61.png" alt="路由模式：输入被分类一次，然后由专门的处理器处理。" loading="lazy"></a><figcaption><span class="figure-number">图 20.2　</span>路由模式：输入被分类一次，然后由专门的处理器处理。</figcaption></figure>

<p><strong>适用场景</strong>：不同任务类型对应不同的最佳 Prompt、工具或模型。如客服分诊、多模态输入处理。</p>

## 20.1.3　并行化（Parallelization） {#parallelization}

<p>多个 LLM 调用并发运行，由一个程序层合并它们的输出。可分为两个子模式：</p>

<ul>
<li><p><strong>分段（Sectioning，fan-out）</strong>：将输入划分为互不相交的块并独立处理——例如对一个代码库同时运行安全、性能和风格检查。</p></li>
<li><p><strong>投票（Voting，冗余）</strong>：用不同随机种子或温度对同一 Prompt 发起 <span class="arithmatex">\(N\)</span> 次调用，然后通过多数投票 <a class="book-citation" id="cite-wang2022selfconsistency-1" href="../../references.md#ref-wang2022selfconsistency" aria-label="参考文献 368">[368]</a>、奖励模型打分或 LLM-as-judge 选出最佳结果。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">并行化示例：代码评审</p>
<ol>
<li><p><strong>并行调用</strong>：安全评审 <span class="arithmatex">\(\|\)</span> 性能评审 <span class="arithmatex">\(\|\)</span> 风格评审</p></li>
<li><p><strong>聚合</strong>：合并所有发现，去重，按严重程度排序</p></li>
</ol>

<p>延迟为 <span class="arithmatex">\(\max\)</span>（各调用），而非 <span class="arithmatex">\(\sum\)</span>（各调用）。</p>
</div>

## 20.1.4　Orchestrator-Workers {#orchestrator-workers}

<p>在该模式下，由 LLM 自身决定如何拆分工作。一个 Orchestrator 模型分析任务、产出子任务计划、将每个子任务分派给 Worker LLM（可能使用不同的 Prompt 或工具），最终将它们的输出合并为一致的结果。与并行化的关键区别在于：分解逻辑是由模型生成的，而非硬编码的。</p>

<figure id="figure-20-3"><a href="../../assets/figures/fig_062_fig62.png" target="_blank" rel="noopener" aria-label="查看图 20.3 原图" title="点击查看原图"><img src="../../assets/figures/fig_062_fig62.png" alt="Orchestrator-workers：LLM 决定如何分解任务，并综合各 Worker 的结果。" loading="lazy"></a><figcaption><span class="figure-number">图 20.3　</span>Orchestrator-workers：LLM 决定如何分解任务，并综合各 Worker 的结果。</figcaption></figure>

<p><strong>适用场景</strong>：开放式问题，子任务的数量和性质无法在设计时穷举——例如“重构这个代码库”需要先理解依赖图，再决定要修改哪些文件。</p>

## 20.1.5　Evaluator-Optimizer {#evaluator-optimizer}

<p>一个双模型反馈循环 <a class="book-citation" id="cite-madaan2023selfrefine-1" href="../../references.md#ref-madaan2023selfrefine" aria-label="参考文献 246">[246]</a>：生成器产出候选输出，由独立的评估器依据显式标准对其打分。若得分低于阈值，则将评估器的评论追加到生成器的上下文中，循环往复，直到达到质量标准或耗尽重试预算。</p>

<figure id="figure-20-4"><a href="../../assets/figures/fig_063_fig63.png" target="_blank" rel="noopener" aria-label="查看图 20.4 原图" title="点击查看原图"><img src="../../assets/figures/fig_063_fig63.png" alt="Evaluator-optimizer：无需训练的迭代式精化。" loading="lazy"></a><figcaption><span class="figure-number">图 20.4　</span>Evaluator-optimizer：无需训练的迭代式精化。</figcaption></figure>

<p><strong>适用场景</strong>：具有明确质量标准的任务——必须通过测试的代码、必须保留语义的翻译、必须符合风格指南的写作。</p>
