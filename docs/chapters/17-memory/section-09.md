# 17.9　评估记忆系统 {#sec:memory-evaluation}

<p>评估智能体记忆颇具挑战，因为记忆操作的质量只能<em>间接</em>显现——通过长时间跨度的下游任务表现。一个对所存事实有完美召回率的记忆系统，若检索出无关上下文或撑爆 LLM 的上下文窗口，仍可能整体失败。</p>

## 17.9.1　评估维度 {#evaluation-dimensions}

<p>LongMemEval <a class="book-citation" id="cite-wu2024longmemeval-1" href="../../references.md#ref-wu2024longmemeval" aria-label="参考文献 384">[384]</a> 提出长期记忆系统必须展现的五项核心能力：</p>

<ol>
<li><p><strong>信息抽取。</strong> 系统能否从对话轮次中识别并存储显著事实？通过事实召回率衡量：有多少真值事实可从记忆中恢复？</p></li>
<li><p><strong>跨会话推理。</strong> 系统能否综合分散在多次过往会话中的信息？例如：“根据我们上周和昨天的对话，项目范围发生了哪些变化？”</p></li>
<li><p><strong>时序推理。</strong> 系统能否正确回答与时间相关的查询？例如“在组织调整<em>之前</em>我说我的优先事项是什么？”这就要求区分不同时间的状态。</p></li>
<li><p><strong>知识更新。</strong> 当事实发生变化时（用户搬到新城市、偏好改变），记忆能否反映最新状态，同时保留历史？</p></li>
<li><p><strong>拒答（Abstention）。</strong> 当系统并无相关记忆时，它能否正确地说“我不知道”，而不是幻觉出一段看似合理却完全是捏造的回忆？</p></li>
</ol>

## 17.9.2　基准（Benchmarks） {#benchmarks}

<p class="table-caption" id="table-17-3"><span class="table-number">表 17.3　</span>评估智能体记忆系统的基准。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>基准</strong></th><th scope="col"><strong>发表会议/期刊</strong></th><th scope="col"><strong>规模</strong></th><th scope="col"><strong>关注点</strong></th></tr></thead>
<tbody><tr><td>LongMemEval <a class="book-citation" id="cite-wu2024longmemeval-2" href="../../references.md#ref-wu2024longmemeval" aria-label="参考文献 384">[384]</a></td><td>ICLR 2025</td><td>500 个问题，可扩展的历史</td><td>五项记忆能力；多会话对话</td></tr>
<tr><td>LOCOMO <a class="book-citation" id="cite-maharana2024locomo-1" href="../../references.md#ref-maharana2024locomo" aria-label="参考文献 247">[247]</a></td><td>EMNLP 2024</td><td>多会话对话</td><td>对话之上的单跳、时序、多跳与开放域问答</td></tr>
<tr><td>InfiniteBench <a class="book-citation" id="cite-zhang2024infinitebench-1" href="../../references.md#ref-zhang2024infinitebench" aria-label="参考文献 428">[428]</a></td><td>ACL 2024</td><td>100K+ Token 上下文</td><td>长上下文召回，并非专门为记忆设计，但能测试上限</td></tr></tbody>
</table>
</div>

## 17.9.3　度量指标 {#metrics}

#### 记忆层面的指标。 {#memory-level-metrics.}

<ul>
<li><p><strong>记忆召回率（Memory Recall）</strong>：<span class="arithmatex">\(\frac{\text{可从记忆中检索到的真值事实数}}{\text{真值事实总数}}\)</span>。衡量存储的完备程度。</p></li>
<li><p><strong>记忆精确率（Memory Precision）</strong>：<span class="arithmatex">\(\frac{\text{top-}k\text{ 检索中相关条目数}}{k}\)</span>。衡量检索的噪声。</p></li>
<li><p><strong>延迟（Latency）</strong>：从查询到取回上下文的耗时（p50 与 p95）。</p></li>
<li><p><strong>Token 效率</strong>：每次查询注入上下文的 Token 总数。越低越好——多余的上下文既降低 LLM 准确率又增加成本。</p></li>
</ul>

#### 下游指标。 {#downstream-metrics.}

<ul>
<li><p><strong>回答准确率</strong>：在记忆条件下最终回答的正确性（EM、F1 或 LLM-as-judge）。</p></li>
<li><p><strong>忠实性（Faithfulness）</strong>：回答是否准确反映记忆中的内容，而没有捏造？</p></li>
<li><p><strong>个性化质量</strong>：用户满意度，通过偏好评分或在记忆增强与无记忆系统之间做 A/B 测试来衡量。</p></li>
<li><p><strong>自相矛盾率</strong>：系统给出与此前所述事实不一致回答的频率。</p></li>
</ul>

#### 运维层面的指标。 {#operational-metrics.}

<ul>
<li><p><strong>写入选择性</strong>：触发记忆写入的轮次比例。过高 <span class="arithmatex">\(\to\)</span> 噪声；过低 <span class="arithmatex">\(\to\)</span> 信息缺口。</p></li>
<li><p><strong>过时率（Staleness）</strong>：尽管已存在更新，仍检索到过时事实的频率。</p></li>
<li><p><strong>存储增长率</strong>：每小时交互所存储的 Token 数。无界增长是不可持续的。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">评估鸿沟</p>
<p>大多数记忆论文都在短基准（10–50 个会话）上做评估。真实的生产 Agent 要运行数月、跨越数千个会话。长周期评估——记忆漂移、矛盾累积与存储膨胀在那里成为主要失败模式——仍是一个开放挑战。从业者应在基准分数之外，对运维指标进行纵向监控。</p>
</div>
