# 12.3　操作范式 {#operational-paradigms}

<p>LLM Agent 通过三种主要的优化方法论来利用专门的轨迹缓冲区：</p>

## 12.3.1　A. 自我纠正与思维细化 {#a.-self-correction-and-thought-refinement-star-reflexion}

<p>此类别中的两种代表性方法是 STaR <a class="book-citation" id="cite-zelikman2022star-1" href="../../references.md#ref-zelikman2022star" aria-label="参考文献 418">[418]</a> 和 Reflexion <a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a>。当 Agent 在一次多步执行轨迹中失败时，次优序列会被保存到缓冲区。该框架随后采样此轨迹，并提示 LLM 对其过去的表现生成显式文本批评：</p>



<div class="arithmatex">\[\text{Critique} \leftarrow \text{LLM}(\mathcal{S}_{\text{failed}},\; \mathcal{A}_{\text{failed}},\; \mathcal{R}_{=0})
\tag{12.3}\]</div>

<p>一旦修正后的轨迹获得正向 reward，它就被移至最优经验池，用于通过微调（在成功轨迹上做 SFT）或 RL（采用二元通过/失败 reward 的 GRPO <a class="book-citation" id="cite-shao2024deepseekmath-1" href="../../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a>）来更新网络权重。</p>

<div class="admonition note"><p class="admonition-title">STaR：自学习推理者</p>
<ol>
<li><p>为一批问题生成推理轨迹</p></li>
<li><p>过滤：仅保留导致正确答案的轨迹</p></li>
<li><p>在成功轨迹上微调模型（SFT）</p></li>
<li><p>重复：改进后的模型在下一次迭代中生成更好的轨迹</p></li>
</ol>

<p>每次迭代都使用模型自身的成功输出作为训练数据，自举其推理能力。</p>
</div>

<div class="admonition note"><p class="admonition-title">Reflexion：语言强化学习</p>
<ol>
<li><p>Agent 尝试一项任务，失败</p></li>
<li><p>Agent 生成一段<strong>语言反思</strong>：“我失败是因为在调用 API 之前没有检查返回类型”</p></li>
<li><p>反思被存储在一个情节记忆缓冲区中</p></li>
<li><p>在下一次尝试中，反思作为经验教训被注入到 Prompt 中</p></li>
<li><p>不需要权重更新——纯粹通过自我批评进行上下文学习</p></li>
</ol>
</div>

## 12.3.2　B. 离策略探索 {#b.-off-policy-exploration-react-tool-use-frameworks}

<p>此范式以 ReAct <a class="book-citation" id="cite-yao2023react-1" href="../../references.md#ref-yao2023react" aria-label="参考文献 408">[408]</a> 及相关工具使用框架为代表，涉及广泛的自主探索。在自主探索过程中（网页导航、数据库查询、代码生成），Agent 记录数千条探索性执行路径。轨迹缓冲区充当一个过滤器：</p>

<ul>
<li><p><strong>成功过滤</strong>：只保留达成目标的轨迹用于训练。</p></li>
<li><p><strong>效率排序</strong>：在成功轨迹中，优先选择最短/最高效的工具使用路径。</p></li>
<li><p><strong>多样性采样</strong>：维持多样化的解决策略集合，以防止模式坍塌。</p></li>
</ul>

<p>优化算法（通常是 GRPO <a class="book-citation" id="cite-shao2024deepseekmath-2" href="../../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a> 或过滤后的 SFT）只在高效、成功的轨迹上计算 Loss，丢弃迂回曲折的运行。</p>

## 12.3.3　C. 非参数化的上下文学习（基于经验的 RAG） {#c.-non-parametric-in-context-learning-rag-over-experiences}

<p>轨迹缓冲区可以不修改神经网络权重，而是作为一个<strong>向量数据库</strong>。给定一个新的用户目标 <span class="arithmatex">\(\mathcal{G}_{\text{new}}\)</span>，系统检索最相关的过往经验：</p>



<div class="arithmatex">\[\boxed{\mathcal{E}_{\text{retrieved}} = \arg\max_{e \in \mathcal{B}} \text{sim}\!\left(\text{Embed}(\mathcal{G}_{\text{new}}),\; \text{Embed}(e)\right)}
\tag{12.4}\]</div>

<p>Top-<span class="arithmatex">\(k\)</span> 个相似的成功历史运行作为 few-shot 示例直接注入到 Prompt 上下文中。这种方法：</p>

<ul>
<li><p>需要<strong>零训练</strong>——纯粹的检索增强生成（Retrieval-Augmented Generation, RAG）</p></li>
<li><p>如果缓冲区中存在相似经验，可即时适应新任务</p></li>
<li><p>随缓冲区规模扩展（更多经验 = 更好的覆盖率）</p></li>
<li><p>与参数化学习互补（罕见情况使用检索，常见模式使用权重）</p></li>
</ul>
