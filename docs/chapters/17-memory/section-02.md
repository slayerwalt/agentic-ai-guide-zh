# 17.2　记忆类型分类 {#sec:memory-taxonomy}

<figure id="fig:memory-taxonomy"><a href="../../assets/figures/fig_052_memory-taxonomy.png" target="_blank" rel="noopener" aria-label="查看图 17.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_052_memory-taxonomy.png" alt="智能体记忆系统的四分类法，与认知科学的划分相对应。每种记忆类型具有截然不同的访问模式、更新频率和检索机制。" loading="lazy"></a><figcaption><span class="figure-number">图 17.1　</span>智能体记忆系统的四分类法，与认知科学的划分相对应。每种记忆类型具有截然不同的访问模式、更新频率和检索机制。</figcaption></figure>

## 17.2.1　工作记忆（短期） {#working-memory-short-term}

<p>工作记忆是 Agent 的<em>活动工作空间</em>：当前正在被操控的信息。在 LLM Agent 中它对应于：</p>

<ul>
<li><p><strong>草稿板（Scratchpads）。</strong> 在产出最终答案之前写入专用缓冲区的中间推理步骤（例如思维链 Chain-of-Thought <a class="book-citation" id="cite-wei2022chain-1" href="../../references.md#ref-wei2022chain" aria-label="参考文献 374">[374]</a>、scratchpad <a class="book-citation" id="cite-nye2021show-1" href="../../references.md#ref-nye2021show" aria-label="参考文献 272">[272]</a>）。</p></li>
<li><p><strong>思维链缓冲区。</strong> 在答案 Token <span class="arithmatex">\(a\)</span> 之前生成的推理 Token 序列 <span class="arithmatex">\(z_1, z_2, \ldots, z_k\)</span>，建模为 <span class="arithmatex">\(p(a \mid x) = \sum_z p(a \mid x, z)\,p(z \mid x)\)</span>。</p></li>
<li><p><strong>对话上下文。</strong> 保留在上下文窗口中的近期轮次历史 <span class="arithmatex">\([(u_1, a_1), \ldots, (u_t, a_t)]\)</span>。</p></li>
</ul>

<p>工作记忆是<em>快速的</em>（零检索延迟——它已经在上下文中）、<em>易失的</em>（上下文清空时即丢失），且<em>容量受限</em>（受 <span class="arithmatex">\(L\)</span> 约束）。</p>

## 17.2.2　情景记忆（基于经验） {#episodic-memory-experience-based}

<p>情景记忆存储<em>具体的过往事件</em>，并按上下文与时间索引。对 Agent 而言：</p>

<ul>
<li><p><strong>过往交互。</strong> 先前对话、任务尝试及其结果的完整或摘要记录。</p></li>
<li><p><strong>成功轨迹。</strong> 可作为少样本范例被检索、以服务于未来相似任务的高 Reward 动作序列。</p></li>
<li><p><strong>失败案例。</strong> 带有根因标注的错误记录，使 Agent 能够避免重复犯错。</p></li>
<li><p><strong>检索增强的情景回忆。</strong> 给定新任务 <span class="arithmatex">\(q\)</span>，检索最相似的 <span class="arithmatex">\(k\)</span> 个过去 Episode <span class="arithmatex">\(\{e_i\}_{i=1}^k\)</span> 并加入上下文。</p></li>
</ul>

<p>情景记忆通常实现为基于 Episode 摘要 Embedding 的向量存储（见 <a href="section-03.md#sec:rag-memory">17.3.1</a> 节）。</p>

## 17.2.3　语义记忆（世界知识） {#semantic-memory-world-knowledge}

<p>语义记忆编码与具体 Episode 解耦的<em>一般性事实与概念</em>：</p>

<ul>
<li><p><strong>事实性知识。</strong> 实体、属性与关系（例如 “Paris 是 France 的首都”）。</p></li>
<li><p><strong>领域概念。</strong> 与 Agent 任务领域相关的定义、分类法与本体。</p></li>
<li><p><strong>知识图谱。</strong> 结构化表示 <span class="arithmatex">\(\mathcal{G} = (\mathcal{V}, \mathcal{E})\)</span>，节点 <span class="arithmatex">\(v \in \mathcal{V}\)</span> 为实体，边 <span class="arithmatex">\(e \in \mathcal{E}\)</span> 为带类型的关系。</p></li>
</ul>

<p>与情景记忆不同，语义记忆是<em>上下文无关的</em>：水在 <span class="arithmatex">\(100^\circ\)</span>C 沸腾这一事实，无论何时何地学到都成立。</p>

## 17.2.4　程序记忆（技能） {#procedural-memory-skills}

<p>程序记忆编码<em>如何做事</em>——已被自动化的技能与动作模式：</p>

<ul>
<li><p><strong>习得的工具使用模式。</strong> 对哪种任务调用哪个 API、如何格式化输入、如何处理错误。</p></li>
<li><p><strong>动作序列。</strong> 多步过程（例如 “部署代码：跑测试 <span class="arithmatex">\(\to\)</span> 构建镜像 <span class="arithmatex">\(\to\)</span> 推送 <span class="arithmatex">\(\to\)</span> 更新 manifest”）。</p></li>
<li><p><strong>作为记忆的 Policy。</strong> 模型权重 <span class="arithmatex">\(\theta\)</span> 本身即编码了程序性知识；在成功轨迹上微调是程序记忆巩固的一种形式。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">记忆类型的分类示例</p>
<p>一个协助软件开发的 Agent 会用到：</p>

<ul>
<li><p><strong>工作记忆</strong>：当前正在编辑的文件、刚收到的错误信息。</p></li>
<li><p><strong>情景记忆</strong>：“上周我在模块 X 第 42 行加了一个空值检查，修复了一个类似的 <code>NullPointerException</code>。”</p></li>
<li><p><strong>语义记忆</strong>：“Python 的 <code>asyncio.gather</code> 并发运行协程；除非设置 <code>return_exceptions=True</code>，否则异常会向上传播。”</p></li>
<li><p><strong>程序记忆</strong>：标准调试工作流——复现 <span class="arithmatex">\(\to\)</span> 隔离 <span class="arithmatex">\(\to\)</span> 提假设 <span class="arithmatex">\(\to\)</span> 验证 <span class="arithmatex">\(\to\)</span> 修复。</p></li>
</ul>
</div>
