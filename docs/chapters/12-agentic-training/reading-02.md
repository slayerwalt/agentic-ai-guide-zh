# 第 12 章 · 训练用例 {#reading-reading-02}

## 12.6　用例：用于生产力副驾的 Agent RL {#use-case-agentic-rl-for-a-productivity-co-pilot}

<p>本节提供了一个完整蓝图，展示如何应用 Agent RL 技术来训练一个跨生产力应用套件（文档、电子表格、演示文稿、电子邮件、消息、云存储）运行的基于 LLM 的副驾。</p>

### 12.6.1　架构概览 {#architecture-overview-dup2}

<figure id="figure-12-2"><a href="../../assets/figures/fig_043_fig43.png" target="_blank" rel="noopener" aria-label="查看图 12.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_043_fig43.png" alt="生产力副驾架构：LLM Agent（带 RL Policy \(\pi_\theta\)）接收用户意图并与多个应用 API 交互。基于任务成功、用户反馈和效率指标的 reward 信号驱动 Policy 改进。" loading="lazy"></a><figcaption><span class="figure-number">图 12.2　</span>生产力副驾架构：LLM Agent（带 RL Policy <span class="arithmatex">\(\pi_\theta\)</span>）接收用户意图并与多个应用 API 交互。基于任务成功、用户反馈和效率指标的 reward 信号驱动 Policy 改进。</figcaption></figure>

### 12.6.2　生产力副驾的形式化 MDP 定义 {#formal-mdp-definition-for-a-productivity-co-pilot}

<p>生产力副驾的环境被形式化为一个部分可观测马尔可夫决策过程（Partially Observable Markov Decision Process, POMDP）：</p>



<div class="arithmatex">\[\boxed{\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle}
\tag{12.14}\]</div>

<ul>
<li><p><span class="arithmatex">\(\mathcal{S}\)</span>：<strong>状态空间</strong>——完整的工作区环境状态：文档内容、邮件会话、日历事件、文件系统、用户权限。<em>不完全可观测</em>：Agent 只能看到 API 查询返回的内容。</p></li>
<li><p><span class="arithmatex">\(\mathcal{A}\)</span>：<strong>动作空间</strong>——结构化的 API 调用（见下文）。每个动作是一个 JSON 对象，指定目标应用、操作和参数。</p></li>
<li><p><span class="arithmatex">\(\mathcal{T}\)</span>：<strong>转移函数</strong>——大多数操作是确定性的（写入文档 <span class="arithmatex">\(\rightarrow\)</span> 文档更新），但依赖网络的动作（邮件投递时间、Teams 可用性）是随机的。</p></li>
<li><p><span class="arithmatex">\(\mathcal{R}\)</span>：<strong>Reward 函数</strong>——多组件（见 Reward 设计章节）。</p></li>
<li><p><span class="arithmatex">\(\Omega\)</span>：<strong>观测空间</strong>——API 响应、渲染后的文档视图、错误消息。</p></li>
<li><p><span class="arithmatex">\(\mathcal{O}\)</span>：<strong>观测函数</strong>——将状态映射到观测（API 响应格式化、为适应上下文窗口限制而截断）。</p></li>
<li><p><span class="arithmatex">\(\gamma = 0.99\)</span>：折扣因子（长视野，通常 10–50 步）。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">具体示例：“总结上周 Project Alpha 邮件并创建一张状态幻灯片”</p>
<p>下面我们追踪一个完整的 POMDP Episode，将每个形式化元素映射到具体实现。</p>

<p><strong>用户请求</strong>：“总结上周 Project Alpha 邮件中的关键决策，并在团队幻灯片中添加一张状态更新幻灯片。”</p>

<span id="table-12-inline-4"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>符号</strong></th><th scope="col"><strong>具体实现</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(s_0\)</span></td><td><em>真实状态</em>：收件箱中有 47 封来自过去 7 天提及 “Project Alpha” 的邮件；SharePoint 上的 PowerPoint 文件 “Q3_Status.pptx” 有 12 张幻灯片；用户拥有编辑权限。</td></tr>
<tr><td><span class="arithmatex">\(o_0\)</span></td><td><em>Agent 观测</em>：用户请求文本 + 系统 Prompt。Agent 此时<em>不知道</em>邮件数量或幻灯片内容。</td></tr>
<tr><td><span class="arithmatex">\(a_1\)</span></td><td><code>outlook.search({query: &quot;Project Alpha&quot;, last_7_days: true})</code></td></tr>
<tr><td><span class="arithmatex">\(\mathcal{T}(s_0, a_1)\)</span></td><td><span class="arithmatex">\(s_1\)</span>：系统检索出 47 封匹配邮件（确定性）。</td></tr>
<tr><td><span class="arithmatex">\(o_1\)</span></td><td>API 返回前 10 封邮件主题 + 发件人 + 时间戳（因上下文限制被截断——<span class="arithmatex">\(\mathcal{O}\)</span> 在起作用）。</td></tr>
<tr><td><span class="arithmatex">\(a_2\)</span></td><td><code>outlook.read({ids: [top_5_thread_ids]})</code>——Agent 挑选最相关的会话。</td></tr>
<tr><td><span class="arithmatex">\(o_2\)</span></td><td>5 个邮件会话的完整正文（<span class="arithmatex">\(\mathcal{O}\)</span> 截断后约 8K Token）。</td></tr>
<tr><td><span class="arithmatex">\(a_3\)</span></td><td><em>内部推理动作</em>：Agent 综合关键决策：(1) 截止日期推迟到 11 月 15 日，(2) 预算已批准，(3) 供应商已选定。</td></tr>
<tr><td><span class="arithmatex">\(a_4\)</span></td><td><code>sharepoint.read({file: &quot;Q3_Status.pptx&quot;, slides: &quot;last&quot;})</code>——检查当前最后一张幻灯片。</td></tr>
<tr><td><span class="arithmatex">\(o_4\)</span></td><td>最后一张幻灯片是 “Q2 Summary”（第 12 张）。Agent 决定添加第 13 张。</td></tr>
<tr><td><span class="arithmatex">\(a_5\)</span></td><td><code>powerpoint.add_slide({file: &quot;Q3_Status.pptx&quot;, position: 13, layout: &quot;Title and Content&quot;, title: &quot;Project Alpha — Week 42 Status&quot;, content: &quot;Key decisions: 1) Deadline: Nov 15...&quot;})</code></td></tr>
<tr><td><span class="arithmatex">\(\mathcal{T}(s_4, a_5)\)</span></td><td><span class="arithmatex">\(s_5\)</span>：幻灯片添加到演示文稿（确定性）。</td></tr>
<tr><td><span class="arithmatex">\(o_5\)</span></td><td>API 返回 <code>{success: true, slide_id: 13}</code>。</td></tr>
<tr><td><span class="arithmatex">\(R(s_5)\)</span></td><td>Reward 组件：<strong>+0.4</strong> 任务完成（幻灯片已创建），<strong>+0.3</strong> 信息质量（提取的决策正确），<strong>+0.2</strong> 格式合规（使用正确布局），<strong>+0.05</strong> 效率（5 个动作，无错误），<strong>-0.0</strong> 安全惩罚。<strong>总分：0.95</strong>。</td></tr></tbody>
</table>
</div>

<p><strong>所体现的 POMDP 关键方面</strong>：</p>

<ul>
<li><p><strong>部分可观测性</strong>：在 <span class="arithmatex">\(t=0\)</span> 时，Agent 不知道存在多少邮件或演示文稿包含什么——它必须查询以发现状态。</p></li>
<li><p><strong>观测函数 <span class="arithmatex">\(\mathcal{O}\)</span></strong>：由于上下文窗口限制，API 返回截断结果（47 封邮件中的前 10 封）。Agent 看到的是真实状态的一个<em>投影</em>。</p></li>
<li><p><strong>随机转移</strong>：如果 Agent 改用 <code>teams.send_message()</code>，投递时机将是不确定的（接收者在线/离线）。</p></li>
<li><p><strong>多步规划</strong>：Agent 必须跨 2 个应用串联 5 个动作，在邮件摘要和幻灯片内容之间保持一致性。</p></li>
<li><p><strong>折扣 <span class="arithmatex">\(\gamma=0.99\)</span></strong>：对 5 步而言，折扣影响很小（<span class="arithmatex">\(0.99^5 = 0.95\)</span>），但对 50 步任务来说至关重要——鼓励高效解决方案。</p></li>
</ul>
</div>

### 12.6.3　动作空间设计 {#action-space-design}

<p>动作空间必须是<strong>结构化的、类型安全的且可组合的</strong>：</p>

<div class="admonition example"><p class="admonition-title">生产力副驾动作 Schema</p>
<div class="highlight"><pre><span></span><code><span class="p">{</span>
  <span class="s2">"action_type"</span><span class="p">:</span> <span class="s2">"api_call"</span><span class="p">,</span>
  <span class="s2">"target_app"</span><span class="p">:</span> <span class="s2">"outlook | excel | word | powerpoint | teams | sharepoint"</span><span class="p">,</span>
  <span class="s2">"operation"</span><span class="p">:</span> <span class="s2">"read | write | search | create | delete | modify"</span><span class="p">,</span>
  <span class="s2">"parameters"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"endpoint"</span><span class="p">:</span> <span class="s2">"/me/messages?$filter=subject eq 'Project X'"</span><span class="p">,</span>
    <span class="s2">"body"</span><span class="p">:</span> <span class="p">{</span> <span class="o">...</span> <span class="p">},</span>             <span class="o">//</span> <span class="n">用于写操作</span>
    <span class="s2">"options"</span><span class="p">:</span> <span class="p">{</span> <span class="s2">"top"</span><span class="p">:</span> <span class="mi">10</span> <span class="p">}</span>     <span class="o">//</span> <span class="n">分页</span><span class="err">、</span><span class="n">过滤</span>
  <span class="p">},</span>
  <span class="s2">"reasoning"</span><span class="p">:</span> <span class="s2">"I need to find relevant emails before summarizing"</span>
<span class="p">}</span>
</code></pre></div>
</div>

<p><strong>按应用划分的动作分类</strong>：</p>

<span id="table-12-inline-5"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>应用</strong></th><th scope="col"><strong>复杂度</strong></th><th scope="col"><strong>关键动作</strong></th></tr></thead>
<tbody><tr><td><strong>Outlook</strong></td><td>中</td><td><code>search</code>、<code>read</code>、<code>draft</code>、<code>send</code>、<code>move</code>、<code>flag</code>、<code>create_rule</code></td></tr>
<tr><td><strong>Excel</strong></td><td>高</td><td><code>read_range</code>、<code>write_range</code>、<code>insert_formula</code>、<code>create_chart</code>、<code>pivot_table</code>、<code>run_macro</code></td></tr>
<tr><td><strong>Word</strong></td><td>中</td><td><code>read_paragraphs</code>、<code>insert_text</code>、<code>format_section</code>、<code>find_replace</code>、<code>insert_table</code></td></tr>
<tr><td><strong>PowerPoint</strong></td><td>中</td><td><code>add_slide</code>、<code>insert_shape</code>、<code>set_text</code>、<code>set_layout</code>、<code>add_image</code>、<code>apply_theme</code></td></tr>
<tr><td><strong>Teams</strong></td><td>低</td><td><code>send_message</code>、<code>create_meeting</code>、<code>search_chat</code>、<code>add_members</code>、<code>post_to_channel</code></td></tr>
<tr><td><strong>SharePoint</strong></td><td>中</td><td><code>list_files</code>、<code>upload</code>、<code>download</code>、<code>search</code>、<code>create_page</code>、<code>set_permissions</code></td></tr></tbody>
</table>
</div>

### 12.6.4　状态表示 {#state-representation}

<p>Agent 在每一步的观测（上下文窗口）：</p>



<div class="arithmatex">\[o_t = [\text{system\_prompt};\; \text{user\_intent};\; \text{tool\_history}_{1:t-1};\; \text{current\_result}_t]
\tag{12.15}\]</div>

<p><strong>上下文预算管理</strong>（对 128K 窗口至关重要）：</p>

<ul>
<li><p><strong>系统 Prompt</strong>：2K Token（能力、安全规则、输出格式）</p></li>
<li><p><strong>用户意图 + 对话</strong>：4K Token</p></li>
<li><p><strong>工具历史</strong>（滑动窗口）：最近 8–12 个动作 + 观测，更早的进行摘要。总计：最多 80K Token。</p></li>
<li><p><strong>当前观测</strong>：最多 32K Token（大型电子表格、邮件会话）</p></li>
<li><p><strong>保留</strong>：10K Token 用于 Agent 的推理 + 下一动作生成</p></li>
</ul>

<p><strong>状态压缩策略</strong>：</p>

<ul>
<li><p><strong>选择性纳入</strong>：只纳入与当前子目标相关的 API 响应（使用辅助的“相关性打分器”）。</p></li>
<li><p><strong>结构化摘要</strong>：将大型电子表格表示为 Schema + 样本行，而非完整数据。</p></li>
<li><p><strong>分层记忆</strong>：将完整轨迹存储在外部；将压缩摘要注入上下文。</p></li>
</ul>

### 12.6.5　Reward 设计：多目标信号 {#reward-design-multi-objective-signal}

<p>生产力副驾的 Reward 函数必须平衡多个目标：</p>



<div class="arithmatex">\[\boxed{R(\tau) = \alpha_1 R_{\text{task}} + \alpha_2 R_{\text{quality}} + \alpha_3 R_{\text{efficiency}} + \alpha_4 R_{\text{safety}} + \alpha_5 R_{\text{user}}}
\tag{12.16}\]</div>

<p class="table-caption" id="table-12-3"><span class="table-number">表 12.3　</span>生产力副驾训练的 Reward 组件。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>组件</strong></th><th scope="col"><strong>权重</strong></th><th scope="col"><strong>信号类型</strong></th><th scope="col"><strong>定义</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(R_{\text{task}}\)</span></td><td>0.40</td><td>二元/标量</td><td>任务成功完成（邮件已发送、文档已创建、公式正确）</td></tr>
<tr><td><span class="arithmatex">\(R_{\text{quality}}\)</span></td><td>0.25</td><td>LLM judge</td><td>输出质量：格式化、清晰度、内容正确性</td></tr>
<tr><td><span class="arithmatex">\(R_{\text{efficiency}}\)</span></td><td>0.15</td><td>标量</td><td>对过多步骤的惩罚：<span class="arithmatex">\(-0.02 \times (\text{num\_steps} - \text{optimal\_steps})\)</span></td></tr>
<tr><td><span class="arithmatex">\(R_{\text{safety}}\)</span></td><td>0.15</td><td>二元</td><td>无不安全动作（未确认即删除、发到错误收件人、权限违规）。任何违规则 <span class="arithmatex">\(R_{\text{safety}} = 0\)</span>。</td></tr>
<tr><td><span class="arithmatex">\(R_{\text{user}}\)</span></td><td>0.05</td><td>稀疏</td><td>可用时的显式用户反馈（拇指向上/向下）</td></tr></tbody>
</table>
</div>

<p><strong>中间 reward（密集信号）</strong>：</p>

<ul>
<li><p>成功的 API 调用（200 响应）：+0.05</p></li>
<li><p>正确的信息检索（通过下游使用验证）：+0.10</p></li>
<li><p>优雅地从错误中恢复（用修正后的参数重试）：+0.08</p></li>
<li><p>API 错误（4xx/5xx）：–0.03</p></li>
<li><p>重复的相同动作（循环检测）：–0.10</p></li>
<li><p>当意图确实模糊时提出澄清问题：+0.05</p></li>
</ul>

### 12.6.6　训练流水线：端到端 {#training-pipeline-end-to-end}

<div class="admonition note"><p class="admonition-title">生产力副驾 RL 训练流水线</p>
<p><strong>阶段 1：监督微调（基础）</strong></p>

<ol>
<li><p>收集 50K–200K 条人工示范的生产力任务轨迹（通过遥测、标注员或合成生成）。</p></li>
<li><p>在 ReAct 格式的（指令，轨迹）对上对基础 LLM 进行 SFT。</p></li>
<li><p>验证：Agent 应在保留任务上达到 40–60% 的任务完成率。</p></li>
</ol>

<p><strong>阶段 2：模拟环境构建</strong></p>

<ol>
<li><p>构建一个带有模拟 API 端点、合成邮箱、文档和日历的<strong>沙盒环境</strong>。</p></li>
<li><p>每个“用户”拥有真实的画像：500+ 封邮件、20+ 份文档、日历事件、Teams 频道。</p></li>
<li><p>任务生成器：产生多样化的指令–验证对：“把 Alice 关于 Q4 预算的所有邮件移到 ‘Finance’ 文件夹” + 验证函数。</p></li>
</ol>

<p><strong>阶段 3：在线 RL 训练（GRPO）</strong></p>

<ol>
<li><p>采样任务 Batch（每次迭代 256 个任务）。</p></li>
<li><p>在沙盒环境中使用 <span class="arithmatex">\(\pi_\theta\)</span> 为每个任务生成 <span class="arithmatex">\(N=8\)</span> 条轨迹。</p></li>
<li><p>执行轨迹，从验证函数收集 reward。</p></li>
<li><p>计算 GRPO 优势（跨每个任务的 8 条轨迹做组归一化）。</p></li>
<li><p>用裁剪目标 + 相对于 SFT 模型的 KL 惩罚来更新 Policy。</p></li>
<li><p>每 500 次迭代：在保留基准上评估（200 个任务，5 个难度级别）。</p></li>
</ol>

<p><strong>阶段 4：人机协同（Human-in-the-Loop）细化</strong></p>

<ol>
<li><p>部署给内部 dogfood 用户（1000+ 用户，2 周）。</p></li>
<li><p>收集拇指向上/向下信号 + 自由文本纠正。</p></li>
<li><p>从 A/B 部署中（旧 Policy 对比新 Policy）构造 DPO 偏好对。</p></li>
<li><p>在人类偏好上应用 1–2 轮 DPO 微调。</p></li>
</ol>
</div>

### 12.6.7　模拟环境架构 {#simulation-environment-architecture}

<div class="admonition example"><p class="admonition-title">沙盒环境（简化版）</p>
<div class="highlight"><pre><span></span><code><span class="k">class</span><span class="w"> </span><span class="nc">ProductivityEnvironment</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">user_profile</span><span class="p">:</span> <span class="n">UserProfile</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">mailbox</span> <span class="o">=</span> <span class="n">SyntheticMailbox</span><span class="p">(</span><span class="n">user_profile</span><span class="o">.</span><span class="n">emails</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">drive</span> <span class="o">=</span> <span class="n">SyntheticOneDrive</span><span class="p">(</span><span class="n">user_profile</span><span class="o">.</span><span class="n">files</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">calendar</span> <span class="o">=</span> <span class="n">SyntheticCalendar</span><span class="p">(</span><span class="n">user_profile</span><span class="o">.</span><span class="n">events</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">teams</span> <span class="o">=</span> <span class="n">SyntheticTeams</span><span class="p">(</span><span class="n">user_profile</span><span class="o">.</span><span class="n">channels</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">step_count</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">max_steps</span> <span class="o">=</span> <span class="mi">50</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">step</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">action</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Tuple</span><span class="p">[</span><span class="n">Observation</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">bool</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""执行动作，返回 (observation, reward, done)。"""</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">step_count</span> <span class="o">+=</span> <span class="mi">1</span>

        <span class="c1"># 路由到对应的应用处理器</span>
        <span class="n">handler</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">get_handler</span><span class="p">(</span><span class="n">action</span><span class="p">[</span><span class="s2">"target_app"</span><span class="p">])</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">result</span> <span class="o">=</span> <span class="n">handler</span><span class="o">.</span><span class="n">execute</span><span class="p">(</span><span class="n">action</span><span class="p">[</span><span class="s2">"operation"</span><span class="p">],</span> <span class="n">action</span><span class="p">[</span><span class="s2">"parameters"</span><span class="p">])</span>
            <span class="n">obs</span> <span class="o">=</span> <span class="n">Observation</span><span class="p">(</span><span class="n">status</span><span class="o">=</span><span class="mi">200</span><span class="p">,</span> <span class="n">body</span><span class="o">=</span><span class="n">result</span><span class="p">)</span>
            <span class="n">reward</span> <span class="o">=</span> <span class="mf">0.05</span>  <span class="c1"># 成功的 API 调用</span>
        <span class="k">except</span> <span class="n">APIError</span> <span class="k">as</span> <span class="n">e</span><span class="p">:</span>
            <span class="n">obs</span> <span class="o">=</span> <span class="n">Observation</span><span class="p">(</span><span class="n">status</span><span class="o">=</span><span class="n">e</span><span class="o">.</span><span class="n">code</span><span class="p">,</span> <span class="n">body</span><span class="o">=</span><span class="nb">str</span><span class="p">(</span><span class="n">e</span><span class="p">))</span>
            <span class="n">reward</span> <span class="o">=</span> <span class="o">-</span><span class="mf">0.03</span>

        <span class="c1"># 检查终止条件</span>
        <span class="n">done</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">step_count</span> <span class="o">&gt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">max_steps</span>
        <span class="k">return</span> <span class="n">obs</span><span class="p">,</span> <span class="n">reward</span><span class="p">,</span> <span class="n">done</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">evaluate</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task</span><span class="p">:</span> <span class="n">Task</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""检查任务目标是否达成（终端 reward）。"""</span>
        <span class="k">return</span> <span class="n">task</span><span class="o">.</span><span class="n">verification_fn</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span>  <span class="c1"># 0.0 或 1.0</span>
</code></pre></div>
</div>

### 12.6.8　任务课程设计 {#task-curriculum-design}

<p>训练有效性关键取决于任务难度的递进：</p>

<p class="table-caption" id="table-12-4"><span class="table-number">表 12.4　</span>生产力副驾课程级别。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>级别</strong></th><th scope="col"><strong>步数</strong></th><th scope="col"><strong>应用数</strong></th><th scope="col"><strong>示例任务</strong></th></tr></thead>
<tbody><tr><td><strong>L1: 单步</strong></td><td>1–2</td><td>1</td><td>“读取我最新的来自 Bob 的邮件”，“A1 单元格里是什么？”</td></tr>
<tr><td><strong>L2: 单应用</strong></td><td>3–5</td><td>1</td><td>“起草一封回复预算邮件并总结要点”</td></tr>
<tr><td><strong>L3: 多步</strong></td><td>5–10</td><td>1</td><td>“从销售数据创建数据透视表，并将 top 表现者加粗格式化”</td></tr>
<tr><td><strong>L4: 跨应用</strong></td><td>5–15</td><td>2–3</td><td>“查找 Q4 预算邮件，提取数字，放入新的 Excel 表”</td></tr>
<tr><td><strong>L5: 复杂工作流</strong></td><td>10–30</td><td>3+</td><td>“准备周报：从 Excel 抽取指标，总结邮件更新，创建 PowerPoint 幻灯片，在 Teams 中分享”</td></tr></tbody>
</table>
</div>

<p><strong>课程策略</strong>：</p>

<ul>
<li><p>训练早期从 80% L1–L2 任务、20% L3 开始。</p></li>
<li><p>当当前级别成功率超过 70% 时晋升下一级。</p></li>
<li><p>始终保留 10–20% 的较简单任务以防止灾难性遗忘（Catastrophic Forgetting）。</p></li>
<li><p>收敛后的最终配比：10% L1、15% L2、25% L3、30% L4、20% L5。</p></li>
</ul>

### 12.6.9　安全与护栏 {#safety-and-guardrails}

<div class="admonition note"><p class="admonition-title">生产力副驾安全框架</p>
<p><strong>硬约束</strong>（动作立即被拒绝，reward = –1.0）：</p>

<ul>
<li><p>未经用户确认即向外部收件人发送邮件/消息</p></li>
<li><p>永久删除文件/邮件（仅允许软删除）</p></li>
<li><p>修改共享资源的权限</p></li>
<li><p>越权访问其他用户的邮箱或文件</p></li>
<li><p>在一个 Batch 中对超过 100 个条目执行动作（防止批量删除/移动事故）</p></li>
</ul>

<p><strong>软约束</strong>（reward 中的惩罚，Agent 应学习避免）：</p>

<ul>
<li><p>未向用户展示预览即发送草稿：–0.2</p></li>
<li><p>未先声明意图即做出不可逆变更：–0.15</p></li>
<li><p>访问敏感标签（机密、律师-当事人）：–0.3</p></li>
<li><p>在没有显式委托的情况下使用“代发”：–0.25</p></li>
</ul>

<p><strong>确认协议</strong>：对任何被归类为“高影响”（发送、删除、对外分享）的动作，Agent 必须：</p>

<ol>
<li><p>用自然语言陈述所计划的动作</p></li>
<li><p>显示将要发送/修改内容的预览</p></li>
<li><p>在执行前等待显式的用户确认</p></li>
</ol>

<p>这在环境层（沙盒拒绝未确认的高影响动作）和 Reward 函数（跳过确认会受罚）两处都被强制执行。</p>
</div>

### 12.6.10　多应用工作流中的信用分配 {#credit-assignment-in-multi-app-workflows}

<p>关键挑战：在一个 20 步的跨应用工作流中，哪些步骤对成功或失败做出了贡献？</p>

<p><strong>方法：分层 Reward 分解</strong></p>

<ol>
<li><p><strong>子目标检测</strong>：将用户指令分解为可验证的子目标：</p>

<ul>
<li><p>“查找 Q4 预算邮件” <span class="arithmatex">\(\rightarrow\)</span> 子目标 1（验证：检索到相关邮件）</p></li>
<li><p>“提取数字” <span class="arithmatex">\(\rightarrow\)</span> 子目标 2（验证：正确数值已解析）</p></li>
<li><p>“创建 Excel 表” <span class="arithmatex">\(\rightarrow\)</span> 子目标 3（验证：表存在且包含正确数据）</p></li>
</ul></li>
<li><p><strong>子目标 reward</strong>：每完成一个子目标就分配中间 reward（每个 <span class="arithmatex">\(r = +0.2\)</span>）。</p></li>
<li><p><strong>轨迹切片</strong>：如果最终任务失败，识别哪个子目标最先失败。仅对该子目标跨度内的动作施加负 reward。</p></li>
<li><p><strong>反事实估计</strong>：“如果这个特定动作不同，任务是否会成功？”——使用 Value 函数来估计。</p></li>
</ol>



<div class="arithmatex">\[R_{\text{step}}(t) = \underbrace{R_{\text{sub-goal}}(t)}_{\text{当前子目标是否成功？}} + \underbrace{\gamma^{T-t} R_{\text{terminal}}}_{\text{折扣后的最终 reward}} + \underbrace{r_{\text{intermediate}}(t)}_{\text{每步 API 成功/失败}}
\tag{12.17}\]</div>

### 12.6.11　扩展与基础设施 {#scaling-and-infrastructure}

<p><strong>算力需求</strong>（针对 70B 参数模型估算）：</p>

<span id="table-12-inline-8"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>组件</strong></th><th scope="col"><strong>资源</strong></th><th scope="col"><strong>说明</strong></th></tr></thead>
<tbody><tr><td>Policy 模型（70B）</td><td>8<span class="arithmatex">\(\times\)</span> A100 80GB（TP=8）</td><td>BF16，生成轨迹</td></tr>
<tr><td>参考模型（70B）</td><td>8<span class="arithmatex">\(\times\)</span> A100 80GB（TP=8）</td><td>冻结，用于 KL 计算</td></tr>
<tr><td>环境 Worker</td><td>128 个 CPU Worker</td><td>每个运行一个沙盒实例</td></tr>
<tr><td>Reward Model / Judge</td><td>4<span class="arithmatex">\(\times\)</span> A100（若用 LLM judge）</td><td>或者若使用基于执行的 reward 则为零</td></tr>
<tr><td>训练（GRPO 更新）</td><td>16<span class="arithmatex">\(\times\)</span> A100（FSDP）</td><td>跨轨迹 Batch 做梯度累积</td></tr>
<tr><td><strong>总计</strong></td><td><strong>40 张 A100 GPU + 128 CPU</strong></td><td>完整训练运行约 5,000 GPU 小时</td></tr></tbody>
</table>
</div>

<p><strong>吞吐量优化</strong>：</p>

<ul>
<li><p><strong>异步 rollout</strong>：将轨迹生成与梯度更新解耦。在训练上一个 Batch 时持续生成。</p></li>
<li><p><strong>批量化环境</strong>：并行运行 128 个沙盒环境，每个处理不同任务。</p></li>
<li><p><strong>KV-cache 共享</strong>：对每个任务的 <span class="arithmatex">\(N=8\)</span> 条轨迹，它们共享相同的 Prompt 前缀——使用前缀缓存避免冗余计算。</p></li>
<li><p><strong>选择性反向传播</strong>：只在动作 Token 上计算梯度（而非观测/系统 Prompt）。将反向传播 FLOPS 减少 40–60%。</p></li>
</ul>

### 12.6.12　评估框架 {#evaluation-framework}

<p class="table-caption" id="table-12-5"><span class="table-number">表 12.5　</span>生产力副驾评估维度。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>指标</strong></th><th scope="col"><strong>目标</strong></th><th scope="col"><strong>度量方式</strong></th></tr></thead>
<tbody><tr><td>任务完成率</td><td><span class="arithmatex">\(&gt;85\%\)</span>（L1–L3），<span class="arithmatex">\(&gt;60\%\)</span>（L4–L5）</td><td>沙盒中的自动化验证</td></tr>
<tr><td>安全违规率</td><td><span class="arithmatex">\(&lt;0.1\%\)</span></td><td>每 1000 个任务的硬约束违规计数</td></tr>
<tr><td>完成的平均步数</td><td>不超过最优值的 <span class="arithmatex">\(1.5\times\)</span></td><td>与已知最短成功轨迹对比</td></tr>
<tr><td>用户满意度（dogfood）</td><td><span class="arithmatex">\(&gt;4.2/5.0\)</span></td><td>内部用户任务后调查</td></tr>
<tr><td>跨应用成功率</td><td><span class="arithmatex">\(&gt;55\%\)</span>（L4–L5）</td><td>需要 2+ 应用的任务</td></tr>
<tr><td>恢复率</td><td><span class="arithmatex">\(&gt;70\%\)</span></td><td>Agent 成功重试失败 API 调用的百分比</td></tr>
<tr><td>延迟（首动作时间）</td><td><span class="arithmatex">\(&lt;3\)</span> 秒</td><td>模型推理 + 动作规划时间</td></tr></tbody>
</table>
</div>

<p><strong>基准套件</strong>（建议）：</p>

<ul>
<li><p><strong>ProdBench-Easy</strong>（200 个任务）：单应用，1–3 步。基线建立。</p></li>
<li><p><strong>ProdBench-Hard</strong>（200 个任务）：跨应用工作流，10–30 步。端到端能力。</p></li>
<li><p><strong>ProdBench-Safety</strong>（100 个任务）：试图触发不安全动作的对抗 Prompt。必须保持 <span class="arithmatex">\(&lt;0.1\%\)</span> 违规率。</p></li>
<li><p><strong>ProdBench-Robustness</strong>（100 个任务）：带模糊指令、注入 API 错误、缺失权限的任务。测试优雅降级。</p></li>
</ul>

### 12.6.13　来自生产部署的经验教训 {#lessons-from-production-deployments}

<div class="admonition tip"><p class="admonition-title">生产力 Agent RL 的实践洞见</p>
<ol>
<li><p><strong>SFT 质量是底线</strong>：RL 只能在 SFT 所提供的基础上改进。如果 SFT 模型无法格式化一个有效的 Graph API 调用，RL 也发现不了。要在阶段 1 的数据质量上重投入。</p></li>
<li><p><strong>Reward hacking 不可避免</strong>：Agent <em>一定</em>会找到捷径。常见例子：</p>

<ul>
<li><p>创建一个空 Excel 文件以“完成”电子表格任务（通过存在性检查）</p></li>
<li><p>在没有实际执行动作的情况下回复“完成”</p></li>
<li><p>利用模糊的验证函数</p></li>
</ul>

<p><strong>缓解措施</strong>：多层验证（格式 + 内容 + 语义正确性）。</p></li>
<li><p><strong>API 速率限制很重要</strong>：在生产中，工作区 API 有限流（429 响应）。用现实的速率限制训练，避免学到滥发并行请求的 Policy。</p></li>
<li><p><strong>上下文窗口是瓶颈</strong>：带丰富 API 响应的 20 步轨迹轻松消耗 80K+ Token。技术手段：观测摘要、选择性历史、分层上下文管理。</p></li>
<li><p><strong>用户意图常常是模糊的</strong>：“清理我的收件箱”对不同用户意味着不同的事情。训练 Agent 在不确定性高时提出澄清问题（恰当澄清给奖励，过度澄清给惩罚）。</p></li>
<li><p><strong>从简单开始，逐步扩展</strong>：从仅 Outlook 任务开始（数量最多，遥测数据最丰富），然后扩展到 Excel，再到跨应用。每个应用都有独特的失败模式。</p></li>
</ol>
</div>

### 12.6.14　完整训练食谱 {#complete-training-recipe}

<p class="table-caption" id="table-12-6"><span class="table-number">表 12.6　</span>生产力副驾 RL 训练的推荐超参数。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>参数</strong></th><th scope="col"><strong>取值</strong></th><th scope="col"><strong>理由</strong></th></tr></thead>
<tbody><tr><td>基础模型</td><td>70B Llama/Mistral</td><td>足够容量进行复杂多步推理</td></tr>
<tr><td>RL 算法</td><td>GRPO</td><td>无需 critic；对长轨迹更省内存</td></tr>
<tr><td>组大小 <span class="arithmatex">\(N\)</span></td><td>8</td><td>在方差减小与算力代价之间取平衡</td></tr>
<tr><td>裁剪 <span class="arithmatex">\(\epsilon\)</span></td><td>0.1</td><td>比标准（0.2）更紧，因长轨迹敏感性</td></tr>
<tr><td>KL 系数 <span class="arithmatex">\(\beta\)</span></td><td>0.04</td><td>对 SFT Policy 的中等约束</td></tr>
<tr><td>Learning rate</td><td><span class="arithmatex">\(5 \times 10^{-7}\)</span></td><td>保守；Agent 任务对大幅更新敏感</td></tr>
<tr><td>Batch size</td><td>256 任务 <span class="arithmatex">\(\times\)</span> 8 轨迹 = 2048</td><td>大 Batch 以稳定 GRPO 归一化</td></tr>
<tr><td>最大轨迹长度</td><td>50 步</td><td>覆盖 95% 的生产力任务</td></tr>
<tr><td>上下文窗口</td><td>128K Token</td><td>长多应用工作流所需</td></tr>
<tr><td>训练迭代数</td><td>3000–5000</td><td>监控评估指标；安全性退化时早停</td></tr>
<tr><td>课程预热</td><td>500 次迭代（仅 L1–L2）</td><td>在复杂任务前建立基本 API 使用</td></tr></tbody>
</table>
</div>

## 12.7　用例：从零构建一个研究 Agent {#use-case-building-a-research-agent-from-scratch}

<p>本用例展示了如何使用本文讨论的技术，构建一个完全自主的<strong>研究 Agent</strong>——一个能够提出假设、检索文献、分析数据、编写代码、运行实验并产出最终报告的 LLM。</p>

### 12.7.1　问题定义 {#problem-definition}

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

### 12.7.2　MDP 形式化 {#mdp-formulation}

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

### 12.7.3　动作空间 {#action-space}

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

### 12.7.4　架构：模型与基础设施选择 {#architecture-model-and-infrastructure-choices}

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

### 12.7.5　Reward 设计 {#reward-design}

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

### 12.7.6　训练流水线 {#training-pipeline}

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

### 12.7.7　示例轨迹：完整 MDP 追踪 {#example-trajectory-full-mdp-trace}

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

### 12.7.8　关键设计决策与权衡 {#key-design-decisions-and-tradeoffs}

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

### 12.7.9　评估 {#evaluation-dup2}

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

### 12.7.10　经验教训与失败模式 {#lessons-and-failure-modes}

<div class="admonition warning"><p class="admonition-title">研究 Agent 训练中的常见失败</p>
<ul>
<li><p><strong>无限循环</strong>：Agent 反复搜索论文却不前进。<em>修复</em>：步数预算 + 对参数相同的重复工具调用的惩罚。</p></li>
<li><p><strong>代码调试螺旋</strong>：Agent 花 20+ 步修一个 Bug。<em>修复</em>：重试次数上限为 3；若代码失败 3 次，放弃当前方法尝试替代方案。</p></li>
<li><p><strong>幻觉引用</strong>：Agent 编造论文标题/结果。<em>修复</em>：通过工具输出验证所有引用存在；惩罚不可验证的断言。</p></li>
<li><p><strong>过早提交</strong>：Agent 提交不完整报告以避免长轨迹的惩罚。<em>修复</em>：设定最低质量阈值（<span class="arithmatex">\(R &gt; 0.4\)</span>）作为有效提交；低于阈值视为失败。</p></li>
<li><p><strong>对 judge 的 reward hacking</strong>：Agent 学会产生在 LLM judge 处得高分但科学上肤浅的文本。<em>修复</em>：轮换 judge 模型；定期在 reward 中纳入人类评估。</p></li>
</ul>
</div>
