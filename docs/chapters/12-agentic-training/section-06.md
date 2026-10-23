# 12.6　用例：用于生产力副驾的 Agent RL {#use-case-agentic-rl-for-a-productivity-co-pilot}

<p>本节提供了一个完整蓝图，展示如何应用 Agent RL 技术来训练一个跨生产力应用套件（文档、电子表格、演示文稿、电子邮件、消息、云存储）运行的基于 LLM 的副驾。</p>

## 12.6.1　架构概览 {#architecture-overview-dup2}

<figure id="figure-12-2"><a href="../../assets/figures/fig_043_fig43.png" target="_blank" rel="noopener" aria-label="查看图 12.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_043_fig43.png" alt="生产力副驾架构：LLM Agent（带 RL Policy \(\pi_\theta\)）接收用户意图并与多个应用 API 交互。基于任务成功、用户反馈和效率指标的 reward 信号驱动 Policy 改进。" loading="lazy"></a><figcaption><span class="figure-number">图 12.2　</span>生产力副驾架构：LLM Agent（带 RL Policy <span class="arithmatex">\(\pi_\theta\)</span>）接收用户意图并与多个应用 API 交互。基于任务成功、用户反馈和效率指标的 reward 信号驱动 Policy 改进。</figcaption></figure>

## 12.6.2　生产力副驾的形式化 MDP 定义 {#formal-mdp-definition-for-a-productivity-co-pilot}

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

## 12.6.3　动作空间设计 {#action-space-design}

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

## 12.6.4　状态表示 {#state-representation}

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

## 12.6.5　Reward 设计：多目标信号 {#reward-design-multi-objective-signal}

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

## 12.6.6　训练流水线：端到端 {#training-pipeline-end-to-end}

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

## 12.6.7　模拟环境架构 {#simulation-environment-architecture}

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

## 12.6.8　任务课程设计 {#task-curriculum-design}

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

## 12.6.9　安全与护栏 {#safety-and-guardrails}

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

## 12.6.10　多应用工作流中的信用分配 {#credit-assignment-in-multi-app-workflows}

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

## 12.6.11　扩展与基础设施 {#scaling-and-infrastructure}

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

## 12.6.12　评估框架 {#evaluation-framework}

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

## 12.6.13　来自生产部署的经验教训 {#lessons-from-production-deployments}

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

## 12.6.14　完整训练食谱 {#complete-training-recipe}

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
