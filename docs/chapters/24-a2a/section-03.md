# 24.3　通信模式 {#sec:a2a:patterns}

<p>多 Agent 系统会根据任务的性质、延迟要求以及参与 Agent 的数量，采用多种不同的通信模式。</p>

## 24.3.1　请求-响应（Request-Response） {#request-response}

<p>最简单的模式：Agent A 把任务发给 Agent B，并等待完整的响应。适用于结果需要在继续之前获得的、简短且定义明确的子任务。</p>

## 24.3.2　流式传输（Streaming） {#streaming}

<p>Agent A 打开一个 SSE 连接；Agent B 在产生部分结果时以流的形式发送出来。非常适合长文本生成（报告、代码）、实时协作或渐进式 UI 更新。</p>

<div class="admonition example"><p class="admonition-title">流式传输模式的应用场景</p>
<p>编排器请求 <code>WritingAgent</code> 起草一份 10 页的技术文档。编排器无需等待 2 分钟以获得完整文档，而是在每一节写完时即流式接收，从而让 <code>ReviewAgent</code> 在后续章节仍在生成时就开始审阅前面的章节——这种流水线将总延迟降低了 40–60%。</p>
</div>

## 24.3.3　多轮交互（Multi-Turn） {#multi-turn-interaction}

<p>某些任务需要迭代式精化。Agent 进入 <code>input-required</code> 状态，编排器提供澄清信息，然后任务恢复执行。这与人类协作流程相似：起草 <span class="arithmatex">\(\to\)</span> 反馈 <span class="arithmatex">\(\to\)</span> 修改。</p>

<div class="highlight"><pre><span></span><code><span class="c1"># 多轮：编排器处理 input-required 状态</span>
<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">run_multiturn_task</span><span class="p">(</span><span class="n">client</span><span class="p">,</span> <span class="n">initial_message</span><span class="p">):</span>
    <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">send_task</span><span class="p">(</span><span class="n">message</span><span class="o">=</span><span class="n">initial_message</span><span class="p">)</span>

    <span class="k">while</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="ow">not</span> <span class="ow">in</span> <span class="p">(</span><span class="s2">"completed"</span><span class="p">,</span> <span class="s2">"failed"</span><span class="p">,</span> <span class="s2">"canceled"</span><span class="p">):</span>
        <span class="k">if</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="o">==</span> <span class="s2">"input-required"</span><span class="p">:</span>
            <span class="c1"># Agent 需要澄清</span>
            <span class="n">clarification_needed</span> <span class="o">=</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">message</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Agent asks: </span><span class="si">{</span><span class="n">clarification_needed</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

            <span class="c1"># 编排器生成或转发澄清回复</span>
            <span class="n">user_reply</span> <span class="o">=</span> <span class="k">await</span> <span class="n">get_clarification</span><span class="p">(</span><span class="n">clarification_needed</span><span class="p">)</span>

            <span class="c1"># 发送回复以继续任务</span>
            <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">send_task</span><span class="p">(</span>
                <span class="n">task_id</span><span class="o">=</span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">,</span>
                <span class="n">message</span><span class="o">=</span><span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>
                         <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"text"</span><span class="p">,</span> <span class="s2">"text"</span><span class="p">:</span> <span class="n">user_reply</span><span class="p">}]}</span>
            <span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="c1"># 仍在执行 —— 延迟后再轮询</span>
            <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
            <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">get_task</span><span class="p">(</span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">task</span>
</code></pre></div>

## 24.3.4　广播（Broadcast） {#broadcast}

<p>编排器将同一条消息同时发送给多个 Agent——适用于发布通告、分发共享上下文，或触发并行独立的工作流。</p>

## 24.3.5　发布-订阅（Publish-Subscribe, Pub-Sub） {#publish-subscribe-pub-sub}

<p>Agent 订阅事件通道（例如 <code>new-document-uploaded</code>、<code>model-retrained</code>）。当事件触发时，所有订阅的 Agent 都会收到通知。这种方式解耦了生产者与消费者，并支持响应式、事件驱动的架构。</p>

## 24.3.6　协商（Negotiation） {#negotiation}

<p>两个 Agent 互相交换提案与反提案，以便在计划、资源分配或方案上达成一致。常见于 Agent 具有不同目标或约束的多 Agent 规划系统中。</p>

<div class="admonition example"><p class="admonition-title">协商模式</p>
<p><code>PlannerAgent</code> 提出一个 5 步研究计划。<code>ResourceAgent</code> 回复说第 3 步（运行一次大规模仿真）将超出计算预算。<code>PlannerAgent</code> 反提议一次规模缩小的仿真。<code>ResourceAgent</code> 批准。随后达成一致的计划被分派给执行型 Agent。</p>
</div>

## 24.3.7　基于拍卖的任务分配（Auction-Based Task Allocation） {#auction-based-task-allocation}

<p>编排器宣布带需求的任务；候选 Agent 提交竞标（预计完成时间、置信度、成本）；编排器把任务授予中标方。这使得在一组 Agent 上进行动态的、市场化的负载均衡成为可能。</p>

<p class="table-caption" id="table-24-1"><span class="table-number">表 24.1　</span>A2A 通信模式总览。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模式</strong></th><th scope="col"><strong>延迟</strong></th><th scope="col"><strong>最佳适用场景</strong></th></tr></thead>
<tbody><tr><td>请求-响应</td><td>低</td><td>简短、定义明确的子任务</td></tr>
<tr><td>流式传输</td><td>低（首 Token）</td><td>长文本生成、实时 UI</td></tr>
<tr><td>多轮</td><td>中</td><td>需要澄清的模糊任务</td></tr>
<tr><td>广播</td><td>低</td><td>共享上下文分发</td></tr>
<tr><td>发布-订阅</td><td>可变</td><td>事件驱动的响应式工作流</td></tr>
<tr><td>协商</td><td>中–高</td><td>资源受限的规划</td></tr>
<tr><td>拍卖</td><td>中</td><td>动态负载均衡</td></tr></tbody>
</table>
</div>
