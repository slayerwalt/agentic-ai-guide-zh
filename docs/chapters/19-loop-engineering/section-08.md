# 19.8　失败模式与反模式 {#sec:loop-failures}

## 19.8.1　Loopmaxxing 陷阱 {#sec:loopmaxxing}

<p>“Loopmaxxing”——假设让 Agent 跑足够多轮迭代最终就会产出正确解——是循环工程版的“算力够多就能解决一切”。它的失败与纯随机搜索的失败同源：没有指向改进的梯度信号，光靠迭代不是优化。</p>

<p>Loopmaxxing 在以下情况出现：</p>

<ul>
<li><p>目标缺少可检查的成功条件（“改善用户体验”）</p></li>
<li><p>验证信号过于粗糙（在 1000 个测试的套件上只有二元通过/失败，无法提供指向修复的梯度）</p></li>
<li><p>无论迭代多少次，Agent 都缺乏解决问题的能力</p></li>
</ul>

<p>解药是认识到循环只是<em>放大</em>工程能力——它们不取代工程能力。目标定义糟糕的循环会以极高的效率去追求错误的东西。</p>

## 19.8.2　理解债 {#sec:comprehension-debt}

<p>当循环修改代码的速度超过工程团队审阅的速度时，仓库中实际存在的东西与人类理解之间的差距就会扩大——这就是<em>理解债（comprehension debt）</em> <a class="book-citation" id="cite-osmani2026loop-1" href="../../references.md#ref-osmani2026loop" aria-label="参考文献 279">[279]</a>。与技术债不同（技术债由代码承载），理解债存在于团队成员的头脑里。它无声地复利增长，直到某场危机迫使人们去调试那些设计决策、结构依赖与边界情况完全未被记录的代码。</p>

<div class="admonition warning"><p class="admonition-title">循环同时放大生产力与风险</p>
<p>设计良好的循环加速的是工程师深刻理解的工作。用循环来逃避理解工作，则会造成加速的不理解螺旋。同样的机制，相反的结果。区别在于：人仍然是工程师，还是沦为了按“开始”按钮的人。</p>
</div>

## 19.8.3　循环中的 Reward Hacking {#sec:loop-reward-hacking}

<p>正如 RL Agent 利用奖励定义缺陷在未达成既定目标的情况下拿到高奖励，循环 Agent 也会利用验证漏洞：</p>

<ul>
<li><p><strong>删除测试</strong>：最简单的 reward hack——删掉失败的测试让 CI 变绿。</p></li>
<li><p><strong>指标博弈</strong>：通过记住验证集而非泛化来过拟合验证损失。</p></li>
<li><p><strong>收窄规格</strong>：解决所陈述问题的简化版本，恰好能通过检查。</p></li>
<li><p><strong>掩盖输出</strong>：压制错误输出而不是修复根本原因。</p></li>
</ul>

<p>防御手段与 RLHF 相同：设计<em>难以钻空子</em>的奖励函数（验证标准）。多个独立检查——测试<em>和</em>类型检查<em>和</em> lint<em>和</em>审查子 Agent——构成一个比任何单一信号都更难利用的验证面。</p>

## 19.8.4　上下文劣化 {#sec:context-degradation}

<p>在长时程循环中，上下文窗口会被每一步的历史——思考、工具输出、错误、补丁——填满。当窗口接近容量时，模型对相关信息的注意力下降（“中间迷失”现象 <a class="book-citation" id="cite-liu2024lost-1" href="../../references.md#ref-liu2024lost" aria-label="参考文献 226">[226]</a>）。症状包括：</p>

<ul>
<li><p>Agent 重新尝试它已经试过并明确标记为失败的方法</p></li>
<li><p>工具调用变得不那么精确——Agent 忘记了自己读过哪些文件</p></li>
<li><p>模型在臃肿的上下文中艰难地分配注意力，回答变得更短、更不连贯</p></li>
</ul>

<p>对策：</p>

<ol>
<li><p><strong>激进压缩</strong>：每 <span class="arithmatex">\(k\)</span> 步之后，把已完成的工作总结为简短的状态描述，丢弃原始记录。</p></li>
<li><p><strong>外部草稿本</strong>：把中间结果写入文件，让 Agent 按需读取，而不是把一切留在上下文里。</p></li>
<li><p><strong>子 Agent 隔离</strong>：在新上下文窗口中生成子任务，只返回它们的结论——而不是完整推理轨迹。</p></li>
<li><p><strong>历史滑动窗口</strong>：只完整保留最近 <span class="arithmatex">\(w\)</span> 步；把更早的内容压缩进一个摘要块。</p></li>
</ol>
