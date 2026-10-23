# 19.7　终止工程 {#sec:termination-engineering}

<p>朴素循环的标志性失败是它永远不会停。终止设计不是事后补丁——它是工程的一半。</p>

## 19.7.1　终止条件 {#sec:termination-conditions}

<p>生产级循环携带多个独立的退出条件：</p>

<ol>
<li><p><strong>目标达成</strong>：验证器确认目标已实现。这是唯一<em>成功</em>的终止。</p></li>
<li><p><strong>硬性迭代上限</strong>：循环步数上限（通常 20–50）。无论其他条件如何，防止无界执行。</p></li>
<li><p><strong>预算耗尽</strong>：token 数、墙钟时间或金钱成本超过预定义上限。</p></li>
<li><p><strong>无进展检测</strong>：最近 <span class="arithmatex">\(k\)</span> 轮迭代没有产生可测量的状态变化——循环在震荡或卡死。这是硬性上限之后最重要的安全机制。</p></li>
<li><p><strong>致命错误</strong>：任何迭代都无法解决的不可恢复状况（凭据缺失、基础设施故障）。</p></li>
</ol>

<div class="admonition note"><p class="admonition-title">无进展检测：停滞断路器</p>
<p>检测停滞需要在迭代间追踪一个进展信号。常见方法：</p>

<ul>
<li><p><strong>输出哈希比较</strong>：如果最近 3 次 Agent 输出或文件状态完全相同，循环在打转。</p></li>
<li><p><strong>错误重复</strong>：同一错误消息连续出现 <span class="arithmatex">\(k\)</span> 轮。</p></li>
<li><p><strong>指标平台期</strong>：数值信号（测试通过率、验证损失）在 <span class="arithmatex">\(k\)</span> 步内没有提升 <span class="arithmatex">\(\epsilon\)</span>。</p></li>
<li><p><strong>Diff 熵</strong>：每轮迭代的代码改动量正趋近于零——Agent 在做越来越小、无关紧要的编辑。</p></li>
</ul>

<p>检测到停滞时，正确的行动是<em>升级</em>，而不是继续迭代。循环保存状态，把控制权交给人类或更高级的编排器。</p>
</div>

## 19.7.2　探索问题 {#sec:loop-exploration}

<p>卡在局部最小值的循环——用微小的变体重复同一种失败方法——正在表现出 RL 中熟悉的探索—利用失败。循环内促进探索的机制：</p>

<ul>
<li><p><strong>基于反思的探索</strong>：<span class="arithmatex">\(k\)</span> 次失败尝试之后，触发一个反思步骤，显式要求模型考虑根本不同的方法（类似于 PPO 中的熵正则化 <a class="book-citation" id="cite-schulman2017proximal-1" href="../../references.md#ref-schulman2017proximal" aria-label="参考文献 319">[319]</a>）。</p></li>
<li><p><strong>温度升级</strong>：连续失败后提高采样温度，鼓励多样化输出。</p></li>
<li><p><strong>策略记忆</strong>：不仅记录失败的动作，还记录失败的<em>策略</em>。“方法 A（修改 handler）失败了 3 次；改试方法 B（重写 schema）。”</p></li>
<li><p><strong>全新子 Agent</strong>：生成一个上下文窗口干净的子 Agent——它不会陷入已累积上下文可能造成的窠臼。</p></li>
</ul>
