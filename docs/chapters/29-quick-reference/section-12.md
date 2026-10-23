# 29.12　常见失效模式与修复 {#common-failure-modes-fixes}

<span id="table-29-inline-8"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>症状</strong></th><th scope="col"><strong>可能原因</strong></th><th scope="col"><strong>修复方法</strong></th></tr></thead>
<tbody><tr><td>奖励上升、质量下降</td><td>奖励黑客（Reward Hacking）</td><td>RM 集成、长度惩罚、增大 <span class="arithmatex">\(\beta\)</span></td></tr>
<tr><td>KL 爆炸（<span class="arithmatex">\(&gt;\)</span>15）</td><td>学习率过高或模式崩溃</td><td>降低学习率、回滚 checkpoint</td></tr>
<tr><td>熵塌缩</td><td>过早收敛</td><td>增大熵系数、提高温度</td></tr>
<tr><td>训练损失 NaN</td><td>梯度爆炸</td><td>降低学习率、增大梯度裁剪、检查数据</td></tr>
<tr><td>5K 步后无提升</td><td>Prompt 分布不当</td><td>Goldilocks 过滤（通过率 20–80%）</td></tr>
<tr><td>基准性能回退</td><td>对齐税（Alignment Tax）</td><td>减小 RL 预算、改用 LoRA、混入 SFT 数据</td></tr>
<tr><td>长度单调增长</td><td>RM 中的长度漏洞</td><td>长度惩罚、用长度控制重训 RM</td></tr>
<tr><td>生成时 OOM</td><td>KV cache 溢出</td><td>减小 batch、增大 TP、使用 PagedAttention</td></tr>
<tr><td>Agent 永久循环</td><td>缺少最大迭代守卫</td><td>设置 <code>max_iterations</code>、加循环检测</td></tr>
<tr><td>工具调用解析失败</td><td>输出格式不一致</td><td>少样本示例、约束解码</td></tr>
<tr><td>RAG 返回无关文档</td><td>embedding / chunking 不佳</td><td>Reranker、混合检索、更小 chunk</td></tr>
<tr><td>多智能体死锁</td><td>循环依赖</td><td>DAG 约束、每个 Agent 设置超时</td></tr></tbody>
</table>
</div>
