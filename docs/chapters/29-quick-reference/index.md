# 第 29 章 · 速查手册 {#quick-reference}

<p>本章汇总了关键公式、架构规格、API 参考以及失效模式诊断，便于开发与调试时快速查阅。</p>

## 29.1　核心 RL 与对齐公式 {#core-rl-alignment-equations}



<div class="arithmatex">\[\begin{align*}
\text{PPO Clip:}&amp;\quad L = \mathbb{E}[\min(r_t\hat{A}_t, \text{clip}(r_t,1{\pm}\epsilon)\hat{A}_t)], \quad r_t = \pi_\theta(a_t|s_t)/\pi_{\text{old}}(a_t|s_t) \tag{29.1}\\

\text{DPO:}&amp;\quad L = -\mathbb{E}[\log\sigma(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)})] \tag{29.2}\\

\text{GRPO:}&amp;\quad \hat{A}_i = (r_i - \mu_G)/\sigma_G, \quad \text{then PPO clip update (no critic)} \tag{29.3}\\

\text{KTO:}&amp;\quad L = \lambda_w(1 - v(y_w)) + \lambda_l \cdot v(y_l), \quad v = \sigma(\beta\log(\pi_\theta/\pi_\text{ref}) - z) \tag{29.4}\\

\text{IPO:}&amp;\quad L = \mathbb{E}[(\log(\pi_\theta(y_w)/\pi_\text{ref}(y_w)) - \log(\pi_\theta(y_l)/\pi_\text{ref}(y_l)) - 1/(2\beta))^2] \tag{29.5}\\

\text{ORPO:}&amp;\quad L = L_\text{SFT}(y_w) - \lambda\log\sigma(\log(\text{odds}(y_w)/\text{odds}(y_l))) \tag{29.6}\\

\text{GAE:}&amp;\quad \hat{A}_t = \textstyle\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \tag{29.7}\\

\text{KL Penalty:}&amp;\quad R_\text{total} = r_\phi(x,y) - \beta D_\text{KL}[\pi_\theta(y|x)\|\pi_\text{ref}(y|x)] \tag{29.8}\\

\text{RM (Bradley-Terry):}&amp;\quad L = -\mathbb{E}[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))] \tag{29.9}\\

\text{Best-of-N:}&amp;\quad y^* = \arg\max_{y_i \sim \pi_\theta(\cdot|x),\, i=1..N} r_\phi(x, y_i)
\tag{29.10}
\end{align*}\]</div>

## 29.2　Transformer 与架构公式 {#transformer-architecture-formulas}



<div class="arithmatex">\[\begin{align*}
\text{Self-Attention:}&amp;\quad \text{Attn}(Q,K,V) = \text{softmax}(QK^\top / \sqrt{d_k}) \cdot V \tag{29.11}\\

\text{Multi-Head:}&amp;\quad \text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O,\quad \text{head}_i = \text{Attn}(XW_i^Q, XW_i^K, XW_i^V) \tag{29.12}\\

\text{RoPE:}&amp;\quad f(x_m, m) = x_m e^{im\theta_j}, \quad \theta_j = 10000^{-2j/d} \tag{29.13}\\

\text{LoRA:}&amp;\quad W&#x27; = W_0 + (\alpha/r) \cdot BA, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k} \tag{29.14}\\

\text{KD (soft targets):}&amp;\quad L_\text{KD} = (1{-}\alpha)L_\text{CE}(y, \hat{y}) + \alpha\, T^2 \cdot \text{KL}(p_T^\text{teacher} \| p_T^\text{student}) \tag{29.15}\\

\text{FFN (SwiGLU):}&amp;\quad \text{FFN}(x) = (\text{Swish}(xW_1) \odot xW_3) W_2
\tag{29.16}
\end{align*}\]</div>

## 29.3　解码方法 {#decoding-methods}

<span id="table-29-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>公式 / 规则</strong></th><th scope="col"><strong>关键参数</strong></th></tr></thead>
<tbody><tr><td>Greedy</td><td><span class="arithmatex">\(y_t = \arg\max_v P(v|y_{&lt;t})\)</span></td><td>—</td></tr>
<tr><td>Beam search</td><td>按联合概率保留 top-<span class="arithmatex">\(B\)</span> 个部分序列</td><td><span class="arithmatex">\(B=4\)</span>–<span class="arithmatex">\(8\)</span></td></tr>
<tr><td>Temperature</td><td><span class="arithmatex">\(P&#x27;(v) = \text{softmax}(\text{logit}_v / T)\)</span></td><td><span class="arithmatex">\(T \in [0.1, 1.5]\)</span></td></tr>
<tr><td>Top-<span class="arithmatex">\(k\)</span></td><td>仅保留 top-<span class="arithmatex">\(k\)</span> 个 logit，其余置零后重新归一化</td><td><span class="arithmatex">\(k=40\)</span>–<span class="arithmatex">\(100\)</span></td></tr>
<tr><td>Top-<span class="arithmatex">\(p\)</span> (nucleus)</td><td>保留最小集合 <span class="arithmatex">\(V&#x27;\)</span> 使 <span class="arithmatex">\(\sum_{v \in V&#x27;} P(v) \geq p\)</span></td><td><span class="arithmatex">\(p=0.9\)</span>–<span class="arithmatex">\(0.95\)</span></td></tr>
<tr><td>Min-<span class="arithmatex">\(p\)</span></td><td>保留满足 <span class="arithmatex">\(P(v) \geq p_\text{min} \cdot P(v_\text{max})\)</span> 的 token</td><td><span class="arithmatex">\(p_\text{min}=0.05\)</span>–<span class="arithmatex">\(0.1\)</span></td></tr>
<tr><td>Repetition penalty</td><td>若 <span class="arithmatex">\(v\)</span> 已出现，则 <span class="arithmatex">\(\text{logit}_v \leftarrow \text{logit}_v / \theta\)</span></td><td><span class="arithmatex">\(\theta=1.1\)</span>–<span class="arithmatex">\(1.3\)</span></td></tr></tbody>
</table>
</div>

## 29.4　系统与并行 {#systems-parallelism}

<span id="table-29-inline-2"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>公式</strong></th><th scope="col"><strong>取值 (70B, BF16)</strong></th><th scope="col"><strong>说明</strong></th></tr></thead>
<tbody><tr><td>模型显存</td><td><span class="arithmatex">\(2P\)</span> 字节</td><td><span class="arithmatex">\(140\)</span> GB（仅权重）</td></tr>
<tr><td>Adam 优化器</td><td><span class="arithmatex">\(2P \times 4\)</span> 字节 (m + v)</td><td><span class="arithmatex">\(280\)</span> GB</td></tr>
<tr><td>完整训练占用</td><td><span class="arithmatex">\(\sim 8P\)</span> 字节</td><td><span class="arithmatex">\(560\)</span> GB（权重 + 优化器 + 梯度）</td></tr>
<tr><td>FSDP 每 GPU 显存</td><td><span class="arithmatex">\(8P / N_\text{GPUs}\)</span></td><td>8 卡时为 <span class="arithmatex">\(70\)</span> GB</td></tr>
<tr><td>生成算术强度</td><td><span class="arithmatex">\(2P / 2P = 1\)</span> FLOP/byte</td><td>严重内存瓶颈</td></tr>
<tr><td>Token 速率（生成）</td><td>HBM_BW <span class="arithmatex">\(/ (2P)\)</span></td><td><span class="arithmatex">\(\sim\)</span>14 tok/s (A100, batch=1)</td></tr>
<tr><td>TP AllReduce / 层</td><td><span class="arithmatex">\(2 \times 2 \cdot \frac{T-1}{T} \cdot bsd\)</span> 字节</td><td><span class="arithmatex">\(\sim\)</span>188 MB (70B, TP=8)</td></tr>
<tr><td>PP 气泡占比</td><td><span class="arithmatex">\((P-1)/(P+M-1)\)</span></td><td><span class="arithmatex">\(P\)</span>=阶段数，<span class="arithmatex">\(M\)</span>=微批数</td></tr>
<tr><td>MFU</td><td>observed_toks <span class="arithmatex">\(\times\)</span> 6<span class="arithmatex">\(P\)</span> / peak_FLOPS</td><td>目标：<span class="arithmatex">\(&gt;40\%\)</span></td></tr></tbody>
</table>
</div>

## 29.5　GPU 硬件规格 {#gpu-hardware-specs}

<span id="table-29-inline-3"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>GPU</strong></th><th scope="col"><strong>显存</strong></th><th scope="col"><strong>带宽 (HBM)</strong></th><th scope="col"><strong>BF16 TFLOPS</strong></th><th scope="col"><strong>NVLink</strong></th><th scope="col"><strong>备注</strong></th></tr></thead>
<tbody><tr><td>A100-80GB</td><td>80 GB HBM2e</td><td>2.0 TB/s</td><td>312</td><td>600 GB/s</td><td>主力型号，广泛可用</td></tr>
<tr><td>H100-80GB</td><td>80 GB HBM3</td><td>3.35 TB/s</td><td>989</td><td>900 GB/s</td><td>当前一代，支持 FP8</td></tr>
<tr><td>H200-141GB</td><td>141 GB HBM3e</td><td>4.8 TB/s</td><td>989</td><td>900 GB/s</td><td>大上下文 / 更少 GPU</td></tr>
<tr><td>B200</td><td>192 GB HBM3e</td><td>8.0 TB/s</td><td>2250</td><td>1800 GB/s</td><td>下一代（2025）</td></tr></tbody>
</table>
</div>

## 29.6　超参数范围 {#hyperparameter-ranges}

<span id="table-29-inline-4"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>参数</strong></th><th scope="col"><strong>典型范围</strong></th><th scope="col"><strong>默认值</strong></th><th scope="col"><strong>说明</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(\beta\)</span> (DPO/KTO)</td><td>0.05–0.5</td><td>0.1</td><td>越大越保守</td></tr>
<tr><td><span class="arithmatex">\(\epsilon\)</span> (PPO clip)</td><td>0.1–0.3</td><td>0.2</td><td>越大更新越激进</td></tr>
<tr><td><span class="arithmatex">\(\gamma\)</span> (GAE 折扣)</td><td>0.99–1.0</td><td>1.0</td><td>情景式任务使用 1.0</td></tr>
<tr><td><span class="arithmatex">\(\lambda\)</span> (GAE)</td><td>0.9–0.99</td><td>0.95</td><td>越小偏差越大、方差越小</td></tr>
<tr><td>KL 系数 (<span class="arithmatex">\(\beta_\text{KL}\)</span>)</td><td>0.01–0.2</td><td>0.05</td><td>自适应目标 KL <span class="arithmatex">\(\approx\)</span> 5–8</td></tr>
<tr><td>学习率 (RLHF)</td><td>1e-7 – 5e-6</td><td>5e-7</td><td>远低于预训练</td></tr>
<tr><td>学习率 (SFT)</td><td>1e-5 – 5e-5</td><td>2e-5</td><td>标准微调范围</td></tr>
<tr><td>LoRA 秩 <span class="arithmatex">\(r\)</span></td><td>8–128</td><td>16–64</td><td>越大容量越大、显存越多</td></tr>
<tr><td>LoRA alpha <span class="arithmatex">\(\alpha\)</span></td><td><span class="arithmatex">\(r\)</span> – <span class="arithmatex">\(2r\)</span></td><td><span class="arithmatex">\(2r\)</span></td><td>缩放因子；<span class="arithmatex">\(\alpha/r\)</span> 为有效尺度</td></tr>
<tr><td>温度（生成）</td><td>0.6–1.0</td><td>0.7</td><td>越低候选越同质</td></tr>
<tr><td>生成数量 <span class="arithmatex">\(K\)</span></td><td>4–64</td><td>4–16</td><td>用于 GRPO / Online DPO / Best-of-N</td></tr>
<tr><td>梯度裁剪范数</td><td>0.5–2.0</td><td>1.0</td><td>防止梯度爆炸</td></tr></tbody>
</table>
</div>

## 29.7　TRL API 速查 {#trl-api-quick-reference}

<span id="table-29-inline-5"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>Trainer</strong></th><th scope="col"><strong>方法</strong></th><th scope="col"><strong>关键配置</strong></th><th scope="col"><strong>数据格式</strong></th></tr></thead>
<tbody><tr><td><code>SFTTrainer</code></td><td>监督微调</td><td><code>packing, max_seq_length</code></td><td>prompt + completion</td></tr>
<tr><td><code>RewardTrainer</code></td><td>奖励模型</td><td><code>center_rewards_coefficient</code></td><td>prompt + chosen + rejected</td></tr>
<tr><td><code>PPOTrainer</code></td><td>PPO</td><td><code>init_kl_coef, target_kl, cliprange</code></td><td>prompts（在线生成）</td></tr>
<tr><td><code>DPOTrainer</code></td><td>DPO / IPO</td><td><code>beta, loss_type=&quot;sigmoid&quot;/&quot;ipo&quot;</code></td><td>prompt + chosen + rejected</td></tr>
<tr><td><code>GRPOTrainer</code></td><td>GRPO</td><td><code>num_generations, beta, use_vllm</code></td><td>prompts + reward_fn</td></tr>
<tr><td><code>OnlineDPOTrainer</code></td><td>Online DPO</td><td><code>num_generations, reward_model_path</code></td><td>prompts（在线生成）</td></tr>
<tr><td><code>KTOTrainer</code></td><td>KTO</td><td><code>desirable_weight, undesirable_weight</code></td><td>prompt + completion + label</td></tr>
<tr><td><code>ORPOTrainer</code></td><td>ORPO</td><td><code>beta</code></td><td>prompt + chosen + rejected</td></tr>
<tr><td><code>Best-of-N (manual)</code></td><td>Best-of-N</td><td><code>n_samples</code></td><td>prompts（推理）</td></tr></tbody>
</table>
</div>

## 29.8　RAG 流水线公式 {#rag-pipeline-formulas}



<div class="arithmatex">\[\begin{align*}
\text{Cosine similarity:}&amp;\quad \text{sim}(q, d) = \frac{q \cdot d}{\|q\| \cdot \|d\|} \tag{29.17}\\

\text{Retrieval:}&amp;\quad \mathcal{D}_k = \text{top-}k_{d \in \mathcal{C}} \; \text{sim}(\text{embed}(q),\; \text{embed}(d)) \tag{29.18}\\

\text{RAG generation:}&amp;\quad P(y|q) = P_\text{LLM}(y \;|\; q, \mathcal{D}_k) \tag{29.19}\\

\text{Chunking overlap:}&amp;\quad \text{stride} = \text{chunk\_size} - \text{overlap} \tag{29.20}\\

\text{Reranker (cross-enc):}&amp;\quad \text{score}(q, d) = \text{MLP}(\text{BERT}([q; d]))
\tag{29.21}
\end{align*}\]</div>

## 29.9　智能体设计模式 {#agentic-design-patterns}

<span id="table-29-inline-6"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模式</strong></th><th scope="col"><strong>结构</strong></th><th scope="col"><strong>最适用于</strong></th></tr></thead>
<tbody><tr><td>ReAct</td><td>Think <span class="arithmatex">\(\to\)</span> Act <span class="arithmatex">\(\to\)</span> Observe <span class="arithmatex">\(\to\)</span> 循环</td><td>通用工具使用型 Agent</td></tr>
<tr><td>Plan-and-Execute</td><td>Plan <span class="arithmatex">\(\to\)</span> 执行步骤 <span class="arithmatex">\(\to\)</span> 修订</td><td>长程、结构化任务</td></tr>
<tr><td>Supervisor</td><td>路由器 <span class="arithmatex">\(\to\)</span> 专家 Agent</td><td>多领域、子任务边界清晰</td></tr>
<tr><td>Swarm (handoffs)</td><td>Agent 移交控制权 + 上下文</td><td>客服、问题升级流程</td></tr>
<tr><td>Hierarchical</td><td>委派 Agent 的树状结构</td><td>复杂任务分解</td></tr>
<tr><td>Human-in-the-loop</td><td>Agent <span class="arithmatex">\(\to\)</span> 审批关卡 <span class="arithmatex">\(\to\)</span> 继续</td><td>高风险、不可逆动作</td></tr></tbody>
</table>
</div>

## 29.10　智能体通信协议 {#agent-communication-protocols}

<span id="table-29-inline-7"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>协议</strong></th><th scope="col"><strong>范围</strong></th><th scope="col"><strong>传输方式</strong></th><th scope="col"><strong>核心概念</strong></th></tr></thead>
<tbody><tr><td>MCP</td><td>工具集成</td><td>stdio / HTTP+SSE</td><td>服务端暴露工具；客户端发现并调用</td></tr>
<tr><td>A2A</td><td>智能体到智能体</td><td>HTTP + JSON-RPC</td><td>带生命周期的任务（submitted<span class="arithmatex">\(\to\)</span>working<span class="arithmatex">\(\to\)</span>done）</td></tr>
<tr><td>OpenAI Function Calling</td><td>工具使用</td><td>API 负载</td><td>在 <code>tools[]</code> 数组中以 JSON schema 描述</td></tr></tbody>
</table>
</div>

## 29.11　上下文窗口预算 {#context-window-budget}



<div class="arithmatex">\[C \geq \underbrace{S}_{\text{system}} + \underbrace{M}_{\text{memory/RAG}} + \underbrace{T}_{\text{tool defs}} + \underbrace{H}_{\text{history}} + \underbrace{R}_{\text{reserved output}}
\tag{29.22}\]</div>

<p><strong>经验法则</strong>（针对 128K 上下文）：</p>

<ul>
<li><p>系统提示：1–4K token（固定）</p></li>
<li><p>工具定义：2–8K（随工具数量增长）</p></li>
<li><p>RAG 上下文：4–16K（top-<span class="arithmatex">\(k\)</span> 个 chunk）</p></li>
<li><p>历史记录：无界增长 <span class="arithmatex">\(\rightarrow\)</span> 汇总或截断</p></li>
<li><p>预留输出：2–8K</p></li>
</ul>

## 29.12　常见失效模式与修复 {#common-failure-modes-fixes}

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

## 29.13　方法选择决策树 {#method-selection-decision-tree}

<ol>
<li><p><strong>是否有成对偏好数据（chosen + rejected）？</strong></p>

<ul>
<li><p>标签噪声大 <span class="arithmatex">\(\rightarrow\)</span> <strong>IPO</strong></p></li>
<li><p>显存受限、尚未做 SFT <span class="arithmatex">\(\rightarrow\)</span> <strong>ORPO</strong></p></li>
<li><p>数据干净、算力有限 <span class="arithmatex">\(\rightarrow\)</span> <strong>DPO</strong></p></li>
<li><p>DPO 进入瓶颈、希望加入探索 <span class="arithmatex">\(\rightarrow\)</span> <strong>Online DPO</strong></p></li>
</ul></li>
<li><p><strong>只有二值反馈（点赞 / 点踩）？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>KTO</strong></p></li>
<li><p><strong>有可验证奖励（数学 / 代码）？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>GRPO</strong></p></li>
<li><p><strong>不惜代价追求最高质量？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>PPO</strong></p></li>
<li><p><strong>想要免训练的改进？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>Best-of-N</strong></p></li>
</ol>

## 29.14　评估指标 {#evaluation-metrics}

<span id="table-29-inline-9"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>指标</strong></th><th scope="col"><strong>范围</strong></th><th scope="col"><strong>衡量内容</strong></th></tr></thead>
<tbody><tr><td>Perplexity</td><td><span class="arithmatex">\([1, \infty)\)</span></td><td>模型的“困惑度”；越低语言建模越好</td></tr>
<tr><td>Win Rate（对比基线）</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>评审/人类偏好的输出比例</td></tr>
<tr><td>BLEU</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>与参考文本的 <span class="arithmatex">\(n\)</span>-gram 重叠（偏精确率）</td></tr>
<tr><td>ROUGE-L</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>与参考文本的最长公共子序列</td></tr>
<tr><td>Pass@<span class="arithmatex">\(k\)</span></td><td><span class="arithmatex">\([0, 1]\)</span></td><td><span class="arithmatex">\(k\)</span> 个代码样本中至少 1 个通过测试的概率</td></tr>
<tr><td>MMLU / GPQA</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>知识/推理基准上的多选准确率</td></tr>
<tr><td>HumanEval</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>生成代码的功能正确率</td></tr>
<tr><td>Faithfulness (RAG)</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>受检索上下文支持的论断比例</td></tr>
<tr><td>Context Relevancy</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>检索内容中与查询相关的比例</td></tr>
<tr><td>Answer Relevancy</td><td><span class="arithmatex">\([0, 1]\)</span></td><td>答案对问题的回应程度</td></tr></tbody>
</table>
</div>

## 29.15　推理与测试时扩展 {#reasoning-test-time-scaling}

<span id="table-29-inline-10"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>算力开销</strong></th><th scope="col"><strong>机制</strong></th></tr></thead>
<tbody><tr><td>思维链（Chain-of-Thought，CoT）</td><td>1.5–3<span class="arithmatex">\(\times\)</span> token</td><td>在 prompt 中加入 “step by step”</td></tr>
<tr><td>自洽性（Self-Consistency）</td><td><span class="arithmatex">\(N \times\)</span> 生成</td><td>采样 <span class="arithmatex">\(N\)</span> 条 CoT 路径，对最终答案多数表决</td></tr>
<tr><td>思维树（Tree-of-Thought, ToT）</td><td><span class="arithmatex">\(B \times D \times\)</span> 生成</td><td>在推理树上 BFS/DFS；对分支评估</td></tr>
<tr><td>Best-of-<span class="arithmatex">\(N\)</span></td><td><span class="arithmatex">\(N \times\)</span> 生成</td><td>采样 <span class="arithmatex">\(N\)</span>，用 RM 打分，取最高</td></tr>
<tr><td>Beam search（推理上）</td><td><span class="arithmatex">\(B \times\)</span> 生成</td><td>维持 top-<span class="arithmatex">\(B\)</span> 个部分推理链</td></tr>
<tr><td>预算强制（Budget forcing）</td><td>可变</td><td>动态为更难问题分配更多 token</td></tr>
<tr><td>验证（ORM/PRM）</td><td><span class="arithmatex">\(N \times\)</span> 生成 + 打分</td><td>生成 <span class="arithmatex">\(N\)</span> 个解，按结果/过程 RM 排序</td></tr></tbody>
</table>
</div>

## 29.16　记忆系统类型 {#memory-system-types}

<span id="table-29-inline-11"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>类型</strong></th><th scope="col"><strong>存储</strong></th><th scope="col"><strong>使用场景</strong></th></tr></thead>
<tbody><tr><td>工作记忆</td><td>上下文窗口</td><td>当前对话、即时工具结果</td></tr>
<tr><td>情景记忆（Episodic Memory）</td><td>向量存储</td><td>历史交互、用户偏好、会话历史</td></tr>
<tr><td>语义记忆（Semantic Memory）</td><td>知识图谱 / embedding</td><td>事实、概念、领域知识</td></tr>
<tr><td>程序性记忆（Procedural Memory）</td><td>技能库 / 代码</td><td>操作手册、学到的工作流</td></tr></tbody>
</table>
</div>

## 29.17　MCP 速查 {#mcp-quick-reference}

<span id="table-29-inline-12"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>原语</strong></th><th scope="col"><strong>方向</strong></th><th scope="col"><strong>有副作用？</strong></th><th scope="col"><strong>用途</strong></th></tr></thead>
<tbody><tr><td>Tools</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务端</td><td>是</td><td>执行动作（创建、修改、删除）</td></tr>
<tr><td>Resources</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务端</td><td>否（只读）</td><td>读取数据（文件、DB 记录、配置）</td></tr>
<tr><td>Prompts</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务端</td><td>否</td><td>常用任务的可复用模板</td></tr>
<tr><td>Sampling</td><td>服务端 <span class="arithmatex">\(\to\)</span> 客户端</td><td>否</td><td>服务端请求客户端调用 LLM 生成</td></tr></tbody>
</table>
</div>

<p><strong>传输方式</strong>：<code>stdio</code>（本地子进程）或 <code>HTTP+SSE</code>（远程、可流式）。<br><strong>发现机制</strong>：客户端在连接初始化时调用 <code>tools/list</code>、<code>resources/list</code>、<code>prompts/list</code>。<br><strong>工具注解</strong>：<code>readOnlyHint</code>、<code>destructiveHint</code>、<code>idempotentHint</code>、<code>openWorldHint</code>。</p>

## 29.18　A2A 协议速查 {#a2a-protocol-quick-reference}

<span id="table-29-inline-13"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>概念</strong></th><th scope="col"><strong>描述</strong></th></tr></thead>
<tbody><tr><td>Agent Card</td><td>位于 <code>/.well-known/agent.json</code> 的 JSON — 名称、技能、支持的内容类型</td></tr>
<tr><td>Task</td><td>工作单元：<code>id</code>、<code>status</code>、<code>artifacts</code>。生命周期：submitted <span class="arithmatex">\(\to\)</span> working <span class="arithmatex">\(\to\)</span> completed/failed</td></tr>
<tr><td>Message</td><td>任务内的通信单元（角色：user/agent；parts：text/file/data）</td></tr>
<tr><td>Artifact</td><td>Agent 产出的输出（结构化数据、文件、生成内容）</td></tr>
<tr><td>Push Notifications</td><td>长任务的 webhook 更新（通过 <code>tasks/pushNotification/set</code>）</td></tr></tbody>
</table>
</div>

<p><strong>关键端点</strong>：<code>tasks/send</code>（创建/更新）、<code>tasks/get</code>（轮询状态）、<code>tasks/sendSubscribe</code>（SSE 流）。</p>

## 29.19　智能体框架对比 {#agent-framework-comparison}

<span id="table-29-inline-14"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>框架</strong></th><th scope="col"><strong>编排方式</strong></th><th scope="col"><strong>多 Agent</strong></th><th scope="col"><strong>最适用于</strong></th></tr></thead>
<tbody><tr><td>LangGraph</td><td>显式状态图</td><td>条件路由</td><td>生产环境：持久化、HITL、精细控制</td></tr>
<tr><td>OpenAI Agents SDK</td><td>声明式 handoff</td><td>基于 handoff</td><td>简单易用：护栏、追踪、快速上手</td></tr>
<tr><td>AutoGen (AG2)</td><td>对话驱动</td><td>GroupChat</td><td>原型开发：代码执行、研究场景</td></tr>
<tr><td>CrewAI</td><td>基于角色的团队</td><td>顺序 / 并行</td><td>低代码：快速演示、简单流水线</td></tr>
<tr><td>Google ADK</td><td>会话 + 事件</td><td>原生 A2A</td><td>企业级：artifact 管理、多模态</td></tr></tbody>
</table>
</div>

## 29.20　智能体 RL 公式 {#agentic-rl-formulas}



<div class="arithmatex">\[\begin{align*}
\text{Trajectory GRPO:}&amp;\quad \hat{A}_i = (R(\tau_i) - \mu_G)/\sigma_G, \quad R(\tau_i) = \sum_{t} r_t^{(\tau_i)} \tag{29.23}\\

\text{Agent reward:}&amp;\quad R = w_1 R_\text{task} + w_2 R_\text{efficiency} + w_3 R_\text{safety}, \quad R_\text{eff} = \max(0, 1 - \text{steps}/N_\text{max}) \tag{29.24}\\

\text{Masking:}&amp;\quad \mathcal{L} = \sum_{t \in \text{agent tokens}} \min(r_t \hat{A}_t,\; \text{clip}(r_t) \hat{A}_t) \quad \text{(mask env outputs)} \tag{29.25}\\

\text{Pass@}k:&amp;\quad 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}, \quad n = \text{total samples},\; c = \text{correct}
\tag{29.26}
\end{align*}\]</div>

## 29.21　智能体安全检查清单 {#agent-security-checklist}

<span id="table-29-inline-15"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>威胁</strong></th><th scope="col"><strong>层级</strong></th><th scope="col"><strong>缓解措施</strong></th></tr></thead>
<tbody><tr><td>Prompt 注入（直接）</td><td>输入</td><td>输入校验、指令层级、分隔符</td></tr>
<tr><td>Prompt 注入（间接）</td><td>工具输出</td><td>将工具输出视为不可信；勿遵循检索文档中的指令</td></tr>
<tr><td>工具滥用</td><td>执行</td><td>最小权限；<code>destructiveHint</code> 关卡；沙箱化</td></tr>
<tr><td>数据外泄</td><td>输出</td><td>输出过滤；限制工具仅访问白名单域</td></tr>
<tr><td>过度自主</td><td>架构</td><td>最大迭代次数；成本预算；人工审批关卡</td></tr>
<tr><td>混淆代理（Confused Deputy）</td><td>多 Agent</td><td>校验任务来源；基于能力的访问控制</td></tr></tbody>
</table>
</div>

## 29.22　智能体评估指标 {#agent-evaluation-metrics}

<span id="table-29-inline-16"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>指标</strong></th><th scope="col"><strong>公式 / 定义</strong></th><th scope="col"><strong>目标</strong></th></tr></thead>
<tbody><tr><td>任务成功率（TSR）</td><td>正确完成数 / 总任务数</td><td><span class="arithmatex">\(&gt;85\%\)</span>（生产）</td></tr>
<tr><td>完成步数</td><td>单次成功任务的平均动作数</td><td>越低越高效</td></tr>
<tr><td>单任务成本</td><td>总 token 数 <span class="arithmatex">\(\times\)</span> 单价</td><td>视预算而定</td></tr>
<tr><td>延迟（TTFC）</td><td>从请求到首条有效输出</td><td>交互场景 <span class="arithmatex">\(&lt;5\)</span> 秒</td></tr>
<tr><td>工具调用准确率</td><td>正确工具选择 / 总调用</td><td><span class="arithmatex">\(&gt;90\%\)</span></td></tr>
<tr><td>恢复率</td><td>重试成功 / 初次失败</td><td><span class="arithmatex">\(&gt;60\%\)</span></td></tr>
<tr><td>人工升级率</td><td>需要人工的任务 / 总任务</td><td><span class="arithmatex">\(&lt;15\%\)</span></td></tr></tbody>
</table>
</div>

## 29.23　关键智能体基准 {#key-agentic-benchmarks}

<span id="table-29-inline-17"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>基准</strong></th><th scope="col"><strong>领域</strong></th><th scope="col"><strong>指标</strong></th><th scope="col"><strong>SOTA (2025)</strong></th></tr></thead>
<tbody><tr><td>SWE-bench Verified</td><td>软件工程</td><td>解决 issue 百分比</td><td><span class="arithmatex">\(\sim\)</span>70%</td></tr>
<tr><td>WebArena</td><td>Web 浏览</td><td>任务成功率</td><td><span class="arithmatex">\(\sim\)</span>40%</td></tr>
<tr><td>OSWorld</td><td>桌面计算机使用</td><td>任务成功率</td><td><span class="arithmatex">\(\sim\)</span>25%</td></tr>
<tr><td>GAIA</td><td>通用 AI 助手</td><td>完全匹配准确率</td><td><span class="arithmatex">\(\sim\)</span>75%（L1）</td></tr>
<tr><td>Tau-bench</td><td>工具使用可靠性</td><td>通过率（5 次试验）</td><td><span class="arithmatex">\(\sim\)</span>65%</td></tr>
<tr><td>HumanEval / MBPP</td><td>代码生成</td><td>Pass@1</td><td><span class="arithmatex">\(&gt;95\%\)</span></td></tr></tbody>
</table>
</div>
