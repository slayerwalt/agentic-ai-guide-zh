# 第 5 章 · PPO——近端策略优化 {#ppo-proximal-policy-optimization}

## 5.1　动机与历史 {#motivation-and-history}

<p><strong>问题</strong>：朴素的 policy gradient 更新对步长没有任何约束。仅仅一次倒霉的 batch 就可能把 policy 推入一个生成垃圾文本的区域 <span class="arithmatex">\(\rightarrow\)</span> 垃圾文本获得低 reward <span class="arithmatex">\(\rightarrow\)</span> 下一次 gradient 让情况更糟 <span class="arithmatex">\(\rightarrow\)</span> 不可挽回地崩溃。</p>

<p><strong>解决方案的演进</strong>：</p>

<ol>
<li><p><strong>TRPO</strong> <a class="book-citation" id="cite-schulman2015trust-1" href="../../references.md#ref-schulman2015trust" aria-label="参考文献 317">[317]</a>（2015）：对新旧 policy 之间的 KL 散度施加约束。效果完美，但需要昂贵的二阶优化（Fisher 信息矩阵、共轭梯度）。</p></li>
<li><p><strong>PPO</strong>（2017） <a class="book-citation" id="cite-schulman2017proximal-1" href="../../references.md#ref-schulman2017proximal" aria-label="参考文献 319">[319]</a>：用一个简单的一阶 clipped 目标实现类似的稳定性。实现复杂度低了 10<span class="arithmatex">\(\times\)</span>，效果几乎一样，且能轻松扩展到分布式训练。</p></li>
</ol>

## 5.2　Clipped 目标 {#the-clipped-objective}

<p>PPO 的核心创新是一个 clipped surrogate 目标：它能阻止破坏性的大幅 policy 更新，同时保持实现简单。</p>



<div class="arithmatex">\[\boxed{L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\hat{A}_t\right)\right]}
\tag{5.1}\]</div>

<p>其中 <span class="arithmatex">\(r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}\)</span> 是概率比。</p>

<div class="admonition tip"><p class="admonition-title">Clipping 直觉——关键洞察</p>
<p><span class="arithmatex">\(\min\)</span> 算子构造了一个<strong>悲观下界</strong>：</p>

<ul>
<li><p><strong>好的 action（<span class="arithmatex">\(\hat{A} &gt; 0\)</span>）</strong>：我们希望提升其概率。Surrogate <span class="arithmatex">\(r\hat{A}\)</span> 会随 <span class="arithmatex">\(r\)</span> 增大而增大，但 clip 将收益上限锁在 <span class="arithmatex">\(r = 1 + \epsilon\)</span>。<em>“别因为一个好例子就贪心。”</em></p></li>
<li><p><strong>坏的 action（<span class="arithmatex">\(\hat{A} &lt; 0\)</span>）</strong>：我们希望降低其概率。<span class="arithmatex">\(r\hat{A}\)</span> 会随 <span class="arithmatex">\(r\)</span> 减小而改善，但 clip 将收益上限锁在 <span class="arithmatex">\(r = 1 - \epsilon\)</span>。<em>“别因为一个坏例子就遗忘得太激进。”</em></p></li>
</ul>

<p>净效应：每次更新步中 policy 的变化最多在 <span class="arithmatex">\(\pm\)</span>20% 之内，既能防止灾难性崩溃，也能防止过度自信地走向特化。</p>
</div>

## 5.3　完整的 PPO Loss {#full-ppo-loss}



<div class="arithmatex">\[L = L^{\text{CLIP}} - c_1 \underbrace{(V_\theta(s_t) - V^{\text{target}}_t)^2}_{\text{value loss}} + c_2 \underbrace{H[\pi_\theta(\cdot|s_t)]}_{\text{entropy bonus}}
\tag{5.2}\]</div>

<ul>
<li><p><strong>Value loss</strong>（<span class="arithmatex">\(c_1 = 0.1\)</span>）：训练 critic 去预测 return；同样被 clip 以保持稳定。</p></li>
<li><p><strong>熵奖励</strong>（<span class="arithmatex">\(c_2 = 0.01\)</span>）：防止过早收敛到确定性 policy，对探索至关重要。</p></li>
</ul>

## 5.4　PPO Gradient 与更新规则的推导 {#derivation-of-the-ppo-gradient-and-update-rule}

<p>本节追溯从 RL 目标到 PPO 更新规则的数学路径，说明 clipped surrogate <em>为何</em>有效。</p>

### 5.4.1　第 1 步：RL 目标 {#step-1-the-rl-objective}

<p>目标是在 policy 下最大化期望累计 reward：</p>



<div class="arithmatex">\[J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T r_t\right]
\tag{5.3}\]</div>

### 5.4.2　第 2 步：Policy Gradient 定理 {#step-2-policy-gradient-theorem}

<p><span class="arithmatex">\(J(\theta)\)</span> 关于 policy 参数的 gradient：</p>



<div class="arithmatex">\[\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right]}
\tag{5.4}\]</div>

<p>其中 <span class="arithmatex">\(\hat{A}_t\)</span> 是优势函数（即在 state <span class="arithmatex">\(s_t\)</span> 下 action <span class="arithmatex">\(a_t\)</span> 相对平均 action 的好坏程度）。用 advantage 替代完整 return 是为了降低方差。</p>

### 5.4.3　第 3 步：离策略数据的重要性采样 {#step-3-importance-sampling-for-off-policy-data}

<p>PPO 使用 <span class="arithmatex">\(\pi_{\theta_{\text{old}}}\)</span> 收集数据，却更新 <span class="arithmatex">\(\pi_\theta\)</span>。为修正这种分布失配，需应用重要性采样：</p>



<div class="arithmatex">\[\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right]
\tag{5.5}\]</div>

<p>定义概率比 <span class="arithmatex">\(r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\)</span>。利用恒等式 <span class="arithmatex">\(\nabla_\theta \log f = \frac{\nabla_\theta f}{f}\)</span>，可得：</p>



<div class="arithmatex">\[\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[\nabla_\theta\, r_t(\theta) \cdot \hat{A}_t\right]
\tag{5.6}\]</div>

<p>这意味着最大化 <strong>surrogate 目标</strong>：</p>



<div class="arithmatex">\[L^{\text{CPI}}(\theta) = \mathbb{E}_t\left[r_t(\theta) \cdot \hat{A}_t\right]
\tag{5.7}\]</div>

### 5.4.4　第 4 步：无约束 Surrogate 的问题 {#step-4-the-problem-with-unconstrained-surrogates}

<p><span class="arithmatex">\(L^{\text{CPI}}\)</span> 是一个合法的目标，但若无约束，单次 gradient 步就可能让 <span class="arithmatex">\(r_t(\theta)\)</span> 远离 1.0，导致：</p>

<ul>
<li><p>重要性权重变得极端 <span class="arithmatex">\(\rightarrow\)</span> 方差升高</p></li>
<li><p>Policy 进入未经检验的区域 <span class="arithmatex">\(\rightarrow\)</span> reward 模型给出不可靠的分数</p></li>
<li><p>灾难性崩溃：policy 生成垃圾，且无法恢复</p></li>
</ul>

<p><strong>TRPO 方案</strong>：约束 <span class="arithmatex">\(D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \leq \delta\)</span>。需要二阶方法（成本高）。</p>

### 5.4.5　第 5 步：PPO 的 Clipped Surrogate（一阶近似） {#step-5-ppos-clipped-surrogate-first-order-approximation}

<p>PPO 用一个 <strong>clipped 目标</strong>取代硬性 KL 约束，仅使用一阶 gradient 即可达到类似行为：</p>



<div class="arithmatex">\[\boxed{L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\hat{A}_t\right)\right]}
\tag{5.8}\]</div>

<p><strong>Gradient 的推导</strong>：</p>

<p>令 <span class="arithmatex">\(L_t = \min(r_t \hat{A}_t,\; \bar{r}_t \hat{A}_t)\)</span>，其中 <span class="arithmatex">\(\bar{r}_t = \text{clip}(r_t, 1{-}\epsilon, 1{+}\epsilon)\)</span>。</p>



<div class="arithmatex">\[\nabla_\theta L_t = \begin{cases}
\nabla_\theta r_t(\theta) \cdot \hat{A}_t &amp; \text{if } r_t \hat{A}_t &lt; \bar{r}_t \hat{A}_t \text{ (unclipped term is smaller)} \\
0 &amp; \text{if } r_t \hat{A}_t \geq \bar{r}_t \hat{A}_t \text{ (clipped term is smaller, gradient = 0)}
\end{cases}
\tag{5.9}\]</div>

<p>展开各条件：</p>

<ul>
<li><p><strong>当 <span class="arithmatex">\(\hat{A}_t &gt; 0\)</span> 且 <span class="arithmatex">\(r_t &lt; 1+\epsilon\)</span></strong>：Gradient 正常传播——policy 被鼓励提升 <span class="arithmatex">\(\pi_\theta(a_t|s_t)\)</span>。</p></li>
<li><p><strong>当 <span class="arithmatex">\(\hat{A}_t &gt; 0\)</span> 且 <span class="arithmatex">\(r_t \geq 1+\epsilon\)</span></strong>：Gradient 为<strong>零</strong>——policy 提升已足够，停止继续推。</p></li>
<li><p><strong>当 <span class="arithmatex">\(\hat{A}_t &lt; 0\)</span> 且 <span class="arithmatex">\(r_t &gt; 1-\epsilon\)</span></strong>：Gradient 正常传播——policy 被鼓励降低 <span class="arithmatex">\(\pi_\theta(a_t|s_t)\)</span>。</p></li>
<li><p><strong>当 <span class="arithmatex">\(\hat{A}_t &lt; 0\)</span> 且 <span class="arithmatex">\(r_t \leq 1-\epsilon\)</span></strong>：Gradient 为<strong>零</strong>——policy 降低已足够，停止继续推。</p></li>
</ul>

### 5.4.6　第 6 步：完整的 PPO 更新规则 {#step-6-the-complete-ppo-update-rule}

<p>将 clipped policy loss、value loss 与熵奖励组合起来：</p>



<div class="arithmatex">\[\boxed{\theta_{k+1} = \theta_k + \alpha \cdot \nabla_\theta \left[L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 H[\pi_\theta]\right]}
\tag{5.10}\]</div>

<p>其中：</p>



<div class="arithmatex">\[\begin{align*}
L^{\text{VF}}(\theta) &amp;= \left(V_\theta(s_t) - V_t^{\text{target}}\right)^2 &amp; &amp;\text{(value function regression loss)} \tag{5.11}\\

H[\pi_\theta] &amp;= -\sum_a \pi_\theta(a|s_t)\log\pi_\theta(a|s_t) &amp; &amp;\text{(entropy of the policy)}
\tag{5.12}
\end{align*}\]</div>

<div class="admonition tip"><p class="admonition-title">小结：它为何有效</p>
<ol>
<li><p><strong>Policy gradient 定理</strong>给出了改进 policy 的方向。</p></li>
<li><p><strong>重要性采样</strong>让我们能在多个 epoch 中复用来自 <span class="arithmatex">\(\pi_{\theta_{\text{old}}}\)</span> 的数据。</p></li>
<li><p><strong>Clipping</strong> 防止重要性权重变得极端，保持更新安全。</p></li>
<li><p><strong><span class="arithmatex">\(\min\)</span> 算子</strong>保证我们始终在 (clipped, unclipped) 中取更保守的一项——对改进设置悲观下界。</p></li>
<li><p><strong>结果</strong>：以概率 1 实现单调改进，且只用一阶 gradient。无需 Hessian、共轭梯度或线搜索。</p></li>
</ol>
</div>

## 5.5　Rollout Buffer 与 Rollout {#rollout-buffer-and-rollouts}

<p>在 PPO 中，数据管理依赖一个被称为 <strong>Rollout Buffer</strong> 的专用短期存储系统。与无限期地在 replay buffer 中保存经验的离策略算法（如 DQN）不同，PPO 需要一种短暂存在的结构，以满足其在策略的数学约束。</p>

### 5.5.1　什么是 Rollout？ {#what-is-a-rollout}

<p><strong>Rollout</strong>（轨迹）是 Agent 在环境中执行当前 policy 所产生的一段交互序列：</p>

<ul>
<li><p><strong>过程</strong>：Agent 观察 state、选择 action、获得 reward，然后转移到下一个 state。重复进行固定步数，或直到 episode 结束。</p></li>
<li><p><strong>在大语言模型/RLHF 中</strong>：一次 rollout 即从数据集中取出一条 prompt，让语言模型一个 token 接一个 token 地生成完整序列，直到遇到文本结束标记。每个 token 就是一个“step”。</p></li>
</ul>

### 5.5.2　Rollout Buffer {#the-rollout-buffer}

<p>Rollout buffer 临时存储 rollout 阶段收集到的全部数据。对每个生成的 token/step，它记录：</p>



<div class="arithmatex">\[\boxed{\mathcal{B} = \left\{ \left(s_t,\; a_t,\; \log\pi_{\theta_{\text{old}}}(a_t|s_t),\; r_t,\; V(s_t)\right) \right\}_{t=1}^{T}}
\tag{5.13}\]</div>

<ul>
<li><p><span class="arithmatex">\(s_t, a_t, r_t\)</span>：步 <span class="arithmatex">\(t\)</span> 的 state、所采取的 action 与 reward。</p></li>
<li><p><span class="arithmatex">\(\log\pi_{\theta_{\text{old}}}(a_t|s_t)\)</span>：在生成该 action 的那一个 policy 下取该 action 的对数概率（计算 ratio 时需要）。</p></li>
<li><p><span class="arithmatex">\(V(s_t)\)</span>：Value function 给出的基线预测（计算 GAE advantage 时需要）。</p></li>
</ul>

### 5.5.3　Rollout Buffer 的生命周期 {#the-rollout-buffer-lifecycle}

<p>该 buffer 按严格的三阶段时钟周期运行：</p>

<ol>
<li><p><strong>收集</strong>：当前 policy 与环境交互，用新鲜轨迹填满 buffer（对一个 70B 模型，batch=128，max_tokens=512：单次 rollout 可达 65K 个 token 级 transition）。</p></li>
<li><p><strong>训练</strong>：对各轨迹计算 GAE advantage。使用 clipped 目标在 mini-batch 上跑 <span class="arithmatex">\(K\)</span> 个 epoch（通常 3–10）的 gradient 下降，更新 policy 权重。</p></li>
<li><p><strong>清空</strong>：整个 buffer 被<strong>彻底清空</strong>。由于 PPO 是 on-policy 的，旧 policy 产生的数据无法安全地复用于下一轮更新——比率 <span class="arithmatex">\(r_t(\theta)\)</span> 会变得过期，clipping 保证也会失效。</p></li>
</ol>

<div class="admonition warning"><p class="admonition-title">Rollout Buffer 与 Replay Buffer 的区别</p>
<p><strong>Replay Buffer</strong>（DQN、SAC）：Off-policy。无限期地存储数百万条 transition。随机采样。数据跨多次更新复用。</p>

<p><strong>Rollout Buffer</strong>（PPO、GRPO）：On-policy。存储一个 batch 的轨迹。使用若干 epoch 后被完全丢弃。每个周期都需要新鲜数据。</p>

<p>这就是为什么 PPO 需要持续生成——每次更新后 buffer 被清空，需要新的 rollout。这也让生成瓶颈（占 wall-clock 时间 60–70%）尤为痛苦。</p>
</div>

<div class="admonition tip"><p class="admonition-title">RLHF 场景中的 vLLM</p>
<p>在 RLHF 训练中，vLLM 被用于<strong>生成阶段</strong>（占 wall-clock 时间 60–70%）。Policy 模型生成 rollout，随后由 reward 模型打分。其主要优势：</p>

<ul>
<li><p><strong>批量生成</strong>：跨多条 prompt 并行生成 256+ 条回答。</p></li>
<li><p><strong>显存效率</strong>：可容纳更多并发生成 <span class="arithmatex">\(\rightarrow\)</span> 在生成瓶颈期 GPU 利用率更高。</p></li>
<li><p><strong>前缀共享</strong>：当每条 prompt 生成 <span class="arithmatex">\(N=8\)</span> 条回答时（如 GRPO），prompt 的 KV 只算一次并被 8 条共享——无冗余 prefill。</p></li>
<li><p><strong>集成</strong>：OpenRLHF、TRL 等框架将 vLLM 作为生成后端，把生成 worker（vLLM）与训练 worker（DeepSpeed/FSDP）分离。</p></li>
</ul>
</div>

## 5.6　用于 RLHF 的 PPO：完整循环 {#ppo-for-rlhf-the-full-loop}

<figure id="figure-5-1"><a href="../../assets/figures/fig_025_fig25.png" target="_blank" rel="noopener" aria-label="查看图 5.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_025_fig25.png" alt="" loading="lazy"></a><figcaption><span class="figure-number">图 5.1　</span></figcaption></figure>

<div class="admonition example"><p class="admonition-title">70B Chat 模型的具体 PPO 步骤</p>
<p><strong>设置</strong>：128 条 prompt 一个 batch，policy 为 Llama-3-70B，最大生成 512 个 token。</p>

<p><strong>第 1 步——生成</strong>：采样 128 条回答（temperature=0.7，top-p=0.9）。耗时占 60%。</p>

<p><strong>第 2 步——打分</strong>：Reward 模型对每个 (prompt, response) 对打分。取值范围：0.2–0.95。</p>

<p><strong>第 3 步——KL</strong>：计算逐 token 的 KL：<span class="arithmatex">\(\text{KL}_t = \log\pi_\theta(y_t|y_{&lt;t}) - \log\pi_\text{ref}(y_t|y_{&lt;t})\)</span>。跨 token 的均值 KL：通常 3–8。</p>

<p><strong>第 4 步——最终 reward</strong>：<span class="arithmatex">\(R = r_\text{RM} - 0.05 \times \text{mean\_KL}\)</span>（只在最后一个 token 给）。</p>

<p><strong>第 5 步——GAE</strong>：使用 value head 的预测，为每个 token 位置计算 <span class="arithmatex">\(\hat{A}_t\)</span>。对 advantage 做 whitening（零均值、单位方差）。</p>

<p><strong>第 6 步——更新</strong>：在大小为 16 的 mini-batch 上跑 4 个 epoch 的 SGD。Clip 比率 <span class="arithmatex">\(\epsilon = 0.2\)</span>，梯度范数 clip 至 1.0。</p>

<p><strong>结果</strong>：Policy 每步约提升 0.005 的胜率。10K 步后：相对于 SFT 绝对提升 5–10%。</p>
</div>

<div class="admonition warning"><p class="admonition-title">大语言模型 RL 中的 Tokenization 陷阱</p>
<p>在计算逐 token KL 惩罚与 advantage 时，请记住：tokenization 决定了什么算一“步”。单个概念性 action（例如输出“2024”）依据 tokenizer 的不同可能跨越 1–4 个 token。这带来若干微妙问题：</p>

<ul>
<li><p><strong>KL 记账</strong>：相同的语义内容若被切成不同数量的 token，逐 token KL 加总会得出不同总额（例如，被切成更多子词的稀有词会获得更高的总 KL 惩罚）。</p></li>
<li><p><strong>Credit 分配</strong>：GAE 按 token 位置分配 advantage——但语义“决策”常跨越多个 token。模型其实只在词首 token 真正“决定”；后续子词 token 基本是确定性的。</p></li>
<li><p><strong>Reward 放置</strong>：只在最后一个 token 给 reward 意味着所有先前 token 必须通过 GAE 向后回传 credit——越长的回答信号被稀释得越厉害。</p></li>
</ul>

<p><strong>缓解措施</strong>：一些系统按序列长度对 KL 做归一化，使用词级 reward shaping，或在语义边界（而非最末 token）处施加 reward。</p>
</div>

## 5.7　细节机制：Logits 与 Policy 更新 {#detailed-mechanics-logits-and-policy-updates}

<figure id="figure-5-2"><a href="../../assets/figures/fig_026_fig26.png" target="_blank" rel="noopener" aria-label="查看图 5.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_026_fig26.png" alt="PPO 端到端流程：从 prompt batch 出发，经过生成、reward 打分、KL 计算、advantage 估计，到 clipped policy 更新。反馈环显示更新后的 policy 又被用于下一次生成。" loading="lazy"></a><figcaption><span class="figure-number">图 5.2　</span>PPO 端到端流程：从 prompt batch 出发，经过生成、reward 打分、KL 计算、advantage 估计，到 clipped policy 更新。反馈环显示更新后的 policy 又被用于下一次生成。</figcaption></figure>

<p>PPO 在内存中维护两份不同的参数状态，它们共享同一神经网络拓扑，但在优化过程中持有不同的权重值：</p>

<div class="admonition note"><p class="admonition-title">核心架构：两个网络</p>
<ol>
<li><p><strong>Policy 网络（<span class="arithmatex">\(\pi_\theta\)</span>）：</strong>由权重 <span class="arithmatex">\(\theta\)</span> 参数化的、处于活动状态的实时网络。优化过程中通过反向传播持续更新。</p></li>
<li><p><strong>旧 Policy 网络（<span class="arithmatex">\(\pi_{\theta_{\text{old}}}\)</span>）：</strong>由权重 <span class="arithmatex">\(\theta_{\text{old}}\)</span> 参数化的冻结快照。在单个优化周期内充当静态锚，防止 policy 漂移过快。</p></li>
</ol>
</div>

### 5.7.1　阶段 1：Rollout（数据收集） {#phase-1-rollout-data-collection}

<p>在数据收集期间，Agent 与环境交互 <span class="arithmatex">\(T\)</span> 步。在每个时间步 <span class="arithmatex">\(t\)</span>：</p>

<ol>
<li><p>环境给出当前 state/观测 <span class="arithmatex">\(s_t\)</span>（对大语言模型而言：prompt + 到目前为止已生成的 token）。</p></li>
<li><p><span class="arithmatex">\(s_t\)</span> 被送入当前网络快照（<span class="arithmatex">\(\theta_{\text{old}}\)</span>）。</p></li>
<li><p>网络输出原始未归一化的值——<strong>logits</strong> <span class="arithmatex">\(z_{\text{old}}\)</span>——一个长度为 <span class="arithmatex">\(|V|\)</span>（词表大小 32K–128K）的向量。</p></li>
<li><p>通过 Softmax 计算概率：</p>



<div class="arithmatex">\[\boxed{P(a \mid s_t) = \text{Softmax}(z_{\text{old}}) = \frac{\exp(z_{\text{old}, a})}{\sum_{j=1}^{|V|} \exp(z_{\text{old}, j})}}
\tag{5.14}\]</div></li>
<li><p>从 <span class="arithmatex">\(P(a \mid s_t)\)</span> 中采样一个 action <span class="arithmatex">\(a_t\)</span>（下一个 token），并将 transition 元组 <span class="arithmatex">\(\langle s_t, a_t, r_t, s_{t+1} \rangle\)</span> 连同 <span class="arithmatex">\(\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\)</span> 一起存入 rollout buffer。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">为何要存对数概率？</p>
<p>在 rollout 阶段把 <span class="arithmatex">\(\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\)</span> 作为标量保存，可以避免在优化期间重新运行冻结的网络。每个 mini-batch 因此可省下一次完整 forward pass——对 70B 模型而言意义重大。</p>
</div>

### 5.7.2　阶段 2：优化循环（Mini-Batch 更新） {#phase-2-optimization-loop-mini-batch-updates}

<p>一旦 rollout buffer 填满，PPO 就在 mini-batch 上跑 <span class="arithmatex">\(K\)</span> 个 epoch（通常 3–10）。在每个 gradient 步中，使用所存 state <span class="arithmatex">\(s_t\)</span> 为两个 policy 同时生成 logits：</p>

<p><strong>旧 Policy 评估</strong>（冻结）：</p>



<div class="arithmatex">\[z_{\text{old}} = f(s_t; \theta_{\text{old}}) \quad \longrightarrow \quad \log \pi_{\theta_{\text{old}}}(a_t \mid s_t) = \text{LogSoftmax}(z_{\text{old}})[a_t]
\tag{5.15}\]</div>

<p><em>实现捷径：复用 rollout 时保存的标量，而非重新计算。</em></p>

<p><strong>实时 Policy 评估</strong>（更新中）：</p>



<div class="arithmatex">\[z_{\text{new}} = f(s_t; \theta) \quad \longrightarrow \quad \log \pi_\theta(a_t \mid s_t) = \text{LogSoftmax}(z_{\text{new}})[a_t]
\tag{5.16}\]</div>

<p>由于每次 mini-batch gradient 步后 <span class="arithmatex">\(\theta\)</span> 都会更新，<span class="arithmatex">\(z_{\text{new}}\)</span> 会在整个优化循环中持续变化，而 <span class="arithmatex">\(z_{\text{old}}\)</span> 则始终保持不变。</p>

### 5.7.3　从 Logits 到概率比 {#from-logits-to-probability-ratio}

<p>PPO 的核心比率衡量某个 action 在新 policy 下相对旧 policy 的概率变化：</p>



<div class="arithmatex">\[\boxed{r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}}
\tag{5.17}\]</div>

<p>为了避免直接除原始概率带来的数值下溢/上溢灾难，计算在<strong>对数空间</strong>中进行：</p>



<div class="arithmatex">\[\begin{align*}
\log \pi_\theta(a_t \mid s_t) &amp;= \text{LogSoftmax}(z_{\text{new}})[a_t] \tag{5.18}\\

\log \pi_{\theta_{\text{old}}}(a_t \mid s_t) &amp;= \text{LogSoftmax}(z_{\text{old}})[a_t]
\tag{5.19}
\end{align*}\]</div>

<p>比率由差值的指数恢复：</p>



<div class="arithmatex">\[\boxed{r_t(\theta) = \exp\!\left(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\right)}
\tag{5.20}\]</div>

<p>该比率随后被代入 PPO 的 clipping 目标：</p>



<div class="arithmatex">\[\boxed{\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\!\left(r_t(\theta)\hat{A}_t, \;\text{clip}(r_t(\theta),\, 1{-}\epsilon,\, 1{+}\epsilon)\,\hat{A}_t\right) \right]}
\tag{5.21}\]</div>

<div class="admonition tip"><p class="admonition-title">Clipping 的工作方式</p>
<ul>
<li><p>若 <span class="arithmatex">\(\hat{A}_t &gt; 0\)</span>（好 action）：比率被 clip 至 <span class="arithmatex">\(1+\epsilon\)</span>——不会对好 action 过度利用。</p></li>
<li><p>若 <span class="arithmatex">\(\hat{A}_t &lt; 0\)</span>（坏 action）：比率被 clip 至 <span class="arithmatex">\(1-\epsilon\)</span>——不会对坏 action 过度惩罚。</p></li>
<li><p><span class="arithmatex">\(\min(\cdot)\)</span> 保证我们始终取更保守的估计。</p></li>
</ul>

<p>结果：在一个信任域内实现单调改进——不会出现灾难性崩溃。</p>
</div>

### 5.7.4　PPO 权重生命周期 {#the-ppo-weight-lifecycle}

<p class="table-caption" id="table-5-1"><span class="table-number">表 5.1　</span><span class="arithmatex">\(\theta\)</span> 与 <span class="arithmatex">\(\theta_{\text{old}}\)</span> 在 PPO 训练各阶段的演化。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>阶段</strong></th><th scope="col"><strong>实时 <span class="arithmatex">\(\theta\)</span></strong></th><th scope="col"><strong>旧 <span class="arithmatex">\(\theta_{\text{old}}\)</span></strong></th><th scope="col"><strong>比率 <span class="arithmatex">\(r_t(\theta)\)</span></strong></th></tr></thead>
<tbody><tr><td>1. Rollout 起始</td><td>活动副本</td><td>同一活动副本</td><td>恒为 <span class="arithmatex">\(1.0\)</span>（按定义）</td></tr>
<tr><td>2. Batch 第 1 步</td><td>计算 gradient</td><td>冻结</td><td><span class="arithmatex">\(1.0\)</span>（初始步）</td></tr>
<tr><td>3. Batch 第 <span class="arithmatex">\(N\)</span> 步</td><td>正在修改（<span class="arithmatex">\(\theta \neq \theta_{\text{old}}\)</span>）</td><td>冻结</td><td>偏离 <span class="arithmatex">\(1.0\)</span>（如 <span class="arithmatex">\(1.06\)</span>、<span class="arithmatex">\(0.94\)</span>）</td></tr>
<tr><td>4. Clipping 生效</td><td>受 <span class="arithmatex">\(\epsilon\)</span> 限制</td><td>冻结</td><td>被锁在边界（<span class="arithmatex">\(1 \pm \epsilon\)</span>）</td></tr>
<tr><td>5. 优化结束</td><td>高度优化后</td><td>被丢弃</td><td>不适用</td></tr>
<tr><td>6. 下一周期</td><td><span class="arithmatex">\(\theta \rightarrow \theta_{\text{old}}\)</span></td><td>接收最新的 <span class="arithmatex">\(\theta\)</span></td><td>重置回 <span class="arithmatex">\(1.0\)</span></td></tr></tbody>
</table>
</div>

### 5.7.5　连续 Action 空间扩展 {#continuous-action-spaces-extension}

<p>对于连续 action 空间（在大语言模型中不常见，但对机器人 RL 很重要），网络输出的是分布参数而非离散 logits：</p>

<ul>
<li><p>预测的均值向量 <span class="arithmatex">\(\mu\)</span></p></li>
<li><p>预测的标准差向量 <span class="arithmatex">\(\sigma\)</span></p></li>
</ul>

<p>对数概率由高斯对数 PDF 计算：</p>



<div class="arithmatex">\[\boxed{\log \pi(a_t \mid s_t) = -\frac{1}{2}\left(\frac{a_t - \mu}{\sigma}\right)^{\!2} - \log(\sigma) - \frac{1}{2}\log(2\pi)}
\tag{5.22}\]</div>

<p>比率 <span class="arithmatex">\(r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})\)</span> 的计算方式完全相同，并被代入同一个 clipping 目标。</p>

## 5.8　TRL 实现 {#trl-implementation}

<p>HuggingFace 的 TRL 库 <a class="book-citation" id="cite-vonwerra2022trl-1" href="../../references.md#ref-vonwerra2022trl" aria-label="参考文献 376">[376]</a> 提供了面向大语言模型的所有主流 RL 方法的生产级实现。</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">PPOConfig</span><span class="p">,</span> <span class="n">PPOTrainer</span><span class="p">,</span> <span class="n">AutoModelForCausalLMWithValueHead</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoTokenizer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">peft</span><span class="w"> </span><span class="kn">import</span> <span class="n">LoraConfig</span>

<span class="c1"># 模型初始化</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">AutoModelForCausalLMWithValueHead</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span>
    <span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">,</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span> <span class="n">device_map</span><span class="o">=</span><span class="s2">"auto"</span><span class="p">,</span>
    <span class="n">peft_config</span><span class="o">=</span><span class="n">LoraConfig</span><span class="p">(</span><span class="n">r</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span> <span class="n">lora_alpha</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span> <span class="n">target_modules</span><span class="o">=</span><span class="p">[</span><span class="s2">"q_proj"</span><span class="p">,</span><span class="s2">"v_proj"</span><span class="p">,</span><span class="s2">"k_proj"</span><span class="p">,</span><span class="s2">"o_proj"</span><span class="p">])</span>
<span class="p">)</span>
<span class="n">tokenizer</span> <span class="o">=</span> <span class="n">AutoTokenizer</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">)</span>

<span class="c1"># PPO 配置，包含全部关键超参</span>
<span class="n">ppo_config</span> <span class="o">=</span> <span class="n">PPOConfig</span><span class="p">(</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">1.5e-6</span><span class="p">,</span>        <span class="c1"># 低学习率，利于稳定</span>
    <span class="n">batch_size</span><span class="o">=</span><span class="mi">128</span><span class="p">,</span>              <span class="c1"># 每步的 prompt 数</span>
    <span class="n">mini_batch_size</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span>          <span class="c1"># 梯度累计的单位</span>
    <span class="n">ppo_epochs</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>                <span class="c1"># 每个 batch 的 epoch 数（复用数据）</span>
    <span class="n">gamma</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>                   <span class="c1"># 不做折扣（单轮）</span>
    <span class="n">lam</span><span class="o">=</span><span class="mf">0.95</span><span class="p">,</span>                    <span class="c1"># GAE lambda</span>
    <span class="n">cliprange</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span>               <span class="c1"># PPO epsilon</span>
    <span class="n">cliprange_value</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span>         <span class="c1"># value function 的 clip 范围</span>
    <span class="n">vf_coef</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>                 <span class="c1"># value loss 系数</span>
    <span class="n">init_kl_coef</span><span class="o">=</span><span class="mf">0.05</span><span class="p">,</span>           <span class="c1"># 初始 KL 惩罚</span>
    <span class="n">target_kl</span><span class="o">=</span><span class="mf">6.0</span><span class="p">,</span>               <span class="c1"># 自适应 KL 目标</span>
    <span class="n">whiten_rewards</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>         <span class="c1"># 对 advantage 做归一化</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">max_grad_norm</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">ppo_trainer</span> <span class="o">=</span> <span class="n">PPOTrainer</span><span class="p">(</span><span class="n">config</span><span class="o">=</span><span class="n">ppo_config</span><span class="p">,</span> <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span> <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
    <span class="n">dataset</span><span class="o">=</span><span class="n">prompt_dataset</span><span class="p">,</span> <span class="n">data_collator</span><span class="o">=</span><span class="n">collator</span><span class="p">)</span>

<span class="c1"># 训练循环</span>
<span class="k">for</span> <span class="n">batch</span> <span class="ow">in</span> <span class="n">ppo_trainer</span><span class="o">.</span><span class="n">dataloader</span><span class="p">:</span>
    <span class="c1"># 1. 生成回答</span>
    <span class="n">query_tensors</span> <span class="o">=</span> <span class="n">batch</span><span class="p">[</span><span class="s2">"input_ids"</span><span class="p">]</span>
    <span class="n">response_tensors</span> <span class="o">=</span> <span class="n">ppo_trainer</span><span class="o">.</span><span class="n">generate</span><span class="p">(</span>
        <span class="n">query_tensors</span><span class="p">,</span> <span class="n">max_new_tokens</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span> <span class="n">temperature</span><span class="o">=</span><span class="mf">0.7</span><span class="p">,</span> <span class="n">top_p</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span> <span class="n">do_sample</span><span class="o">=</span><span class="kc">True</span>
    <span class="p">)</span>
    <span class="c1"># 2. 用 reward 模型打分</span>
    <span class="n">texts</span> <span class="o">=</span> <span class="p">[</span><span class="n">tokenizer</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">r</span><span class="p">,</span> <span class="n">skip_special_tokens</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">response_tensors</span><span class="p">]</span>
    <span class="n">rewards</span> <span class="o">=</span> <span class="p">[</span><span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">(</span><span class="n">reward_model</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">q</span><span class="p">,</span> <span class="n">r</span><span class="p">))</span> <span class="k">for</span> <span class="n">q</span><span class="p">,</span> <span class="n">r</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">batch</span><span class="p">[</span><span class="s2">"query"</span><span class="p">],</span> <span class="n">texts</span><span class="p">)]</span>
    <span class="c1"># 3. PPO 更新（内部完成 KL、GAE、clipping）</span>
    <span class="n">stats</span> <span class="o">=</span> <span class="n">ppo_trainer</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">query_tensors</span><span class="p">,</span> <span class="n">response_tensors</span><span class="p">,</span> <span class="n">rewards</span><span class="p">)</span>
    <span class="c1"># 监控指标：stats["ppo/mean_scores"]、stats["ppo/policy/approx_kl"]</span>
</code></pre></div>

## 5.9　关键超参 {#critical-hyperparameters}

<span id="table-5-inline-2"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>参数</strong></th><th scope="col"><strong>典型值</strong></th><th scope="col"><strong>设错的影响</strong></th></tr></thead>
<tbody><tr><td><code>cliprange</code></td><td>0.2</td><td>过低：学不到东西。过高：不稳定。</td></tr>
<tr><td><code>init_kl_coef</code></td><td>0.01–0.1</td><td>过低：reward hacking。过高：卡在 SFT 状态。</td></tr>
<tr><td><code>target_kl</code></td><td>4–8</td><td>自适应控制器的目标值；越低越保守。</td></tr>
<tr><td><code>ppo_epochs</code></td><td>4</td><td>过多：在单 batch 上过拟合。过少：浪费生成算力。</td></tr>
<tr><td><code>learning_rate</code></td><td><span class="arithmatex">\(1{-}5 \times 10^{-6}\)</span></td><td>过高：灾难性遗忘（Catastrophic Forgetting）。</td></tr>
<tr><td><code>batch_size</code></td><td>64–256</td><td>越大：梯度更平滑，但生成算力更多。</td></tr>
<tr><td><code>temperature</code></td><td>0.7–1.0</td><td>越低：探索越少。越高：advantage 越嘈杂。</td></tr></tbody>
</table>
</div>
