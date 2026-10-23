# 5.5　Rollout Buffer 与 Rollout {#rollout-buffer-and-rollouts}

<p>在 PPO 中，数据管理依赖一个被称为 <strong>Rollout Buffer</strong> 的专用短期存储系统。与无限期地在 replay buffer 中保存经验的离策略算法（如 DQN）不同，PPO 需要一种短暂存在的结构，以满足其在策略的数学约束。</p>

## 5.5.1　什么是 Rollout？ {#what-is-a-rollout}

<p><strong>Rollout</strong>（轨迹）是 Agent 在环境中执行当前 policy 所产生的一段交互序列：</p>

<ul>
<li><p><strong>过程</strong>：Agent 观察 state、选择 action、获得 reward，然后转移到下一个 state。重复进行固定步数，或直到 episode 结束。</p></li>
<li><p><strong>在大语言模型/RLHF 中</strong>：一次 rollout 即从数据集中取出一条 prompt，让语言模型一个 token 接一个 token 地生成完整序列，直到遇到文本结束标记。每个 token 就是一个“step”。</p></li>
</ul>

## 5.5.2　Rollout Buffer {#the-rollout-buffer}

<p>Rollout buffer 临时存储 rollout 阶段收集到的全部数据。对每个生成的 token/step，它记录：</p>



<div class="arithmatex">\[\boxed{\mathcal{B} = \left\{ \left(s_t,\; a_t,\; \log\pi_{\theta_{\text{old}}}(a_t|s_t),\; r_t,\; V(s_t)\right) \right\}_{t=1}^{T}}
\tag{5.13}\]</div>

<ul>
<li><p><span class="arithmatex">\(s_t, a_t, r_t\)</span>：步 <span class="arithmatex">\(t\)</span> 的 state、所采取的 action 与 reward。</p></li>
<li><p><span class="arithmatex">\(\log\pi_{\theta_{\text{old}}}(a_t|s_t)\)</span>：在生成该 action 的那一个 policy 下取该 action 的对数概率（计算 ratio 时需要）。</p></li>
<li><p><span class="arithmatex">\(V(s_t)\)</span>：Value function 给出的基线预测（计算 GAE advantage 时需要）。</p></li>
</ul>

## 5.5.3　Rollout Buffer 的生命周期 {#the-rollout-buffer-lifecycle}

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
