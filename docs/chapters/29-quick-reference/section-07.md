# 29.7　TRL API 速查 {#trl-api-quick-reference}

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
