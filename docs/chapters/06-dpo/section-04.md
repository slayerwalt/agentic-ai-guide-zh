# 6.4　TRL 实现 {#trl-implementation-dup2}

<p>下面给出一个使用 HuggingFace TRL 的最小可运行示例。</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoModelForCausalLM</span><span class="p">,</span> <span class="n">AutoTokenizer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">peft</span><span class="w"> </span><span class="kn">import</span> <span class="n">LoraConfig</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">datasets</span><span class="w"> </span><span class="kn">import</span> <span class="n">load_dataset</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">AutoModelForCausalLM</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">,</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span> <span class="n">attn_implementation</span><span class="o">=</span><span class="s2">"flash_attention_2"</span><span class="p">)</span>
<span class="n">tokenizer</span> <span class="o">=</span> <span class="n">AutoTokenizer</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span><span class="p">)</span>

<span class="c1"># 数据集格式：{"prompt": str, "chosen": str, "rejected": str}</span>
<span class="n">dataset</span> <span class="o">=</span> <span class="n">load_dataset</span><span class="p">(</span><span class="s2">"argilla/ultrafeedback-binarized-preferences"</span><span class="p">)</span>

<span class="n">lora_config</span> <span class="o">=</span> <span class="n">LoraConfig</span><span class="p">(</span><span class="n">r</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span> <span class="n">lora_alpha</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span> <span class="n">lora_dropout</span><span class="o">=</span><span class="mf">0.05</span><span class="p">,</span>
    <span class="n">target_modules</span><span class="o">=</span><span class="p">[</span><span class="s2">"q_proj"</span><span class="p">,</span><span class="s2">"k_proj"</span><span class="p">,</span><span class="s2">"v_proj"</span><span class="p">,</span><span class="s2">"o_proj"</span><span class="p">,</span><span class="s2">"gate_proj"</span><span class="p">,</span><span class="s2">"up_proj"</span><span class="p">,</span><span class="s2">"down_proj"</span><span class="p">])</span>

<span class="n">dpo_config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./dpo_output"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>                    <span class="c1"># KL 正则强度</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">5e-7</span><span class="p">,</span>          <span class="c1"># 学习率非常低，利于稳定</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"sigmoid"</span><span class="p">,</span>         <span class="c1"># 标准 DPO loss</span>
    <span class="n">max_length</span><span class="o">=</span><span class="mi">2048</span><span class="p">,</span>             <span class="c1"># 最大序列长度</span>
    <span class="n">max_prompt_length</span><span class="o">=</span><span class="mi">1024</span><span class="p">,</span>      <span class="c1"># prompt 截断长度</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>  <span class="c1"># 有效 batch = 16</span>
    <span class="n">gradient_checkpointing</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>          <span class="c1"># DPO 易过拟合——只跑 1 个 epoch！</span>
    <span class="n">warmup_ratio</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
    <span class="n">logging_steps</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span>
    <span class="n">eval_strategy</span><span class="o">=</span><span class="s2">"steps"</span><span class="p">,</span>
    <span class="n">eval_steps</span><span class="o">=</span><span class="mi">200</span><span class="p">,</span>
    <span class="n">save_strategy</span><span class="o">=</span><span class="s2">"steps"</span><span class="p">,</span>
    <span class="n">save_steps</span><span class="o">=</span><span class="mi">500</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>             <span class="c1"># 配合 LoRA 时，ref = base 模型（无需另外拷贝！）</span>
    <span class="n">args</span><span class="o">=</span><span class="n">dpo_config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">[</span><span class="s2">"train"</span><span class="p">],</span>
    <span class="n">eval_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">[</span><span class="s2">"test"</span><span class="p">],</span>
    <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
    <span class="n">peft_config</span><span class="o">=</span><span class="n">lora_config</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
<span class="c1"># 需要监控的关键指标：train/rewards/chosen、train/rewards/rejected、train/rewards/margins</span>
</code></pre></div>
