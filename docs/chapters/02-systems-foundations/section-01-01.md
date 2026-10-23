# 2.1.1　为什么深度学习要用 GPU？ {#why-gpus-for-deep-learning}

<p>GPU 和 CPU 代表了根本不同的硬件设计哲学。理解这种差异能解释为何 LLM 训练在 GPU 上能快 100–1000 倍。</p>

<div class="admonition tip"><p class="admonition-title">CPU 与 GPU——基本设计哲学</p>
<ul>
<li><p><strong>CPU</strong> 针对<em>延迟</em>进行优化——它们以尽可能快的速度执行少量线程，配备大型缓存、分支预测器和乱序执行单元。现代 CPU 拥有 8–96 个核心。</p></li>
<li><p><strong>GPU</strong> 针对<em>吞吐量</em>进行优化——它们并行执行数千个线程，每个线程只做简单工作。现代 GPU 拥有数千个“核心”（执行单元），它们被组织成流式多处理器（Streaming Multiprocessor，SM）。</p></li>
</ul>

<p>深度学习工作负载以矩阵乘法（在 <span class="arithmatex">\(O(n^2)\)</span> 数据上执行 <span class="arithmatex">\(O(n^3)\)</span> 操作）为主，这是天然并行的。70B 模型的单次 transformer 前向传播每个 token 需要约 140 TFLOP 计算量——非常适合 GPU 的吞吐能力。</p>
</div>
