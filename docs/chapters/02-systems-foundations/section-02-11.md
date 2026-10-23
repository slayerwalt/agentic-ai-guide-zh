# 2.2.11　前缀缓存（自动 Prompt 缓存） {#prefix-caching-automatic-prompt-caching}

<p>当多个请求共享相同前缀（系统 prompt、few-shot 示例）时：</p>

<ol>
<li><p>对每个逻辑块的 token 内容计算哈希。</p></li>
<li><p>新请求到达时，检查是否有前缀块已在缓存中。</p></li>
<li><p>命中时：跳过这些 token 的 prefill，直接复用物理 KV 块。首 token 延迟（Time-to-first-token）大幅下降。</p></li>
<li><p>驱逐：LRU 策略。缓存块仅在内存压力需要时才被释放。</p></li>
</ol>

<p><strong>影响</strong>：对于具有长系统提示（2K+ token、所有用户共享）的聊天应用，前缀缓存可降低 TTFT 60–80%。</p>
