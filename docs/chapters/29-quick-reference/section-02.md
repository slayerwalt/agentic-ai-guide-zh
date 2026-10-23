# 29.2　Transformer 与架构公式 {#transformer-architecture-formulas}



<div class="arithmatex">\[\begin{align*}
\text{Self-Attention:}&amp;\quad \text{Attn}(Q,K,V) = \text{softmax}(QK^\top / \sqrt{d_k}) \cdot V \tag{29.11}\\

\text{Multi-Head:}&amp;\quad \text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O,\quad \text{head}_i = \text{Attn}(XW_i^Q, XW_i^K, XW_i^V) \tag{29.12}\\

\text{RoPE:}&amp;\quad f(x_m, m) = x_m e^{im\theta_j}, \quad \theta_j = 10000^{-2j/d} \tag{29.13}\\

\text{LoRA:}&amp;\quad W&#x27; = W_0 + (\alpha/r) \cdot BA, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k} \tag{29.14}\\

\text{KD (soft targets):}&amp;\quad L_\text{KD} = (1{-}\alpha)L_\text{CE}(y, \hat{y}) + \alpha\, T^2 \cdot \text{KL}(p_T^\text{teacher} \| p_T^\text{student}) \tag{29.15}\\

\text{FFN (SwiGLU):}&amp;\quad \text{FFN}(x) = (\text{Swish}(xW_1) \odot xW_3) W_2
\tag{29.16}
\end{align*}\]</div>
