# 29.1　核心 RL 与对齐公式 {#core-rl-alignment-equations}



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
