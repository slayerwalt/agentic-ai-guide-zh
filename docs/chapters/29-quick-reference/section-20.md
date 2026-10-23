# 29.20　智能体 RL 公式 {#agentic-rl-formulas}



<div class="arithmatex">\[\begin{align*}
\text{Trajectory GRPO:}&amp;\quad \hat{A}_i = (R(\tau_i) - \mu_G)/\sigma_G, \quad R(\tau_i) = \sum_{t} r_t^{(\tau_i)} \tag{29.23}\\

\text{Agent reward:}&amp;\quad R = w_1 R_\text{task} + w_2 R_\text{efficiency} + w_3 R_\text{safety}, \quad R_\text{eff} = \max(0, 1 - \text{steps}/N_\text{max}) \tag{29.24}\\

\text{Masking:}&amp;\quad \mathcal{L} = \sum_{t \in \text{agent tokens}} \min(r_t \hat{A}_t,\; \text{clip}(r_t) \hat{A}_t) \quad \text{(mask env outputs)} \tag{29.25}\\

\text{Pass@}k:&amp;\quad 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}, \quad n = \text{total samples},\; c = \text{correct}
\tag{29.26}
\end{align*}\]</div>
