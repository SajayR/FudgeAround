Refer to TUTORIAL.md to understand the codebase. You are free to edit the entire codebase as you see fit in order to implement the following task--including train.py.
Make sure to do extensive testing for your implementations
# BS-ReLoRA: formal spec

## Objects and notation (per linear layer (W\in\mathbb{R}^{m\times n}))

* Loss (\mathcal{L}). Batch activations into the layer (X_t\in\mathbb{R}^{B\times n}); upstream gradient (G_t=\partial \mathcal{L}/\partial y_t\in\mathbb{R}^{B\times m}).
* Full gradient (not materialized if you don’t want): ( \nabla W_t = G_t^\top X_t \in \mathbb{R}^{m\times n}).
* Adam moments on (W): (m_t, v_t) (elementwise), bias-corrected (\hat m_t,\hat v_t). Adam step (as a *matrix*):
  [
  S_t ;=; \eta,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} ;\in; \mathbb{R}^{m\times n}.
  ]
* Low-rank budget (r), oversampling (p) (e.g., 4), target rank (r^\star=r+p).
* Cycle index (k=0,1,2,\dots). LoRA factors per cycle: (A_k\in\mathbb{R}^{r\times n}), (B_k\in\mathbb{R}^{m\times r}). Update (\Delta W_k = B_k A_k).
* (Optional) memory subspaces (U_{\text{mem}}\in\mathbb{R}^{m\times R_L}, V_{\text{mem}}\in\mathbb{R}^{n\times R_R}) containing orthonormal columns that summarize prior cycles.

---

## Cycle structure (for each (k))

### Phase 1: **Probe** with K full steps


**Mode P-shadow.**

1. Freeze (W) **positions** (no param updates), but **do update** Adam’s internal moments on (W) for K steps:

   * For (t=1..K): run forward/backward, accumulate batch gradients (g_t) and update (m_t, v_t) as Adam would. Do **not** change (W).
2. For each step, define the *would-be* preconditioned step (S_t=\eta,\hat m_t/(\sqrt{\hat v_t}+\epsilon)).
3. Form a **windowed estimate** of Adam’s step (take an EMA or average):
   [
   \bar S ;=; \frac{1}{K}\sum_{t=1}^{K} S_t \quad \text{(or EMA with decay (\beta_{\text{probe}}\in[0.8,0.95])).}
   ]

---

### Phase 2: **Subspace extraction** (rank (r))

**S-svd.**

* Compute **randomized SVD** of (\bar S) to rank (r^\star) (oversampling (p), 1 power iteration):
  [
  \bar S \approx U_{r^\star},\Sigma_{r^\star},V_{r^\star}^\top \quad\Rightarrow\quad U_r, V_r.
  ]

Hence we get the top-(r) left/right subspaces of the *Adam step*, not just raw gradients.

---

### Phase 3: **Soft deflation** (avoid retreading)

Project away the part already spanned by previous cycles, but not too aggressively:

[
\tilde U = \mathrm{qr}!\big(,(I - \lambda_U U_{\text{mem}}U_{\text{mem}}^\top),U_r\big),\quad
\tilde V = \mathrm{qr}!\big(,(I - \lambda_V V_{\text{mem}}V_{\text{mem}}^\top),V_r\big),
]
with (\lambda_U,\lambda_V\in[0.3,1.0]) (e.g., 0.5) controlling **soft** deflation.
Append (\tilde U,\tilde V) to the memories (re-orthonormalize; cap memory dims).

---

### Phase 4: **Low-rank phase** (LoRA for (N) steps)

**L-constrained.**

* Parameterize (B_k = \tilde U R), (A_k = L \tilde V^\top) with learnable **cores** (L,R\in\mathbb{R}^{r\times r}).
* Only (L,R) are trainable this phase (and any biases, norms, etc., as usual).
* Update equals ( \Delta W_k = \tilde U, (R L), \tilde V^\top).

---

### Phase 5: **Merge** and state handling

* Merge in-place: (W \leftarrow W + \Delta W_k) (use your effective factors if you also kept C-ReLoRA rotations).
* **Adapter Adam state:** set (m_{A_k}=m_{B_k}=0); shrink (v) (e.g., (\times 0.5)) so next cycles start stable but fresh.
* **Base (W) Adam state:** do a **magnitude-aware damp** (avoid divergence while flushing stale direction):

  * Let (\rho = |\Delta W_k|_F/(|W|_F+\epsilon)).
  * Momentum: (m_W \leftarrow \alpha_m(\rho), m_W) with (\alpha_m\in[0.01,0.5]) for typical (\rho\sim10^{-3}).
  * Second moment: (v_W \leftarrow \alpha_v(\rho), v_W + (1-\alpha_v),v_{\min}) with (\alpha_v \in [0.5,0.9]).
  * (Nice extra) **project-and-damp along the new span**:
    [
    m_W \leftarrow m_W - \gamma \big(\tilde U\tilde U^\top m_W \tilde V\tilde V^\top\big),\ \gamma\in[0.5,1].
    ]
* Optional **LR cooldown** on base (W) for 100–200 steps (e.g., ×0.5), then restore.

Repeat next cycle.


### TL;DR

Do **K=32** full steps to *measure* the **Adam step’s top-(r) subspace**, **soft-deflate** it against earlier cycles, **execute** most of the work with LoRA inside that subspace for (N) steps, **merge**, and **damp** states so momentum doesn’t drag you backward. This is the cleanest way to “emulate full-rank Adam with a low-rank budget,” and it stays elegant, analyzable, and inexpensive.
