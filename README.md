# sft-mode-collapse

> **Investigating mode collapse in supervised fine-tuning when multiple valid outputs exist per input, and a training objective that prevents it.**

## The Problem

Many real tasks have multiple correct outputs for a single input. A programming problem has many valid solutions, a prompt has many good translations, a book page has many high-quality study guide questions. Standard supervised fine-tuning (SFT) treats each (input, output) pair independently, but when the training data contains these **equivalence sets** of valid outputs, the model reliably collapses to generating only one mode — even when it's trained on all of them.

## Why It Happens

The core issue is the training objective itself. Negative log-likelihood (NLL) minimization sharpens the model's output distribution with every gradient step, concentrating probability mass on fewer and fewer outputs regardless of how diverse the training data is. This is not a side effect of how we arrange or batch the data — it's a structural property of what the loss function optimizes toward.

### NLL sharpening is the primary cause

The cross-entropy gradient for each training example takes the form `P(k|x) - 1[k=y]`: it pushes probability toward the target token and drains it from everything else. Over many updates, this concentrates the distribution — the model becomes increasingly confident, assigning more mass to a shrinking set of outputs.

This operates even when there is no conflict between training examples at all. A model trained on a single randomly chosen solution per problem (no conflicting targets, no gradient interference) loses *more* diversity than one trained on all solutions — collapsing from 8.17 to 3.84 clusters, compared to ~5.0 clusters for multi-output training. NLL sharpening alone accounts for the majority of diversity loss.

Given enough training, the optimal solution under NLL is unimodal by construction ([GX-Chen et al., 2025](https://arxiv.org/abs/2510.09683)). Mode collapse is not an optimization failure. It is the intended behavior of the objective.

### Conflicting gradients help, not hurt

This is counterintuitive. When solutions A and B both appear in training for the same input, they provide opposing gradients at their divergence points — A says "increase the logit for `for`," B says "increase the logit for `while`." The averaged gradient pushes both tokens toward equal probability, effectively trying to maintain both modes.

This averaging creates a weak protective effect against collapse. Multiple valid outputs inject stochastic noise into the optimization that partially resists NLL sharpening. Our experiments confirm this: multi-output training preserves ~1 additional cluster compared to single-output training (5.0 vs 3.84 clusters). The "contradictory supervision" from equivalence sets is actually a form of implicit diversity regularization.

### But the protection isn't strong enough

The averaging equilibrium — equal probability at each branch point — is unstable in practice for two reasons.

**Autoregressive compounding.** Even if the model achieves P(`for`) = P(`while`) = 0.5 at a divergence point, downstream tokens have only one correct continuation per branch. The model sharpens each branch independently. A tiny perturbation at the branch point cascades: if P(`for`) nudges to 0.51, the entire A-path gets slightly stronger training signal, which makes `for` more likely in the next update, and so on. Small biases compound into full mode collapse through the sequential structure of generation.

**The softmax bottleneck.** A single hidden state at the divergence point must produce a distribution over next tokens. The model's finite capacity limits how well it can represent a truly multimodal distribution from one representation ([Yang et al., ICLR 2018](https://arxiv.org/abs/1711.03953)). Under pressure from NLL sharpening, committing to one mode is easier to represent than maintaining balanced coverage of several.

### Batch composition controls speed, not outcome

When solutions A and B appear in the same batch, their opposing gradients partially cancel within one step, producing a weaker net update. When they appear in separate batches, the model takes full steps toward A and then full steps toward B, oscillating. The per-step dynamics differ, but the loss landscape — and therefore the optimum — does not.

Our collision rate sweep confirms this: training with 0% within-batch collisions produces the same diversity as training with 50% or 75% collisions (~4.5–5.4 clusters across all conditions). Even 100% collision rate, where gradient cancellation is maximal and loss barely decreases, still collapses to ~5 clusters. Batch ordering is a second-order effect. The objective function drives the collapse.

### Shuffling delays the inevitable

Randomizing batch order reduces within-batch collisions, which adds stochastic noise that slows convergence. But the optimum doesn't change — it's a property of the loss, not the data ordering. Given enough epochs, a shuffled model collapses too. Shuffling treats the symptom (correlated gradients within a batch) rather than the disease (an objective whose solution is unimodal).

### Temperature cannot recover lost modes

Vérine et al. (ICML 2025) prove that if a model hasn't been trained toward coverage, raising temperature at inference time spreads probability mass to garbage outputs, not to the valid modes that were lost. Diversity must be baked into training.

### Summary

| Force | Effect on diversity | Magnitude |
|-------|-------------------|-----------|
| NLL sharpening | Destroys diversity by concentrating the distribution | Dominant (~40% of clusters lost) |
| Conflicting gradients (multi-output) | Weakly preserves diversity via implicit noise | Small (~1 cluster preserved) |
| Batch collision rate | Controls collapse speed, not outcome | Negligible |
| Shuffling | Delays collapse, doesn't prevent it | Temporary |
| Temperature scaling | Cannot recover lost modes | Zero |

## The Fix: Equivalence Coverage Optimization (ECO)

Since the problem is the objective, the fix must change the objective. ECO adds a **coverage term** to the training loss that explicitly penalizes probability imbalance across valid outputs:

```
L_ECO = L_NLL + λ · Var(log P(y_i | x))    for y_i ∈ equivalence set
```

The variance penalty says: if `P(solution_A) >> P(solution_B)`, push A down and B up. This changes what the optimum *is* — the loss-minimizing solution must now assign high, *balanced* probability across all valid outputs, not concentrate on one. This directly counteracts NLL sharpening within the equivalence set, stabilizing the branch-point equilibrium that is otherwise unstable.

Crucially, ECO works regardless of batch composition, data ordering, or training duration, because it modifies the loss landscape itself rather than relying on stochastic noise to resist collapse.

**Why not simpler alternatives?**
- **Label smoothing** spreads probability to *all* tokens uniformly, wasting mass on invalid outputs. ECO targets only the valid set.
- **Mixture of Experts** (Shen et al., ICML 2019) requires architectural changes, knowing K modes upfront, and substantial overhead. ECO is a single loss term.
- **Minimum Risk Training** requires expensive sampling at training time. ECO uses quantities already computed in the forward pass.
- **KL to a uniform target** over valid outputs is theoretically clean but requires computing full sequence-level probabilities for all valid outputs on every step. ECO's variance penalty is a cheap approximation — you're already computing `log P(y_i)` for whichever examples are in the batch, and the variance over those is nearly free.

ECO is the **minimal, tractable modification** to standard SFT that fixes the problem. One extra term, computed from quantities already available, no sampling, no architecture changes, no separate target distribution.

## Experimental Design

We use code generation as a testbed because output correctness is objectively verifiable through test execution — unlike natural language tasks where "valid" is subjective.

### Dataset: CodeContests Equivalence Sets

We construct equivalence sets from [CodeContests](https://github.com/google-deepmind/code_contests) (Li et al., 2022):

1. Extract problems with ≥5 Python3 solutions from the training split
2. Normalize and deduplicate solutions
3. Verify correctness against complete test suites (batched execution, parallel workers)
4. Cluster by algorithmic features (data structures, control flow, library usage)

**Result:** 1,334 problems, mean 18.6 verified solutions per problem, mean 9.0 algorithmically distinct clusters per problem.

### Experiments

Three conditions establish whether the problem exists and if ECO fixes it:

| Condition | Batch ordering | Loss function | Tests |
|-----------|---------------|---------------|-------|
| Grouped SFT | Same-problem solutions adjacent | Standard NLL | Expected: Collapse happens fast |
| Shuffled SFT | Randomly ordered | Standard NLL | Expected: Collapse happens slowly |
| Grouped ECO | Same-problem solutions adjacent | NLL + coverage | Expected: Collapse is prevented |

If Grouped ECO matches or exceeds Shuffled SFT in output diversity, that demonstrates a loss-level fix that works regardless of batch composition — important because batch ordering can't always be controlled (distributed training, streaming data, curriculum learning).

### Diversity Metrics

All measured on held-out problems via sampling multiple generations per prompt:

- **Cluster coverage** — how many algorithmically distinct approaches appear in outputs
- **Self-BLEU** — surface lexical diversity (lower = more diverse)
- **Unique solutions** — count of syntactically distinct generations

## Key References

**The objective itself causes collapse:**
- GX-Chen et al. (2025). *KL-Regularized Reinforcement Learning is Designed to Mode Collapse.* [[arXiv]](https://arxiv.org/abs/2510.20817)
- Vérine et al. (ICML 2025). *Improving Diversity in Language Models: When Temperature Fails, Change the Loss.* [[arXiv]](https://arxiv.org/abs/2508.09654)
- Kirk et al. (ICLR 2024). *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* [[arXiv]](https://arxiv.org/abs/2310.06452)

**Diversity-aware training (closest related work):**
- Lanchantin et al. (Meta, 2025). *DivPO.* Achieves diversity through data curation rather than loss modification. [[arXiv]](https://arxiv.org/abs/2501.18101)
- Chang et al. (ICLR 2025). *Diverse Preference Learning for Capabilities and Alignment.* Decouples entropy from KL regularization for global diversity control. [[arXiv]](https://arxiv.org/html/2511.08594v1)
- Wu et al. (NeurIPS 2025). *GEPO.* Variance penalty on log-probability ratios within equivalent groups for code translation. [[arXiv]](https://arxiv.org/abs/2505.12723)

**The multi-output problem across domains:**
- Li et al. (Science, 2022). *Competition-Level Code Generation with AlphaCode.* CodeContests dataset. [[arXiv]](https://arxiv.org/abs/2203.07814)
- Shen et al. (ICML 2019). *Mixture Models for Diverse Machine Translation.* K expert decoders for multi-modal translation. [[arXiv]](https://arxiv.org/abs/1902.07816)
- Parker-Holder et al. (NeurIPS 2020). *Effective Diversity in Population Based Reinforcement Learning.* Differentiable set-level diversity loss. [[arXiv]](https://arxiv.org/abs/2002.00632)
