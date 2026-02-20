# sft-mode-collapse

> **Investigating mode collapse in supervised fine-tuning when multiple valid outputs exist per input, and a training objective that prevents it.**

## The Problem

Many real tasks have multiple correct outputs for a single input. A programming problem has many valid solutions, a prompt has many good translations, a book page has many high quality study guide questions. Standard supervised fine-tuning (SFT) treats each (input, output) pair independently, but when the training data contains these **equivalence sets** of valid outputs, the model reliably collapses to generating only one mode, even when it's trained on all of them.

## Why It Happens

This phenomenon is fairly intuituve as it's a property of the objective itself.

Consider an example input X with all equally valid example outputs in the training set: A, B, and C.

**Standard NLL provides contradictory supervision when multiple valid outputs exist.** In a programming problem where solution A uses `for i in range(n):` and solution B uses `while queue:`, at the token position where these solutions diverge, the gradient from example A says "increase the logit for `for`, decrease the logit for `while`." The gradient from example B says the exact opposite. Negative log-likelihood (NLL) has no way to express "both A and B are correct" — each training example independently asserts that its specific token sequence is *the* right answer. When two examples disagree about what comes next for the same input, the loss function sees contradictory supervision.

**The only way to minimize loss under contradictory targets is to pick one and commit.** Once solution A gets slightly ahead of solution B , even if it was just from initialization noise, a positive feedback loop begins. A has lower loss, its gradient is smaller `(1 - P(A|x))`, while B's larger gradient partially *suppresses* A at their divergence point through the shared softmax. Given enough training, the optimal solution under NLL is unimodal by construction ([GX-Chen et al., 2025](https://arxiv.org/abs/2510.09683)). Mode collapse in this case is constructed into how optimization happens.

**Batch composition controls how fast this happens.** When solutions A and B appear in the same gradient update, their opposing gradients cancel directly within the single step. The model receives a single averaged signal that points toward whichever mode currently dominates. The more valid solutions that collide within a batch, the stronger this averaging effect. And it will occur at *every* divergence point between every pair of valid solutions in the batch.

**Shuffling the data slows the collapse but doesn't prevent it.** Randomizing batch order to reduce or eliminate within-batch collisions adds some stochastic noise that delays convergence to the unimodal optimum. But the optimum itself doesn't change — it's a property of the loss function, not the data ordering. Given enough epochs, a shuffled model collapses too. Shuffling treats the symptom (correlated gradients within a batch) rather than the disease (an objective whose solution is unimodal).

**Temperature cannot recover lost modes.** Vérine et al. (ICML 2025) prove that if a model hasn't been trained toward coverage, raising temperature at inference time spreads probability mass to garbage outputs, not to the valid modes that were lost. Diversity must be baked into training.

## The Fix: Equivalence Coverage Optimization (ECO)

We propose a simple augmentation to the opimization, called ECO, which adds a **coverage term** to the training loss that explicitly penalizes probability imbalance across valid outputs:

```
L_ECO = L_NLL + λ · Var(log P(y_i | x))    for y_i ∈ equivalence set
```

The variance penalty says: if `P(solution_A) >> P(solution_B)`, push A down and B up. This changes what the optimum *is* such that more than one valid solution can be effecitvely learned. The loss-minimizing solution must now assign high, *balanced* probability across all valid outputs, not concentrate on one. This holds regardless of batch composition, data ordering, or training duration.

**Why not simpler alternatives?**
- **Label smoothing** spreads probability to *all* tokens uniformly, wasting mass on invalid outputs. ECO targets only the valid set.
- **Mixture of Experts** (Shen et al., ICML 2019) requires architectural changes, knowing K modes upfront, and substantial overhead. ECO is a simpler, single loss term.
- **Minimum Risk Training** requires expensive sampling at training time. ECO uses quantities already computed in the forward pass.
- **KL to a uniform target** over valid outputs is theoretically clean but requires computing full sequence-level probabilities for all valid outputs on every step. ECO's variance penalty is a cheap approximation — you're already computing `log P(y_i)` for whichever examples are in the batch, and the variance over those is nearly free.

ECO's is the **minimal, tractable modification** to standard SFT that fixes the problem, without introducing any additional issues or failure points in instances when it's not needed. With one extra term, computed from quantities already available, ECO provides a simple fix to this mode collapse problem.

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
| Grouped SFT | Same-problem solutions adjacent | Standard NLL | Expectd: Collapse happens fast |
| Shuffled SFT | Randomly ordered | Standard NLL | Expected: Collapse happens slowly |
| Grouped ECO | Same-problem solutions adjacent | NLL + coverage | Expected: Collapse is prevented |

If Grouped ECO matches or exceeds Shuffled SFT in output diversity, that demonstrates a loss-level fix that works regardless of batch composition. This is important because batch ordering can't always be controlled (distributed training, streaming data, curriculum learning).

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
- Chang et al. (ICLR 2025). *Diverse Preference Learning for Capabilities and Alignment* Decouples entropy from KL regularization for global diversity control. [[arXiv]](https://arxiv.org/html/2511.08594v1)

**The multi-output problem across domains:**
- Li et al. (Science, 2022). *Competition-Level Code Generation with AlphaCode.* CodeContests dataset. [[arXiv]](https://arxiv.org/abs/2203.07814)
- Shen et al. (ICML 2019). *Mixture Models for Diverse Machine Translation.* K expert decoders for multi-modal translation. [[arXiv]](https://arxiv.org/abs/1902.07816)
- Parker-Holder et al. (NeurIPS 2020). *Effective Diversity in Population Based Reinforcement Learning.* Differentiable set-level diversity loss. [[arXiv]](https://arxiv.org/abs/2002.00632)
