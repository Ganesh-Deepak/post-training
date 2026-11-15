# Post-Training in Practice: curriculum

*Course 2 of the library · Model track · 20 chapters · about 25 hours · status: preview. Generated from `curricula/posttraining.txt` in the hub repository; edit it there.*

How to take a base or instruct model and make it measurably better at your task. You will decide whether to fine-tune at all, build and decontaminate a dataset, run SFT, preference optimisation and RL with verifiable rewards on real open models, keep what the model already knew, and prove the result with the methods of the Evaluation course. Transformers Ch 23–25 derived the algorithms; this course runs them.

**For:** Engineers who understand SFT, DPO and GRPO on paper and need to make them work on real models and real data.

**By the end you can:**

- Decide between prompting, retrieval, fine-tuning and distillation with a cost model
- Build, filter, deduplicate and decontaminate a training set
- Train LoRA and full fine-tunes that beat a prompted baseline, and measure what was forgotten
- Run DPO-family methods and GRPO-family RL with verifiers, and diagnose them from their logs
- Evaluate and ship a fine-tune honestly

**Before you start:** Transformers From Scratch Ch 23 (SFT, LoRA, RLHF, and PPO); Transformers From Scratch Ch 24 (DPO From First Principles); Transformers From Scratch Ch 25 (DPO Variants, GRPO, and RLVR); Evaluation & Experimentation for LLM Systems Ch 06 (Comparing Two Systems)

**Current to November 2026:** The open recipes of 2025–26 (Tülu 3, OLMo 3: SFT → DPO → scaled RLVR); DAPO, Dr. GRPO, GSPO and CISPO as fixes to GRPO; asynchronous RL with inference engines and the trainer–sampler mismatch it creates; on-policy distillation; rubrics as rewards for tasks without verifiers; multi-turn agentic RL; model merging; "LoRA without regret" — when LoRA matches full fine-tuning and when it cannot.

**Not covered:** Pre-training from scratch (Transformers Ch 11 and 21), and closed fine-tuning APIs except as a baseline.

**Reference package:** `ptkit/` · **Semantic hues:** dim-1 = the policy; dim-2 = the reference model; dim-3 = the data; dim-4 = the reward; dim-5 = the verifier


## Part I · Decide and prepare

*Most of the result is decided before training starts.*

### 01 · Should You Fine-Tune at All? (55 min)

*Prompting, retrieval, fine-tuning and distillation, costed.*

Builds on: Transformers From Scratch Ch 23 (SFT, LoRA, RLHF, and PPO); Evaluation & Experimentation for LLM Systems Ch 01 (Defining "Good" Before Writing Code)

1. **What fine-tuning can and cannot add** — behaviour, format and skill, but rarely new facts
2. **The four options** — prompt, retrieve, fine-tune, distil — and the failure each one fixes
3. **The cost model** — data, training, evaluation, and serving an adapter versus a longer prompt
4. **Strong baselines first** — the best prompt, few-shot, and a larger model are the bar to beat
5. **Worked example** — one task, four approaches, one table

**Build:** a cost calculator for the four options; a baseline harness used in every later chapter.

### 02 · Anatomy of a Modern Post-Training Recipe (60 min)

*What each stage of an open model's pipeline buys.*

Builds on: Transformers From Scratch Ch 23 (SFT, LoRA, RLHF, and PPO); Transformers From Scratch Ch 25 (DPO Variants, GRPO, and RLVR)

1. **The 2025–26 open recipes** — mid-training, SFT, DPO, RLVR, and why that order
2. **What each stage changes** — measured on the released checkpoints of a fully open model
3. **Reasoning and non-reasoning variants** — thinking budgets and how they are trained
4. **Where the compute goes** — RL is now a large share of post-training compute
5. **Reading a technical report** — the tables that matter and the ones that do not

**Build:** an evaluation sweep over the intermediate checkpoints of an open model.

### 03 · Chat Templates, Masks, and Packing (60 min)

*The data plumbing that silently breaks runs.*

Builds on: Transformers From Scratch Ch 03 (Tokenization: Text → Integers); Transformers From Scratch Ch 23 (SFT, LoRA, RLHF, and PPO)

1. **Chat templates** — roles, special tokens, and the bugs a wrong template causes
2. **Loss masking** — train on assistant turns only, and verify it
3. **Packing with document masks** — many examples per sequence without attention leaking between them
4. **Tool calls and reasoning traces in the template**
5. **Tests for the pipeline** — round-trips, mask audits and token counts before any GPU time

**Build:** a data collator with packing and document masks, and the tests that prove both.

### 04 · Datasets: Sourcing, Filtering, Mixing (70 min)

*Quality, diversity, and the mixture.*

Builds on: Transformers From Scratch Ch 11 (Training a Transformer); this course Ch 03 (Chat Templates, Masks, and Packing)

1. **Where data comes from** — public sets, your logs, experts, and models
2. **Quality filtering** — heuristics, classifier scores, and LLM grading, with their failure modes
3. **Deduplication** — exact, near-duplicate (MinHash) and semantic
4. **Diversity** — measuring coverage of skills and intents
5. **Mixing ratios** — how much of each source, and how to tune the mixture cheaply

**Build:** a filtering and dedup pipeline with a report of what each stage removed.

### 05 · Decontamination (45 min)

*Make sure you are not training on your test.*

Builds on: Evaluation & Experimentation for LLM Systems Ch 15 (Contamination and Leakage); this course Ch 04 (Datasets: Sourcing, Filtering, Mixing)

1. **Leakage paths** — public benchmarks, paraphrases, and synthetic data generated from test items
2. **N-gram and embedding-similarity checks against every eval you will report**
3. **Choosing thresholds** — measured false positives and false negatives
4. **Decontaminating synthetic data at the prompt and the answer**

**Build:** a decontamination step that runs in the pipeline and fails the build on leakage.

### 06 · Synthetic Data (70 min)

*Generation, verification, and collapse.*

Builds on: this course Ch 04 (Datasets: Sourcing, Filtering, Mixing); this course Ch 05 (Decontamination)

1. **Generation patterns** — self-instruct, persona-driven prompts, and evolving difficulty
2. **Rejection sampling** — generate many, keep the verified ones
3. **Verification** — programmatic checks, judges, and agreement between models
4. **Mode collapse and model collapse** — measured as lost diversity over generations
5. **Licences and terms** — which model outputs you may train on

**Build:** a synthetic-data generator with verification and a diversity monitor.


## Part II · Supervised fine-tuning

*Teaching a format, a skill, a voice.*

### 07 · SFT That Works (70 min)

*The hyperparameters that matter and the ones that do not.*

Builds on: Transformers From Scratch Ch 11 (Training a Transformer); this course Ch 03 (Chat Templates, Masks, and Packing)

1. **Learning rate first** — sweeps, and why the right value depends on the model size
2. **Epochs and overfitting** — validation loss is not the target; task score is
3. **Batch size, schedule and warmup for fine-tuning**
4. **Full fine-tuning** — memory, optimiser state and gradient checkpointing on one node
5. **Diagnosing an SFT run from its curves**

**Build:** an SFT trainer on an open model, with a learning-rate sweep and its evaluation.

### 08 · LoRA and QLoRA in Practice (70 min)

*Rank, targets, learning rate — and when LoRA falls short.*

Builds on: Transformers From Scratch Ch 23 (SFT, LoRA, RLHF, and PPO)

Related: Mathematics for Machine Learning, from First Principles Ch 05 (The SVD Explains Half the Field)

1. **Settings that matter** — all linear layers, rank, alpha, and a higher learning rate than full fine-tuning
2. **When LoRA matches full fine-tuning** — and when capacity runs out (large datasets, RL against SFT)
3. **QLoRA** — NF4, double quantisation, paged optimisers, and the quality cost measured
4. **Variants** — DoRA, PiSSA, rsLoRA: what each changes, compared on one task
5. **Merging and serving adapters**

**Build:** LoRA versus full fine-tuning on one task, with a rank sweep and a memory table.

### 09 · Forgetting, Measured (60 min)

*What the model lost while it learned your task.*

Builds on: this course Ch 07 (SFT That Works); Evaluation & Experimentation for LLM Systems Ch 06 (Comparing Two Systems)

1. **A capability-retention suite** — the general skills you must not lose
2. **Measuring forgetting** — paired scores before and after, with intervals
3. **Mitigations** — replay data, lower learning rates, LoRA, and KL penalties
4. **Model merging** — task arithmetic, TIES and averaging checkpoints
5. **The trade-off curve** — task gain against retained capability

**Build:** a retention suite and a merge experiment that recovers lost capability.

### 10 · Distillation (65 min)

*Small models taught by large ones.*

Builds on: this course Ch 07 (SFT That Works)

Related: Mathematics for Machine Learning, from First Principles Ch 13 (KL in Practice)

1. **Sequence-level distillation** — train on the teacher's outputs
2. **Logit distillation** — matching distributions, and the choice of forward or reverse KL
3. **On-policy distillation** — the student samples, the teacher grades every token
4. **Distilling reasoning** — traces, lengths, and what transfers
5. **Choosing the teacher** — bigger is not always better for a small student

**Build:** sequence-level, logit and on-policy distillation compared on one task.


## Part III · Preferences

*Learning from comparisons.*

### 11 · Preference Data (55 min)

*Collection, bias, and on-policy pairs.*

Builds on: Transformers From Scratch Ch 23 (SFT, LoRA, RLHF, and PPO); Evaluation & Experimentation for LLM Systems Ch 03 (Labels and Agreement)

1. **Where pairs come from** — people, models, and your own sampled outputs
2. **Annotator bias** — length, confidence and style preferences, measured
3. **On-policy against off-policy pairs** — why pairs from your own model work better
4. **Synthetic preferences** — judge-labelled pairs and their systematic errors

**Build:** a preference-pair builder from your own samples, with a bias audit.

### 12 · Reward Models (65 min)

*Training, evaluating, and hacking them.*

Builds on: Transformers From Scratch Ch 23 (SFT, LoRA, RLHF, and PPO); this course Ch 11 (Preference Data)

1. **Bradley–Terry reward models on a real backbone**
2. **Evaluating a reward model** — accuracy on held-out pairs, and what benchmarks miss
3. **Generative reward models** — a judge that reasons before it scores
4. **Process reward models** — scoring steps, not answers
5. **Reward hacking** — over-optimisation curves and ensembles that slow it down

**Build:** a reward model with a held-out evaluation and a best-of-n over-optimisation curve.

### 13 · The DPO Family on One Dataset (70 min)

*DPO, IPO, KTO, ORPO and SimPO, compared fairly.*

Builds on: Transformers From Scratch Ch 24 (DPO From First Principles); this course Ch 11 (Preference Data)

1. **One dataset, one budget, five losses** — a fair comparison
2. **Length exploitation** — and length-normalised and length-controlled variants
3. **Choosing β** — the KL trade-off, measured
4. **Iterative and online DPO** — sampling new pairs from the current policy
5. **Reading DPO logs** — chosen and rejected log-probabilities, margins and accuracy

**Build:** a single trainer with five losses and the comparison report.


## Part IV · Reinforcement learning

*Rewards from verifiers, rubrics and environments.*

### 14 · RL Infrastructure (75 min)

*Rollouts, trainers, and the mismatch between them.*

Builds on: Transformers From Scratch Ch 25 (DPO Variants, GRPO, and RLVR)

Related: Inference Engineering at Scale Ch 03 (Inside a Serving Engine)

1. **The loop** — sample with an inference engine, score, train, sync weights
2. **Synchronous against asynchronous RL** — throughput, and policy lag
3. **The trainer–sampler mismatch** — different kernels, different probabilities; importance-sampling corrections
4. **Weight synchronisation and memory** — colocated against separate GPU pools
5. **Throughput accounting** — where the time goes in an RL step

**Build:** a minimal RL loop with an inference engine, with the mismatch measured and corrected.

### 15 · GRPO and Its Fixes (75 min)

*DAPO, Dr. GRPO, GSPO and CISPO in practice.*

Builds on: Transformers From Scratch Ch 25 (DPO Variants, GRPO, and RLVR); this course Ch 14 (RL Infrastructure)

1. **GRPO's known problems** — length bias, zero-variance groups, entropy collapse, clipping
2. **DAPO** — clip-higher, dynamic sampling, token-level loss, overlong shaping
3. **Dr. GRPO and GSPO** — removing the length bias; sequence-level ratios
4. **CISPO and other clipped-importance variants**
5. **Monitoring a run** — entropy, response length, reward and KL, and what each pattern means

**Build:** GRPO and three fixes on one verifiable task, with diagnostic dashboards.

### 16 · Verifiable Rewards (65 min)

*Verifiers, sandboxes, and reward hacking.*

Builds on: this course Ch 15 (GRPO and Its Fixes)

Related: Agents & Tool Use Ch 12 (Sandboxing and Least Privilege)

1. **Math and exact-answer verifiers** — extraction and equivalence
2. **Code verifiers** — unit tests in sandboxes, timeouts and partial credit
3. **Format rewards and their side effects**
4. **Difficulty filtering and curricula** — keep prompts the model sometimes solves
5. **Reward hacking in RLVR** — test tampering, special-casing, and how to detect it

**Build:** a sandboxed code verifier and a difficulty-filtered training set.

### 17 · Rewards Without Verifiers (60 min)

*Rubrics, judges, and AI feedback.*

Builds on: Evaluation & Experimentation for LLM Systems Ch 11 (Rubrics, Reward Models, and Goodhart); this course Ch 12 (Reward Models)

1. **Rubrics as rewards** — expert criteria scored by a judge, per rollout
2. **LLM-judge rewards** — their noise, their bias, and how the policy exploits them
3. **Constitutional and AI-feedback methods**
4. **Mixing verifiable and judged rewards in one run**

**Build:** rubric-reward RL on an open-ended task, with a human audit of the gains.

### 18 · Multi-Turn and Agentic RL (75 min)

*Environments, tools, and credit over many steps.*

Builds on: this course Ch 16 (Verifiable Rewards)

Related: Agents & Tool Use Ch 16 (Improving Agents)

1. **Environments** — tools, state, and episode termination
2. **Credit assignment across turns** — outcome rewards, turn-level rewards and masks for tool output
3. **Long rollouts** — truncation, context limits, and asynchronous sampling
4. **Environment design** — tasks the model can learn from, and hackable rewards to avoid

**Build:** RL on a small tool-use environment, with trajectory-level evaluation.


## Part V · Ship it

*Prove it, then serve it.*

### 19 · Evaluating a Fine-Tune Honestly (55 min)

*Baselines, retention, safety, and significance.*

Builds on: Evaluation & Experimentation for LLM Systems Ch 06 (Comparing Two Systems); this course Ch 09 (Forgetting, Measured)

1. **The comparison table** — prompted baseline, larger model, and your fine-tune
2. **Paired significance on the task, and retention on everything else**
3. **Safety regression** — fine-tuning can remove refusal behaviour; measure it
4. **Writing the model card** — data, training, evaluation and known limits

**Build:** the evaluation report generator used for the capstone.

### 20 · Capstone: Beat the Prompt (300 min)

*An open model, your task, a measured win.*

Builds on: this course Ch 19 (Evaluating a Fine-Tune Honestly)

Related: Inference Engineering at Scale Ch 16 (Many Models and Many Adapters)

1. **Choose the task and the baseline** — and write down the bar before training
2. **The dataset** — sourced, filtered, deduplicated and decontaminated
3. **SFT, then preference optimisation or RL** — with logs you can explain
4. **Forgetting and safety reports**
5. **Serving the result** — merged or as an adapter, quantised, with its cost

**Build:** the full pipeline, the model card, and the paired evaluation.

## Capstone

Take an open 1–8B model to a measured win over the best prompt on a task you choose: a decontaminated dataset, an SFT and a preference or RL stage, a forgetting report, and a paired evaluation with intervals.
