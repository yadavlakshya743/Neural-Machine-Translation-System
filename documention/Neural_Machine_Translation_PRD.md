**Product Requirements Document (PRD)**

**Neural Machine Translation & Localization Platform**

# **Product Goal**

Create a production-oriented Generative AI translation platform that learns to generate high-quality English↔Hindi translations using a custom Transformer encoder-decoder model, grounded in “Attention Is All You Need” (NeurIPS 2017\) and trained on the IIT Bombay English-Hindi Corpus. The system will combine a trained generative translation model, rigorous NLP evaluation, experiment/model tracking, a FastAPI inference backend, and an interactive React localization workspace.

# **Product Scope**

The product is an end-to-end Neural Machine Translation (NMT) system rather than a simple API wrapper around a pre-trained translation service. The core generative model, training pipeline, decoding pipeline, evaluation workflow, and application layer must be integrated and reproducible. The primary supported directions are English→Hindi and Hindi→English.

* Custom Transformer encoder-decoder implementation suitable for training and inference.  
* Subword tokenization and language-aware preprocessing for English and Hindi.  
* Official IIT Bombay English-Hindi Corpus split for model development and evaluation.  
* Training, validation, checkpointing, experiment configuration, and model versioning.  
* Greedy decoding and beam-search decoding, with configurable inference parameters.  
* BLEU and chrF++ evaluation using reproducible evaluation scripts.  
* FastAPI REST service for translation and model/evaluation metadata.  
* React \+ Tailwind CSS localization workspace consuming the backend through Axios.  
* Authentication/user-session handling, translation history, validation, logging, and error handling.  
* Automated unit/integration tests, architecture documentation, setup instructions, and deployment.

# **Feature Priorities**

| Feature | Priority | Requirement |
| :---- | :---- | :---- |
| Corpus ingestion & preprocessing | Must | Acquire, validate, clean, normalize and prepare the IIT Bombay English-Hindi corpus using the official split. |
| Subword tokenization | Must | Train/use reproducible tokenizers and special tokens for both translation directions. |
| Transformer NMT model | Must | Implement and train an encoder-decoder Transformer with attention, masking, positional information and cross-attention. |
| Training pipeline | Must | Configurable training, validation, checkpoints, optimization, mixed precision where supported, and reproducible experiments. |
| Bidirectional translation | Must | Support English→Hindi and Hindi→English. |
| Advanced decoding | Must | Provide greedy decoding and beam search; expose decoding configuration in the application. |
| BLEU \+ chrF++ evaluation | Must | Evaluate against the official corpus test split and produce a reproducible analysis report. |
| FastAPI inference backend | Must | Expose versioned REST endpoints for translation, health, model metadata and relevant application functions. |
| React localization workspace | Must | Interactive translation UI with language selection, source/target editors, results, status and history. |
| Authentication / sessions | Must | Basic user authentication/session handling for protected application features. |
| Translation history | Must | Persist user translation activity and allow retrieval of previous translations. |
| Experiment/model registry | Should | Track model version, tokenizer, configuration, dataset version and evaluation metrics. |
| Evaluation dashboard | Should | Visualize BLEU, chrF++, loss curves, direction-wise results and experiment comparisons. |
| Translation quality insights | Should | Surface confidence/quality-related metadata and representative error examples without claiming semantic certainty. |
| Deployment & observability | Should | Containerized deployment, health checks, structured logs and production-style configuration. |
| Notifications | Could | Optional training/evaluation completion or system-status notifications. |

# **Target Users**

* Localization specialist — translates and reviews English/Hindi content efficiently.  
* Content platform user — generates translations through an interactive application.  
* ML/NLP engineer — trains, evaluates, compares and versions translation models.  
* Project reviewer/mentor — inspects model quality, experiments, architecture, testing and reproducibility.  
* Administrator — manages users, sessions and system/model configuration.

**Persona Pain Points**

* Localization specialist — manual comparison and preservation of meaning/style across English and Hindi is time-consuming.  
* Content platform user — needs fast, consistent translations without leaving the application.  
* ML/NLP engineer — needs reproducible experiments, model versions and objective evaluation instead of opaque outputs.  
* Project reviewer/mentor — needs visible evidence of engineering depth, measurable improvement and reproducibility.  
* Administrator — needs controlled access, reliable service behavior and traceable user activity.

# **User Stories**

**US-001**

As a localization user, I want to select English↔Hindi translation direction so that I can translate content between both supported languages.

**Acceptance criteria**

* Both directions are available.  
* The selected source and target language are clearly displayed.  
* Input is validated against the selected source language where applicable.

**US-002**

As a localization user, I want to enter or paste text and receive a generated translation so that I can use the system as a translation workspace.

**Acceptance criteria**

* Source text can be submitted through the UI.  
* The generated translation is displayed with clear request status.  
* API/network/model failures produce a user-readable error state.

**US-003**

As an ML engineer, I want the system to use a custom Transformer encoder-decoder model so that the project demonstrates the core Generative AI architecture rather than relying only on an external translation API.

**Acceptance criteria**

* Encoder-decoder Transformer is implemented and trainable.  
* Self-attention, cross-attention, masking and positional information are implemented correctly.  
* A saved checkpoint can be loaded independently for inference.

**US-004**

As an ML engineer, I want a reproducible corpus preprocessing and tokenization pipeline so that training and evaluation can be repeated consistently.

**Acceptance criteria**

* Corpus version/source is documented.  
* Official train/validation/test split is preserved.  
* Preprocessing and tokenizer configuration are saved with experiments.

**US-005**

As an ML engineer, I want to train and validate multiple model configurations so that I can tune the translation system based on measurable results.

**Acceptance criteria**

* Training configuration is externalized.  
* Training/validation metrics are recorded.  
* Best checkpoints can be identified and restored.

**US-006**

As a localization user, I want beam search and configurable decoding so that I can compare different generation strategies.

**Acceptance criteria**

* Greedy decoding is supported.  
* Beam search is supported.  
* Decoding parameters are validated and applied by the inference service.

**US-007**

As a reviewer, I want BLEU and chrF++ scores on the official test split so that I can objectively evaluate translation quality.

**Acceptance criteria**

* Both metrics are computed with a reproducible evaluation script.  
* Results are reported separately for English→Hindi and Hindi→English where applicable.  
* The report includes methodology, configuration and representative examples.

**US-008**

As a user, I want my translation history to be available so that I can revisit previous work.

**Acceptance criteria**

* Authenticated users can view their own history.  
* History records source text, generated translation, direction, model version and timestamp.  
* Users cannot access another user's history.

**US-009**

As an ML engineer, I want model/version metadata so that every generated translation can be associated with a known model and tokenizer.

**Acceptance criteria**

* Active model version is identifiable.  
* Checkpoint/tokenizer/configuration metadata are recorded.  
* The API exposes model metadata for the UI.

**US-010**

As a reviewer, I want experiment and evaluation results visible in a dashboard so that I can understand how the system evolved.

**Acceptance criteria**

* Training/evaluation results can be compared.  
* BLEU/chrF++ and loss metrics are presented clearly.  
* Dataset/model configuration is visible for each experiment.

**US-011**

As an API consumer, I want documented REST endpoints so that the translation service can be tested and integrated independently of the React frontend.

**Acceptance criteria**

* FastAPI provides versioned endpoints.  
* Request/response schemas are validated.  
* OpenAPI documentation is available.  
* Automated API tests cover success and failure paths.

**US-012**

As a project reviewer, I want a reproducible deployment and README so that I can run and verify the system.

**Acceptance criteria**

* Setup steps are documented.  
* Architecture diagram and data flow are documented.  
* Model/evaluation reproduction steps are documented.  
* Tests and expected commands/results are documented.

# **Functional Requirements**

* FR-001 — Translation: accept source text, selected direction and decoding configuration; return generated target text and model metadata.  
* FR-002 — Directions: support English→Hindi and Hindi→English through a consistent inference interface.  
* FR-003 — Preprocessing: normalize and tokenize source/target sequences consistently with training.  
* FR-004 — Model loading: load a selected trained checkpoint and tokenizer without retraining.  
* FR-005 — Decoding: support greedy and beam-search generation with bounded generation length.  
* FR-006 — Evaluation: run BLEU and chrF++ on the official test split and persist/report results.  
* FR-007 — History: store authenticated translation requests/results and associated metadata.  
* FR-008 — Model metadata: expose active model version, tokenizer version and relevant configuration.  
* FR-009 — Authentication: implement basic registration/login/session or token-based access for protected features.  
* FR-010 — API reliability: validate inputs, return structured errors, provide health/readiness endpoints, and log relevant events.  
* FR-011 — Frontend: provide responsive translation, history and evaluation/model views.  
* FR-012 — Testing: unit-test core preprocessing/inference utilities and integration-test critical API workflows.

# **Generative AI / Model Requirements**

* The central AI component is an autoregressive sequence-to-sequence Transformer, treated as the project's Generative AI model.  
* Architecture must be grounded in Vaswani et al., “Attention Is All You Need” (NeurIPS 2017).  
* The implementation must include token embeddings, positional representation, multi-head self-attention, feed-forward networks, residual connections, normalization, decoder causal masking and encoder-decoder cross-attention.  
* Training must use teacher forcing with an appropriate token-level objective such as cross-entropy; label smoothing should be evaluated as an improvement rather than assumed beneficial.  
* Optimization should use Adam/AdamW with a documented learning-rate strategy; gradient clipping and mixed precision may be enabled where hardware supports it.  
* Checkpointing must preserve model state, optimizer/scheduler state where required, tokenizer information and experiment configuration.  
* Inference must be deterministic/reproducible where requested and must record decoding configuration.  
* Advanced model work should include controlled experiments/ablations such as model size, dropout, label smoothing, tokenizer choice or decoding strategy, with conclusions based on evaluation results.  
* The project must distinguish model-generated output from ground-truth references and must not claim that BLEU/chrF++ alone proves semantic correctness.

# **Data Requirements**

* Primary dataset: IIT Bombay English-Hindi Corpus.  
* The official corpus split must be used for final evaluation; no random replacement split should be substituted for the official test set.  
* Data pipeline must include validation, duplicate/invalid record handling, normalization and length statistics.  
* Tokenizer and preprocessing artifacts must be versioned/reproducible.  
* Training, validation and test data must remain isolated to prevent evaluation leakage.  
* Dataset provenance, license/usage constraints and acquisition steps must be documented in the README.

# **Evaluation & Quality Requirements**

* Primary metrics: BLEU and chrF++.  
* Final metrics must be calculated on the official corpus test split.  
* Evaluation must be reported by translation direction.  
* Include baseline-vs-final comparison and at least one controlled ablation/experiment.  
* Include qualitative examples covering successful translations and common failure patterns.  
* Report training/validation loss and relevant training configuration.  
* Evaluation scripts must be runnable independently from the web application.  
* Use a standard, reproducible metric implementation such as sacreBLEU and record the metric configuration/version.

# **System Architecture**

The platform follows a layered architecture: React frontend → FastAPI REST API → translation/application services → Transformer inference engine and persistence layer. The offline ML pipeline handles corpus preparation, tokenization, training, checkpointing and evaluation. The same versioned model/tokenizer artifacts are then loaded by the inference service.

* Frontend: React, Tailwind CSS, Axios.  
* Backend: Python, FastAPI, Pydantic, authentication/session layer.  
* ML: Python, PyTorch, custom Transformer components, tokenization utilities.  
* Evaluation: sacreBLEU and chrF++ tooling.  
* Persistence: relational database or equivalent structured store for users, history and model/experiment metadata.  
* Deployment: containerized services with environment-based configuration, health checks and reproducible startup.

# **Product States**

TRAINING\_DATA\_READY → BASELINE\_TRAINED → MODEL\_TRAINED → MODEL\_EVALUATED → MODEL\_REGISTERED → API\_READY → UI\_READY → INTEGRATION\_TESTED → DEPLOYED

Runtime translation flow: REQUEST\_RECEIVED → VALIDATED → TOKENIZED → GENERATING → DECODED → RESPONSE\_READY

Failure states: DATA\_VALIDATION\_FAILED, TRAINING\_FAILED, CHECKPOINT\_LOAD\_FAILED, TOKENIZATION\_FAILED, INFERENCE\_FAILED, API\_ERROR

# **Security & Reliability Requirements**

* Authentication/session information must not be exposed to other users.  
* Users may access only their own translation history unless an explicit administrative role is implemented.  
* Secrets and credentials must be supplied through environment configuration, not committed to source control.  
* API inputs must be validated and bounded to prevent oversized or pathological requests.  
* Model inference failures must return structured API errors without exposing internal stack traces.  
* Health/readiness checks must distinguish application availability from model readiness where practical.  
* Logging must avoid unnecessarily storing sensitive user text in unrestricted logs.

# **Testing Requirements**

* Unit tests for preprocessing and normalization.  
* Unit tests for tokenizer input/output and special-token handling.  
* Unit tests for attention/masking or core model utilities where practical.  
* Inference tests using a fixed checkpoint and known sample inputs.  
* API integration tests for authentication, translation, validation, history and health endpoints.  
* Frontend integration tests for critical translation workflow where tooling permits.  
* Regression tests to ensure model/API changes do not silently break the inference pipeline.  
* A documented test command and expected pass criteria.

**Risks & Mitigations**

| Risk | Probability | Impact | Mitigation | Trigger / Response |
| :---- | :---- | :---- | :---- | :---- |
| GPU / training-time constraints | High | High | Use configurable model sizes, mixed precision, gradient accumulation, checkpointing, early stopping and resumable training. Establish a smaller baseline first. | If training exceeds available compute/time, reduce configuration while preserving the evaluation protocol. |
| Low BLEU / chrF++ | Medium | High | Run baseline, controlled tuning, decoding comparisons and error analysis; do not overfit the test set. | If targets are missed, report the strongest reproducible model and documented failure analysis. |
| Hindi tokenization / normalization | Medium | High | Validate Unicode normalization, Indic text handling, subword coverage and sequence-length statistics before full training. | Inspect preprocessing/tokenizer coverage before retraining. |
| Evaluation leakage | Low | Critical | Keep official train/validation/test split isolated; version data artifacts and prohibit test-set tuning. | Invalidate and rerun any experiment with detected overlap/leakage. |
| Overfitting / poor generalization | Medium | High | Monitor validation metrics, checkpoint best models, regularize and use early stopping. | Restore best validation checkpoint and investigate degradation. |
| Inference latency / memory | Medium | Medium | Load model once, bound generation length, optimize decoding and expose readiness checks. | Profile tokenization/decoding and optimize before scaling infrastructure. |

# **Documentation & Delivery Requirements**

* README with prerequisites, installation, environment configuration and run instructions.  
* Architecture diagram showing frontend, backend, model service, storage and training/evaluation pipeline.  
* Model architecture documentation with tensor/data flow and major hyperparameters.  
* Dataset preparation and official split documentation.  
* Training reproduction instructions including configuration and checkpoint handling.  
* Evaluation report containing BLEU, chrF++, methodology, baseline/final comparison, ablation results and error analysis.  
* API documentation with request/response examples.  
* Testing instructions and deployment instructions.  
* Final demonstration covering the model, evaluation evidence, API and interactive application.

# **Project Phases**

**Phase 1 — Foundation & Baseline**

Study the grounding paper and Transformer encoder-decoder fundamentals; scaffold the FastAPI backend and React frontend shell; acquire and explore the IIT Bombay English-Hindi Corpus; establish preprocessing and tokenization; build and validate a baseline translation pipeline.

**Phase 2 — Model Training, Evaluation & Product Build**

Train and tune the full Transformer encoder-decoder model; add bidirectional translation and advanced decoding; expose inference through FastAPI; run BLEU/chrF++ evaluation; implement model/checkpoint metadata; build React translation, history and evaluation screens.

**Phase 3 — Integration, Hardening & Delivery**

Complete frontend-backend integration; add authentication/session handling, tests, logging and error handling; polish UI/UX; complete experiment comparison and error analysis; finalize evaluation against grading criteria; deploy the application; prepare README, architecture documentation, demo and final presentation.

**Success Metrics & Target Thresholds**

Targets are engineering/viva benchmarks, not claims of state-of-the-art performance. Final results must record the exact split, tokenizer, model configuration and metric implementation/version.

| Metric | Direction | Minimum Acceptable Target | Stretch Target |
| :---- | :---- | :---- | :---- |
| BLEU | English → Hindi | ≥ 20 | ≥ 25 |
| chrF++ | English → Hindi | ≥ 45 | ≥ 50 |
| BLEU | Hindi → English | ≥ 22 | ≥ 28 |
| chrF++ | Hindi → English | ≥ 50 | ≥ 55 |
| Baseline → Final BLEU | Both directions | Measurable improvement | ≥ 10% relative improvement |
| Critical API tests | Backend | 100% pass | 100% pass \+ regression suite |

# **Acceptance / Definition of Done**

* A trained Transformer-based English↔Hindi model can be loaded and used for real inference without retraining.  
* Both translation directions are available through the application.  
* BLEU and chrF++ are reported on the official corpus test split with reproducible evaluation.  
* A baseline and improved model/experiment comparison is documented.  
* FastAPI exposes a tested translation API and relevant supporting endpoints.  
* React provides a usable localization workspace connected to the backend.  
* Authentication/session handling and user-specific translation history work correctly.  
* Model/tokenizer/version metadata is traceable for generated translations.  
* Unit/integration tests cover the critical ML and application pipeline.  
* README, architecture diagram, reproduction steps, evaluation report and final demo materials are complete.  
* The system is deployable as a coherent end-to-end application.

Product lifecycle: PROJECT\_INITIALIZED → DATA\_READY → BASELINE\_TRAINED → MODEL\_TRAINED → MODEL\_EVALUATED → MODEL\_REGISTERED → API\_READY → UI\_READY → DEPLOYED

Translation request lifecycle: DRAFT → VALIDATING → TOKENIZING → TRANSLATING → DECODING → COMPLETED

Translation failure lifecycle: Any runtime stage → FAILED → ERROR\_REPORTED

Experiment lifecycle: CONFIGURED → DATA\_READY → TRAINING → VALIDATING → EVALUATING → REGISTERED

Experiment failure lifecycle: Any experiment stage → FAILED → DIAGNOSTICS\_RECORDED

# **Product Principles**

* Model before wrapper — the core generative Transformer is the central engineering artifact, not an external API hidden behind a UI.  
* Evidence before claims — translation quality claims must be supported by reproducible BLEU/chrF++ evaluation and qualitative analysis.  
* Reproducibility by default — dataset split, preprocessing, tokenizer, configuration, checkpoint and evaluation setup must be traceable.  
* Experiment scientifically — improvements should be demonstrated through controlled comparisons rather than assumed.  
* Production-minded engineering — API contracts, authentication, testing, observability, error handling and deployment are first-class requirements.  
* Human-readable output — the localization workspace should make generated translations easy to inspect, copy, compare and review.  
* Clear uncertainty — automated metrics are evidence of system performance, not a guarantee of perfect semantic correctness.  
* Every output is traceable — generated text should be associated with translation direction, model/version and decoding context where practical.

# **Grounding & References**

Paper: Vaswani et al., “Attention Is All You Need,” NeurIPS 2017\.

Primary dataset: IIT Bombay English-Hindi Corpus.

Technology stack: PyTorch, FastAPI, React, Tailwind CSS, Axios, sacreBLEU, chrF++ tooling.

