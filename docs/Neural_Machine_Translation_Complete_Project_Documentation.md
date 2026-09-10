**Neural Machine Translation & Localization Platform**

*Complete OJT / Semester Project Documentation*  
*Generative AI • NLP / Machine Translation • English ↔ Hindi*

# **1\. Project Overview**

## **Project Identity**

| Attribute | Definition |
| :---- | :---- |
| Track | Generative AI |
| Domain | NLP / Neural Machine Translation / Localization |
| Project type | End-to-end applied Generative AI \+ ML engineering system |
| Languages | English and Hindi |
| Directions | English → Hindi and Hindi → English |
| Core model | Custom Transformer encoder-decoder |
| Grounding | Attention Is All You Need — NeurIPS 2017 |
| Dataset | IIT Bombay English-Hindi Corpus |
| ML | Python \+ PyTorch |
| Backend | FastAPI |
| Frontend | React \+ Tailwind CSS |
| HTTP | Axios |
| Evaluation | BLEU \+ chrF++ |
| Constraint | GPU / training compute availability |

## **Executive Summary**

The project is a Generative AI-based Neural Machine Translation platform that learns to generate English↔Hindi translations from parallel training data. The central AI artifact is a Transformer encoder-decoder implemented/configured for this project and grounded in the architecture introduced by Vaswani et al. in “Attention Is All You Need.” The model is trained on the IIT Bombay English-Hindi Corpus and evaluated on the official corpus split.

The trained model is then integrated into a production-oriented application. FastAPI provides the inference API and React provides an interactive localization workspace. The system also includes decoding strategies, model/experiment tracking, translation history, authentication, evaluation reporting, testing, observability and deployment documentation.

## **Core Product Flow**

IITB corpus → validation/cleaning → normalization → subword tokenization → official train/validation/test split → Transformer training → checkpointing → evaluation → model registration → FastAPI inference → React localization workspace → translation history/evaluation/model information.

## **MVP**

* IIT Bombay corpus acquisition and validation.  
* Official split preservation and reproducible preprocessing.  
* Subword tokenizer pipeline.  
* Custom Transformer encoder-decoder in PyTorch.  
* Training/validation/checkpoint pipeline.  
* English→Hindi and Hindi→English support.  
* Greedy and beam-search decoding.  
* BLEU and chrF++ evaluation.  
* FastAPI inference and health endpoints.  
* React translation workspace.  
* Basic authentication/session handling and translation history.  
* Unit/integration tests, README, architecture documentation and deployment.

## **Out of Scope**

* Training a foundation-scale LLM from scratch.  
* Claiming state-of-the-art performance without evidence.  
* Using Google Translate as the core translation engine.  
* Training/tuning on the final test set.  
* Guaranteeing semantic correctness from automatic metrics.  
* Uncontrolled autonomous retraining from user content.

# **2\. Business Requirements Document (BRD)**

## **Problem Statement**

Localization teams and multilingual content platforms need to translate large amounts of English and Hindi content. The project addresses both the machine-learning problem of learning translation from aligned bilingual data and the engineering problem of delivering the trained generative model through a reliable, measurable and usable application.

## **Vision**

Build a reproducible, measurable and production-oriented Generative AI localization workspace that learns English↔Hindi translation from parallel corpus data and exposes the resulting model through an interactive application.

## **Objectives**

* Implement the core Transformer NMT architecture instead of only wrapping a third-party translation API.  
* Train a generative sequence-to-sequence model using the IIT Bombay corpus.  
* Support both translation directions.  
* Develop robust English/Hindi preprocessing and subword tokenization.  
* Compare baseline and improved configurations through controlled experiments.  
* Evaluate with BLEU and chrF++ on the official test split.  
* Compare greedy and beam-search generation.  
* Serve the selected model through versioned FastAPI APIs.  
* Provide React localization, history and evaluation/model views.  
* Maintain reproducibility, security, testing and deployment evidence.

## **Personas**

| Persona | Goals | Pain Points |
| :---- | :---- | :---- |
| Localization Specialist | Translate/review content efficiently | Manual bilingual comparison is repetitive and time-consuming. |
| Content Platform User | Get consistent translations quickly | Needs translation without leaving the workflow. |
| ML/NLP Engineer | Train and improve the model | Needs reproducible experiments, versions and meaningful metrics. |
| Project Reviewer/Mentor | Assess technical depth | Needs measurable results and defensible architecture. |
| Administrator | Control access and reliability | Needs user isolation, health checks and traceability. |

## **Business Requirements**

| ID | Requirement | Priority | Acceptance |
| :---- | :---- | :---- | :---- |
| BR-001 | Select translation direction | Must | EN→HI and HI→EN available |
| BR-002 | Submit source text | Must | Validated text accepted |
| BR-003 | Generate translation | Must | Loaded trained checkpoint returns output |
| BR-004 | Advanced decoding | Must | Greedy and beam search work |
| BR-005 | Evaluate model | Must | BLEU/chrF++ on official test split |
| BR-006 | Reproduce experiments | Must | Dataset/tokenizer/config/model recorded |
| BR-007 | Trace model version | Must | Output identifies model/version |
| BR-008 | Persist history | Must | User sees only own records |
| BR-009 | Evaluation dashboard | Should | Metrics/results are viewable |
| BR-010 | Health/readiness | Should | API and model readiness are distinguishable |

## **Non-Functional Requirements**

* Reproducibility of preprocessing, training and evaluation.  
* Bounded input and generation length.  
* Model loaded once per service process.  
* Structured API errors.  
* Authentication for protected features.  
* User-specific history isolation.  
* Versioned model artifacts.  
* Basic latency/resource observability.  
* Responsive and accessible UI.

## **Success Metrics**

| Metric | Direction | Minimum | Stretch |
| :---- | :---- | :---- | :---- |
| BLEU | English→Hindi | ≥20 | ≥25 |
| chrF++ | English→Hindi | ≥45 | ≥50 |
| BLEU | Hindi→English | ≥22 | ≥28 |
| chrF++ | Hindi→English | ≥50 | ≥55 |
| Baseline→Final BLEU | Both | Measurable improvement | ≥10% relative |
| Critical API tests | Backend | 100% pass | 100% \+ regression |
| Reproducibility | ML | Checkpoint restore \+ evaluation | Full config/tokenizer/checkpoint traceability |

These are predefined engineering/viva benchmarks, not claims of state-of-the-art quality. Final results must record the exact split, configuration and metric/tool version.

## **Risks**

| Risk | Probability | Impact | Mitigation |
| :---- | :---- | :---- | :---- |
| GPU/training-time limits | High | High | Small baseline first; mixed precision, gradient accumulation, checkpoint/resume and configurable model size. |
| Low BLEU/chrF++ | Medium | High | Baseline comparison, controlled experiments, tokenization checks, regularization and error analysis. |
| Hindi Unicode/tokenization issues | Medium | High | Validate normalization, subword coverage and sequence statistics. |
| Evaluation leakage | Low | Critical | Keep official test split isolated; freeze configuration before final test evaluation. |
| Overfitting | Medium | High | Validation monitoring, best-checkpoint selection, regularization and early stopping. |
| Inference latency/memory | Medium | Medium | Load once, bound generation, profile decoding and optimize deployment. |
| Checkpoint/tokenizer mismatch | Low | High | Version and validate model/tokenizer/config as a single artifact set. |

## **Product States**

Translation request: DRAFT → VALIDATING → TOKENIZING → TRANSLATING → DECODING → COMPLETED

Translation failure: Any stage → FAILED → ERROR\_REPORTED

Experiment: CONFIGURED → DATA\_READY → TRAINING → VALIDATING → EVALUATING → REGISTERED

Experiment failure: Any stage → FAILED → DIAGNOSTICS\_RECORDED

# **3\. Product Requirements Document (PRD)**

## **Product Goal**

Create a professional Generative AI localization workspace that trains and serves a custom Transformer-based English↔Hindi neural machine translation model, provides rigorous NLP evaluation and experiment/model tracking, and delivers an interactive full-stack translation experience.

## **Feature Priorities**

| Feature | Priority |
| :---- | :---- |
| Corpus ingestion/preprocessing | Must |
| Subword tokenization | Must |
| Custom Transformer NMT | Must |
| Training/validation/checkpoints | Must |
| English↔Hindi translation | Must |
| Greedy \+ beam decoding | Must |
| BLEU \+ chrF++ | Must |
| FastAPI inference | Must |
| React localization workspace | Must |
| Authentication/session | Must |
| Translation history | Must |
| Model/version registry | Should |
| Evaluation dashboard | Should |
| Experiment comparison | Should |
| Error analysis | Should |
| Deployment/observability | Should |
| Notifications | Could |

## **User Stories**

**US-001**

As a localization user, I want to select English↔Hindi direction so that I can translate between both languages.

**Acceptance criteria**

* Both directions selectable.  
* Direction is visible.

**US-002**

As a user, I want to submit text and receive generated translation so that I can use the platform for localization.

**Acceptance criteria**

* Input validated.  
* Output/status displayed.  
* Failure handled clearly.

**US-003**

As an ML engineer, I want a custom Transformer encoder-decoder so that the project demonstrates core Generative AI architecture.

**Acceptance criteria**

* Attention/masking components implemented.  
* Checkpoint can be restored.

**US-004**

As an ML engineer, I want reproducible preprocessing/tokenization so that experiments can be repeated.

**Acceptance criteria**

* Official split preserved.  
* Tokenizer/config recorded.

**US-005**

As an ML engineer, I want controlled experiments so that model improvements are evidence-based.

**Acceptance criteria**

* Configuration and metrics recorded.  
* Best checkpoint identifiable.

**US-006**

As a user, I want greedy and beam decoding so that generation strategies can be compared.

**Acceptance criteria**

* Both strategies work.  
* Parameters validated/bounded.

**US-007**

As a reviewer, I want BLEU and chrF++ so that quality is objectively measurable.

**Acceptance criteria**

* Official test split used.  
* Both directions reported.

**US-008**

As a user, I want translation history so that I can revisit work.

**Acceptance criteria**

* History contains text/direction/time/model.  
* Access is user-scoped.

**US-009**

As an ML engineer, I want model metadata so generated output is traceable.

**Acceptance criteria**

* Model/tokenizer versions are identifiable.

**US-010**

As a reviewer, I want experiment comparison and error analysis so that I can understand model evolution.

**Acceptance criteria**

* Baseline/final and controlled experiment evidence exists.

**US-011**

As an API consumer, I want documented REST endpoints so that the model is independently usable.

**Acceptance criteria**

* Versioned API, schemas and tests exist.

**US-012**

As a reviewer, I want reproducible deployment documentation so that I can verify the project.

**Acceptance criteria**

* README, architecture, setup, tests and evaluation steps exist.

## **Product Principles**

* Model before wrapper — the trained NMT model and pipeline are the central artifact.  
* Reproducibility before claims.  
* Unseen test data for final evaluation.  
* Evidence before improvement claims.  
* Generative output is not guaranteed truth.  
* Traceability by default.  
* Production-minded engineering.

# **4\. UX Requirements**

## **Information Architecture**

1. Login / Session  
2. Dashboard  
3. Translation Workspace  
4. Translation History  
5. Evaluation Dashboard  
6. Experiments  
7. Model Registry / Model Information  
8. Settings

## **Primary User Flow**

Login → select direction → enter text → choose decoding → validate → translate → generate/decode → display → save history → inspect model/evaluation.

## **Key Screens**

| Screen | Requirements |
| :---- | :---- |
| Login | Authentication, validation, loading/error states |
| Dashboard | Recent activity, model version, evaluation summary |
| Translation Workspace | Source/target panels, direction, decoding controls, model metadata, copy/save |
| History | User-specific translations, time, direction, model version |
| Evaluation | BLEU, chrF++, direction-wise results, baseline/final comparison |
| Experiments | Configuration, checkpoint, metrics, dataset/tokenizer version |
| Model Information | Active version, architecture, tokenizer, evaluation metrics |

# **5\. Technical Requirements Document (TRD)**

## **Architecture**

Frontend: React \+ Tailwind CSS \+ Axios  
Backend/API: Python \+ FastAPI \+ Pydantic  
AI/ML: Python \+ PyTorch \+ custom Transformer  
Evaluation: sacreBLEU \+ chrF++ tooling  
Persistence: structured database for users/history/experiments/model metadata  
Artifacts: versioned model checkpoints \+ tokenizer \+ configuration  
Deployment: containerized services appropriate to available compute

## **Technology Rationale**

* PyTorch provides control over Transformer implementation and training.  
* FastAPI keeps model serving in the Python runtime and provides typed REST/OpenAPI interfaces.  
* React provides reusable interactive localization components.  
* Tailwind enables consistent UI styling.  
* Axios provides explicit HTTP integration.  
* sacreBLEU/chrF++ provide reproducible translation evaluation.

## **Technical Requirements**

| ID | Requirement |
| :---- | :---- |
| TR-001 | Versioned REST API |
| TR-002 | Validated and bounded translation requests |
| TR-003 | Custom Transformer encoder-decoder |
| TR-004 | Reproducible tokenizer/preprocessing |
| TR-005 | Versioned checkpoints/model artifacts |
| TR-006 | Greedy \+ beam decoding |
| TR-007 | BLEU/chrF++ evaluation pipeline |
| TR-008 | Authenticated history persistence |
| TR-009 | Health/readiness endpoints |
| TR-010 | Unit/integration test suite |

## **NFR Targets**

* Non-generation API p95 target \<2 seconds in student deployment.  
* Generation length bounded.  
* Model loaded once per process.  
* Secrets outside source control.  
* Evaluation reproducible from documented configuration.

# **6\. High-Level Design (HLD)**

## **Architecture**

User → React Localization Workspace → FastAPI REST API → Translation Service → Tokenizer → Transformer Encoder/Decoder → Decoding Engine → Generated Translation → FastAPI → React.

Offline ML path: IITB Corpus → preprocessing → tokenizer → DataLoader → Transformer training → checkpoints → evaluation → model registry → inference service.

## **Components**

| Component | Responsibility |
| :---- | :---- |
| React UI | Translation, history, evaluation/model views |
| FastAPI | Auth, validation, translation, metadata, health |
| Translation Service | Preprocess/tokenize/inference orchestration |
| Tokenizer | Text↔subword ID conversion |
| Transformer | Generative sequence-to-sequence translation |
| Decoder | Greedy/beam generation |
| Evaluation | BLEU/chrF++ and reports |
| Database | Users, history, experiments, metadata |
| Artifact Store | Checkpoints, tokenizer and configs |
| Observability | Logs, latency, errors and versions |

## **End-to-End Data Flow**

9. Acquire and validate corpus.  
10. Normalize and tokenize.  
11. Preserve official train/validation/test split.  
12. Create batches and masks.  
13. Train Transformer and compute loss.  
14. Backpropagate and update weights.  
15. Track validation and checkpoints.  
16. Evaluate frozen candidates on isolated test data.  
17. Register model with tokenizer/configuration.  
18. FastAPI loads selected model.  
19. React sends validated translation request.  
20. Tokenizer converts text to IDs.  
21. Encoder produces contextual representations.  
22. Decoder autoregressively generates target tokens.  
23. Decoding strategy selects output sequence.  
24. Detokenize target IDs into text.  
25. Return translation plus relevant model/decoding metadata.  
26. Persist history for authenticated users.

## **Scalability & Reliability**

* Keep training and inference workloads separate.  
* Use bounded generation and batching where useful.  
* Introduce job queues for long-running evaluation/training if needed.  
* Load immutable model artifacts once.  
* Support rollback to a known-good model.  
* Use structured errors, timeouts and readiness checks.

# **7\. Generative AI / Transformer Architecture**

## **Why Generative AI?**

The model generates a new target-language sequence. It predicts the probability distribution of the next token conditioned on the source sentence and previously generated target tokens. Repeating this process generates the complete translation.

## **Why Encoder-Decoder?**

The encoder processes the source-language sequence and produces contextual representations. The decoder generates the target sequence token by token while attending to previously generated target tokens and to the encoder's source representations.

## **Encoder**

* Receives source token IDs.  
* Uses token embeddings and positional information.  
* Uses multi-head self-attention to model source-token relationships.  
* Uses feed-forward networks.  
* Uses residual connections and normalization.  
* Produces contextual source representations.

## **Decoder**

* Receives previously generated target tokens.  
* Uses masked self-attention to prevent seeing future target tokens.  
* Uses cross-attention to access encoder representations.  
* Uses feed-forward layers, residual connections and normalization.  
* Projects representations to vocabulary logits.  
* Generates tokens until EOS or maximum length.

## **Attention**

Scaled dot-product attention uses Query (Q), Key (K) and Value (V): Attention(Q,K,V) \= softmax(QKᵀ / √dₖ)V. Multi-head attention performs several learned attention projections in parallel.

## **Positional Information**

A Transformer does not inherently encode sequence order through recurrence, so positional information is added to token representations to distinguish positions.

## **Masks**

* Padding masks stop padded positions from affecting attention.  
* Causal/look-ahead masks stop the decoder from using future target tokens during training.

## **Cross-Attention**

Cross-attention connects decoder generation to the encoded source. It allows the decoder to determine which source representations are relevant while predicting each target token.

## **Training**

Training uses teacher forcing: the correct previous target tokens are supplied while the model learns to predict the next token. Cross-entropy is the primary token-level objective. Adam/AdamW, learning-rate scheduling, gradient clipping, checkpointing, validation and early stopping are part of the training design.

## **Inference**

At inference the target sentence is unknown. The decoder starts with a beginning-of-sequence token, predicts a target token, feeds that generated token back into the decoder, and repeats until EOS or the maximum length.

## **Decoding**

| Method | Description | Role |
| :---- | :---- | :---- |
| Greedy | Select highest-probability next token at each step | Fast baseline |
| Beam Search | Maintain multiple candidate sequences and select the strongest completed sequence | Explore alternatives / quality-latency comparison |

## **What 'Custom Transformer' Means**

It means the project implements/configures and trains the Transformer NMT pipeline for English-Hindi translation instead of simply calling a completed external translation service. It does not claim that the project invented the Transformer. The project contribution is adapting the architecture, training it, experimenting with it, evaluating it and deploying it.

## **Advanced Model Experiments**

* Baseline versus improved model.  
* Model depth/width experiments.  
* Dropout/regularization experiments.  
* Tokenizer/subword experiments.  
* Label smoothing evaluation.  
* Greedy versus beam search comparison.  
* Sentence-length and rare-token analysis.  
* Translation error analysis.  
* Checkpoint comparison and rollback.

# **8\. Data & Evaluation Design**

## **Dataset**

Primary dataset: IIT Bombay English-Hindi Corpus. The official train/validation/test split is preserved for final evaluation. Dataset provenance, acquisition and usage constraints must be documented.

## **Data Pipeline**

27. Acquire and validate corpus.  
28. Check source-target alignment.  
29. Normalize Unicode/text safely.  
30. Handle invalid or duplicate records under documented rules.  
31. Train/configure reproducible subword tokenizer.  
32. Convert text to token IDs.  
33. Create padded batches and masks.  
34. Keep test data isolated.

## **Evaluation**

* BLEU measures n-gram overlap with reference translations.  
* chrF++ uses character/word n-gram information and can capture useful morphological/spelling similarities.  
* Report both metrics for both directions.  
* Compare baseline and final model.  
* Perform at least one controlled experiment/ablation.  
* Include qualitative examples and error categories.  
* Record exact evaluation tool/configuration version.

## **Leakage Controls**

* Never train on test examples.  
* Do not select hyperparameters using test data.  
* Keep final test data isolated from training/validation code.  
* Freeze final configuration before final test evaluation.  
* Apply the same documented preprocessing consistently.

# **9\. Database / Data Design**

| Entity | Key Fields |
| :---- | :---- |
| users | id, name, email, role, auth/session metadata, createdAt |
| translationRequests | id, userId, sourceLanguage, targetLanguage, sourceText, outputText, decodingStrategy, modelVersion, timestamp, status |
| models | id, modelName, modelVersion, artifactRef, tokenizerVersion, configuration, metrics, status |
| experiments | id, name, datasetVersion, tokenizerVersion, configuration, training/validation/test metrics, checkpointRef, status |
| evaluationRuns | id, modelVersion, datasetVersion, direction, BLEU, chrF++, configuration, createdAt |
| auditEvents | id, userId, action, metadata, timestamp |

# **API Specification**

Base URL: /api/v1

| Method | Endpoint | Purpose |
| :---- | :---- | :---- |
| POST | /auth/login | Authenticate |
| POST | /auth/logout | End session |
| POST | /translations | Generate translation |
| GET | /translations | List own history |
| GET | /translations/{id} | Retrieve translation |
| GET | /models/active | Active model metadata |
| GET | /models | Registered models |
| GET | /experiments | Experiments |
| GET | /evaluations | Evaluation results |
| GET | /health | Application health |
| GET | /ready | Model/service readiness |

## **Example Request**

{"sourceLanguage":"en","targetLanguage":"hi","text":"I am going to college.","decoding":{"strategy":"beam\_search","beamSize":4,"maxLength":64}}

## **Example Response**

{"translation":"मैं कॉलेज जा रहा हूँ।","sourceLanguage":"en","targetLanguage":"hi","modelVersion":"transformer-v1","decodingStrategy":"beam\_search","requestId":"req\_123"}

## **Standard Errors**

{"error":{"code":"INFERENCE\_FAILED","message":"Translation could not be generated.","requestId":"req\_123"}}

# **10\. Low-Level Design (LLD)**

## **Repository Structure**

neural-machine-translation-platform/  
├── docs/  
├── frontend/  
├── backend/  
├── ml/  
│   ├── data/  
│   ├── tokenization/  
│   ├── transformer/  
│   ├── training/  
│   ├── inference/  
│   ├── decoding/  
│   ├── evaluation/  
│   └── configs/  
├── models/  
├── experiments/  
├── evaluation/  
├── tests/  
├── infrastructure/  
└── scripts/

## **Key ML Interfaces**

| Interface | Responsibilities |
| :---- | :---- |
| TokenizerService | encode/decode, vocabulary and special-token validation |
| TranslationModel | load, forward, save, metadata |
| Decoder | greedy\_decode, beam\_search |
| TrainingService | train, validate, checkpoint, resume |
| EvaluationService | BLEU, chrF++, reports |
| TranslationService | validate, tokenize, infer, decode, return metadata |
| ModelRegistry | register, retrieve, activate, rollback |

## **Sequence**

React → POST /translations → FastAPI validation → TranslationService → tokenizer → encoder → decoder/cross-attention → decoding → detokenization → response → React → history persistence.

## **Design Patterns**

* Service layer for orchestration.  
* Strategy pattern for decoding strategies.  
* Repository pattern for persistence.  
* Dependency injection for API testability.  
* Adapter abstraction where multiple artifact/storage implementations are actually needed.  
* Avoid patterns without a concrete need.

# **11\. Security Design**

* Token/session authentication.  
* User-scoped authorization for history.  
* Input and generation bounds.  
* HTTPS in deployed environments.  
* Secrets in environment/secret management.  
* No credentials in source control.  
* Avoid logging full translation content by default.  
* CORS and secure headers configured appropriately.  
* Dependency scanning.  
* Immutable/versioned model artifacts after registration.

# **12\. Testing Strategy**

## **Test Levels**

35. Unit  
36. Integration  
37. API  
38. UI  
39. End-to-end  
40. Security  
41. Performance  
42. AI/ML evaluation

## **Sample Tests**

| ID | Scenario | Expected Result | Priority |
| :---- | :---- | :---- | :---- |
| TEST-001 | Valid translation request | Valid translation response | High |
| TEST-002 | Invalid language direction | Validation error | High |
| TEST-003 | Empty/oversized input | Rejected | High |
| TEST-004 | Tokenizer special tokens | Expected encode/decode | High |
| TEST-005 | Fixed checkpoint inference | Valid stable output | High |
| TEST-006 | Tokenizer/checkpoint mismatch | Load blocked | High |
| TEST-007 | Beam search | Valid bounded generation | Medium |
| TEST-008 | Unauthorized history | 403/denied | High |
| TEST-009 | Model unavailable | Graceful error | High |
| TEST-010 | Evaluation script | BLEU/chrF++ generated | High |
| TEST-011 | Health/readiness | Correct status | High |

## **CI/CD**

Feature branch → pull request → lint/unit tests → API/integration tests → dependency/security scan → build → container checks → staging/demo deployment → smoke tests.

## **Quality Gates**

* Required tests pass.  
* No secrets detected.  
* Model/tokenizer/configuration compatible.  
* Evaluation changes include evidence.  
* Deployment health checks pass.

# **13\. Observability**

## **Logs**

* Request ID  
* Translation status  
* Model version  
* Decoding strategy  
* Latency  
* Error code  
* Experiment/job ID

Do not log sensitive source/target content by default.

## **Metrics**

* API latency  
* Inference latency  
* Tokenization latency  
* Error rate  
* Model load time  
* Requests by direction  
* Generation length  
* Model version usage  
* CPU/GPU/memory utilization where available  
* Evaluation metrics

# **14\. Deployment Architecture**

Development → Testing → Staging/Demo → Production-like student deployment

* React as static web application.  
* FastAPI as container/service.  
* Versioned model artifacts.  
* Database for users/history/metadata.  
* GPU inference when resources justify it; smaller CPU-compatible model/configuration can be supported.  
* Training environment separated from inference where practical.

## **GPU Constraint Strategy**

* Establish a small baseline first.  
* Use mixed precision when supported.  
* Use gradient accumulation when memory is limited.  
* Checkpoint frequently and resume.  
* Keep model hyperparameters configurable.  
* Run cheap experiments before full runs.  
* Use CPU for preprocessing/evaluation where practical.  
* Choose model size appropriate for deployment.

## **Rollback**

Keep model artifacts versioned and activate a specific registered version. If a new model fails quality or operational checks, reactivate the previous known-good version.

# **15\. Cost / Resource Analysis**

For this project, training compute is the main resource constraint. Track GPU hours, memory utilization, training time, storage and hosting cost where applicable.

* Cache preprocessing/evaluation artifacts.  
* Avoid unnecessary full retraining.  
* Use smaller experiments before large runs.  
* Batch evaluation/inference where useful.  
* Bound generation length.  
* Keep only necessary model artifacts.

# **16\. 12-Week Project Roadmap**

| Phase | Weeks | Deliverables |
| :---- | :---- | :---- |
| Research & understanding | 1 | Transformer paper, NMT fundamentals, project scope |
| Requirements & design | 2 | BRD, PRD, UX, metrics, risks |
| Data foundation | 3 | IITB corpus, validation, split, normalization |
| Tokenizer \+ baseline | 4 | Tokenizer, DataLoader, baseline model |
| Transformer implementation | 5 | Attention, encoder, decoder, masks, training loop |
| Training | 6 | Validation, optimization, checkpoints, resume |
| Evaluation | 7 | BLEU/chrF++, experiments, error analysis |
| Advanced inference | 8 | Beam search, model registry |
| Backend | 9 | FastAPI, auth, history, model/evaluation APIs |
| Frontend | 10 | React workspace, history, evaluation/model views |
| Integration/testing | 11 | E2E, security, observability, regression |
| Deployment/final | 12 | Deployment, docs, final report, demo/viva |

# **17\. Team Responsibilities**

| Area | Owner 1 | Owner 2 | Shared |
| :---- | :---- | :---- | :---- |
| ML/NMT | Transformer/training | Data/tokenization/evaluation | Model decisions |
| Backend | API architecture | ML integration | Auth/testing |
| Frontend | Translation workspace | Evaluation/model views | UX/integration |
| Documentation | PRD/UX/API | ML/evaluation | HLD/LLD/README |
| Deployment | Frontend/API | Model/evaluation | CI/CD/security/observability |

Both students should understand and defend the complete architecture.

# **18\. Architecture Decision Records (ADRs)**

**ADR-001 — Custom Transformer**

Decision: Implement/train the core Transformer NMT pipeline.  
Rationale: Demonstrates actual GenAI engineering.  
Trade-off: Higher complexity and GPU demand.

**ADR-002 — FastAPI**

Decision: Use FastAPI as model-serving backend.  
Rationale: Python-native serving and validation.  
Trade-off: Backend remains Python-centric.

**ADR-003 — React Workspace**

Decision: Use React \+ Tailwind for product UI.  
Rationale: Interactive reusable localization workflow.  
Trade-off: Additional frontend scope.

**ADR-004 — BLEU \+ chrF++**

Decision: Use both metrics on official test split.  
Rationale: Complementary translation-quality evidence.  
Trade-off: Automatic metrics are imperfect.

**ADR-005 — Model Versioning**

Decision: Register checkpoint \+ tokenizer \+ config \+ metrics.  
Rationale: Traceability and rollback.  
Trade-off: Requires artifact metadata.

**ADR-006 — Bidirectional NMT**

Decision: Support English↔Hindi.  
Rationale: Broader capability and comparative evaluation.  
Trade-off: More training/evaluation work.

# **19\. Traceability Matrix**

| Business | Product | Technical | Component | Data/Artifact | Test |
| :---- | :---- | :---- | :---- | :---- | :---- |
| BR-001 | Direction | TR-001/002 | Translation API | Tokenizer/model | TEST-002 |
| BR-002 | Translation input | TR-002 | POST /translations | Requests | TEST-001/003 |
| BR-003 | Transformer generation | TR-003 | TranslationService | Checkpoint | TEST-005 |
| BR-004 | Decoding | TR-006 | Decoder | Decoding config | TEST-007 |
| BR-005 | Evaluation | TR-007 | EvaluationService | Evaluation runs | TEST-010 |
| BR-006 | Reproducibility | TR-004/005 | Training/Registry | Config/checkpoint | TEST-006 |
| BR-007 | Traceability | TR-005 | Model API | Model registry | TEST-005 |
| BR-008 | History | TR-008 | History API | Requests | TEST-008 |
| BR-010 | Readiness | TR-009 | Health/Ready | Operational data | TEST-011 |

# **20\. Interview / Viva Preparation**

## **Product Questions**

43. What is the project?  
44. What problem does it solve?  
45. Why is it Generative AI?  
46. Why English-Hindi?  
47. Why Transformer?  
48. What is custom about it?  
49. Why not use Google Translate?  
50. Who are the users?  
51. What are success targets?  
52. What are major risks?

## **Transformer Questions**

53. What is an encoder?  
54. What is a decoder?  
55. Why encoder-decoder?  
56. What is attention?  
57. What are Q/K/V?  
58. What is multi-head attention?  
59. What is cross-attention?  
60. What is masked attention?  
61. Why positional information?  
62. How does generation work?

## **Training/NLP**

63. What is tokenization?  
64. Why subwords?  
65. What is a parallel corpus?  
66. What is teacher forcing?  
67. What is cross-entropy?  
68. What is backpropagation?  
69. Why Adam/AdamW?  
70. What is overfitting?  
71. Why validation/test?  
72. What is leakage?

## **Evaluation**

73. What is BLEU?  
74. What is chrF++?  
75. Why both?  
76. Can BLEU guarantee correctness?  
77. What is your baseline?  
78. How will you prove improvement?  
79. What is an ablation?  
80. What errors will you analyze?

## **Architecture**

81. Why FastAPI?  
82. Why React?  
83. What happens after Translate is clicked?  
84. Why model versioning?  
85. How is history secured?  
86. How do you handle model failure?  
87. How do you deploy with limited GPU?  
88. How would you scale inference?

## **Scenario Questions**

89. GPU becomes unavailable. What do you do?  
90. BLEU is below target. What do you investigate?  
91. Hindi output has malformed tokens. What do you inspect?  
92. Tokenizer changes after training. Can the checkpoint still be used?  
93. Beam search improves quality but increases latency. What trade-off do you make?  
94. New model is worse than current model. How do you prevent deployment?  
95. API is healthy but model cannot load. What should readiness show?  
96. User sends an extremely long request. What happens?  
97. Test data accidentally enters training. What do you do?  
98. Two model versions differ. How do you decide which is better?

## **Strong Answer Principle**

For every major decision, explain WHAT was built, WHY it was chosen, HOW it works, HOW it is evaluated, and WHAT trade-off was accepted.

# **21\. Final Project Quality Score**

| Dimension | Score |
| :---- | :---- |
| Problem Clarity | /10 |
| UX | /10 |
| Technical Complexity | /10 |
| Architecture | /10 |
| Code Quality | /10 |
| Scalability | /10 |
| Security | /10 |
| Testing | /10 |
| Deployment | /10 |
| AI/ML Quality | /10 |
| Innovation | /10 |
| Documentation | /10 |
| Production Readiness | /10 |

1–3: Basic prototype; major gaps.  
4–6: Functional project with some engineering discipline.  
7–8: Strong product-company-style project.  
9–10: Exceptional engineering/product/research depth with strong evidence.

## **Distinction-Level Criteria**

* Clear NMT problem and localization use case.  
* Correct Transformer encoder-decoder implementation.  
* Strong attention/masking/training/generation understanding.  
* Reproducible corpus/tokenizer/training pipeline.  
* BLEU/chrF++ on isolated official test split.  
* Baseline-to-final improvement with controlled evidence.  
* Bidirectional translation and advanced decoding.  
* Production-oriented FastAPI service.  
* Interactive React localization workspace.  
* Model/tokenizer/experiment traceability.  
* Security, testing, deployment and observability.  
* Clear trade-offs and honest limitations.

# **22\. Final Project Summary**

Problem: Build an English↔Hindi neural translation system for localization.  
Solution: Train and evaluate a custom Transformer encoder-decoder on the IIT Bombay English-Hindi Corpus, then serve it through FastAPI and a React localization workspace.  
Core AI: Attention-based encoder-decoder with autoregressive generation.  
Evidence: BLEU \+ chrF++, baseline/final comparison, controlled experiments and error analysis.  
Engineering: Versioned models, API, authentication, history, testing, observability and deployment.  
Key distinction: The project is not merely a UI around an existing translation API; the trained and evaluated NMT model and reproducible ML pipeline are the central deliverables.