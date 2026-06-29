# Safety Router Golden Dataset Pipeline - Kế hoạch thực hiện

> **Ngày tạo:** 2026-06-17
> **Ngày cập nhật:** 2026-06-29
> **Trạng thái:** Step 8 - Phase 1 done, chờ Phase 2 trên Colab

---

## Mục lục
1. [Mục tiêu](#1-mục-tiêu)
2. [Trạng thái hiện tại](#2-trạng-thái-hiện-tại)
3. [Pipeline Flow](#3-pipeline-flow)
4. [Cấu trúc file](#4-cấu-trúc-file)
5. [API Configuration](#5-api-configuration)
6. [Scripts](#6-scripts)
7. [Usage](#7-usage)
8. [Ghi chú quan trọng](#8-ghi-chú-quan-trọng)
9. [Files](#9-files)
10. [Model Configuration](#10-model-configuration)
11. [Data Format](#11-data-format-json-fields)

---

## 1. Mục tiêu

Tạo golden dataset để train ML router cho Safety Router.

### Query Generation Models (sinh queries - chạy CẢ 3):
- `MiniMax-M2.7` (PRIMARY - MiniMax API)
- `DeepSeek-V4-Pro` (Alibaba API)
- `qwen3-next-80b-a3b-thinking` (Alibaba API)

### Response Generation Models (sinh response - 2 models):
- **local**: `Qwen/Qwen3-4B-Instruct-2507` (Colab GPU)
- **high**: `gemini-3.1-flash-lite` (API call)

**Dataset size:**
- 330 queries × 3 query-models × 2 response-models × 2 languages = **3,960 responses**

---

## 2. Trạng thái hiện tại

### ✅ Đã hoàn thành

| Thành phần | Status | File/Ghi chú |
|------------|--------|--------------|
| Policy CSV parsing | ✅ Hoàn thành | `policy.csv` (12 policies) |
| `.env` configuration | ✅ Hoàn thành | API keys đầy đủ |
| Smoke test | ✅ Hoàn thành | `test_api.py` |
| MiniMax API | ✅ Hoạt động | 1M tokens free |
| Alibaba API | ✅ Hoạt động | Backup only |
| Pipeline cleanup | ✅ Hoàn thành | Đã xóa template-based, giữ LLM-based |
| Separate scripts | ✅ Hoàn thành | 6 scripts riêng biệt |
| Steps 1-7 (VNI only) | ✅ Hoàn thành | Queries → responses → judge → review → difficulty → augmentation → split |
| Phase 1: Convert to routing format | ✅ Hoàn thành | 874 queries, 1,748 routing rows, `--skip-embeddings` |
| llm_data.json + mfrouter.yaml | ✅ Hoàn thành | Config trỏ artifacts/routing/, text_dim=1024 |

### 🔴 Cần làm

| # | Task | Priority | Ghi chú |
|---|------|----------|---------|
| 9 | Phase 2: Generate embeddings (Colab) | Cao | Alibaba text-embedding-v3, 874 queries |
| 10 | Train MFRouter (Colab) | Cao | Notebook/Colab |
| 11 | Evaluate MFRouter | Trung | Test accuracy, confusion matrix |

---

## 3. Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│           STEP 1: Query Generation (3 MODELS - SEPARATE RUNS)    │
│  Script: scripts/generate_queries.py                             │
│  Language: vi (Vietnamese) or eng (English)                      │
│  Models (run SEPARATELY - each generates 330 queries):           │
│      1. MiniMax-M2.7 (MiniMax API)                              │
│      2. DeepSeek-V4-Pro (Alibaba API)                            │
│      3. qwen3-next-80b-a3b-thinking (Alibaba API)                │
│  Backup Models (if primary fails):                              │
│      - qwen3.6-plus (Alibaba API)                                │
│      - qwen3.7-max (Alibaba API)                                 │
│  Output per model per language:                                  │
│      - queries_minimax_vni.jsonl (330 queries)                   │
│      - queries_deepseek_vni.jsonl (330 queries)                  │
│      - queries_qwen_vni.jsonl (330 queries)                      │
│      (same for eng)                                              │
│  Total: 330 queries × 3 models × 2 languages = 1,980 queries   │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│     STEP 2: Response Generation (Qwen3-4B - COLAB)               │
│  Script: scripts/run_local_generation.py                         │
│  Model: Qwen/Qwen3-4B-Instruct-2507                            │
│  Input: ALL 6 query files (upload to Colab)                     │
│  Note: Only TARGET policies (query.policy_ids) are passed     │
│        to prompt - not all 12 policies                          │
│  Output: answers_{model}_local_{lang}.jsonl                     │
│      - answers_minimax_local_vni.jsonl                          │
│      - answers_deepseek_local_vni.jsonl                         │
│      - answers_qwen_local_vni.jsonl                             │
│      (same pattern for eng)                                      │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│         STEP 3: Response Generation (Gemini - API)               │
│  Script: scripts/run_gemini_generation.py                         │
│  Model: gemini-3.1-flash-lite (API)                              │
│  Input: ALL 6 query files                                       │
│  Note: Only TARGET policies (query.policy_ids) are passed       │
│        to prompt - not all 12 policies                            │
│  Output: answers_{model}_gemini_{lang}.jsonl                    │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│               STEP 4: Merge Responses                               │
│  Script: scripts/merge_and_export.py                              │
│  Command: merge                                                  │
│  Input: 6 response files (3 query models × 2 response models)    │
│  Output: merged_{lang}.jsonl                                      │
│  Note: Uses --rebase-query-id (default) to ensure each provider  │
│        query gets unique query_id (e.g., deepseek/Q0001)          │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 5: LLM-as-Judge                          │
│  Script: scripts/judge_responses.py                                │
│  Judge ENG: DeepSeek-V4-Pro / qwq-max                            │
│  Judge VNI: qwen3-235b / DeepSeek-V3.2                           │
│  Output: judged_{lang}.jsonl                                     │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│          STEP 5a: Human Review (Streamlit App)                    │
│  Script: scripts/streamlit_human_review.py                         │
│  Review: TẤT CẢ cases                                            │
│  Edit: response, judgment, consensus if needed                   │
│  Output: reviewed_{lang}.jsonl (with human_review flag)          │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│           STEP 6: Determine Easy/Hard (Automatic)                  │
│  Script: scripts/determine_difficulty.py                           │
│  Logic:                                                           │
│      easy: local_correct AND gemini_correct                        │
│      hard: local_wrong AND gemini_correct                          │
│  Note: Both wrong = excluded (not useful for training)            │
│  Output: Adds "difficulty" field (easy/hard)                      │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│          STEP 6b: Synthetic Hard Augmentation                     │
│  Script: scripts/synthetic_hard_augmentation.py                  │
│  Input: golden_dataset_{lang}.jsonl                               │
│  Source pattern: hard samples + selected easy seeds               │
│  Output: golden_dataset_augmented_{lang}.jsonl                    │
│  Note: Generate harder queries on local machine via Alibaba API   │
│        using `ALIBABA_SYN_DATA_DEEPSEEK` from `.env`              │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│               STEP 7: Export + Split                              │
│  Script: scripts/merge_and_export.py                              │
│  Commands: export, split                                          │
│  Final Output:                                                   │
│      - golden_dataset_vni.jsonl (with difficulty label)           │
│      - golden_dataset_eng.jsonl (with difficulty label)           │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│     STEP 8: Convert to MFRouter Routing Format                    │
│  Script: scripts/convert_to_routing_ds.py                         │
│  Phase 1 (LOCAL - no torch):                                      │
│    --skip-embeddings → JSONL + unique_query_texts.txt            │
│  Phase 2 (COLAB - torch + httpx):                                 │
│    --embeddings-only → gọi Alibaba text-embedding-v3             │
│  Input: artifacts/golden/splits_vni/{train,test,dev}.jsonl       │
│  Embedding: Alibaba text-embedding-v3 (1024-dim)                 │
│  Logic:                                                           │
│      difficulty=easy → local=1.0, gemini=1.0                      │
│      difficulty=hard → local=0.0, gemini=1.0                      │
│  Output:                                                          │
│      - artifacts/routing/routing_data_{train,test,dev}.jsonl     │
│      - artifacts/routing/query_data_{train,test,dev}.jsonl       │
│      - artifacts/routing/query_embeddings.pt                     │
│      - artifacts/routing/unique_query_texts.txt                  │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                 STEP 9: Train MFRouter                             │
│  Notebook: notebooks/mfrouter/01_mfrouter_training.ipynb          │
│  Config: configs/model_config_train/mfrouter.yaml                │
│  Model: Matrix Factorization Router (latent_dim=128)              │
│  Output: saved_models/mfrouter/mfrouter_vni.pkl                   │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│               STEP 10: Evaluate MFRouter                           │
│  Notebook: notebooks/mfrouter/02_mfrouter_inference.ipynb         │
│  Metrics: Top-1 accuracy, per-policy routing table                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Cấu trúc file

```
LLMRouter/
├── policy.csv                        # Source policy (12 policies)
├── policy_normalized.json           # Normalized policies
├── test_api.py                      # Smoke test
├── test_generated_queries.json      # 3 samples đã test
│
├── .env                             # ✅ API keys đã có
│
├── safety/
│   └── dataset/
│       └── pipeline.py              # SafetyGoldenDatasetBuilder
│                                    # Chỉ giữ LLM-based methods
│
├── scripts/
│   ├── generate_queries.py          # STEP 1: Generate queries (MiniMax)
│   ├── run_local_generation.py       # STEP 2: Qwen3-4B on Colab
│   ├── run_gemini_generation.py     # STEP 3: Gemini API
│   ├── judge_responses.py            # STEP 4: LLM-as-Judge
│   └── merge_and_export.py           # STEP 5: Merge + Export
│
└── artifacts/
    └── safety_queries/
        # STEP 1: Query generation (3 models × 2 languages = 6 files)
        ├── queries_minimax_vni.jsonl     # 330 queries
        ├── queries_minimax_eng.jsonl     # 330 queries
        ├── queries_deepseek_vni.jsonl    # 330 queries
        ├── queries_deepseek_eng.jsonl    # 330 queries
        ├── queries_qwen_vni.jsonl        # 330 queries
        ├── queries_qwen_eng.jsonl        # 330 queries
        # STEP 2: Qwen3-4B answers (6 files) - output from local model
        ├── answers_minimax_local_vni.jsonl
        ├── answers_minimax_local_eng.jsonl
        ├── answers_deepseek_local_vni.jsonl
        ├── answers_deepseek_local_eng.jsonl
        ├── answers_qwen_local_vni.jsonl
        ├── answers_qwen_local_eng.jsonl
        # STEP 3: Gemini answers (6 files) - output from Gemini API
        ├── answers_minimax_gemini_vni.jsonl
        ├── answers_minimax_gemini_eng.jsonl
        ├── answers_deepseek_gemini_vni.jsonl
        ├── answers_deepseek_gemini_eng.jsonl
        ├── answers_qwen_gemini_vni.jsonl
        ├── answers_qwen_gemini_eng.jsonl
        # STEP 4: After merge (6 answer files merged into 1)
        ├── merged_vni.jsonl
        ├── merged_eng.jsonl
        # STEP 5: After judge
        ├── judged_vni.jsonl
        ├── judged_eng.jsonl
        # STEP 5a: After human review
        ├── reviewed_vni.jsonl
        ├── reviewed_eng.jsonl
        # STEP 6: After determine_difficulty (golden dataset with difficulty label)
        └── golden_dataset_*.jsonl

    └── routing/
        # STEP 8: MFRouter routing format
        ├── routing_data_train.jsonl
        ├── routing_data_test.jsonl
        ├── routing_data_dev.jsonl
        ├── query_data_train.jsonl
        ├── query_data_test.jsonl
        ├── query_data_dev.jsonl
        ├── query_embeddings.pt          # Alibaba text-embedding-v3 (1024-dim)
        └── llm_data.json                # LLM candidates (local + gemini)

saved_models/
    └── mfrouter/
        └── mfrouter_vni.pkl             # Trained MFRouter model
```

---

## 5. API Configuration

### Query Generation Models (sinh queries - chạy CẢ 3)

| Model | ENV | Priority | Note |
|-------|-----|----------|------|
| MiniMax-M2.7 | `MINIMAX_*` | **1** | 1M tokens free, ưu tiên dùng |
| DeepSeek-V4-Pro | `ALIBABA_QUERY_DEEPSEEK` | **2** | Alibaba API |
| qwen3-next-80b-a3b-thinking | `ALIBABA_QUERY_QWEN` | **3** | Alibaba API |

### Response Generation Models (sinh response - 2 models)

| Tier | Model | ENV | Chạy trên |
|------|-------|-----|-----------|
| **local** | Qwen/Qwen3-4B-Instruct-2507 | `LOCAL_GENERATION_MODEL` | **Colab GPU** |
| **high** | gemini-3.1-flash-lite | `GEMINI_GENERATION_NAME` | **Local API** |

### Judge Models (đánh giá responses)

| Ngôn ngữ | Primary | Backup |
|----------|---------|--------|
| ENG | DeepSeek-V4-Pro | qwq-max |
| VNI | qwen3-235b | DeepSeek-V3.2 |
| SUB | glm-5.1 | qwen3.7-plus |

---

## 6. Scripts

### 6.1 generate_queries.py

Sinh queries bằng LLM (3 models - chạy riêng từng model).

```bash
# Vietnamese - MiniMax
python scripts/generate_queries.py \
    --language vi --model minimax \
    --output artifacts/safety_queries/queries_minimax_vni.jsonl

# Vietnamese - DeepSeek
python scripts/generate_queries.py \
    --language vi --model deepseek \
    --output artifacts/safety_queries/queries_deepseek_vni.jsonl

# Vietnamese - Qwen
python scripts/generate_queries.py \
    --language vi --model qwen \
    --output artifacts/safety_queries/queries_qwen_vni.jsonl

# English - same pattern with --language eng

# Options
--language vi|eng           # Language (REQUIRED)
--model minimax|deepseek|qwen # Model to use (REQUIRED)
--output <path>              # Output file (REQUIRED)
--single-per-policy 5        # Samples per policy per complexity
--multi-groups 15            # Number of multi-policy groups
--multi-per-group 3          # Samples per multi-policy group
--no-policy-per-complexity 20 # No-policy samples per complexity
```

### 6.2 run_local_generation.py

Chạy Qwen3-4B trên Colab (GPU). Hỗ trợ resume - nếu output file đã tồn tại, sẽ skip các query đã xử lý.

```bash
# Colab setup
!git clone https://github.com/your-repo/LLMRouter.git
%cd LLMRouter
!pip install transformers torch accelerate

# Run generation (tự động resume nếu file đã có)
!python scripts/run_local_generation.py \
    --input artifacts/safety_queries/queries_minimax_vni.jsonl \
    --output artifacts/safety_queries/answers_minimax_local_vni.jsonl \
    --model-name "Qwen/Qwen3-4B-Instruct-2507" \
    --batch-size 8

# Download output
# from google.colab import files
# files.download("artifacts/safety_queries/answers_minimax_local_vni.jsonl")
```

### 6.3 run_gemini_generation.py

Gọi Gemini API. Hỗ trợ resume - nếu output file đã tồn tại, sẽ skip các query đã xử lý.

```bash
# Vietnamese - Gemini (tự động resume nếu file đã có)
python scripts/run_gemini_generation.py \
    --input artifacts/safety_queries/queries_minimax_vni.jsonl \
    --output artifacts/safety_queries/answers_minimax_gemini_vni.jsonl

# English - Gemini
python scripts/run_gemini_generation.py \
    --input artifacts/safety_queries/queries_minimax_eng.jsonl \
    --output artifacts/safety_queries/answers_minimax_gemini_eng.jsonl

# Options
--model-name "gemini-3.1-flash-lite"
--delay 1.0  # Delay between calls
```

### 6.4 merge_and_export.py

Merge answers từ 6 response files (split được thực hiện ở Step 7)

```bash
# Merge 6 answer files (3 query models × 2 response models)
# Uses --rebase-query-id (default) to prefix query_id with provider name
python scripts/merge_and_export.py merge \
    --inputs artifacts/answers/answers_minimax_local_vni.jsonl \
            artifacts/answers/answers_deepseek_local_vni.jsonl \
            artifacts/answers/answers_qwen_local_vni.jsonl \
            artifacts/answers/answers_minimax_gemini_vni.jsonl \
            artifacts/answers/answers_deepseek_gemini_vni.jsonl \
            artifacts/answers/answers_qwen_gemini_vni.jsonl \
    --output artifacts/answers/merged_vni.jsonl

# Options
--rebase-query-id        # Prefix query_id with provider (e.g., deepseek/Q0001) (default)
--no-rebase-query-id     # Keep original query_id (may cause collisions)
```

### 6.5 judge_responses.py

LLM-as-Judge đánh giá responses. Hỗ trợ resume - nếu output file đã tồn tại, sẽ skip các query đã được judge.

```bash
python scripts/judge_responses.py \
    --input artifacts/answers/merged_vni.jsonl \
    --output artifacts/answers/judged_vni.jsonl \
    --language vi

# Options
--delay 1.0  # Delay between calls
```

### 6.6 streamlit_human_review.py

Streamlit app cho human review TẤT CẢ cases. Cho phép edit response, judgment, consensus.

```bash
# Run streamlit app
streamlit run scripts/streamlit_human_review.py -- \
    --input artifacts/safety_queries/judged_vni.jsonl \
    --output artifacts/safety_queries/reviewed_vni.jsonl

# App features:
# - View all queries with local/gemini responses and judge verdicts
# - Edit any field (response, judgment, consensus)
# - Filter by query_id, policy, difficulty
# - Mark as reviewed
# - Export after review
```

### 6.7 determine_difficulty.py

Tự động xác định easy/hard queries dựa trên judge results.

```bash
python scripts/determine_difficulty.py \
    --input artifacts/safety_queries/reviewed_vni.jsonl \
    --output artifacts/safety_queries/golden_dataset_vni.jsonl

# Logic:
#   easy: local_correct=True AND gemini_correct=True
#   hard: local_correct=False AND gemini_correct=True
#   (Both wrong = excluded from dataset)
```

### 6.8 synthetic_hard_augmentation.py

Script này giờ ghi incremental ra file, nên chạy được an toàn trên Colab và có thể resume.

```bash
# Tạo thêm hard samples và ghi dần từng dòng ra output
python scripts/synthetic_hard_augmentation.py \
  --input artifacts/safety_queries/golden_dataset_vni.jsonl \
  --output artifacts/safety_queries/golden_dataset_augmented_vni.jsonl \
  --ratio 0.5 \
  --resume

# Nếu muốn tạo lại từ đầu, tắt resume
python scripts/synthetic_hard_augmentation.py \
  --input artifacts/safety_queries/golden_dataset_vni.jsonl \
  --output artifacts/safety_queries/golden_dataset_augmented_vni.jsonl \
  --ratio 0.5 \
  --no-resume
```

### 6.9 convert_to_routing_ds.py

Convert golden dataset splits sang MFRouter routing format. **2 phase:**

**Phase 1 (local — không cần torch/GPU):**
```bash
python scripts/convert_to_routing_ds.py \
    --input-dir artifacts/golden/splits_vni \
    --output-dir artifacts/routing \
    --skip-embeddings

# Output Phase 1:
#   artifacts/routing/routing_data_{train,test,dev}.jsonl   (embedding_id=-1)
#   artifacts/routing/query_data_{train,test,dev}.jsonl
#   artifacts/routing/unique_query_texts.txt                 (874 dòng → input cho Phase 2)
```

**Phase 2 (Colab — cần torch + httpx):**
```bash
# Trên Colab sau khi clone repo:
# !pip install torch httpx
# Đảm bảo .env có ALIBABA_URL, ALIBABA_API_KEY, ALIBABA_EMBEDDING

!python scripts/convert_to_routing_ds.py \
    --output-dir artifacts/routing \
    --embeddings-only

# Output Phase 2:
#   artifacts/routing/query_embeddings.pt          (1024-dim, 874 vectors)
#   Backfill embedding_id vào routing_data JSONL
#   Backfill embedding_id vào query_data JSONL
```

**Performance mapping:**
```
  difficulty=easy  → local=1.0, gemini=1.0  (both correct)
  difficulty=hard  → local=0.0, gemini=1.0  (local wrong, gemini right)
```

**Filtering:**
```
  - Bỏ records có judge_consensus="fail"
  - Bỏ records không có difficulty
  - Giữ "uncertain" nếu difficulty xác định được
```

### 6.10 MFRouter Training (STEP 9)

Sau khi có routing data, train MFRouter qua notebook.

```bash
# Chạy trên local (nếu có GPU) hoặc Colab
# Mở notebook và chạy:
#   notebooks/mfrouter/01_mfrouter_training.ipynb

# Config mặc định:
#   - data_path → artifacts/routing/
#   - text_dim: 1024 (Alibaba text-embedding-v3)
#   - latent_dim: 128
#   - epochs: 5
```

---

## 7. Usage

### Complete workflow

```bash
# STEP 1: Generate queries (3 models × 2 languages = 6 files)
# Vietnamese queries
python scripts/generate_queries.py --language vi --model minimax --output artifacts/safety_queries/queries_minimax_vni.jsonl
python scripts/generate_queries.py --language vi --model deepseek --output artifacts/safety_queries/queries_deepseek_vni.jsonl
python scripts/generate_queries.py --language vi --model qwen --output artifacts/safety_queries/queries_qwen_vni.jsonl

# English queries
python scripts/generate_queries.py --language eng --model minimax --output artifacts/safety_queries/queries_minimax_eng.jsonl
python scripts/generate_queries.py --language eng --model deepseek --output artifacts/safety_queries/queries_deepseek_eng.jsonl
python scripts/generate_queries.py --language eng --model qwen --output artifacts/safety_queries/queries_qwen_eng.jsonl

# Push to repo
git add artifacts/safety_queries/queries_*.jsonl
git commit -m "Generated queries"
git push

# STEP 2: Run on Colab (Qwen3-4B) for all 6 query files
# - Clone repo on Colab
# - Upload all 6 queries_*.jsonl files
# - Run for each: python scripts/run_local_generation.py --input ... --output answers_*_local_*.jsonl
# - Download answers_*_local_*.jsonl files

# STEP 3: Run Gemini (local) for all 6 query files
for lang in vni eng; do
  for model in minimax deepseek qwen; do
    python scripts/run_gemini_generation.py \
      --input artifacts/safety_queries/queries_${model}_${lang}.jsonl \
      --output artifacts/safety_queries/answers_${model}_gemini_${lang}.jsonl
  done
done

# STEP 4: Merge (combine all 6 answer files into one)
# Uses --rebase-query-id (default) so each provider's query gets unique ID
python scripts/merge_and_export.py merge \
  --inputs artifacts/safety_queries/answers_minimax_local_vni.jsonl \
          artifacts/safety_queries/answers_deepseek_local_vni.jsonl \
          artifacts/safety_queries/answers_qwen_local_vni.jsonl \
          artifacts/safety_queries/answers_minimax_gemini_vni.jsonl \
          artifacts/safety_queries/answers_deepseek_gemini_vni.jsonl \
          artifacts/safety_queries/answers_qwen_gemini_vni.jsonl \
  --output artifacts/safety_queries/merged_vni.jsonl \
  --rebase-query-id

# STEP 5: Judge (evaluate merged responses)
python scripts/judge_responses.py --input merged_vni.jsonl --output judged_vni.jsonl --language vi

# STEP 5a: Human Review (Streamlit app)
streamlit run scripts/streamlit_human_review.py -- --input judged_vni.jsonl --output reviewed_vni.jsonl
# Review TẤT CẢ cases trên giao diện web, edit nếu cần
# Sau khi review xong, click "Export" trong app

# STEP 6: Determine Easy/Hard
python scripts/determine_difficulty.py \
    --input artifacts/safety_queries/reviewed_vni.jsonl \
    --output artifacts/safety_queries/golden_dataset_vni.jsonl

# STEP 6b: Synthetic hard augmentation (optional)
python scripts/synthetic_hard_augmentation.py \
  --input artifacts/safety_queries/golden_dataset_vni.jsonl \
  --output artifacts/safety_queries/golden_dataset_augmented_vni.jsonl \
  --ratio 0.5

# STEP 7: Split train/dev/test
python scripts/merge_and_export.py split \
    --input artifacts/safety_queries/golden_dataset_augmented_vni.jsonl \
    --output-dir artifacts/safety_queries/splits_vni \
    --train-ratio 0.7 \
    --dev-ratio 0.15 \
    --test-ratio 0.15

# STEP 8: Convert to MFRouter routing format
# Phase 1 (local - không cần GPU):
python scripts/convert_to_routing_ds.py \
    --input-dir artifacts/golden/splits_vni \
    --output-dir artifacts/routing \
    --skip-embeddings

# Phase 2 (Colab):
#   !pip install torch httpx
#   !python scripts/convert_to_routing_ds.py --output-dir artifacts/routing --embeddings-only

# STEP 9: Train MFRouter (Colab notebook)
# notebooks/mfrouter/01_mfrouter_training.ipynb
#   - Config: configs/model_config_train/mfrouter.yaml
#   - Model saved to: saved_models/mfrouter/mfrouter_vni.pkl

# STEP 10: Evaluate (Colab notebook)
# notebooks/mfrouter/02_mfrouter_inference.ipynb
```

---

## 8. Ghi chú quan trọng

1. **3 Query Generation Models** - chạy CẢ 3 (MiniMax, DeepSeek-V4-Pro, qwen3-next-80b)
2. **MiniMax ưu tiên** - 1M tokens free, dùng trước
3. **2 Response Generation Models** - Qwen3-4B (Colab) + Gemini (API)
4. **Colab cho Qwen3-4B** - local machine không chạy nổi 4B model
5. **Không lưu keys trong code** - dùng `.env`
6. **Resume mode** - `run_gemini_generation.py`, `run_local_generation.py`, và `judge_responses.py` đều hỗ trợ resume. Nếu output file đã tồn tại, sẽ tự động skip các query đã xử lý.
7. **MFRouter Phase 1 (local)** - `convert_to_routing_ds.py --skip-embeddings` chạy local không cần torch/GPU.
8. **MFRouter Phase 2 (Colab)** - `--embeddings-only` cần torch + httpx, chạy trên Colab để gọi Alibaba embedding API.
9. **MFRouter Training (Colab)** - Notebook cần GPU, chạy trên Colab với notebook `01_mfrouter_training.ipynb`.

---

## 9. Files

| File | Mô tả |
|------|-------|
| `safety/dataset/pipeline.py` | SafetyGoldenDatasetBuilder (LLM-based only) |
| `scripts/generate_queries.py` | Generate queries using LLM (3 models) |
| `scripts/run_local_generation.py` | Qwen3-4B on Colab |
| `scripts/run_gemini_generation.py` | Gemini API calls |
| `scripts/judge_responses.py` | LLM-as-Judge |
| `scripts/streamlit_human_review.py` | Human review UI (Streamlit) |
| `scripts/determine_difficulty.py` | Auto determine easy/hard |
| `scripts/synthetic_hard_augmentation.py` | Synthetic hard query augmentation |
| `scripts/convert_to_routing_ds.py` | Convert golden dataset → MFRouter routing format |
| `scripts/merge_and_export.py` | Merge + Split only (export via determine_difficulty.py) |

## 10. Model Configuration (.env)

```
# Query Generation (3 models - ALL RUN)
MINIMAX_QUERY_NAME=MiniMax-M2.7
ALIBABA_QUERY_DEEPSEEK=DeepSeek-V4-Pro
ALIBABA_QUERY_QWEN=qwen3-next-80b-a3b-thinking
ALIBABA_QUERY_SUB1=qwen3.6-plus (backup)
ALIBABA_QUERY_SUB2=qwen3.7-max (backup)

# Response Generation (2 models)
LOCAL_GENERATION_MODEL=Qwen/Qwen3-4B-Instruct-2507
GEMINI_GENERATION_NAME=gemini-3.1-flash-lite

# Embedding Model (for MFRouter)
ALIBABA_EMBEDDING=text-embedding-v3
```

---

## 11. Data Format (JSON Fields)

### 11.1 `answers_*_local_*.jsonl` (Step 2 - 3)

```json
{
  "query_id": "string",
  "query": "string",
  "policy_ids": ["P01", "P02"],
  "policy_names": ["string"],
  "designed_complexity": "low|medium|high",
  "policy_match_type": "single_policy|multi_policy|no_policy",
  "group_type": "string",
  "language": "vi|en",
  "query_model": "minimax|deepseek|qwen",
  "model_name": "Qwen/Qwen3-4B-Instruct-2507",
  "model_response": "string",
  "response_time": 0.0,
  "error": null
}
```

### 11.2 `answers_*_gemini_*.jsonl` (Step 3)

```json
{
  "query_id": "string",
  "query": "string",
  "policy_ids": ["P01", "P02"],
  "policy_names": ["string"],
  "designed_complexity": "low|medium|high",
  "policy_match_type": "single_policy|multi_policy|no_policy",
  "group_type": "string",
  "language": "vi|en",
  "query_model": "minimax|deepseek|qwen",
  "model_name": "gemini-3.1-flash-lite",
  "model_response": "string",
  "response_time": 0.0,
  "error": null
}
```

### 11.3 `merged_vni.jsonl` (Step 4)

```json
{
  "query_id": "string",              // Rebased: provider/query_id (e.g., "deepseek/Q0001")
  "original_query_id": "string",      // Original query_id from source file
  "provider": "minimax|deepseek|qwen", // Source provider
  "query": "string",
  "policy_ids": ["P01", "P02"],
  "policy_names": ["string"],
  "designed_complexity": "low|medium|high",
  "policy_match_type": "single_policy|multi_policy|no_policy",
  "group_type": "string",
  "language": "vi|en",
  "metadata": {
    "expected_behavior": "string",
    "reason": "string"
  },
  "responses": {
    "Qwen/Qwen3-4B-Instruct-2507": {
      "response": "string",
      "response_time": 0.0,
      "error": null
    },
    "gemini-3.1-flash-lite": {
      "response": "string",
      "response_time": 0.0,
      "error": null
    }
  }
}
```

**Note:** With `--rebase-query-id` (default), each provider's queries get unique IDs.
Without it, queries with same query_id from different providers would overwrite each other.

### 11.4 `judged_vni.jsonl` (Step 5)

```json
{
  "query_id": "string",
  "query": "string",
  "policy_ids": ["P01", "P02"],
  "policy_names": ["string"],
  "designed_complexity": "low|medium|high",
  "policy_match_type": "single_policy|multi_policy|no_policy",
  "group_type": "string",
  "language": "vi|en",
  "metadata": {
    "expected_behavior": "string",
    "reason": "string"
  },
  "responses": {
    "Qwen/Qwen3-4B-Instruct-2507": {
      "response": "string",
      "response_time": 0.0,
      "error": null
    },
    "gemini-3.1-flash-lite": {
      "response": "string",
      "response_time": 0.0,
      "error": null
    }
  },
  "judge_result": {
    "evaluations": {
      "local_model_response": {
        "is_correct": true|false,
        "reasoning": "string"
      },
      "gemini_model_response": {
        "is_correct": true|false,
        "reasoning": "string"
      }
    },
    "consensus": "pass|fail|uncertain",
    "local_correct": true|false|null,
    "gemini_correct": true|false|null,
    "judge_model": "string",
    "judge_type": "primary|backup",
    "call_duration": 0.0,
    "attempts_used": 0,
    "error": null
  },
  "consensus_status": "pass|fail|uncertain",
  "pass": true|false|null
}
```

### 11.5 `reviewed_vni.jsonl` (Step 5a - after human review)

```json
{
  "query_id": "string",
  "query": "string",
  "policy_ids": ["P01", "P02"],
  "policy_names": ["string"],
  "designed_complexity": "low|medium|high",
  "policy_match_type": "single_policy|multi_policy|no_policy",
  "group_type": "string",
  "language": "vi|en",
  "metadata": {
    "expected_behavior": "string",
    "reason": "string"
  },
  "responses": {
    "Qwen/Qwen3-4B-Instruct-2507": {
      "response": "string",
      "response_time": 0.0,
      "error": null
    },
    "gemini-3.1-flash-lite": {
      "response": "string",
      "response_time": 0.0,
      "error": null
    }
  },
  "judge_result": {
    "evaluations": {
      "local_model_response": {
        "is_correct": true|false,
        "reasoning": "string"
      },
      "gemini_model_response": {
        "is_correct": true|false,
        "reasoning": "string"
      }
    },
    "consensus": "pass|fail|uncertain",
    "local_correct": true|false|null,
    "gemini_correct": true|false|null,
    "judge_model": "string",
    "judge_type": "primary|backup",
    "call_duration": 0.0,
    "attempts_used": 0,
    "error": null
  },
  "consensus_status": "pass|fail|uncertain",
  "pass": true|false|null,
  "human_reviewed": true
}
```

### 11.6 `golden_dataset_vni.jsonl` (Step 6 - final)

```json
{
  "query_id": "string",
  "query": "string",
  "policy_ids": ["P01", "P02"],
  "designed_complexity": "low|medium|high",
  "group_type": "string",
  "language": "vi|en",
  "difficulty": "easy|hard",
  "local_response": "string",
  "gemini_response": "string",
  "judge_consensus": "pass|fail|uncertain",
  "human_reviewed": true
}
```

**Notes:**
- `difficulty=easy`: local_correct=True AND gemini_correct=True
- `difficulty=hard`: local_correct=False AND gemini_correct=True
- Records where both models agree on wrong or uncertain are excluded
- `judge_consensus` is the normalized consensus from Step 5/5a (`pass|fail|uncertain`)
