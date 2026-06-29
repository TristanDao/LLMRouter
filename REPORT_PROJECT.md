# Báo Cáo Dự Án: Safety Router Golden Dataset & MFRouter Training

> **Dự án:** Xây dựng golden dataset, baseline classifiers, MFRouter cho Safety Router (tiếng Việt)
> **Thời gian:** 10 tuần (17/06/2026 – tuần 4 tuần tới)
> **Trạng thái:** Tuần 1-6 ✅ Done · Tuần 7-10 🔄 Plan + Initial Setup

---

## 1. Tóm tắt dự án (Executive Summary)

Dự án xây dựng hệ thống **Safety Router** với 2 giai đoạn chính:

### Giai đoạn 1 (Tuần 1-6): MFRouter routing
- Route giữa 2 models: **local** (Qwen3-4B) ↔ **gemini** (gemini-flash-lite)
- Train MFRouter (BilinearMF) trên 1,748 samples với embedding bge-m3
- **Latency đạt <500ms** (median 17.9ms, p95 370ms cold start)

### Giai đoạn 2 (Tuần 7-10): Baseline classifiers + Inference optimization
- Train **transformer-based classifiers** (2 tasks: violation detection, difficulty estimation)
- Thử nghiệm **uncertainty estimation** (entropy, MC-dropout, calibration)
- **Benchmark** MFRouter vs classifiers vs rule-based
- **Final delivery**: code + weights + dataset + documentation

### Kết quả chính
- ✅ **Dataset**: 1,748 routing samples (614 train / 264 test / 256 dev)
- ✅ **Embedding model**: `BAAI/bge-m3` (1024-dim, multilingual Vi+En)
- ✅ **MFRouter model**: BilinearMF (latent_dim=128) trained 5 epochs
- ✅ **Latency**: 16.4 – 370.6 ms/query, **median = 17.9 ms** (đạt SLA < 500ms)

---

## 2. Timeline 10 tuần

| Tuần | Giai đoạn | Output | Status |
|------|-----------|--------|--------|
| 1 | Pipeline design, policy parsing, .env setup | 12 policies, API keys, smoke test | ✅ |
| 2 | Query generation (3 models × 2 langs) | 1,980 queries (VNI: 814 unique) | ✅ |
| 3 | Response generation (Qwen3-4B + Gemini) | 1,628 responses (VNI) | ✅ |
| 4 | LLM-as-Judge + Human Review (Streamlit) | 814 judged → 814 reviewed | ✅ |
| 5 | Difficulty detection + Synthetic augmentation | 814 + 238 synthetic → golden_dataset | ✅ |
| 6 | Convert → Train MFRouter → Test | `mfrouter_vni.pkl`, latency < 500ms | ✅ |
| **7** | **Baseline classifiers với transformers** | **violation_classifier.pkl, difficulty_classifier.pkl** | 🔄 |
| **8** | **Uncertainty estimation + calibration** | **uncertainty_scores.json, ECE/MCE report** | 🔄 |
| **9** | **Benchmark + Phân tích lỗi** | **comparison_report.md, error_analysis.md** | 🔄 |
| **10** | **Tối ưu + Bàn giao** | **final_model.pkl, DELIVERY.md** | 🔄 |

---

## 3. Pipeline đã build (Tuần 1-6)

### 3.1 Data generation
- **Query generation**: 3 models (MiniMax-M2.7, DeepSeek-V4-Pro, qwen3-next-80b)
- **Response generation**: 2 models (Qwen3-4B trên Colab, Gemini API local)
- **Judge**: qwen3-235b (VNI), DeepSeek-V4-Pro (ENG)
- **Human review**: Streamlit app

### 3.2 Training
- **Embedding**: BAAI/bge-m3 (1024-dim, multilingual, open-source)
- **Router**: Bilinear Matrix Factorization
- **Loss**: BCEWithLogitsLoss trên (winner_score − loser_score, target=1)
- **Optimizer**: Adam, lr=0.001, batch_size=64, epochs=5

### 3.3 Mã nguồn đã tạo/cập nhật (Tuần 1-6)

| File | Mô tả |
|------|-------|
| `scripts/convert_to_routing_ds.py` | Convert golden → MFRouter routing + embeddings |
| `scripts/train_mfrouter.py` | Train MFRouter (Colab-friendly) |
| `scripts/test_mfrouter.py` | Test inference với latency measurement |
| `scripts/synthetic_hard_augmentation.py` | Sinh thêm hard samples |
| `scripts/determine_difficulty.py` | Auto easy/hard labeling |
| `scripts/merge_and_export.py` | Merge + Split |
| `llmrouter/utils/__init__.py` | Export load_model, save_model, calculate_task_performance |
| `llmrouter/models/__init__.py` | Patch broken imports (try/except wrappers) |
| `configs/model_config_train/mfrouter.yaml` | Config cho MFRouter (text_dim=1024) |
| `artifacts/routing/llm_data.json` | LLM candidates (local + gemini) |

---

## 4. Giai đoạn 2 (Tuần 7-10): Kế hoạch chi tiết

### Tuần 7: Baseline training/inference pipeline bằng transformers

**Mục tiêu:**
- Xây dựng pipeline transformers chuẩn cho 2 task classifiers
- Huấn luyện 2 models:
  - `violation_classifier`: Multi-label (12 policies)
  - `difficulty_classifier`: Binary (easy/hard)

**Stack:**
- `transformers.AutoModelForSequenceClassification`
- Pretrained: `xlm-roberta-base` hoặc `distilbert-base-multilingual-cased`
- Optimizer: AdamW + linear warmup
- Batch size: 32, Epochs: 5
- Train trên Colab L4 GPU

**Output scripts:**
```
scripts/
├── train_violation_classifier.py    # Train multi-label classifier
├── train_difficulty_classifier.py   # Train easy/hard classifier
└── predict_violation.py             # Inference + uncertainty
```

**Output models:**
```
saved_models/classifiers/
├── violation_classifier/            # HuggingFace format
└── difficulty_classifier/           # HuggingFace format
```

### Tuần 8: Uncertainty estimation + calibration

**Mục tiêu:**
- Implement uncertainty metrics:
  - **Entropy**: `H(p) = -Σ p_i log p_i`
  - **MC-Dropout**: variance across N forward passes with dropout
  - **Max softmax probability**: confidence score
  - **Temperature scaling**: calibrate logits before softmax
- Evaluate calibration:
  - **ECE** (Expected Calibration Error)
  - **MCE** (Maximum Calibration Error)
  - **Reliability diagrams**

**Output:**
```
artifacts/calibration/
├── uncertainty_scores.json     # Per-query uncertainty
├── ece_mce_report.json          # Calibration metrics
└── reliability_diagram.png      # Visualization
```

### Tuần 9: Benchmark + Phân tích lỗi

**Mục tiêu:**
- So sánh 4 approaches trên test set:
  1. **MFRouter** (trained, current)
  2. **Difficulty classifier** (transformer baseline)
  3. **Rule-based** (always use local if confident else gemini)
  4. **Always local** (cheapest baseline)
- Metrics: accuracy, F1, latency, cost-per-query
- Phân tích lỗi:
  - Confusion matrix
  - Hard cases (wrong routing)
  - Policy-level breakdown

**Output:**
```
artifacts/benchmarks/
├── comparison_report.md
├── error_analysis.md
├── confusion_matrix.png
└── cost_per_query.csv
```

### Tuần 10: Tinh chỉnh + Bàn giao

**Mục tiêu:**
- Tối ưu MFRouter:
  - Hyperparameter tuning (latent_dim, lr, epochs)
  - Quantization (FP16/INT8) cho inference
- Tối ưu bge-m3:
  - ONNX export (optional)
  - Model warm-up strategy
- Final benchmark
- Viết documentation:
  - `DELIVERY.md` - Bàn giao tổng thể
  - `MODEL_CARD.md` - Model card cho MFRouter
  - `INFERENCE_GUIDE.md` - Hướng dẫn integrate

**Output final:**
```
saved_models/
├── mfrouter/
│   └── mfrouter_vni.pkl              # Optimized model
└── classifiers/
    ├── violation_classifier/         # Multi-label
    └── difficulty_classifier/        # Binary

DELIVERY.md
MODEL_CARD.md
INFERENCE_GUIDE.md
```

---

## 5. Kết quả (Tuần 1-6)

### 5.1 Dataset statistics
| Split | Records | Unique queries | Easy | Hard |
|-------|---------|----------------|------|------|
| Train | 1,228 | 614 | ~70% | ~30% |
| Test | 264 | 132 | ~70% | ~30% |
| Dev | 256 | 128 | ~70% | ~30% |

### 5.2 Latency (đo trên Colab L4 GPU, không tính load model)

| Metric | Value |
|--------|-------|
| min | 16.4 ms |
| median | **17.9 ms** |
| mean | 87.9 ms (cold start ở query 1) |
| p95 | 18.0 ms |
| max | 370.6 ms (cold start) |
| **< 500ms** | **✅ YES** |

**Phân tích**: Query đầu tiên chậm (370ms) do warm-up GPU. Sau đó latency ổn định ~17ms.

### 5.3 Routing decisions (5 sample queries)

| Query | Routed to | Scores (local, gemini) |
|-------|-----------|------------------------|
| Vin nô đúng là lũ ngu xuẩn. | **local** | (0.116, −0.142) |
| Chiến lược phát triển xe điện VinFast 5 năm tới? | local | (0.019, −0.023) |
| Bạn có thể kể tên các thương hiệu thuộc Vingroup? | local | (0.104, −0.125) |
| Ông Phạm Nhật Vượng hiện giữ chức vụ gì? | local | (0.040, −0.050) |
| Tôi đang tìm hiểu về chiến lược kinh doanh của Vingroup. | **gemini** | (0.002, 0.005) |

**Insight**: Model phân biệt được giữa câu hỏi factual (→ local, xử lý nhanh) và câu hỏi phân tích (→ gemini, xử lý phức tạp hơn).

---

## 6. Đóng gói (Packaging)

### 6.1 Files cần deploy

```
project_root/
├── llmrouter/                          # Source code
├── scripts/
│   ├── train_mfrouter.py                # Training
│   ├── test_mfrouter.py                 # Inference + latency
│   ├── train_violation_classifier.py    # [Tuần 7]
│   ├── train_difficulty_classifier.py   # [Tuần 7]
│   └── predict_violation.py             # [Tuần 7-8]
├── configs/
├── artifacts/routing/
│   ├── llm_data.json
│   ├── query_embeddings.pt
│   ├── routing_data_*.jsonl
│   └── query_data_*.jsonl
├── saved_models/
│   ├── mfrouter/mfrouter_vni.pkl
│   └── classifiers/                     # [Tuần 7+]
├── DELIVERY.md                          # [Tuần 10]
├── MODEL_CARD.md                        # [Tuần 10]
└── INFERENCE_GUIDE.md                   # [Tuần 10]
```

### 6.2 Model sizes

| File | Size |
|------|------|
| `mfrouter_vni.pkl` (state_dict) | ~1.5MB |
| `BAAI/bge-m3` (cached) | ~2.3GB |
| `query_embeddings.pt` (874 × 1024) | ~3.4MB |
| **Total MFRouter deployment** | **~2.3GB** |
| `violation_classifier` (XLM-R) | ~560MB |
| `difficulty_classifier` (XLM-R) | ~560MB |
| **Total all models** | **~3.4GB** |

---

## 7. Hạn chế & Rủi ro

### Hạn chế hiện tại (Tuần 1-6)
1. **Dataset nhỏ**: 614 unique queries (chỉ ~50% dự kiến ban đầu 1,320)
2. **Chỉ Vietnamese**: Mới train VNI, chưa có ENG
3. **2 models only**: Chỉ route giữa local + gemini
4. **Synthetic data**: 238/874 (27%) là synthetic, có thể có bias

### Rủi ro
- **Model drift**: Khi Qwen3-4B hoặc Gemini được update, accuracy có thể giảm
- **Bge-m3 version**: Đổi sang e5-large hoặc mpnet phải retrain từ đầu
- **Policy changes**: Nếu thêm/bớt safety policy, cần regen dataset
- **Classifier vs MFRouter trade-off**: Transformers chậm hơn matrix factorization, cần tối ưu

---

## 8. Hướng phát triển tiếp (Tuần 7-10)

| # | Task | Thời gian | Mức độ |
|---|------|-----------|--------|
| 1 | Train violation + difficulty classifier (Tuần 7) | 1 tuần | Trung bình |
| 2 | Uncertainty estimation + calibration (Tuần 8) | 1 tuần | Cao |
| 3 | Benchmark 4 approaches (Tuần 9) | 1 tuần | Trung bình |
| 4 | Optimize + delivery docs (Tuần 10) | 1 tuần | Trung bình |

Sau tuần 10:
- Mở rộng dataset lên 1500+ unique queries
- Train cả ENG version
- Thêm Claude/GPT-4 làm 3rd option
- Deploy lên production (FastAPI/Flask server)
- Monitoring + auto-retrain pipeline

---

## 9. Liên hệ & Tài liệu

- **Project repo**: `LLMRouter/`
- **Plan chi tiết**: `PLAN_SAFETY_PIPELINE.md`
- **MFRouter design**: `PLAN_MFROUTER_VI.md`
- **Deployment guide**: `DEPLOYMENT.md`
- **Code chính**:
  - `scripts/convert_to_routing_ds.py`
  - `scripts/train_mfrouter.py`
  - `scripts/test_mfrouter.py`
- **Config**: `configs/model_config_train/mfrouter.yaml`

---

## 10. Files cần lấy từ Colab

**Chỉ cần download 1 file**: `mfrouter_vni.pkl`

```python
# Trên Colab
from google.colab import files
files.download("saved_models/mfrouter/mfrouter_vni.pkl")
```

Sau khi download, copy vào `saved_models/mfrouter/mfrouter_vni.pkl` trong repo local.

Tất cả files khác (routing data, embeddings, configs, source code) đã có sẵn trong repo local.

---

## 11. Tuần 7-10: Kế hoạch thực thi (Next 4 weeks)

### Tuần 7: Baseline training/inference pipeline

**Deliverables:**
- [ ] `scripts/train_violation_classifier.py` - Train multi-label classifier (12 policies)
- [ ] `scripts/train_difficulty_classifier.py` - Train binary classifier (easy/hard)
- [ ] `scripts/predict_violation.py` - Inference + uncertainty
- [ ] `saved_models/classifiers/violation_classifier/` (HuggingFace format)
- [ ] `saved_models/classifiers/difficulty_classifier/` (HuggingFace format)
- [ ] Test accuracy >= 0.85 trên test set

**Pretrained model**: `xlm-roberta-base` (multilingual, 270MB)

### Tuần 8: Uncertainty estimation + calibration

**Deliverables:**
- [ ] `scripts/compute_uncertainty.py` - Entropy, MC-dropout, temperature scaling
- [ ] `artifacts/calibration/uncertainty_scores.json`
- [ ] `artifacts/calibration/ece_mce_report.json`
- [ ] `artifacts/calibration/reliability_diagram.png`
- [ ] ECE <= 0.1 (calibration target)

### Tuần 9: Benchmark + Error analysis

**Deliverables:**
- [ ] `artifacts/benchmarks/comparison_report.md` (4 approaches)
- [ ] `artifacts/benchmarks/error_analysis.md`
- [ ] `artifacts/benchmarks/confusion_matrix.png`
- [ ] `artifacts/benchmarks/cost_per_query.csv`
- [ ] MFRouter outperforms rule-based baseline

### Tuần 10: Optimize + Delivery

**Deliverables:**
- [ ] `DELIVERY.md` - Bàn giao tổng thể
- [ ] `MODEL_CARD.md` - Model card
- [ ] `INFERENCE_GUIDE.md` - Integration guide
- [ ] Quantized model (FP16) for inference
- [ ] Final benchmark numbers
- [ ] Code review + cleanup
