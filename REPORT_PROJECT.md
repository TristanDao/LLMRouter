# Báo Cáo Dự Án: Difficulty and Uncertainty Estimation in Guardrails Systems

> **Dự án:** Difficulty and Uncertainty Estimation in Guardrails Systems
> **Sinh viên thực hiện:** Đào Phước Thịnh
> **Mentor:** Đoàn Thành Khang
> **Thời gian:** 6 tuần (17/06/2026 – 29/06/2026)
> **Trạng thái:** Hoàn thành

---

## 1. Kết quả chính

- ✅ **Dataset**: 1,748 routing samples (614 train / 264 test / 256 dev)
- ✅ **Embedding model**: `BAAI/bge-m3` (1024-dim, multilingual Vi+En)
- ✅ **MFRouter model**: BilinearMF (latent_dim=128) trained 5 epochs
- ✅ **Latency**: 16.4 – 370.6 ms/query, **median = 17.9 ms** (đạt SLA < 500ms)

### Latency (đo trên Colab L4 GPU, không tính load model)

| Metric      | Value                          |
| ----------- | ------------------------------ |
| min         | 16.4 ms                        |
| median      | **17.9 ms**                    |
| mean        | 87.9 ms (cold start ở query 1) |
| p95         | 18.0 ms                        |
| max         | 370.6 ms (cold start)          |
| **< 500ms** | **✅ YES**                     |

### Dataset statistics

**874 unique queries** = 475 real (golden) + 399 synthetic (từ `synthetic_hard_augmentation.py`)

> **Tại sao Records = 2 × Unique queries?**
> MFRouter (Bilinear Matrix Factorization) học theo cặp: với mỗi query, model phải biết **cả local lẫn gemini** trả lời tốt như thế nào để quyết định route. Nên mỗi unique query → **2 records** trong `routing_data_*.jsonl` (1 cho local, 1 cho gemini), cùng `query_id` nhưng khác `model_name` + `performance`.
>
> ```jsonl
> {"query": "...", "query_id": "Q001", "model_name": "local",  "performance": 0.85}
> {"query": "...", "query_id": "Q001", "model_name": "gemini", "performance": 0.62}
> ```
>
> ⇒ **874 unique × 2 = 1,748 records** dùng để train MFRouter.

| Split     | Unique queries | Synthetic     | Real          | × 2 → | Records (training) |
| --------- | -------------- | ------------- | ------------- | ----- | ------------------ |
| Train     | 614            | 275 (45%)     | 339 (55%)     | × 2   | 1,228              |
| Test      | 132            | 57 (43%)      | 75 (57%)      | × 2   | 264                |
| Dev       | 128            | 67 (52%)      | 61 (48%)      | × 2   | 256                |
| **Total** | **874**        | **399 (46%)** | **475 (54%)** | × 2   | **1,748**          |

### Routing decisions (5 sample queries)

| Query                                                    | Routed to  | Scores (local, gemini) |
| -------------------------------------------------------- | ---------- | ---------------------- |
| Vin nô đúng là lũ ngu xuẩn.                              | **local**  | (0.116, −0.142)        |
| Chiến lược phát triển xe điện VinFast 5 năm tới?         | local      | (0.019, −0.023)        |
| Bạn có thể kể tên các thương hiệu thuộc Vingroup?        | local      | (0.104, −0.125)        |
| Ông Phạm Nhật Vượng hiện giữ chức vụ gì?                 | local      | (0.040, −0.050)        |
| Tôi đang tìm hiểu về chiến lược kinh doanh của Vingroup. | **gemini** | (0.002, 0.005)         |

**Insight**: Model phân biệt được giữa câu hỏi factual (→ local, xử lý nhanh) và câu hỏi phân tích (→ gemini, xử lý phức tạp hơn).

---

## 2. Sản phẩm bàn giao

### 2.1 Code

```
project_root/
├── llmrouter/                          # Source code
│   ├── models/                         # Model definitions (BilinearMF, ...)
│   ├── utils/                          # load_model, save_model, ...
│   └── ...
├── scripts/
│   ├── convert_to_routing_ds.py        # Convert golden → MFRouter routing + embeddings
│   ├── train_mfrouter.py               # Training (Colab-friendly)
│   ├── test_mfrouter.py                # Inference + latency measurement
│   ├── synthetic_hard_augmentation.py   # Sinh thêm hard samples
│   ├── determine_difficulty.py         # Auto easy/hard labeling
│   └── merge_and_export.py             # Merge + Split
├── configs/
│   └── model_config_train/mfrouter.yaml
```

### 2.2 Weights

```
saved_models/
└── mfrouter/
    └── mfrouter_vni.pkl                # ~1.5MB, BilinearMF (latent_dim=128)
```

> Bge-m3 (`~2.3GB`) sẽ tự động download lần đầu khi chạy inference.

### 2.3 Dataset

```
artifacts/routing/
├── llm_data.json                       # LLM candidates (local + gemini)
├── query_embeddings.pt                 # bge-m3 embeddings (874 × 1024)
├── routing_data_*.jsonl                # MFRouter training/test/dev
└── query_data_*.jsonl
```

### 2.4 Tài liệu hướng dẫn

| File                      | Mô tả                                           |
| ------------------------- | ----------------------------------------------- |
| `REPORT_PROJECT.md`       | Báo cáo tổng thể dự án (file này)               |
| `PLAN_SAFETY_PIPELINE.md` | Plan chi tiết 6 tuần pipeline                   |
| `PLAN_MFROUTER_VI.md`     | MFRouter design (BilinearMF, loss, hyperparams) |
| `README.md`               | Hướng dẫn cơ bản                                |

### 2.5 Model sizes (tổng)

| File                   | Size       |
| ---------------------- | ---------- |
| `mfrouter_vni.pkl`     | ~1.5MB     |
| `BAAI/bge-m3` (cached) | ~2.3GB     |
| `query_embeddings.pt`  | ~3.4MB     |
| **Total deployment**   | **~2.3GB** |

---

## 3. Hạn chế

1. **Dataset nhỏ**: 475 unique real queries (chỉ ~36% dự kiến ban đầu 1,320)
2. **Chỉ Vietnamese**: Mới train VNI, chưa có ENG
3. **2 models only**: Chỉ route giữa local + gemini
4. **Synthetic data**: 399/874 (46%) là synthetic (tăng từ golden gốc 475 → 874 unique), có thể có bias

### Rủi ro

- **Model drift**: Khi Qwen3-4B hoặc Gemini được update, accuracy có thể giảm
- **Bge-m3 version**: Đổi sang e5-large hoặc mpnet phải retrain từ đầu
- **Policy changes**: Nếu thêm/bớt safety policy, cần regen dataset

---

## 4. Ý nghĩa các file trong `artifacts/answers/`

Toàn bộ pipeline tạo dataset chạy qua các bước sau; mỗi file là output của 1 bước:

```
   [Tuần 1-2]              [Tuần 3]                [Tuần 4]               [Tuần 5]
 queries_*.jsonl  ──►  answers_*_{local,gemini}  ──►  judged_vni.jsonl  ──►  merged_vni.jsonl
   (input)              _vni.jsonl                    (LLM-as-Judge)         (chọn winner/loser)
                                                                                    │
                                                                                    ▼
                                                                           reviewed_vni.jsonl
                                                                         (sau human review)
                                                                                    │
                                                                                    ▼
                                                                           golden_dataset_vni.jsonl
                                                                          (chuẩn hóa, gán difficulty)
                                                                                    │
                                                                                    ▼
                                                                       golden_dataset_augmented_vni.jsonl
                                                                          (thêm synthetic hard)
```

### Chi tiết từng file

| File                                                       | Records           | Mô tả                                                                                                                                                                       |
| ---------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `safety_queries/queries_{qwen,deepseek,minimax}_vni.jsonl` | 1,980 tổng        | **Query generation**: 3 LLM generators sinh 660 queries mỗi cái, tổng 1,980 (sau dedup còn 814 unique)                                                                      |
| `answers/answers_{provider}_{local,gemini}_vni.jsonl`      | 6 files × 310-330 | **Response generation**: 2 response models (local = Qwen3-4B, gemini = Gemini API) trả lời cho queries từ mỗi provider. Tổng 6 file = 3 query providers × 2 response models |
| `answers/judged_vni.jsonl`                                 | 970               | **LLM-as-Judge**: Judge LLM chấm điểm response của local vs gemini → `judge_result` (winner/loser), `consensus_status`, `pass`                                              |
| `answers/merged_vni.jsonl`                                 | 970               | **Merge**: gộp judged + metadata, chuẩn bị cho routing dataset                                                                                                              |
| `answers/reviewed_vni.jsonl`                               | 970               | **Human review**: Streamlit app, người review sửa lại label của judge (nếu judge sai) → `human_reviewed=True`                                                               |
| `answers/golden_dataset_vni.jsonl`                         | 812               | **Golden dataset gốc**: query + 2 responses + difficulty (easy/hard) + human_reviewed. Sau khi dedup & filter quality                                                       |
| `answers/golden_dataset_augmented_vni.jsonl`               | 882               | **Golden dataset + synthetic hard**: thêm 406 synthetic hard queries từ `synthetic_hard_augmentation.py`                                                                    |
| `routing/routing_data_{train,test,dev}.jsonl`              | 1,748             | **MFRouter training data**: mỗi query có 2 records (1 cho local, 1 cho gemini) với `performance` score                                                                      |
| `routing/query_data_{train,test,dev}.jsonl`                | 874               | **Unique queries** trong routing data (dùng để build embedding)                                                                                                             |
| `routing/query_embeddings.pt`                              | 874 × 1024        | **bge-m3 embeddings** của tất cả unique queries (precomputed)                                                                                                               |
| `routing/llm_data.json`                                    | -                 | Metadata: tên 2 LLM candidates (local + gemini)                                                                                                                             |
| `routing/unique_query_texts.txt`                           | 874               | Danh sách unique query text (cho dedup/debug)                                                                                                                               |

### Flow rút gọn

```
Query gen (3 LLMs) → Response gen (2 LLMs) → Judge → Human review
    → Golden (gán difficulty) → Augment synthetic → Routing dataset
    → bge-m3 embedding → Train MFRouter
```
