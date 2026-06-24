## Pipeline train model

#### Bước 7: Train ML router

Sau khi có `difficulty` label (easy/hard) từ golden dataset pipeline, dùng dataset để train ML router.

Feature đầu vào gồm:

- Semantic embedding của query.
- TF-IDF / BoW / n-gram.
- Text/count features + Feature logic/ cấu trúc + Feature nhạy cảm
- Policy similarity features.

A. Nhóm đặc trưng Logic & Cấu trúc
Structural Features
Logical Feature
Adversarial & Formattin
Domain Intent Heuristics
Vingroup & Lãnh đạo
Violence & Illegal
Jailbreak / Evasion Features
B. Nhóm đặc trưng Ngữ nghĩa
Dùng các model encoder siêu nhanh (như bge-small-en-v1.5 hoặc PhoBERT cho tiếng Việt) để tính Cosine Similarity.
Chi tiết Feature engineer

1. Nhóm Đặc trưng Cấu trúc & Độ dài (Structural Features)
   word_count: Tổng số từ trong query.
   char_length: Tổng số ký tự.
   avg_word_per_sentence: Trung bình số từ trên mỗi câu (dấu hiệu của việc diễn đạt dài dòng, vòng vo).
   sentence_count: Số lượng câu hỏi/mệnh đề độc lập trong một lần chat.
2. Nhóm Đặc trưng Logic & Mệnh đề (Logical Features)
   conditional_count (Điều kiện): Đếm các từ: "nếu", "thì", "giả sử", "trường hợp", "đặt giả thiết". Câu nào có nhiều từ này chắc chắn logic từ medium đến high.
   contrast_count (Tương phản/Bẻ lái): Đếm các từ: "nhưng", "tuy nhiên", "mặc dù", "ngược lại", "khác với".
   reasoning_count (Suy luận): Đếm các từ: "tại sao", "nguyên nhân", "lý do", "giải thích".
   sequential_count (Tuần tự): Đếm các từ: "bước 1", "đầu tiên", "sau đó", "tiếp theo", "cuối cùng".
3. Nhóm Đặc trưng Dấu câu & Định dạng (Adversarial & Formatting)
   quote_count: Đếm số lượng dấu ngoặc kép "" hoặc ngoặc đơn ''. Dấu hiệu điển hình của việc trích dẫn một đoạn văn bản lạ hoặc ra lệnh gài bẫy.
   bracket_count: Đếm số lượng ngoặc tròn (), ngoặc vuông [], hoặc ngoặc nhọn {}. Rất phổ biến trong các query liên quan đến JSONL, code, hoặc ép định dạng đầu ra.
   backtick_count: Đếm dấu `. Dấu hiệu chắc chắn 90% là user đang hỏi về lập trình hoặc format Markdown.
   question_mark_count: Đếm dấu ?. Nếu một query có từ 3 dấu hỏi chấm trở lên, đó thường là một chuỗi câu hỏi dồn dập cần xử lý ngắt nhịp (Uncertainty) hoặc chia nhỏ tác vụ.
4. Nhóm Đặc trưng Từ khóa Đặc thù (Domain Intent Heuristics)
   is_math_logic: Chứa các từ "tính toán", "phép tính", "đạo hàm", "phương trình", "logic".
   is_coding: Chứa các từ "code", "hàm", "biến", "lỗi", "bug", "thuật toán", "tối ưu".
5. Nhóm Đặc trưng Rủi ro Thương hiệu & Cá nhân (Vingroup & Lãnh đạo)
   vingroup_entity_count: Đếm sự xuất hiện của các từ "Vingroup", "VinFast", "Vinhomes", "Vin"...
   vip_name_count: Đếm danh sách VIP đã nêu trong policy: "Phạm Nhật Vượng", "Tô Lâm", "Tập Cận Bình", "Bác Hồ"... (Chỉ cần có tên VIP xuất hiện, độ lắt léo/rủi ro tự động tăng).
   slang_insult_count: Đếm các từ lóng công kích đặc thù: "Vin nô", "Vượng Vin", "Vin nát", "đồng chí X", "bọn lừa đảo".
6. Nhóm Đặc trưng Rủi ro Xã hội & Pháp lý (Violence & Illegal)
   harm_action_count: Đếm các động từ mạnh: "tự tử", "chết", "đánh", "giết", "đầu độc", "chế tạo", "đốt".
   gambling_count: Đếm các từ "cá độ", "casino", "đặt cược", "tỷ lệ kèo", "đá gà".
7. Nhóm Đặc trưng "Lách luật" (Jailbreak / Evasion Features)
   roleplay_count: Đếm cụm từ: "viết tiểu thuyết", "kịch bản phim", "đóng vai", "giả sử bạn là". (User thường dùng cách này để lách luật bạo lực, khiêu dâm trẻ em hoặc chế tạo bom).
   opinion_seeking_count: Đếm cụm từ: "quan điểm của", "nghĩ gì về", "nhận xét thế nào". (Dấu hiệu của bẫy chính trị hoặc bẫy Vingroup).
   rumor_framing_count: Đếm cụm từ: "nghe nói", "có phải", "thực hư chuyện", "đồn rằng". (Dấu hiệu của vi phạm Policy số 6: Tin đồn).

### 12. Các phương pháp ML Router để test

Có nhiều phương pháp router có sẵn trong codebase. Thứ tự test đề xuất dựa trên độ phức tạp và hiệu quả:

#### Thứ tự test đề xuất

| Priority | Router | Type | Lý do test trước |
|----------|--------|------|------------------|
| **1st** | **MLP Router** | Neural Network | Train nhanh (sklearn), inference nhanh, tune được nhiều hyperparams, phù hợp dataset ~2K |
| **2nd** | **SVM Router** | Kernel-based | RBF kernel bắt non-linear patterns, memory efficient (chỉ lưu support vectors) |
| **3rd** | **RouterDC** | Contrastive Learning | State-of-the-art, multilingual mDeBERTa, cần GPU, phức tạp hơn |
| **4th** | **KNN Router** | Instance-based | No training, interpretable, inference O(N) chậm - tốt làm baseline |
| **Skip** | **ELO Router** | Rating | Query-agnostic - luôn chọn 1 model, không phù hợp bài toán này |

#### 12.1 MLP Router (Test đầu tiên)

**Đặc điểm:**
- Framework: sklearn's `MLPClassifier`
- Hidden layers default: `[128, 64]`
- Activation: ReLU
- Solver: Adam (tốt cho dataset ~2K)

**Input features:**
- Concatenate: embeddings + TF-IDF + count features → feature vector
- Cần normalize features trước khi train

**Pros:**
- ✅ Train nhanh (minutes)
- ✅ Inference nhanh (O(1))
- ✅ Tune được nhiều hyperparameters
- ✅ Works well với ~2K samples

**Cons:**
- ❌ Cần normalize features
- ❌ Có thể overfit nếu không tune kỹ

**Config path:** `llmrouter/models/mlprouter/`

#### 12.2 SVM Router (Test thứ hai)

**Đặc điểm:**
- Framework: sklearn's `SVC` với RBF kernel
- Kernel: RBF (default, recommended)
- Regularization: C parameter (default=1.0)
- Probability: True để có confidence scores

**Input features:**
- Features cần normalize (SVM sensitive to feature scales)
- Kernel trick map vào high-dimensional space

**Pros:**
- ✅ Kernel trick bắt non-linear patterns hiệu quả
- ✅ Memory efficient (chỉ lưu support vectors)
- ✅ Margin maximization → robust generalization

**Cons:**
- ❌ Chậm nếu dataset > 10K (O(n²) to O(n³))
- ❌ Cần normalize features
- ❌ Hyperparameter sensitive (C, gamma)

**Config path:** `llmrouter/models/svmrouter/`

**Hyperparameter tuning guide:**
```
Low C (0.01-0.1):   Wide margin, more generalization
Medium C (1.0):      Balanced (default)
High C (10-100):     Narrow margin, less misclassification tolerance

Low gamma (0.001):   Smooth decision boundary
Medium gamma:        Balanced (use "scale")
High gamma (1-10):   Complex boundary, risk of overfitting
```

#### 12.3 RouterDC (Test thứ ba - nếu cần)

**Đặc điểm:**
- Framework: PyTorch + HuggingFace Transformers
- Backbone: mDeBERTa-v3-base (~280M params)
- Learning: Dual contrastive (sample-LLM + sample-sample + cluster)
- Cần GPU mạnh

**Pros:**
- ✅ State-of-the-art encoder
- ✅ Multilingual (100+ languages)
- ✅ Multi-level contrastive learning

**Cons:**
- ❌ Cần GPU (16GB+ recommended)
- ❌ Training chậm
- ❌ Cold start cho new LLMs
- ❌ Nhiều hyperparameters phức tạp

**Config path:** `llmrouter/models/routerdc/`

#### 12.4 KNN Router (Baseline)

**Đặc điểm:**
- Framework: sklearn's `KNeighborsClassifier`
- No training - lazy learning
- K value: 5 (default, thử 3, 7, 10)

**Pros:**
- ✅ No training required
- ✅ Interpretable - inspect được neighbors
- ✅ Incremental learning - thêm data dễ
- ✅ Works well với small data

**Cons:**
- ❌ Inference O(N) - chậm với large dataset
- ❌ Memory intensive (lưu tất cả examples)
- ❌ Curse of dimensionality

**Config path:** `llmrouter/models/knnrouter/`

#### 12.5 ELO Router (Không phù hợp - Skip)

**Lý do skip:**
- Query-agnostic - luôn chọn highest-rated model duy nhất
- Không tận dụng được query features
- Bài toán này cần query-specific routing

**Config path:** `llmrouter/models/elorouter/`

---

### 13. Test Plan chi tiết

```
Phase 1: MLP Router (1-2 ngày)
├── Prepare features: embeddings + TF-IDF + count features
├── Train sklearn MLPClassifier ([128, 64], Adam, 500 iterations)
├── Evaluate: accuracy, F1, confusion matrix, FN rate
└── Tune hyperparameters nếu cần

Phase 2: SVM Router (1-2 ngày)
├── Normalize features
├── Train sklearn SVC (RBF kernel, C=1.0, gamma='scale')
├── Compare vs MLP
└── Evaluate: accuracy, F1, confusion matrix, FN rate

Phase 3: RouterDC (3-5 ngày) - optional
├── Chỉ nếu Phase 1+2 chưa đủ tốt
├── Cần GPU setup
└── Complex contrastive learning setup
```

### 14. Mục tiêu metric cần đạt

| Metric | Target | Ghi chú |
|--------|--------|---------|
| Accuracy | > 85% | Tổng quát |
| Macro F1 | > 80% | Cân bằng easy/hard |
| Recall (hard) | > 90% | Quan trọng nhất - không bỏ lọt hard cases |
| FN rate (hard→easy) | < 5% | Lỗi nghiêm trọng nhất |

### 15. Inference deployment

Sau khi train xong, model có thể deploy:

```python
# Local inference
router = MLPRouter(yaml_path="configs/model_config_test/mlprouter.yaml")
result = router.route_single({"query": "..."})
# result['model_name'] → "Qwen3-4B-Instruct-2507" hoặc "gemini-3.1-flash-lite"
```

Model ML có thể thử:

Model ML và các router /home/thinh/projects/VSF/LLMRouter/llmrouter/models

#### Bước 8: Đánh giá router

Metric chính không chỉ là accuracy tổng, mà cần tập trung vào lỗi route sai xuống model nhỏ.

Các metric nên dùng:

- Accuracy.
- Macro F1.
- Recall của nhóm hard (cần Gemini - model mạnh).
- False negative rate của nhóm hard (easy bị classify thành hard thì không sao, nhưng hard bị classify thành easy thì nghiêm trọng).
- Confusion matrix.

Lỗi quan trọng nhất cần giảm:

```text
Đáng lẽ cần hard (Gemini) nhưng router lại chọn easy (Qwen3-4B local).
```

---

### 9. Kết quả kỳ vọng

Sau khi hoàn thành golden dataset và benchmark 2 model, ta sẽ có:

1. Bộ query khoảng 1,980 queries bao phủ 12 policies.
2. Kết quả so sánh local (Qwen3-4B) / high (Gemini) trên từng query.
3. Nhãn difficulty thực nghiệm cho từng query:
   - **easy**: local_correct=True AND gemini_correct=True → dùng Qwen3-4B (local)
   - **hard**: local_correct=False AND gemini_correct=True → cần Gemini (high)
   - (Both wrong = excluded, không hữu ích cho training)
4. Bộ dữ liệu dùng để train ML router phân loại easy/hard.
5. Báo cáo cho biết:
   - Model local xử lý tốt nhóm nào.
   - Nhóm nào cần Gemini.
   - Policy nào khó nhất.
   - Dạng query nào dễ bị bỏ lọt.
   - Router nên ưu tiên escalation trong các trường hợp nào.

---

### 10. Tóm tắt ngắn

Golden dataset được thiết kế theo 3 nhóm: single-policy, multi-policy và no-policy. Single-policy dùng để test từng policy riêng lẻ, multi-policy dùng để test các case chồng chéo nhiều policy, còn no-policy dùng để kiểm tra over-refusal. Các mức low/medium/high ban đầu chỉ là designed complexity để cân bằng dữ liệu, không phải nhãn độ khó cuối cùng.

Sau khi sinh query, mỗi query sẽ được chạy qua 2 model: **local** (Qwen3-4B-Instruct-2507) và **high** (Gemini-3.1-flash-lite). Kết quả của các model sẽ được đánh giá thông qua bước LLM-as-Judge, trong đó một model mạnh hơn sẽ so sánh output với expected label và đưa ra nhận định đúng/sai, mức độ phù hợp với policy, confidence và lý do đánh giá. Các trường hợp có độ tin cậy thấp, model bất đồng mạnh hoặc thuộc nhóm borderline sẽ được chuyển sang bước Human Review để kiểm tra lại và xác nhận nhãn cuối cùng.

Sau khi hoàn tất bước LLM-as-Judge và Human Review, kết quả được dùng để gán **difficulty label** thực nghiệm:
- **easy**: local_correct=True AND gemini_correct=True → route xuống local model
- **hard**: local_correct=False AND gemini_correct=True → route lên Gemini

Difficulty label này là nhãn chính để train ML router bằng semantic embedding, TF-IDF/count features và policy similarity features.

Mục tiêu cuối cùng là tối ưu chi phí bằng cách dùng Qwen3-4B local cho các case easy, đồng thời tránh bỏ lọt các query vi phạm policy cần Gemini, đồng thời đảm bảo chất lượng nhãn thông qua quy trình đánh giá kết hợp giữa LLM-as-Judge và Human Review.

---

### 11. Mapping giữa Golden Dataset và Router Labels

| Golden Dataset Field | Router Logic | Model Gọi |
|---------------------|--------------|-----------|
| `difficulty = easy` | Route xuống local | Qwen3-4B-Instruct-2507 |
| `difficulty = hard` | Route lên high | gemini-3.1-flash-lite |
| `human_reviewed = true` | Đã được human verify | Dùng judgment cuối cùng |

**Excluded from training:**
- Queries where both models are wrong (both_correct=False) → không hữu ích cho routing
- Queries with `uncertain` consensus → cần human review trước khi train
