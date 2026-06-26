# Kế Hoạch Triển Khai MFRouter Cho Custom Task Tiếng Việt

> Mục tiêu: dùng golden dataset sau `scripts/determine_difficulty.py` để train `MFRouter` cho bài toán tiếng Việt.

## 1. Kết Luận Nhanh

- Không cần “model embedding tiếng Việt”.
- `MFRouter` học **model embedding latent** trong lúc train.
- Thứ cần cho tiếng Việt là **query embedding** phù hợp ngôn ngữ.

## 2. Mục Tiêu

1. Chuyển golden dataset sang format train cho `MFRouter`.
2. Tạo query embeddings cho tiếng Việt.
3. Train MFRouter để route giữa các model khả dụng.
4. Kiểm tra inference với custom task.

## 3. Phạm Vi Dữ Liệu

### Input hiện có

- `reviewed_*.jsonl`
- `golden_dataset_*.jsonl` từ `scripts/determine_difficulty.py`

### Output cần tạo

- `routing_data_train.jsonl`
- `routing_data_test.jsonl`
- `query_embeddings.pt` hoặc `.npy`
- `mfrouter.yaml` trỏ đúng data mới

## 4. Schema Train Cho MFRouter

Mỗi query nên được expand thành nhiều dòng, mỗi dòng ứng với một model:

```json
{
  "query": "...",
  "query_id": "...",
  "model_name": "local|gemini|...",
  "performance": 0.0,
  "embedding_id": 12
}
```

### Gợi ý gán nhãn

- `easy`: cả local và gemini đúng -> có thể giữ hoặc loại tùy mục tiêu routing.
- `hard`: local sai, gemini đúng -> đây là tín hiệu học routing tốt nhất.
- `both wrong` / `uncertain`: nên loại khỏi train.

## 5. Query Embedding

### Khuyến nghị

Với tiếng Việt, nên dùng embedding đa ngôn ngữ thay vì Longformer tiếng Anh mặc định.

### Ưu tiên

1. Multilingual sentence embedding.
2. Vietnamese-friendly embedding model.
3. Chỉ giữ Longformer nếu test thấy ổn và cần ít thay đổi nhất.

### Lý do

- `MFRouter` không cần embedding cho model.
- Nó cần vector hóa query để học tương quan query-model.

## 6. Pipeline Đề Xuất

### Bước 1: Tạo golden dataset

- Chạy `scripts/determine_difficulty.py`
- Lưu `golden_dataset_vi.jsonl`

### Bước 2: Chuẩn hóa thành routing data

- Với mỗi query, tạo 1 dòng cho mỗi candidate model.
- Gán `performance` theo kết quả đúng/sai.
- Giữ `embedding_id` để map sang query embedding.

### Bước 3: Sinh query embeddings

- Encode `user_prompt` hoặc `query`.
- Lưu tensor embedding theo thứ tự và gán chỉ số.

### Bước 4: Train MFRouter

- Dùng `configs/model_config_train/mfrouter.yaml`.
- Trỏ `routing_data_train` và `query_embedding_data` sang file mới.

### Bước 5: Evaluate

- Chạy inference trên tập test.
- Đo accuracy routing hoặc top-1 selection.

## 7. Chỗ Cần Xem Lại Trong Code

1. `llmrouter/models/mfrouter/router.py`
- Logic build pairwise samples đang dựa trên `performance` và `embedding_id`.

2. `llmrouter/models/mfrouter/trainer.py`
- Dùng precomputed embeddings nếu có.

3. `configs/model_config_train/mfrouter.yaml`
- Cần thay data path theo dataset mới.

4. `scripts/determine_difficulty.py`
- Đang xuất `golden_dataset_*` tốt cho bước chuẩn bị dữ liệu.

## 8. Decision Cần Chốt

1. Chỉ route giữa `local` và `gemini`, hay nhiều model hơn?
2. Dùng embedding multilingual nào?
3. Có giữ các sample `easy` trong train không, hay chỉ dùng `hard`?

## 9. Kế Hoạch Ngắn Gọn

1. Export golden dataset VI.
2. Convert sang `routing_data_train`.
3. Generate query embeddings.
4. Train MFRouter.
5. Test inference.

## 10. Notebook Chạy Trên Colab

Bạn có thể chạy notebook bằng cách `git clone` repo trên Colab rồi mở trực tiếp các file sau:

- Train: `notebooks/mfrouter/01_mfrouter_training.ipynb`
- Infer: `notebooks/mfrouter/02_mfrouter_inference.ipynb`

### Mẫu setup trên Colab

```python
!git clone <REPO_URL>
%cd LLMRouter
```

Sau đó mở:

```text
notebooks/mfrouter/01_mfrouter_training.ipynb
notebooks/mfrouter/02_mfrouter_inference.ipynb
```

### Ghi chú

- Vì máy local yếu, nên ưu tiên chạy training trong Colab.
- Nếu notebook đang dùng path tương đối, giữ nguyên cấu trúc repo khi clone là đủ.
- Nếu cần, ta có thể chỉnh notebook để mặc định đọc data từ `artifacts/` hoặc từ Google Drive mount.
