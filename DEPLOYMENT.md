# Deployment Guide - Safety Router MFRouter

> Hướng dẫn đóng gói và deploy model MFRouter đã train.

---

## 1. Files cần tải từ Colab về local

Trên Colab, chạy các lệnh sau để download:

```python
from google.colab import files
import os

# 1. Trained model
files.download("saved_models/mfrouter/mfrouter_vni.pkl")
```

**Hoặc nếu muốn download cả folder:**
```python
!zip -r saved_models.zip saved_models/
files.download("saved_models.zip")
```

### Files cần thiết (đã có sẵn trong repo local)

| File | Có sẵn local? | Cần upload Colab? |
|------|---------------|-------------------|
| `llmrouter/` source code | ✅ Yes | ❌ No (Colab clone from git) |
| `scripts/train_mfrouter.py` | ✅ Yes | ❌ No |
| `scripts/test_mfrouter.py` | ✅ Yes | ❌ No |
| `scripts/convert_to_routing_ds.py` | ✅ Yes | ❌ No |
| `configs/model_config_train/mfrouter.yaml` | ✅ Yes | ❌ No |
| `artifacts/routing/*` | ✅ Yes (do local script) | ❌ No |
| **`saved_models/mfrouter/mfrouter_vni.pkl`** | ❌ **No (chỉ có trên Colab)** | **Cần download về** |

---

## 2. Files cần có trên production server

```
deploy/
├── llmrouter/                          # Source code (git clone hoặc copy)
├── scripts/
│   ├── test_mfrouter.py                # Inference script
│   └── convert_to_routing_ds.py        # Nếu cần re-embed
├── configs/
│   └── model_config_train/mfrouter.yaml
├── artifacts/
│   └── routing/
│       ├── llm_data.json
│       └── (không cần routing_data nếu chỉ inference)
├── saved_models/
│   └── mfrouter/
│       └── mfrouter_vni.pkl            # ← Download từ Colab
└── .env                                # API keys
```

---

## 3. Setup trên server mới

### 3.1 Install dependencies
```bash
pip install torch transformers pandas numpy pyyaml
```

### 3.2 Download BAAI/bge-m3 model
Lần đầu chạy `test_mfrouter.py`, model sẽ tự động download từ HuggingFace (~2.3GB).

Hoặc download thủ công để tránh download mỗi lần restart:
```bash
python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('BAAI/bge-m3', cache_dir='/path/to/cache')
AutoModel.from_pretrained('BAAI/bge-m3', cache_dir='/path/to/cache')
"
```

### 3.3 Test inference
```bash
python scripts/test_mfrouter.py \
    --config configs/model_config_train/mfrouter.yaml \
    --model-path saved_models/mfrouter/mfrouter_vni.pkl \
    --embedding-model BAAI/bge-m3 \
    --device cuda \
    --text "Cho tôi hỏi về Vingroup" \
    --text "VinFast sản xuất xe gì?"
```

---

## 4. Embedding model - Cẩn thận

**Critical**: Inference PHẢI dùng cùng `BAAI/bge-m3` như training.

Nếu đổi embedding model:
- Cần re-run Phase 2 (`convert_to_routing_ds.py --embeddings-only`)
- Cần re-train MFRouter
- Cần update `text_dim` trong `mfrouter.yaml`

---

## 5. Liên hệ bge-m3 model file

Model sẽ được cache tại:
- Linux: `~/.cache/huggingface/hub/`
- Colab: `/root/.cache/huggingface/hub/`

---

## 6. MFRouter model size

| File | Size |
|------|------|
| `mfrouter_vni.pkl` | ~1.5MB |
| `BAAI/bge-m3` (pytorch_model.bin) | ~2.27GB |
| `query_embeddings.pt` | ~3.4MB |
| **Total per deployment** | **~2.3GB** |

---

## 7. Quick sanity check trên server mới

```bash
# 1. Test model load
python -c "
import torch
emb = torch.load('artifacts/routing/query_embeddings.pt', map_location='cpu')
print(f'Embeddings: {len(emb)}, dim={emb[0].shape[0]}')
"

# 2. Test inference
python scripts/test_mfrouter.py \
    --model-path saved_models/mfrouter/mfrouter_vni.pkl \
    --text "test query" \
    --device cuda

# 3. Test latency
python scripts/test_mfrouter.py \
    --model-path saved_models/mfrouter/mfrouter_vni.pkl \
    --text-file my_queries.txt \
    --device cuda
```

Expected output: routing decisions + latency < 500ms.

---

## 8. Files KHÔNG cần deploy (chỉ training artifacts)

- `artifacts/routing/routing_data_*.jsonl` - chỉ dùng để train
- `artifacts/routing/query_data_*.jsonl` - chỉ dùng để test/eval
- `artifacts/routing/unique_query_texts.txt` - intermediate file
- `artifacts/answers/*.jsonl` - golden dataset gốc (đã được convert)
- `artifacts/golden/splits_vni/*.jsonl` - splits gốc

Nếu chỉ cần inference (không cần re-train), KHÔNG cần các file này.
