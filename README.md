# LLMRouter

Repo này hiện có hai lớp nội dung:

1. **Luồng safety / golden dataset** đang là hướng làm việc chính.
2. **Các router và notebook cũ từ paper codebase** vẫn còn nằm trong repo, nhưng đa phần là legacy hoặc tùy chọn.

## Cấu trúc thư mục

```text
.
├── safety/                      # Canonical safety namespace: tasks, router, dataset, common
├── llmrouter/                   # Core package: prompts, data, evaluation, models
├── scripts/                     # Entry points CLI cho safety pipeline
├── configs/                     # YAML config cho pipeline / model / prompt-related settings
├── notebooks/                   # Notebook demo / training / inference cũ
├── archive/                     # Mã và tài liệu legacy đã tách ra
├── policy.csv                   # Nguồn policy cho safety pipeline
└── artifacts/                   # Output sinh ra từ pipeline
```

## Luồng safety hiện tại

Luồng chính của repo là:

1. Đọc `policy.csv` hoặc policy markdown.
2. Sinh bộ query golden.
3. Chạy router nội bộ để ước lượng `route`, `risk`, `difficulty`, `violation_status`.
4. Nếu bật cấu hình, gọi thêm các model tier `local / medium / high`.
5. Xuất `train.jsonl`, `dev.jsonl`, `test.jsonl`, `golden_dataset.jsonl`, `manifest.json`.

### Entry point

- `scripts/generate_safety_queries.py`: chỉ sinh query, không benchmark model.
- `scripts/run_safety_evaluation.py`: chạy end-to-end pipeline, gồm sinh query, routing, judge, export.
- `safety/dataset/pipeline.py`: chứa logic chính của builder.

## Prompt lưu ở đâu

Repo này tách prompt theo 3 kiểu:

### 1. Prompt YAML dùng chung

Nằm trong `llmrouter/prompts/`:

- `task_prompts/`: prompt cho benchmark/task format.
- `agentic_role/`: prompt cho agent / decomposition / multi-round reasoning.
- `router_prompts/`: prompt cho router-specific cases.
- `data_prompts/`: prompt cho chuyển đổi dữ liệu.

Các prompt này được load bằng:

```python
from llmrouter.prompts import load_prompt_template
template = load_prompt_template("task_mc")
```

### 2. Prompt riêng cho safety task

Nằm trong `safety/tasks/`:

- `safety/tasks/task_prompts/task_safety_aware_policy.yaml`
- `safety/tasks/safety_aware_policy.py`

File `safety/tasks/safety_aware_policy.py` đăng ký task `safety_aware_policy` và dùng `load_prompt_template("task_safety_aware_policy")`.

Quan trọng: `llmrouter/prompts/__init__.py` ưu tiên tìm prompt trong `safety/tasks/` trước, rồi mới fallback sang `llmrouter/prompts/`. Nghĩa là prompt safety có thể override prompt built-in nếu trùng tên.

### 3. Prompt hardcode trong code

Một số router đời cũ vẫn giữ prompt trực tiếp trong Python, ví dụ:

- `llmrouter/models/router_r1/prompt_pool.py`

Đây là ngoại lệ, không phải cách lưu prompt khuyến nghị cho task mới.

## Query generation nằm ở đâu

Phần sinh query của safety pipeline không nằm trong YAML prompt file. Nó nằm trong code:

- `safety/dataset/pipeline.py`

Trong đó builder tạo 3 nhóm query:

- `single_policy`
- `multi_policy`
- `no_policy`

Các template sinh query hiện được viết trực tiếp trong Python, ví dụ các cụm prompt tiếng Việt cho mức `low / medium / high`.

## Tổ chức code quan trọng nhất

### `safety/router/`

Chứa:

- router nội bộ
- pipeline sinh golden dataset
- config cho safety workflow

### `safety/tasks/`

Chứa:

- registry task
- system prompt riêng cho safety
- logic format prompt cho benchmark safety

### `llmrouter/prompts/`

Chứa:

- prompt YAML dùng chung cho toàn repo
- loader `load_prompt_template()`

### `llmrouter/data/`

Chứa:

- pipeline dữ liệu cho các router legacy
- embedding / evaluation utilities
- sample config và README riêng

### `llmrouter/models/`

Chứa nhiều router family:

- `knnrouter`
- `graphrouter`
- `mfrouter`
- `personalizedrouter`
- `router_r1`
- `hybrid_llm`
- `automix`

Nếu bạn chỉ làm safety golden dataset, phần này chủ yếu là tham khảo hoặc legacy.

## Chạy nhanh

Sinh query:

```bash
python scripts/generate_safety_queries.py \
  --config configs/safety/query_generation.yaml
```

Chạy end-to-end:

```bash
python scripts/run_safety_evaluation.py \
  --config configs/safety/router.yaml \
  --policy-file policy.csv \
  --output-dir artifacts/safety_golden
```

## Gợi ý đọc code

Nếu bạn đang mới vào repo, đọc theo thứ tự này:

1. `AGENT.md`
2. `REPO_MAP.md`
3. `safety/tasks/README.md`
4. `safety/router/README.md`
5. `safety/dataset/README.md`
6. `llmrouter/prompts/README.md`

## Ghi chú

- `archive/` là nơi chứa mã và tài liệu cũ.
- Output sinh ra từ pipeline nên để trong `artifacts/`.
- Nếu thêm prompt mới, ưu tiên YAML trong `llmrouter/prompts/` hoặc `safety/tasks/task_prompts/` thay vì hardcode trong Python.
