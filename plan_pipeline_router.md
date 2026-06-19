## Pipeline train model

#### Bước 7: Train ML router

Sau khi có `route_label`, dùng dataset để train ML router.

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
   conditional_count (Điều kiện): Đếm các từ: "nếu", "thì", "giả sử", "trường hợp", "đặt giả thiết". Câu nào có nhiều từ này chắc chắn logic từ mức Medium đến High.
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

Model ML có thể thử:

Model ML và các router /home/thinh/projects/VSF/LLMRouter/llmrouter/models

#### Bước 8: Đánh giá router

Metric chính không chỉ là accuracy tổng, mà cần tập trung vào lỗi route sai xuống model nhỏ.

Các metric nên dùng:

- Accuracy.
- Macro F1.
- Recall của nhóm cần escalation.
- False negative rate của escalation.
- Confusion matrix.

Lỗi quan trọng nhất cần giảm:

```text
Đáng lẽ cần medium/high nhưng router lại chọn local.
```

---

### 9. Kết quả kỳ vọng

Sau khi hoàn thành golden dataset và benchmark 3 model, ta sẽ có:

1. Bộ query khoảng 1,000 mẫu bao phủ 12 policy.
2. Kết quả so sánh local / medium / high trên từng query.
3. Nhãn route_label thực nghiệm cho từng query.
4. Bộ dữ liệu dùng để train ML router.
5. Báo cáo cho biết:
   - Model local xử lý tốt nhóm nào.
   - Nhóm nào cần medium/high.
   - Policy nào khó nhất.
   - Dạng query nào dễ bị bỏ lọt.
   - Router nên ưu tiên escalation trong các trường hợp nào.

---

### 10. Tóm tắt ngắn

Golden dataset được thiết kế theo 3 nhóm: single-policy, multi-policy và no-policy. Single-policy dùng để test từng policy riêng lẻ, multi-policy dùng để test các case chồng chéo nhiều policy, còn no-policy dùng để kiểm tra over-refusal. Các mức low/medium/high ban đầu chỉ là designed complexity để cân bằng dữ liệu, không phải nhãn độ khó cuối cùng.

Sau khi sinh query, mỗi query sẽ được chạy qua 3 model local, medium và high. Kết quả của các model sẽ được đánh giá thông qua bước LLM-as-Judge, trong đó một model mạnh hơn sẽ so sánh output với expected label và đưa ra nhận định đúng/sai, mức độ phù hợp với policy, confidence và lý do đánh giá. Các trường hợp có độ tin cậy thấp, model bất đồng mạnh hoặc thuộc nhóm borderline sẽ được chuyển sang bước Human Review để kiểm tra lại và xác nhận nhãn cuối cùng.

Sau khi hoàn tất bước LLM-as-Judge và Human Review, kết quả được dùng để gán route_label thực nghiệm: local, medium, high hoặc human_review. Route_label này là nhãn chính để train ML router bằng semantic embedding, TF-IDF/count features và policy similarity features.

Mục tiêu cuối cùng là tối ưu chi phí bằng cách dùng model nhỏ cho các case đủ đơn giản, đồng thời tránh bỏ lọt các query vi phạm policy cần model mạnh hơn, đồng thời đảm bảo chất lượng nhãn thông qua quy trình đánh giá kết hợp giữa LLM-as-Judge và Human Review.
