# 📑 Tóm tắt Bài báo Khoa học: RouteProfile

**Tiêu đề gốc**: `RouteProfile: Graph-Based Profiling for Cold-Start LLM Routing`  
**Tác giả**: Jingjun Xu, Hongji Pu, Tao Feng, Haozhen Zhang, Jiaxuan You, Ge Liu  
*(University of Illinois Urbana-Champaign & Nanyang Technological University)*  
**Mã nguồn**: [ulab-uiuc/RouteProfile](https://github.com/ulab-uiuc/RouteProfile)

---

## 🎯 1. Bối cảnh & Bài toán "Cold-Start LLM Routing"

Trong hệ sinh thái LLM phát triển như vũ bão, các mô hình có thế mạnh và điểm yếu rất khác nhau đối với từng lĩnh vực (toán, code, lý luận, kiến thức). Điều này dẫn đến sự ra đời của **LLM Router** (Bộ định tuyến LLM) nhằm chọn ra mô hình tối ưu nhất cho từng câu hỏi để tối ưu hóa cả hiệu năng lẫn chi phí.

Tuy nhiên, các hệ thống định tuyến truyền thống gặp phải **nút thắt cổ chai thích ứng (adaptation bottleneck)**:
* **Chi phí re-train khổng lồ**: Mỗi khi có một LLM mới được phát hành, hệ thống phải chạy thử nghiệm quy mô lớn trên hàng ngàn câu hỏi để lấy dữ liệu tương tác (Query-Response-Reward), sau đó huấn luyện lại (re-train) toàn bộ bộ định tuyến.
* **Bài toán Cold-Start**: Làm thế nào để định tuyến câu hỏi cho một LLM mới ra mắt mà **chưa hề có bất kỳ lịch sử tương tác nào**?

### Quan sát cốt lõi của tác giả:
Các LLM mới khi ra mắt luôn đi kèm các **tín hiệu công khai (public signals)** như: *họ mô hình (model family), mô tả kỹ thuật (model card), điểm số benchmark (reported benchmark scores) và lĩnh vực benchmark (benchmark domains)*. Bài báo nghiên cứu cách khai thác các tín hiệu công khai này để xây dựng hồ sơ (profile) cho LLM, hỗ trợ định tuyến ngay lập tức mà không cần dữ liệu lịch sử.

---

## 🎨 2. Phương pháp RouteProfile: Hồ sơ hóa LLM dựa trên Đồ thị

`RouteProfile` tổ chức các thông tin công khai không đồng nhất, thưa thớt thành một **Đồ thị không đồng nhất (Heterogeneous Graph)** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

* **Các nút (Nodes - $\mathcal{V}$)**:
  * Nút Mô hình ($v_m$)
  * Nút Họ mô hình ($v_f$) - đại diện cho dòng kiến trúc, nhà phát triển (ví dụ: Qwen, Llama).
  * Nút Benchmark ($v_b$) - điểm số kiểm thử công khai (ví dụ: MMLU, GSM8K).
  * Nút Domain ($v_d$) - lĩnh vực bao phủ của benchmark (ví dụ: Math, Coding).
* **Các cạnh (Edges - $\mathcal{E}$)**:
  * Kết nối Model-Family ($e_{mf}$)
  * Kết nối Model-Benchmark ($e_{mb}$) - chứa trọng số chính là điểm số hiệu năng công khai.
  * Kết nối Benchmark-Domain ($e_{bd}$)

### Không gian Thiết kế Hồ sơ (4 Chiều cốt lõi):

Tác giả nghiên cứu cách hàm tổng hợp thông tin đồ thị $f$ tạo ra Profile cho LLM qua 4 chiều:
1. **Organizational Form (Hình thức Tổ chức)**: 
   * *Flat*: Nối trực tiếp văn bản mô tả thành chuỗi phẳng.
   * *Structured*: Sử dụng cấu trúc đồ thị thông qua mạng tích chập đồ thị (GNN).
2. **Representation Type (Loại biểu diễn)**:
   * *Text*: Tóm tắt bằng ngôn ngữ tự nhiên sử dụng LLM mạnh (như GPT-4o-mini).
   * *Embedding*: Mã hóa văn bản thành vector đặc trưng dày đặc (dense vectors).
3. **Aggregation Depth (Độ sâu tổng hợp)**: Khoảng cách lan truyền thông tin trên đồ thị (Hop $\in \{0, 1, 2, 3, 4\}$).
4. **Learning Configuration (Cấu hình học tập)**:
   * *Training-free*: Tổng hợp trực tiếp không qua huấn luyện.
   * *Trainable*: Huấn luyện mô hình GNN tự giám sát (self-supervised) để khôi phục các đặc trưng bị che (masked reconstruction objective).

---

## 🛠️ 3. Ba Bộ Định Tuyến (Routers) được Đánh giá

Để đo lường chất lượng của hồ sơ LLM (LLM Profiles), bài báo đánh giá trên 3 bộ định tuyến đại diện cho 3 cơ chế định tuyến khác nhau:

### 1. SimRouter (Bộ định tuyến không tham số - Không huấn luyện)
* **Cơ chế**: Đo lường trực tiếp độ tương đồng (ví dụ: Cosine Similarity) giữa Vector câu hỏi của User (Query Embedding) và Vector hồ sơ của candidate LLM (LLM Profile).
* **Đặc điểm**: Hoàn toàn không huấn luyện. Đây là bộ định tuyến thuần túy nhất để đánh giá trực tiếp chất lượng biểu diễn ngữ nghĩa của Profile.

### 2. MLPRouter (Bộ định tuyến học máy - Trainable)
* **Cơ chế**: Sử dụng hai mạng thần kinh nhân tạo (MLP - Multi-Layer Perceptron) độc lập để chiếu Query Embedding và LLM Profile Embedding vào một không gian ẩn chung (shared latent space). Sau đó xếp hạng các mô hình dựa trên độ tương đồng trong không gian chiếu này.
* **Đặc điểm**: Được huấn luyện trên dữ liệu tương tác của các LLM cũ, sau đó kiểm tra khả năng tích hợp mô hình mới (unseen LLM) dựa trên hồ sơ công khai của nó.

### 3. GraphRouter (Bộ định tuyến cấu trúc đồ thị)
* **Cơ chế**: Xây dựng đồ thị không đồng nhất chứa cả câu hỏi của người dùng và các candidate LLM. Áp dụng mạng đồ thị GNN để mô tả sự tương tác ngữ cảnh sâu sắc giữa Task, Query và LLM.
* **Đặc điểm**: Phức tạp nhất, khai thác triệt để các mối quan hệ cấu trúc giữa các thực thể.

---

## 🏆 4. Các Phát hiện Quan trọng từ Thực nghiệm

Qua kiểm thử trên 12 bộ dữ liệu (Toán, Lý luận, Code, Kiến thức) và 8 LLM ứng viên (Qwen2, Llama3, Gemma2, Mistral, Mixtral,...), bài báo rút ra 3 kết luận lớn:

1. **Cấu trúc Đồ thị vượt trội hơn hẳn dạng phẳng (Flat baseline)**: Trong chế độ cold-start không huấn luyện, việc tổ chức thông tin dưới dạng đồ thị có cấu trúc giúp nâng cao rõ rệt hiệu năng định tuyến của `SimRouter`.
2. **Thông tin "Họ mô hình" (Model Family) đáng tin cậy hơn thông tin "Lĩnh vực" (Benchmark Domain)**: Việc biết một mô hình thuộc kiến trúc nào (ví dụ: dòng Llama) giúp suy luận ra năng lực của nó tốt hơn việc chỉ dựa vào điểm số các bài test domain công khai (vốn dễ bị overfit hoặc dữ liệu bẩn).
3. **Tích hợp LLM mới cần sự phối hợp (Co-design)**: Để tích hợp hoàn hảo một LLM mới vào hệ thống đã đóng băng (freeze), việc kết hợp giữa GNN dạng nhúng (Embedding-based GNN) và học máy MLP mang lại kết quả tối ưu nhất.
