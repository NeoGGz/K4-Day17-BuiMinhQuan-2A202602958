# Báo Cáo Phân Tích & Benchmark Hệ Thống Memory Cho AI Agent
**Phase 2, Track 3, Day 17: Memory Systems for AI Agent**

---

## 1. Tổng Quan Kiến Trúc Hệ Thống Memory

Bài lab triển khai và so sánh thực nghiệm giữa 2 kiến trúc Agent trên cùng tập dữ liệu benchmark chuẩn tiếng Việt:

1. **Baseline Agent (Agent A)**:
   - **Cơ chế**: Chỉ sở hữu *Short-term Memory* (Within-session memory) lưu trong RAM theo `thread_id`.
   - **Hạn chế**: Khi bước sang một phiên mới (fresh `thread_id`), Agent mất toàn bộ ngữ cảnh quá khứ, không có khả năng nhận diện hay nhớ lại thông tin người dùng.

2. **Advanced Agent (Agent B)**:
   - **Cơ chế 3 lớp (3-tier Memory Architecture)**:
     - **Short-term Memory**: Lưu trữ ngữ cảnh hội thoại cục bộ hiện tại.
     - **Persistent Memory (`User.md`)**: Lưu trữ hồ sơ người dùng bền vững trên ổ đĩa (`state/profiles/<user>/User.md`) theo định dạng Markdown có cấu trúc.
     - **Compact Memory**: Bộ quản lý bộ nhớ nén (`CompactMemoryManager`), tự động tóm tắt các lượt hội thoại cũ khi tổng số token vượt ngưỡng (`compact_threshold_tokens`), đồng thời giữ lại $K$ lượt gần nhất (`compact_keep_messages`).
   - **Mở rộng Đa Provider**: Hỗ trợ 6 provider (`openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`) linh hoạt cho cả chế độ live và offline.

---

## 2. Kết Quả Thực Nghiệm Benchmark

Dữ liệu benchmark gồm 2 bộ test:
- **Standard Benchmark (`data/conversations.json`)**: 10 hội thoại, mỗi hội thoại 10 lượt của người dùng `dungct`, kèm 14 câu hỏi recall chéo phiên (hỏi ở thread mới).
- **Long-Context Stress Benchmark (`data/advanced_long_context.json`)**: 1 hội thoại siêu dài 16 lượt mang nhiều dữ liệu nhiễu và tin tức của `dungct_stress`, kèm 3 câu hỏi recall tổng hợp.

### Bảng 1: Standard Benchmark (10 hội thoại thường)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 1,548 | 14,641 | 4% | 0.42 | 0 B | 0 |
| **Advanced Agent** | 5,035 | 33,602 | **100%** | **1.00** | 343 B | 3 |

### Bảng 2: Long-Context Stress Benchmark (16 lượt tải nặng)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 256 | 22,144 | 0% | 0.40 | 0 B | 0 |
| **Advanced Agent** | 1,778 | **11,148** | **100%** | **1.00** | 277 B | **14** |

---

## 3. Phân Tích Chuyên Sâu Các Trade-Off

### 3.1. Vì sao Advanced Agent đạt Recall vượt trội so với Baseline (100% vs 0-4%)?
- **Bản chất**: Baseline Agent chỉ duy trì state trong phạm vi một `thread_id`. Khi người dùng mở thread mới để hỏi câu hỏi recall (`conv-01-recall-0`,...), lịch sử của thread mới hoàn toàn trống rỗng $\rightarrow$ Baseline phản hồi theo mẫu ngây thơ ("chưa có thông tin") và đạt recall gần như bằng 0.
- **Ưu thế của Persistent Memory**: Advanced Agent trích xuất các thuộc tính cốt lõi (Tên, Nơi ở, Nghề nghiệp, Đồ uống yêu thích, Style trả lời...) và ghi vào `User.md`. Bất kể ở thread mới nào, hồ sơ này luôn được nạp vào system prompt, giúp agent trả lời chính xác 100% các dữ kiện đã lưu.

### 3.2. Vì sao Advanced Agent tốn token hơn ở hội thoại ngắn?
- Nhìn vào Bảng 1 (Standard Benchmark), `Prompt tokens processed` của Advanced (33,602) cao hơn Baseline (14,641).
- **Nguyên nhân**: Ở mỗi lượt hội thoại ngắn, Advanced Agent luôn phải kéo thêm overhead của file `User.md` và phần tóm tắt ngữ cảnh vào prompt context. Khi các lượt hội thoại chưa đủ dài để vượt ngưỡng compact, chi phí "cõng" hồ sơ người dùng liên tục qua từng lượt làm tổng prompt token tăng lên. Đây là chi phí đánh đổi (trade-off) cần thiết để đổi lấy năng lực nhớ lâu dài.

### 3.3. Vì sao Compact Memory giúp Advanced Agent thắng thế ở hội thoại dài?
- Nhìn vào Bảng 2 (Stress Benchmark), `Prompt tokens processed` của Advanced Agent chỉ là **11,148 tokens**, giảm gần **50%** so với Baseline (**22,144 tokens**), đồng thời thực hiện **14 lần compactions**.
- **Giải thích**:
  - Với Baseline Agent, chi phí prompt mỗi lượt $i$ là tổng độ dài của tất cả các lượt từ $1$ đến $i$ ($O(N^2)$ context expansion). Với các đoạn văn bản dài hàng trăm chữ, lượng token bị nhân lên theo cấp số nhân.
  - Với Advanced Agent, khi tổng token trong buffer vượt ngưỡng `compact_threshold_tokens` (600 tokens), toàn bộ các tin nhắn cũ bị nén thành một bản tóm tắt súc tích, chỉ giữ lại đúng $K$ tin nhắn gần nhất. Do đó, kích thước ngữ cảnh mỗi lượt được giới hạn trên một hằng số $O(K + \text{summary})$, giúp tiết kiệm triệt để tài nguyên prompt khi đàm thoại kéo dài.

### 3.4. Tốc độ tăng trưởng file memory (`User.md`) và các rủi ro đi kèm
- Trong quá trình chạy, `User.md` của `dungct` tăng từ 0 lên 343 bytes, của `dungct_stress` là 277 bytes.
- **Rủi ro kỹ thuật khi mở rộng production**:
  1. **Unbounded Memory Growth**: Nếu người dùng trò chuyện qua nhiều tháng, việc ghi mọi sở thích vụn vặt sẽ khiến `User.md` phình to, vượt quá context window hoặc làm loãng sự chú ý của model.
  2. **Context Poisoning / Stale Facts**: Lưu sai sự thật hoặc không dọn dẹp các sự thật cũ sẽ khiến Agent hành động sai lệch.
  3. **Privacy & Security Leak**: Ghi trực tiếp PII (Personally Identifiable Information) hoặc thông tin nhạy cảm vào file markdown không mã hóa tiềm ẩn nguy cơ rò rỉ dữ liệu.

---

## 4. Các Tính Năng Bonus Đã Triển Khai (Mục Tiêu Điểm 90 - 100)

Hệ thống đã tích hợp 5 kỹ thuật nâng cao vượt khung rubric cơ bản:

1. **Phân Biệt Câu Hỏi vs Cung Cấp Sự Thật (Question vs Fact Discrimination)**:
   - Cơ chế regex guardrail tự động phát hiện và bỏ qua các câu hỏi truy vấn thông tin (ví dụ: *"Bạn có biết DũngCT không?"*, *"Mình tên gì?"*), ngăn chặn việc agent nhầm lẫn câu hỏi thành thông tin người dùng mới.

2. **Loại Bỏ Nhiễu & Phát Hiện Nói Đùa (Noise & Joke Rejection)**:
   - Nhận diện các bối cảnh nói đùa (ví dụ: *"đùa với đồng nghiệp rằng hay chuyển sang product manager..."*) $\rightarrow$ kiên quyết giữ nghề nghiệp thực tế `MLOps engineer`.
   - Nhận diện các sự kiện tạm thời (ví dụ: *"Hà Nội chỉ là nơi mình vừa bay ra họp..."*) $\rightarrow$ không nhầm Hà Nội thành nơi thường trú.

3. **Xử Lý Xung Đột & Đính Chính Thực Thể (Conflict Handling & Corrections)**:
   - Khi người dùng đưa ra đính chính mới (ví dụ: chuyển nơi ở từ Đà Nẵng $\rightarrow$ Huế, hoặc từ Huế $\rightarrow$ Đà Nẵng trong stress test), hàm `upsert_fact` tự động ghi đè giá trị cũ trong `User.md` mà không để tồn tại 2 trạng thái mâu thuẫn song song.

4. **Ngưỡng Tin Cậy (Confidence Thresholding)**:
   - Hàm `upsert_fact(..., confidence=..., min_confidence=0.6)` đảm bảo chỉ những thông tin có độ chắc chắn $\ge 60\%$ mới được ghi vào bộ nhớ vĩnh viễn. Các phát biểu phỏng đoán hoặc không chắc chắn bị loại bỏ ngay từ đầu.

5. **Cơ Chế Suy Giảm Bộ Nhớ (Memory Decay & Pruning)**:
   - Cung cấp hàm `apply_decay` giúp giảm dần trọng số ưu tiên của các thông tin cũ không còn được nhắc lại, hỗ trợ tự động cắt tỉa các thuộc tính lỗi thời nhằm kiểm soát kích thước file memory bền vững.

---

## 5. Hướng Dẫn Kiểm Thử & Chạy Nghiệm Thu

### Cài đặt môi trường ảo
```bash
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama python-dotenv tabulate pytest chardet
```

### Chạy Unit Test (100% Pass)
```bash
.\.venv\Scripts\python.exe -m pytest src/test_agents.py -v
```
*Kết quả: 7/7 bài test PASSED (bao gồm test `User.md`, compact trigger, cross-session recall, prompt token reduction, confidence threshold, noise filtering, và memory decay).*

### Chạy Benchmark So Sánh
```bash
.\.venv\Scripts\python.exe src/benchmark.py
```
*Kết quả in 2 bảng so sánh Standard Benchmark và Long-Context Stress Benchmark theo đúng chuẩn định dạng.*
