# Báo cáo LAB Day 22: LangSmith + Prompt Versioning

## Thông tin sinh viên

- Họ và tên: Nguyễn Danh Thành
- Mã số sinh viên: 2A202600581
- Chủ đề: RAG Pipeline, LangSmith tracing, Prompt Hub A/B routing, RAGAS evaluation, Guardrails validators
- Provider hiện tại: OpenRouter
- Cấu hình `.env`: `PROVIDER=openrouter`, `OPENROUTER_MODEL=openai/gpt-4o-mini`
- Embeddings: OpenRouter không có embeddings endpoint chuẩn, nên project dùng `LocalHashEmbeddings` fallback khi `OPENAI_API_KEY` chưa được cấu hình thật.

## 1. RAG Pipeline với LangSmith

Đã hoàn thiện `src/01_langsmith_rag_pipeline.py`:

- Tải `data/knowledge_base.txt`.
- Chia văn bản bằng `RecursiveCharacterTextSplitter` với `chunk_size=500`, `chunk_overlap=50`.
- Tạo FAISS vector store từ embedding provider được chọn trong `.env`.
- Xây dựng LCEL chain theo luồng `retriever -> prompt -> LLM -> StrOutputParser`.
- Gắn `@traceable(name="rag-query", tags=["rag", "step1"])` để gửi trace lên LangSmith.
- Chạy toàn bộ 50 câu hỏi trong `SAMPLE_QUESTIONS`.

Lệnh chạy:

```bash
cd src
python 01_langsmith_rag_pipeline.py
```

Evidence cần nộp sau khi chạy có network/API key:

- `evidence/01_langsmith_traces.png`: ảnh LangSmith project hiển thị tối thiểu 50 traces.

Kết quả chạy thực tế với OpenRouter:

- Đã chạy đủ 50 câu hỏi.
- Script báo đã gửi 50 traces lên LangSmith project `day22-lab`.
- FAISS vectorstore được tạo từ 107 chunks.
- Evidence cần chụp thêm: `evidence/01_langsmith_traces.png`.

## 2. Prompt Hub và A/B Routing

Đã hoàn thiện `src/02_prompt_hub_ab_routing.py`:

- Tạo 2 prompt version:
  - `nguyen-danh-thanh-rag-prompt-v1`: phong cách ngắn gọn, 2-4 câu.
  - `nguyen-danh-thanh-rag-prompt-v2`: phong cách chuyên gia, có cấu trúc, 3-5 câu.
- Push 2 prompt lên LangSmith Prompt Hub bằng `client.push_prompt`.
- Pull prompt từ Hub bằng `client.pull_prompt`, có fallback local nếu Hub tạm lỗi.
- A/B routing tất định bằng `MD5(request_id) % 2`.
- Gắn `@traceable(name="ab-rag-query", tags=["ab-test", "step2"])`.

Lệnh chạy và lưu log:

```bash
cd src
python 02_prompt_hub_ab_routing.py | tee ../evidence/02_ab_routing_log.txt
```

Evidence cần nộp:

- `evidence/02_prompt_hub.png`: ảnh Prompt Hub có 2 prompt version.
- `evidence/02_ab_routing_log.txt`: log 50 câu hỏi với nhãn `prompt-v1` hoặc `prompt-v2`.

Kết quả chạy thực tế với OpenRouter:

- Prompt V1: `nguyen-danh-thanh-rag-prompt-v1`.
- Prompt V2: `nguyen-danh-thanh-rag-prompt-v2`.
- Khi chạy lại, LangSmith trả `409 Nothing to commit` vì prompt không đổi so với commit mới nhất; đây là trạng thái hợp lệ, sau đó script vẫn pull cả 2 prompt từ Hub.
- Routing: V1 = 19 câu, V2 = 31 câu, tổng = 50 câu.
- Log đã lưu tại `evidence/02_ab_routing_log.txt`.

## 3. RAGAS Evaluation

Đã hoàn thiện `src/03_ragas_evaluation.py`:

- Chạy cả 50 QA pairs qua cả 2 prompt version.
- Thu thập `question`, `reference`, `answer`, `contexts`.
- Tạo `EvaluationDataset` từ các `SingleTurnSample`.
- Đánh giá 4 metric: `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`.
- Tính trung bình từng metric bằng `numpy.mean`.
- Lưu báo cáo vào `data/ragas_report.json`.

Lệnh chạy:

```bash
cd src
python 03_ragas_evaluation.py
cp ../data/ragas_report.json ../evidence/03_ragas_report.json
```

Evidence cần nộp:

- `evidence/03_ragas_scores.png`: ảnh terminal hiển thị bảng điểm V1/V2.
- `evidence/03_ragas_report.json`: bản sao của report JSON.

Kết quả chạy thực tế với OpenRouter:

- Đã sinh outputs cho 50 QA pairs với V1 và 50 QA pairs với V2.
- RAGAS V1:
  - `faithfulness`: 0.8481
  - `answer_relevancy`: 0.6878
  - `context_recall`: 0.7400
  - `context_precision`: 0.5683
- RAGAS V2:
  - `faithfulness`: 0.9444
  - `answer_relevancy`: 0.7653
  - `context_recall`: 0.7083
  - `context_precision`: 0.6429
- So sánh:
  - V2 thắng `faithfulness`, `answer_relevancy`, `context_precision`.
  - V1 thắng `context_recall`.
- Mục tiêu rubric đạt: best faithfulness = 0.9444 >= 0.8.
- `data/ragas_report.json` và `evidence/03_ragas_report.json` đã được lưu dưới dạng JSON hợp lệ.
- Log terminal đã lưu tại `evidence/03_ragas_scores.txt`; có thể chụp ảnh từ đoạn cuối log để tạo `evidence/03_ragas_scores.png`.

## 4. Guardrails AI Validators

Đã hoàn thiện `src/04_guardrails_validator.py`:

- `PIIDetector` dùng regex để phát hiện và redact:
  - Email
  - Số điện thoại
  - SSN
  - Số thẻ tín dụng
- `JSONFormatter` tự sửa:
  - Markdown code fences
  - Single quotes
  - Trailing commas
  - JSON formatting chuẩn bằng `json.dumps`
- Dùng `Guard().use(Validator(on_fail=OnFailAction.FIX))`.

Đã chạy demo bằng `.venv` và lưu log:

```bash
./.venv/bin/python src/04_guardrails_validator.py | tee evidence/04_pii_demo_log.txt
cp evidence/04_pii_demo_log.txt evidence/04_json_demo_log.txt
```

Kết quả chính:

- Email được đổi thành `[EMAIL_REDACTED]`.
- Số điện thoại được đổi thành `[PHONE_REDACTED]`.
- SSN được đổi thành `[SSN_REDACTED]`.
- Credit card được đổi thành `[CREDIT_CARD_REDACTED]`.
- JSON có markdown fences, single quotes và trailing comma được sửa thành JSON hợp lệ.
- Trường hợp hoàn toàn không phải JSON trả về Fail đúng kỳ vọng.

## Ghi chú kiểm chứng

- Đã kiểm tra compile bằng Python trong `.venv`: các script chính không có lỗi cú pháp.
- `./.venv/bin/python src/config.py` trả về `Provider: OPENROUTER`.
- Bước 1 đã chạy đủ 50 câu hỏi và gửi traces lên LangSmith.
- Bước 2 đã pull prompt từ Hub và chạy đủ 50 câu hỏi A/B.
- Bước 3 đã chạy RAGAS cho cả V1 và V2; target faithfulness đạt.
- Bước 4 đã chạy thành công trong môi trường local và không phụ thuộc LLM provider.
- Không nên commit `.env`.
- Khi chạy Guardrails trong môi trường offline có thể xuất hiện warning OpenTelemetry do package cố export telemetry ra ngoài; warning này không làm fail validator.
