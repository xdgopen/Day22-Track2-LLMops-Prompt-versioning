# Evidence Summary

Sinh viên: Nguyễn Danh Thành - 2A202600581

Provider hiện tại trong `.env`: OpenRouter

- `PROVIDER=openrouter`
- `OPENROUTER_MODEL=openai/gpt-4o-mini`

Ghi chú: OpenRouter không có embeddings endpoint chuẩn, project dùng `LocalHashEmbeddings` fallback khi không có OpenAI embedding key thật.

Các file evidence cần có theo rubric:

- `01_langsmith_traces.png`: chụp LangSmith dashboard sau khi chạy bước 1.
- `02_prompt_hub.png`: chụp Prompt Hub có 2 prompt `nguyen-danh-thanh-rag-prompt-v1` và `nguyen-danh-thanh-rag-prompt-v2`.
- `02_ab_routing_log.txt`: đã tạo bằng `python src/02_prompt_hub_ab_routing.py`.
- `03_ragas_scores.png`: chụp terminal bảng điểm RAGAS.
- `03_ragas_report.json`: đã copy từ `data/ragas_report.json`.
- `04_pii_demo_log.txt`: đã tạo từ demo Guardrails.
- `04_json_demo_log.txt`: đã tạo từ demo Guardrails.

Phân tích ngắn V1/V2:

- V1 ưu tiên câu trả lời ngắn, có `context_recall` cao hơn V2: 0.7400 so với 0.7083.
- V2 có cấu trúc tốt hơn và thắng 3/4 chỉ số: `faithfulness` 0.9444, `answer_relevancy` 0.7653, `context_precision` 0.6429.
- Kết luận: V2 là prompt tốt hơn cho submission vì đạt faithfulness cao nhất và vượt ngưỡng 0.8 rõ ràng.
