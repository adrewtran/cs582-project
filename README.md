# CS 582 CRM Sales Opportunities

Group 2: Hong Thai Phan, Nguyen Khanh An Tran, Hoang Thien Bao Bui.

Project giữ đúng đề tài CRM đã báo cáo với giáo sư. Bản nâng cấp gồm **xác suất Won/Lost**, **lý do của model** và **Sales Assistant đề xuất review actions kèm bằng chứng**. Chạy CPU, không cần GPU hoặc API trả phí.

**Hướng dẫn demo từng bước: [docs/PROFESSOR_DEMO_GUIDE.md](docs/PROFESSOR_DEMO_GUIDE.md).** Giải thích code: [docs/PROJECT_WALKTHROUGH.md](docs/PROJECT_WALKTHROUGH.md).

## Chạy trên Google Colab

1. Tải **Code → Download ZIP** và `notebooks/CRM_Sales_Opportunities.ipynb` từ cùng branch. Trước PR merge, dùng **adrewtran/cs582-project → codex/crm-history-agent**; sau merge dùng **thai-phan/cs582-project → main**.
2. Mở [Google Colab](https://colab.research.google.com/), **File → Upload notebook**, chọn notebook vừa tải.
3. Runtime Python **3.12**, accelerator **None/CPU**. Setup dừng nếu phiên bản chưa được kiểm thử.
4. **Runtime → Run all**. Upload đúng một ZIP source khi được hỏi. Không upload `.patch` hoặc ZIP chỉ có kết quả.
5. Notebook cài môi trường riêng, chạy tests, 6 model × 2 feature sets, xuất tài liệu, rồi demo bằng model thật.
6. Kiểm tra `PASS: full`, bảng A/B và `CRM SALES ASSISTANT`. Cell cuối tải ZIP kết quả; lưu trước khi runtime ngắt.

Không cần đổi tài khoản GitHub kết nối ChatGPT. Colab miễn phí có quota; xem [FAQ chính thức](https://research.google.com/colaboratory/faq.html). Nhóm vẫn cần xác nhận upload/download trong Colab và import Google Slides; thực thi code cell trong môi trường phát triển không xác nhận các UI này.

## Một lần setup, một lệnh full run

Từ thư mục chứa README, Python 3.12:

```bash
python scripts/setup_cpu.py
.venv-crm/bin/python -m src.run_project
```

Runner thực hiện validate/join, split, history, 12 lượt fit CPU, selection trên validation, evaluation, calibration, explanations/SHAP, leakage audit, assistant, report, PPTX và script. Bất kỳ model bắt buộc nào lỗi đều dừng pipeline.

```bash
# Verification
.venv-crm/bin/python -m pytest -q
.venv-crm/bin/python -m pip check

# Demo từ saved bundle, không train lại
.venv-crm/bin/python -m src.demo
.venv-crm/bin/python -m src.demo --opportunity-id 01XZ9CRY --json

# Smoke test, không dùng số liệu báo cáo
.venv-crm/bin/python -m src.run_project --quick

# Giữ output riêng
.venv-crm/bin/python -m src.run_project --output reports/crm/my_run
.venv-crm/bin/python -m src.demo --bundle reports/crm/my_run/model_bundle.joblib

# Regenerate tài liệu từ A/B outputs
.venv-crm/bin/python -m src.upgrade_deliverables reports/crm/final

# Chạy experiment không sinh tài liệu; hoặc raw-only để kiểm tra tương thích
.venv-crm/bin/python -m src.run_project --no-documents
.venv-crm/bin/python -m src.run_project --raw-only --output reports/crm/raw_only
```

`--quick` ghi `reports/crm/smoke`, nhãn `SMOKE_TEST_NOT_FINAL`. Full mặc định ghi `reports/crm/final`. Rerun cùng folder ghi đè artifact cùng tên; dùng output khác để giữ kết quả. Linux CPU đã kiểm thử; không tuyên bố đã kiểm thử Windows/macOS. Joblib chỉ được load từ nguồn tin cậy.

## Kết quả và novelty

Nguồn số liệu hiện tại: [ablation_comparison.csv](reports/crm/final/ablation_comparison.csv). A là 18 features gốc; B thêm 19 features lịch sử, cùng split. Bảng chứa cả validation/test metrics riêng cho 12 lượt fit.

Validation chọn **history / Logistic Regression**: validation AUC khoảng 0.5692, test AUC **0.5168** so với raw LR **0.5238** (−0.0070). History chưa cải thiện kết quả này. Raw CatBoost có test AUC khoảng 0.5424 nhưng không được chọn sau khi xem test. Calibrated Brier của selected model khoảng 0.2489, vẫn kém Dummy 0.2421. Đây chưa phải model đủ tốt cho sales prioritization.

| Đóng góp | Implementation | Bằng chứng |
|---|---|---|
| Leakage audit | `analysis.leakage_control` | `leakage_audit.csv`, diagnostic C |
| Lịch sử đúng thời điểm | `history.build_history` | Row audit, coverage, invariance tests |
| Sales Assistant | `sales_agent.recommend` | Rule IDs, evidence, rationale, batch JSONL và real-model CLI |

Đây là đóng góp ứng dụng có kiểm chứng, không phải thuật toán mới. CatBoost là comparator, SHAP hỗ trợ giải thích. Không có bằng chứng tăng doanh thu hoặc causal uplift. [NOVELTY_AND_RESULTS.md](docs/NOVELTY_AND_RESULTS.md) có đối chiếu rubric và trả lời ESL.

## Data và protocol

[Maven CRM Sales Opportunities](https://mavenanalytics.io/data-playground/crm-sales-opportunities) mô tả công ty phần cứng B2B giả lập. Raw CSV giữ nguyên trong `data/crm`; SHA-256 ghi trong manifest. Các synthetic experiments nhóm thêm trong `data/crm_synthetic` và scripts riêng được giữ nguyên; runner chính không trộn chúng vào kết quả CRM gốc.

- 8.800 opportunities, 85 accounts, 7 products, 35 sales-team rows và một data dictionary.
- 6.711 closed: 4.238 Won / 2.473 Lost. Score 1.589 dated Engaging, trong đó 1.088 thiếu account. Không gán open thành Lost; không score 500 Prospecting thiếu ngày.
- Train 2.975 / validation 583 / test 1.361; purge 1.792 outcomes chưa biết ở cutoff. Ngày cắt 2017-07-15 và 2017-09-18; cùng ngày cùng tập.
- History archive chỉ chứa **train**. Query tại T chỉ dùng deal khác có **close_date < T**, loại closure cùng ngày và chính ID. Outcome validation/test không cập nhật archive.
- Agent/account/product: prior count, smoothed win rate, mean value/cycle, cold-start. Thêm 4 global statistics, tổng 19 history features.
- Smoothing cố định 5. Khi history thiếu dùng eligible global prior; global rỗng: p=.5, count/value/cycle=0. Missing account không gộp thành một khách hàng.
- Current `close_value`, `close_date`, `deal_stage`, `opportunity_id` không là predictor. Past mean value chỉ từ deal khác đã đóng trước T.
- Preprocessing fit train; model/epoch/threshold/calibration chọn bằng validation. `selection_lock.json` tạo trước test. Test đã từng được xem nên là exploratory holdout, không phải independent external test.

TabNet có thể khác giữa CPU/runtime dù cùng seed và pins; không chọn lần có test đẹp hơn. Frozen histories tạo khác biệt phân phối giữa train và các period sau. Account/team/product là snapshot tĩnh. Closed-only sampling và missing open accounts hạn chế khả năng áp dụng thực tế.

## Outputs và diễn giải

| File/field | Ý nghĩa |
|---|---|
| `experiments/raw`, `experiments/history` | Chi tiết A/B: metrics, figures, configs, model và split |
| `history_audit.csv`, `history_coverage.csv` | Đối chiếu eligibility theo thời gian và mức đủ history |
| `open_deal_predictions.csv` | Probabilities, legacy win-priority và reference sensitivities |
| `sales_assistant_outputs.jsonl` | Tách `model_prediction` và `agent_recommendation`, một dòng mỗi deal |
| `loss_risk` | HIGH nếu P(win)<.40; MEDIUM nếu <.70; LOW nếu ≥.70; khác High win-priority |
| `actions` | 2–4 review requests có rule ID, evidence, rationale; không tự gửi email hoặc giảm giá |
| `timing` | Retrospective snapshot nếu engagement trước lúc model sẵn sàng |
| `rf_shap_*` | Diagnostic riêng cho RF chưa calibration; không gán thành explanation của selected model khác |
| `model_bundle.joblib` | Saved model, preprocessing, reference, threshold, frozen history archive |

Reference sensitivity thay một input bằng train reference; không additive và không causal. Correlated features có thể khiến perturbation không thực tế. Rules kiểm tra missing/thin history hoặc supported weak patterns, rồi yêu cầu verify status và human review. Chưa có user study hoặc intervention study.

## Tài liệu cho professor

- [Hướng dẫn demo](docs/PROFESSOR_DEMO_GUIDE.md)
- [Báo cáo](reports/crm/final/deliverables/CRM_Final_Report.md), cùng folder có DOCX
- [PowerPoint 12 slides](reports/crm/final/deliverables/CRM_Final_Presentation.pptx)
- [ESL script 10 phút, 3 người](reports/crm/final/deliverables/SPEAKER_SCRIPT_ESL.md), cùng folder có DOCX
- [Tóm tắt kết quả](reports/crm/final/deliverables/RESULTS_SUMMARY.md)
- [Verification record](docs/VERIFICATION.md)

PPTX có native text/table và speaker notes. Template trong `assets`, normal pipeline chỉ cần Python để điền nội dung, không cần design API/Node. Nhóm inspect sau Google Slides import. Giới hạn 1–8 slides trước đây dành cho progress forum; xác nhận giới hạn final. Chưa tự ghi đóng góp thực tế, merge PR hoặc nộp bài.

## Khi nào coi là chạy xong?

Manifest `status=complete`, `mode=full`; comparison 12 dòng; selected table 6 model; confusion counts cộng 1.361. Open scores 1.589 và win+loss=1. History không có closure cùng/sau query date. Live demo khớp exported score. Tests kiểm tra invariance, model fits, rules, empty inputs và artifact consistency.

`scripts/verify_notebook.py` chạy nguyên 7 code cells tuần tự trong một Python process; không kiểm chứng Jupyter/Colab UI. Tests pass chứng minh phần mềm hoạt động, không chứng minh predictive utility.
