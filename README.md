# CS 582 — Predicting CRM Sales Opportunities

Group 2: **Hong Thai Phan · Nguyen Khanh An Tran · Hoang Thien Bao Bui**.

Project giữ đúng đề tài CRM đã báo cáo với giáo sư. Input là bốn bảng CRM; hai output chính là **xác suất Won/Lost** và **các yếu tố ảnh hưởng tới điểm dự đoán**. Priority là nhãn minh họa, chưa phải chính sách bán hàng đã được kiểm chứng.

**Bắt đầu:** dùng `notebooks/CRM_Sales_Opportunities.ipynb` và ZIP chứa source mới. Không cần quyền push GitHub, GPU hoặc máy local mạnh để chạy.

## 1. Kết quả thực tế

Bản full đã chạy trong môi trường phát triển Python 3.12 / CPU. Dummy, Logistic Regression (LR), Random Forest (RF), MLP và TabNet đều được train thật. Kết quả chính nằm trong **`reports/crm/final/`**.

| Model | Test ROC-AUC | Accuracy | F1 (Won) |
|---|---:|---:|---:|
| Dummy prior | 0.5000 | 0.6003 | 0.7502 |
| Logistic Regression | 0.5238 | 0.4372 | 0.2620 |
| Random Forest | 0.5194 | 0.5900 | 0.7388 |
| MLP | 0.4954 | 0.4849 | 0.5135 |
| TabNet | 0.5173 | 0.5628 | 0.6685 |

LR được chọn bằng **validation ROC-AUC**, trước khi đọc test. Threshold tối ưu macro-F1 trên validation, không mặc định 0.5 và không tối ưu F1(Won) trên test. Accuracy/F1 cao của Dummy đến từ luôn đoán Won. Kết quả **chưa chứng minh khả năng ưu tiên sales hữu ích**; không đưa `close_value` vào input để cải thiện số liệu giả tạo.

Calibration LR làm Brier test đổi 0.2794 → 0.2443, vẫn kém Dummy 0.2421 (thấp hơn tốt hơn). Calibration không tăng ROC-AUC. Đây là nghiên cứu có hạn chế, chưa phải mô hình production.

## 2. Chạy trên Google Colab — hướng dẫn từng bước

Colab có gói miễn phí nhưng quota/tài nguyên không được đảm bảo: [Google FAQ](https://research.google.com/colaboratory/faq.html). Project chỉ cần CPU; không cần mua Pro.

1. Tải ZIP source bàn giao và `CRM_Sales_Opportunities.ipynb`. Giữ ZIP để upload; không cần cài Python trên laptop.
2. Mở [Google Colab](https://colab.research.google.com/) và đăng nhập Google của bạn.
3. Chọn **File → Upload notebook**, chọn `.ipynb` vừa tải.
4. **Runtime → Change runtime type**: hardware accelerator **None/CPU**. Môi trường đã kiểm thử là **Python 3.12**. Nếu Google đổi phiên bản, chọn runtime 3.12 nếu có; setup dừng rõ ràng với phiên bản chưa kiểm thử.
5. Chọn **Runtime → Run all**.
6. Cell đầu yêu cầu upload: chọn **đúng một ZIP source code mới**, không chọn `.patch` hoặc ZIP chỉ chứa kết quả.
7. Chờ setup và pipeline xong. Lần đầu tải dependency có thể mất vài phút. Notebook dùng môi trường Python riêng, không cần restart kernel giữa các cell.
8. Cell cuối tải ZIP kết quả. Lưu trước khi Colab ngắt session; storage của runtime là tạm thời.

Notebook gọi cùng command với CLI. Các code cell được thực thi tuần tự bằng Python trong môi trường phát triển. Môi trường này chặn socket của Jupyter kernel, nên **chưa xác nhận kernel Jupyter hoặc session Colab đăng nhập của nhóm**. Nhóm vẫn cần tự đăng nhập, chọn file upload/download và xác nhận một lần.

### Khi source mới đã được push

Có thể chọn tab **GitHub** trong Colab, tìm `thai-phan/cs582-project` và **đúng feature branch hoặc commit mới**. Trước PR merge, `main` có thể vẫn là code cũ. Mở notebook từ GitHub không tự tải cả repo vào runtime; notebook này vẫn hỗ trợ upload ZIP.

Muốn clone thay upload: thêm cell trước cell đầu và thay placeholder:

```python
import os, subprocess
from pathlib import Path
PROJECT_DIR = Path('/content/cs582-project')
if PROJECT_DIR.exists():
    raise RuntimeError('Dùng runtime mới để tránh trộn hai bản code.')
subprocess.run(['git', 'clone', 'https://github.com/thai-phan/cs582-project.git', str(PROJECT_DIR)], check=True)
subprocess.run(['git', '-C', str(PROJECT_DIR), 'checkout', 'TEN_BRANCH_HOAC_COMMIT_DA_PUSH'], check=True)
os.environ['CRM_PROJECT_ROOT'] = str(PROJECT_DIR)
```

Không cần đổi kết nối GitHub `adrewtran` của ChatGPT. Repo private có thể cần xác thực riêng khi clone; upload ZIP tránh bước đó.

## 3. Chạy terminal (tùy chọn)

Trong thư mục chứa README, dùng Python 3.12:

```bash
python scripts/setup_cpu.py
.venv-crm/bin/python -m src.run_project
.venv-crm/bin/python -m pytest -q
```

Sau setup, **một command full run** là `.venv-crm/bin/python -m src.run_project`.

Pipeline: validate/join → split → EDA → 5 models → validation selection → calibration → test → explanations/SHAP → open scores → paper/slides/script. Một bước lỗi sẽ làm command trả lỗi và manifest ghi `failed`; không âm thầm bỏ model.

Trên Windows, đường dẫn interpreter là `.venv-crm\Scripts\python.exe`. Môi trường đã xác nhận là Linux CPU; không tuyên bố đã kiểm thử Windows.

```bash
# Smoke test: tiny budgets, KHÔNG dùng kết quả này báo cáo
.venv-crm/bin/python -m src.run_project --quick
# Giữ lần chạy riêng
.venv-crm/bin/python -m src.run_project --output reports/crm/my_run
# Experiment only
.venv-crm/bin/python -m src.run_project --no-documents
# Tạo lại paper/slides từ kết quả có sẵn
.venv-crm/bin/python -m src.deliverables reports/crm/final
```

`--quick` ghi `reports/crm/smoke/` với nhãn **SMOKE_TEST_NOT_FINAL**. Rerun cùng output folder ghi đè artifact cùng tên; dùng `--output` khác để giữ lần trước. File bạn tự thêm không bị xóa.

## 4. Verify — chạy đúng là thế nào?

| Kiểm tra | Kết quả với data đi kèm |
|---|---|
| `run_manifest.json` | `status: complete`, `mode: full` |
| `test_metrics.csv` | 5 model, đủ metric, không NaN |
| Raw data | 8,800 pipeline; 85 accounts; 7 products; 35 teams |
| Labels | 6,711 closed: 4,238 Won / 2,473 Lost |
| Split | train 2,975; validation 583; test 1,361; purged 1,792 |
| `open_deal_predictions.csv` | 1,589 rows; win_probability + loss_probability = 1 |
| Missing account | 1,088 rows được flag |
| `shap_audit.json` | RF additivity error < 1e-5 |
| `deliverables/` | PPTX, DOCX, Markdown và ESL script |

Notebook tự kiểm tra điều kiện chính. Tests còn kiểm tra date availability, tied dates, invalid IDs/stages/dates/joins, metric formulas, cả năm model thật, calibration và reload bundle cho cùng prediction. **Tests pass không có nghĩa model đủ tốt về kinh doanh.** Hai lần fit TabNet trên cùng môi trường cuối cho cùng prediction; môi trường CPU trước cho AUC 0.4924 thay vì 0.5173. Seed không bảo đảm bitwise giống nhau giữa mọi nền tảng; early stopping có thể chọn epoch khác. Không chọn kết quả tốt hơn để quyết định model: tất cả đều yếu, LR vẫn được chọn trên validation. Đối chiếu versions, hashes và platform khi reproduce.

## 5. Data và feature policy

Nguồn: [Maven CRM Sales Opportunities](https://mavenanalytics.io/data-playground/crm-sales-opportunities), công ty B2B bán phần cứng máy tính giả lập. Bốn CSV và dictionary ở `data/crm/`; raw files giữ nguyên. Số dòng được tính từ file, không lấy số đếm website. Manifest lưu SHA-256.

- Join theo product, account và sales_agent, kiểm tra many-to-one và không làm tăng row count.
- Loader chuẩn hóa `GTXPro → GTX Pro` và `technolgy → technology`.
- Categorical: product, series, sector, office_location, regional_office, manager, sales_agent.
- Numeric: sales_price, revenue, employees, revenue_per_employee, year_established, account_age_at_engage, engage_year/month/quarter/dayofweek, is_subsidiary.
- **Loại khỏi predictor:** close_value, close_date, deal_stage, opportunity_id. close_date chỉ kiểm tra label có sẵn; stage tạo target; ID truy vết.
- Chỉ train Won/Lost; không gán open thành Lost. Score Engaging, loại 500 Prospecting thiếu engage_date.
- Imputation, encoding, scaling chỉ fit training; các tập còn lại chỉ transform.

Split mới giữ deals cùng ngày ở cùng period và purge outcome chưa biết tại cutoff. Dataset từng được xem trong các lần thử trước, nên gọi là **exploratory holdout**, không phải test hoàn toàn mới. Các bảng account/team/product là snapshot, không đảm bảo từng giá trị đúng ở thời điểm lịch sử. Accounts/agents có thể lặp lại giữa period; chưa đánh giá riêng khách hàng hoàn toàn mới.

## 6. Output và novelty

| Cột/file | Ý nghĩa |
|---|---|
| win_probability / loss_probability | P(Won) calibrated / 1 − P(Won) |
| predicted_outcome | Threshold từ validation, map qua calibration |
| priority | High ≥0.70; Medium ≥0.40; Low <0.40; heuristic chưa validate |
| positive_factors / negative_factors | Các input làm probability cao/thấp hơn khi so với train-reference |
| reference_deltas_json | Tất cả chênh lệch; **không phải SHAP, không cộng thành prediction** |
| input_warning / account_missing | Cảnh báo missing account, không giấu việc imputation/unknown category |
| scoring_context | `frozen_model_snapshot_demo`, không phải historical replay |
| rf_shap_* | SHAP riêng cho **raw RF**, không gán nhầm thành explanation của LR calibrated |
| model_bundle.joblib | Frozen model + calibration + features/reference/threshold |

Reference sensitivity thay từng input bằng median/mode của train rồi đo chênh lệch P(Won). Đây là hành vi của model, không chứng minh đổi input sẽ thay đổi kết quả thực tế. Thay riêng một feature tương quan có thể tạo tổ hợp không thực tế. Chỉ load joblib từ project đáng tin cậy do định dạng này deserialize Python objects.

Đối chiếu novelty đã hứa và lý do kết quả còn yếu: [`docs/NOVELTY_AND_RESULTS.md`](docs/NOVELTY_AND_RESULTS.md).

**Novelty cho project môn học:** phối hợp prediction, explanation, leakage audit, label availability và input-quality warning thành workflow tái tạo được. Không tuyên bố thuật toán mới, causal effect, expected revenue hoặc tăng doanh thu đã được chứng minh.

## 7. File để báo cáo

Tất cả trong `reports/crm/final/`:

- `deliverables/CRM_Final_Report.docx` và `.md`: phương pháp, kết quả, hạn chế, related work và references.
- `deliverables/CRM_Final_Presentation.pptx`: 12 slides editable, Arial, speaker notes. Upload Drive → Open with Google Slides; kiểm tra layout sau import. [Hướng dẫn Google](https://support.google.com/docs/answer/9310378?hl=en).
- `deliverables/SPEAKER_SCRIPT_ESL.docx` và `.md`: 3 người × 4 slides, khoảng 8–10 phút; kèm Q&A.
- `deliverables/TEAM_REVIEW.md`: các bước nhóm cần xác nhận trước nộp.
- `test_metrics.csv`, `calibration_test.csv`, `leakage_audit.csv`, `priority_test.csv`: bằng chứng số liệu.
- `figures/`: ROC, confusion matrices, EDA, calibration, importance, SHAP.
- `split_manifest.csv`, `model_training.json`, `run_manifest.json`: row assignments, configurations, versions và hashes.

Giới hạn 1–8 slides trong thông báo trước dành cho progress forum. Deck 12 slides này cho final; kiểm tra giới hạn final nếu thầy có thông báo riêng. Chưa ghi video, chưa nộp bài, chưa tự ghi đóng góp thực tế của thành viên.

## 8. Cam kết → bằng chứng

| Cam kết | Đã có |
|---|---|
| CRM Won/Lost + join/clean/EDA | `src/datasets/crm.py`, `src/analysis.py`, data-quality/EDA outputs |
| LR, RF, MLP, TabNet | `src/models.py`, 5 rows metrics gồm control |
| Accuracy/Precision/Recall/F1/AUC/confusion | `src/evaluation.py`, test metrics và figures |
| Feature importance + optional SHAP | native/permutation CSV, RF SHAP và additivity audit |
| Giải thích input/decision | per-deal explanation, missing-input warning |
| Reproducible/free CPU | setup, runner, notebook, tests |
| Paper + slides | editable team-review drafts |

`docs/COMMITMENT_AUDIT.md` lưu audit ban đầu và trạng thái khắc phục. Proposal CRM gốc: `docs/CS582_Group2_Original_CRM_Proposal.docx`; Markdown ở root là transcription. Hướng Leads/Bank/Telco cũ đã bị xóa khỏi repo (lấy lại từ commit `c5bcfa6` nếu cần).

## 9. Push branch / mở PR vào repo của nhóm

Repo đích là `thai-phan/cs582-project`, base `main`. Luồng PR đã dùng cho project là từ fork có sẵn `adrewtran/cs582-project`; fork này có quyền push, dù kết nối hiện chưa có quyền push trực tiếp vào repo đích. Branch bàn giao: `codex/crm-completion`, dựa trên `main` tại `f37826f`. Tên trong danh sách Contributors không tự cấp write permission.

Trong lần thử terminal/API ngày 2026-10-06, terminal thiếu xác thực và kết nối GitHub trả 403 khi tạo blob trên fork. Người dùng sau đó cho phép dùng trình duyệt đã đăng nhập để xuất bản branch/PR. Quyền của tài khoản trên repo không đồng nghĩa app đang kết nối có quyền ghi. Trạng thái PR hiện tại nằm trên GitHub; chi tiết kiểm chứng ở `docs/VERIFICATION.md`.

Không xóa repo, không upload ZIP thành một file source, không force-push. Dùng checkout hiện tại hoặc giải nén ZIP bàn giao rồi copy nội dung vào clone sạch. ZIP không chứa `.git` hay môi trường Python.

```bash
git remote -v
git status --short
git fetch origin
git switch -c codex/crm-completion
.venv-crm/bin/python -m pytest -q
```

Nếu branch đã tồn tại, kiểm tra rồi dùng `git switch codex/crm-completion`. Không reset/xóa các thay đổi CRM đang chưa commit. Nếu main đã tiến thêm, đối chiếu và xử lý conflict, không ghi đè mù.

Stage theo path, **không dùng `git add .`** để tránh stage nhầm file ngoài scope:

```bash
git add .gitignore AGENTS.md README.md PLAN.md Group2_Project_Proposal.md requirements.txt
git add src tests scripts notebooks/CRM_Sales_Opportunities.ipynb docs
git add reports/crm/final
git diff --cached --stat
git diff --cached --name-only
```

Đọc diff; đảm bảo không chứa dữ liệu ngoài scope. Sau đó:

```bash
git commit -m "Complete CRM model comparison and reproducible course deliverables"
# Thêm remote một lần; nếu đã có, kiểm tra bằng git remote -v.
git remote add fork https://github.com/adrewtran/cs582-project.git
git push -u fork codex/crm-completion
gh pr create --repo thai-phan/cs582-project --base main --head adrewtran:codex/crm-completion --title "Complete CRM ML project and reproducible deliverables" --body-file docs/PR_BODY.md
```

Không có `gh`: dùng GitHub **Compare across forks**, chọn base `thai-phan:main`, head `adrewtran:codex/crm-completion`, copy `docs/PR_BODY.md`. Để nhóm review; không tự merge. Nếu có quyền push trực tiếp, có thể dùng branch trên repo đích. Nếu gặp 403, kiểm tra account/app permission; không thay đổi kết nối của project khác.

Nếu dùng patch bàn giao: áp vào clone sạch đúng base commit ghi trong handoff. `git apply --check crm-completion.patch` rồi `git apply --index crm-completion.patch`. **Không viết `git apply -- --index ...`**: dấu `--` khiến Git hiểu `--index` là tên file. Không apply lại patch vào workspace đã có những changes này.

## 10. Troubleshooting

| Vấn đề | Xử lý |
|---|---|
| ModuleNotFoundError | Chạy setup; dùng đúng `.venv-crm/bin/python`, không nhầm kernel |
| Python khác 3.12 | Chọn runtime 3.12; không sửa pin đại rồi gọi là verified |
| Thiếu CSV | Upload ZIP source có `data/crm/`, không chỉ notebook/ZIP kết quả |
| Không có GPU | Bình thường: tất cả model dùng CPU |
| Colab quota/ngắt | Lưu output; chạy lại khi có CPU quota |
| TabNet best weights warning | Early stopping dùng best checkpoint, không phải skip model |
| SHAP dependency lỗi | Dùng đúng env/pins; giữ traceback và manifest để chẩn đoán |
| AUC quanh 0.5 | Kết quả thật; audit leakage nếu một score cao bất thường |
| High priority rất ít | Không ép threshold cho đẹp; band chưa validate nghiệp vụ |
| Numbers khác slide cũ | Chỉ dùng `reports/crm/final/`, không trộn kết quả cũ |
| Sửa slide rồi rerun bị mất | Sửa generator `src/deliverables.py` hoặc giữ bản PowerPoint sửa tay tên khác |

Cấu trúc: `data/crm/` inputs → `src/` pipeline → `reports/crm/final/` outputs; `tests/` kiểm thử; `scripts/` setup/notebook; `docs/` contract/audit/PR.
