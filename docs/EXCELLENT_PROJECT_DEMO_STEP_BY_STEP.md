# Hướng dẫn demo từng bước cho giáo sư — CRM Agent (CS 582, Group 2)

> **Cách đọc tài liệu này.** Phần hướng dẫn chuẩn bị viết bằng **tiếng Việt**. Những câu bạn **nói với giáo sư** viết bằng **tiếng Anh đơn giản**, đặt trong khung `🗣️ Say:`. Mọi lệnh đều copy/dán được. Không có con số nào trong tài liệu này được bịa ra: các con số ghi "verified run" là từ lần chạy đầy đủ đã kiểm chứng, lưu trong `reports/crm/final/`. Nếu bạn chạy lại toàn bộ pipeline, hãy đọc số thật trên màn hình theo hướng dẫn "Cách đọc số" ở từng bước.

**Một người điều khiển toàn bộ demo** (người giữ máy tính). Ba thành viên vẫn nói phần slide riêng của mình theo `docs/ESL_PRESENTATION_TRANSCRIPT.md`.

**Hai chế độ:**
- **Chế độ đầy đủ** (≈ 20–25 phút): làm STEP 0 → STEP 11.
- **Chế độ nhanh 3–5 phút** (sau khi đã cài đặt): xem mục cuối "LIVE DEMO 3–5 PHÚT".

**Các con số đã kiểm chứng trong lần chạy đầy đủ (verified run)** — dùng để so sánh với màn hình. Bảng này được điền từ `reports/crm/final/` sau lần chạy đầy đủ; xem mục "Số liệu tham chiếu" ở cuối tài liệu.

---

## STEP 0 — Chuẩn bị (làm trước ngày thuyết trình, ≈ 20–40 phút, chỉ một lần)

### 0.1 Repository và nhánh (branch) cần tải

- Code nằm ở **PR #4**: https://github.com/thai-phan/cs582-project/pull/4
- Nhánh của PR: `codex/crm-history-agent` trên fork `adrewtran/cs582-project`.
- PR này **chưa merge** vào `main`. Vì vậy **không** dùng nhánh `main` cho demo agent.

### 0.2 Cài công cụ

| Hệ điều hành | Cài Git | Cài Python 3.12 |
|---|---|---|
| macOS | `git --version` (nếu chưa có, máy sẽ gợi ý cài Command Line Tools) | `brew install python@3.12` (cần Homebrew: https://brew.sh) |
| Windows | https://git-scm.com/download/win | https://www.python.org/downloads/release/python-3120/ → tick **"Add python.exe to PATH"** |
| Linux (Ubuntu) | `sudo apt install git` | `sudo apt install python3.12 python3.12-venv` |

Kiểm tra phiên bản (phải là **3.12.x**; 3.13/3.14 sẽ bị từ chối):

```bash
python3.12 --version          # macOS / Linux
py -3.12 --version            # Windows (PowerShell)
```

### 0.3 Tải code và chuyển sang nhánh PR #4

Cách A (trực tiếp từ fork, khuyên dùng):

```bash
git clone https://github.com/adrewtran/cs582-project.git
cd cs582-project
git checkout codex/crm-history-agent
git log --oneline -1          # ghi lại mã commit để biết bạn đang demo đúng phiên bản
```

Cách B (từ repo chính, lấy đúng đầu PR #4):

```bash
git clone https://github.com/thai-phan/cs582-project.git
cd cs582-project
git fetch origin pull/4/head:pr-4
git checkout pr-4
```

Nếu đã clone trước đó: `git fetch --all` rồi `git checkout codex/crm-history-agent` và `git pull`.

### 0.4 Cài thư viện (CPU, không cần GPU, không cần API key)

```bash
python3.12 scripts/tools/setup_cpu.py          # macOS / Linux
py -3.12 scripts\tools\setup_cpu.py            # Windows
```

Script này tạo thư mục `.venv-crm`, cài PyTorch bản CPU, rồi cài `requirements.txt` (scikit-learn, pytorch-tabnet, catboost, shap, …) và chạy `pip check`. Lần đầu mất khoảng 5–15 phút (tùy mạng).

**Trong toàn bộ tài liệu, `PY` nghĩa là:**
- macOS / Linux: `.venv-crm/bin/python`
- Windows: `.venv-crm\Scripts\python.exe`

Ví dụ trên Windows, thay `.venv-crm/bin/python -m src.agent_demo` bằng `.venv-crm\Scripts\python.exe -m src.agent_demo`.

### 0.5 Dữ liệu và model cần có

Không cần tải gì thêm. Repo đã chứa:
- Dữ liệu CRM thật: `data/crm/` (4 bảng: `sales_pipeline.csv`, `accounts.csv`, `products.csv`, `sales_teams.csv`). **Chỉ đọc; không sửa.**
- Model đã train và kết quả: `reports/crm/final/` (thư mục `models/`, `metrics/`, `predictions/`, `agent/`, `figures/`, …).

### 0.6 Kiểm tra cài đặt

```bash
.venv-crm/bin/python -c "import sklearn, torch, pytorch_tabnet, catboost; print('ok')"
.venv-crm/bin/python -m src.agent_demo --check
```

Kết quả mong đợi: dòng `ok`, sau đó một danh sách `[OK] ...` và cuối cùng là **`CHECK PASSED`**.

Kiểm tra sâu hơn (không bắt buộc, ≈ 15–25 phút, có train model nhỏ thật trên CPU):

```bash
.venv-crm/bin/python -m pytest -q
```

### 0.7 Xử lý sự cố khi chuẩn bị

| Triệu chứng | Cách xử lý |
|---|---|
| `Verified runtime is Python 3.12` | Bạn đang dùng Python khác 3.12. Dùng đúng `python3.12` (macOS/Linux) hoặc `py -3.12` (Windows). |
| `python3.12: command not found` (macOS) | Đóng và mở lại Terminal; hoặc dùng `/opt/homebrew/bin/python3.12` (Apple Silicon) hay `/usr/local/bin/python3.12` (Intel). |
| `No module named src` | Bạn không đứng ở thư mục gốc của repo. Chạy `cd cs582-project` trước. |
| `[MISSING] ...` khi `--check` | Chạy lại toàn bộ pipeline: `.venv-crm/bin/python -m src.run_project` (≈ 45–60 phút trên CPU laptop; làm **trước** buổi thuyết trình). |
| Lỗi khi cài `torch` trên Windows | Cài "Microsoft Visual C++ Redistributable", rồi chạy lại `setup_cpu.py`. |
| Muốn làm lại từ đầu | Xóa thư mục `.venv-crm` rồi chạy lại bước 0.4. |
| Chữ tiếng Việt/ký tự lạ trên Windows | Dùng Windows Terminal hoặc chạy `chcp 65001` trước. |

**Những gì đã được kiểm chứng:** toàn bộ cài đặt, test và demo đã chạy thành công trên **Linux (WSL2, Ubuntu, Python 3.12, CPU)**. **Chưa kiểm chứng** trên macOS, Windows gốc hoặc Google Colab trong lần nâng cấp này. Vì vậy hãy làm STEP 0 trên chính máy sẽ dùng để demo, ít nhất một ngày trước buổi thuyết trình.

---

## STEP 1 — Xác minh model đã train và kết quả đã lưu (≈ 1 phút)

**Lệnh:**
```bash
.venv-crm/bin/python -m src.agent_demo --check
```

**Kết quả mong đợi:** danh sách file `[OK]` (manifest, model bundle, model card, selection lock, metrics, predictions, kết quả agent), rồi 4 dòng tóm tắt và `CHECK PASSED`.

**Cách đọc số:** dòng `Selected on validation BEFORE test: history/logistic_regression (test_evaluated_at_selection=False)` cho biết model được chọn **chỉ dựa vào tập validation**, trước khi nhìn tập test. Dòng `Selected model test ROC-AUC 0.5168` là kết quả test thật (gần 0.5 = gần ngẫu nhiên).

**File có thể mở:** `reports/crm/final/models/selection_lock.json` (chỉ ra 12 ứng viên và quy tắc chọn), `reports/crm/final/run_manifest.json`.

**Chỉ vào màn hình:** dòng `test_evaluated_at_selection=False` và dòng `Agent experiment: complete ...`.

🗣️ **Say:** "First, I check that everything is already trained and saved. We do not train anything live. The model was chosen using only the validation data, before we looked at the test data. Here you can see it: test evaluated at selection is false."

**Bước này chứng minh:** model thật, đã đóng băng (frozen), chọn không gian lận trên test.

**Nếu lỗi:** xem bảng 0.7. Nếu thiếu file → chạy `.venv-crm/bin/python -m src.run_project` trước buổi demo.

---

## STEP 2 — Chạy dự đoán ML thông thường (Baseline A) (≈ 1 phút)

**Lệnh:**
```bash
.venv-crm/bin/python -m src.agent_demo --traditional
```

**Kết quả mong đợi:** một deal thật được chọn theo quy tắc cố định (deal Engaging đầu tiên có account đầy đủ, được tạo sau ngày model có mặt). Trong verified run đó là **01XZ9CRY** (sản phẩm GTX Plus Pro, account Initech). Màn hình in `P(Won)`, `P(Lost)`, ngưỡng (threshold), `Predicted outcome`, `priority band`, các yếu tố làm tăng/giảm điểm, và `Warning`.

**Cách đọc số:** `P(Won)` là xác suất thắng đã hiệu chỉnh (calibrated). Nếu `P(Won)` nhỏ hơn `threshold` thì dự đoán là `Lost`. Các số `delta P(Won)` là mức thay đổi điểm khi đặt một input về giá trị tham chiếu (trung vị/mode của tập train). Đây là liên hệ dự đoán, **không phải nguyên nhân**.

**Chỉ vào màn hình:** dòng `Warning: none` và câu cuối "This is where a traditional pipeline stops".

🗣️ **Say:** "This is a traditional machine learning pipeline. It gives one number, a label and some factors. That is all. A person still has to look up the account, check the history and the data quality, and decide what to do. Notice the warning line says none. Remember that, because the agent will find a problem here."

**Bước này chứng minh:** baseline truyền thống chạy thật trên model đã lưu.

**Nếu lỗi:** `No module named src` → `cd` về thư mục repo. `FileNotFoundError` → làm STEP 1.

---

## STEP 3 — Chạy trợ lý dựa trên luật hiện có (Baseline B) (≈ 1 phút)

**Lệnh:**
```bash
.venv-crm/bin/python -m src.assistant_demo
```

**Kết quả mong đợi:** cùng deal **01XZ9CRY**, cùng xác suất như STEP 2; một mức `Loss risk` (LOW/MEDIUM/HIGH theo ngưỡng cố định 0.40/0.70), ba lý do, và 2–4 hành động cố định (ví dụ `verify_status`, `human_review`) với phần `Evidence` và `Why`.

**Chỉ vào màn hình:** dòng `Fixed-rule review actions (printed only; nothing is executed)`.

🗣️ **Say:** "This is our earlier rule-based assistant. It uses the same model and the same deal. It adds fixed IF-THEN rules. It always runs the same rules in the same order. It only prints advice. It does not create a task, it does not keep a trace, and it cannot refuse a dangerous request."

**Bước này chứng minh:** baseline B có thật và dùng cùng đầu vào — so sánh công bằng.

**Nếu lỗi:** giống STEP 2.

---

## STEP 4 — Chạy agent tự động mới (≈ 2 phút)

**Lệnh (một lệnh duy nhất):**
```bash
.venv-crm/bin/python -m src.agent_demo
```

(Muốn dừng giữa các phần để giải thích, thêm `--pause` và nhấn Enter để đi tiếp.)

**Kết quả mong đợi:** 5 phần có tiêu đề:
1. `[1/5] LOADING THE FROZEN, ALREADY-TRAINED SYSTEM` — tên model, mã SHA-256 của model bundle, ROC-AUC validation và khoảng tin cậy 95%, ngưỡng, số deal.
2. `[2/5] THE AGENT STARTS` — mục tiêu (goal) và từng bước `Step n PLAN / ACT / EVALUATE / DECIDE`.
3. `[3/5] DECISION AND ACTIONS` — quyết định, khuyến nghị, chất lượng bằng chứng, task đã tạo.
4. `COMPARISON ON THE SAME DEAL` — so sánh A, B, C.
5. `[5/5] SAVED ARTIFACTS` — danh sách file đã ghi, kết thúc bằng `DEMO COMPLETE`.

**Chỉ vào màn hình:** dòng `Step  1 GOAL ...`.

🗣️ **Say:** "Now the new agent. I type one command. The agent loads the same frozen model. It sets its own goal: decide the safest supported review action for this deal. From now on, I do not touch anything. The agent decides each step by itself."

**Bước này chứng minh:** agent chạy hoàn toàn offline, tự động, trên model và dữ liệu thật.

**Nếu lỗi:** xem STEP 1. Nếu màn hình quá nhỏ, phóng to font Terminal (Ctrl/Cmd +) trước buổi demo.

---

## STEP 5 — Quan sát agent tự chọn và thực thi công cụ (≈ 2–3 phút)

Vẫn là output của STEP 4. Đọc từ trên xuống:

| Dòng trên màn hình | Ý nghĩa |
|---|---|
| `PLAN options (need - cost): get_opportunity 0.98` | Agent chấm điểm từng công cụ: **utility = need − cost**. Chỉ chạy công cụ nếu utility ≥ 0.10. |
| `why ...: ...` | Lý do bằng lời cho lựa chọn đó. |
| `ACT get_account_information({...}) -> OK` | Agent **thực sự gọi** hàm Python cục bộ, kèm tham số. |
| `source: data/crm/accounts.csv [Initech]` | Nguồn dữ liệu chính xác của bằng chứng. |
| `result: ...` | Agent đánh giá kết quả vừa nhận. |
| `check_data_quality ... warnings: ['outside_training_range']` | Agent phát hiện input nằm ngoài phạm vi dữ liệu train. |
| `decision-relevant: the result could change the action (...)` | **Value of information**: agent chỉ lấy thêm bằng chứng nếu nó có thể thay đổi quyết định. |
| `EVALUATE evaluate_evidence ... Evidence quality ...` | Tổng hợp chất lượng bằng chứng, mâu thuẫn, độ tin cậy của model. |
| `DECIDE candidates, most specific first` | 5 hành động ứng viên; agent chọn hành động **cụ thể nhất mà bằng chứng cho phép**. `(missing: ...)` cho biết điều kiện nào chưa đạt. |
| `ACT create_review_task(...) -> OK` | Hành động an toàn duy nhất được tự động thực thi: ghi một file task **cục bộ**. |

**Cách đọc số:** trong verified run, deal 01XZ9CRY có `P(Won)` thấp hơn ngưỡng hơn 0.05 (risk `at_risk`). Bằng chứng đủ (≥ 0.70), không có mâu thuẫn, nên quyết định là **`ESCALATE_AT_RISK_REVIEW`**: cần người xem trước. Agent cũng ghi cảnh báo `outside_training_range`, vì một số input lịch sử lớn hơn bất kỳ giá trị nào model đã thấy khi train. Baseline A ở STEP 2 thì ghi `Warning: none`.

🗣️ **Say:** "Look at each step. The agent plans: it gives every tool a score, need minus cost, and runs the best one. Then it acts: it really calls the tool, and it shows the data source. Then it evaluates the result. Here the data-quality check found that some inputs are outside the training range. The traditional pipeline did not show this. At the end, the agent lists five possible actions and picks the most specific one that the evidence supports. The only thing it does by itself is write a local review task file. It never emails a customer or changes the CRM."

**Bước này chứng minh:** agent có quan sát, lập kế hoạch, dùng công cụ thật, tự đánh giá, và chỉ thực thi hành động an toàn.

**Nếu lỗi:** nếu output khác số trong bảng tham chiếu, kiểm tra `git log --oneline -1` có đúng commit PR #4 không, và chạy `--check`.

---

## STEP 6 — Mở trace (vết quyết định) và giải thích từng trường quan trọng (≈ 2–3 phút)

**File cần mở** (đường dẫn được in ở cuối STEP 4):
- `reports/crm/agent_demo/traces/01XZ9CRY.full_agent.report.md` — bản dễ đọc (bảng từng bước).
- `reports/crm/agent_demo/traces/01XZ9CRY.full_agent.trace.json` — bản máy đọc được.
- `reports/crm/agent_demo/review_tasks/<task_id>.json` — task đã tạo.

Mở bằng VS Code, hoặc:
```bash
cat reports/crm/agent_demo/traces/01XZ9CRY.full_agent.report.md     # macOS / Linux
type reports\crm\agent_demo\traces\01XZ9CRY.full_agent.report.md    # Windows
```

**Các trường cần chỉ vào trong file JSON:**

| Trường | Giải thích |
|---|---|
| `policy_version` | Phiên bản chính sách quyết định (mọi hằng số nằm trong `src/agent/policy.py`). |
| `goal` | Mục tiêu agent tự đặt. |
| `model.name`, `model.feature_set`, `model.bundle_sha256` | Chính xác model nào đã được dùng (mã băm file). |
| `steps[].candidates` | Các công cụ được cân nhắc, kèm `need`, `cost`, `utility`, `reason`. |
| `steps[].chosen`, `steps[].tool_call.arguments` | Công cụ được chọn và tham số thật. |
| `steps[].tool_call.sources` | Bảng/khóa dữ liệu làm nguồn bằng chứng. |
| `steps[].tool_call.attempts` | Số lần thử (tự thử lại 1 lần nếu lỗi). |
| `steps[].evaluation` | Agent đánh giá kết quả. |
| `decision.action`, `decision.why_not_more_specific` | Quyết định cuối và lý do không chọn hành động cụ thể hơn. |
| `decision.assessment.evidence_checks` | 7 mục kiểm tra bằng chứng, mỗi mục có trọng số cố định. |
| `actions_executed`, `blocked_actions`, `approval_requests` | Hành động đã làm / bị chặn / chờ người duyệt. |
| `stopping_reason`, `step_budget` | Vì sao dừng; giới hạn tối đa 14 bước. |
| `external_side_effects` | Luôn bằng 0. |
| `content_sha256` | Mã băm nội dung (bỏ timestamp). Chạy lại sẽ ra **cùng mã** → tái lập được. |

🗣️ **Say:** "Every decision leaves a trace. This file records the goal, every tool the agent considered with its score, the exact arguments, the data source, the result, and why the agent stopped. It also records the model file hash. The content hash is the same every time we run it, so the run is reproducible and auditable."

**Bước này chứng minh:** quyết định có thể kiểm toán, tái lập, và có nguồn gốc rõ ràng.

**Nếu lỗi:** thư mục `reports/crm/agent_demo/` bị xóa mỗi lần chạy demo mới. Nếu cần file đã kiểm chứng sẵn, mở bản tham chiếu: `reports/crm/final/agent/scenario_runs/S1/traces/01XZ9CRY.full_agent.report.md` (đây là **output đã lưu của lần chạy đã kiểm chứng**, không phải chạy trực tiếp).

---

## STEP 7 — Tình huống thiếu thông tin (missing account) (≈ 1–2 phút)

**Lệnh:**
```bash
.venv-crm/bin/python -m src.agent_demo --scenario missing-account
```

**Kết quả mong đợi:** một deal Engaging thật **không có account** (verified run: **UU7N7LZK**). Agent:
- không gọi `get_account_information` (lý do: `not applicable: the opportunity has no account key`);
- `check_data_quality` báo `Blocking issues: ['account_missing']`;
- vẫn tính điểm, nhưng chỉ làm **ngữ cảnh** (need 0.30), không giải thích và không kiểm tra committee (need 0 / không liên quan);
- quyết định **`DATA_COMPLETION`**: tạo task "sửa dữ liệu CRM trước".

Phần `COMPARISON` cho thấy: A vẫn đưa ra nhãn và priority dựa trên điểm nằm ngoài vùng train; B gắn mức rủi ro và thêm hành động `complete_account`.

🗣️ **Say:** "Now a real deal with no account information. Most open deals in this data have this problem. The agent notices the missing account, so it does not trust the score. It does not waste time explaining a score that is out of support. It creates a data-completion task instead. The traditional pipeline still gives a label and a priority for this deal."

**Bước này chứng minh:** agent nhận diện dữ liệu thiếu và từ chối khuyến nghị không có cơ sở.

**Biến thể khác (nếu còn thời gian):** `--scenario borderline` (điểm sát ngưỡng → `REVIEW_UNCERTAIN`), `--scenario conflict` (điểm mâu thuẫn với lịch sử account), `--scenario invalid` (deal đã đóng → `ABSTAIN`), `--scenario tool-failure` (model bị lỗi giả lập → thử lại 1 lần rồi `ABSTAIN`, không bịa xác suất), `--scenario prohibited` (yêu cầu gửi email/sửa CRM → **bị chặn**, chờ người duyệt).

**Nếu lỗi:** chạy `--check`.

---

## STEP 8 — Agent tự động xét duyệt nhiều deal (≈ 2 phút)

**Lệnh:**
```bash
.venv-crm/bin/python -m src.agent_demo --batch 10 --budget 3
```

**Kết quả mong đợi:**
- `[2/5] BATCH GOAL`: chọn deal nào cần người xem trước, trong 10 deal Engaging gần nhất, tối đa 3 task.
- `[3/5]`: mỗi deal một dòng: quyết định, `P(Won)`, chất lượng bằng chứng, số công cụ đã dùng.
- `[4/5] REVIEW QUEUE`: hàng đợi xếp theo **khóa phân loại** (escalation trước, rồi bằng chứng tốt hơn, rồi P(Won) thấp hơn, rồi mở lâu hơn). Chỉ 3 dòng đầu ghi `TASK CREATED`; phần còn lại ghi `over budget`.
- `Data-completion queue`: các deal thiếu account được tách riêng, **không** vào hàng đợi bán hàng.

**File mở:** `reports/crm/agent_demo/batch_summary.json`.

🗣️ **Say:** "In real work, a sales manager has many deals and little time. Here the agent screens ten real deals, investigates the ones where more evidence matters, and builds a review queue with at most three tasks. Deals with missing data go to a separate data queue. The probability only breaks ties inside the same evidence level. We do not treat a weak probability as business value."

**Bước này chứng minh:** lập kế hoạch nhiều đối tượng với ngân sách (budget) và mục tiêu rõ ràng.

**Nếu lỗi:** giảm `--batch 5`.

---

## STEP 9 — Trình bày so sánh có kiểm soát và ablation (≈ 2–3 phút)

**File cần mở:**
- Hình: `reports/crm/final/figures/agent_comparison.png` và `reports/crm/final/figures/agent_ablation.png`.
- Bảng: `reports/crm/final/agent/system_comparison.csv` (A, B, C và 4 ablation trên **cùng 1,589 deal Engaging**).
- Bảng: `reports/crm/final/agent/scenario_results.csv` (10 kịch bản × 3 hệ thống).
- Bảng: `reports/crm/final/agent/ablation_summary.csv`.

**Cách đọc số** (cột trong `system_comparison.csv`):

| Cột | Ý nghĩa |
|---|---|
| `evidence_types_mean` | Số loại bằng chứng trung bình được trích dẫn (tối đa 7). |
| `claim_verification_rate` | Tỷ lệ khẳng định được **kiểm chứng lại độc lập** từ CSV gốc / model bundle. |
| `data_issue_recall` | Tỷ lệ vấn đề dữ liệu thật (thiếu account, chấm điểm hồi tố, bản ghi cũ) được phát hiện. |
| `unsupported_label_rate` | Tỷ lệ nhãn định hướng (Won/Lost, HIGH/LOW, escalate/monitor) gắn cho deal **ngoài vùng train** hoặc **sát ngưỡng**. |
| `manual_steps_mean` | Số bước con người còn phải tự làm (trên checklist 9 bước). |
| `external_actions_executed` | Hành động ra bên ngoài đã thực hiện (luôn 0). |
| `tool_calls_mean`, `runtime_ms_mean` | Chi phí: số lần gọi công cụ và thời gian mỗi deal. |

Các con số đã kiểm chứng nằm ở mục "Số liệu tham chiếu" cuối tài liệu và trên slide "Baseline vs agent".

🗣️ **Say:** "We compared three systems on exactly the same 1,589 real open deals and the same frozen model. The agent cites more types of evidence, finds the data problems, and gives no directional label when the score is out of support or too close to the threshold. Every claim it makes is re-checked from the raw files. Then we turned off one part of the agent at a time. Without the reliability checks, unsupported labels come back. Without planning, the agent calls more tools but makes the same decisions, so planning saves cost. These are software and workflow measurements. They are not sales results."

**Bước này chứng minh:** đóng góp của từng thành phần được đo bằng thí nghiệm, không chỉ mô tả.

---

## STEP 10 — Kết quả khoa học và giới hạn (≈ 2 phút)

**File mở:** `reports/crm/final/metrics/feature_set_comparison.csv`, `reports/crm/final/agent/replay_outcomes.csv`, `NOVELTY_AND_RESULTS.md`.

**Cách đọc số:**
- `feature_set_comparison.csv`: 12 dòng (6 model × đặc trưng raw/history). Cột `selected=True` là model được khóa trên validation (history/Logistic Regression). Test ROC-AUC của mọi model nằm gần 0.5.
- Có model đạt test ROC-AUC cao hơn (ví dụ CatBoost raw), nhưng **chúng tôi không đổi model sau khi xem test**; đó là quy tắc chống gian lận.
- `replay_outcomes.csv`: chạy lại agent trên 1,361 deal test đã đóng **với kết quả bị che**, rồi mới so với kết quả thật. Đây là đánh giá offline, **không phải** học online hay chứng minh tăng doanh số.

🗣️ **Say:** "Here are the honest results. On the real data, no model ranks deals much better than chance. The selected model has a test ROC-AUC of about 0.52. Some other model looks a bit better on test, but we do not switch models after seeing the test data. The agent does not make the model more accurate. What it improves is how safely and transparently the score is used. We did not measure any increase in sales."

**Bước này chứng minh:** tách bạch ba loại khẳng định: chất lượng dự đoán, chất lượng quy trình agent, và giá trị kinh doanh (chưa được chứng minh).

---

## STEP 11 — Kết thúc bằng tuyên bố về tính mới (novelty) (≈ 1 phút)

**Mở slide** "What our project adds" (bảng đóng góp) hoặc `docs/AGENT_ARCHITECTURE.md` mục 5.

🗣️ **Say:** "To summarize our novelty: we did not invent a new learning algorithm. Our contribution is the agent architecture. First, it plans which evidence to collect by asking whether that evidence could change the decision. Second, it refuses to give a confident label when the data is missing, borderline, conflicting, or when the model is at chance level. Third, it acts on its own only inside safe, local limits, and it blocks external actions for human approval. Fourth, every decision has a replayable trace. We measured each of these against two baselines and with ablations, on real data, and we report the weak prediction results honestly."

**Bước này chứng minh:** tính mới nằm ở kiến trúc agent và có bằng chứng thí nghiệm.

---

## LIVE DEMO 3–5 PHÚT (sau khi đã cài đặt)

Mở sẵn một Terminal ở thư mục repo, font lớn. Chạy lần lượt:

| Thời gian | Lệnh | Câu nói (tiếng Anh) |
|---|---|---|
| 0:00–0:30 | `.venv-crm/bin/python -m src.agent_demo --check` | "Everything is already trained. The model was chosen on validation only." |
| 0:30–2:30 | `.venv-crm/bin/python -m src.agent_demo` | "One command. The agent sets a goal, plans each tool by need minus cost, calls real tools, checks the data, and picks the most specific action the evidence supports. It writes only a local review task." (chỉ vào `outside_training_range`, `DECIDE`, `create_review_task`, `COMPARISON`) |
| 2:30–3:30 | `.venv-crm/bin/python -m src.agent_demo --scenario missing-account --quiet` | "With a missing account, the agent does not trust the score. It asks to fix the data first." |
| 3:30–4:30 | `.venv-crm/bin/python -m src.agent_demo --scenario prohibited --quiet` | "If someone asks it to email the customer, the safety guard blocks it and queues it for a human." |
| 4:30–5:00 | mở `reports/crm/final/figures/agent_comparison.png` | "On all 1,589 real open deals, the agent gives no unsupported labels and leaves fewer manual steps. It does not change the model's accuracy." |

**Phương án dự phòng (nếu máy demo lỗi):** mở output **đã lưu từ lần chạy đã kiểm chứng**: `reports/crm/final/agent/scenario_runs/S1/traces/01XZ9CRY.full_agent.report.md`, `S2/...`, `S7/...` và `reports/crm/final/agent/scenario_results.csv`. Nói rõ: "This is the saved output of our verified run, not a live run."

---

## Số liệu tham chiếu (verified run, từ `reports/crm/final/`)

| Mục | Giá trị (verified run) | File nguồn |
|---|---|---|
| Model được chọn (khóa trên validation trước khi xem test) | history / Logistic Regression | `models/selection_lock.json` |
| ROC-AUC validation của model được chọn | 0.569 (95% CI 0.523–0.616) | `models/model_card.json` |
| ROC-AUC test của model được chọn | 0.5168 (raw LR: 0.5238; 12 cặp: 0.480–0.542) | `metrics/feature_set_comparison.csv` |
| Ngưỡng quyết định (đã hiệu chỉnh) | 0.6109 | `run_manifest.json` |
| Deal Engaging được chấm điểm / thiếu account | 1,589 / 1,088 | `predictions/open_deal_predictions.csv` |
| Deal demo mặc định (STEP 2–6) | 01XZ9CRY: P(Won) 0.534 → `ESCALATE_AT_RISK_REVIEW`, 12 bước | `agent/scenario_runs/S1/` |
| Deal thiếu account (STEP 7) | UU7N7LZK → `DATA_COMPLETION` | `agent/scenario_runs/S2/` |
| Nhãn định hướng không có cơ sở (A / B / agent) | 1,242 / 487 / 0 | `agent/system_comparison.csv` |
| Phát hiện vấn đề dữ liệu (A / B / agent) | 0.28 / 0.64 / 1.00 | `agent/system_comparison.csv` |
| Bước thủ công còn lại, trên 9 (A / B / agent) | 5.0 / 3.0 / 2.22 | `agent/system_comparison.csv` |
| Kịch bản đạt (A / B / agent) | 4/10 / 6/10 / 10/10 | `agent/scenario_results.csv` |
| Ablation: bỏ kiểm tra độ tin cậy | 80% quyết định thay đổi; 53% nhãn không có cơ sở | `agent/ablation_summary.csv` |
| Ablation: bỏ lập kế hoạch | quyết định giữ nguyên; +2.1 lần gọi công cụ; chậm 2.9× | `agent/ablation_summary.csv` |
| Tái lập | 200/200 trace giống hệt; 880/880 lần gọi công cụ phát lại khớp | `agent/experiment.json` |
| Replay có che kết quả (1,361 deal test) | deal bị escalate thắng 62.3% so với 60.0% trung bình → không tốt hơn ngẫu nhiên | `agent/replay_outcomes.csv` |

Đường dẫn tương đối so với `reports/crm/final/`. Nếu bạn chạy lại `python -m src.run_project` trên máy khác, một số số liệu của mạng nơ-ron (TabNet, MLP) có thể khác nhẹ; khi đó hãy đọc số trên màn hình. Hai lần chạy liên tiếp trên máy kiểm chứng (Linux, CPU) cho kết quả giống hệt nhau.
