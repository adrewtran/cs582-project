# Giải thích source code — Dự đoán cơ hội bán hàng CRM bằng Machine Learning

Đây là project môn **CS 582 (Nhóm 2)**. Mục tiêu: dựa vào dữ liệu CRM của một công ty B2B (giả lập) bán phần cứng máy
tính, **dự đoán một deal (cơ hội bán hàng) sẽ Won (thắng) hay Lost (thua)**, đồng thời **giải thích yếu tố nào làm xác
suất thắng cao hơn hoặc thấp hơn**.

Toàn bộ quy trình chạy bằng **một lệnh**:

```bash
.venv-crm/bin/python -m src.run_project
```

Lệnh này đọc dữ liệu, chia tập, train 5 model, chọn model, hiệu chỉnh xác suất (calibration), đánh giá, giải thích, chấm
điểm các deal đang mở, rồi tự sinh báo cáo, slide và kịch bản thuyết trình. Kết quả chính thức nằm trong
`reports/crm/final/`.

---

## 1. Bức tranh tổng thể

Pipeline chia thành **3 bước chạy độc lập** (train → evaluate → predict).
`src/run_project.py` nối cả 3 bước lại thành một lệnh. Paper và slide do nhóm tự viết (`CRM_IEEE_Paper.docx`,
`CRM_Final.pptx`); code không sinh tài liệu.

```
data/crm/*.csv (4 bảng, chỉ đọc)
        │
        ▼
src/data/            crm.py (kiểm tra, làm sạch, join, tạo feature, nhãn Won/Lost)
                     split.py (chia as-of theo thời gian) · preprocess.py (impute, one-hot, scale; chỉ fit trên train)
        │
        ▼
src/train.py         train 5 model (src/models/zoo.py) → chọn model trên validation → calibrate
                     (src/models/calibration.py) → lưu model (src/models/bundle.py) vào models/
        │
        ▼
src/evaluate.py      load model đã lưu → metric test (src/evaluation/metrics.py), biểu đồ (figures.py),
                     leakage/priority (audits.py), importance + SHAP (src/explain/importance.py),
                     so sánh cách chia + kiểm tra giải thích (novelty_checks.py)
        │
        ▼
src/predict.py       load model_bundle.joblib → chấm điểm deal đang mở hoặc file CSV deal mới,
                     kèm yếu tố giải thích (src/explain/reference.py) và nhóm ưu tiên (priority.py)
        │
        ▼
reports/crm/final/   data/ models/ metrics/ explain/ checks/ predictions/ figures/
```

Đường dẫn các thư mục con của kết quả được định nghĩa ở **một chỗ duy nhất**: `src/outputs.py`.

---

## 2. Dữ liệu (`data/crm/`)

Nguồn: bộ dữ liệu **Maven CRM Sales Opportunities**, gồm 4 bảng CSV:

| Bảng                 | Nội dung                                                                                                                                                          | Khóa             |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| `sales_pipeline.csv` | Mỗi dòng là một deal: agent, sản phẩm, khách hàng, giai đoạn (`deal_stage`), ngày bắt đầu (`engage_date`), ngày đóng (`close_date`), giá trị đóng (`close_value`) | `opportunity_id` |
| `accounts.csv`       | Thông tin khách hàng: ngành, năm thành lập, doanh thu, số nhân viên, địa điểm, công ty mẹ                                                                         | `account`        |
| `products.csv`       | Sản phẩm: dòng sản phẩm (series), giá bán                                                                                                                         | `product`        |
| `sales_teams.csv`    | Nhân viên sales: quản lý, văn phòng khu vực                                                                                                                       | `sales_agent`    |

`deal_stage` có 4 giá trị: **Won**, **Lost** (đã đóng), **Engaging** (đang làm việc) và **Prospecting** (mới tiềm năng).

Số liệu thực tế: 8.800 deal, trong đó 6.711 deal đã đóng (4.238 Won / 2.473 Lost).

---

## 3. Giải thích từng module

### 3.1 `src/data/dataset.py` — Container dữ liệu dùng chung

- Lớp **`Dataset`** (dataclass) gói một bảng đã làm sạch cùng vai trò của từng cột:
    - `frame`: bảng dữ liệu có nhãn,
    - `target`: tên cột nhãn 0/1,
    - `categorical` và `numeric`: danh sách cột phân loại / cột số,
    - `report`: các thống kê thu được khi làm sạch,
    - `extra`: dữ liệu phụ (ví dụ các deal đang mở).
- `features()` **chỉ trả về** các cột feature đã khai báo, nên các cột "rò rỉ" (như `close_value`) không bao giờ lọt vào
  model, dù vẫn có trong `frame` để kiểm tra/EDA.
- `__post_init__` kiểm tra: đủ cột, không có cột vừa là categorical vừa là numeric, nhãn chỉ là 0/1.
- Các hàm tiện ích: `strip_strings` (cắt khoảng trắng), `blank_to_na` (chuỗi rỗng → NaN), `basic_report` (đếm số dòng/tỉ
  lệ dương)...

### 3.2 `src/data/crm.py` — Đọc và chuẩn bị dữ liệu CRM

Đây là loader chính, hàm vào là **`build()`**:

1. **`load_raw`**: đọc 4 CSV, cắt khoảng trắng, chuyển chuỗi rỗng thành missing.
2. **`validate_raw`**: kiểm tra đủ cột bắt buộc, khóa chính không rỗng và không trùng, `deal_stage` hợp lệ, cột số không
   âm/không vô cực.
3. **`clean_pipeline`**:
    - Sửa lỗi tên sản phẩm `GTXPro → GTX Pro` (để join khớp với bảng products).
    - Parse ngày theo định dạng `%m/%d/%y`; ngày sai thì báo lỗi.
    - Deal đã đóng phải có `close_date`, và `close_date` không được trước `engage_date`.
4. **`clean_accounts`**: sửa lỗi chính tả `technolgy → technology`, tạo `is_subsidiary` (có công ty mẹ hay không) và
   `revenue_per_employee`.
5. **`join_tables`**: join pipeline với products, accounts, sales_teams theo quan hệ *many-to-one*, và **kiểm tra số
   dòng không thay đổi** sau khi join (tránh nhân bản dòng do khóa trùng).
6. **`add_derived_features`**: tạo feature thời gian **chỉ từ `engage_date`** (năm, tháng, quý, thứ trong tuần) và tuổi
   công ty tại thời điểm bắt đầu deal. Không bao giờ dùng `close_date` làm feature.
7. Tạo nhãn **`is_won`**: Won = 1, Lost = 0. Deal đang mở **không được gán là Lost**.
8. Tách riêng:
    - `labeled`: deal đã đóng (dùng để train/đánh giá), sắp xếp theo `engage_date`.
    - `scorable_open_deals`: deal **Engaging** có `engage_date` — đây là những deal sẽ được chấm điểm. Deal Prospecting
      (500 deal, không có `engage_date`) bị loại.

**Feature được dùng:**

- Categorical: `product`, `series`, `sector`, `office_location`, `regional_office`, `manager`, `sales_agent`.
- Numeric: `sales_price`, `revenue`, `employees`, `revenue_per_employee`, `year_established`, `account_age_at_engage`,
  `engage_year/month/quarter/dayofweek`, `is_subsidiary`.

**Cột bị cấm làm feature (`LEAKAGE_COLUMNS`):** `close_date`, `close_value`, `deal_stage`, `opportunity_id`. Lý do: các
cột này chỉ biết **sau khi** deal đóng (hoặc chính là nhãn / chỉ là ID). Dùng chúng sẽ cho kết quả "đẹp giả tạo".

### 3.3 `src/data/split.py` — Chia dữ liệu theo thời gian (`asof_split`)

Thay vì chia ngẫu nhiên, project chia theo **thời gian** để mô phỏng tình huống thật: dùng quá khứ để dự đoán tương lai.

- Sắp xếp theo `engage_date`, lấy mốc ở **60%** (`validation_start`) và **80%** (`test_start`).
- **Train**: deal bắt đầu *và* đã đóng trước `validation_start`.
- **Validation**: deal bắt đầu trong khoảng [60%, 80%) *và* đã đóng trước `test_start`.
- **Test**: deal bắt đầu từ `test_start` trở đi.
- Deal nào chưa biết kết quả tại mốc cắt thì bị **"purge"** (loại bỏ), vì ở thời điểm đó ta chưa thể biết nhãn của nó.
  Các deal cùng ngày luôn nằm cùng một tập.
- Mỗi tập phải có đủ cả hai lớp Won và Lost.
- Trả về đối tượng `Split` gồm chỉ số của từng tập và một `manifest` ghi vai trò của từng dòng (`train`, `validation`,
  `test`, `purged_train`, `purged_validation`).

Kết quả: train 2.975, validation 583, test 1.361, purged 1.792 dòng.

### 3.4 `src/data/preprocess.py` — Tiền xử lý

Hàm `make_preprocessor` tạo một `ColumnTransformer` của scikit-learn:

- **Cột phân loại**: điền giá trị thiếu bằng `"Missing"`, rồi **one-hot encode** (category lạ ở tập sau sẽ bị bỏ qua
  thay vì gây lỗi).
- **Cột số**: điền thiếu bằng **median**; với LR, MLP, TabNet thì thêm **chuẩn hóa** (`StandardScaler`). Random Forest
  và Dummy không cần chuẩn hóa.

Quan trọng: preprocessor **chỉ được fit trên tập train**; validation/test chỉ dùng `transform` để tránh rò rỉ thông tin.

### 3.5 `src/models/zoo.py` — 5 model

Hàm `fit_model(name, ...)` train một model trên CPU (seed cố định = 42) và trả về `FittedModel` (gồm preprocessor,
classifier, lịch sử train, cấu hình, thời gian fit):

| Model                 | Mô tả                                                                                                                                                                                    |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dummy_prior`         | Model "đối chứng": luôn dự đoán theo tỉ lệ Won trong train. Dùng làm mốc so sánh tối thiểu.                                                                                              |
| `logistic_regression` | Hồi quy logistic, `max_iter=2000`.                                                                                                                                                       |
| `random_forest`       | 300 cây (40 cây ở chế độ `--quick`), `min_samples_leaf=5`.                                                                                                                               |
| `mlp`                 | Mạng nơ-ron 2 lớp ẩn (64, 32), ReLU. Train từng epoch bằng `partial_fit`, theo dõi ROC-AUC trên validation, **early stopping** sau 15 epoch không cải thiện, giữ lại phiên bản tốt nhất. |
| `tabnet`              | Mạng TabNet (deep learning cho dữ liệu bảng, dùng PyTorch), early stopping với patience 12. Nếu thiếu thư viện thì **báo lỗi rõ ràng chứ không âm thầm bỏ qua**.                         |

`FittedModel.predict_proba` tự áp dụng preprocessor rồi gọi classifier, đồng thời kiểm tra xác suất trả về hợp lệ.

### 3.6 `src/evaluation/metrics.py` và `src/models/calibration.py` — Metric, chọn model, calibration

- **`metrics(y, p, threshold)`**: tính Accuracy, Balanced Accuracy, Precision, Recall, F1, Macro-F1, ROC-AUC, Average
  Precision, Brier score, Log loss và ma trận nhầm lẫn (TN, FP, FN, TP).
- **`choose_threshold`**: thử ngưỡng từ 0.10 đến 0.90 (bước 0.01), chọn ngưỡng cho **macro-F1 cao nhất trên validation**
  (không mặc định 0.5, không tối ưu trên test).
- **`select_model`**: chọn model có **ROC-AUC validation cao nhất**; hòa thì xét Brier thấp hơn, rồi tới tên model. Việc
  chọn model được "khóa" **trước khi nhìn kết quả test**.
- **`CalibratedModel`** (`src/models/calibration.py`): hiệu chỉnh xác suất theo kiểu sigmoid (Platt scaling): `p_mới = sigmoid(a · logit(p_cũ) + b)`,
  với `a > 0` để giữ nguyên thứ tự xếp hạng. Tham số học trên validation bằng L-BFGS-B. **Model gốc được giữ nguyên (
  frozen)**, không train lại.

### 3.7 `src/evaluation/figures.py`, `audits.py` và `src/explain/importance.py` — Phân tích và biểu đồ

Các hàm này chỉ dùng để **chẩn đoán/báo cáo**, không dùng để chọn model hay threshold:

- **`eda`** (`figures.py`): biểu đồ phân bố Won/Lost, số deal theo tháng, tỉ lệ thắng theo sản phẩm, và tỉ lệ thiếu dữ liệu giữa deal
  đã đóng và deal đang mở.
- **`evaluation_figures`**: đường ROC, đường Precision–Recall, ma trận nhầm lẫn cho từng model, biểu đồ reliability (độ
  tin cậy của xác suất) trước/sau calibration.
- **`learning_curves`**: đường loss và ROC-AUC theo epoch cho MLP và TabNet.
- **`native_importances`** (`src/explain/importance.py`, cùng `permutation_importance` và `shap_audit`): độ quan trọng feature "gốc" — hệ số của LR, impurity importance của RF, attention importance
  của TabNet.
- **`permutation_importance`**: xáo trộn từng feature trên test và đo ROC-AUC giảm bao nhiêu.
- **`leakage_control`** (`audits.py`, cùng `priority_checks`): thí nghiệm **cố ý gian lận** — train một cây quyết định chỉ với `close_value`. Kết quả cao bất
  thường cho thấy vì sao phải loại các cột rò rỉ. Đây chỉ là minh họa, không phải kết quả hợp lệ.
- **`shap_audit`**: tính **TreeSHAP** cho Random Forest (bản gốc, không phải model được chọn) trên mẫu test, và kiểm tra
  tính cộng (tổng SHAP + giá trị kỳ vọng ≈ xác suất dự đoán, sai số < 1e-5).
- **`priority_checks`**: kiểm tra tỉ lệ thắng thực tế trong từng nhóm ưu tiên (Low/Medium/High) và độ nhạy khi đổi
  ngưỡng nhóm.

### 3.8 `src/explain/reference.py` — Giải thích từng deal đang mở

- **`make_reference`**: tạo một "deal tham chiếu" từ tập train — giá trị phổ biến nhất (mode) cho cột phân loại và
  median cho cột số.
- **`reference_sensitivities`**: với mỗi deal, lần lượt thay **từng feature** bằng giá trị tham chiếu rồi đo xác suất
  Won thay đổi bao nhiêu. Chênh lệch dương nghĩa là giá trị hiện tại của feature đó đang **kéo xác suất thắng lên**.
- **`explain_rows`**: chấm điểm bất kỳ tập deal nào có đủ feature; **`explain_open`** gọi nó cho tất cả deal Engaging.
  Kết quả có các cột:
    - `win_probability`, `loss_probability`,
    - `predicted_outcome` (theo threshold đã chọn trên validation),
    - `priority`: High (≥ 0.70), Medium (≥ 0.40), Low (< 0.40),
    - `positive_factors` / `negative_factors`: 3 yếu tố tăng/giảm xác suất nhiều nhất,
    - `reference_deltas_json`: toàn bộ chênh lệch,
    - `account_missing` / `input_warning`: cảnh báo khi deal thiếu thông tin khách hàng.

Lưu ý: đây **không phải SHAP**, các chênh lệch không cộng lại thành xác suất, và **không phải quan hệ nhân quả**.

### 3.9 `src/explain/priority.py` — Hàm hỗ trợ CRM

- `priority_group(p)`: chuyển xác suất thành nhãn High/Medium/Low (ngưỡng là heuristic, chưa được kiểm chứng nghiệp vụ).

### 3.10 `src/train.py`, `src/evaluate.py`, `src/predict.py`, `src/run_project.py` — Các bước chạy

| Bước | Lệnh | Việc làm |
|---|---|---|
| Train | `python -m src.train` | Tạo `run_manifest.json`; `build()` → `asof_split()` → lưu `data/split_manifest.csv`, `data/data_quality.json`; ghi SHA-256 dữ liệu/mã nguồn, phiên bản thư viện, commit git; train 5 model, chọn threshold riêng từng model trên validation (`metrics/validation_metrics.csv`); **chọn model** theo validation rồi **calibrate**; lưu `models/trained_models.joblib` (cả 5 model) và `models/model_bundle.joblib` (model đã calibrate). **Không chấm điểm deal test nào.** |
| Evaluate | `python -m src.evaluate` | Load model đã lưu; EDA; đánh giá 5 model trên test (`metrics/test_metrics.csv`), trước/sau calibration (`metrics/calibration_test.csv`), xác suất từng deal test; biểu đồ; importance, permutation, SHAP (`explain/`); leakage, priority, so sánh cách chia, kiểm tra giải thích (`checks/`). Không thể thay đổi model đã chọn. |
| Predict | `python -m src.predict` | Load `model_bundle.joblib`, chấm điểm deal Engaging → `predictions/open_deal_predictions.csv`. Với `--input deals_moi.csv` (định dạng `sales_pipeline.csv`, cần `engage_date`) thì join với các bảng tham chiếu và chấm điểm deal mới. |
| Tất cả | `python -m src.run_project` | Chạy train → evaluate → predict (truyền dữ liệu trong bộ nhớ), rồi đặt trạng thái `complete`. Lỗi ở bất kỳ bước nào → `failed` kèm thông báo lỗi. |

Tùy chọn dòng lệnh của `run_project` (và `train`):

| Tham số          | Ý nghĩa                                                                                            |
|------------------|----------------------------------------------------------------------------------------------------|
| `--quick`        | Chạy nhanh với ngân sách nhỏ (smoke test), ghi vào `reports/crm/smoke/`, **không dùng để báo cáo** |
| `--output PATH`  | Ghi kết quả ra thư mục khác                                                                        |

Chỉ load file joblib do chính project này tạo ra (joblib có thể chạy code khi unpickle).

---

## 4. Scripts và test

| File                         | Chức năng                                                                                                                                                                                                  |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `scripts/tools/setup_cpu.py`       | Tạo môi trường ảo `.venv-crm` và cài thư viện CPU đã kiểm thử (yêu cầu Python 3.12)                                                                                                                        |
| `scripts/simulation/`        | Sinh dữ liệu mô phỏng có công thức thắng biết trước và chạy lại 5 model trên đó (`run_simulated_models.py`, `run_simulated_seeds.py`) |
| `scripts/analysis/`          | Ước tính doanh thu 3 tháng (minh họa) và xếp hạng sản phẩm trên dữ liệu mô phỏng |
| `scripts/tools/package_handoff.py` | Đóng gói ZIP và patch để bàn giao, không động vào git index                                                                                                                                                |
| `tests/`                     | Kiểm thử bằng `pytest`: dữ liệu và validate, chia thời gian (kể cả trường hợp cùng ngày), công thức metric, fit thật cả 5 model, calibration, giải thích, chạy lại từng bước từ model đã lưu, chấm điểm file CSV mới và chạy tích hợp toàn bộ |

---

## 5. Kết quả hiện tại (`reports/crm/final/`)

| Model                           | Test ROC-AUC | Accuracy | F1 (Won) |
|---------------------------------|-------------:|---------:|---------:|
| Dummy prior                     |       0.5000 |   0.6003 |   0.7502 |
| Logistic Regression (được chọn) |       0.5238 |   0.4372 |   0.2620 |
| Random Forest                   |       0.5180 |   0.5922 |   0.7422 |
| MLP                             |       0.4954 |   0.4849 |   0.5135 |
| TabNet                          |       0.5068 |   0.5356 |   0.6368 |

Số liệu từ lần chạy trên macOS ngày 2026-10-09. Lần chạy trước trên Linux cho RF 0.5194 và TabNet 0.5173; LR, MLP và mọi kết luận không đổi.

**Diễn giải trung thực:**

- ROC-AUC của mọi model đều **gần 0.5** (gần như đoán ngẫu nhiên). Với các feature có sẵn trước khi deal đóng, dữ liệu
  này gần như **không chứa tín hiệu mạnh** để dự đoán Won/Lost.
- Accuracy/F1 cao của Dummy chỉ vì nó luôn đoán Won (lớp chiếm đa số ~60%).
- Calibration cải thiện Brier score của LR (0.2794 → 0.2443) nhưng vẫn kém Dummy (0.2421) và không làm tăng ROC-AUC.
- Project **cố ý không dùng** `close_value` để "làm đẹp" số liệu; thí nghiệm leakage cho thấy nếu dùng thì điểm sẽ cao
  bất thường nhưng vô giá trị trong thực tế.

---

## 6. Các nguyên tắc thiết kế quan trọng

1. **Chống rò rỉ dữ liệu (leakage)**: loại các cột chỉ biết sau khi đóng deal; preprocessor chỉ fit trên train; chia
   theo thời gian và purge deal chưa biết kết quả.
2. **Mọi lựa chọn dựa trên validation**: model, số epoch, threshold, calibration — test chỉ dùng để đánh giá cuối.
3. **Không gán deal đang mở là Lost**; chỉ chấm điểm deal Engaging; cảnh báo khi thiếu thông tin khách hàng.
4. **Tái lập được**: seed cố định, ghi hash dữ liệu/mã nguồn, phiên bản thư viện, platform.
5. **Không giấu lỗi**: model nào lỗi thì cả pipeline báo `failed`, không âm thầm bỏ qua.
6. **Trung thực về hạn chế**: dataset đã từng được xem trước (test không hoàn toàn "sạch"), dữ liệu account/team là
   snapshot tĩnh, validation dùng lại nhiều lần, ngưỡng priority chỉ là heuristic, giải thích là hành vi của model chứ
   không phải nhân quả.
