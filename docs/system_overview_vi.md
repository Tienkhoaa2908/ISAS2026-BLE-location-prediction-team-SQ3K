# Tổng quan hệ thống SQ3K — ISAS 2026 BLE Location Prediction

Tài liệu này dùng để các thành viên trong nhóm nắm toàn bộ hệ thống trước phần thuyết trình và Q&A. Nội dung bám theo mã nguồn, notebook và kết quả đã khóa trong repository.

## 1. Bài toán của nhóm đang giải quyết

Mục tiêu là dự đoán vị trí trong nhà từ dữ liệu Bluetooth Low Energy (BLE). Hệ thống nhận các gói BLE theo thời gian, chuyển chúng thành các fingerprint có độ dài cố định, sau đó dự đoán một trong 22 lớp vị trí.

Phương pháp của nhóm có hai phần chính:

1. một mô hình chính dùng cửa sổ 10 giây để đưa ra dự đoán mặc định;
2. một mô hình specialist dùng ngữ cảnh 50 giây để xem xét lại một số trường hợp khó thuộc nhóm phòng bệnh.

Specialist không được thay thế mô hình chính trên toàn bộ dữ liệu. Một directional gate quyết định khi nào specialist được phép sửa dự đoán của mô hình chính.

Luồng tổng quát:

```text
BLE events + location intervals
            |
            v
10-second fingerprints, stride 2 seconds
            |
            v
175 features
            |
            v
Hierarchical Random Forest
            |
            +------------------------------+
            |                              |
            | main prediction              |
            |                              v
            |                       Directional gate
            |                              ^
            |                              |
            +---- 50-second context -------+
                        |
                        v
                 119 context features
                        |
                        v
              14-room RF specialist

Directional gate
      |
      v
selected corrected prediction
      |
      v
centered temporal smoothing
      |
      v
final sequence
```

Điểm cần nhớ là mô hình chính luôn là bộ ra quyết định mặc định. Specialist chỉ có quyền can thiệp trong một phạm vi nhỏ và khi thỏa các điều kiện đã cố định.

---

## 2. Dữ liệu đi vào pipeline

Dữ liệu challenge không được đưa lên repository public. Repository chỉ chứa code, config, notebook và kết quả tổng hợp. Khi chạy lại pipeline cần file riêng `data.zip`.

Các file dữ liệu chính mà pipeline sử dụng gồm:

```text
data/derived/train_fingerprints.csv
data/derived/5f_ble_merged_clean(friend_data).csv
data/derived/BLE_processed.csv
data/labels/user_97.csv
data/labels/5f_label_loc_train.csv
data/test/BLE_Test_predict.csv
data/cache/cache_oof_train_fingerprints_stack_rf_r10_f5_proba.npz
```

`train_fingerprints.csv` là bảng fingerprint 10 giây đã được tạo từ BLE event-level data và location intervals. Pipeline có script để tạo lại bảng này và kiểm tra SHA-256 nhằm bảo đảm dữ liệu tái tạo đúng với bảng đã dùng trong thí nghiệm.

Các kiểm tra khóa của bảng fingerprint:

- 15,382 windows được giữ lại;
- 22 lớp vị trí;
- 175 đặc trưng số;
- SHA-256:

```text
d95ce428732af277c74bf8bd3fb16301c9aaeddbbb1c46a6b394905bfe5b3877
```

---

## 3. Cách tạo fingerprint 10 giây

Dữ liệu BLE ban đầu là các quan sát theo timestamp. Hệ thống gom chúng thành các cửa sổ thời gian dài 10 giây, với stride 2 giây.

```text
window length = 10 s
stride        = 2 s
overlap       = 8 s
```

Nhãn của mỗi window được xác định theo midpoint của window. Với window bắt đầu tại thời điểm `t`, midpoint là `t + 5 s`. Nếu midpoint nằm trong một location interval hợp lệ thì window nhận location đó làm nhãn.

### 3.1. 125 đặc trưng của cửa sổ hiện tại

Có 25 BLE beacon. Với mỗi beacon, hệ thống lấy 5 thống kê:

1. mean RSSI;
2. presence;
3. RSSI standard deviation;
4. maximum RSSI;
5. packet count.

Do đó:

```text
25 beacons × 5 features = 125 features
```

### 3.2. 50 đặc trưng lịch sử

Ngoài cửa sổ hiện tại, hệ thống dùng thông tin từ 8 retained windows trước đó. Với mỗi beacon, hai đại lượng được tổng hợp là:

- mean packet count;
- mean RSSI standard deviation.

Do đó:

```text
25 beacons × 2 history features = 50 features
```

Tổng số đặc trưng của main path:

```text
125 current-window features
+ 50 history features
= 175 features
```

Mục đích của 10-second path là giữ độ phân giải thời gian tương đối ngắn để không làm mất các lần ghé phòng ngắn.

---

## 4. Main model: Hierarchical Random Forest

Main model là mô hình dự đoán mặc định của hệ thống.

Thay vì đưa 22 lớp vào một Random Forest phẳng duy nhất, nhóm dùng cấu trúc hai tầng.

### Tier 1

Phân biệt hai nhóm lớn:

- numbered patient rooms;
- common areas.

### Tier 2

Sau khi Tier 1 xác định nhóm, Tier 2 dự đoán vị trí cụ thể trong nhóm đó.

Ý tưởng của cấu trúc này là tách quyết định cấp cao trước khi giải quyết nhầm lẫn chi tiết giữa các vị trí cùng loại.

Cấu hình Random Forest chính:

```text
n_estimators = 400
max_features = sqrt
class_weight = balanced_subsample
```

Để xử lý class imbalance, Borderline-SMOTE type 1 được áp dụng chỉ bên trong training fold, với target support cho minority class là 200.

Điểm cần nhớ cho Q&A: SMOTE không được chạy trước khi chia validation. Làm như vậy sẽ làm synthetic samples mang thông tin từ validation vào training và gây leakage.

---

## 5. Vì sao có thêm specialist 50 giây

Main model chỉ nhìn một fingerprint 10 giây cùng phần history ngắn. Một số patient rooms vẫn khó phân biệt khi tín hiệu trong một đoạn ngắn không đủ rõ.

Nhóm vì vậy tạo thêm một representation độc lập trên context 50 giây.

Specialist chỉ được huấn luyện trên 14 patient-room classes:

```text
501, 502, 503, 506, 508, 510, 511,
512, 513, 515, 518, 520, 522, 523
```

Nó không phải một classifier 22 lớp thay thế main model.

### 5.1. 119 context features

Trong mỗi context 50 giây, với từng beacon, hệ thống lấy:

- mean RSSI;
- packet count;
- maximum RSSI;
- RSSI standard deviation.

```text
25 beacons × 4 = 100 features
```

Ngoài ra có 19 global/context features, gồm các thống kê RSSI tổng thể, số lượng mẫu, độ phủ beacon và biểu diễn thời gian trong ngày.

```text
100 per-beacon features
+ 19 global/context features
= 119 features
```

Các context features được xây dựng không dùng nhãn tương lai để tạo feature. Khi validation, các specialist training rows có cùng exact 50-second context bin với validation được loại khỏi training.

Specialist dùng Random Forest:

```text
n_estimators = 1200
class_weight = balanced_subsample
max_features = sqrt
random_state = 42
```

---

## 6. Directional Long-Context Specialist Correction

Đây là phần quan trọng nhất của phương pháp.

Nếu chỉ lấy specialist thay main model hoặc blend hai model trên toàn bộ dữ liệu, specialist có thể làm hỏng những dự đoán mà main model vốn đã làm tốt. Vì vậy DLCS chỉ cho specialist sửa một số dự đoán cụ thể.

### 6.1. Main model vẫn giữ quyền quyết định

Mỗi sample trước hết có:

- class dự đoán của main model;
- probability distribution của main model;
- probability distribution của specialist trong 14-room scope.

Specialist probability được kết hợp với probability của main model trong room scope để tạo candidate correction. Trong implementation hiện tại, trọng số specialist là:

```text
alpha = 0.75
```

Nhưng candidate đó chưa được dùng ngay. Nó còn phải đi qua gate.

### 6.2. Điều kiện để gate được kích hoạt

Một sample chỉ được xem xét correction khi:

- có 50-second context hợp lệ;
- main prediction nằm trong specialist room scope;
- top-1 và top-2 class của main model đều nằm trong room scope;
- margin giữa top-1 và top-2 của main model không lớn hơn 0.30.

Điều kiện margin có ý nghĩa đơn giản: nếu main model đang rất chắc chắn thì specialist không được can thiệp.

### 6.3. Protected source classes

Một số class được bảo vệ, không cho specialist override khi chúng là prediction của main model:

```text
501, 502, 516, 522, 523
```

Lý do là trong giai đoạn development, broad specialist gây negative transfer đáng kể lên các class này.

### 6.4. Các target được phép correction

Frozen gate hiện chỉ cho candidate correction vào các target:

```text
508, 510, 513, 515, 518, 520
```

### 6.5. Confusion clusters

Correction còn phải phù hợp với confusion cluster của target. Ví dụ:

```text
508 -> {508, 511, 520}
510 -> {510, 511, 512}
513 -> {501, 513, 515}
515 -> {501, 513, 515}
518 -> {515, 517, 518, 520, 522}
520 -> {518, 520, 522, 523}
```

Nếu top-1 và top-2 của main model không nằm trong cluster phù hợp thì correction bị chặn.

### 6.6. Candidate margin

Cuối cùng, specialist candidate phải thắng source class đủ một margin nhất định. Margin mặc định là 0.03. Một số cặp có threshold riêng cao hơn.

Ví dụ:

```text
522 -> 520 : 0.12
523 -> 520 : 0.12
501 -> 513 : 0.10
501 -> 515 : 0.10
```

Chỉ khi qua toàn bộ các điều kiện trên thì prediction mới bị override.

Đây là lý do tên phương pháp có từ `Directional`: correction không xảy ra tự do giữa mọi cặp class mà chỉ xảy ra theo các hướng đã được kiểm soát.

---

## 7. Gate thực sự thay đổi bao nhiêu prediction

Trong repeated confirmation, gate chỉ override khoảng 48 đến 60 predictions trong mỗi repetition, dưới 0.4% của 15,382 fingerprints.

Điểm này quan trọng khi giải thích phương pháp. DLCS không cố thay đổi hàng loạt prediction. Nó giữ main model trên gần như toàn bộ dữ liệu và chỉ can thiệp ở một số trường hợp được gate xác định là phù hợp.

Trong development protocol:

```text
Main Macro-F1 = 0.5367
DLCS Macro-F1 = 0.5487
```

Mức cải thiện xuất hiện ở cả 4 confirmation repetitions. Một số patient rooms hưởng lợi rõ hơn, ví dụ room 508 có mức tăng F1 lớn nhất trong phân tích per-class.

Không nên diễn giải các con số này thành kết luận rằng DLCS luôn tốt hơn ở mọi điều kiện. Phần validation bên dưới giải thích rõ giới hạn đó.

---

## 8. Temporal smoothing

Prediction theo từng 10-second window có thể dao động giữa các window liền nhau. Pipeline có thêm bước centered temporal smoothing để ổn định chuỗi prediction.

Đây là hậu xử lý theo chuỗi, không phải một phần của Random Forest hay specialist.

Điểm cần nhớ:

- main sequence và DLCS sequence đều được áp dụng cùng một smoothing rule khi so sánh;
- centered smoothing sử dụng cả các prediction ở phía trước và phía sau vị trí hiện tại;
- vì vậy đây là offline/non-causal smoothing, không nên mô tả như một thuật toán real-time causal deployment.

---

## 9. Hai protocol validation và lý do phải có cả hai

### 9.1. Repeated session-grouped validation

Đây là development protocol dùng để so sánh main model với DLCS.

Nhóm dùng 4 confirmation repetitions:

```text
random_state = 10, 11, 12, 13
5 folds mỗi repetition
```

Kết quả cho thấy DLCS cải thiện Main ở cả 4 repetitions.

Tuy nhiên, sau đó nhóm audit lại đơn vị validation và phát hiện một vấn đề.

Fingerprint 10 giây dùng stride 2 giây nên các window chồng lấn mạnh. Một physical visit có thể bị chia thành nhiều contiguous session groups. Audit cho thấy:

```text
332 labeled visits
80 visits bị split qua nhiều session groups
= 24.1%
```

Khoảng 44.3–44.8% validation windows có một window khác của cùng labeled visit xuất hiện trong training ở repeated session split.

Vì vậy session-grouped result được xem là within-period diagnostic, không phải bằng chứng mạnh cho cross-day generalization.

### 9.2. Leave-one-day-out validation

Nhóm bổ sung fixed-22-class leave-one-day-out, viết tắt LODO.

Mỗi lần:

```text
1 ngày được giữ hoàn toàn làm validation
3 ngày còn lại dùng để train
```

Điều này ngăn một physical visit của cùng ngày xuất hiện ở cả train và validation.

Dưới LODO, DLCS vẫn tạo một cải thiện nhỏ trên raw predictions nhưng lợi ích đó không còn sau khi áp dụng temporal smoothing. Vì vậy paper không kết luận rằng DLCS giải quyết được cross-day signal shift.

Điểm cần trả lời nếu bị hỏi: việc nhóm chủ động audit lại validation và bổ sung LODO là để tách rõ hiệu quả trong cùng deployment period với khả năng generalize sang ngày khác.

---

## 10. Vì sao final hidden-test submission không dùng DLCS

Đây là điểm tất cả thành viên cần nhớ để tránh trả lời mâu thuẫn.

Paper nghiên cứu DLCS và chứng minh selective correction có ích trong development protocol. Nhưng khi chọn pipeline để tạo hidden-test submission, nhóm ưu tiên kết quả từ strict LODO.

Sau common centered smoothing, Main model tốt hơn DLCS một chút trong LODO. Vì vậy file prediction cuối cùng được tạo bằng:

```text
Hierarchical Main RF
        +
centered temporal smoothing
```

không dùng DLCS correction trong final hidden-test sequence.

Đây không phải mâu thuẫn. Paper nghiên cứu một correction mechanism và đồng thời báo cáo giới hạn của nó. Khi phải chọn một cấu hình duy nhất để submit hidden test, nhóm chọn cấu hình có robustness tốt hơn trong protocol nghiêm ngặt hơn.

Locked hidden-test output:

```text
62,222 rows
SHA-256:
4b1a80a214b53b531a40467a515e2afbe4350b7b0533144ec1415c3eb9a43d80
```

Hidden-test score cuối cùng thuộc phía Ban Tổ chức và không được suy đoán từ training validation.

---

## 11. Pipeline final test inference

`notebooks/test_prediction.ipynb` và `scripts/test_inference.py` thực hiện các bước sau:

1. đọc `train_fingerprints.csv`;
2. đọc raw `BLE_Test_predict.csv`;
3. chuyển timestamp test về format dùng bởi fingerprint builder;
4. tạo test fingerprints bằng cùng window 10 s, stride 2 s và cùng feature definitions như training;
5. train hierarchical RF trên toàn bộ labeled training fingerprints;
6. dự đoán test windows;
7. áp dụng centered temporal smoothing;
8. map window-level prediction trở lại từng packet row của file test gốc;
9. tạo `SQ3K_prediction.csv`;
10. kiểm tra row count, missing prediction, class validity và SHA-256.

Không có hidden-test label được sử dụng để chọn threshold hoặc model.

---

## 12. Cấu trúc repository

```text
README.md
requirements.txt

config/
  beacon_coords.json
  room_centers.json

docs/
  data.md
  system_overview_vi.md

notebooks/
  training.ipynb
  test_prediction.ipynb

src/
  build_fingerprints.py
  model_core.py
  train_rf.py

  models/
    hier_adj.py
    resample_wrap.py
    train_resample.py

  dlcs/
    common.py
    evaluation_core.py
    session_grouped_validation.py
    verify_results.py

scripts/
  prepare_data.py
  verify_fingerprints.py
  session_grouped_validation.py
  lodo_validation.py
  test_inference.py

results/
  session_grouped/
  lodo/
```

### Vai trò các phần chính

`src/build_fingerprints.py`
: tạo fingerprint 10 giây từ BLE events và location intervals.

`src/models/`
: implementation của hierarchical Random Forest và resampling.

`src/dlcs/common.py`
: xây 50-second context, 119 context features và temporal smoothing utilities.

`src/dlcs/evaluation_core.py`
: train main model, train specialist, directional gate và metric logic dùng trong validation.

`scripts/session_grouped_validation.py`
: entry point để chạy repeated session-grouped confirmation.

`scripts/lodo_validation.py`
: chạy fixed-22-class leave-one-day-out.

`scripts/test_inference.py`
: tạo final hidden-test CSV.

`notebooks/training.ipynb`
: notebook để chạy preprocessing, fingerprint verification, session-grouped validation và LODO.

`notebooks/test_prediction.ipynb`
: notebook final inference.

---

## 13. Cách chạy lại

### Bước 1: cài dependencies

```bash
python -m pip install -r requirements.txt
```

### Bước 2: chuẩn bị private data

```bash
python scripts/prepare_data.py \
  --archive /path/to/data.zip \
  --out runtime/Dataset
```

### Bước 3: kiểm tra fingerprint reconstruction

```bash
python scripts/verify_fingerprints.py \
  --data-root runtime/Dataset \
  --out runtime/rebuilt_train_fingerprints.csv
```

### Bước 4: repeated session-grouped confirmation

```bash
python scripts/session_grouped_validation.py \
  --data-root runtime/Dataset \
  --outdir runtime/session_grouped \
  --trees 1200
```

### Bước 5: LODO

```bash
python scripts/lodo_validation.py \
  --data-root runtime/Dataset \
  --outdir runtime/lodo \
  --specialist-trees 1200 \
  --force
```

### Bước 6: final test inference

```bash
python scripts/test_inference.py \
  --data-root runtime/Dataset \
  --out SQ3K_prediction.csv
```

Lệnh cuối phải in:

```text
matches_locked_submission True
```

---

## 14. Những câu Q&A mà cả nhóm cần trả lời được

### Tại sao không dùng một classifier 22 lớp duy nhất?

Nhóm dùng hierarchical classifier để tách quyết định giữa patient rooms và common areas trước, sau đó mới dự đoán class cụ thể. Đây là main model đã được giữ cố định trước khi thêm DLCS.

### Tại sao dùng Random Forest?

Input là engineered tabular BLE fingerprints với nhiều thống kê RSSI, presence, count và history. Random Forest phù hợp với dạng feature này, xử lý quan hệ phi tuyến và không yêu cầu lượng dữ liệu lớn như các temporal deep models. Quan trọng hơn, mục tiêu nghiên cứu của paper nằm ở decision architecture và selective correction, không phải đề xuất một backbone deep-learning mới.

### Tại sao cần 10 giây và 50 giây cùng lúc?

10-second path giữ độ phân giải thời gian và là predictor chính. 50-second path cung cấp context dài hơn cho một số patient-room ambiguity. Nếu dùng long context cho mọi prediction, nó có thể làm mờ transition và gây negative transfer.

### Vì sao specialist chỉ có 14 rooms?

Specialist được thiết kế cho patient-room confusions, không phải thay thế global classifier. Giới hạn scope giúp nó tập trung vào các class mà long context có khả năng hữu ích.

### Vì sao cần gate?

Specialist không luôn tốt hơn main model. Gate giữ nguyên main prediction khi model đang chắc chắn hoặc khi candidate correction không phù hợp với confusion structure đã validate. Điều này làm giảm negative transfer.

### Vì sao gọi là directional correction?

Vì correction không được phép tự do từ mọi source sang mọi target. Source, target, confusion cluster và margin đều bị giới hạn. Một số source classes còn được bảo vệ hoàn toàn.

### Gate có sửa nhiều prediction không?

Không. Trong repeated confirmation, mỗi repetition chỉ có khoảng 48–60 overrides, dưới 0.4% dataset. Đây là targeted correction.

### Kết quả tốt nhất của DLCS là gì?

Trong repeated session-grouped confirmation, Macro-F1 tăng từ 0.5367 lên 0.5487 và cải thiện ở cả 4 confirmation repetitions. Không nên trình bày con số này như bằng chứng cross-day generalization vì protocol có same-visit exposure.

### Nhóm xử lý leakage như thế nào?

Borderline-SMOTE chỉ chạy trong training fold. Specialist training rows có exact 50-second context trùng với validation được purge. Sau đó nhóm còn audit physical visits và bổ sung leave-one-day-out để đánh giá day separation.

### Tại sao audit rồi vẫn giữ session-grouped result trong paper?

Vì nó là protocol dùng trong development và cho biết DLCS hoạt động thế nào trong cùng deployment period. Paper giữ kết quả đó nhưng hạ mức diễn giải của nó, đồng thời dùng LODO làm robustness check.

### Vì sao final test không dùng DLCS dù paper nói về DLCS?

Vì strict LODO sau smoothing cho thấy main model ổn định hơn một chút. Nhóm chọn main + smoothing cho hidden test thay vì chọn cấu hình chỉ vì nó đẹp hơn ở development validation.

### Hạn chế lớn nhất hiện tại là gì?

Cross-day distribution shift chưa được giải quyết. Một số class rất hiếm hoặc không xuất hiện trong training portion của một LODO fold nên không thể học đầy đủ. Centered smoothing cũng là offline/non-causal.

### Có dùng hidden-test data để tune model không?

Không dùng hidden-test labels hoặc hidden-test score để chọn model, gate hay threshold. Final inference chỉ áp dụng pipeline đã khóa lên hidden test.

---

## 15. Các con số nên nhớ

Không cần thuộc mọi metric. Cả nhóm nên nhớ các số sau:

```text
22 location classes
25 BLE beacons
10 s main window
2 s stride
175 main features
50 s specialist context
119 specialist features
14 specialist room classes
400 trees in the main Random Forest
1200 trees in the specialist Random Forest
15,382 retained training fingerprints

Main development Macro-F1: 0.5367
DLCS development Macro-F1: 0.5487
4/4 confirmation repetitions improved
about 48–60 overrides per repetition
< 0.4% predictions changed by the gate

final hidden-test rows: 62,222
```

Không cần chủ động đọc các LODO F1 thấp trong presentation. Nếu được hỏi về robustness thì giải thích kết luận: DLCS còn một small raw gain trong cross-day evaluation nhưng gain không giữ được sau common temporal smoothing, nên paper không claim cross-day improvement.

---

## 16. Cách hiểu ngắn nhất về đóng góp của bài

Nếu cần giải thích bài trong khoảng 20–30 giây:

> Hệ thống dùng một hierarchical Random Forest trên fingerprint 10 giây làm predictor chính. Nhóm bổ sung một Random Forest specialist dùng context BLE 50 giây cho 14 patient rooms. Specialist không được thay main model trực tiếp mà phải qua một directional gate dựa trên uncertainty, class scope, confusion clusters và probability margins. Gate chỉ sửa một số rất ít prediction và cải thiện nhất quán trong repeated development validation. Sau đó nhóm audit validation và bổ sung leave-one-day-out, qua đó xác định rằng cross-day shift vẫn là giới hạn chưa được giải quyết. Vì vậy final hidden-test submission dùng main model cộng temporal smoothing, là cấu hình ổn định hơn trong strict LODO.

Đây là nội dung cốt lõi mà tất cả thành viên nên thống nhất khi trả lời Q&A.
