# 📊 HOTEL BOOKING CANCELLATION ANALYSIS (DATA MINING PROJECT)

## 📌 1. Giới thiệu

Dự án này áp dụng các kỹ thuật **Data Mining** và **Machine Learning** để phân tích hành vi đặt phòng khách sạn và dự đoán khả năng **huỷ phòng (cancellation)**.

Mục tiêu:

* Phát hiện các yếu tố ảnh hưởng đến việc huỷ phòng
* Xây dựng mô hình dự đoán huỷ phòng
* Phân tích hành vi khách hàng
* Trực quan hoá dữ liệu

---

## 🎯 2. Các kỹ thuật sử dụng

### 🔹 2.1 Tiền xử lý dữ liệu

* Xử lý missing values
* Rời rạc hoá:

  * `lead_time`
  * `country`
  * `channel`
* Loại bỏ **data leakage**
  (reservation_status, reservation_status_date)

---

### 🔹 2.2 Khai phá luật kết hợp (Apriori)

* Tìm các tổ hợp thuộc tính liên quan đến huỷ phòng
* Các chỉ số:

  * Support
  * Confidence
  * Lift

👉 Ví dụ:

* Khách đặt sớm + Non Refund → khả năng huỷ cao

---

### 🔹 2.3 Phân cụm (Clustering)

* Thuật toán: KMeans
* Chuẩn hoá dữ liệu trước khi phân cụm
* Profiling:

  * Xác định cụm có tỷ lệ huỷ cao

---

### 🔹 2.4 Phân lớp (Classification)

Các mô hình sử dụng:

* Logistic Regression
* Random Forest
* SVM

Xử lý mất cân bằng:

* SMOTE
* class_weight

Đánh giá:

* F1-score
* PR-AUC

---

### 🔹 2.5 Semi-Supervised Learning

* Giả lập thiếu nhãn
* Self-training với ngưỡng tin cậy cao
* Phân tích pseudo-label

---

### 🔹 2.6 Chuỗi thời gian (Time Series)

* Phân tích tỷ lệ huỷ theo tháng
* Dự báo đơn giản bằng shift

---

## 🧠 3. Kiến trúc project

```
DATA_MINING_PROJECT/
│
├── app.py                 # Web dashboard (Flask)
├── run_pipeline.py        # Chạy toàn bộ pipeline
│
├── data/
│   └── raw/
│       └── hotel_bookings.csv
│
├── templates/             # HTML giao diện
├── static/                # Ảnh biểu đồ
│
└── src/
    ├── data/
    │   └── cleaner.py
    │
    ├── features/
    │   └── builder.py
    │
    ├── models/
    │   ├── trainer.py
    │   ├── cluster.py
    │
    ├── mining/
    │   └── apriori.py
    │
    ├── evaluation/
    │   └── evaluator.py
    │
    ├── semi/
    │   └── self_training.py
    │
    └── timeseries/
        └── ts_model.py
```

---

## ⚙️ 4. Cách chạy project

### 🔹 4.1 Cài đặt thư viện

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask mlxtend imbalanced-learn
```

---

### 🔹 4.2 Chạy pipeline

```bash
python run_pipeline.py
```

---

### 🔹 4.3 Chạy web

```bash
python app.py
```

Mở trình duyệt:

```
http://127.0.0.1:5000
```

---

## 📊 5. Kết quả

Project cung cấp:

* 📈 Biểu đồ EDA theo tháng
* 🔥 Ma trận tương quan
* 📉 PR Curve
* 🎯 Clustering khách hàng
* 📊 Time series cancellation

---

## ⚠️ 6. Lưu ý

* Dataset lớn → nên dùng `.sample()` để tránh lag
* SVM có thể chạy chậm → đã tối ưu kernel linear
* SMOTE có thể tốn tài nguyên

---

## 📚 7. Tài liệu tham khảo

* Data Mining: Concepts and Techniques – Jiawei Han
* Hands-On Machine Learning – Aurélien Géron
* Scikit-learn Documentation
* MLxtend Documentation
* Hotel Booking Dataset (Kaggle)

---

## 🚀 8. Hướng phát triển

* Thêm XGBoost / LightGBM
* Dashboard tương tác (Plotly)
* Triển khai web online
* Tối ưu feature engineering

---

## 👨‍💻 9. Tác giả

Đào Duy Hồng

---
