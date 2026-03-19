# 📊 HOTEL BOOKING CANCELLATION ANALYSIS (DATA MINING PROJECT)

## 📌 1. Giới thiệu
Dự án áp dụng **Data Mining & Machine Learning** để phân tích hành vi đặt phòng và dự đoán khả năng **huỷ phòng (cancellation)**.

### 🎯 Mục tiêu
- Phân tích yếu tố ảnh hưởng đến huỷ phòng  
- Xây dựng mô hình dự đoán  
- Phân cụm hành vi khách hàng  
- Trực quan hoá dữ liệu  

---

## 🧠 2. Kỹ thuật sử dụng

### 🔹 Tiền xử lý
- Xử lý missing values  
- Rời rạc hoá:
  - lead_time
  - country
  - channel  
- Loại bỏ **data leakage**

---

### 🔹 Association Rule (Apriori)
- Tìm tổ hợp thuộc tính liên quan huỷ phòng  
- Metrics:
  - Support  
  - Confidence  
  - Lift  

---

### 🔹 Clustering
- KMeans  
- Chuẩn hoá dữ liệu  
- Xác định cụm có tỷ lệ huỷ cao  

---

### 🔹 Classification
Models:
- Logistic Regression  
- Random Forest  
- SVM  

Xử lý imbalance:
- SMOTE  
- class_weight  

Đánh giá:
- F1-score  
- PR-AUC  

---

### 🔹 Semi-Supervised Learning
- Giả lập thiếu nhãn  
- Self-training  
- Phân tích pseudo-label  

---

### 🔹 Time Series
- Phân tích tỷ lệ huỷ theo tháng  
- Dự báo bằng phương pháp shift  

---

## 📂 3. Cấu trúc project
