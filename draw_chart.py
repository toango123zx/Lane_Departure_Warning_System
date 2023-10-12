import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
np.random.seed(0)
n_samples = 200
X = np.random.randn(n_samples, 2)

# Thêm một số điểm nhiễu xa
outliers = np.random.uniform(low=-10, high=10, size=(10, 2))
X = np.concatenate([X, outliers])

# Áp dụng thuật toán DBSCAN để loại bỏ điểm nhiễu
print(type(X))
eps = 10.0  # Độ lớn của cửa sổ
min_samples = 5  # Số điểm tối thiểu trong mỗi cụm
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X)

# Lấy các điểm không phải điểm nhiễu
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
print(type(core_samples_mask))
print(core_samples_mask)
print(type(X))
print(X)
non_noise_points = X[core_samples_mask]

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))

# Vẽ các điểm trước khi loại bỏ nhiễu
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='b', alpha=0.6)
plt.title('Các điểm trước khi loại bỏ nhiễu')
plt.xlabel('X')
plt.ylabel('Y')

# Vẽ các điểm sau khi loại bỏ nhiễu
plt.subplot(1, 2, 2)
plt.scatter(non_noise_points[:, 0], non_noise_points[:, 1], c='r', alpha=0.6)
plt.title('Các điểm sau khi loại bỏ nhiễu')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
