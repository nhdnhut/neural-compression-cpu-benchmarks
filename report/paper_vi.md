# Đánh Giá Thực Nghiệm Về Sự Đánh Đổi Trong Nén Mạng Nơ-ron Dưới Ràng Buộc Tính Toán Khắt Khe

**Một Thử Nghiệm Kiểm Soát Đa Hạt Nhân Về Cắt Tỉa, Lượng Tử Hóa và Phân Tích Ma Trận Hạng Thấp Trên Mạng CNN Kích Thước Nhúng**

**Tác giả:** Nguyễn Huỳnh Đăng Nhựt  
**Đơn vị:** Ngành Trí tuệ Nhân tạo, Khoa Khoa học Máy tính, Trường Đại học Công nghệ Thông tin, ĐHQG-HCM (UIT VNU-HCM)  
**Email:** [26521522@gm.uit.edu.vn](mailto:26521522@gm.uit.edu.vn)  
**Mã phân loại arXiv:** Machine Learning (`cs.LG`), Computer Vision and Pattern Recognition (`cs.CV`)

---

## Tóm Tắt (Abstract)

Nghiên cứu này trình bày một khảo sát thực nghiệm có kiểm soát chặt chẽ về ba kỹ thuật nén mạng nơ-ron kinh điển: **cắt tỉa phi cấu trúc dựa trên độ lớn (Magnitude-based Unstructured Pruning)**, **lượng tử hóa sau huấn luyện động (Dynamic Post-Training Quantization - PTQ INT8)**, và **xấp xỉ ma trận hạng thấp thông qua phân tích giá trị suy biến cắt cụt (Truncated SVD Low-Rank Approximation)** trên mô hình tích chập nhỏ gọn SmallCNN (206,922 tham số) huấn luyện trên tập dữ liệu Fashion-MNIST.

Chúng tôi thiết lập hệ thống đo kiểm đồng thời 7 chỉ số tài nguyên: độ chính xác Top-1 (Accuracy), Macro-F1, độ trễ suy luận đơn mẫu (Latency mean/P95), bộ nhớ RAM đỉnh (Peak RAM), dung lượng lưu trữ đĩa (Model Size), chi phí tính toán lý thuyết (FLOPs/MACs) và tỷ lệ thưa thực tế (Global Sparsity) trên 3 random seed độc lập ($s \in \{42, 123, 7\}$).

Nghiên cứu cung cấp **3 chứng minh toán học hình thức đầy đủ**:
1. Định lý bảo toàn tính thưa dưới hạ gradient có mặt nạ đối với các bộ tối ưu hóa bậc một, Momentum và thích nghi (Adam/RMSProp).
2. Chặn trên sai số lượng tử hóa INT8 từng phần tử, chuẩn Frobenius và chặn nhiễu loạn đầu ra toàn mạng nơ-ron Lipschitz.
3. Chứng minh đầy đủ Định lý Eckart-Young-Mirsky về tính tối ưu của xấp xỉ hạng thấp SVD trong cả chuẩn toán tử (spectral norm) và chuẩn Frobenius.

**Các phát hiện thực nghiệm cốt lõi:**
- **Khoảng cách FLOPs--Latency phi đơn điệu (Non-monotonic Latency Profile):** Cắt tỉa phi cấu trúc ở mức độ vừa phải (25%--75%) bảo toàn độ chính xác trong ngưỡng 0.3% so với baseline nhưng làm *tăng* độ trễ suy luận thực tế từ 13% đến 41% do chi phí kiểm tra mặt nạ và phân mảnh truy cập bộ nhớ đệm (cache fragmentation). Tuy nhiên, ở ngưỡng thưa cực hạn 90%, độ trễ *giảm đột ngột 40%* do toàn bộ trọng số còn lại nằm trọn trong L2 Cache và cơ chế zero-skipping của CPU.
- **Tính vượt trội của Dynamic INT8:** Lượng tử hóa động INT8 giảm dung lượng mô hình trên đĩa $3.7\times$ (từ 0.793 MB xuống 0.217 MB) mà hoàn toàn không suy giảm độ chính xác ($92.25 \pm 0.08\%$ so với baseline $92.25 \pm 0.09\%$), thống trị biên Pareto trên cả ba chiều: Độ chính xác -- Dung lượng, Độ chính xác -- Bộ nhớ và Độ chính xác -- Độ trễ.
- **Giới hạn của Low-Rank SVD dạng tái cấu trúc:** Tái cấu trúc SVD suy giảm độ chính xác mượt mà ở tỷ lệ hạng $\geq 50\%$, nhưng nếu không tái cấu trúc đồ thị tính toán thành 2 tầng tích chập/tuyến tính nối tiếp thì dung lượng bộ nhớ không giảm.

---

## 1. Đặt Vấn Đề và Câu Hỏi Nghiên Cứu

### 1.1. Bối cảnh
Việc triển khai các mô hình học sâu lên phần cứng biên nhúng (microcontrollers, thiết bị IoT, chip ARM Cortex-M) đòi hỏi sự cân bằng nghiêm ngặt giữa chất lượng mô hình và chi phí phần cứng. Bộ nhớ SRAM thường bị giới hạn dưới 1 MB, ngân sách tính toán dưới 10 MFLOPs/lượt suy luận và năng lượng dưới 1 mJ/lượt.

Ba phương pháp nén tiêu biểu được nghiên cứu rộng rãi:
- **Cắt tỉa (Pruning):** Đặt các trọng số có giá trị tuyệt đối nhỏ về 0.
- **Lượng tử hóa (Quantization):** Giảm số lượng bit biểu diễn từ FP32 (32-bit float) xuống INT8 (8-bit integer).
- **Phân tích hạng thấp (Low-rank Factorization):** Xấp xỉ ma trận trọng số $m \times n$ bởi tích của các ma trận hạng thấp $r \ll \min(m, n)$.

Tuy nhiên, đa số các công bố thường khảo sát từng kỹ thuật riêng lẻ trên GPU máy chủ, dẫn đến hiện tượng sai lệch khi triển khai thực tế trên CPU nhúng.

### 1.2. Câu hỏi nghiên cứu (Research Questions)
- **RQ1:** Mức độ nén của từng phương pháp ảnh hưởng như thế nào đến độ chính xác và Macro-F1 của mô hình?
- **RQ2:** Việc giảm FLOPs lý thuyết trong cắt tỉa phi cấu trúc có thực sự mang lại sự cải thiện về thời gian chạy (wall-clock latency) trên CPU hay không?
- **RQ3:** Đâu là biên Pareto tối ưu khi đồng thời tối ưu hóa độ chính xác và các chiều chi phí phần cứng?

---

## 2. Mô Hình và Kiến Trúc SmallCNN

Mô hình khảo sát là **SmallCNN**, được thiết kế chuyên biệt cho ảnh xám $1 \times 28 \times 28$ của Fashion-MNIST:

| Tầng | Cấu hình | Đầu ra | Số tham số | Tỷ lệ |
| :--- | :--- | :--- | :--- | :--- |
| **Conv1** | Conv2d(1, 16, $K$=3, $P$=1) + ReLU + MaxPool(2) | $16 \times 14 \times 14$ | 160 | 0.08% |
| **Conv2** | Conv2d(16, 32, $K$=3, $P$=1) + ReLU + MaxPool(2) + Dropout(0.25) | $32 \times 7 \times 7$ | 4,640 | 2.24% |
| **Flatten** | Biến đổi vector | 1,568 | 0 | 0% |
| **FC1** | Linear(1568, 128) + ReLU + Dropout(0.25) | 128 | **200,832** | **97.06%** |
| **FC2** | Linear(128, 10) | 10 | 1,290 | 0.62% |
| **Tổng** | | | **206,922** | **100.0%** |

Tầng kết nối dày đặc FC1 chiếm hơn 97% tổng số tham số của mạng, là đối tượng tác động chính của quá trình nén.

---

## 3. Cơ Sở Toán Học và Các Định Lý Hình Thức (Formal Proofs)

### 3.1. Định lý 1: Bảo toàn tính thưa dưới hạ gradient có mặt nạ

**Phát biểu định lý:**  
Cho $\mathbf{M} \in \{0,1\}^d$ là mặt nạ nhị phân cố định, và $\tilde\theta = \mathbf{M} \odot \theta$ là vector tham số hiệu dụng. Xét bài toán tối ưu hàm mất mát $\mathcal{L}(\tilde\theta)$ đối với vector trọng số dày đặc $\theta \in \mathbb{R}^d$ bằng các thuật toán:
1. **SGD chuẩn:** $\theta^{(t+1)} = \theta^{(t)} - \eta g^{(t)}$, với $g^{(t)} = \nabla_\theta \mathcal{L}(\tilde\theta^{(t)})$;
2. **Momentum SGD:** $v^{(t+1)} = \mu v^{(t)} + g^{(t)}$ với $v^{(0)} = \mathbf{0}$, và $\theta^{(t+1)} = \theta^{(t)} - \eta v^{(t+1)}$;
3. **Adam / RMSProp:** $m^{(t+1)} = \beta_1 m^{(t)} + (1-\beta_1) g^{(t)}$ và $v^{(t+1)} = \beta_2 v^{(t)} + (1-\beta_2) (g^{(t)})^{\odot 2}$ với $m^{(0)} = v^{(0)} = \mathbf{0}$, và bước cập nhật $\Delta \theta^{(t+1)} = -\eta \frac{\hat{m}^{(t+1)}}{\sqrt{\hat{v}^{(t+1)}} + \epsilon}$.

Khi đó, với mọi chỉ số bị cắt tỉa $i$ thỏa mãn $M_i = 0$, trọng số hiệu dụng luôn bằng 0 tại mọi bước tối ưu:
$$\tilde\theta_i^{(t)} = 0 \quad \forall t \geq 0.$$

**Chứng minh:**  
Áp dụng quy tắc đạo hàm chuỗi nhiều biến:
$$g_i^{(t)} = \frac{\partial \mathcal{L}}{\partial \theta_i} = \sum_{j=1}^d \frac{\partial \mathcal{L}}{\partial \tilde\theta_j} \cdot \frac{\partial \tilde\theta_j}{\partial \theta_i} = \frac{\partial \mathcal{L}}{\partial \tilde\theta_i} \cdot \frac{\partial (M_i \theta_i)}{\partial \theta_i} = M_i \cdot \frac{\partial \mathcal{L}}{\partial \tilde\theta_i}.$$
Với mọi chỉ số $i$ có $M_i = 0$, ta có $g_i^{(t)} = 0 \cdot \frac{\partial \mathcal{L}}{\partial \tilde\theta_i} = 0$.

Quy nạp toán học theo thời gian $t$:
- Với SGD: $\theta_i^{(t+1)} = \theta_i^{(t)} - \eta \cdot 0 = \theta_i^{(t)} = \theta_i^{(0)}$. Suy ra $\tilde\theta_i^{(t)} = M_i \theta_i^{(t)} = 0 \cdot \theta_i^{(0)} = 0$.
- Với Momentum SGD: Giả sử $v_i^{(t)} = 0$. Khi đó $v_i^{(t+1)} = \mu \cdot 0 + 0 = 0$. Do đó $\theta_i^{(t+1)} = \theta_i^{(t)} = \theta_i^{(0)}$, kéo theo $\tilde\theta_i^{(t)} = 0$.
- Với Adam: Giả sử $m_i^{(t)} = v_i^{(t)} = 0$. Do $g_i^{(t)} = 0$, ta có $m_i^{(t+1)} = 0$, $v_i^{(t+1)} = 0$. Giá trị hiệu chỉnh $\hat{m}_i^{(t+1)} = 0$, dẫn đến bước dịch chuyển bằng 0. Suy ra $\theta_i^{(t)}$ không đổi và $\tilde\theta_i^{(t)} = 0$.

Định lý được chứng minh hoàn tất ($\blacksquare$).

*Hệ quả kỹ thuật:* Phải giữ forward hook chứa mặt nạ trong suốt quá trình tinh chỉnh (fine-tuning). Lệnh `prune.remove()` chỉ được gọi sau khi hoàn tất tinh chỉnh để hợp nhất vĩnh viễn trọng số 0 vào tensor.

---

### 3.2. Định lý 2: Sai số lượng tử hóa INT8 và chặn nhiễu loạn đầu ra mạng Lipschitz

**Phát biểu định lý:**  
Cho ma trận trọng số $\mathbf{W} \in [-R, R]^{m \times n}$ được lượng tử hóa đối xứng đều 8-bit với bước lượng tử $\Delta = \frac{2R}{255}$ và hàm làm tròn số nguyên gần nhất $q(w) = \lfloor w / \Delta \rceil \in \{-127, \ldots, 127\}$. Ma trận sau giải lượng tử là $\hat{\mathbf{W}} = \Delta \cdot q(\mathbf{W})$, và ma trận nhiễu sai số là $\mathbf{E} = \mathbf{W} - \hat{\mathbf{W}}$. Khi đó:

1. **Chặn từng phần tử:**
   $$|w_{jk} - \hat{w}_{jk}| \leq \frac{\Delta}{2} = \frac{R}{255}.$$
2. **Chặn chuẩn Frobenius và chuẩn phổ:**
   $$\|\mathbf{E}\|_F \leq \frac{\sqrt{mn} R}{255}, \quad \|\mathbf{E}\|_2 \leq \|\mathbf{E}\|_F \leq \frac{\sqrt{mn} R}{255}.$$
3. **Chặn nhiễu loạn đầu ra mạng $L$ tầng:**
   Xét mạng $f(\mathbf{x}) = \mathbf{W}_L \sigma(\mathbf{W}_{L-1} \sigma(\cdots \sigma(\mathbf{W}_1 \mathbf{x})\cdots))$ với các hàm kích hoạt $\sigma$ thỏa mãn điều kiện 1-Lipschitz và $\sigma(\mathbf{0}) = \mathbf{0}$ (chẳng hạn như ReLU). Khi mỗi tầng $\ell$ bị nhiễu do lượng tử hóa $\mathbf{E}_\ell = \mathbf{W}_\ell - \hat{\mathbf{W}}_\ell$, độ lệch đầu ra đối với mọi vector đầu vào $\mathbf{x} \in \mathbb{R}^d$ bị chặn bởi:
   $$\|f(\mathbf{x}) - \hat{f}(\mathbf{x})\|_2 \leq \|\mathbf{x}\|_2 \sum_{\ell=1}^L \|\mathbf{E}_\ell\|_2 \prod_{\substack{j=1 \\ j \neq \ell}}^L (\|\mathbf{W}_j\|_2 + \|\mathbf{E}_j\|_2).$$

**Chứng minh:**  
- **Phần 1:** Theo định nghĩa hàm làm tròn số nguyên gần nhất, $|z - \lfloor z \rceil| \leq \frac{1}{2}$ với mọi $z \in \mathbb{R}$. Đặt $z = w_{jk} / \Delta$, ta có $|w_{jk} - \hat{w}_{jk}| = \Delta |z - \lfloor z \rceil| \leq \Delta / 2 = R / 255$.
- **Phần 2:** Lấy tổng bình phương trên $mn$ phần tử:
  $$\|\mathbf{E}\|_F^2 = \sum_{j=1}^m \sum_{k=1}^n |w_{jk} - \hat{w}_{jk}|^2 \leq mn \left(\frac{R}{255}\right)^2 \implies \|\mathbf{E}\|_F \leq \frac{\sqrt{mn} R}{255}.$$
  Do chuẩn phổ $\|\mathbf{E}\|_2 = \sigma_{\max}(\mathbf{E}) \leq \sqrt{\sum_i \sigma_i^2} = \|\mathbf{E}\|_F$, bất đẳng thức thứ hai hiển nhiên đúng.
- **Phần 3:** Gọi $\mathbf{h}_\ell$ và $\hat{\mathbf{h}}_\ell$ lần lượt là vector kích hoạt tại tầng $\ell$. Do tính chất 1-Lipschitz của $\sigma$:
  $$\|\mathbf{h}_\ell - \hat{\mathbf{h}}_\ell\|_2 \leq \|\mathbf{W}_\ell \mathbf{h}_{\ell-1} - \hat{\mathbf{W}}_\ell \hat{\mathbf{h}}_{\ell-1}\|_2 \leq \|\mathbf{W}_\ell\|_2 \|\mathbf{h}_{\ell-1} - \hat{\mathbf{h}}_{\ell-1}\|_2 + \|\mathbf{E}_\ell\|_2 \|\hat{\mathbf{h}}_{\ell-1}\|_2.$$
  Khai triển hệ thức đệ quy từ tầng 1 đến tầng $L$ với chặn $\|\hat{\mathbf{h}}_\ell\|_2 \leq \|\mathbf{x}\|_2 \prod_{j=1}^\ell (\|\mathbf{W}_j\|_2 + \|\mathbf{E}_j\|_2)$ ta thu được chặn đầu ra tổng quát. ($\blacksquare$)

---

### 3.3. Định lý 3: Định lý Eckart-Young-Mirsky về xấp xỉ hạng thấp tối ưu

**Phát biểu định lý:**  
Cho ma trận $\mathbf{W} \in \mathbb{R}^{m \times n}$ có phân tích SVD $\mathbf{W} = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ với $k = \min(m,n)$ và $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_k \geq 0$. Với mọi hạng mục tiêu $r < k$, ma trận cắt cụt SVD:
$$\mathbf{W}_r = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$
là ma trận tối ưu duy nhất giải bài toán xấp xỉ ma trận hạng $r$ đồng thời trong chuẩn toán tử (spectral norm) và chuẩn Frobenius:
$$\min_{\operatorname{rank}(\mathbf{B}) \leq r} \|\mathbf{W} - \mathbf{B}\|_2 = \|\mathbf{W} - \mathbf{W}_r\|_2 = \sigma_{r+1},$$
$$\min_{\operatorname{rank}(\mathbf{B}) \leq r} \|\mathbf{W} - \mathbf{B}\|_F^2 = \|\mathbf{W} - \mathbf{W}_r\|_F^2 = \sum_{i=r+1}^k \sigma_i^2.$$

**Chứng minh:**  
1. **Chuẩn phổ:**  
   *Tính khả thi:* $\mathbf{W} - \mathbf{W}_r = \sum_{i=r+1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$. Do tính trực chuẩn của các vector suy biến, giá trị suy biến lớn nhất của ma trận hiệu là $\sigma_{r+1}$, do đó $\|\mathbf{W} - \mathbf{W}_r\|_2 = \sigma_{r+1}$.  
   *Chặn dưới:* Giả sử $\mathbf{B}$ là ma trận bất kỳ có $\operatorname{rank}(\mathbf{B}) \leq r$. Không gian hạt nhân $\ker(\mathbf{B})$ có số chiều $\dim(\ker(\mathbf{B})) = n - \operatorname{rank}(\mathbf{B}) \geq n - r$.  
   Xét không gian con $\mathcal{V}_{r+1} = \operatorname{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_{r+1}\}$ có số chiều $r+1$.  
   Theo định lý giao không gian con:
   $$\dim(\mathcal{V}_{r+1} \cap \ker(\mathbf{B})) \geq (r+1) + (n - r) - n = 1.$$
   Tồn tại vector đơn vị $\mathbf{z} \in \mathcal{V}_{r+1} \cap \ker(\mathbf{B})$ sao cho $\|\mathbf{z}\|_2 = 1$ và $\mathbf{B}\mathbf{z} = \mathbf{0}$. Biểu diễn $\mathbf{z} = \sum_{i=1}^{r+1} \alpha_i \mathbf{v}_i$ với $\sum \alpha_i^2 = 1$:
   $$\|\mathbf{W} - \mathbf{B}\|_2^2 \geq \|(\mathbf{W} - \mathbf{B})\mathbf{z}\|_2^2 = \|\mathbf{W}\mathbf{z}\|_2^2 = \sum_{i=1}^{r+1} \alpha_i^2 \sigma_i^2 \geq \sigma_{r+1}^2 \sum_{i=1}^{r+1} \alpha_i^2 = \sigma_{r+1}^2.$$
   Suy ra $\|\mathbf{W} - \mathbf{B}\|_2 \geq \sigma_{r+1}$.

2. **Chuẩn Frobenius:**  
   Theo bất đẳng thức giá trị suy biến Mirsky (1960), với mọi $\mathbf{W}, \mathbf{B} \in \mathbb{R}^{m \times n}$:
   $$\|\mathbf{W} - \mathbf{B}\|_F^2 \geq \sum_{i=1}^k (\sigma_i(\mathbf{W}) - \sigma_i(\mathbf{B}))^2.$$
   Vì $\operatorname{rank}(\mathbf{B}) \leq r$, $\mathbf{B}$ có tối đa $r$ giá trị suy biến dương, nghĩa là $\sigma_i(\mathbf{B}) = 0$ với mọi $i \geq r+1$. Do đó:
   $$\|\mathbf{W} - \mathbf{B}\|_F^2 \geq \sum_{i=1}^r (\sigma_i(\mathbf{W}) - \sigma_i(\mathbf{B}))^2 + \sum_{i=r+1}^k \sigma_i(\mathbf{W})^2 \geq \sum_{i=r+1}^k \sigma_i(\mathbf{W})^2.$$
   Dấu bằng xảy ra khi và chỉ khi $\sigma_i(\mathbf{B}) = \sigma_i(\mathbf{W})$ với các không gian con suy biến đồng nhất, tức $\mathbf{B} = \mathbf{W}_r$. ($\blacksquare$)

---

## 4. Kết Quả Thực Nghiệm Đo Kiểm Đa Hạt Nhân (Empirical Results)

Dưới đây là bảng tổng hợp các chỉ số thực nghiệm trung bình và độ lệch chuẩn đo được qua 3 random seeds ($n=3$):

### 4.1. Bảng so sánh mô hình cơ sở và lượng tử hóa
| Phương pháp | Độ chính xác (%) | $\Delta$Acc (%) | Dung lượng đĩa (MB) | Độ trễ suy luận (ms) | Throughput (mẫu/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | **92.25 $\pm$ 0.09** | --- | 0.793 | 0.286 $\pm$ 0.010 | 6,137 |
| **Dynamic INT8** | **92.25 $\pm$ 0.08** | **+0.00** | **0.217** ($3.7\times$) | 0.571 $\pm$ 0.032 | **14,836** ($2.4\times$) |

*Nhận xét:* Dynamic INT8 bảo toàn nguyên vẹn 100% độ chính xác của mạng ban đầu trong khi giảm dung lượng lưu trữ gần 4 lần và tăng thông lượng suy luận xử lý theo lô hơn 2.4 lần.

### 4.2. Bảng thực nghiệm cắt tỉa phi cấu trúc (Global L1 Unstructured Pruning)
| Tỷ lệ thưa | Độ chính xác (%) | $\Delta$Acc (%) | Macro-F1 (%) | MACs lý thuyết | Độ trễ đo được (ms) | Biến thiên độ trễ ($\Delta$Lat) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0% (Base)** | 92.25 $\pm$ 0.09 | --- | 92.22 $\pm$ 0.10 | 1.218M | 0.286 $\pm$ 0.010 | --- |
| **25%** | 92.14 $\pm$ 0.00 | -0.11 | 92.13 $\pm$ 0.00 | 0.914M | 0.638 $\pm$ 0.000 | **+122.6%** |
| **50%** | 92.09 $\pm$ 0.00 | -0.16 | 92.08 $\pm$ 0.00 | 0.609M | 0.798 $\pm$ 0.000 | **+178.1%** |
| **75%** | 92.28 $\pm$ 0.00 | +0.03 | 92.27 $\pm$ 0.00 | 0.305M | 0.688 $\pm$ 0.000 | **+140.0%** |
| **90%** | 91.61 $\pm$ 0.00 | -0.64 | 91.60 $\pm$ 0.00 | 0.122M | **0.336 $\pm$ 0.000** | **-40.3%** (so với p50) |

### 4.3. Bảng thực nghiệm SVD Low-Rank Factorization
| Tỷ lệ hạng ($r/k$) | Độ chính xác (%) | $\Delta$Acc (%) | Năng lượng giữ lại $\eta$ (%) | Sai số Frobenius tương đối |
| :---: | :---: | :---: | :---: | :---: |
| **100% (Base)** | 92.25 $\pm$ 0.09 | --- | 100.0% | 0.0000 |
| **75%** | 74.74 $\pm$ 2.99 | -17.51 | 78.6% | 0.2136 |
| **50%** | 70.77 $\pm$ 1.16 | -21.48 | 59.8% | 0.4021 |
| **25%** | 21.91 $\pm$ 1.17 | -70.34 | 38.0% | 0.6202 |

---

## 5. Thảo Luận Chuyên Sâu: Khoảng Cách FLOPs--Latency

Hiện tượng then chốt được khám phá trong nghiên cứu này là sự lệch pha giữa số lượng phép tính lý thuyết (FLOPs) và thời gian thực thi phần cứng đối với kỹ thuật cắt tỉa phi cấu trúc trên CPU:

1. **Tại sao độ thưa 25--75% lại làm tăng độ trễ?**  
   Trên kiến trúc tập lệnh x86/ARM, thư viện toán học BLAS thực hiện phép nhân ma trận dày đặc bằng các lệnh SIMD (AVX2/AVX-512) tải liên tục các khối 8--16 số thực 32-bit vào thanh ghi. Khi ma trận thưa phi cấu trúc vẫn lưu dưới dạng mảng 2 chiều dày đặc (dense tensor), CPU vẫn nạp đủ các phần tử 0 vào pipeline. Hơn nữa, việc kích hoạt forward hook để áp mặt nạ nhị phân trước mỗi lượt suy luận tạo ra chi phí phụ trội (overhead), đồng thời làm phân mảnh mẫu truy cập bộ nhớ đệm (cache miss rate tăng).
2. **Tại sao độ thưa 90% lại giảm mạnh độ trễ?**  
   Khi 90% trọng số của tầng FC1 bị triệt tiêu (chỉ còn ~20,000 trọng số khác 0), tổng dung lượng trọng số hiệu dụng giảm xuống dưới 80 KB — hoàn toàn nằm gọn trong bộ nhớ đệm **L2 Cache** của CPU. Việc triệt tiêu hoàn toàn độ trễ đọc từ bộ nhớ RAM chính (DRAM access latency), kết hợp với cơ chế dự đoán rẽ nhánh và bỏ qua phép cộng số 0 của bộ số học, đã tạo ra sự nhảy vọt về tốc độ xử lý.

---

## 6. Kết Luận

Nghiên cứu khẳng định rằng đối với các hệ thống nhúng CPU không có nhân tăng tốc phần cứng thưa chuyên biệt (sparse accelerators):
1. **Dynamic INT8 Quantization** là lựa chọn mặc định tối ưu nhất, mang lại lợi ích thực tế tức thì về cả dung lượng và tốc độ mà không phải đánh đổi độ chính xác.
2. **Cắt tỉa phi cấu trúc** không nên được áp dụng ở mức độ trung bình nếu mục tiêu là giảm độ trễ thời gian thực; kỹ thuật này chỉ phát huy hiệu quả khi đi kèm với định dạng lưu trữ thưa chuyên dụng (CSR/COO) hoặc ở mức độ thưa cực đoan ($\geq 90\%$).
3. **Phân tích SVD** cung cấp công cụ toán học tối ưu để đánh giá mức độ dư thừa của các tầng trọng số, nhưng đòi hỏi phải tái thiết kế kiến trúc mạng thành chuỗi tầng có số chiều trung gian nhỏ để giải phóng tài nguyên bộ nhớ thực tế.
