import numpy as np
import bin2dec_dec2bin as sp
import decoder_euclidien as de


# ====================== MRAM CHANNEL (VECTORIZED) ======================
def MRAM_channel(data, sigma0_ratio, P1, offset_mu=0.0, offset_sigma_ratio=0.0):
    """
    Mô phỏng kênh STT-MRAM (cascaded channel model) — vectorized.

    Tham số cố định (hard-coded):
        mu_0, mu_1     : điện trở trung bình mức 0 và mức 1 (1.0 và 2.0 kOhm)
        P0 = P1 / 100  : lỗi ghi 1->0, thấp hơn P1 đúng 2 bậc
        Pr = P1 / 100  : nhiễu đọc, thấp hơn P1 đúng 2 bậc

    Tham số điều chỉnh (truyền vào):
        data               : mảng numpy 1D hoặc 2D gồm các bit 0/1
        sigma0_ratio       : σ0/μ0 = σ1/μ1 (ví dụ: 0.09 tương ứng 9%)
        P1                 : xác suất lỗi ghi 0->1
        offset_mu          : μofs — độ lệch điện trở trung bình do nhiệt độ (mặc định 0)
        offset_sigma_ratio : σofs/μ1 — độ dao động của offset (mặc định 0)

    Trả về:
        output : mảng giá trị thực sau kênh Gaussian (cùng shape với data)
    """
    mu_0, mu_1 = 1.0, 2.0
    P0 = P1 / 100
    Pr = P1 / 100

    # Xác suất kênh BAC + Z kết hợp (write-0 direction)
    p0 = (P0 / 2) * (1 - Pr)          # P(0→1 lỗi)
    p1 = P1 / 2 + (1 - P1 / 2) * Pr   # P(1→1 đúng) bù lại

    sigma_0   = mu_0 * sigma0_ratio
    sigma_1   = mu_1 * sigma0_ratio
    sigma_ofs = offset_sigma_ratio * mu_1

    shape = data.shape

    # ---- Bước 1: Kênh BAC (vectorized) ----
    rand_vals = np.random.rand(*shape)
    temp = np.where(
        data == 0,
        (rand_vals <= p0).astype(int),          # 0→1 lỗi với xác suất p0
        (rand_vals <= (1 - p1)).astype(int)     # 1 giữ nguyên với xác suất (1-p1)
    )

    # ---- Bước 2: Kênh Gaussian (vectorized) ----
    noise = np.random.randn(*shape)
    if sigma_ofs > 0:
        ofs = offset_mu + sigma_ofs * np.random.randn(*shape)
    else:
        ofs = offset_mu

    output = np.where(
        temp == 0,
        mu_0 + sigma_0 * noise,
        mu_1 + ofs + sigma_1 * noise
    )

    return output


# ====================== LOAD CODEBOOK 7b9b ======================
codebook_7b9b = np.loadtxt("output.txt", dtype=int)   # shape (128, 9)
assert codebook_7b9b.shape == (128, 9), \
    f"Codebook sai shape: {codebook_7b9b.shape}, cần (128, 9)"

# ====================== THAM SỐ HỆ THỐNG ======================
k        = 7        # số bit thông tin
n        = 9        # số bit mã hóa
blk      = 10000    # số khung
sigma_mu = 0.08     # sigma0_ratio = 8%
P1       = 2e-4     # xác suất lỗi ghi 0->1
alpha    = 2.5      # hệ số attenuator tối ưu

# ====================== TẠO DỮ LIỆU GỐC NGẪU NHIÊN ======================
print("Đang tạo dữ liệu gốc 7-bit...")
in_bits = np.random.randint(0, 2, size=(blk, k), dtype=int)

# ====================== MÃ HÓA 7b9b (VECTORIZED) ======================
print("Đang mã hóa 7b9b...")
indices = np.array([int(sp.bin2dec_ndc(in_bits[i])) for i in range(blk)])
out_enc = codebook_7b9b[indices]   # (blk, n) — index trực tiếp vào codebook
print("Mã hóa hoàn tất!")

# ====================== QUA KÊNH STT-MRAM (VECTORIZED) ======================
print(f"Đang truyền qua kênh STT-MRAM (sigma_mu={sigma_mu}, P1={P1})...")
received = MRAM_channel(out_enc, sigma_mu, P1)   # toàn bộ (blk, n) một lần

# ====================== ATTENUATOR + GIẢI MÃ ======================
received_att = received / alpha
print("Đang giải mã bằng Minimum Euclidean Distance...")
total_bit_errors, total_frame_errors = de.decode_euclidean(
    received_att, codebook_7b9b, k, in_bits
)

# ====================== KẾT QUẢ ======================
BER = total_bit_errors   / (blk * k)
FER = total_frame_errors / blk

print("\n" + "=" * 55)
print("           KẾT QUẢ GIẢI MÃ 7b9b QUA KÊNH STT-MRAM")
print("=" * 55)
print(f"Số khung (blk)             : {blk:,}")
print(f"Tổng số bit thông tin      : {blk * k:,}")
print(f"Tổng số bit mã hóa         : {blk * n:,}")
print(f"sigma0_ratio (σ0/μ0)       : {sigma_mu:.4f}  ({sigma_mu*100:.1f}%)")
print(f"P1 (xác suất lỗi ghi 0->1) : {P1:.2e}")
print(f"Attenuator alpha           : {alpha}")
print(f"Bit Error Rate  (BER)      : {BER:.2e}")
print(f"Frame Error Rate (FER)     : {FER:.2e}")
print("=" * 55)
