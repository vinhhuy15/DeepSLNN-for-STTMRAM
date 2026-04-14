"""
config.py — Tham số hệ thống và hàm tiện ích dùng chung cho tất cả Simulate_figure*.py
"""
import numpy as np
import os
import bin2dec_dec2bin as sp

# ====================== THAM SỐ HỆ THỐNG ======================
K         = 7        # số bit thông tin
N         = 9        # số bit mã hóa
P1        = 2e-4     # xác suất lỗi ghi 0->1
ALPHA     = 2.5      # attenuator tối ưu
MU_0      = 1.0
MU_1      = 2.0
THRESHOLD = (MU_0 + MU_1) / 2   # = 1.5

MIN_ERRORS = 500     # số lỗi bit tối thiểu mỗi điểm (Monte Carlo)
BATCH      = 5000    # số frame mỗi vòng lặp

# ====================== LOAD CODEBOOK ======================
codebook_7b9b = np.loadtxt("output.txt", dtype=int)
assert codebook_7b9b.shape == (128, 9), \
    f"Codebook sai shape: {codebook_7b9b.shape}, cần (128, 9)"

# Bảng tra decode: index → mảng bit k-bit
_decode_table = np.array(
    [[int(b) for b in format(i, 'b').zfill(K)] for i in range(2 ** K)],
    dtype=int
)  # (128, K)


# ====================== MRAM CHANNEL (VECTORIZED, BATCH) ======================
def MRAM_channel_batch(data_batch, sigma0_ratio, P1,
                       offset_mu=0.0, offset_sigma_ratio=0.0):
    """
    Truyền toàn bộ batch (blk, n) qua kênh STT-MRAM một lần.
    Nhanh hơn gọi từng frame nhờ NumPy vectorization.

    Tham số:
        data_batch         : (blk, bits) — mảng bit 0/1
        sigma0_ratio       : σ0/μ0
        P1                 : xác suất lỗi ghi 0→1
        offset_mu          : μofs (mặc định 0)
        offset_sigma_ratio : σofs/μ1 (mặc định 0)

    Trả về:
        output : (blk, bits) — giá trị thực sau kênh
    """
    P0 = P1 / 100
    Pr = P1 / 100
    p0 = (P0 / 2) * (1 - Pr)
    p1 = P1 / 2 + (1 - P1 / 2) * Pr

    sigma_0   = MU_0 * sigma0_ratio
    sigma_1   = MU_1 * sigma0_ratio
    sigma_ofs = offset_sigma_ratio * MU_1

    shape = data_batch.shape

    # BAC vectorized
    rand_vals = np.random.rand(*shape)
    temp = np.where(
        data_batch == 0,
        (rand_vals <= p0).astype(int),
        (rand_vals <= (1 - p1)).astype(int)
    )

    # Gaussian vectorized
    noise = np.random.randn(*shape)
    if sigma_ofs > 0:
        ofs = offset_mu + sigma_ofs * np.random.randn(*shape)
    else:
        ofs = float(offset_mu)

    output = np.where(
        temp == 0,
        MU_0 + sigma_0 * noise,
        MU_1 + ofs + sigma_1 * noise
    )
    return output


# ====================== ENCODE BATCH ======================
def encode_batch(in_bits):
    """
    Mã hóa batch (blk, k) → (blk, n) bằng codebook LUT.
    """
    indices = np.array([int(sp.bin2dec_ndc(row)) for row in in_bits])
    return codebook_7b9b[indices]


# ====================== DECODE: EUCLIDEAN (VECTORIZED) ======================
def decode_euclidean_batch(received_att, in_bits):
    """
    Giải mã batch bằng ML / min Euclidean distance.
    received_att : (blk, n)
    in_bits      : (blk, k)
    Trả về: (total_bit_errors, total_frame_errors)
    """
    dists   = np.sum((received_att[:, np.newaxis, :] -
                      codebook_7b9b[np.newaxis, :, :]) ** 2, axis=2)
    best    = np.argmin(dists, axis=1)
    out_dec = _decode_table[best]
    errs    = np.sum(out_dec != in_bits, axis=1)
    return int(np.sum(errs)), int(np.sum(errs > 0))


# ====================== DECODE: DETECTOR (THRESHOLD) ======================
def decode_detector_batch(received_coded, in_bits):
    """
    Threshold detection → Euclidean decode trên tín hiệu đã detect.
    received_coded : (blk, n) — tín hiệu thực chưa /alpha
    in_bits        : (blk, k)
    Trả về: (total_bit_errors, total_frame_errors)
    """
    detected = (received_coded >= THRESHOLD).astype(float)  # (blk, n)
    dists    = np.sum((detected[:, np.newaxis, :] -
                       codebook_7b9b[np.newaxis, :, :]) ** 2, axis=2)
    best     = np.argmin(dists, axis=1)
    out_dec  = _decode_table[best]
    errs     = np.sum(out_dec != in_bits, axis=1)
    return int(np.sum(errs)), int(np.sum(errs > 0))


# ====================== RAW DATA (THRESHOLD ONLY) ======================
def simulate_raw_batch(in_bits, sigma0_ratio, P1,
                       offset_mu=0.0, offset_sigma_ratio=0.0):
    """
    Truyền raw in_bits (k bit, không encode) qua kênh, threshold detect.
    Trả về: (total_bit_errors, total_frame_errors)
    """
    rx       = MRAM_channel_batch(in_bits, sigma0_ratio, P1,
                                  offset_mu, offset_sigma_ratio)
    detected = (rx >= THRESHOLD).astype(int)
    errs     = np.sum(detected != in_bits, axis=1)
    return int(np.sum(errs)), int(np.sum(errs > 0))


# ====================== RESUME HELPER ======================
def load_checkpoint(path):
    """Load kết quả đã lưu (npy dict). Trả về dict hoặc None."""
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True).item())
    return None

def save_checkpoint(path, data):
    """Lưu kết quả vào file .npy để resume sau."""
    np.save(path, data)
