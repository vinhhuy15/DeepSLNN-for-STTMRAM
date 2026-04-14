"""
Simulate_figure7_combined.py 
So sánh BER/FER: Raw Data vs Euclidean Decoder vs Detector vs Deep SLNN
Mô phỏng bám sát Figure 7 của bài báo gốc.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from config import (
    K, N, P1, ALPHA, MIN_ERRORS, BATCH,
    codebook_7b9b,
    MRAM_channel_batch, encode_batch,
    decode_euclidean_batch, decode_detector_batch, simulate_raw_batch,
    load_checkpoint, save_checkpoint
)

# Nhập thêm module SLNN
from slnn_decoder import load_slnn, decode_slnn_batch

# ====================== THAM SỐ FIGURE 7 ======================
OFFSET_MU          = 0.0
OFFSET_SIGMA_RATIO = 0.0
sigma_mu_list      = np.linspace(0.08, 0.15, 8)   # 8% -> 15%, 8 điểm

# Tên file checkpoint mới để chứa thêm data của SLNN
CKPT_FILE = "figure7_combined_checkpoint.npy"
SLNN_MODEL_PATH = "deep_slnn_model.pt"

# ====================== HEADER ======================
print("=" * 70)
print("  FIGURE 7 (COMBINED) — BER & FER vs sigma_0/mu_0 (8% -> 15%)")
print(f"  P1={P1:.0e}, alpha={ALPHA}")
print(f"  offset_mu={OFFSET_MU}, offset_sigma_ratio={OFFSET_SIGMA_RATIO}")
print(f"  Dừng khi >= {MIN_ERRORS} bit lỗi mỗi điểm | BATCH={BATCH} frames")
print("=" * 70)

# ====================== LOAD SLNN MODEL ======================
if not os.path.exists(SLNN_MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy mô hình SLNN tại {SLNN_MODEL_PATH}. Vui lòng train mô hình trước!")
print(f"[INFO] Đang tải mô hình Deep SLNN từ {SLNN_MODEL_PATH}...")
slnn_model = load_slnn(SLNN_MODEL_PATH)

# ====================== LOAD CHECKPOINT ======================
ckpt = load_checkpoint(CKPT_FILE)
if ckpt is None:
    ckpt = {
        "BER_decoder": [], "FER_decoder": [],
        "BER_detector": [], "FER_detector": [],
        "BER_raw": [], "FER_raw": [],
        "BER_slnn": [], "FER_slnn": [],  # Thêm track cho SLNN
        "done_indices": []
    }
    print("  Bắt đầu mới (không có checkpoint).")
else:
    print(f"  Tiếp tục từ checkpoint — đã xong {len(ckpt['done_indices'])} điểm.")

# ====================== VÒNG LẶP CHÍNH ======================
for idx, sigma_mu in enumerate(sigma_mu_list):
    if idx in ckpt["done_indices"]:
        print(f"  sigma_mu = {sigma_mu*100:.0f}% — đã có trong checkpoint, bỏ qua.")
        continue

    pct = sigma_mu * 100
    dec_bit=0; dec_frm=0; dec_bits_tot=0; dec_frm_tot=0
    det_bit=0; det_frm=0; det_bits_tot=0; det_frm_tot=0
    raw_bit=0; raw_frm=0; raw_bits_tot=0; raw_frm_tot=0
    sln_bit=0; sln_frm=0; sln_bits_tot=0; sln_frm_tot=0

    print(f"\n  sigma_mu = {pct:.0f}% ...", end=' ', flush=True)

    # Vòng lặp dừng khi TẤT CẢ các bộ giải mã đều đạt đủ số lỗi tối thiểu
    while (dec_bit < MIN_ERRORS or det_bit < MIN_ERRORS or 
           raw_bit < MIN_ERRORS or sln_bit < MIN_ERRORS):

        in_bits = np.random.randint(0, 2, size=(BATCH, K), dtype=int)
        out_enc = encode_batch(in_bits)

        received_coded = MRAM_channel_batch(
            out_enc, sigma_mu, P1,
            offset_mu=OFFSET_MU, offset_sigma_ratio=OFFSET_SIGMA_RATIO
        )

        # [1] Proposed code: Euclidean decoder
        if dec_bit < MIN_ERRORS:
            be, fe = decode_euclidean_batch(received_coded / ALPHA, in_bits)
            dec_bit += be; dec_frm += fe
            dec_bits_tot += BATCH * K; dec_frm_tot += BATCH

        # [2] Proposed code: detector output
        if det_bit < MIN_ERRORS:
            be, fe = decode_detector_batch(received_coded, in_bits)
            det_bit += be; det_frm += fe
            det_bits_tot += BATCH * K; det_frm_tot += BATCH

        # [3] Raw data: không encode, threshold only
        if raw_bit < MIN_ERRORS:
            raw_in = np.random.randint(0, 2, size=(BATCH, K), dtype=int)
            be, fe = simulate_raw_batch(
                raw_in, sigma_mu, P1,
                offset_mu=OFFSET_MU, offset_sigma_ratio=OFFSET_SIGMA_RATIO
            )
            raw_bit += be; raw_frm += fe
            raw_bits_tot += BATCH * K; raw_frm_tot += BATCH
            
        # [4] Deep SLNN Decoder (MỚI THÊM)
        if sln_bit < MIN_ERRORS:
            # Lưu ý: SLNN tự normalize bên trong nên truyền thẳng received_coded
            be, fe = decode_slnn_batch(slnn_model, received_coded, in_bits)
            sln_bit += be; sln_frm += fe
            sln_bits_tot += BATCH * K; sln_frm_tot += BATCH

    # Tính toán tỉ lệ
    ber_dec = dec_bit / dec_bits_tot
    fer_dec = dec_frm / dec_frm_tot
    ber_det = det_bit / det_bits_tot
    fer_det = det_frm / det_frm_tot
    ber_raw = raw_bit / raw_bits_tot
    fer_raw = raw_frm / raw_frm_tot
    ber_sln = sln_bit / sln_bits_tot
    fer_sln = sln_frm / sln_frm_tot

    # Cập nhật checkpoint
    ckpt["BER_decoder"].append(ber_dec);  ckpt["FER_decoder"].append(fer_dec)
    ckpt["BER_detector"].append(ber_det); ckpt["FER_detector"].append(fer_det)
    ckpt["BER_raw"].append(ber_raw);      ckpt["FER_raw"].append(fer_raw)
    ckpt["BER_slnn"].append(ber_sln);     ckpt["FER_slnn"].append(fer_sln)
    ckpt["done_indices"].append(idx)
    
    save_checkpoint(CKPT_FILE, ckpt)

    print(f"\n      BER_raw={ber_raw:.2e} | BER_det={ber_det:.2e}")
    print(f"      BER_dec={ber_dec:.2e} | BER_sln={ber_sln:.2e}")
    print(f"      Frames ~ {max(dec_frm_tot, raw_frm_tot, sln_frm_tot):,}", flush=True)

# ====================== VẼ FIGURE 7 KẾT HỢP ======================
x = sigma_mu_list * 100

BER_decoder  = ckpt["BER_decoder"]
FER_decoder  = ckpt["FER_decoder"]
BER_detector = ckpt["BER_detector"]
FER_detector = ckpt["FER_detector"]
BER_raw      = ckpt["BER_raw"]
FER_raw      = ckpt["FER_raw"]
BER_slnn     = ckpt["BER_slnn"]
FER_slnn     = ckpt["FER_slnn"]

fig, ax = plt.subplots(figsize=(10, 7))

# --- Trục BER (nét liền) ---
ax.semilogy(x, BER_raw,      'r-*',  label='BER - Raw data (Detector)', markersize=9, linewidth=1.8)
ax.semilogy(x, BER_detector, 'k-o',  label='BER - Proposed (Detector)', markersize=7, linewidth=1.8)
ax.semilogy(x, BER_decoder,  'k-^',  label='BER - Proposed (Euclidean)', markersize=7, linewidth=1.8)
# Đường Deep SLNN mới
ax.semilogy(x, BER_slnn,     'b-s',  label='BER - Proposed (Deep SLNN)', markersize=7, linewidth=2.0)

# --- Trục FER (nét đứt) ---
ax.semilogy(x, FER_raw,      'r--*', label='FER - Raw data (Detector)', markersize=9, linewidth=1.5)
ax.semilogy(x, FER_detector, 'k--o', label='FER - Proposed (Detector)', markersize=7, linewidth=1.5)
ax.semilogy(x, FER_decoder,  'k--^', label='FER - Proposed (Euclidean)', markersize=7, linewidth=1.5)
ax.semilogy(x, FER_slnn,     'b--s', label='FER - Proposed (Deep SLNN)', markersize=7, linewidth=1.5)

# Cấu hình đồ thị
ax.set_xlabel(r'Tỷ lệ nhiễu đọc $\sigma_0/\mu_0$ (%)', fontsize=13)
ax.set_ylabel('BER & FER', fontsize=13)
ax.set_title(r'Figure 7: So sánh Raw data, Euclidean Decoder và Deep SLNN Decoder' '\n'
             r'$P_1=2\times10^{-4}$, no offset, $\alpha=2.5$', fontsize=12, fontweight='bold')

# Chia legend thành 2 cột cho gọn
ax.legend(fontsize=9, loc='upper left', ncol=2)
ax.set_xlim([8, 15])
ax.set_ylim([1e-4, 1e0])
ax.set_xticks(range(8, 16))

# Bật lưới log chuẩn
ax.grid(True, which='major', linestyle='-', alpha=0.6)
ax.grid(True, which='minor', linestyle='--', alpha=0.3)

plt.tight_layout()
output_image = 'figure7_combined.png'
plt.savefig(output_image, dpi=300)
print(f"\n[HOÀN TẤT] Đã lưu đồ thị tổng hợp tại: {output_image}")
plt.show()