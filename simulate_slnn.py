"""
simulate_slnn.py — So sánh BER/FER: Euclidean decoder vs Deep SLNN decoder
"""

import numpy as np
import os

from config import (
    K, N, P1, ALPHA, MU_1,
    MIN_ERRORS, BATCH,
    codebook_7b9b,
    MRAM_channel_batch,
    encode_batch,
    decode_euclidean_batch,
)

from slnn_decoder import train_slnn, decode_slnn_batch, save_slnn, load_slnn

# ═══════════════════════════════════════════════════════════════════════════
# THAM SỐ MÔ PHỎNG
# ═══════════════════════════════════════════════════════════════════════════

SIGMA_LIST = [round(x * 0.01, 2) for x in range(8, 16)]
SIGMA_TRAIN = SIGMA_LIST  

# Cấu hình Deep SLNN
SLNN_NR_TRAIN   = 1_500_000   # Tăng dữ liệu do mạng sâu hơn
SLNN_H1         = 128         # Layer ẩn 1
SLNN_H2         = 64          # Layer ẩn 2
SLNN_EPOCHS     = 30
SLNN_BATCH      = 256
SLNN_LR         = 0.01
SLNN_PATIENCE   = 5

MODEL_SAVE_PATH = "deep_slnn_model.pt"

MIN_ERRORS_SIM  = MIN_ERRORS  
BATCH_SIM       = BATCH        
MAX_FRAMES      = 5_000_000    

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 1: TRAIN
# ═══════════════════════════════════════════════════════════════════════════
def get_slnn_model(force_retrain: bool = False):
    if not force_retrain and os.path.exists(MODEL_SAVE_PATH):
        print(f"\n[INFO] Load model cũ từ {MODEL_SAVE_PATH}")
        return load_slnn(MODEL_SAVE_PATH)

    print("\n[INFO] Bắt đầu train Deep SLNN decoder...")
    model = train_slnn(
        codebook    = codebook_7b9b,
        sigma_mu    = SIGMA_TRAIN,
        P1          = P1,
        nr_train    = SLNN_NR_TRAIN,
        h1          = SLNN_H1,
        h2          = SLNN_H2,
        epochs      = SLNN_EPOCHS,
        batch_size  = SLNN_BATCH,
        lr          = SLNN_LR,
        patience    = SLNN_PATIENCE,
        seed        = 42,
        verbose     = True,
    )
    save_slnn(model, MODEL_SAVE_PATH)
    return model

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 2: MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════
def simulate_one_sigma(slnn_model, sigma: float):
    euc_bit_err = euc_frm_err = 0
    sln_bit_err = sln_frm_err = 0
    total_frames = 0

    while (euc_bit_err < MIN_ERRORS_SIM or sln_bit_err < MIN_ERRORS_SIM) \
            and total_frames < MAX_FRAMES:

        in_bits = np.random.randint(0, 2, size=(BATCH_SIM, K), dtype=int)
        out_enc = encode_batch(in_bits)                             

        received = MRAM_channel_batch(out_enc, sigma, P1)          

        # Euclidean 
        received_att = received / ALPHA
        be, fe = decode_euclidean_batch(received_att, in_bits)
        euc_bit_err += be
        euc_frm_err += fe

        # Deep SLNN 
        be2, fe2 = decode_slnn_batch(slnn_model, received, in_bits)
        sln_bit_err += be2
        sln_frm_err += fe2

        total_frames += BATCH_SIM

    total_bits = total_frames * K
    return {
        'sigma'       : sigma,
        'total_frames': total_frames,
        'euc_BER': euc_bit_err / total_bits  if total_bits > 0 else 0,
        'euc_FER': euc_frm_err / total_frames if total_frames > 0 else 0,
        'sln_BER': sln_bit_err / total_bits  if total_bits > 0 else 0,
        'sln_FER': sln_frm_err / total_frames if total_frames > 0 else 0,
    }

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 3: IN KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════════════
def run_simulation(slnn_model):
    print(f"\n{'='*72}")
    print(f"   SO SÁNH BER: Euclidean Decoder vs Deep SLNN Decoder")
    print(f"{'='*72}")
    print(f"  {'σ/μ':>5} | {'Euc BER':>12} | {'SLNN BER':>12} | {'ΔBER':>8} | Frames")
    print(f"  {'-'*72}")

    results = []
    for sigma in SIGMA_LIST:
        r = simulate_one_sigma(slnn_model, sigma)
        if r['euc_BER'] > 0:
            delta = (r['euc_BER'] - r['sln_BER']) / r['euc_BER'] * 100
            arrow = "↓" if delta > 0 else "↑"
            delta_str = f"{arrow}{abs(delta):.1f}%"
        else:
            delta_str = "   N/A"

        print(f"  {sigma*100:4.0f}% | {r['euc_BER']:12.3e} | {r['sln_BER']:12.3e} | {delta_str:>8} | {r['total_frames']:,}")
        results.append(r)
    return results

def save_results_csv(results, path: str = "BER_Euclidean_vs_Deep_SLNN.csv"):
    rows = [[r['sigma'] * 100, r['euc_BER'], r['euc_FER'], r['sln_BER'], r['sln_FER'], r['total_frames']] for r in results]
    np.savetxt(path, np.array(rows), delimiter=",", header="sigma_pct,euc_BER,euc_FER,slnn_BER,slnn_FER,total_frames", comments="", fmt=["%.1f", "%.6e", "%.6e", "%.6e", "%.6e", "%d"])
    print(f"\n[INFO] Đã lưu CSV → {path}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Đặt force_retrain=True để bắt mô hình học lại với dữ liệu và kiến trúc mới
    slnn_model = get_slnn_model(force_retrain=True)
    results = run_simulation(slnn_model)
    save_results_csv(results)

    print(f"\n{'='*72}")
    print("  TÓM TẮT")
    euc_bers = [r['euc_BER'] for r in results]
    sln_bers = [r['sln_BER'] for r in results]
    gains = [(e - s) / e * 100 for e, s in zip(euc_bers, sln_bers) if e > 0]
    if gains:
        print(f"  Cải thiện BER trung bình : {np.mean(gains):.1f}%")
        print(f"  Cải thiện BER tốt nhất  : {max(gains):.1f}%  (tại σ={SIGMA_LIST[gains.index(max(gains))]*100:.0f}%)")
    print(f"{'='*72}")