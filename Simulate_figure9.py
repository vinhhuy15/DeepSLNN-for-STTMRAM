import numpy as np
import matplotlib.pyplot as plt
from config import (
    K, N, P1, ALPHA, MIN_ERRORS, BATCH,
    codebook_7b9b,
    MRAM_channel_batch, encode_batch,
    decode_euclidean_batch, decode_detector_batch, simulate_raw_batch,
    load_checkpoint, save_checkpoint
)

# ====================== THAM SỐ FIGURE 9 ======================
OFFSET_MU          = -0.2    # μofs = -0.2 kΩ
OFFSET_SIGMA_RATIO = 0.07    # σofs/μ1 = 7%
sigma_mu_list      = np.linspace(0.02, 0.10, 9)   # 2% -> 10%, 9 điểm

CKPT_FILE = "figure9_checkpoint.npy"

# ====================== HEADER ======================
print("=" * 65)
print("  FIGURE 9 — BER & FER vs sigma_0/mu_0 (2% -> 10%)")
print(f"  P1={P1:.0e}, alpha={ALPHA}")
print(f"  offset_mu={OFFSET_MU} kOhm, offset_sigma_ratio={OFFSET_SIGMA_RATIO*100:.0f}%")
print(f"  Dừng khi >= {MIN_ERRORS} bit lỗi mỗi điểm | BATCH={BATCH} frames")
print("=" * 65)

# ====================== LOAD CHECKPOINT ======================
ckpt = load_checkpoint(CKPT_FILE)
if ckpt is None:
    ckpt = {
        "BER_decoder": [], "FER_decoder": [],
        "BER_detector": [], "FER_detector": [],
        "BER_raw": [], "FER_raw": [],
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

    print(f"\n  sigma_mu = {pct:.0f}% ...", end=' ', flush=True)

    while (dec_bit < MIN_ERRORS or det_bit < MIN_ERRORS or raw_bit < MIN_ERRORS):

        in_bits = np.random.randint(0, 2, size=(BATCH, K), dtype=int)
        out_enc = encode_batch(in_bits)

        received_coded = MRAM_channel_batch(
            out_enc, sigma_mu, P1,
            offset_mu=OFFSET_MU, offset_sigma_ratio=OFFSET_SIGMA_RATIO
        )

        if dec_bit < MIN_ERRORS:
            be, fe = decode_euclidean_batch(received_coded / ALPHA, in_bits)
            dec_bit += be; dec_frm += fe
            dec_bits_tot += BATCH * K; dec_frm_tot += BATCH

        if det_bit < MIN_ERRORS:
            be, fe = decode_detector_batch(received_coded, in_bits)
            det_bit += be; det_frm += fe
            det_bits_tot += BATCH * K; det_frm_tot += BATCH

        if raw_bit < MIN_ERRORS:
            raw_in = np.random.randint(0, 2, size=(BATCH, K), dtype=int)
            be, fe = simulate_raw_batch(
                raw_in, sigma_mu, P1,
                offset_mu=OFFSET_MU, offset_sigma_ratio=OFFSET_SIGMA_RATIO
            )
            raw_bit += be; raw_frm += fe
            raw_bits_tot += BATCH * K; raw_frm_tot += BATCH

    ber_dec = dec_bit / dec_bits_tot
    fer_dec = dec_frm / dec_frm_tot
    ber_det = det_bit / det_bits_tot
    fer_det = det_frm / det_frm_tot
    ber_raw = raw_bit / raw_bits_tot
    fer_raw = raw_frm / raw_frm_tot

    ckpt["BER_decoder"].append(ber_dec);  ckpt["FER_decoder"].append(fer_dec)
    ckpt["BER_detector"].append(ber_det); ckpt["FER_detector"].append(fer_det)
    ckpt["BER_raw"].append(ber_raw);      ckpt["FER_raw"].append(fer_raw)
    ckpt["done_indices"].append(idx)
    save_checkpoint(CKPT_FILE, ckpt)

    print(f"BER_dec={ber_dec:.2e} | BER_det={ber_det:.2e} | "
          f"BER_raw={ber_raw:.2e} | frames~{max(dec_frm_tot, raw_frm_tot):,}",
          flush=True)

# ====================== VẼ FIGURE 9 ======================
x = sigma_mu_list * 100

fig, ax = plt.subplots(figsize=(9, 6))
ax.semilogy(x, ckpt["BER_raw"],      'r-^',  label='BER - raw data w/o coding, detector output', markersize=7, linewidth=1.8)
ax.semilogy(x, ckpt["BER_detector"], 'b-o',  label='BER - proposed code, detector output',        markersize=7, linewidth=1.8)
ax.semilogy(x, ckpt["BER_decoder"],  'k-s',  label='BER - proposed code, decoder output',         markersize=7, linewidth=1.8)
ax.semilogy(x, ckpt["FER_raw"],      'r--^', label='.FER - raw data w/o coding, detector output', markersize=7, linewidth=1.5)
ax.semilogy(x, ckpt["FER_detector"], 'b--o', label='.FER - proposed code, detector output',       markersize=7, linewidth=1.5)
ax.semilogy(x, ckpt["FER_decoder"],  'k--s', label='.FER - proposed code, decoder output',        markersize=7, linewidth=1.5)

ax.set_xlabel(r'$\sigma_0/\mu_0$ (%)', fontsize=13)
ax.set_ylabel('BER & FER', fontsize=13)
ax.set_title(r'Figure 9: Proposed code vs Raw data' '\n'
             r'$\mu_{ofs}=-0.2\ k\Omega$, $\sigma_{ofs}/\mu_1=7\%$, $\alpha=2.5$',
             fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim([2, 10])
ax.set_ylim([1e-4, 1e0])
ax.set_xticks(range(2, 11))
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('figure9.png', dpi=150)
print("\n\nĐã lưu xong! figure9.png")
plt.show()
