import numpy as np
import matplotlib.pyplot as plt
from config import (
    K, N, P1, ALPHA, MIN_ERRORS, BATCH,
    codebook_7b9b,
    MRAM_channel_batch, encode_batch,
    decode_euclidean_batch,
    load_checkpoint, save_checkpoint
)

# ====================== THAM SỐ FIGURE 10 ======================
OFFSET_MU     = -0.2
sigma_mu_list = np.linspace(0.02, 0.10, 9)   # 2% -> 10%, 9 điểm

CASES = [
    {'offset_sigma_ratio': 0.04,
     'label': r'BER - proposed code, $\sigma_{ofs}/\mu_1=4\%$',
     'color': 'b', 'marker': 'o',
     'ckpt': 'figure10_4pct_checkpoint.npy'},
    {'offset_sigma_ratio': 0.07,
     'label': r'BER - proposed code, $\sigma_{ofs}/\mu_1=7\%$',
     'color': 'k', 'marker': '^',
     'ckpt': 'figure10_7pct_checkpoint.npy'},
]

print("=" * 65)
print("  FIGURE 10 — BER vs sigma_0/mu_0 (2% -> 10%)")
print(f"  P1={P1:.0e}, alpha={ALPHA}, offset_mu={OFFSET_MU} kOhm")
print(f"  So sánh: offset_sigma_ratio = 4% vs 7%")
print(f"  Dừng khi >= {MIN_ERRORS} bit lỗi mỗi điểm | BATCH={BATCH} frames")
print("=" * 65)

# ====================== VÒNG LẶP CHÍNH ======================
results = []

for case in CASES:
    offset_sigma_ratio = case['offset_sigma_ratio']
    ckpt_file          = case['ckpt']

    ckpt = load_checkpoint(ckpt_file)
    if ckpt is None:
        ckpt = {"BER_list": [], "done_indices": []}
        print(f"\n  === offset_sigma_ratio = {offset_sigma_ratio*100:.0f}% — bắt đầu mới ===")
    else:
        print(f"\n  === offset_sigma_ratio = {offset_sigma_ratio*100:.0f}% — "
              f"tiếp tục, đã xong {len(ckpt['done_indices'])} điểm ===")

    for idx, sigma_mu in enumerate(sigma_mu_list):
        if idx in ckpt["done_indices"]:
            print(f"    sigma_mu = {sigma_mu*100:.0f}% — bỏ qua (checkpoint).")
            continue

        pct = sigma_mu * 100
        dec_bit=0; dec_bits_tot=0; dec_frm_tot=0

        print(f"\n    sigma_mu = {pct:.0f}% ...", end=' ', flush=True)

        while dec_bit < MIN_ERRORS:
            in_bits = np.random.randint(0, 2, size=(BATCH, K), dtype=int)
            out_enc = encode_batch(in_bits)

            received_coded = MRAM_channel_batch(
                out_enc, sigma_mu, P1,
                offset_mu=OFFSET_MU, offset_sigma_ratio=offset_sigma_ratio
            )

            be, fe = decode_euclidean_batch(received_coded / ALPHA, in_bits)
            dec_bit      += be
            dec_bits_tot += BATCH * K
            dec_frm_tot  += BATCH

        ber = dec_bit / dec_bits_tot
        ckpt["BER_list"].append(ber)
        ckpt["done_indices"].append(idx)
        save_checkpoint(ckpt_file, ckpt)

        print(f"BER={ber:.2e} | frames~{dec_frm_tot:,}", flush=True)

    results.append(ckpt["BER_list"])

# ====================== VẼ FIGURE 10 ======================
x = sigma_mu_list * 100

fig, ax = plt.subplots(figsize=(9, 6))
for i, case in enumerate(CASES):
    ax.semilogy(x, results[i],
                color=case['color'], marker=case['marker'],
                linestyle='-', label=case['label'],
                markersize=7, linewidth=1.8)

ax.set_xlabel(r'$\sigma_0/\mu_0$ (%)', fontsize=13)
ax.set_ylabel('BER', fontsize=13)
ax.set_title(r'Figure 10: Proposed code — effect of offset variation' '\n'
             r'$\mu_{ofs}=-0.2\ k\Omega$, $\alpha=2.5$, $P_1=2\times10^{-4}$',
             fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim([2, 10])
ax.set_ylim([1e-4, 1e-1])
ax.set_xticks(range(2, 11))
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('figure10.png', dpi=150)
print("\n\nĐã lưu xong! figure10.png")
plt.show()
