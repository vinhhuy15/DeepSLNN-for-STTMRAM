import numpy as np
import bin2dec_dec2bin as sp


def decode_euclidean(received_att, codebook_7b9b, k, in_bits):
    """
    Giải mã dữ liệu bằng phương pháp Minimum Euclidean Distance.

    Tham số:
        received_att   : Mảng numpy (blk, n) chứa dữ liệu đã nhận sau attenuator.
        codebook_7b9b  : Mảng numpy (128, n) chứa codebook.
        k              : Số bit thông tin (ví dụ: 7).
        in_bits        : Mảng numpy (blk, k) chứa dữ liệu gốc 7-bit để so sánh lỗi.

    Trả về:
        total_bit_errors  : Tổng số lỗi bit.
        total_frame_errors: Tổng số lỗi khung.
    """
    blk = received_att.shape[0]

    # ---- Vectorized: tính khoảng cách Euclidean toàn bộ batch cùng lúc ----
    # received_att : (blk, n)  →  expand thành (blk, 1, n)
    # codebook     : (128, n)  →  expand thành (1, 128, n)
    # dists        : (blk, 128)
    dists = np.sum((received_att[:, np.newaxis, :] - codebook_7b9b[np.newaxis, :, :]) ** 2, axis=2)
    best_idx = np.argmin(dists, axis=1)   # (blk,)

    # Giải mã từng index → mảng bit (blk, k)
    # Dùng bảng tra để tránh vòng lặp Python
    decode_table = np.array(
        [[int(b) for b in format(i, 'b').zfill(k)] for i in range(2 ** k)],
        dtype=int
    )  # (128, k)
    out_dec = decode_table[best_idx]   # (blk, k)

    # Đếm lỗi
    bit_err_per_frame = np.sum(out_dec != in_bits, axis=1)   # (blk,)
    total_bit_errors   = int(np.sum(bit_err_per_frame))
    total_frame_errors = int(np.sum(bit_err_per_frame > 0))

    return total_bit_errors, total_frame_errors
