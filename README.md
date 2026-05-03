# STT-MRAM 7/9-Rate Sparse Code AI-Driven Decoders

This repository contains the simulation framework, implementation, and evaluation scripts for AI-driven decoders targeting 7/9-rate sparse codes in Spin-Transfer Torque MRAM (STT-MRAM). It explores and benchmarks advanced Deep Learning mechanisms—specifically a Deep Single-Layer Neural Network (SLNN) and a Graph Neural Network (GNN)—against a traditional Euclidean Distance baseline.

## 📖 Project Overview

STT-MRAM memory cells suffer from asymmetric write errors, read disturb issues, and non-Gaussian channel impairments (like resistance spread and temperature-induced offset). Traditional Maximum Likelihood (ML) decoders using Euclidean distance treat each codeword independently and fail to fully exploit the combinatorial structure of the underlying 7/9 codebook (where codewords specifically have a Hamming weight of 2 or 4).

This project shifts from traditional "single-shot" Euclidean distance comparison to an **AI-driven** approach:
1.  **Deep SLNN Decoder**: A deep neural network pipeline that maps noisy multi-dimensional inputs into 128 codeword classes.
2.  **Graph Neural Network (GNN) Decoder**: Formulates the decoding process as a Node-Classification problem on a graph. The graph is constructed based on the Hamming distance between codewords, letting the GNN propagate logical structures to overcome severe asymmetric channel inferences.

## 📂 Repository Structure

### Core Components
- `config.py`: Global configuration logic, channel parameters (e.g., offsets, variances), and tuning variables.
- `Encoder_with_channel.py`: Implements the 7/9-rate sparse code encoding process and an end-to-end simulation of the cascaded STT-MRAM channel behavior (incorporating BAC write errors, Z-channel read disturbs, and Gaussian mixture sensing noise).

### Decoders
- `decoder_euclidien.py`: The baseline traditional decoder implementation computing the minimum Euclidean distance.
- `slnn_decoder.py`: Architecture for the Deep SLNN learning algorithm.
- `gnn_decoder.py`: Architecture for the Graph Neural Network-based decoding mechanism, utilizing spatial message passing over the codebook's graph.

### Training Scripts
- `train_gnn.py`: Pipeline to train the Graph Neural Network.
- `simulate_slnn.py` / `deep_slnn_model.pt`: Logic for training the Deep SLNN model over augmented channel data and its saved model weights.

### Simulations & Benchmarking
- `Simulate_figure7.py` to `Simulate_figure10.py`: Automated evaluation scripts. These simulate different channel conditions across varying noise ranges ($\sigma/\mu$ from 8% to 15%) and plot comparisons outperforming the baseline algorithm.
- `BER_Euclidean_vs_Deep_SLNN.csv` & `*.npy`: Checkpoints and serialized output logs from the automated simulation measurements mapping the Bit Error Rate (BER) and Frame Error Rate (FER).

## 🚀 Getting Started

### Prerequisites
Make sure you have an active Python working environment. Typical dependencies include:
* `Python 3.8+`
* `torch` (PyTorch)
* `torch-geometric` (for the GNN decoder)
* `numpy`
* `pandas`
* `matplotlib` / `scipy`

Install dependencies using pip:
```bash
pip install torch torchvision torchaudio torch-geometric numpy pandas matplotlib scipy
```

### Usage

1. **Verify the Configuration:**
   Review `config.py` to ensure dataset sizes, learning rates, epochs, and baseline MRAM channel definitions meet your computational capacity.

2. **Train AI Models:**
   Generate models adapted to dynamic environment offsets:
   ```bash
   python train_gnn.py
   python simulate_slnn.py
   ```

3. **Run Performance Evaluations:**
   To benchmark BER/FER versus varying $\sigma/\mu$ intervals:
   ```bash
   python Simulate_figure7.py
   ```
   (Similarly, execute `Simulate_figure8.py`, `9`, and `10` for different noise permutations and visualizations).

## 📈 Methodology
*   **Data Augmentation:** The models are actively trained using dynamic augmentation simulating physical anomalies like drifting $\mu$ offsets natively inside the training loops.
*   **Early Stopping:** Both GNN and SLNN feature native step-learning-rate (`StepLR`) adjustments and patience-based Early Stopping logic to halt computing securely when loss parameters effectively plateau.

## 📜 References
- Summarized theoretical constraints and designs can be explored in `new_way.md` and `decoder_architecture_summary.md` found within this workspace.
