# DRAM PIM Accelerator Simulation API

This repository provides a PyTorch-based simulation environment for our DRAM PIM (Processing-in-Memory) accelerator, specifically optimized for Binary Neural Networks (BNN).

This repository includes:
* **PIM API (`PIM_function`)**: A functional simulation of the hardware macro, including analog non-idealities.
* **MNIST Training**: A complete example script demonstrating how to train a Binary Neural Network (BNN) on the MNIST dataset using the API.

---

## Hardware Specifications
The API is strictly constrained by the following physical hardware parameters:

* **PIM Array Size** 96 (`NUM_ROWS`) x 128 (`NUM_COLS`)
* **Compute Cycle**: 50 (`NUM_CYCLE`)
* **Precision**: 1-bit Weight, 1-bit Input, 1-bit Output

Input and Weight tensors must match these shapes exactly to satisfy hardware guardrails:

### Required Tensor Dimensions
| Tensor | Parameterized Shape | Actual Shape |
| :--- | :--- | :--- |
| **Input** | `NUM_CYCLE` x `NUM_ROWS` | **50 x 96** |
| **Weight** | `NUM_COLS` x `NUM_ROWS` | **128 x 96** |
| **Output** | `NUM_CYCLE` x `NUM_COLS` | **50 x 128** |

---

## Modeled Non-Idealities
To ensure software results match our silicon measurements, the following analog effects are included in the `sim` mode:
* **Sense Amp (SA) Thermal Noise**: Gaussian noise with $\sigma = 7$ is applied to the partial sums before SA quantization.
* **SA Quantization**: 1-bit binarization via `torch.sign`.
* **Gradient Approximation**: Straight-Through Estimator (STE) with gradient clipping at $\pm 30$ to prevent divergence.

---

## Directory Tree
```text
.
├── library/             # API library
│   ├── __init__.py      # Package initialization
│   ├── pim.py           # PIM Accelerator API
│   └── models.py        # Example layers and models
├── data/                # Dataset
├── main_pim.py          # Main training script
└── README.md            # It's me
```

---

## Quick Start
1. Ensure you have PyTorch and Torchvision installed: 
```bash 
pip install torch torchvision
```
2. Run the main training script: 
```bash 
python main_pim.py
```