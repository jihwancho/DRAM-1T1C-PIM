# DRAM PIM Accelerator Simulation API

This repository provides a PyTorch-based simulation environment for our analog DRAM PIM (Processing-in-Memory) accelerator, specifically optimized for Binary Neural Networks (BNN).

This repository includes:
* **PIM API**: A functional simulation of the hardware macro, including analog non-idealities.
* **MNIST Training**: A complete example script demonstrating how to train a Binary Neural Network (BNN) on the MNIST dataset using the API.

---

## PIM API (`PIM_function`)
### Usage
```python
output = PIM_function.apply(input, weight, mode='sim')
```
### Description
The `PIM_function` performs binary matrix multiplication, incorporating the non-idealities of the analog PIM accelerator.

$$OUT = \text{sgn}(IN_b \cdot W_b^T + \epsilon)$$ 

Where: 
* $\text{sgn}()$ is the sign function.
* $IN_b = \text{sgn}(IN)$ 
* $W_b = \text{sgn}(W)$
* $\epsilon \sim \mathcal{N}(0, 7^2)$ represents the Sense Amp (SA) thermal noise.

### Hardware Constraints
The API is strictly constrained by the following physical hardware parameters:

* **PIM Array Size**: 96 (`NUM_ROWS`) x 128 (`NUM_COLS`)
* **Compute Cycle**: 50 (`NUM_CYCLE`)
* **Precision**: 1-bit Weight, 1-bit Input, 1-bit Output

### Required Tensor Dimensions
Input and Weight tensors must match the following shapes exactly to satisfy hardware constraints:
| Tensor | Parameterized Shape | Actual Shape |
| :--- | :--- | :--- |
| **Input** | `NUM_CYCLE` x `NUM_ROWS` | **50 x 96** |
| **Weight** | `NUM_COLS` x `NUM_ROWS` | **128 x 96** |
| **Output** | `NUM_CYCLE` x `NUM_COLS` | **50 x 128** |

### Parameters
* **mode**
  - **`'sim'`(default)**: Simulation mode, used in simulation.
  - **`'hardware'`**: Hardware mode, calling the actual PIM chip driver used in measurement. 

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