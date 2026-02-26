import torch
import torch.nn as nn
from torch.autograd import Function

# Hardware Constraints defined by the PIM array architecture
NUM_CYCLE = 50
NUM_ROWS = 96
NUM_COLS = 128

# PIM API: performs binary matrix multiplication including non-idealities of analog PIM
# Input, weight, and output dimensions are fixed to match the hardware constraints of the PIM array  
class PIM_function(Function):
    @staticmethod
    def forward(ctx, input, weight, mode='sim'):
        """
        Hardware Dimensions:
        Input (Batch x In_features): 50 x 96
        Weight (Out_features x In_features): 128 x 96
        Output (Batch x Out_features): 50 x 128
        """
        # Hardware Guardrails: Check dimensions before execution
        # Input shape check
        assert input.shape == (NUM_CYCLE, NUM_ROWS), f"Hardware Error: Input must be {NUM_CYCLE}x{NUM_ROWS}, got {list(input.shape)}"
        # Weight shape check
        assert weight.shape == (NUM_COLS, NUM_ROWS), f"Hardware Error: Weight must be {NUM_COLS}x{NUM_ROWS}, got {list(weight.shape)}"
        
        # Simulation Mode 
        if mode == 'sim':
            # 1. Binarize weights and inputs (+1/-1)
            weight_b = torch.sign(weight)
            input_b = torch.sign(input)
            
            # 2. Perform Matmul
            output_ideal = torch.matmul(input_b, weight_b.t())
            
            # 3. Add Sense Amp (SA) Thermal noise (Gaussian noise with mean=0 and std=7)
            noise_sigma = 7
            noise = torch.randn_like(output_ideal) * noise_sigma
            output_noise = output_ideal + noise
            
            # Save tensors for backward pass
            ctx.save_for_backward(input, weight, output_noise)

            # 4. 1-bit Output quantization in SA
            output_b = torch.sign(output_noise)

            return output_b
        
        # Hardware Mode
        elif mode == 'hardware':
            # Call PIM chip driver
            # return hardware_driver.run(input, weight)
            pass

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, output_noise = ctx.saved_tensors
        
        # 4. Output quantization: sign() --> Straight-Through Estimator (STE) with Gradient clipping 
        # Gradient clipping [-30, 30] (Hardtanh [-30, 30]) to prevent output_noise from exploding  
        grad_output[output_noise.abs() > 30] = 0
        
        # 3. Addition --> Gradients pass through unchanged

        # 2. Matmul --> Gradients are computed with respect to the binarized weights and inputs
        grad_input = grad_output.matmul(torch.sign(weight))
        grad_weight = grad_output.t().matmul(torch.sign(input))
        
        # 1. Binarization: sign() --> STE
        # Apply gradient clipping to keep latent weights within the clamp range (-1, 1)
        grad_weight[weight.abs() > 1] = 0
        
        return grad_input, grad_weight, None