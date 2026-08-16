import torch
import matplotlib.pyplot as plt
from typing import List
from src.nn import Sequential, Tanh

def plot_activation_distribution(model: Sequential, figsize=(18, 4)):
    """Tanh activation and saturation in all layers"""
    plt.figure(figsize=figsize)
    legends = []
    tanh_layers = [layer for layer in model.layers[:-1] if isinstance(layer, Tanh)]
    
    for i, layer in enumerate(tanh_layers):
        t = layer.out
        sat_pct = (t.abs() > 0.97).float().mean().item() * 100
        print(f"Layer {i} (Tanh) | Mean: {t.mean():+.2f} | Std: {t.std():.2f} | Saturated: {sat_pct:5.2f}%")
        
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach().cpu(), hy.detach().cpu())
        legends.append(f"Layer {i} (Tanh)")
        
    plt.legend(legends)
    plt.title("Activation Distribution (Tanh Outputs)")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()


def plot_gradient_distribution(model: Sequential, figsize=(18, 4)):
    """Tanh activation gradient distribution in all layers"""
    plt.figure(figsize=figsize)
    legends = []
    tanh_layers = [layer for layer in model.layers[:-1] if isinstance(layer, Tanh)]
    
    for i, layer in enumerate(tanh_layers):
        t = layer.out.grad
        print(f"Layer {i} (Tanh grad) | Mean: {t.mean():+e} | Std: {t.std():e}")
        
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach().cpu(), hy.detach().cpu())
        legends.append(f"Layer {i} (Tanh grad)")
        
    plt.legend(legends)
    plt.title("Activation Gradients Distribution (dL/dOut)")
    plt.xlabel("Gradient Value")
    plt.ylabel("Density")
    plt.show()


def plot_weights_gradient_distribution(parameters: List[torch.Tensor], figsize=(18, 4)):
    """2D weights gradient distribution and gradient/data update ratio"""
    plt.figure(figsize=figsize)
    legends = []
    weight_params = [p for p in parameters if p.ndim == 2]
    
    for i, p in enumerate(weight_params):
        t = p.grad
        ratio = t.std() / p.std()
        print(f"Weight {tuple(p.shape)!s:12s} | Mean: {t.mean():+e} | Std: {t.std():e} | Grad:Data Ratio: {ratio:e}")
        
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach().cpu(), hy.detach().cpu())
        legends.append(f"Param {i} {tuple(p.shape)}")
        
    plt.legend(legends)
    plt.title("Weights Gradient Distribution")
    plt.xlabel("Gradient Value")
    plt.ylabel("Density")
    plt.show()


def plot_update_to_data_ratio(ud_history: List[List[float]], parameters: List[torch.Tensor], figsize=(18, 4)):
    """Update-to-Data Ratio (training step/params std)"""
    plt.figure(figsize=figsize)
    legends = []
    weight_indices = [i for i, p in enumerate(parameters) if p.ndim == 2]
    
    for idx in weight_indices:
        plt.plot([step_stats[idx] for step_stats in ud_history])
        legends.append(f"Param {idx} {tuple(parameters[idx].shape)}")
        
    plt.plot([0, len(ud_history)], [-3, -3], 'k--', label="Target Threshold (~1e-3)")
    plt.legend(legends)
    plt.title("Update-to-Data Ratio History (log10 scale)")
    plt.xlabel("Training Step")
    plt.ylabel("log10(lr * grad.std / data.std)")
    plt.show()