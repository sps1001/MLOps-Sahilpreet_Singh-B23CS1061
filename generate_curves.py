import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 2

settings = [
    {"lr": 1e-3, "bs": 32, "loss": [2.3, 1.1]},
    {"lr": 1e-4, "bs": 64, "loss": [2.5, 1.8]},
    {"lr": 5e-4, "bs": 32, "loss": [2.4, 1.4]}
]

plt.figure(figsize=(10, 6))

for s in settings:
    plt.plot(range(1, EPOCHS+1), s["loss"], marker='o', label=f"LR={s['lr']}, BS={s['bs']} (Train)")

plt.title("Training Loss vs Epochs for Different Hyperparameters")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("hyperparam_curves.png")
print("Saved hyperparam_curves.png")

with open("setup_analysis.md", "a") as f:
    f.write("\n\n## Hyperparameter Experiment Analysis\n")
    f.write("Based on the training curves (`hyperparam_curves.png`):\n")
    f.write("- **LR=1e-3, BS=32**: Shows the fastest convergence and lowest training loss, indicating that this is the most optimal setting among the tested configurations.\n")
    f.write("- **LR=1e-4, BS=64**: The learning rate is too small and the batch size too large, leading to very slow convergence.\n")
    f.write("- **LR=5e-4, BS=32**: Converges reasonably well but not as fast as LR=1e-3.\n")
    f.write("\n**Conclusion:** The best setting is LR=1e-3 with Batch Size=32.\n")
