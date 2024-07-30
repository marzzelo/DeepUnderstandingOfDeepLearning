import torch
from rich import print

# Crear un tensor con dimensiones de tamaño 1
tensor_original = torch.randn(1, 3, 1, 4)

# Aplicar torch.squeeze para eliminar dimensiones de tamaño 1
tensor_squeezed = torch.squeeze(tensor_original)

# Mostrar el tensor original y el tensor resultante
print("Tensor original:")
print(tensor_original)
print("Forma del tensor original:", tensor_original.shape)

print("\nTensor después de aplicar torch.squeeze:")
print(tensor_squeezed)
print("Forma del tensor después de aplicar torch.squeeze:", tensor_squeezed.shape)
