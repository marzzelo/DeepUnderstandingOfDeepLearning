import torch

# Create a 3D tensor with requires_grad=True
tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)

# Define a simple operation on the tensor
output = (tensor**2).sum()
print(f"Output: {output}")

# Call backward() to compute gradients
output.backward()

# Print the gradients
print(tensor.grad)

# make a 3d plot of the function z = (tensor**2).sum() and its gradient at the point (1, 2)
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a grid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = (x**2 + y**2)

# Plot the function
ax.plot_surface(x, y, z, alpha=0.5)

# Plot the gradient at the point (1, 2)
grad = tensor.grad.numpy()
# ax.quiver(1, 2, (grad[0, 0]), (grad[0, 1]), color='r')
# TypeError: Axes3D.quiver() missing 2 required positional arguments: 'V' and 'W'
ax.quiver(1, 2, 0, grad[0, 0], grad[0, 1], 0, color='r')

plt.show()


