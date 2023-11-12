# Papildoma užduotis (papildomi 2 balai)
# Išspręskite paviršiaus aproksimavimo uždavinį, kai tinklas turi du įėjimus ir vieną išėjimą.
# Rekomenduojama literatūra
# - Neural Networks and Learning Machines (3rd Edition), <...> psl., <...> lentelė


#       -> X-\
#      /-> X-\
#I1   / -> X-\
#    | -> X -> O
#I2   \ -> X-/
#      \-> X-/
#       ->X-/

import math
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

def activation(x):
    return 1 / (1 + np.exp(-x))

def activation_derivative(x):
    return x * (1 - x)

def desired_value(x, z):
    return (1 + 0.6 * np.sin(2 * np.pi * x / 0.7)) + (0.3 * np.sin(2 * np.pi * z)) / 2

x1 = np.linspace(0.05, 1.0, 20)
x2 = np.linspace(0.05, 1.0, 20)
x1, x2 = np.meshgrid(x1, x2)

d = desired_value(x1, x2)

w111 = random.uniform(0, 1)
w112 = random.uniform(0, 1)
b11 = random.uniform(0, 1)
w211 = random.uniform(0, 1)
w212 = random.uniform(0, 1)
b21 = random.uniform(0, 1)
w311 = random.uniform(0, 1)
w312 = random.uniform(0, 1)
b31 = random.uniform(0, 1)
w411 = random.uniform(0, 1)
w412 = random.uniform(0, 1)
b41 = random.uniform(0, 1)
w511 = random.uniform(0, 1)
w512 = random.uniform(0, 1)
b51 = random.uniform(0, 1)
w611 = random.uniform(0, 1)
w612 = random.uniform(0, 1)
b61 = random.uniform(0, 1)

w12 = random.uniform(0, 1)
w22 = random.uniform(0, 1)
w32 = random.uniform(0, 1)
w42 = random.uniform(0, 1)
w52 = random.uniform(0, 1)
w62 = random.uniform(0, 1)
b12 = random.uniform(0, 1)
n = 0.1

for i in range(10000):
    for j in range(len(x1)):
        v11 = x1[j] * w111 + x2[j] * w112 + b11
        y11 = activation(v11)
        v21 = x1[j] * w211 + x2[j] * w212 + b21
        y21 = activation(v21)
        v31 = x1[j] * w311 + x2[j] * w312 + b31
        y31 = activation(v31)
        v41 = x1[j] * w411 + x2[j] * w412 + b41
        y41 = activation(v41)
        v51 = x1[j] * w511 + x2[j] * w512 + b51
        y51 = activation(v51)
        v61 = x1[j] * w611 + x2[j] * w612 + b61
        y61 = activation(v61)

        v12 = y11*w12 + y21*w22 + y31*w32 + y41*w42 + y51*w52 + y61*w62 + b12
        y12 = v12

        e = d[j] - y12

        delta_out = e
        delta_hidden1 = activation_derivative(y11) * (w12*delta_out)
        delta_hidden2 = activation_derivative(y21) * (w22*delta_out)
        delta_hidden3 = activation_derivative(y31) * (w32*delta_out)
        delta_hidden4 = activation_derivative(y41) * (w42*delta_out)
        delta_hidden5 = activation_derivative(y51) * (w52*delta_out)
        delta_hidden6 = activation_derivative(y61) * (w62*delta_out)

        w12 = w12 + n * delta_out * y11
        w22 = w22 + n * delta_out * y21
        w32 = w32 + n * delta_out * y31
        w42 = w42 + n * delta_out * y41
        w52 = w52 + n * delta_out * y51
        w62 = w62 + n * delta_out * y61
        b12 = b12 + n * delta_out

        w111 = w111 + n*delta_hidden1 * x1[j]
        w112 = w112 + n*delta_hidden1 * x2[j]
        b11 = b11 + n*delta_hidden1
        w211 = w211 + n*delta_hidden2 * x1[j]
        w212 = w212 + n*delta_hidden2 * x2[j]
        b21 = b21 + n*delta_hidden2
        w311 = w311 + n*delta_hidden3 * x1[j]
        w312 = w312 + n*delta_hidden3 * x2[j]
        b31 = b31 + n*delta_hidden3
        w411 = w411 + n*delta_hidden4 * x1[j]
        w412 = w412 + n*delta_hidden4 * x2[j]
        b41 = b41 + n*delta_hidden4
        w511 = w511 + n*delta_hidden5 * x1[j]
        w512 = w512 + n*delta_hidden5 * x2[j]
        b51 = b51 + n*delta_hidden5
        w611 = w611 + n*delta_hidden6 * x1[j]
        w612 = w612 + n*delta_hidden6 * x2[j]
        b61 = b61 + n*delta_hidden6
output_values = []
for j in range(len(x1)):
    v11 = x1[j] * w111 + x2[j] * w112 + b11
    y11 = activation(v11)
    v21 = x1[j] * w211 + x2[j] * w212 + b21
    y21 = activation(v21)
    v31 = x1[j] * w311 + x2[j] * w312 + b31
    y31 = activation(v31)
    v41 = x1[j] * w411 + x2[j] * w412 + b41
    y41 = activation(v41)
    v51 = x1[j] * w511 + x2[j] * w512 + b51
    y51 = activation(v51)
    v61 = x1[j] * w611 + x2[j] * w612 + b61
    y61 = activation(v61)
    v12 = y11*w12 + y21*w22 + y31*w32 + y41*w42 + y51*w52 + y61*w62+ b12
    output_values.append(v12)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, d, cmap='viridis', alpha=0.5)  # Plot the desired values as a surface

# Plot the output values as points
ax.scatter(x1, x2, output_values, color='red', s=50, label='Output Values')

# Set labels
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Desired Value / Output Value')

# Show the plot
plt.show()
# # Create a 3D surface plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x1, x2, d, cmap='viridis')
#
# # Set labels
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Desired Value')
#
# # Show the plot
# plt.show()
