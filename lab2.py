 # IS-Lab2 (LT)
#Intelektualiosios sistemos. Antrojo laboratorinio darbo užduotis.
# Tikslas
#Išmokti savarankiškai suprogramuoti paprasto netiesinio aproksimatoriaus mokymo (parametrų skaičiavimo) algoritmą.
# Užduotys (maks. 8 balai)
#1. Sukurkite daugiasluoksnio perceptrono koeficientams apskaičiuoti skirtą programą. Daugiasluoksnis perceptronas turi atlikti aproksimatoriaus funkciją. Daugiasluoksnio perceptrono struktūra:
#- vienas įėjimas (įėjime paduodamas 20 skaičių vektorius X, su reikšmėmis intervale nuo 0 iki 1, pvz., x = 0.1:1/22:1; ).
#- vienas išėjimas (pvz., išėjime tikimasi tokio norimo atsako, kurį galima būtų apskaičiuoti pagal formulę: y = (1 + 0.6\*sin(2\*pi\*x/0.7)) + 0.3\*sin(2\*pi\*x))/2; - kuriamas neuronų tinklas turėtų "modeliuoti/imituoti šios formulės elgesį" naudodamas visiškai kitokią matematinę išraišką nei ši);
#- vienas paslėptasis sluoksnis su hiperbolinio tangento arba sigmoidinėmis aktyvavimo funkcijomis neuronuose (neuronų skaičius: 4-8);
#- tiesine aktyvavimo funkcija išėjimo neurone;
# mokymo algoritmas - Backpropagation (atgalinio sklidimo).
# Papildoma užduotis (papildomi 2 balai)
#Išspręskite paviršiaus aproksimavimo uždavinį, kai tinklas turi du įėjimus ir vieną išėjimą.
# Rekomenduojama literatūra
#- Neural Networks and Learning Machines (3rd Edition), <...> psl., <...> lentelė



#       ->X-\
#     /-> X-\
#    / -> X-\
#I  | ->  X -> O
#    \ -> X-/
#     \-> X-/
#       ->X-/

import math
import random
import matplotlib.pyplot as plt

def activation(x):
    return 1 / (1 + math.exp(-x))

def activation_derivative(x):
    return x * (1 - x)

def desired_value(x):
    return (1 + 0.6*math.sin(2*math.pi*x/0.7)) + (0.3*math.sin(2*math.pi*x))/2

x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
d = list(map(desired_value, x))

w11 = random.uniform(0, 1)
b11 = random.uniform(0, 1)
w21 = random.uniform(0, 1)
b21 = random.uniform(0, 1)
w31 = random.uniform(0, 1)
b31 = random.uniform(0, 1)
w41 = random.uniform(0, 1)
b41 = random.uniform(0, 1)
w51 = random.uniform(0, 1)
b51 = random.uniform(0, 1)
w61 = random.uniform(0, 1)
b61 = random.uniform(0, 1)

w12 = random.uniform(0, 1)
w22 = random.uniform(0, 1)
w32 = random.uniform(0, 1)
w42 = random.uniform(0, 1)
w52 = random.uniform(0, 1)
w62 = random.uniform(0, 1)
b12 = random.uniform(0, 1)
print("Starting Weights:")
print(f"w11: {w11}, w21: {w21}, w31: {w31}, w41: {w41}, w51: {w51}, w61: {w61}")
print(f"b11: {b11}, b21: {b21}, b31: {b31}, b41: {b41}, b51: {b51}, b61: {b61}")
print(f"w12: {w12}, w22: {w22}, w32: {w32}, w42: {w42}, w52: {w52}, w62: {w62}")
print(f"b12: {b12}")
n = 0.5

for i in range(10000):
    for j in range(len(x)):
        v11 = x[j] * w11 + b11
        y11 = activation(v11)
        v21 = x[j] * w21 + b21
        y21 = activation(v21)
        v31 = x[j] * w31 + b31
        y31 = activation(v31)
        v41 = x[j] * w41 + b41
        y41 = activation(v41)
        v51 = x[j] * w51 + b51
        y51 = activation(v51)
        v61 = x[j] * w61 + b61
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

        w11 = w11 + n*delta_hidden1 * x[j]
        b11 = b11 + n*delta_hidden1
        w21 = w21 + n*delta_hidden2 * x[j]
        b21 = b21 + n*delta_hidden2
        w31 = w31 + n*delta_hidden3 * x[j]
        b31 = b31 + n*delta_hidden3
        w41 = w41 + n*delta_hidden4 * x[j]
        b41 = b41 + n*delta_hidden4
        w51 = w51 + n*delta_hidden5 * x[j]
        b51 = b51 + n*delta_hidden5
        w61 = w61 + n*delta_hidden6 * x[j]
        b61 = b61 + n*delta_hidden6
output_values = []
for j in range(len(x)):
    v11 = x[j] * w11 + b11
    y11 = activation(v11)
    v21 = x[j] * w21 + b21
    y21 = activation(v21)
    v31 = x[j] * w31 + b31
    y31 = activation(v31)
    v41 = x[j] * w41 + b41
    y41 = activation(v41)
    v51 = x[j] * w51 + b51
    y51 = activation(v51)
    v61 = x[j] * w61 + b61
    y61 = activation(v61)
    v12 = y11*w12 + y21*w22 + y31*w32 + y41*w42 + y51*w52 + y61*w62+ b12
    output_values.append(v12)
print("Final Weights:")
print(f"w11: {w11}, w21: {w21}, w31: {w31}, w41: {w41}, w51: {w51}, w61: {w61}")
print(f"b11: {b11}, b21: {b21}, b31: {b31}, b41: {b41}, b51: {b51}, b61: {b61}")
print(f"w12: {w12}, w22: {w22}, w32: {w32}, w42: {w42}, w52: {w52}, w62: {w62}")
print(f"b12: {b12}")

plt.plot(x, d, 'b')
plt.plot(x, output_values, 'rx')
plt.show()
