# -*- coding: utf-8 -*-
"""Copy of treino-regressao-linear.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zpjdRLpaIvq6OIL1p02GVpPR7Ycmy5RJ

**Pressione SHIFT ENTER para executar a célula**

# Regressão Linear Simples

Este notebook irá auxiliar você a ajustar um modelo de regressão linear "do zero" usando somente a biblioteca `numpy`.
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

"""#### Importando o conjunto de dados
Neste momento, vamos utilizar o conjunto de dados com duas colunas: *Horas de estudo* e *Nota final*, respectivamente.
"""


def compute_cost(theta_0, theta_1, data):
    """
    Calcula o erro quadratico medio

    Args:
        theta_0 (float): intercepto da reta
        theta_1 (float): inclinacao da reta
        data (np.array): matriz com o conjunto de dados, x na coluna 0 e y na coluna 1

    Retorna:
        float: o erro quadratico medio
    """
    total_cost = 0

    for row in data:
        x = row[0]
        y = row[1]
        total_cost=total_cost+(y-(x*theta_0+theta_1))**2
    total_cost = total_cost / (len(data))
    print("total_cost: ", total_cost)
    return total_cost


"""#### Teste do Erro Quadratico Medio

Seu EQM para a reta horizontal (theta_1 = 0) interceptando y em 0 (theta_0 = 0) deve ser 5565.107834483211


"""
def testa_EQM():
    theta_0_test = 0
    theta_1_test = 0
    # comparacao de floats com tolerancia 1E-11
    if abs(compute_cost(theta_0_test, theta_1_test, data) - 5565.107834483211) < 1e-11:
        print("Erro Quadratico Medio passou no teste")
    else:
        print("ERRO NO CALCULO DO ERRO QUADRATICO MEDIO!")


"""#### Define as funções de Gradiente Descendente"""


def step_gradient(theta_0_current, theta_1_current, data, alpha):
    """Calcula um passo em direção ao EQM mínimo

    Args:
        theta_0_current (float): valor atual de theta_0
        theta_1_current (float): valor atual de theta_1
        data (np.array): vetor com dados de treinamento (x,y)
        alpha (float): taxa de aprendizado / tamanho do passo

    Retorna:
        tupla: (theta_0, theta_1) os novos valores de theta_0, theta_1
    """

    theta_0_updated = 0
    theta_1_updated = 0

    der_theta_0 = 0
    der_theta_1 = 0

    for i in range(len(data)):
        h = theta_0_current + data[i,0] * theta_1_current
        der_theta_0 = der_theta_0+(h - data[i, 1])
    der_theta_0 = der_theta_0 * (2 / len(data))

    for i in range(len(data)):
        h = theta_0_current + data[i,0] * theta_1_current
        der_theta_1 = der_theta_1+((h - data[i, 1]) * data[i, 0])
    der_theta_1 = der_theta_1 * (2 / len(data))

    theta_0_updated = theta_0_current - (alpha * der_theta_0)
    theta_1_updated = theta_1_current - (alpha * der_theta_1)

    return theta_0_updated, theta_1_updated


"""### Teste da funcao step_gradient"""


def testa_step_gradient():
    # dataset copiado do Quiz de Otimizacao Continua
    other_data = np.array([[1, 3], [2, 4], [3, 4], [4, 2]])

    new_theta0, new_theta1 = step_gradient(1, 1, other_data, alpha=0.1)
    # comparacao de floats com tolerancia 1E-11
    if abs(new_theta0 - 0.95) < 1e-11:
        print("Atualizacao de theta0 OK")
    else:
        print("ERRO NA ATUALIZACAO DE theta0!")

    if abs(new_theta1 - 0.55) < 1e-11:
        print("Atualizacao de theta1 OK")
    else:
        print("ERRO NA ATUALIZACAO DE theta1!")


def gradient_descent(data, starting_theta_0, starting_theta_1, learning_rate, num_iterations):
    """executa a descida do gradiente

    Args:
        data (np.array): dados de treinamento, x na coluna 0 e y na coluna 1
        starting_theta_0 (float): valor inicial de theta0
        starting_theta_1 (float): valor inicial de theta1
        learning_rate (float): hyperparâmetro para ajustar o tamanho do passo durante a descida do gradiente
        num_iterations (int): hyperparâmetro que decide o número de iterações que cada descida de gradiente irá executar

    Retorna:
        list : os primeiros dois parâmetros são o Theta0 e Theta1, que armazena o melhor ajuste da curva. O terceiro e quarto parâmetro, são vetores com o histórico dos valores para Theta0 e Theta1.
    """

    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1

    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []

    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x   Theta0)
    theta_0_progress = []
    theta_1_progress = []

    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    num_iterations = 10
    for i in range(num_iterations):
        cost_graph.append(compute_cost(theta_0, theta_1, data))
        theta_0, theta_1 = step_gradient(theta_0, theta_1, data, alpha=0.0001)
        theta_0_progress.append(theta_0)
        theta_1_progress.append(theta_1)

    return [theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress]

def minmax_norm(column):
    norm_col=np.empty(column.size)
    max_x=column.max()
    min_x=column.min()
    for x in data['Id']:
        x_new=(x-min_x)/(max_x-min_x)
        norm_col=norm_col.append(x_new)
    return norm_col

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Otimização por descida de gradiente')
    parser.add_argument('filename', metavar='filename', type=str)
    parser.add_argument('iterations', metavar='iterations', type=int)
    parser.add_argument('natributos', metavar='natributos', type=int)
    args = parser.parse_args()

    data = np.genfromtxt(open(args.filename,'r'), delimiter=',',names=True)
    
    
    theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress = gradient_descent( data, starting_theta_0=0, starting_theta_1=0, learning_rate=0, num_iterations=10)
# Imprimir parâmetros otimizados
    print("Theta_0 otimizado: ", theta_0)
    print("Theta_1 otimizado: ", theta_1)

# Imprimir erro com os parâmetros otimizados
    print("Custo minimizado: ", compute_cost(theta_0, theta_1, data))


