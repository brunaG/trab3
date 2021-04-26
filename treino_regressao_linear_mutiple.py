import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

def compute_cost(theta_0, theta_1, data):
    """
    Calcula o erro quadratico medio
    Args:
        theta_0 (float): intercepto da reta
        theta_1 (float): lista inclinacao da reta p/ cada var
        data (np.array): matriz com o conjunto de dados, vars na coluna 1:N-1 e y na coluna N
    Retorna:
        float: o erro quadratico medio
    """
    total_cost = 0
    dependent_value=0

    for row in data:
        y = row[-1]
        for n in range(len(theta_1)):
            dependent_value=dependent_value+row[n]*theta_1[n]
        total_cost=total_cost+(y-(dependent_value+theta_0))**2
    total_cost = total_cost / (len(data))
    #print("total_cost: ", total_cost)
    return total_cost


def step_gradient(theta_0_current, theta_n_current, data, alpha):
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
    theta_n_updated = []

    der_theta_0 = 0

    hatH_list=[]
    nfeats=len(data[0])-1

    for i in range(len(data)):
        hatH=theta_0_current
        for n in range(nfeats):
            hatH=hatH+data[i][n]*theta_n_current[n]
        hatH_list.append(hatH)

    for i in range(len(data)):
    	der_theta_0 = der_theta_0+(hatH_list[i] - data[i][-1])
    der_theta_0 = der_theta_0 * (2 / float(len(data)))
    theta_0_updated = theta_0_current - (alpha * der_theta_0)

    for n in range(nfeats): #For each var
        theta_n_grad=0
        for i in range(len(data)): #For each row
            theta_n_grad = theta_n_grad+((hatH_list[i] - data[i][nfeats-1]) * data[i][n])
        theta_n_grad = theta_n_grad * (2 / float(len(data)))
        theta_n_grad=theta_n_current[n]-(alpha * theta_n_grad)
        theta_n_updated.append(theta_n_grad)

    return theta_0_updated, theta_n_updated


def gradient_descent(data, starting_theta_0, starting_theta_n, learning_rate, num_iterations):
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
    theta_n = starting_theta_n

    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []

    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x   Theta0)
    theta_0_progress = []
    theta_n_progress = []
    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    for i in range(num_iterations):
        cost_graph.append(compute_cost(theta_0, theta_n, data))
        theta_0, theta_n = step_gradient(theta_0, theta_n, data, learning_rate)
        theta_0_progress.append(theta_0)
        theta_n_progress.append(theta_n)

    return [theta_0, theta_n, cost_graph, theta_0_progress, theta_n_progress]

def minmax_norm(column):
    norm_col=np.empty(0)
    max_x=column.max()
    min_x=column.min()
    for x in column:
        x_new=(x-min_x)/(max_x-min_x)
        norm_col=np.append(norm_col, x_new)
    return norm_col

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Otimização por descida de gradiente')
    parser.add_argument('filename', metavar='filename', type=str)
    parser.add_argument('iterations', metavar='iterations', type=int)
    parser.add_argument('natributos', metavar='natributos', type=int)
    args = parser.parse_args()
    data = np.genfromtxt(open(args.filename,'r'), delimiter=',',names=True)
    p=3

    if(args.natributos==1): #3.1
        grlivArea=minmax_norm(data['GrLivArea'])
        #Predicted value on the last column
        new_data=np.c_[grlivArea,data['SalePrice']]
        theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress = gradient_descent( new_data, starting_theta_0=100, starting_theta_n=[0.0], learning_rate=0.001, num_iterations=300)
    elif(args.natributos==2): #3.1
        grlivArea = minmax_norm(data['GrLivArea'])
        overallQual = minmax_norm(data['OverallQual'])
        new_data = np.c_[grlivArea,overallQual,data['SalePrice']]
        theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress = gradient_descent(new_data,starting_theta_0=100,starting_theta_n=[0.0,0.0],learning_rate=0.001,num_iterations=300)
    elif(args.natributos==5): #3.1
        grlivArea = minmax_norm(data['GrLivArea'])
        overallQual = minmax_norm(data['OverallQual'])
        overallCond = minmax_norm(data['OverallCond'])
        garageArea = minmax_norm(data['GarageArea'])
        yearBuilt = minmax_norm(data['YearBuilt'])
        new_data = np.c_[grlivArea, overallQual, overallCond, garageArea, yearBuilt, data['SalePrice']]
        theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress = gradient_descent(new_data,starting_theta_0=100,starting_theta_n=[0.0, 0.0, 0.0, 0.0, 0.0],learning_rate=0.001,num_iterations=100)

# Imprimir parâmetros otimizados
    #print("Theta_0 otimizado: ", theta_0)
    #print("Theta_1 otimizado: ", theta_1)

# Imprimir erro com os parâmetros otimizados
    #print("Custo minimizado: ", compute_cost(theta_0, theta_1, new_data))

    if(args.natributos==1):
        print("Theta_0: ", theta_0)
        print("Theta_1: ", theta_1)
        print ('Erro quadratico medio: ', compute_cost(theta_0, theta_1, new_data))
    if(args.natributos==2):
        print("Theta_0: ", theta_0)
        print("Theta_1: ", theta_1[0])
        print("Theta_2: ", theta_1[1])
        print ('Erro quadratico medio: ', compute_cost(theta_0, theta_1, new_data))
    if(args.natributos==5):
        print("Theta_0: ", theta_0)
        print("Theta_1: ", theta_1[0])
        print("Theta_2: ", theta_1[1])
        print("Theta_3: ", theta_1[2])
        print("Theta_4: ", theta_1[3])
        print("Theta_5: ", theta_1[4])
        print ('Erro quadratico medio: ', compute_cost(theta_0, theta_1, new_data))
