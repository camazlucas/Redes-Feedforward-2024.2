# Questão 1
import numpy as np
#1.1

#Gradiente
def gradF(w):
    y = np.array([
        3*(w[0]**2)*(w[1]**2) - 6*(w[0]**2)*w[1] - 3*(w[1]**2) + 6*w[1],
        2*w[1]*(w[0]**3) - 2*(w[0]**3) - 6*w[0]*w[1] + 6*w[0]
    ])
    return y

#Hessiana
def Hessi(w):
    y = np.array([
        [6*w[0]*(w[1]**2) - 12*w[0]*w[1], 6*(w[0]**2)*w[1] - 6*(w[0]**2) - 6*w[1] + 6],
        [6*(w[0]**2)*w[1] - 6*(w[0]**2) - 6*w[1] + 6, 2*(w[0]**3) - 6*w[0]]
    ])
    return y

#1.2
#Pontos
w01 = [0, 0]
w02 = [1, 1]
w03 = [-1, 1]
w04 = [1, -1]

grad1 = gradF(w01)
grad2 = gradF(w02)
grad3 = gradF(w03)
grad4 = gradF(w04)

Hessi1 = Hessi(w01)
Hessi2 = Hessi(w02)
Hessi3 = Hessi(w03)
Hessi4 = Hessi(w04)

#1.3
#F(1,-1) = -6

#1.4
#rtg = (0,-1)t + (1, -1)

#1.5
def NewRap(w):
    h = np.linalg.inv(Hessi(w))
    g = gradF(w)[:, np.newaxis]
    w0 = np.array(w)[:, np.newaxis]
    prod = np.dot(h,g)
    y = w0 - prod
    return y

wnr = NewRap(w04)

print('NewRapson aplicado 1.5 =', wnr)

#Como o módulo do gradiente do ponto gerado é 0 então chegou em um ponto extremo.

#1.6

wrep = NewRap(w01)

print('NewRapson aplicado 1.6 =', wrep)

#Continuou no mesmo ponto pois é um ponto extremo.

#1.7
def alfaotimo(w):
    d = -((gradF(w))[:, np.newaxis])
    alfacima = np.dot(gradF(w), d)
    alfabaixo = np.dot(np.dot(-gradF(w), Hessi(w)),d)
    y = -(alfacima/alfabaixo)
    return y

alfa = alfaotimo(w04)
print('Valor do alfa otimo =', alfa)
wotimo = w04 - alfa*gradF(w04)

print('Ponto encontrado com o alfaotimo :', wotimo)

#Como o alfaotimo é negativo o ponto extremo encontrado é um máximo. 
     

