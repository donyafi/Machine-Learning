from numpy import *
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def Gradient_Descent(Learning_rate,K,epoch,X,Y,m_new, b_new, tolerance):
    Lr=Learning_rate
    tolerance=tolerance
    K=K
    X=X
    Y=Y
    m_new=m_new
    b_new=b_new
    epoch=epoch
    List_m , List_b ,costs =[] , [] ,[]
    for i in range(epoch):
          Y_0 = m_new*X + b_new  # The current value of Y
          cost= mean_squared_error(Y,Y_0)
          costs.append(cost)
          if (cost > tolerance):
              List_m.append(m_new)
              List_b.append(b_new)
              D_m = (-2/K) * sum(X * (Y - Y_0))  # Derivative wrt m
              D_b = (-2/K) * sum(Y - Y_0)  # Derivative wrt b
              m_new = m_new - Lr * D_m  # Update m
              b_new = b_new - Lr * D_b  # Update b
                
             
          else :
              print('line slope :',m_new,'intercept point', b_new,'error:', cost)
              
    print ('line slope :',m_new,'intercept point', b_new,'error:', cost) 
    plt.plot(costs , List_m ,color='red')
    plt.show()
    plt.scatter(X,Y)
    plt.plot(X, Y_0 , color='red')  # regression line
    plt.show()
    return


#generating samples
tolerance=0.01
m, b = 10 ,-2   # target parametrs
Learning_rate = 0.0001  # The learning Rate
K = 500  # The number of samples
epoch = 10000 # The number of iterations to perform gradient descent
X=random.randint(0,100,500)
X=X.reshape((500,1))

noise = random.normal(0,0.2,500)   #random noise
noise = noise.reshape ((500,1))

Y= m*X + b + noise #real Y

m_new, b_new = 1,0   # new parametrs , What we have

Gradient_Descent(Learning_rate,K,epoch,X,Y,m_new, b_new, tolerance)
    


