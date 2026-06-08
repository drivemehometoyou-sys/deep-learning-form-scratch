import numpy as np
from math_basics import numerical_gradient
from math_basics import function_2

def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x = init_x.copy()

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    
    return x

if __name__ == '__main__':
    init_x = np.array([-3.0,-4.0])
    print(gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100))
    print(gradient_descent(function_2,init_x=init_x,lr=1,step_num=100))
    print(gradient_descent(function_2,init_x=init_x,lr=10.0,step_num=100))
    print(gradient_descent(function_2,init_x=init_x,lr=1e-10,step_num=100))
    print(gradient_descent(function_2,init_x=init_x,lr=10,step_num=100))



    

    
