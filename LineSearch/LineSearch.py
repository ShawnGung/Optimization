import numpy as np
import matplotlib.pyplot as plt

"""
consider the function fx = (x1 - 4)^4 + (x2 -3)^2 + 4(x3 +5)^4
"""


def func_val(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return np.power(x1 - 4, 4) + np.power(x2 - 3, 2) + 4 * np.power(x3 + 5, 4)


def gradient(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return np.array([4 * np.power(x1 - 4, 3), 2 * (x2 - 3), 16 * np.power(x3 + 5, 3)])

def gradient2(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    row1 = [12 * np.power(x1 - 4, 2),0,0]
    row2 = [0,2,0]
    row3 = [0,0,48 * np.power(x3 + 5, 2)]
    return np.array([row1,row2,row3])


def isStop(gradient, two_param, two_funcValue, tol=0.00001):
    """
    stopping criteria
    :return:
    """
    # the norm2 of gradient < tol
    if np.linalg.norm(gradient) < tol:
        return True

    # l2 norm of params changing
    xk = two_param[0]
    xk1 = two_param[1]
    if np.linalg.norm(xk1 - xk) < tol:
        return True

    # abs of the changes of function value
    fxk = two_funcValue[0]
    fxk1 = two_funcValue[1]
    if np.abs(fxk1 - fxk) < tol:
        return True
    return False


def constant_stepsize(x, step_size=0.0001, maxInteration=50):
    plotValue = []  # for plotting
    funcValue = []
    params = []
    fx0 = func_val(x)
    funcValue.append([1000, fx0])
    plotValue.append(fx0)
    initializedX = np.array([100, 100, 100])
    params.append([initializedX, x0])
    grad = gradient(x)

    k = 0
    while not isStop(grad, params[-1], funcValue[-1]) and k < maxInteration:
        x = x - step_size * grad
        grad = gradient(x)
        params.append([params[-1][1], x])
        val = func_val(x)
        funcValue.append([funcValue[-1][1], val])
        plotValue.append(val)
        k+=1
    return plotValue

def back_tracking(x,ca = 0.5,cb = 0.8,maxInteration = 50):
    plotValue = []  # for plotting
    funcValue = []
    params = []
    fx0 = func_val(x)
    funcValue.append([1000, fx0])
    plotValue.append(fx0)
    initializedX = np.array([100, 100, 100])
    params.append([initializedX, x0])
    grad = gradient(x)
    k = 0
    while not isStop(grad, params[-1], funcValue[-1]) and k < maxInteration:

        step_size = 1

        # back_tracking
        while True:
            if func_val(x - step_size*grad) > func_val(x) - ca*step_size*np.linalg.norm(grad):
                step_size = cb*step_size
            else:
                break
        x = x - step_size * grad
        grad = gradient(x)
        params.append([params[-1][1], x])
        val = func_val(x)
        funcValue.append([funcValue[-1][1], val])
        plotValue.append(val)
        k+=1
    return plotValue


def newton(x,maxInteration = 50):
    plotValue = []  # for plotting
    funcValue = []
    params = []
    fx0 = func_val(x)
    funcValue.append([1000, fx0])
    plotValue.append(fx0)
    initializedX = np.array([100, 100, 100])
    params.append([initializedX, x0])
    grad = gradient(x)
    k = 0
    while not isStop(grad, params[-1], funcValue[-1]) and k < maxInteration:
        # newtown
        grad2 = gradient2(x)
        x = x - np.linalg.pinv(grad2).dot(grad)
        grad = gradient(x)
        params.append([params[-1][1], x])
        val = func_val(x)
        funcValue.append([funcValue[-1][1], val])
        plotValue.append(val)
        k+=1
    return plotValue


def BFGS(B,two_params,two_grad):
    g0 = two_grad[0]
    g1 = two_grad[1]
    x0 = two_params[0]
    x1 = two_params[1]

    y = g1 - g0
    y = np.reshape(y,[3,1])
    s = x1 - x0
    s = np.reshape(s,[3,1])
    u = y
    v = B.dot(s)
    alpha = 1/(y.T.dot(s))
    beta = -1/(s.T.dot(B).dot(s))

    update_B = B + alpha*u.dot(u.T) + beta*v.dot(v.T)
    return update_B

def quasi_newton(x,B0,maxInteration = 50):
    plotValue = []  # for plotting
    funcValue = []
    params = []
    grad = []
    fx0 = func_val(x)
    funcValue.append([1000, fx0])
    plotValue.append(fx0)
    initializedX = np.array([100, 100, 100])
    params.append([initializedX, x0])
    grad.append([100,gradient(x)])
    k = 0
    B = []
    B.append(B0)
    while not isStop(grad[-1][1], params[-1], funcValue[-1]) and k < maxInteration:

        x = x - np.linalg.pinv(B[-1]).dot(grad[-1][1])
        grad.append([grad[-1][1],gradient(x)])
        params.append([params[-1][1], x])
        val = func_val(x)
        funcValue.append([funcValue[-1][1], val])
        plotValue.append(val)
        k+=1

        # BFGS
        B.append(BFGS(B[-1],params[-1], grad[-1]))
    return plotValue

x0 = np.array([4, 2, -1])

label = []
value = constant_stepsize(x0)
length = len(value)
x1 = np.linspace(0, length, length)
l1, = plt.plot(x1, value,'b')
label.append(l1)

value2 = back_tracking(x0)
length2 = len(value2)
x2 = np.linspace(0, length2, length2)
l2, = plt.plot(x2, value2,'r')
label.append(l2)

value3 = newton(x0)
length3 = len(value3)
x3 = np.linspace(0, length3, length3)
l3, = plt.plot(x3, value3,'k')
label.append(l3)

value4 = quasi_newton(x0,B0 = np.eye(3))
length4 = len(value4)
x4 = np.linspace(0, length4, length4)
l4, = plt.plot(x4, value4,'m')
label.append(l4)

#choose a better B0
B0 = np.array([[0,0,0],[0,2,0],[0,0,1000]])
value5 = quasi_newton(x0,B0 = B0)
length5 = len(value5)
x5 = np.linspace(0, length5, length5)
l5, = plt.plot(x5, value5,'c')
label.append(l5)

order = ['constant stepsize:0.0001','backTracking','newton','quasi_newton,B0 = I','quasi_newton,better B0']
plt.legend(label, order, loc='upper right')
plt.ylim(0 ,1200)
plt.show()

