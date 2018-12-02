import numpy as np

# 激活函数

def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    #     缓存变量、提供给反向传播的函数使用
    cache = Z
    
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    #     缓存变量、提供给反向传播的函数使用
    cache = Z 
    return A, cache

#激活函数的导数
def relu_backward(dA, cache):

    #cache -- 反向传播前一层的缓存（Z）
    Z = cache
    
    dZ = np.array(dA, copy=True) # 初始化dz
    
    # 当Z小于0，导数为0，当z大于0，导数为其本身
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
   
    #cache -- 反向传播前一层的缓存（Z）
    Z = cache
    #sigmoid()
    s = 1/(1+np.exp(-Z))
    #计算导数
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

