import numpy as np
import matplotlib.pyplot as plt
import h5py


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

# 从h5格式的数据集中加载数据
def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    #保证维度
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# 初始化两层神经网络各层参数

def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
        n_x -- 输入层size
        n_h -- 隐含层size
        n_y -- 输出层size
    
    Returns:
        W1 -- weight matrix of shape (n_h, n_x)
        b1 -- bias vector of shape (n_h, 1)
        W2 -- weight matrix of shape (n_y, n_h)
        b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
   
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters      


# 初始化深层神经网络的各层参数

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
        layer_dims -- 包含各层的size
    
    Returns:
    parameters -- 
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l], 1)
    """
    
#     np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # 有几层网络

    for l in range(1, L):
        #初始化第l层的W矩阵和b向量
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        #小心symmetry breaking
        parameters['b' + str(l)]= np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

# 单层网络-向前传播-激活函数的-输入值计算

def linear_forward(A, W, b):
    """
    Arguments:
    A -- 前一层的激活值向量
    W -- 当前层的W矩阵——shape (size of current layer, size of previous layer)
    b -- 当前层的偏移向量——shape (size of the current layer, 1)

    Returns:
    Z -- 激活函数的输入值
    cache -- 记住当前层的参数供反向传播快速计算使用
    """
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

# 单层网络-向前传播-激活函数的-输出值计算

def linear_activation_forward(A_prev, W, b, activation):
    """
    Arguments:
    A_prev -- 前一层的激活值向量
    W -- 当前层的W矩阵——shape (size of current layer, size of previous layer)
    b -- 当前层的偏移向量——shape (size of the current layer, 1)
    activation -- 激活函数的选择——"sigmoid" or "relu"

    Returns:
    A -- 激活函数的输出值
    cache -- 记住当前层的参数供反向传播快速计算使用
             contain:
                 1)linear_cache-(A,W,b)
                 2)activation_cache-当前层的激活值
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

# 实现向前传播

def L_model_forward(X, parameters):
    """
    Arguments:
    X -- 输入矩阵-shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- 最终的输出值
    caches -- 从0到L-1层的cache-linear_activation_forward()
    """

    caches = []
    A = X
    # 神经网络的层数
    L = len(parameters) // 2                  
    
    #前L-1层的传播
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)],"relu")
        caches.append(cache)
    
#     第L层的计算
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):
    """
    Arguments:
    AL -- 向前传播的最终输出向量-shape (1, number of examples)
    Y -- 实际的标签值

    Returns:
    cost -- 损失值
    """
    
    m = Y.shape[1]
    #按照函数计算损失值
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # e.g. [[17]] into 17
    assert(cost.shape == ())
    
    return cost

# 单层网络-反向传播-激活函数梯度的-输出值计算

def linear_backward(dZ, cache):
    """
    Arguments:
    dZ -- 当前层的dZ
    cache -- 当前层的正向传播的cache（A,W,b）

    Returns:
    dA_prev -- 前一层的梯度输入
    dW -- 损失函数对W的梯度
    db -- 损失函数对b的梯度
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    #利用公式计算各参数的梯度
    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)

    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

# 单层网络-反向传播-激活函数梯度的-输入值计算
#此过程与具体的激活函数有关
def linear_activation_backward(dA, cache, activation):
    """
    Arguments:
    dA -- 当前层的dA
    cache -- 向前传播的cache(linear_cache, activation_cache)
    activation -- 激活函数类型
    
    Returns:
    dA_prev -- 前一层的dA
    dW -- 损失函数对W的梯度
    db -- 损失函数对b的梯度
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        #计算当前层的dZ
        dZ = relu_backward(dA,activation_cache)
        
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

# 实现向后传播

def L_model_backward(AL, Y, caches):
    """
    Arguments:
    AL -- 最后一层的输出向量
    Y -- 实际的标签值
    caches --从0到L-1层的cache——linear_activation_forward()
    
    Returns:
    grads --每一层的梯度值
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # 层数
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # 保证二者的维度相等
    
    # 初始化第L层的dA
    dAL =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) 
    
    #计算第L层的梯度
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
   #计算第L-1——0层的梯度
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

# 更新函数的计算

def update_parameters(parameters, grads, learning_rate):
    """
    Arguments:
    parameters -- 初始化的神经网络参数
    grads -- output of L_model_backward
    
    Returns:
    parameters:
        contain:
          parameters["W" + str(l)] = ... 
          parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # 层数
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads["db" + str(l+1)]
        
    return parameters

#根据参数预测，并输出精度

def predict(X, y, parameters):
    """
    Arguments:
    X -- 输入数据
    parameters -- 训练好的模型参数
    
    Returns:
    p -- 对于输入数据的预测标签
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # 网络层数
    p = np.zeros((1,m))
    
    # 向前传播
    probas, caches = L_model_forward(X, parameters)

    
    # 将probas转化成0/1标签
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
            
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

# def print_mislabeled_images(classes, X, y, p):
#     """
#     Plots images where predictions and truth were different.
#     X -- dataset
#     y -- true labels
#     p -- predictions
#     """
#     a = p + y
#     mislabeled_indices = np.asarray(np.where(a == 1))
#     plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
#     num_images = len(mislabeled_indices[0])
#     for i in range(num_images):
#         index = mislabeled_indices[1][i]
        
#         plt.subplot(2, num_images, i + 1)
#         plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
#         plt.axis('off')
#         plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
