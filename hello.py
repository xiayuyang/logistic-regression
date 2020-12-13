#coding=utf-8
import  unittest
import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

def sigmoid(z):
    """
        参数：
            z  - 任何大小的标量或numpy数组。

        返回：
            s  -  sigmoid（z）
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
        此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0。

        参数：
            dim  - 我们想要的w矢量的大小（或者这种情况下的参数数量）

        返回：
            w  - 维度为（dim，1）的初始化向量。
            b  - 初始化的标量（对应于偏差）
    """
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    """
        实现前向和后向传播的成本函数及其梯度。
        参数：
            w  - 权重，大小不等的数组（num_px * num_px * 3，1）
            b  - 偏差，一个标量
            X  - 矩阵类型为（num_px * num_px * 3，训练数量）
            Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)

        返回：
            cost- 逻辑回归的负对数似然成本
            dw  - 相对于w的损失梯度，因此与w相同的形状 dw = dJ/dw
            db  - 相对于b的损失梯度，因此与b的形状相同 db = dJ/db
    """
    # 训练集数量
    m = X.shape[1]
    # 矩阵 1*训练数量
    A = sigmoid(np.dot(w.T, X) + b)
    # *是对应元素相乘
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dz = A - Y
    dw = (1 / m) * np.dot(X, dz.T)
    db = (1 / m) * np.sum(dz)
    # print(str(dw.shape) + str(w.shape))
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    # 变成一个数
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
        此函数通过运行梯度下降算法来优化w和b

        参数：
            w  - 权重，大小不等的数组（num_px * num_px * 3，1）
            b  - 偏差，一个标量
            X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
            Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
            num_iterations  - 优化循环的迭代次数
            learning_rate  - 梯度下降更新规则的学习率
            print_cost  - 每100步打印一次损失值

        返回：
            params  - 包含权重w和偏差b的字典
            grads  - 包含权重和偏差相对于成本函数的梯度的字典
            成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。

        提示：
        我们需要写下两个步骤并遍历它们：
            1）计算当前参数的成本和梯度，使用propagate（）。
            2）使用w和b的梯度下降法则更新参数。
    """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("迭代的次数： {0}， 误差值： {1}".format(str(i), str(cost)))
    params = {
        "w": w,
        "b": b
    }
    return params, costs


def predict(w, b, X):
    """
        使用学习逻辑回归参数logistic （w，b）预测标签是0还是1，

        参数：
            w  - 权重，大小不等的数组（num_px * num_px * 3，1）
            b  - 偏差，一个标量
            X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据

        返回：
            Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）

    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    assert(Y_prediction.shape == (1, m))
    return Y_prediction





train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
# 训练集里图片的数量。
m_train = train_set_y_orig.shape[1]
# 测试集里图片的数量。
m_test = test_set_y_orig.shape[1]
# 训练、测试集里面的图片的宽度和高度（均为64x64）。
num_px = train_set_x_orig.shape[1]

# 现在看一看我们加载的东西的具体情况
# print ("训练集的数量: m_train = " + str(m_train))
# print ("测试集的数量 : m_test = " + str(m_test))
# print ("每张图片的宽/高 : num_px = " + str(num_px))
# print ("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("训练集_图片的维数 : " + str(train_set_x_orig.shape))
# print ("训练集_标签的维数 : " + str(train_set_y_orig.shape))
# print ("测试集_图片的维数: " + str(test_set_x_orig.shape))
# print ("测试集_标签的维数: " + str(test_set_y_orig.shape))

# 训练集的数量: m_train = 209
# 测试集的数量 : m_test = 50
# 每张图片的宽/高 : num_px = 64
# 每张图片的大小 : (64, 64, 3)
# 训练集_图片的维数 : (209, 64, 64, 3)
# 训练集_标签的维数 : (1, 209)
# 测试集_图片的维数: (50, 64, 64, 3)
# 测试集_标签的维数: (1, 50)


#将训练集的维度降低并转置
train_set_x_flatter = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
#将测试集的维度降低并转置
test_set_x_flatter = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# 标准化数据集
train_set_x = train_set_x_flatter / 255
test_set_x = test_set_x_flatter / 255

# print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
# print ("训练集_标签的维数 : " + str(train_set_y.shape))
# print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
# print ("测试集_标签的维数 : " + str(test_set_y.shape))

# 训练集降维最后的维度： (12288, 209)
# 训练集_标签的维数 : (1, 209)
# 测试集降维之后的维度: (12288, 50)
# 测试集_标签的维数 : (1, 50)


# index = 25
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print(str(classes[0]))  #b'non-cat'
# print("train_set_y=" + str(train_set_y_orig)) #你也可以看一下训练集里面的标签是什么样的。





# 打印出当前的训练标签值
# 使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1
# print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
# 只有压缩后的值才能进行解码操作
# print("y=" + str(train_set_y_orig[:, index]) + ", it's a " + classes[np.squeeze(train_set_y_orig[:, index])].decode("utf-8") + "' picture")


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.1, print_cost = False):
    """
        通过调用之前实现的函数来构建逻辑回归模型

        参数：
            X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
            Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
            X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
            Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
            num_iterations  - 表示用于优化参数的迭代次数的超参数
            learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
            print_cost  - 设置为true以每100次迭代打印成本

        返回：
            d  - 包含有关模型信息的字典。
    """
    m = X_train.shape[0]
    w, b = initialize_with_zeros(m)
    params, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_predict_train = predict(w, b, X_train)
    Y_predict_test = predict(w, b, X_test)

    print("训练集准确性： {0}%".format(str(100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100)))
    print("训练集准确性： {0}%".format(str(100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100)))
    d = {
        "costs": costs,
        "Y_predict_train": Y_predict_train,
        "Y_predict_test": Y_predict_test,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iteration": num_iterations
    }
    return d


# d = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations=2000, learning_rate=0.005, print_cost=True)
# costs = np.squeeze(d["costs"])
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations(per hundreds)')
# plt.title('learning_rate =' + str(d['learning_rate']))
# plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print('learning_rate = ' + str(i))
    models[str(i)] = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations=2000, learning_rate=i, print_cost=False)
    print('\n' + "........................................." + '\n')
for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]['costs']), label=(str(i)))
plt.xlabel('cost')
plt.ylabel('iterations')
plt.legend(loc='best')
plt.show()
