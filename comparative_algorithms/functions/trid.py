import numpy as np

TARGET = [45, 36, 55, 36, 30, 75, 12, 30, 36]
RECIPE = [40, 40, 40, 40, 40, 40, 40, 40, 40]  # N=1 fitness=0.354
X_ortho_values = np.array([
    [0, 81.0],
    [0, 81.0],
    [0, 81.0],
    [0, 81.0],
    [0, 81.0],
    [0, 81.0],
    [0, 81.0],
    [0, 81.0],
    [0, 81.0],
])

Y_dim, X_dim = len(TARGET), X_ortho_values.shape[0]
X_input = [f'x{i}' for i in range(1, X_dim+1)]
Y_input = [f'y{i}' for i in range(1, Y_dim+1)]

p = np.array([
       [ 0.00145429, -0.07222127,  0.04321999,  0.04673882, -0.1822698 , -0.01274662, -0.02621028,  0.05516759,  0.02776576],
       [ 0.02828105, -0.08578419, -0.03181315,  0.16777049,  0.02880079, -0.22408154,  0.03211648, -0.10179542, -0.0142158 ],
       [-0.00200437, -0.04711348,  0.03852606, -0.04376164, -0.04622416, -0.08771239,  0.01384661, -0.04598728, -0.00923173],
       [ 0.09897362, -0.02017143,  0.01893266, -0.20324325, -0.23269404, -0.057402  ,  0.06639101,  0.11600775,  0.24823044],
       [ 0.04204345,  0.25444987,  0.05124475, -0.01611584,  0.04490149,  0.12646177, -0.17232999,  0.00393661, -0.16978052],
       [ 0.07637359, -0.10224535, -0.02775009, -0.09784421, -0.00277058,  0.08651781,  0.20836008, -0.15342548,  0.0479573 ],
       [-0.13332282,  0.29748621,  0.07167938,  0.00084145,  0.08116574, -0.14710304, -0.01919353,  0.04104313,  0.14378563],
       [-0.16739995, -0.10391723,  0.16058294,  0.02126849, -0.08830528,  0.0993015 ,  0.01851429, -0.02773241, -0.1892801 ],
       [-0.01304632,  0.00152659,  0.0225917 ,  0.03358456,  0.12623971,  0.08311278,  0.00924015,  0.0084943 , -0.01232895]
       ])
a = np.array([0.1, 0.1, 0.1, -0.04, 0.05, -0.03, 0.02, -0.05, 0.1])
b = np.array([1, 0.1, 1, -0.01, 0.01, 0.3, 0.02, -0.05, 0.5])
c = np.array([17, 0, 28, 45, 0, 54, 0, 45, 40])


def trid(X, pj, aj, bj, cj):
    dim = X.shape[0]
    if X.ndim > 1:
        pj = pj.reshape(-1, 1)

    term1 = np.sum((pj*X - 1) ** 2, axis=0)
    term2 = sum((pj[i]*X[i] * pj[i-1]*X[i-1] for i in range(1, dim)))
    f = aj * term1 - bj * term2 + cj
    
    return f


def calculate_trid(X, y_list=Y_input, dim=Y_dim):
    """
    用于计算 Rosenbrock 函数值
    如要批量运算，需要在输入X和输出performs的numpy形式加转置：
        function(X.T, ...) 
        return np.array(performs).T


    Parameters
    ----------
    X : numpy
        需要计算的配方.
    y_list : 字符串列表，形如 ['y1', 'y2', 'y5']
        待计算的性能列表.
    dim : int
        性能维数，对于此Rosenbrock函数应设为9.
        

    Returns
    -------
    performs : list
        输出对应性能值.

    """
    performs=[]
    for y in y_list:
        k = (int)(y[1:])-1
        assert (0 <= k < dim), f"传入函数参数 y{k+1} 超出预定义的 Y 范围 [y1~y{dim}]"            
        performs.append(trid(X, p[k], a[k], b[k], c[k]))
    return np.array(performs)


# 绘图
import matplotlib.pyplot as plt
def plot_fx(xi, yi):
    x_min, x_max = X_ortho_values[xi, 0], X_ortho_values[xi, -1]
    x = np.linspace(x_min, x_max, 100)    # 创建x值数组 (1000个点)    
    base_X = (X_ortho_values.T[-1] - X_ortho_values.T[0]) / 2  # 9维原始数组
    X_matrix = np.tile(base_X, (100, 1))
    X_matrix[:, xi] = x  # 替换第xi列的值（广播机制）
    y = np.array([calculate_trid(X, [f'y{yi+1}']) for X in X_matrix])
    # y = g(x)  # 计算对应的y值
    plt.figure(figsize=(10, 6))  # 创建图形和坐标轴
    
    # 绘制函数曲线
    plt.plot(x, y, 'b-', linewidth=2, label=f'$f(x{xi+1})$')
    # 添加标题和标签
    plt.title('Function Plot: $f(x)$', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('g(x)', fontsize=12)
    # 设置网格和范围
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(x_min, x_max)
    # 添加图例
    plt.legend(loc='best', fontsize=10)
    # # 高亮显示关键点
    # min_idx = np.argmin(y)
    # max_idx = np.argmax(y)
    # plt.plot(x[min_idx], y[min_idx], 'ro', markersize=8, label=f'Min: ({x[min_idx]:.3f}, {y[min_idx]:.3f})')
    # plt.plot(x[max_idx], y[max_idx], 'go', markersize=8, label=f'Max: ({x[max_idx]:.3f}, {y[max_idx]:.3f})')
    # # 添加关键点标注
    # plt.annotate(f'({x[min_idx]:.3f}, {y[min_idx]:.3f})', 
    #               (x[min_idx], y[min_idx]), 
    #               # xytext=(0.2, 0.1), 
    #               arrowprops=dict(arrowstyle='->', connectionstyle='arc3'),
    #               fontsize=10)
    # plt.annotate(f'({x[max_idx]:.3f}, {y[max_idx]:.3f})', 
    #               (x[max_idx], y[max_idx]), 
    #               # xytext=(0.7, 1.0), 
    #               arrowprops=dict(arrowstyle='->', connectionstyle='arc3'),
    #               fontsize=10)
    # 添加水平零点线
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    # 显示图例并调整布局
    plt.legend()
    plt.tight_layout()
    
    plt.show()  # 显示图像
    
    

# 计算测试
if __name__ == "__main__":
    
    for xi in range(9):
        plot_fx(xi, 8)
    
    # X = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    #              [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
    X = np.array(RECIPE)
    # X = np.vstack([X, X])
    # X = np.array([1, 3.0, 15, 1.75, 2.129, 1.1, 0.488, 29.98, 2.5])
    # # 如果从Excel复制字符串（由制表符 \t 分隔的数字）
    # numbers_str = input("输入配方参数集X:")
    # numbers_list = numbers_str.split('\t')
    # X = np.array([float(num) for num in numbers_list])  # 字符串转先浮点数，再转np数组    
    
    y_performances = calculate_trid(X)
    print(f"计算结果：{np.round(y_performances, 4)}")
    print(f"目标差距：{np.round(TARGET - y_performances, 4)}")
    print(f"达标个数：{np.sum(TARGET - y_performances <= 0, axis=-1)}")

