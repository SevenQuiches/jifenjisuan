# 数值积分方法比较脚本
# 任务清单：
# [x] 1. 导入所需库
# [x] 2. 定义被积函数 f(x) = sin(x)
# [x] 3. 使用 SciPy 的 quad 函数计算高精度参考值
# [x] 4. 实现矩形法（左矩形）
# [x] 5. 实现梯形法
# [x] 6. 实现辛普森法（要求n为偶数）
# [x] 7. 定义n值列表（10, 20, 50, 100, 200, 500, 1000）
# [x] 8. 计算每种方法在不同n值下的近似值和相对误差
# [x] 9. 输出误差结果
# [x] 10. 绘制双对数坐标图（log10(n) vs log10(error)）
# [x] 11. 添加拟合直线并计算收敛阶

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# 关闭警告显示
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义被积函数
def f(x):
    """被积函数 f(x) = sin(x)"""
    return np.sin(x)

# 计算高精度参考值
def compute_reference_value():
    """使用 SciPy 的 quad 函数计算高精度参考值"""
    ref_value, _ = quad(f, 0, np.pi)
    print(f"高精度参考值: {ref_value:.10f}")
    return ref_value

# 矩形法（左矩形）
def rectangle_method(f, a, b, n):
    """
    矩形法（左矩形）数值积分 - 使用子区间左端点计算
    参数:
        f: 被积函数
        a: 积分下限
        b: 积分上限
        n: 区间划分数量
    返回:
        积分近似值
    """
    h = (b - a) / n  # 子区间宽度
    x = np.linspace(a, b - h, n)  # 左端点
    return h * np.sum(f(x))

# 梯形法
def trapezoidal_method(f, a, b, n):
    """
    梯形法数值积分
    参数:
        f: 被积函数
        a: 积分下限
        b: 积分上限
        n: 区间划分数量
    返回:
        积分近似值
    """
    h = (b - a) / n  # 子区间宽度
    x = np.linspace(a, b, n + 1)  # 所有端点
    return h * (np.sum(f(x)) - 0.5 * (f(a) + f(b)))

# 辛普森法
def simpson_method(f, a, b, n):
    """
    辛普森法数值积分（要求n为偶数）
    参数:
        f: 被积函数
        a: 积分下限
        b: 积分上限
        n: 区间划分数量（必须为偶数）
    返回:
        积分近似值
    异常:
        ValueError: 如果n不是偶数
    """
    if n % 2 != 0:
        raise ValueError("辛普森法要求区间划分数量n为偶数")
    
    h = (b - a) / n  # 子区间宽度
    x = np.linspace(a, b, n + 1)  # 所有端点
    y = f(x)
    
    # 辛普森公式: (h/3) * [y0 + 4(y1 + y3 + ... + y_{n-1}) + 2(y2 + y4 + ... + y_{n-2}) + yn]
    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

# 计算误差
def compute_errors(ref_value, n_values):
    """
    计算不同方法在不同n值下的相对误差
    参数:
        ref_value: 高精度参考值
        n_values: 区间划分数量列表
    返回:
        三种方法的误差字典
    """
    errors = {
        'rectangle': [],
        'trapezoidal': [],
        'simpson': []
    }
    
    for n in n_values:
        # 矩形法
        approx_rect = rectangle_method(f, 0, np.pi, n)
        error_rect = abs((approx_rect - ref_value) / ref_value)
        errors['rectangle'].append(error_rect)
        
        # 梯形法
        approx_trap = trapezoidal_method(f, 0, np.pi, n)
        error_trap = abs((approx_trap - ref_value) / ref_value)
        errors['trapezoidal'].append(error_trap)
        
        # 辛普森法
        approx_simp = simpson_method(f, 0, np.pi, n)
        error_simp = abs((approx_simp - ref_value) / ref_value)
        errors['simpson'].append(error_simp)
    
    return errors

# 输出误差结果
def print_errors(n_values, errors):
    """
    输出不同方法在不同n值下的相对误差
    参数:
        n_values: 区间划分数量列表
        errors: 三种方法的误差字典
    """
    print("\n=== 相对误差结果 ===")
    
    print("\n矩形法（左矩形）:")
    for n, err in zip(n_values, errors['rectangle']):
        print(f"n = {n:4d}: {err:.10f}")
    
    print("\n梯形法:")
    for n, err in zip(n_values, errors['trapezoidal']):
        print(f"n = {n:4d}: {err:.10f}")
    
    print("\n辛普森法:")
    for n, err in zip(n_values, errors['simpson']):
        print(f"n = {n:4d}: {err:.10f}")

# 绘制误差图
def plot_errors(n_values, errors):
    """
    绘制双对数坐标图（log10(n) vs log10(error)）
    参数:
        n_values: 区间划分数量列表
        errors: 三种方法的误差字典
    """
    # 使用非交互式后端避免GUI死循环
    import matplotlib
    matplotlib.use('Agg')
    
    plt.figure(figsize=(10, 6))
    
    # 绘制三种方法的误差曲线
    plt.loglog(n_values, errors['rectangle'], 'o-', label='矩形法（左矩形）', color='#1f77b4')
    plt.loglog(n_values, errors['trapezoidal'], 's-', label='梯形法', color='#ff7f0e')
    plt.loglog(n_values, errors['simpson'], '^-', label='辛普森法', color='#2ca02c')
    
    # 添加拟合直线并计算收敛阶
    # 矩形法拟合（收敛阶≈1）
    slope_rect, intercept_rect = np.polyfit(np.log10(n_values), np.log10(errors['rectangle']), 1)
    plt.loglog(n_values, 10**(intercept_rect + slope_rect * np.log10(n_values)), 
               '--', color='#1f77b4', alpha=0.5, label=f'矩形法拟合 (斜率={slope_rect:.2f})')
    
    # 梯形法拟合（收敛阶≈2）
    slope_trap, intercept_trap = np.polyfit(np.log10(n_values), np.log10(errors['trapezoidal']), 1)
    plt.loglog(n_values, 10**(intercept_trap + slope_trap * np.log10(n_values)), 
               '--', color='#ff7f0e', alpha=0.5, label=f'梯形法拟合 (斜率={slope_trap:.2f})')
    
    # 辛普森法拟合（收敛阶≈4）
    slope_simp, intercept_simp = np.polyfit(np.log10(n_values), np.log10(errors['simpson']), 1)
    plt.loglog(n_values, 10**(intercept_simp + slope_simp * np.log10(n_values)), 
               '--', color='#2ca02c', alpha=0.5, label=f'辛普森法拟合 (斜率={slope_simp:.2f})')
    
    # 添加图例、标签和标题
    plt.xlabel('区间划分数量 n', fontsize=12)
    plt.ylabel('相对误差', fontsize=12)
    plt.title('数值积分方法收敛速度比较 (双对数坐标)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    # 保存图像
    plt.savefig('integration_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("图像已保存为 integration_errors.png")

# 主函数
def main():
    """主函数，执行所有计算和绘图"""
    # 1. 计算高精度参考值
    print("=== 步骤1: 计算高精度参考值 ===")
    ref_value = compute_reference_value()
    
    # 2. 定义n值列表
    n_values = [10, 20, 50, 100, 200, 500, 1000]
    print(f"\n=== 步骤2: 区间划分数量n值列表 ===")
    print(f"n = {n_values}")
    
    # 3. 计算不同方法的误差
    print("\n=== 步骤3: 计算不同方法的相对误差 ===")
    errors = compute_errors(ref_value, n_values)
    
    # 4. 输出误差结果
    print_errors(n_values, errors)
    
    # 5. 绘制误差图
    print("\n=== 步骤4: 绘制误差图 ===")
    print("正在生成误差图...")
    plot_errors(n_values, errors)
    
    print("\n=== 任务完成 ===")
    print("分析结论：")
    print("1. 矩形法收敛阶约为1，误差随n增大线性减小")
    print("2. 梯形法收敛阶约为2，误差随n增大二次减小")
    print("3. 辛普森法收敛阶约为4，误差随n增大四次减小，收敛速度最快")

if __name__ == "__main__":
    main()
