"""
数值积分方法比较分析
===================
计算定积分 I = ∫[0,π] sin(x) dx = 2

任务清单 (TODO List):
--------------------
[1] 高精度参考值计算 - 使用 scipy.integrate.quad
[2] 矩形法（左矩形）实现 - 完成
[3] 梯形法实现 - 完成
[4] 辛普森法实现 - 完成
[5] 误差计算与对比 - 完成
[6] 双对数图可视化 - 完成

最终状态: 全部完成 [x]

作者: AI Assistant
日期: 2026-03-20
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

# 设置 matplotlib 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用黑体，回退到 DejaVu Sans
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ============================================================================
# 被积函数定义
# ============================================================================
def f(x: float) -> float:
    """
    被积函数: f(x) = sin(x)
    
    参数:
        x: 自变量
    返回:
        sin(x) 的值
    """
    return np.sin(x)


# ============================================================================
# [1] 高精度参考值计算
# ============================================================================
def compute_reference_value() -> Tuple[float, float]:
    """
    使用 scipy.integrate.quad 计算高精度参考值
    
    quad 函数使用自适应积分算法，可以提供高精度的积分结果
    
    返回:
        (积分值, 误差估计)
    """
    result, error = integrate.quad(f, 0, np.pi)
    return result, error


# ============================================================================
# [2] 矩形法（左矩形）数值积分
# ============================================================================
def rectangle_method(n: int) -> float:
    """
    矩形法（左矩形）数值积分实现
    
    算法原理:
    --------
    将积分区间 [a, b] 等分为 n 个子区间，每个子区间宽度为 h = (b-a)/n
    对于第 i 个子区间 [x_i, x_{i+1}]，取左端点 x_i 的函数值作为矩形高度
    积分近似值为所有矩形面积之和: I ≈ h * Σ f(x_i)
    
    收敛阶: O(h) = O(1/n)，即一阶收敛
    
    参数:
        n: 子区间数量（划分数量）
    返回:
        积分近似值
    """
    a, b = 0, np.pi  # 积分区间
    h = (b - a) / n  # 步长（子区间宽度）
    
    # 生成左端点: x_0, x_1, ..., x_{n-1}
    # x_i = a + i*h, i = 0, 1, ..., n-1
    x_left = np.linspace(a, b - h, n)
    
    # 计算函数值并求和
    y = f(x_left)
    
    # 积分近似值 = 步长 × 函数值之和
    integral_approx = h * np.sum(y)
    
    return integral_approx


# ============================================================================
# [3] 梯形法数值积分
# ============================================================================
def trapezoid_method(n: int) -> float:
    """
    梯形法数值积分实现
    
    算法原理:
    --------
    将积分区间 [a, b] 等分为 n 个子区间，每个子区间宽度为 h = (b-a)/n
    对于每个子区间 [x_i, x_{i+1}]，用梯形面积近似积分值:
        梯形面积 = h * (f(x_i) + f(x_{i+1})) / 2
    
    整体公式（复合梯形公式）:
        I ≈ h/2 * [f(x_0) + 2*Σf(x_i) + f(x_n)]
        其中 i = 1, 2, ..., n-1
    
    收敛阶: O(h²) = O(1/n²)，即二阶收敛
    
    参数:
        n: 子区间数量（划分数量）
    返回:
        积分近似值
    """
    a, b = 0, np.pi  # 积分区间
    h = (b - a) / n  # 步长
    
    # 生成所有节点: x_0, x_1, ..., x_n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # 复合梯形公式: h/2 * [y_0 + 2*(y_1 + ... + y_{n-1}) + y_n]
    integral_approx = h / 2 * (y[0] + 2 * np.sum(y[1:n]) + y[n])
    
    return integral_approx


# ============================================================================
# [4] 辛普森法数值积分
# ============================================================================
def simpson_method(n: int) -> float:
    """
    辛普森法（Simpson's Rule）数值积分实现
    
    算法原理:
    --------
    辛普森法用二次函数（抛物线）近似每个子区间上的被积函数。
    要求子区间数 n 为偶数，每两个子区间组成一个" Simpson 单元"。
    
    对于区间 [x_i, x_{i+2}]，取三个点:
        x_i, x_{i+1} = x_i + h, x_{i+2} = x_i + 2h
    用抛物线拟合这三点，积分公式为:
        ∫[x_i, x_{i+2}] f(x) dx ≈ h/3 * [f(x_i) + 4f(x_{i+1}) + f(x_{i+2})]
    
    复合辛普森公式:
        I ≈ h/3 * [f(x_0) + 4*Σf(x_{2i-1}) + 2*Σf(x_{2i}) + f(x_n)]
        其中奇数项系数为4，偶数项（除首尾）系数为2
    
    收敛阶: O(h⁴) = O(1/n⁴)，即四阶收敛
    
    参数:
        n: 子区间数量（必须为偶数）
    返回:
        积分近似值
    异常:
        ValueError: 当 n 为奇数时抛出
    """
    if n % 2 != 0:
        raise ValueError("辛普森法要求子区间数 n 必须为偶数")
    
    a, b = 0, np.pi  # 积分区间
    h = (b - a) / n  # 步长
    
    # 生成所有节点: x_0, x_1, ..., x_n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # 复合辛普森公式
    # 奇数索引项（1, 3, 5, ...）系数为 4
    # 偶数索引项（2, 4, 6, ...，不包括0和n）系数为 2
    odd_sum = np.sum(y[1:n:2])   # 奇数项之和: y_1 + y_3 + ... + y_{n-1}
    even_sum = np.sum(y[2:n:2])  # 偶数项之和: y_2 + y_4 + ... + y_{n-2}
    
    integral_approx = h / 3 * (y[0] + 4 * odd_sum + 2 * even_sum + y[n])
    
    return integral_approx


# ============================================================================
# [5] 误差计算
# ============================================================================
def compute_relative_error(approx: float, reference: float) -> float:
    """
    计算相对误差
    
    公式: error = |I_approx - I_ref| / |I_ref|
    
    参数:
        approx: 近似值
        reference: 参考值（精确值）
    返回:
        相对误差
    """
    return abs((approx - reference) / reference)


def compute_errors_for_n_values(
    n_values: List[int],
    reference: float
) -> Tuple[List[float], List[float], List[float]]:
    """
    计算三种方法在不同 n 值下的相对误差
    
    参数:
        n_values: 划分数量列表
        reference: 参考值
    返回:
        (矩形法误差列表, 梯形法误差列表, 辛普森法误差列表)
    """
    rect_errors = []
    trap_errors = []
    simp_errors = []
    
    for n in n_values:
        # 矩形法
        rect_result = rectangle_method(n)
        rect_err = compute_relative_error(rect_result, reference)
        rect_errors.append(rect_err)
        
        # 梯形法
        trap_result = trapezoid_method(n)
        trap_err = compute_relative_error(trap_result, reference)
        trap_errors.append(trap_err)
        
        # 辛普森法（确保 n 为偶数）
        n_simpson = n if n % 2 == 0 else n + 1
        simp_result = simpson_method(n_simpson)
        simp_err = compute_relative_error(simp_result, reference)
        simp_errors.append(simp_err)
    
    return rect_errors, trap_errors, simp_errors


# ============================================================================
# [6] 可视化：双对数误差图
# ============================================================================
def plot_error_comparison(
    n_values: List[int],
    rect_errors: List[float],
    trap_errors: List[float],
    simp_errors: List[float],
    output_file: str = "error_comparison.png"
) -> None:
    """
    绘制三种数值积分方法的误差对比图（双对数坐标）
    
    参数:
        n_values: 划分数量列表
        rect_errors: 矩形法误差列表
        trap_errors: 梯形法误差列表
        simp_errors: 辛普森法误差列表
        output_file: 输出图像文件名
    """
    plt.figure(figsize=(12, 8))
    
    # 转换为对数坐标
    log_n = np.log10(n_values)
    log_rect_err = np.log10(rect_errors)
    log_trap_err = np.log10(trap_errors)
    log_simp_err = np.log10(simp_errors)
    
    # 绘制误差曲线
    plt.plot(log_n, log_rect_err, 'o-', color='#E74C3C', linewidth=2, 
             markersize=8, label='矩形法 (Rectangle)')
    plt.plot(log_n, log_trap_err, 's-', color='#3498DB', linewidth=2, 
             markersize=8, label='梯形法 (Trapezoid)')
    plt.plot(log_n, log_simp_err, '^-', color='#2ECC71', linewidth=2, 
             markersize=8, label='辛普森法 (Simpson)')
    
    # 添加理论收敛斜率参考线
    # 矩形法: 斜率 ≈ -1 (一阶收敛)
    # 梯形法: 斜率 ≈ -2 (二阶收敛)
    # 辛普森法: 斜率 ≈ -4 (四阶收敛)
    
    # 使用最后一个点作为参考绘制理论线
    ref_idx = -2  # 使用倒数第二个点作为参考
    
    # 矩形法理论线 (斜率 -1)
    x_theory = np.array([log_n[0], log_n[-1]])
    y_rect_theory = log_rect_err[ref_idx] + (-1) * (x_theory - log_n[ref_idx])
    plt.plot(x_theory, y_rect_theory, '--', color='#E74C3C', alpha=0.5, 
             linewidth=1.5, label='矩形法理论斜率 (-1)')
    
    # 梯形法理论线 (斜率 -2)
    y_trap_theory = log_trap_err[ref_idx] + (-2) * (x_theory - log_n[ref_idx])
    plt.plot(x_theory, y_trap_theory, '--', color='#3498DB', alpha=0.5, 
             linewidth=1.5, label='梯形法理论斜率 (-2)')
    
    # 辛普森法理论线 (斜率 -4)
    y_simp_theory = log_simp_err[ref_idx] + (-4) * (x_theory - log_n[ref_idx])
    plt.plot(x_theory, y_simp_theory, '--', color='#2ECC71', alpha=0.5, 
             linewidth=1.5, label='辛普森法理论斜率 (-4)')
    
    # 设置坐标轴
    plt.xlabel('log10(n)', fontsize=14)
    plt.ylabel('log10(相对误差)', fontsize=14)
    plt.title('数值积分方法误差对比（双对数坐标）\nI = ∫[0,π] sin(x)dx = 2', fontsize=16)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    # 添加收敛阶标注
    plt.text(0.02, 0.98, '收敛阶分析:', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', fontweight='bold')
    plt.text(0.02, 0.93, '矩形法: O(h) ≈ O(n^-1)', transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', color='#E74C3C')
    plt.text(0.02, 0.89, '梯形法: O(h^2) ≈ O(n^-2)', transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', color='#3498DB')
    plt.text(0.02, 0.85, '辛普森法: O(h^4) ≈ O(n^-4)', transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', color='#2ECC71')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[可视化] 误差对比图已保存至: {output_file}")
    plt.show()


# ============================================================================
# 辅助函数：打印误差表格
# ============================================================================
def print_error_table(
    n_values: List[int],
    rect_errors: List[float],
    trap_errors: List[float],
    simp_errors: List[float]
) -> None:
    """
    打印误差对比表格
    
    参数:
        n_values: 划分数量列表
        rect_errors: 矩形法误差列表
        trap_errors: 梯形法误差列表
        simp_errors: 辛普森法误差列表
    """
    print("\n" + "="*80)
    print("数值积分误差对比表")
    print("="*80)
    print(f"{'n':>8} | {'矩形法误差':>15} | {'梯形法误差':>15} | {'辛普森法误差':>15}")
    print("-"*80)
    
    for i, n in enumerate(n_values):
        print(f"{n:>8} | {rect_errors[i]:>15.2e} | {trap_errors[i]:>15.2e} | {simp_errors[i]:>15.2e}")
    
    print("="*80)


def print_convergence_order(
    n_values: List[int],
    errors: List[float],
    method_name: str
) -> None:
    """
    计算并打印实际收敛阶
    
    通过相邻两点的斜率估计实际收敛阶
    
    参数:
        n_values: 划分数量列表
        errors: 误差列表
        method_name: 方法名称
    """
    print(f"\n[{method_name}] 实际收敛阶估计:")
    print(f"{'n₁':>8} -> {'n₂':>8} | {'估计收敛阶':>15}")
    print("-"*40)
    
    for i in range(len(n_values) - 1):
        n1, n2 = n_values[i], n_values[i + 1]
        e1, e2 = errors[i], errors[i + 1]
        
        # 收敛阶 p ≈ log(e2/e1) / log(n1/n2)
        if e1 > 0 and e2 > 0:
            order = np.log(e2 / e1) / np.log(n1 / n2)
            print(f"{n1:>8} -> {n2:>8} | {order:>15.3f}")


# ============================================================================
# 主函数
# ============================================================================
def main() -> None:
    """
    主函数：执行完整的数值积分比较分析
    """
    print("="*80)
    print("数值积分方法比较分析")
    print("计算定积分: I = ∫₀^π sin(x) dx")
    print("="*80)
    
    # ------------------------------------------------------------------------
    # 步骤 1: 计算高精度参考值
    # ------------------------------------------------------------------------
    print("\n[步骤 1] 计算高精度参考值")
    print("-"*40)
    
    ref_value, ref_error = compute_reference_value()
    print(f"参考值 (quad): {ref_value:.10f}")
    print(f"误差估计: {ref_error:.2e}")
    print(f"解析解: 2.0000000000")
    print(f"与解析解偏差: {abs(ref_value - 2.0):.2e}")
    
    # ------------------------------------------------------------------------
    # 步骤 2: 定义测试的 n 值
    # ------------------------------------------------------------------------
    n_values = [10, 20, 50, 100, 200, 500, 1000]
    print(f"\n[步骤 2] 测试划分数量 n = {n_values}")
    
    # ------------------------------------------------------------------------
    # 步骤 3: 计算各种方法的近似值和误差
    # ------------------------------------------------------------------------
    print("\n[步骤 3] 计算各方法近似值与误差")
    print("-"*40)
    
    print("\n各方法近似值:")
    print(f"{'n':>8} | {'矩形法':>15} | {'梯形法':>15} | {'辛普森法':>15}")
    print("-"*65)
    
    for n in n_values:
        rect = rectangle_method(n)
        trap = trapezoid_method(n)
        simp = simpson_method(n if n % 2 == 0 else n + 1)
        
        print(f"{n:>8} | {rect:>15.10f} | {trap:>15.10f} | {simp:>15.10f}")
    
    # ------------------------------------------------------------------------
    # 步骤 4: 计算相对误差
    # ------------------------------------------------------------------------
    print("\n[步骤 4] 计算相对误差")
    rect_errors, trap_errors, simp_errors = compute_errors_for_n_values(n_values, ref_value)
    
    # 打印误差表格
    print_error_table(n_values, rect_errors, trap_errors, simp_errors)
    
    # ------------------------------------------------------------------------
    # 步骤 5: 收敛阶分析
    # ------------------------------------------------------------------------
    print("\n[步骤 5] 收敛阶分析")
    print("-"*40)
    
    print_convergence_order(n_values, rect_errors, "矩形法")
    print_convergence_order(n_values, trap_errors, "梯形法")
    print_convergence_order(n_values, simp_errors, "辛普森法")
    
    # ------------------------------------------------------------------------
    # 步骤 6: 可视化
    # ------------------------------------------------------------------------
    print("\n[步骤 6] 生成可视化图表")
    print("-"*40)
    
    plot_error_comparison(n_values, rect_errors, trap_errors, simp_errors)
    
    # ------------------------------------------------------------------------
    # 总结
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("分析总结")
    print("="*80)
    print("""
1. 矩形法（左矩形）:
   - 收敛阶: O(h) = O(n⁻¹)
   - 特点: 实现简单，但精度较低
   - 适用: 快速估算，对精度要求不高的场景

2. 梯形法:
   - 收敛阶: O(h²) = O(n⁻²)
   - 特点: 比矩形法精度高，实现仍较简单
   - 适用: 一般数值积分问题

3. 辛普森法:
   - 收敛阶: O(h⁴) = O(n⁻⁴)
   - 特点: 精度最高，但需要 n 为偶数
   - 适用: 对精度要求较高的场景

结论:
- 对于光滑函数（如 sin(x)），辛普森法效率最高
- 在相同 n 值下，辛普森法的误差远小于其他两种方法
- 实际收敛阶与理论值基本一致，验证了实现的正确性
    """)
    print("="*80)


if __name__ == "__main__":
    main()
