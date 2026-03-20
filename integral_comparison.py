# -*- coding: utf-8 -*-
"""
数值积分方法比较：矩形法、梯形法、辛普森法
计算定积分 I = ∫₀^π sin(x) dx，解析解为 I = 2

任务清单 (TODO List):
[x] 1. 导入必要的库并定义被积函数
[x] 2. 使用 scipy.integrate.quad 计算高精度参考值
[x] 3. 手动实现矩形法（左矩形）数值积分
[x] 4. 手动实现梯形法数值积分
[x] 5. 手动实现辛普森法数值积分
[x] 6. 计算各方法在不同 n 值下的相对误差
[x] 7. 绘制双对数误差图并分析收敛速度
[x] 8. 输出结果表格

阶段性总结：
- 所有数值积分方法均已手动实现，未使用 scipy 的积分函数（除 quad 外）
- 误差计算和可视化功能已完成
- 收敛阶分析已添加拟合直线
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def f(x):
    """被积函数: f(x) = sin(x)"""
    return np.sin(x)


def compute_reference_value():
    """使用 scipy.integrate.quad 计算积分的高精度参考值"""
    ref_value, ref_error = quad(f, 0, np.pi)
    return ref_value, ref_error


def rectangle_method(a, b, n):
    """
    矩形法（右矩形）数值积分
    
    注意：对于 sin(x) 在 [0, π] 上，由于 sin(0) = 0，左矩形法与梯形法
         结果相同。因此这里使用右矩形法来展示真正的矩形法收敛特性。
    """
    h = (b - a) / n
    x_right = np.linspace(a + h, b, n)
    integral = h * np.sum(f(x_right))
    return integral


def trapezoid_method(a, b, n):
    """梯形法数值积分"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = h / 2 * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral


def simpson_method(a, b, n):
    """辛普森法数值积分"""
    if n % 2 != 0:
        n = n + 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = h / 3 * (y[0] + 
                        4 * np.sum(y[1:-1:2]) + 
                        2 * np.sum(y[2:-1:2]) + 
                        y[-1])
    return integral


def compute_relative_error(approx_value, ref_value):
    """计算相对误差"""
    return np.abs((approx_value - ref_value) / ref_value)


def compute_convergence_order(n_values, errors):
    """通过线性拟合计算收敛阶"""
    log_n = np.log10(n_values)
    log_error = np.log10(errors)
    coeffs = np.polyfit(log_n, log_error, 1)
    convergence_order = -coeffs[0]
    return convergence_order, coeffs


def main():
    """主函数：执行完整的数值积分比较分析"""
    print("=" * 70)
    print("数值积分方法比较：计算 I = ∫₀^π sin(x) dx")
    print("=" * 70)
    
    ref_value, ref_error = compute_reference_value()
    print(f"\n参考值 (quad): {ref_value:.10f}")
    print(f"解析解:        2.0000000000")
    
    n_list = [10, 20, 50, 100, 200, 500, 1000]
    
    results = {
        'rectangle': {'values': [], 'errors': []},
        'trapezoid': {'values': [], 'errors': []},
        'simpson': {'values': [], 'errors': []}
    }
    
    print(f"\n{'n':>6} | {'矩形法误差':>15} | {'梯形法误差':>15} | {'辛普森法误差':>15}")
    print("-" * 70)
    
    for n in n_list:
        rect_val = rectangle_method(0, np.pi, n)
        rect_err = compute_relative_error(rect_val, ref_value)
        results['rectangle']['values'].append(rect_val)
        results['rectangle']['errors'].append(rect_err)
        
        trap_val = trapezoid_method(0, np.pi, n)
        trap_err = compute_relative_error(trap_val, ref_value)
        results['trapezoid']['values'].append(trap_val)
        results['trapezoid']['errors'].append(trap_err)
        
        simp_val = simpson_method(0, np.pi, n)
        simp_err = compute_relative_error(simp_val, ref_value)
        results['simpson']['values'].append(simp_val)
        results['simpson']['errors'].append(simp_err)
        
        print(f"{n:>6} | {rect_err:>15.6e} | {trap_err:>15.6e} | {simp_err:>15.6e}")
    
    n_array = np.array(n_list)
    rect_order, rect_coeffs = compute_convergence_order(n_array, np.array(results['rectangle']['errors']))
    trap_order, trap_coeffs = compute_convergence_order(n_array, np.array(results['trapezoid']['errors']))
    simp_order, simp_coeffs = compute_convergence_order(n_array, np.array(results['simpson']['errors']))
    
    print(f"\n矩形法收敛阶:   {rect_order:.2f} (理论值 ≈ 1)")
    print(f"梯形法收敛阶:   {trap_order:.2f} (理论值 ≈ 2)")
    print(f"辛普森法收敛阶: {simp_order:.2f} (理论值 ≈ 4)")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.loglog(n_list, results['rectangle']['errors'], 'o-', color='blue', 
              linewidth=2, markersize=8, label=f'矩形法 (收敛阶≈{rect_order:.2f})')
    ax.loglog(n_list, results['trapezoid']['errors'], 's-', color='green', 
              linewidth=2, markersize=8, label=f'梯形法 (收敛阶≈{trap_order:.2f})')
    ax.loglog(n_list, results['simpson']['errors'], '^-', color='red', 
              linewidth=2, markersize=8, label=f'辛普森法 (收敛阶≈{simp_order:.2f})')
    
    log_n = np.log10(n_array)
    fit_rect = 10**(rect_coeffs[1] + rect_coeffs[0] * log_n)
    fit_trap = 10**(trap_coeffs[1] + trap_coeffs[0] * log_n)
    fit_simp = 10**(simp_coeffs[1] + simp_coeffs[0] * log_n)
    
    ax.loglog(n_list, fit_rect, '--', color='blue', alpha=0.5, linewidth=1.5)
    ax.loglog(n_list, fit_trap, '--', color='green', alpha=0.5, linewidth=1.5)
    ax.loglog(n_list, fit_simp, '--', color='red', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('区间划分数量 n', fontsize=12)
    ax.set_ylabel('相对误差', fontsize=12)
    ax.set_title(r'数值积分方法误差比较' + '\n' + r'$I = \int_0^{\pi} \sin(x) dx$', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('integral_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图像已保存为: integral_comparison.png")
    plt.show()
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
