% main.m
% 算法与测试函数综合测试框架
clc; clear all; close all;
rng(42)
% ========================================================
% 1. 控制开关设置 (在这里进行切换)
% ========================================================
% 函数选择开关: 1 = Ackley 函数, 2 = Shubert 函数
funcChoice = 2; 

% 运行模式开关: 
% 1 = 仅运行 PSO
% 2 = 仅运行 IPSO
% 3 = 仅运行 TabuSearch
% 4 = 三个算法同时运行并绘制对比图
runMode = 4;

% ========================================================
% 2. 基础参数与测试函数动态初始化
% ========================================================
noP = 30;             % 种群规模 / 邻域解数量
Max_iteration = 1000;  % 最大迭代次数

% 根据选择动态配置函数维度和边界
switch funcChoice
    case 1
        disp('---- 当前测试函数: Ackley ----');
        dim = 20;          % Ackley 通常测试较高维度 (可根据需要修改)
        lb = -32.768;      % 标准 Ackley 下界
        ub = 32.768;       % 标准 Ackley 上界
        fobj = @ackley_func;
    case 2
        disp('---- 当前测试函数: Shubert ----');
        dim = 10;           % Shubert 函数标准测试为 2 维
        lb = -10;          % 标准 Shubert 下界
        ub = 10;           % 标准 Shubert 上界
        fobj = @shubert_func;
    otherwise
        error('无效的函数选择！');
end

% ========================================================
% 3. 算法执行与结果绘图
% ========================================================
figure('Name', '收敛曲线对比', 'Color', 'w');
hold on; grid on;

switch runMode
    case 1
        disp('正在运行: 标准 PSO...');
        [bestScore, bestPos, trace_PSO] = PSO(noP, Max_iteration, lb, ub, dim, fobj);
        plot(trace_PSO, 'LineWidth', 1.5, 'Color', 'b', 'DisplayName', 'PSO');
        disp(['最优适应度: ', num2str(bestScore)]);
        
    case 2
        disp('正在运行: 改进 IPSO...');
        [bestScore, bestPos, trace_IPSO] = IPSO(noP, Max_iteration, lb, ub, dim, fobj);
        plot(trace_IPSO, 'LineWidth', 1.5, 'Color', 'r', 'DisplayName', 'IPSO');
        disp(['最优适应度: ', num2str(bestScore)]);
        
    case 3
        disp('正在运行: 连续型 Tabu Search...');
        [bestScore, bestPos, trace_TS] = TabuSearch(noP, Max_iteration, lb, ub, dim, fobj);
        plot(trace_TS, 'LineWidth', 1.5, 'Color', 'g', 'DisplayName', 'TabuSearch');
        disp(['最优适应度: ', num2str(bestScore)]);
        
    case 4
        disp('正在运行: PSO, IPSO, TabuSearch 综合对比...');
        
        % 运行三个算法
        [score1, pos1, trace_PSO]  = PSO(noP, Max_iteration, lb, ub, dim, fobj);
        [score2, pos2, trace_IPSO] = IPSO(noP, Max_iteration, lb, ub, dim, fobj);
        [score3, pos3, trace_TS]   = TabuSearch(noP, Max_iteration, lb, ub, dim, fobj);
        
        % 绘制三条收敛曲线
        plot(trace_PSO,  'LineWidth', 1.5, 'Color', 'b', 'LineStyle', '--', 'DisplayName', '标准 PSO');
        plot(trace_IPSO, 'LineWidth', 1.5, 'Color', 'r', 'LineStyle', '-',  'DisplayName', '改进 IPSO');
        plot(trace_TS,   'LineWidth', 1.5, 'Color', 'g', 'LineStyle', '-.', 'DisplayName', '连续型 TS');
        
        % 打印对比结果
        disp('---- 最终运行结果对比 ----');
        disp(['PSO        最优值: ', num2str(score1)]);
        disp(['IPSO       最优值: ', num2str(score2)]);
        disp(['TabuSearch 最优值: ', num2str(score3)]);
end

% 统一图表格式设置
title('算法收敛曲线对比', 'FontSize', 14);
xlabel('进化代数 (Iterations)', 'FontSize', 12);
ylabel('最优适应度 (Best Fitness)', 'FontSize', 12);
legend('Location', 'northeast', 'FontSize', 11);
set(gca, 'YScale', 'log'); % 使用对数坐标轴更容易观察后期精度，如果报错可注释掉这句
% 保存为 600 DPI 的 PNG 图片
% print(gcf, 'Shubert convergence_curve.png', '-dpng', '-r600');
print(gcf, 'Ackley convergence_curve.png', '-dpng', '-r600');


% ========================================================
% 4. 目标函数定义区 (Local Functions)
% ========================================================

% --- Ackley 函数定义 ---
% 全局最小值: f(0,0,...,0) = 0
function z = ackley_func(x)
    d = length(x);
    sum1 = sum(x.^2);
    sum2 = sum(cos(2*pi.*x));
    
    term1 = -20 * exp(-0.2 * sqrt(sum1 / d));
    term2 = -exp(sum2 / d);
    
    z = term1 + term2 + 20 + exp(1);
end

% --- Shubert 函数定义 ---
% 全局最小值: -186.7309 (对于2维)
function z = shubert_func(x)
    z = 1;
    for i = 1:length(x)
        s = 0;
        for j = 1:5
            s = s + j * cos((j + 1) * x(i) + j);
        end
        z = z * s;
    end
end