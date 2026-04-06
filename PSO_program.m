%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 自适应变异PSO算法（可选择适应度函数：funShubert 或 fun）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% 1. 参数初始化（可根据问题调整）
c1 = 1.4;          % 个体加速度因子
c2 = 1.5;          % 社会加速度因子
maxgen = 500;      % 进化代数（最大迭代次数）
sizepop = 50;      % 种群规模（粒子数量）
w = 0.8;           % 惯性权重
Vmax = 5; Vmin = -5;  % 速度上下界
popmax = 10; popmin = -10;  % 位置（粒子）上下界

% 选择适应度函数：'funShubert' 或 'fun'
fitnessFunc = 'funShubert';  % 可改为 'fun' 测试另一个函数

% 初始化种群、速度、适应度、收敛曲线
pop = zeros(sizepop, 2);   % 种群（sizepop行，2列，对应x1、x2）
V = zeros(sizepop, 2);     % 速度
fitness = zeros(sizepop, 1); % 每个粒子的适应度
trace = zeros(maxgen, 1);  % 每代最优适应度记录（收敛曲线）

%% 2. 随机初始化种群和速度
for i = 1:sizepop
    pop(i, :) = 5 * rands(1, 2);  % 初始位置（范围：[-5,5]，rands生成[-1,1]随机数）
    V(i, :) = rands(1, 2);        % 初始速度（范围：[-1,1]）
    % 计算适应度（根据选择的适应度函数）
    if strcmp(fitnessFunc, 'funShubert')
        fitness(i) = funShubert(pop(i, :));
    else
        fitness(i) = fun(pop(i, :));
    end
end

%% 3. 个体极值和群体极值初始化
[bestfitness, bestindex] = min(fitness);  % 找初始种群中的最优解（最小化问题）
Gbest = pop(bestindex, :);                % 全局最佳位置
fitnessGbest = bestfitness;               % 全局最佳适应度
Pbest = pop;                              % 个体最佳位置（初始为种群）
fitnessPbest = fitness;                   % 个体最佳适应度（初始为种群适应度）

%% 4. 迭代寻优（自适应变异PSO）
for i = 1:maxgen
    for j = 1:sizepop
        % 速度更新
        V(j, :) = w*V(j, :) + c1*rand*(Pbest(j, :) - pop(j, :)) + ...
                  c2*rand*(Gbest - pop(j, :));
        % 速度边界处理
        V(j, V(j, :) > Vmax) = Vmax;
        V(j, V(j, :) < Vmin) = Vmin;
        
        % 位置更新
        pop(j, :) = pop(j, :) + V(j, :);
        % 位置边界处理
        pop(j, pop(j, :) > popmax) = popmax;
        pop(j, pop(j, :) < popmin) = popmin;
        
        % 自适应变异（概率0.1，rand>0.9时变异）
        if rand > 0.9
            pop(j, :) = rands(1, 2);  % 重新随机初始化该粒子
        end
        
        % 计算新适应度
        if strcmp(fitnessFunc, 'funShubert')
            fitness(j) = funShubert(pop(j, :));
        else
            fitness(j) = fun(pop(j, :));
        end
    end
    
    % 个体最优更新（最小化问题，适应度更小则更新）
    for j = 1:sizepop
        if fitness(j) < fitnessPbest(j)
            Pbest(j, :) = pop(j, :);
            fitnessPbest(j) = fitness(j);
        end
    end
    
    % 群体最优更新（最小化问题，适应度更小则更新）
    [current_best, current_index] = min(fitness);
    if current_best < fitnessGbest
        Gbest = pop(current_index, :);
        fitnessGbest = current_best;
    end
    
    % 记录当前代最优适应度
    trace(i) = fitnessGbest;
    % 显示当前最优解
    disp(['第', num2str(i), '代 全局最优位置：', num2str(Gbest), ...
          ' 适应度：', num2str(fitnessGbest)]);
end

%% 5. 结果分析（绘制收敛曲线）
figure;
plot(trace, 'LineWidth', 1.5);
title('PSO算法收敛曲线（适应度函数：' + fitnessFunc + '）');
xlabel('进化代数');
ylabel('最优适应度值');
grid on;