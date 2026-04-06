% 自适应变异PSO算法求解舒伯特函数最小值
clc; clear; close all;

%% 1. 参数初始化
c1 = 1.4;          % 个体加速度因子
c2 = 1.5;          % 社会加速度因子
maxgen = 500;      % 进化代数（最大迭代次数）
sizepop = 50;      % 种群规模（粒子数量）
w = 0.8;           % 惯性权重
Vmax = 5; Vmin = -5;  % 速度上下界
popmax = 10; popmin = -10;  % 位置（粒子）上下界

% 初始化种群、速度、适应度、收敛曲线
pop = zeros(sizepop, 2);   % 种群（sizepop行，2列，对应x1、x2）
V = zeros(sizepop, 2);     % 速度
fitness = zeros(sizepop, 1); % 每个粒子的适应度
trace = zeros(maxgen, 1);  % 每代最优适应度记录（收敛曲线）

%% 2. 随机初始化种群和速度
for i = 1:sizepop
    pop(i, :) = 5 * rands(1, 2);  % 初始位置（范围：[-5,5]，rands生成[-1,1]随机数）
    V(i, :) = rands(1, 2);        % 初始速度（范围：[-1,1]）
    fitness(i) = funShubert(pop(i, :));  % 计算初始适应度
end

%% 3. 初始化个体极值（Pbest）和群体极值（Gbest）
[bestfitness, bestindex] = min(fitness);  % 找到初始种群中适应度最小的粒子（因求最小值）
Gbest = pop(bestindex, :);       % 全局最优位置
fitnessGbest = bestfitness;      % 全局最优适应度
Pbest = pop;                     % 个体最优位置（初始为自身）
fitnessPbest = fitness;           % 个体最优适应度（初始为自身适应度）

%% 4. 迭代优化（进化过程）
for i = 1:maxgen
    for j = 1:sizepop
        % --------------------- 速度更新 ---------------------
        V(j, :) = w * V(j, :) ...
            + c1 * rand() * (Pbest(j, :) - pop(j, :)) ...  % 个体认知部分
            + c2 * rand() * (Gbest - pop(j, :));          % 社会认知部分
        
        % 速度边界处理（超过Vmax/Vmin则截断）
        V(j, V(j, :) > Vmax) = Vmax;
        V(j, V(j, :) < Vmin) = Vmin;
        
        % --------------------- 位置更新 ---------------------
        pop(j, :) = pop(j, :) + V(j, :);
        
        % 位置边界处理（超过popmax/popmin则截断）
        pop(j, pop(j, :) > popmax) = popmax;
        pop(j, pop(j, :) < popmin) = popmin;
        
        % --------------------- 自适应变异 ---------------------
        if rand() > 0.9  % 10%概率发生变异（随机重置位置）
            pop(j, :) = rands(1, 2);  % 重置为[-1,1]随机数，再乘以5？原代码是rands(1,2)，这里保持原逻辑
        end
        
        % --------------------- 计算新适应度 ---------------------
        fitness(j) = funShubert(pop(j, :));
    end
    
    % --------------------- 个体最优更新 ---------------------
    for j = 1:sizepop
        if fitness(j) < fitnessPbest(j)  % 若当前适应度优于个体历史最优
            Pbest(j, :) = pop(j, :);    % 更新个体最优位置
            fitnessPbest(j) = fitness(j);  % 更新个体最优适应度
        end
    end
    
    % --------------------- 群体最优更新 ---------------------
    for j = 1:sizepop
        if fitness(j) < fitnessGbest  % 若当前适应度优于群体历史最优
            Gbest = pop(j, :);        % 更新群体最优位置
            fitnessGbest = fitness(j);  % 更新群体最优适应度
        end
    end
    
    % --------------------- 记录收敛曲线 ---------------------
    trace(i) = fitnessGbest;
    disp(['第', num2str(i), '代，全局最优位置：', num2str(Gbest), ...
          '，全局最优适应度：', num2str(fitnessGbest)]);
end

%% 5. 结果可视化
figure;
plot(trace, 'LineWidth', 2);
title('最优个体适应度收敛曲线');
xlabel('进化代数');
ylabel('适应度（舒伯特函数值）');
grid on;