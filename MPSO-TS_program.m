%% 增强型混合PSO-TS算法求解舒伯特函数最小值
% 特点：
% 1. 在个体极值更新时加入变异操作，增强种群多样性
% 2. 在PSO迭代后期嵌入禁忌搜索，进行局部精细搜索
% 3. 动态调整变异概率，前期探索后期收敛

clc; clear; close all;

%% 1. 舒伯特函数定义
function y = shubert(x)
    % 计算舒伯特函数值
    % 输入: x = [x1, x2] 二维向量
    % 输出: 函数值y
    h1 = 0;
    h2 = 0;
    for i = 1:5
        h1 = h1 + i * cos((i + 1) * x(1) + i);
        h2 = h2 + i * cos((i + 1) * x(2) + i);
    end
    y = h1 * h2;
end

%% 2. 个体极值变异操作
function pbest_mutated = mutate_pbest(pbest, pbest_fitness, lb, ub, mutation_rate, iteration, max_iter)
    % 对个体极值实施变异操作
    % 输入:
    %   pbest - 当前个体最优位置
    %   pbest_fitness - 当前个体最优适应度
    %   lb, ub - 搜索边界
    %   mutation_rate - 基础变异概率
    %   iteration - 当前迭代次数
    %   max_iter - 最大迭代次数
    % 输出: 变异后的个体极值
    
    % 动态调整变异强度，前期大变异，后期小变异
    progress = iteration / max_iter;
    adaptive_rate = mutation_rate * (1 - progress * 0.8);
    
    pbest_mutated = pbest;
    
    % 判断是否进行变异
    if rand() < adaptive_rate
        % 随机选择变异类型
        mutation_type = randi(3);
        
        switch mutation_type
            case 1
                % 高斯变异
                sigma = 0.1 * (ub - lb) * (1 - progress);
                pbest_mutated = pbest + sigma * randn(size(pbest));
                
            case 2
                % 均匀变异
                mutation_range = 0.2 * (ub - lb) * (1 - progress);
                pbest_mutated = pbest + mutation_range * (2 * rand(size(pbest)) - 1);
                
            case 3
                % 柯西变异（长尾分布，有利于跳出局部最优）
                cauchy_scale = 0.05 * (ub - lb) * (1 - progress);
                cauchy_sample = cauchyrnd(0, cauchy_scale, size(pbest));
                pbest_mutated = pbest + cauchy_sample;
        end
        
        % 边界处理
        pbest_mutated = min(max(pbest_mutated, lb), ub);
        
        % 评估变异后的适应度
        mutated_fitness = shubert(pbest_mutated);
        
        % 如果变异后适应度变差，有一定概率接受（模拟退火思想）
        if mutated_fitness > pbest_fitness && rand() < 0.3 * (1 - progress)
            pbest_mutated = pbest;  % 保留原解
        end
    end
end

%% 3. 柯西分布随机数生成
function r = cauchyrnd(location, scale, size_array)
    % 生成柯西分布随机数
    u = rand(size_array) - 0.5;
    r = location + scale * tan(pi * u);
end

%% 4. 禁忌搜索局部优化
function [best_solution, best_fitness] = tabu_search_enhanced(initial_solution, initial_fitness, lb, ub, ts_params)
    % 增强禁忌搜索算法
    % 输入:
    %   initial_solution - 初始解
    %   initial_fitness - 初始适应度
    %   lb, ub - 搜索边界
    %   ts_params - 禁忌搜索参数结构体
    % 输出:
    %   best_solution - 最优解
    %   best_fitness - 最优适应度
    
    % 提取参数
    max_iter = ts_params.max_iter;
    tabu_size = ts_params.tabu_size;
    neighbor_size = ts_params.neighbor_size;
    aspiration_criteria = ts_params.aspiration_criteria;
    
    % 初始化
    current_solution = initial_solution;
    current_fitness = initial_fitness;
    best_solution = current_solution;
    best_fitness = current_fitness;
    
    % 动态步长参数
    initial_step = 0.5;
    min_step = 0.01;
    step_decay = 0.95;
    
    % 禁忌表初始化
    tabu_list = zeros(tabu_size, 2);
    tabu_fitness = inf(1, tabu_size);
    tabu_index = 1;
    
    % 禁忌搜索主循环
    for iter = 1:max_iter
        % 动态调整搜索步长
        current_step = max(min_step, initial_step * (step_decay^iter));
        
        % 生成邻域解
        best_candidate = current_solution;
        best_candidate_fitness = current_fitness;
        
        for k = 1:neighbor_size
            % 生成候选解（使用自适应扰动）
            if iter < max_iter/2
                % 前期：较大范围的扰动
                perturbation = current_step * (2 * rand(1, 2) - 1);
            else
                % 后期：较小范围的精细搜索
                perturbation = 0.1 * current_step * randn(1, 2);
            end
            
            candidate = current_solution + perturbation;
            
            % 边界处理
            candidate = min(max(candidate, lb), ub);
            
            % 检查是否在禁忌表中
            is_tabu = false;
            for t = 1:tabu_size
                if ~isinf(tabu_fitness(t)) && all(abs(tabu_list(t, :) - candidate) < 1e-8)
                    is_tabu = true;
                    break;
                end
            end
            
            % 计算候选解适应度
            candidate_fitness = shubert(candidate);
            
            % 如果候选解更优且不是禁忌解，或者满足渴望准则
            if (~is_tabu && candidate_fitness < best_candidate_fitness) || ...
               (candidate_fitness < best_fitness - aspiration_criteria)
                best_candidate = candidate;
                best_candidate_fitness = candidate_fitness;
            end
        end
        
        % 更新当前解
        current_solution = best_candidate;
        current_fitness = best_candidate_fitness;
        
        % 更新禁忌表
        tabu_list(tabu_index, :) = current_solution;
        tabu_fitness(tabu_index) = current_fitness;
        tabu_index = mod(tabu_index, tabu_size) + 1;
        
        % 更新全局最优
        if current_fitness < best_fitness
            best_solution = current_solution;
            best_fitness = current_fitness;
        end
    end
end

%% 5. 增强型混合PSO-TS主算法
function [global_best, global_best_fitness, convergence_curve] = enhanced_pso_ts()
    % 主算法：PSO+变异+TS混合优化
    
    % 算法参数
    dim = 2;                % 问题维度
    pop_size = 30;          % 种群规模
    max_iter = 300;         % 最大迭代次数
    w = 0.9;                % 惯性权重初始值
    w_damp = 0.99;          % 惯性权重衰减系数
    c1 = 2.0;               % 个体学习因子
    c2 = 2.0;               % 社会学习因子
    
    % 变异参数
    mutation_rate = 0.3;    % 变异概率
    ts_start_iter = 200;    % TS开始迭代次数
    
    % 禁忌搜索参数
    ts_params.max_iter = 50;
    ts_params.tabu_size = 15;
    ts_params.neighbor_size = 30;
    ts_params.aspiration_criteria = 1e-3;
    
    % 搜索空间
    lb = -10 * ones(1, dim);
    ub = 10 * ones(1, dim);
    
    % 初始化种群
    pop_pos = lb + (ub - lb) .* rand(pop_size, dim);
    pop_vel = zeros(pop_size, dim);
    pop_fitness = zeros(pop_size, 1);
    
    % 计算初始适应度
    for i = 1:pop_size
        pop_fitness(i) = shubert(pop_pos(i, :));
    end
    
    % 初始化个体最优和全局最优
    pbest_pos = pop_pos;
    pbest_fitness = pop_fitness;
    [global_best_fitness, gbest_idx] = min(pop_fitness);
    global_best = pop_pos(gbest_idx, :);
    
    % 收敛曲线记录
    convergence_curve = zeros(max_iter + ts_params.max_iter, 1);
    
    fprintf('开始增强型混合PSO-TS优化...\n');
    fprintf('总迭代次数: %d (PSO: %d, TS: %d)\n\n', ...
        max_iter + ts_params.max_iter, max_iter, ts_params.max_iter);
    
    %% 阶段1: 增强PSO（带个体极值变异）
    fprintf('=== 阶段1: 增强PSO（带个体极值变异）===\n');
    
    for iter = 1:max_iter
        % 更新惯性权重
        w = w * w_damp;
        
        % 更新每个粒子
        for i = 1:pop_size
            % 更新速度
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            pop_vel(i, :) = w * pop_vel(i, :) + ...
                c1 * r1 .* (pbest_pos(i, :) - pop_pos(i, :)) + ...
                c2 * r2 .* (global_best - pop_pos(i, :));
            
            % 速度边界限制
            vel_max = 0.2 * (ub - lb);
            pop_vel(i, pop_vel(i, :) > vel_max) = vel_max;
            pop_vel(i, pop_vel(i, :) < -vel_max) = -vel_max;
            
            % 更新位置
            pop_pos(i, :) = pop_pos(i, :) + pop_vel(i, :);
            
            % 位置边界处理
            pop_pos(i, pop_pos(i, :) > ub) = ub;
            pop_pos(i, pop_pos(i, :) < lb) = lb;
            
            % 计算新适应度
            new_fitness = shubert(pop_pos(i, :));
            
            % 更新个体最优（带变异操作）
            if new_fitness < pbest_fitness(i)
                pbest_pos(i, :) = pop_pos(i, :);
                pbest_fitness(i) = new_fitness;
            else
                % 对个体极值实施变异
                pbest_pos(i, :) = mutate_pbest(pbest_pos(i, :), pbest_fitness(i), ...
                    lb, ub, mutation_rate, iter, max_iter);
                pbest_fitness(i) = shubert(pbest_pos(i, :));
            end
            
            % 更新全局最优
            if pbest_fitness(i) < global_best_fitness
                global_best = pbest_pos(i, :);
                global_best_fitness = pbest_fitness(i);
            end
        end
        
        % 记录收敛曲线
        convergence_curve(iter) = global_best_fitness;
        
        % 显示进度
        if mod(iter, 50) == 0
            fprintf('PSO迭代 %d/%d: 当前最优 = %.8f\n', ...
                iter, max_iter, global_best_fitness);
        end
    end
    
    fprintf('PSO阶段完成，最优解: [%.6f, %.6f], 适应度: %.6f\n', ...
        global_best(1), global_best(2), global_best_fitness);
    
    %% 阶段2: 禁忌搜索局部优化
    fprintf('\n=== 阶段2: 禁忌搜索局部优化 ===\n');
    
    [global_best, global_best_fitness] = tabu_search_enhanced(...
        global_best, global_best_fitness, lb, ub, ts_params);
    
    % 记录TS阶段的收敛曲线
    for i = 1:ts_params.max_iter
        convergence_curve(max_iter + i) = global_best_fitness;
    end
    
    fprintf('TS阶段完成，最优解: [%.8f, %.8f], 适应度: %.8f\n', ...
        global_best(1), global_best(2), global_best_fitness);
    
    %% 最终结果显示
    fprintf('\n=== 优化结果 ===\n');
    fprintf('最优解: x1 = %.8f, x2 = %.8f\n', global_best(1), global_best(2));
    fprintf('最优适应度: %.8f\n', global_best_fitness);
    fprintf('理论最小值: -186.7309088\n');
    fprintf('误差: %.8f\n\n', abs(global_best_fitness + 186.7309088));
end

%% 6. 主程序
% 运行增强型混合PSO-TS算法
[best_solution, best_fitness, convergence_curve] = enhanced_pso_ts();

%% 7. 结果可视化
% 创建多子图显示结果
figure('Position', [100, 100, 1400, 500]);

% 子图1: 收敛曲线
subplot(1, 3, 1);
plot(convergence_curve, 'b-', 'LineWidth', 2);
hold on;
plot([300, 300], ylim, 'r--', 'LineWidth', 1.5);
xlabel('迭代次数', 'FontSize', 12);
ylabel('最优适应度', 'FontSize', 12);
title('增强PSO-TS收敛曲线', 'FontSize', 14);
legend('适应度', 'PSO/TS切换点', 'Location', 'best');
grid on;
set(gca, 'YScale', 'log');

% 子图2: 函数等高线与最优解
subplot(1, 3, 2);
% 生成网格
[x1_grid, x2_grid] = meshgrid(linspace(-10, 10, 200), linspace(-10, 10, 200));
z_grid = zeros(size(x1_grid));

% 计算网格点函数值
for i = 1:size(x1_grid, 1)
    for j = 1:size(x1_grid, 2)
        z_grid(i, j) = shubert([x1_grid(i, j), x2_grid(j)]);
    end
end

% 绘制等高线
contourf(x1_grid, x2_grid, z_grid, 50, 'LineStyle', 'none');
hold on;
plot(best_solution(1), best_solution(2), 'rp', 'MarkerSize', 20, 'MarkerFaceColor', 'r');
xlabel('x1', 'FontSize', 12);
ylabel('x2', 'FontSize', 12);
title('函数等高线与最优解', 'FontSize', 14);
colorbar;
grid on;

% 子图3: 适应度值分布
subplot(1, 3, 3);
% 生成随机点评估适应度分布
n_samples = 1000;
samples = -10 + 20 * rand(n_samples, 2);
sample_fitness = zeros(n_samples, 1);
for i = 1:n_samples
    sample_fitness(i) = shubert(samples(i, :));
end

histogram(sample_fitness, 30, 'FaceColor', 'g', 'FaceAlpha', 0.7);
hold on;
xline(best_fitness, 'r-', 'LineWidth', 2);
xlabel('适应度值', 'FontSize', 12);
ylabel('频数', 'FontSize', 12);
title('适应度值分布与最优解', 'FontSize', 14);
legend('随机采样分布', '最优适应度', 'Location', 'best');
grid on;

% 显示最优解信息
annotation('textbox', [0.02, 0.02, 0.96, 0.1], ...
    'String', sprintf('最优解: [%.8f, %.8f], 最优适应度: %.8f, 误差: %.8f', ...
    best_solution(1), best_solution(2), best_fitness, abs(best_fitness + 186.7309088)), ...
    'FontSize', 10, 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', 'BackgroundColor', [0.9, 0.9, 0.9]);

%% 8. 三维曲面可视化
figure;
surf(x1_grid, x2_grid, z_grid, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
hold on;
plot3(best_solution(1), best_solution(2), best_fitness, ...
    'ro', 'MarkerSize', 15, 'MarkerFaceColor', 'r', 'LineWidth', 2);
xlabel('x1', 'FontSize', 12);
ylabel('x2', 'FontSize', 12);
zlabel('f(x1, x2)', 'FontSize', 12);
title('舒伯特函数三维曲面与最优解', 'FontSize', 14);
colormap('jet');
colorbar;
view(45, 30);
grid on;
