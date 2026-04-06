function [best_solution, best_fitness, convergence_curve] = tabu_search_shubert()
    % 禁忌搜索算法求解舒伯特函数最小值
    % 舒伯特函数：f(x1, x2) = sum_{i=1}^{5} i*cos((i+1)*x1+i) * sum_{i=1}^{5} i*cos((i+1)*x2+i)
    % 在定义域[-10, 10]^2上有多个局部极小点
    
    % 清空工作区
    clc; clear; close all;
    
    % 参数设置
    max_iter = 1000;          % 最大迭代次数
    tabu_size = 20;          % 禁忌表大小
    neighbor_size = 50;      % 邻域大小
    step_size = 0.1;         % 邻域搜索步长
    aspiration_criteria = 5; % 渴望水平（允许禁忌移动的条件）
    
    % 定义域
    lower_bound = -10;
    upper_bound = 10;
    
    % 初始化
    current_solution = lower_bound + (upper_bound - lower_bound) * rand(1, 2);
    current_fitness = shubert(current_solution);
    
    best_solution = current_solution;
    best_fitness = current_fitness;
    
    % 禁忌表初始化
    tabu_list = zeros(tabu_size, 2);
    tabu_fitness = inf(1, tabu_size);
    tabu_index = 1;
    
    % 收敛曲线
    convergence_curve = zeros(max_iter, 1);
    
    disp('开始禁忌搜索...');
    fprintf('初始解: [%.4f, %.4f], 适应度: %.4f\n', ...
        current_solution(1), current_solution(2), current_fitness);
    
    % 禁忌搜索主循环
    for iter = 1:max_iter
        % 生成邻域解
        neighbors = zeros(neighbor_size, 2);
        neighbor_fitness = zeros(neighbor_size, 1);
        
        for k = 1:neighbor_size
            % 在当前位置附近生成随机扰动
            perturbation = step_size * (2 * rand(1, 2) - 1);
            candidate = current_solution + perturbation;
            
            % 边界处理
            candidate = max(candidate, lower_bound);
            candidate = min(candidate, upper_bound);
            
            neighbors(k, :) = candidate;
            neighbor_fitness(k) = shubert(candidate);
        end
        
        % 寻找最佳候选解（考虑禁忌约束）
        best_candidate_fitness = inf;
        best_candidate_idx = 1;
        best_candidate = neighbors(1, :);
        
        for k = 1:neighbor_size
            candidate_fitness = neighbor_fitness(k);
            candidate = neighbors(k, :);
            
            % 检查是否在禁忌表中
            is_tabu = false;
            for t = 1:tabu_size
                if ~isinf(tabu_fitness(t)) && all(abs(tabu_list(t, :) - candidate) < 1e-6)
                    is_tabu = true;
                    break;
                end
            end
            
            % 评估候选解
            if candidate_fitness < best_candidate_fitness
                % 如果不是禁忌解，或者满足渴望准则
                if ~is_tabu || candidate_fitness < best_fitness - aspiration_criteria
                    best_candidate_fitness = candidate_fitness;
                    best_candidate_idx = k;
                    best_candidate = candidate;
                end
            end
        end
        
        % 更新当前解
        current_solution = best_candidate;
        current_fitness = best_candidate_fitness;
        
        % 更新禁忌表
        tabu_list(tabu_index, :) = best_candidate;
        tabu_fitness(tabu_index) = best_candidate_fitness;
        tabu_index = mod(tabu_index, tabu_size) + 1;
        
        % 更新全局最优解
        if current_fitness < best_fitness
            best_solution = current_solution;
            best_fitness = current_fitness;
            
            fprintf('迭代 %d: 找到新最优解: [%.6f, %.6f], 适应度: %.6f\n', ...
                iter, best_solution(1), best_solution(2), best_fitness);
        end
        
        % 记录收敛曲线
        convergence_curve(iter) = best_fitness;
        
        % 显示进度
        if mod(iter, 100) == 0
            fprintf('迭代 %d/%d, 当前最优适应度: %.6f\n', iter, max_iter, best_fitness);
        end
    end
    
    % 输出最终结果
    fprintf('\n===== 禁忌搜索完成 =====\n');
    fprintf('最优解: x1 = %.8f, x2 = %.8f\n', best_solution(1), best_solution(2));
    fprintf('最优适应度: %.8f\n', best_fitness);
    fprintf('理论最小值: -186.7309088\n');
    
    % 绘制收敛曲线
    figure;
    plot(1:max_iter, convergence_curve, 'b-', 'LineWidth', 1.5);
    xlabel('迭代次数');
    ylabel('最优适应度');
    title('禁忌搜索收敛曲线');
    grid on;
    
    % 绘制搜索空间和最优解
    figure;
    [X, Y] = meshgrid(linspace(lower_bound, upper_bound, 200), ...
                      linspace(lower_bound, upper_bound, 200));
    Z = zeros(size(X));
    
    for i = 1:size(X, 1)
        for j = 1:size(X, 2)
            Z(i, j) = shubert([X(i, j), Y(i, j)]);
        end
    end
    
    surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    colormap('jet');
    colorbar;
    hold on;
    
    % 标记最优解
    plot3(best_solution(1), best_solution(2), best_fitness, ...
          'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 2);
    xlabel('x1');
    ylabel('x2');
    zlabel('f(x1, x2)');
    title('舒伯特函数曲面及最优解');
    view(45, 30);
end

function y = shubert(x)
    % 舒伯特函数
    % 输入: x = [x1, x2] 二维向量
    % 输出: 函数值
    
    h1 = 0;
    h2 = 0;
    
    for i = 1:5
        h1 = h1 + i * cos((i + 1) * x(1) + i);
        h2 = h2 + i * cos((i + 1) * x(2) + i);
    end
    
    y = h1 * h2;
end

% 主程序调用
if __name__ == '__main__'
    [best_solution, best_fitness, convergence_curve] = tabu_search_shubert();
end
