function [gBestScore, gBest, cg_curve] = TabuSearch(noP, Max_iteration, lb, ub, dim, fobj)
    % ---------------------------------------------------------
    % 连续型禁忌搜索算法 (Continuous Tabu Search)
    % noP           : 每次迭代生成的邻居解数量 (Neighborhood Size)
    % Max_iteration : 最大迭代次数
    % lb, ub        : 变量下界和上界
    % dim           : 变量维度
    % fobj          : 目标函数句柄
    % ---------------------------------------------------------

    % 初始化参数
    ub = ub .* ones(1, dim);
    lb = lb .* ones(1, dim);
    
    % 禁忌搜索专属参数
    TL = round(Max_iteration * 0.2);     % 禁忌长度 (Tabu Length)，根据迭代次数设定
    TabuList = zeros(TL, dim);           % 禁忌表，记录近期访问过的位置
    tabu_index = 1;                      % 禁忌表更新指针
    tabu_count = 0;                      % 当前禁忌表中的记录数
    tabu_radius = norm(ub - lb) * 0.05;  % 禁忌半径：与禁忌表内解的距离小于此值则被视为“禁忌”
    
    % 初始化当前解
    currentPos = (ub - lb) .* rand(1, dim) + lb;
    currentCost = fobj(currentPos);
    
    % 初始化全局最优解
    gBest = currentPos;
    gBestScore = currentCost;
    
    cg_curve = zeros(1, Max_iteration);
    
    % 禁忌搜索主循环
    for t = 1:Max_iteration
        bestNeighborCost = inf;
        bestNeighborPos = currentPos;
        
        % 自适应步长：随着迭代进行，搜索步长逐渐减小 (从 20% 空间递减)
        step_size = (ub - lb) .* (1 - t/Max_iteration) * 0.2; 
        
        % 1. 生成邻域解并评估
        for i = 1:noP
            % 在当前解附近添加高斯随机扰动生成邻居
            neighbor = currentPos + step_size .* randn(1, dim);
            
            % 越界处理
            neighbor = max(neighbor, lb);
            neighbor = min(neighbor, ub);
            
            % 检查是否触发禁忌
            isTabu = false;
            for j = 1:tabu_count
                % 如果新解距离禁忌表中某个解太近，则视为禁忌
                if norm(neighbor - TabuList(j, :)) < tabu_radius
                    isTabu = true;
                    break;
                end
            end
            
            % 计算邻居适应度
            nCost = fobj(neighbor);
            
            % 破禁准则 (Aspiration Criterion)：
            % 如果该解被禁忌，但它的适应度比历史全局最优还要好，则无视禁忌
            if nCost < gBestScore
                isTabu = false; 
            end
            
            % 记录当前非禁忌的最优邻居
            if ~isTabu && (nCost < bestNeighborCost)
                bestNeighborCost = nCost;
                bestNeighborPos = neighbor;
            end
        end
        
        % 2. 异常处理：如果所有邻居都被禁忌且没有触发破禁准则
        if bestNeighborCost == inf
            % 随机生成一个全新解以跳出局部空间
            bestNeighborPos = (ub - lb) .* rand(1, dim) + lb;
            bestNeighborCost = fobj(bestNeighborPos);
        end
        
        % 3. 移动到最佳邻居解
        currentPos = bestNeighborPos;
        currentCost = bestNeighborCost;
        
        % 4. 更新全局最优解 (Best Solution Ever Found)
        if currentCost < gBestScore
            gBestScore = currentCost;
            gBest = currentPos;
        end
        
        % 5. 更新禁忌表 (Update Tabu List - FIFO 队列)
        TabuList(tabu_index, :) = currentPos;
        tabu_index = tabu_index + 1;
        if tabu_index > TL
            tabu_index = 1; % 环形队列重置指针
        end
        if tabu_count < TL
            tabu_count = tabu_count + 1;
        end
        
        % 记录收敛曲线
        cg_curve(t) = gBestScore;
    end
end