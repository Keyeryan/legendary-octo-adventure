function [gBestScore, gBest, cg_curve] = IPSO(noP, Max_iteration, lb, ub, dim, fobj)
    % 初始化参数
    ub = ub .* ones(1, dim);
    lb = lb .* ones(1, dim);
    wMax=0.9;
    wMin=0.2;
    c1=2;
    c2=2;
    vMax = (ub - lb) * 0.2;
    vMin = -vMax;

    iter = Max_iteration;
    pBestScore = zeros(noP, 1);
    pBest = zeros(noP, dim);
    gBest = zeros(1, dim);
    cg_curve = zeros(1, iter);
    vel = zeros(noP, dim);
    pos = zeros(noP, dim);

    % 种群初始化
    for i = 1:size(pos, 1)
        for j = 1:size(pos, 2)
            pos(i, j) = (ub(j) - lb(j)) * rand() + lb(j);
            vel(i, j) = rand();
        end
    end
    
    for i = 1:noP
        pBestScore(i) = inf;
    end
    gBestScore = inf;

    % 主循环
    for t = 1:iter
        for i = 1:size(pos, 1)
            % 越界处理
            Flag4ub = pos(i, :) > ub;
            Flag4lb = pos(i, :) < lb;
            pos(i, :) = (pos(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            
            % 计算适应度
            fitness = fobj(pos(i, :));
            
            % 个体最优更新
            if (pBestScore(i) > fitness)
                pBestScore(i) = fitness;
                pBest(i, :) = pos(i, :);
            end
            % 群体最优更新
            if (gBestScore > fitness)
                gBestScore = fitness;
                gBest = pos(i, :);
            end
        end
        
        % 线性递减惯性权重
        w = wMax - t * ((wMax - wMin) / iter);
        
        % 速度和位置更新
        for i = 1:size(pos, 1)
            for j = 1:size(pos, 2)
                vel(i, j) = w * vel(i, j) + c1 * rand() * (pBest(i, j) - pos(i, j)) + c2 * rand() * (gBest(j) - pos(i, j));
                
                % 速度限制
                if vel(i, j) > vMax(j)
                    vel(i, j) = vMax(j);
                end
                if vel(i, j) < vMin(j)
                    vel(i, j) = vMin(j);
                end
                
                % 位置更新
                pos(i, j) = pos(i, j) + vel(i, j);
            end
            
            % ==========================================
            % 改进策略：自适应变异 (Adaptive Mutation)
            % ==========================================
            if rand() > 0.999
                for j = 1:size(pos, 2)
                     % 在搜索空间内重新随机分配位置
                     pos(i, j) = (ub(j) - lb(j)) * rand() + lb(j);
                end
            end
            % ==========================================
        end
        
        % 记录每次迭代的全局最优值
        cg_curve(t) = gBestScore;
    end
end