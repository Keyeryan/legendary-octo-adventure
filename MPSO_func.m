function y = funShubert(x)
% 函数用于计算粒子适应度值
% x       input  输入粒子（二维向量，x(1)、x(2)）
% y       output 粒子适应度值
    h1 = 0;
    h2 = 0;
    for i = 1:5
        h1 = h1 + i * cos((i + 1) * x(1) + i);
        h2 = h2 + i * cos((i + 1) * x(2) + i);
    end
    y = h1 * h2;  % 舒伯特函数（最小化问题，y越小越优）
end