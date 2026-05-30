%% Q3_DFD.m  ─  问题三：PSO 压力优化  数据流图（DFD）
%  符号规范：Gane-Sarson 标准（与 Q2_DFD.m 一致）
%
%  运行：直接 F5 或命令行输入 Q3_DFD
%  输出：Q3_DFD.png（当前目录，150 DPI）

clear; clc; close all;

FONT = 'SimHei';   % 修改为本机可用的中文字体

%% ─── 画布 ────────────────────────────────────────────────────────────────────
fig = figure('Color','w','Units','pixels','Position',[50 50 1680 1260]);
ax  = axes(fig,'Position',[0.03 0.05 0.94 0.91]);
hold(ax,'on');  axis(ax,'off');
set(ax,'XLim',[0 24],'YLim',[0 23],'DataAspectRatio',[1 1 1]);

title(ax,'问题三：PSO 压力优化  数据流图（DFD）  [Gane-Sarson 规范]',...
      'FontName',FONT,'FontSize',13,'FontWeight','bold');

%% ─── 尺寸参数 ────────────────────────────────────────────────────────────────
EW=3.0; EH=1.1;    % 外部实体
PW=3.8; PH=1.6;    % 处理过程
SW=5.6; SH=1.0;    % 数据存储

%% ─── 坐标（中心点 cx, cy）────────────────────────────────────────────────────
%  三列布局：左列 x≈2  中列 x≈9.5  右列 x≈19.5
EE1=[2.0 21.5];  EE2=[19.5 1.0];

% 处理过程（中列，纵向等间距）
P1=[9.5 21.5];  P2=[9.5 18.0];  P3=[9.5 14.5];
P4=[9.5 11.0];  P5=[9.5  7.5];  P6=[9.5  4.0];

% 数据存储（右列，对应各处理过程行）
D1=[19.5 21.5];  % 稳态CO基准值（对应 P1 行）
D2=[ 2.0 18.0];  % 历史压力分位数表（左列，对应 P2 行）
D3=[19.5 18.0];  % 18维压力调节范围
D4=[19.5 14.5];  % Q2 XGBoost 预测模型（对应 P3/P4 行之间）
D5=[19.5 11.0];  % 粒子群状态矩阵
D6=[19.5  7.5];  % 全局最优解记录

%% ─── 外部实体 ────────────────────────────────────────────────────────────────
ext(ax,FONT, EE1,EW,EH, {'工况','监测系统'});
ext(ax,FONT, EE2,EW,EH, {'压力调控','执行系统'});

%% ─── 处理过程 ────────────────────────────────────────────────────────────────
proc(ax,FONT, P1,PW,PH, {'P1','稳态 CO 计算','(固定点迭代 α=0.5×50 次)'});
proc(ax,FONT, P2,PW,PH, {'P2','调节边界确定','(10%~90% 分位数截取)'});
proc(ax,FONT, P3,PW,PH, {'P3','PSO 粒子群初始化','(40 粒子 × 18 维)'});
proc(ax,FONT, P4,PW,PH, {'P4','稳态适应度评估','+ 可靠性惩罚'});
proc(ax,FONT, P5,PW,PH, {'P5','自适应粒子更新','(w:0.9→0.4, α:50→500)'});
proc(ax,FONT, P6,PW,PH, {'P6','收敛判断','& 输出最优解'});

%% ─── 数据存储 ────────────────────────────────────────────────────────────────
dstore(ax,FONT, D1,SW,SH,'D1',{'稳态 CO 基准值','(3495.4 mg/m³)'});
dstore(ax,FONT, D2,SW,SH,'D2',{'历史压力分位数表','(10%~90%, 18 维)'});
dstore(ax,FONT, D3,SW,SH,'D3',{'18 维压力调节范围','[下限, 上限]'});
dstore(ax,FONT, D4,SW,SH,'D4',{'Q2 XGBoost 预测模型','(R²=0.9992)'});
dstore(ax,FONT, D5,SW,SH,'D5',{'粒子群状态矩阵','(40×18 位置+速度)'});
dstore(ax,FONT, D6,SW,SH,'D6',{'全局最优解记录','(最优压力+最优 CO)'});

%% ─── 数据流 ──────────────────────────────────────────────────────────────────

%-- 初始化阶段 --

% EE1 → P1（当前负压）
fl(ax,FONT, EE1(1)+EW/2, EE1(2), P1(1)-PW/2, P1(2), '18 个风箱当前负压值');

% D4(Q2模型) → P1（CO预测函数，折线：D4左侧 → 绕上 → P1右侧入口）
%   路径：D4左边 → 向左折 → 向上至P1行 → 向右至P1右侧 → 进入P1
pts24_P1 = [D4(1)-SW/2        D4(2);
            D4(1)-SW/2-1.2    D4(2);
            D4(1)-SW/2-1.2    P1(2)+PH/2+0.5;
            P1(1)+PW/2        P1(2)+PH/2+0.5;
            P1(1)+PW/2        P1(2)];
flpath(ax,FONT, pts24_P1, 'CO 预测函数', 2);

% P1 → D1（稳态CO）
fl(ax,FONT, P1(1)+PW/2, P1(2), D1(1)-SW/2, D1(2), '稳态 CO=3495.4 mg/m³');

% D1 → P2（基准值，折线：D1左侧 → 绕进P2右侧）
pts_D1P2 = [D1(1)-SW/2        D1(2);
            D1(1)-SW/2-0.8    D1(2);
            D1(1)-SW/2-0.8    P2(2)+PH/2+0.4;
            P2(1)+PW/2        P2(2)+PH/2+0.4;
            P2(1)+PW/2        P2(2)];
flpath(ax,FONT, pts_D1P2, '稳态 CO 基准值', 2);

% EE1 → P2（当前工况参考，折线：沿EE1下方向左折 → 进P2左侧）
pts_E1P2 = [EE1(1)+EW/4      EE1(2)-EH/2;
            EE1(1)+EW/4      P2(2)+PH/2+0.9;
            P2(1)-PW/2       P2(2)+PH/2+0.9;
            P2(1)-PW/2       P2(2)];
flpath(ax,FONT, pts_E1P2, '当前工况参考值', 2);

% D2 → P2（历史分位数）
fl(ax,FONT, D2(1)+SW/2, D2(2), P2(1)-PW/2, P2(2), '18 维历史分位数');

% P2 → D3（调节范围）
fl(ax,FONT, P2(1)+PW/2, P2(2), D3(1)-SW/2, D3(2), '18 维压力 [下限, 上限]');

% D3 → P3（边界，折线：D3左侧 → 绕进P3右侧）
pts_D3P3 = [D3(1)-SW/2       D3(2);
            D3(1)-SW/2-0.8   D3(2);
            D3(1)-SW/2-0.8   P3(2)+PH/2+0.4;
            P3(1)+PW/2       P3(2)+PH/2+0.4;
            P3(1)+PW/2       P3(2)];
flpath(ax,FONT, pts_D3P3, '调节范围边界', 2);

% P3 → D5（初始粒子矩阵，斜向右下）
fl(ax,FONT, P3(1)+PW/2, P3(2), D5(1)-SW/2, D5(2)+SH/3, ...
    {'40×18 初始位置','+ 速度矩阵'});

%-- 迭代核心循环 --

% D5 → P4（读取粒子位置）
fl(ax,FONT, D5(1)-SW/2, D5(2), P4(1)+PW/2, P4(2), '当前粒子位置（18 维压力）');

% D4 → P4（XGBoost预测，斜向左下）
fl(ax,FONT, D4(1)-SW/2, D4(2), P4(1)+PW/2, P4(2)-PH/4, 'XGBoost CO 预测');

% P4 → P5（目标值，向下）
fl(ax,FONT, P4(1), P4(2)-PH/2, P5(1), P5(2)+PH/2, ...
    {'目标值 = CO_稳态 + α·Σexp(-10·d)'});

% P5 → D5（更新粒子，斜向右上）
fl(ax,FONT, P5(1)+PW/2, P5(2)+PH/4, D5(1)-SW/2, D5(2)-SH/4, '更新粒子位置与速度');

% P5 → D6（写入全局最优）
fl(ax,FONT, P5(1)+PW/2, P5(2), D6(1)-SW/2, D6(2), '写入新全局最优');

% D6 → P6（读取最优）
fl(ax,FONT, D6(1)-SW/2, D6(2), P6(1)+PW/2, P6(2), '读取当前最优解');

%-- 反馈回路（左侧折线） --
% P6 → P4（未收敛，沿左侧向上）
loop_x = 2.8;
pts_fb = [P6(1)-PW/2    P6(2);
          loop_x        P6(2);
          loop_x        P4(2);
          P4(1)-PW/2    P4(2)];
flpath(ax,FONT, pts_fb, {'未达 150 次迭代','继续评估'}, 2);

%-- 输出阶段 --
% P6 → EE2（最优方案）
fl(ax,FONT, P6(1)+PW/2, P6(2), EE2(1)-EW/2, EE2(2), ...
    {'最优 18 维压力配置','CO=1299.5 mg/m³ / 减排 62.8%','可靠性: 1/18 贴近边界'});

%% ─── 图例（左下角）─────────────────────────────────────────────────────────
ly = 0.55;
ext   (ax,FONT, [1.6 ly], 1.7, 0.65, '外部实体');
proc  (ax,FONT, [4.5 ly], 2.1, 0.65, 'Px 处理过程');
dstore(ax,FONT, [8.4 ly], 3.2, 0.65, 'Dx', '数据存储');
fl    (ax,FONT, 10.7, ly, 12.3, ly,  '数据流');

%% ─── 保存 ────────────────────────────────────────────────────────────────────
drawnow;
print(fig, 'Q3_DFD', '-dpng', '-r150');
fprintf('  ✓  Q3_DFD.png 已保存\n');


%% ═══════════════════════ 本 地 函 数 ═════════════════════════════════════════

function ext(ax, font, pos, w, h, lbl)
rectangle('Parent',ax, 'Position',[pos(1)-w/2, pos(2)-h/2, w, h], ...
          'EdgeColor','k', 'FaceColor','w', 'LineWidth',1.4);
text(pos(1), pos(2), lbl, 'Parent',ax, ...
     'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
     'FontName',font, 'FontSize',9, 'Interpreter','none');
end

function proc(ax, font, pos, w, h, lbl)
rectangle('Parent',ax, 'Position',[pos(1)-w/2, pos(2)-h/2, w, h], ...
          'Curvature',[0.18 0.35], ...
          'EdgeColor','k', 'FaceColor','w', 'LineWidth',1.4);
text(pos(1), pos(2), lbl, 'Parent',ax, ...
     'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
     'FontName',font, 'FontSize',9, 'Interpreter','none');
end

function dstore(ax, font, pos, w, h, id, nm)
x = pos(1)-w/2;  y = pos(2)-h/2;  tab = w*0.20;
patch(ax, [x x+w x+w x], [y y y+h y+h], [0.91 0.91 0.91], 'EdgeColor','none');
line(ax, [x x+w], [y+h y+h], 'Color','k', 'LineWidth',1.4);
line(ax, [x x+w], [y   y  ], 'Color','k', 'LineWidth',1.4);
line(ax, [x+tab x+tab], [y y+h], 'Color','k', 'LineWidth',1.0);
text(x+tab/2, pos(2), id, 'Parent',ax, ...
     'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
     'FontName',font, 'FontSize',9, 'FontWeight','bold', 'Interpreter','none');
text(x+tab+(w-tab)/2, pos(2), nm, 'Parent',ax, ...
     'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
     'FontName',font, 'FontSize',9, 'Interpreter','none');
end

function fl(ax, font, x1, y1, x2, y2, lbl)
line(ax, [x1 x2], [y1 y2], 'Color','k', 'LineWidth',0.9);
arrowhead(ax, x1,y1, x2,y2);
if nargin > 6 && ~isempty(lbl)
    text((x1+x2)/2, (y1+y2)/2, lbl, 'Parent',ax, ...
         'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
         'FontName',font, 'FontSize',7.5, 'Interpreter','none', ...
         'BackgroundColor','w', 'Margin',2);
end
end

function flpath(ax, font, pts, lbl, li)
n = size(pts,1);
for i = 1:n-1
    line(ax, [pts(i,1) pts(i+1,1)], [pts(i,2) pts(i+1,2)], ...
         'Color','k', 'LineWidth',0.9);
end
arrowhead(ax, pts(n-1,1),pts(n-1,2), pts(n,1),pts(n,2));
if nargin > 3 && ~isempty(lbl)
    if nargin < 5,  li = 1;  end
    mx = (pts(li,1)+pts(li+1,1))/2;
    my = (pts(li,2)+pts(li+1,2))/2;
    text(mx, my, lbl, 'Parent',ax, ...
         'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
         'FontName',font, 'FontSize',7.5, 'Interpreter','none', ...
         'BackgroundColor','w', 'Margin',2);
end
end

function arrowhead(ax, x1, y1, x2, y2)
dx = x2-x1;  dy = y2-y1;
L  = max(hypot(dx,dy), 1e-9);
ux = dx/L;   uy = dy/L;
px = -uy;    py =  ux;
hw = 0.13;   hl = 0.27;
xh = [x2-hl*ux + hw*px,  x2,  x2-hl*ux - hw*px];
yh = [y2-hl*uy + hw*py,  y2,  y2-hl*uy - hw*py];
patch(ax, xh, yh, 'k', 'EdgeColor','k');
end
