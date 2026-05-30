%% Q2_DFD.m  ─  问题二：CO 预测模型建立  数据流图（DFD）
%  符号规范：Gane-Sarson 标准
%    外部实体 → 矩形（白底黑边）
%    处理过程 → 圆角矩形（白底黑边，Px 编号）
%    数据存储 → 双横线+左侧ID列（灰底，Gane-Sarson 开口矩形）
%    数据流   → 黑色实线 + 实心三角箭头 + 线上文字标注
%
%  运行：直接 F5 或命令行输入 Q2_DFD
%  输出：Q2_DFD.png（当前目录，150 DPI）
%
%  字体说明：Windows 中文版默认含 SimHei；
%            Linux/Mac 可将 FONT 改为系统已有的中文字体名称。

clear; clc; close all;

FONT = 'SimHei';   % 修改为本机可用的中文字体

%% ─── 画布 ────────────────────────────────────────────────────────────────────
fig = figure('Color','w','Units','pixels','Position',[50 50 1680 1050]);
ax  = axes(fig,'Position',[0.02 0.06 0.96 0.90]);
hold(ax,'on');  axis(ax,'off');
set(ax,'XLim',[0 24],'YLim',[0.5 16.5],'DataAspectRatio',[1 1 1]);

title(ax,'问题二：CO 预测模型建立  数据流图（DFD）  [Gane-Sarson 规范]',...
      'FontName',FONT,'FontSize',13,'FontWeight','bold');

%% ─── 尺寸参数 ────────────────────────────────────────────────────────────────
EW=2.8; EH=1.0;    % 外部实体  宽×高
PW=3.3; PH=1.5;    % 处理过程  宽×高
SW=5.2; SH=1.0;    % 数据存储  宽×高

%% ─── 坐标（中心点 cx, cy）────────────────────────────────────────────────────
%  三列布局：左列 x≈2  中列 x≈9  右列 x≈19
EE1=[2.0 14.5];  EE2=[18.5 1.5];
P1=[9.0 14.5];  P2=[9.0 11.5];  P3=[5.5 8.5];  P4=[12.5 8.5];  P5=[9.0 5.5];
D1=[19.2 14.5]; D2=[2.2 11.5];  D3=[19.2 11.5]; D4=[19.2 8.5];

%% ─── 外部实体 ────────────────────────────────────────────────────────────────
ext(ax,FONT, EE1,EW,EH, {'烧结','生产系统'});
ext(ax,FONT, EE2,EW,EH, {'模型','使用方'});

%% ─── 处理过程 ────────────────────────────────────────────────────────────────
proc(ax,FONT, P1,PW,PH, {'P1','数据采集与预处理'});
proc(ax,FONT, P2,PW,PH, {'P2','特征工程'});
proc(ax,FONT, P3,PW,PH, {'P3','PSO 超参数','寻优'});
proc(ax,FONT, P4,PW,PH, {'P4','XGBoost','模型训练'});
proc(ax,FONT, P5,PW,PH, {'P5','模型评估','验证'});

%% ─── 数据存储 ────────────────────────────────────────────────────────────────
dstore(ax,FONT, D1,SW,SH,'D1','原始工况数据库');
dstore(ax,FONT, D2,SW,SH,'D2',{'清洗后数据集','（2425 条）'});
dstore(ax,FONT, D3,SW,SH,'D3','84 维特征数据集');
dstore(ax,FONT, D4,SW,SH,'D4',{'XGBoost 预测模型','R²=0.9992'});

%% ─── 数据流 ──────────────────────────────────────────────────────────────────
% EE1 → P1
fl(ax,FONT, EE1(1)+EW/2, EE1(2), P1(1)-PW/2, P1(2), ...
    {'原始压力/温度','/速度/CO 记录'});

% P1 → D1（写入原始记录）
fl(ax,FONT, P1(1)+PW/2, P1(2), D1(1)-SW/2, D1(2), '写入原始记录');

% P1 → D2（清洗结果，斜向左下）
fl(ax,FONT, P1(1)-PW/4, P1(2)-PH/2, D2(1)+SW/3, D2(2)+SH/2, ...
    {'2425 条','清洗后记录'});

% D2 → P2
fl(ax,FONT, D2(1)+SW/2, D2(2), P2(1)-PW/2, P2(2), '读取清洗数据');

% P2 → D3（特征矩阵）
fl(ax,FONT, P2(1)+PW/2, P2(2), D3(1)-SW/2, D3(2), ...
    {'84 维特征矩阵','(物理+梯度+统计+CO 滞后)'});

% D3 → P3（CV 折叠特征集，斜向左下）
fl(ax,FONT, D3(1)-SW/2, D3(2)-SH/4, P3(1)+PW/2, P3(2)+PH/4, ...
    {'特征集','(CV 折叠)'});

% D3 → P4（训练特征集，斜向左下）
fl(ax,FONT, D3(1)-SW/4, D3(2)-SH/2, P4(1)+PW/4, P4(2)+PH/2, '训练特征集');

% P3 → P4（最优超参数）
fl(ax,FONT, P3(1)+PW/2, P3(2), P4(1)-PW/2, P4(2), ...
    {'最优超参数','(lr=0.206, depth=3, n=262)'});

% P4 → D4（保存模型）
fl(ax,FONT, P4(1)+PW/2, P4(2), D4(1)-SW/2, D4(2), '保存训练完成模型');

% D4 → P5（调用预测模型，斜向左下）
fl(ax,FONT, D4(1)-SW/4, D4(2)-SH/2, P5(1)+PW/3, P5(2)+PH/2, '调用预测模型');

% D2 → P5（测试集，折线路径：向下沿左列 → 右折至 P5）
pts = [D2(1)       D2(2)-SH/2;
       D2(1)       4.0;
       P5(1)-PW/2  4.0;
       P5(1)-PW/2  P5(2)-PH/2];
flpath(ax,FONT, pts, {'测试集样本','(695 条)'}, 2);

% P5 → EE2（评估报告）
fl(ax,FONT, P5(1)+PW/2, P5(2), EE2(1)-EW/2, EE2(2), ...
    {'R²=0.9992 / MAE=50.1','评估报告'});

% D4 → EE2（部署模型）
fl(ax,FONT, D4(1), D4(2)-SH/2, EE2(1), EE2(2)+EH/2, '部署 XGBoost 预测模型');

%% ─── 图例（左下角）─────────────────────────────────────────────────────────
ly = 0.9;
ext   (ax,FONT, [1.6 ly], 1.7, 0.6, '外部实体');
proc  (ax,FONT, [4.5 ly], 2.0, 0.6, 'Px 处理过程');
dstore(ax,FONT, [8.3 ly], 3.1, 0.6, 'Dx', '数据存储');
fl    (ax,FONT, 10.6, ly, 12.2, ly, '数据流');

%% ─── 保存 ────────────────────────────────────────────────────────────────────
drawnow;
print(fig, 'Q2_DFD', '-dpng', '-r150');
fprintf('  ✓  Q2_DFD.png 已保存\n');


%% ═══════════════════════ 本 地 函 数 ═════════════════════════════════════════

function ext(ax, font, pos, w, h, lbl)
%EXT  外部实体：矩形，白底黑边
rectangle('Parent',ax, 'Position',[pos(1)-w/2, pos(2)-h/2, w, h], ...
          'EdgeColor','k', 'FaceColor','w', 'LineWidth',1.4);
text(pos(1), pos(2), lbl, 'Parent',ax, ...
     'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
     'FontName',font, 'FontSize',9, 'Interpreter','none');
end

function proc(ax, font, pos, w, h, lbl)
%PROC  处理过程：圆角矩形，白底黑边
rectangle('Parent',ax, 'Position',[pos(1)-w/2, pos(2)-h/2, w, h], ...
          'Curvature',[0.18 0.35], ...
          'EdgeColor','k', 'FaceColor','w', 'LineWidth',1.4);
text(pos(1), pos(2), lbl, 'Parent',ax, ...
     'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
     'FontName',font, 'FontSize',9, 'Interpreter','none');
end

function dstore(ax, font, pos, w, h, id, nm)
%DSTORE  Gane-Sarson 数据存储：双横线 + 左侧 ID 列，灰底
x = pos(1)-w/2;  y = pos(2)-h/2;  tab = w*0.20;
% 灰色底色
patch(ax, [x x+w x+w x], [y y y+h y+h], [0.91 0.91 0.91], 'EdgeColor','none');
% 上、下边线
line(ax, [x x+w], [y+h y+h], 'Color','k', 'LineWidth',1.4);
line(ax, [x x+w], [y   y  ], 'Color','k', 'LineWidth',1.4);
% 左侧竖分隔线
line(ax, [x+tab x+tab], [y y+h], 'Color','k', 'LineWidth',1.0);
% ID 编号（加粗）
text(x+tab/2, pos(2), id, 'Parent',ax, ...
     'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
     'FontName',font, 'FontSize',9, 'FontWeight','bold', 'Interpreter','none');
% 名称
text(x+tab+(w-tab)/2, pos(2), nm, 'Parent',ax, ...
     'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
     'FontName',font, 'FontSize',9, 'Interpreter','none');
end

function fl(ax, font, x1, y1, x2, y2, lbl)
%FL  数据流：实线 + 实心三角箭头 + 标注文字
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
%FLPATH  折线数据流（直角连线），pts 为 N×2 路径点矩阵
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
%ARROWHEAD  在 (x2,y2) 处绘制实心三角箭头，方向由 (x1,y1)→(x2,y2) 决定
dx = x2-x1;  dy = y2-y1;
L  = max(hypot(dx,dy), 1e-9);
ux = dx/L;   uy = dy/L;   % 单位方向向量
px = -uy;    py =  ux;    % 垂直向量
hw = 0.13;   hl = 0.27;   % 半宽 / 箭头长（数据单位）
xh = [x2-hl*ux + hw*px,  x2,  x2-hl*ux - hw*px];
yh = [y2-hl*uy + hw*py,  y2,  y2-hl*uy - hw*py];
patch(ax, xh, yh, 'k', 'EdgeColor','k');
end
