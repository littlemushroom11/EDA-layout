# EDA-project

#### 项目简介

本项目主要针对芯片布局的物理设计这一问题进行了分析与探索。分别使用模拟退火算法和强化学习算法完成了对标准单元的优化布局。在算法中考虑了布局宽度、布线复杂度、对称性、引脚密度以及DRC规则的因素，能够对布局结果进行评估，并根据评估结果继续调整布局。

#### 模拟退火

- eda.ml为主程序
- cells.spi为数据集
- placement文件夹中包含部分标准单元的布局结果
- 使用`python main.py <placement_file> <cell_name> <netlist>`命令查看对单个标准单元布局结果placement_file的评价成绩

#### 强化学习

- main.py为主程序

- 使用ISPD数据集

- test.nodes，test.nets，test.pl为测试用例