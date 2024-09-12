import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from parse_node_pl import nodes
from nets_file import nets
import matplotlib.pyplot as plt
import time
start_time = time.time()

# 将字体设置成包含中文的字体，使matplotlib能够正常显示中文
plt.rc("font", family='YouYuan')

# Environment部分
class Environment:
    def __init__(self, grid_size_x,grid_size_y, num_nodes,nodes,nets):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.num_nodes = num_nodes    #模块数量
        self.state = None
        self.node_coords = []   #记录模块坐标
        self.area_all = 0
        self.overlap_last=0   #记录上一个布局的重叠面积（为了方便计算reward）
        self.nodes_list=nodes.node_list
        self.nets=nets

    def reset(self):
        # 初始化环境状态和模块坐标
        self.state = np.zeros((self.grid_size_x, self.grid_size_y))
        self.node_coords = self.initialize_modules() #调用初始化模块坐标函数
        self.get_area_all()     #计算area_all
        return self.state

    def get_area_all(self):            #获取node的总面积，只会在开始时计算一次
        for node in self.nodes_list:
            self.area_all += node.width * node.height

    def initialize_modules(self):
        # 随机初始化模块坐标    并更新二维表
        for i in range(self.num_nodes):     #每个模块随机选择一个坐标
            if self.nodes_list[i].movetype=='movable':
                x = np.random.randint(0, self.grid_size_x-self.nodes_list[i].width)      #防止坐标超过网格边界，因此减去其宽度
                y = np.random.randint(0, self.grid_size_y-self.nodes_list[i].height)     #该函数取值为[x,y)
                self.node_coords.append((x, y))
            else:
                self.node_coords.append((self.nodes_list[i].ll_xcoord,self.nodes_list[i].ll_ycoord))
        for i,coord in enumerate(self.node_coords):      #根据当前的坐标情况更新state的值
            for j in range(self.nodes_list[i].width):
                for k in range(self.nodes_list[i].height):
                    self.state[coord[0]+j, coord[1]+k]=1
        self.overlap_last=self.overlap_last
        return self.node_coords

    def get_overlap(self):
        area_sum = 0
        for i in self.state:         #遍历二维数组，计算area_sum(重叠的面积只会被计算一次，和area_all不一样)
            for j in i:
                area_sum += j
        overlap=self.area_all - area_sum
        return overlap

    def get_state(self):
        # 返回当前环境状态（二维表）
        self.state = np.zeros((self.grid_size_x, self.grid_size_y))
        for i, coord in enumerate(self.node_coords):  # 根据当前的坐标情况更新state的值
            for j in range(self.nodes_list[i].width):
                for k in range(self.nodes_list[i].height):
                    self.state[coord[0] + j, coord[1] + k] = 1
        return self.state

    def step(self, actions,cth):
        # 执行动作并返回新状态、奖励和完成标志
        self.move_node(actions)     #执行action
        new_state = self.get_state()       #更新网格上模块分布位置
        reward,cost,overlap,wirelength = self.calculate_reward(cth)      #计算is_reward
        print("当前cost值为：", cost,",总线长=",wirelength,",重叠面积=",overlap)
        done = self.is_done(cost,cth)                    #重新计算is_done
        return new_state, reward,cost ,overlap,wirelength,done,self.node_coords

    def move_node(self,actions):
        #print("执行动作")
        # 根据动作移动模块
        #print('action=',action)
        #print(self.node_coords)
        #print(actions)
        #print(self.num_nodes)
        for node in range(self.num_nodes):
            #print(node)
            if self.nodes_list[node].movetype=='terminal':
                continue
                #print('节点',node,"为固定节点")
            else:
                #print(node)
                action=actions[node]
                #print("非固定节点",node,'执行动作',action)
                x, y = self.node_coords[node]
                #print(x,',',y)
                if action == 0:  # 向左移动
                    x = max(0, x - 1)
                elif action == 1:  # 向右移动
                    x = min(self.grid_size_x - 1-self.nodes_list[node].width, x + 1)
                elif action == 2:  # 向下移动
                    y = max(0, y - 1)
                else:# action == 3:  # 向上移动
                    y = min(self.grid_size_y - 1-self.nodes_list[node].height, y + 1)
                self.node_coords[node] = (x, y)
                #print(self.node_coords[node])
        #print(self.node_coords)

    def calculate_reward(self,cth):
        wirelength=0
        area_sum=0
        for net in self.nets:  #对于每一个线网(nets是存放net的列表)
            pin0=net.pins[0]
            node_=self.nodes_list[pin0.node]
            x = self.node_coords[pin0.node][0] + node_.width / 2 + pin0.x_offset
            y = self.node_coords[pin0.node][1] + node_.height / 2 + pin0.y_offset
            x_min,x_max,y_min,y_max=x,x,y,y
            for pin in net.pins[1:]:   #d对于该net的每一个引脚
                node_=self.nodes_list[pin.node]
                x=node_.ll_xcoord+node_.width/2+pin.x_offset
                y=node_.ll_ycoord+node_.height/2+pin.y_offset
                if x<x_min:
                    x_min=x
                if y<y_min:
                    y_min=y
                if x>x_max:
                    x_max=x
                if y>y_max:
                    y_max=y
            wirelength=wirelength+(x_max-x_min)+(y_max-y_min)      #加上该net的半周长
        #print("当前总线长为：",wirelength)
        overlap=self.get_overlap()
        #print("当前重叠面积为：",overlap)
        overlap_last=self.overlap_last
        self.overlap_last=overlap     #为下一次计算reward更新该参数
        w1=3                #超参数
        cost=wirelength+w1*overlap
        beta=1.3              #超参数
        delta=cth*beta-cth
        if overlap>overlap_last:
            return -50,cost,overlap,wirelength
        elif cost < cth:
            return 300,cost,overlap,wirelength
        elif cost>cth and cost<beta*cth:
            return 200*(cost-cth)/(delta),cost,overlap,wirelength
        else:
            return -1,cost,overlap,wirelength

    def is_done(self,cost,cth):
        if cost<cth*0.9:
            return True
        else:
            return False


# Agent部分
class Agent:
    def __init__(self, state_size, action_size,num_nodes):
        self.state_size = state_size                       #输入神经网络的格式
        self.action_size = action_size                     #动作类型个数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 0.6  # 探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.epsilon_min = 0.01  # 最小探索率
        self.learning_rate = 0.001  # 学习率
        self.num_nodes=num_nodes
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=self.state_size))#32 个卷积核，生成 32 个不同的特征图，每个卷积核学习提取不同的特征
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu',kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(self.action_size*self.num_nodes, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state,num_nodes):
        max_q_action=[]
        if np.random.rand() <= self.epsilon:
            for i in range(num_nodes):
                action=np.random.randint(self.action_size)
                max_q_action.append(action)
            return max_q_action
        else:
            state = np.reshape(state, (1, *self.state_size))
            q_values = self.model.predict(state) # <class 'numpy.ndarray'> (1, 25)
            for i in range(num_nodes):
                action=0
                s=5*i
                max_q=q_values[0][s]
                for j in range(5):
                    if q_values[0][s+j]>max_q:
                        action=j
                        max_q=q_values[0][s+j]
                max_q_action.append(action)
            return max_q_action

    def train(self, state, actions, reward, next_state, done,num_nodes):
        state = np.reshape(state, (1, *self.state_size))
        next_state = np.reshape(next_state, (1, *self.state_size))
        target = self.model.predict(state)
        #print(target)
        if done:
            for i in range(num_nodes):
                target[0][i*5+actions[i]] = reward
        else:
            for i in range(num_nodes):
                target[0][i*5+actions[i]] = reward + self.gamma * np.amax(self.model.predict(next_state)[0])      #TD算法，用部分真实的reward和预测的Qt+1  来接近真实的return
        self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



    # 主程序
def main():
    max_=[0,0,0,0]          #分别为  max_x, max_y, max_width, max_height， 用列表只是为了方便写代码
    for node in nodes.node_list:
        inf=[node.ll_xcoord,node.ll_ycoord,node.width,node.height]
        i=0
        for i in range(4):
            if inf[i]>max_[i]:
                max_[i]=inf[i]
    max_x, max_y, max_width, max_height = max_[0],max_[1],max_[2],max_[3]
    '''
    grid_size_x = max_x + max_width        #网格宽度
    grid_size_y = max_y + max_height         #网格高度
    '''
    grid_size_x = 1400  # 网格宽度
    grid_size_y = 1400  # 网格高度

    print(grid_size_x)
    print(grid_size_y)
    num_nodes = nodes.numnodes       #模块的数量
    state_size = (grid_size_x, grid_size_y, 1)         #输入神经网络的格式
    action_size = 5  # 上、下、左、右、不动

    cost_list=[]
    area_list=[]
    length_list=[]
    env = Environment(grid_size_x,grid_size_y, num_nodes,nodes,nets.nets_list)
    agent = Agent(state_size, action_size,num_nodes)
    state = env.reset()
    reward, cost,overlap,wirelength = env.calculate_reward(15000)
    print("初始时，cost=", cost,",总线长=",wirelength,"重叠面积=",overlap)
    cost_list.append(cost)
    area_list.append(overlap)
    length_list.append(wirelength)

    cth = 0.9 * cost

    num_episodes = 8
    cths = [cth, cth*0.8, cth*0.65,cth*0.5 ,cth*0.4]
    cth_index=0

    for episode in range(num_episodes):

        state = np.reshape(state, state_size)   #该步骤是为了整合成神经网络输入的格式
        done = False
        env.episode = 0
        cth = cths[cth_index]
        cth_index=cth_index + 1
        while not done:      #如果满足终止条件的话，终止
            #print(state)
            actions = agent.act(state,num_nodes)
            #print(max_q_node,'节点执行动作',action)

            next_state, reward,cost,overlap,wirelength,done,node_coords = env.step(actions,cth)      #更新相关参数
            cost_list.append(cost)
            area_list.append(overlap)
            length_list.append(wirelength)
            #print(node_coords)
            next_state = np.reshape(next_state, state_size)
            agent.train(state, actions, reward, next_state, done,num_nodes)
            state = next_state
            env.episode += 1
            if env.episode>100:
                print("当前轮次下cost=",cost,'，cth=',cth)
                cth_index=cth_index-1
                done=True
        print(f"Episode {episode + 1} completed.")
        if cth_index==5:
            break
    x_=list(range(len(cost_list)))
    plt.plot(x_,cost_list)
    plt.title('Cost')
    plt.xlabel('Episode')
    plt.ylabel('cost')
    plt.show()

    x_ = list(range(len(length_list)))
    plt.plot(x_, length_list)
    plt.title('总线长')
    plt.xlabel('Episode')
    plt.ylabel('wirelength')
    plt.show()

    x_ = list(range(len(area_list)))
    plt.plot(x_, area_list)
    plt.title('重叠面积')
    plt.xlabel('Episode')
    plt.ylabel('overlap_area')
    plt.show()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"运行时间: {execution_time} 秒")

if __name__ == "__main__":
    main()

