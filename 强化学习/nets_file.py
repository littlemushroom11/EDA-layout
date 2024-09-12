# Define the Net and Pin classes
class Net:
    def __init__(self, name,degree):
        self.net_name = name
        self.degree=degree
        self.pins = []

class Pin:
    def __init__(self, node,direction, x_offset, y_offset):
        self.node = node          #int类型，记录的是对应节点的下标
        self.direction=direction
        self.x_offset = x_offset
        self.y_offset = y_offset

class Nets:
    def __init__(self):
        self.net_num=0
        self.pin_num=0
        self.nets_list=[]

nets=Nets()

# 打开文件
with open('test.nets', 'r') as file:
    # 读取文件的第一行
    i=0
    while i<5:
        line = file.readline()
        i=i+1
    words = line.split()
    nets.net_num=int(words[2])

    line=file.readline()
    words = line.split()
    nets.pin_num=int(words[2])
    line = file.readline()

    for line in file:
        words=line.split()
        if words[0]=='NetDegree':
            degree=int(words[2])
            net=Net(words[3],degree)
            i=0
        else:
            i=i+1
            pin = Pin(int(words[0][1:]),words[1],float(words[3]),float(words[4]))
            net.pins.append(pin)
            if i==degree:
              nets.nets_list.append(net)

'''
for i in nets.nets_list:
    print(i.net_name)
for j in i.pins:
    print(j.node,"  ",j.direction,"  ",j.x_offset,"  ",j.y_offset)
'''