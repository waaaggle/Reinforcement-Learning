import matplotlib.pyplot as plt

def show_loss(**kwargs):
    fig = plt.figure(figsize=(8,6))    #创建指定大小画布，不指定默认会创建一张
    #同时画多条损失线，参数名为损失名，参数值为损失值！！！
    for key,value in kwargs.items():
        label = str(key).replace("_", " ")
        plt.plot(range(len(value)), value, marker='o', label=label)  #画线
    plt.xlabel('epoch')  #x轴标签
    plt.ylabel('loss')  #y轴标签
    plt.title('Loss Decreasing During Training')   #标题
    plt.legend()   #显示右上角图例，多条线同时画时有用
    # plt.grid(True)   #显示网格线
    plt.show()

if __name__ == '__main__':
    loss = [0.1*x for x in range(30, 1, -1)]
    show_loss(test_loss = loss)
