import matplotlib.pyplot as plt

# 数据
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# 创建图表
plt.plot(x, y)

# 添加标题和标签
plt.title("Simple Plot")
plt.xlabel("x-axis")
plt.ylabel("y-axis")

# 显示图表
plt.show()