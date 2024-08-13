import numpy as np
import matplotlib.pyplot as plt
import imageio

def fpstodur(fps):
    dur = 1/fps*1000
    return dur

path_gif = '/SanDisk/Li/LowRank_ModifiedTheta_SNN/PYTHON/gif_file/'
# 定义生成每一帧的函数
def generate_frame(t):
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x + np.pi * t)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.xlim(0, 2*np.pi)
    plt.ylim(-1.5, 1.5)
    return plt

# # 使用with语句创建一个临时的图形，以避免内存中积累太多图形
# frames = []  # 存储帧的数组
# for t in range(30):  # 生成30帧
#     with generate_frame(t / 10.0).canvas.printing():
#         frame = plt.gcf()  # 获取当前图形
#         frame.canvas.draw()  # 绘制图形
#         image = np.frombuffer(frame.canvas.tostring_rgb(), dtype='uint8')
#         image = image.reshape(frame.canvas.get_width_height()[::-1] + (3,))
#         frames.append(image)
#     plt.close(frame)  # 现在我们关闭图形

# 生成帧
frames = []
for t in range(40):  # 生成30帧
    frame = generate_frame(t/10.0)  # t/10.0 使得正弦波有旋转的效果
    frame.savefig(f'{path_gif}frame_{t}.png')  # 保存每一帧为PNG
    plt.close()  # 关闭图形，避免显示
    frames.append(imageio.imread(f'{path_gif}frame_{t}.png'))  # 读取每一帧

# 保存为GIF
imageio.mimwrite(f'{path_gif}sine_wave.gif', frames, duration=fpstodur(15),loop = 0)  # duration (in ms) fps = 1/(0.001*duration) 
# imageio.mimwrite(f'{path_gif}sine_wave.gif', frames, fps=10)  # duration (in ms)


