import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import interpolate
from scipy.signal import savgol_filter

N = 3001
xf = np.zeros((N, 2))
# print(xf)

# 生成波形
xf[:, 0] = np.linspace(0, 10 * np.pi, N)
# print(np.sin(xf[:,0]))
xf[:, 1] = np.sin(xf[:, 0])


# print(xf)
def inter(x, y, range, interval):
    """根据输入的m数据，range范围，会保留正负，interval内插间隔"""
    u, indices = np.unique(x, return_index=True)  # 略去其中的重复值
    fx = interpolate.interp1d(x[indices], y[indices], kind="linear",
                              fill_value="extrapolate")  # 'linear','zero', 'slinear', 'quadratic', 'cubic'
    internumber = int((range[1] - range[0]) / interval + 1)
    x = np.linspace(range[0], range[1], internumber)
    intery = np.zeros([x.size, 2])
    intery[:, 0] = x
    intery[:, 1] = fx(x)
    # plt.plot(m[:,1],m[:,2],intery[:,0],intery[:,1])
    # plt.show()
    # print(intery)
    return intery[:, 0], intery[:, 1], internumber


#
def sdhfft_inter(x, y):
    z = rfft(y)  # 傅里叶变换
    # print(type(Z))
    # print(Z)
    fftx = rfftfreq(len(x), x[1] - x[0])  # 获取傅里叶变换的横坐标(数量，间隔）（生成的数量是N//2)
    plt.plot(fftx, np.abs(z))
    plt.show()
    return fftx, np.abs(z)


def smooth(y):
    # 对数组执行Savitzky-Golay平滑 window_length=len(y)
    smoothed = savgol_filter(y, y.shape[0], polyorder=5)
    Diff = y - smoothed  # 差值，即ΔR
    # 绘制原始数据和平滑后的数据
    return smoothed, Diff


def plot_academic(data1):
    fig = plt.figure(figsize=(19.2, 10.8))


range = [8, 14]
range2 = [0.0714, 0.125]
interval = 0.01
interval2 = 0.00005
font = 14
data = np.loadtxt('Sheet1.dat', delimiter='\t')
x = data[:, 0]/10000
y = data[:, 1]
x_inter, y_inter, num = inter(x, y, range, interval)
y_smooth, y_diff = smooth(y_inter)
x_inv = 1 / x_inter
x_inter2, y_inter2, num2 = inter(x_inv, y_diff, range2, interval2)
freq, amp = sdhfft_inter(x_inter2, y_inter2)

data = np.column_stack((x_inter, y_inter, y_smooth, y_diff))
np.savetxt("data.txt", data, delimiter=",", header="x_inter,y_inter,y_smooth,y_diff")
# 保存幅值数据
fft_data = np.column_stack((freq, amp))
np.savetxt("fft.txt", fft_data, delimiter=",", header="Frequency,Amplitude")

fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=100)

axs[0, 0].plot(x, y * 1000, '-', linewidth=2.0, label='FeGe', color="black")
axs[0, 0].plot(x_inter, y_smooth * 1000, '--', label='Smoothed', color="red", linewidth=2.0)
axs[0, 0].tick_params(labelsize=14)
axs[0, 0].legend(prop={'size': 14})
axs[0, 0].set_xlabel('$\mu_{0}\mathrm{H(T)}$', fontsize=16)
#axs[0, 0].set_ylabel('$\mathrm{R_{xx}(m\Omega})$', fontsize=16)
axs[0, 0].set_ylabel('$\mathrm{M(m emu})$', fontsize=16)
axs[0, 0].spines['top'].set_linewidth(1.5)
axs[0, 0].spines['bottom'].set_linewidth(1.5)
axs[0, 0].spines['left'].set_linewidth(1.5)
axs[0, 0].spines['right'].set_linewidth(1.5)
axs[0, 0].tick_params(direction='in', width=1.5)
axs[0, 0].text(-0.1, 1.0, '(a)', transform=axs[0, 0].transAxes,
                va='top', ha='right', fontsize=16, fontweight='bold')


axs[0, 1].plot(x_inter, y_diff*1000000,'-', linewidth=2.0, color="black")
axs[0, 1].tick_params(labelsize=14)
#axs[0, 1].legend(prop={'size': 14})
axs[0, 1].set_xlabel('$\mu_{0}H\mathrm{(T)}$', fontsize=16)
#axs[0, 1].set_ylabel('$\mathrm{\Delta R_{xx}(\mu\Omega)}$', fontsize=16)
axs[0, 1].set_ylabel('$\mathrm{\Delta M(\mu emu)}$', fontsize=16)
axs[0, 1].spines['top'].set_linewidth(1.5)
axs[0, 1].spines['bottom'].set_linewidth(1.5)
axs[0, 1].spines['left'].set_linewidth(1.5)
axs[0, 1].spines['right'].set_linewidth(1.5)
axs[0, 1].tick_params(direction='in', width=1.5)


axs[0, 1].text(-0.1, 1.0, '(b)', transform=axs[0, 1].transAxes,
                va='top', ha='right', fontsize=16, fontweight='bold')


axs[1, 0].plot(x_inv, y_diff*1000000,'-', linewidth=2.0, color="black")
# axs[1, 0].plot(x_inter2, y_inter2)
axs[1, 0].tick_params(labelsize=14)
#axs[1, 0].legend(prop={'size': 14})
axs[1, 0].set_xlabel('$1/\mu_{0}H\mathrm{(T^{-1})}$', fontsize=16)
#axs[1, 0].set_ylabel('$\mathrm{\Delta R_{xx}(\mu\Omega)}$', fontsize=16)
axs[1, 0].set_ylabel('$\mathrm{\Delta M(\mu emu)}$', fontsize=16)
axs[1, 0].spines['top'].set_linewidth(1.5)
axs[1, 0].spines['bottom'].set_linewidth(1.5)
axs[1, 0].spines['left'].set_linewidth(1.5)
axs[1, 0].spines['right'].set_linewidth(1.5)
axs[1, 0].tick_params(direction='in', width=1.5)


axs[1, 0].text(-0.1, 1.0, '(c)', transform=axs[1, 0].transAxes,
                va='top', ha='right', fontsize=16, fontweight='bold')


axs[1, 1].plot(freq, amp*1000,'-', linewidth=2.0, color="black")
axs[1, 1].set_xlim([0, 3000])
axs[1, 1].set_ylim([0, 6])
axs[1, 1].tick_params(labelsize=14)
axs[1, 1].legend(prop={'size': 14})
axs[1, 1].set_xlabel('Frequency(T)', fontsize=16)
axs[1, 1].set_ylabel('FFT Amp. (arb.units)', fontsize=16)
axs[1, 1].spines['top'].set_linewidth(1.5)
axs[1, 1].spines['bottom'].set_linewidth(1.5)
axs[1, 1].spines['left'].set_linewidth(1.5)
axs[1, 1].spines['right'].set_linewidth(1.5)
axs[1, 1].tick_params(direction='in', width=1.5)
axs[1, 1].text(-0.1, 1.0, '(d)', transform=axs[1, 1].transAxes,
                va='top', ha='right', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('FeGe.png', dpi=300, bbox_inches='tight')
plt.show()
