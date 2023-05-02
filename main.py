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
    return intery[:, 0], intery[:, 1], internumber


#
def sdhfft_inter(x, y):
    z = rfft(y)  # 傅里叶变换
    # print(type(Z))
    # print(Z)
    fftx = rfftfreq(len(x), x[1] - x[0])  # 获取傅里叶变换的横坐标(数量，间隔）（生成的数量是N//2)
    #plt.plot(fftx, np.abs(z))
    #plt.show()
    return fftx, np.abs(z)


def smooth(y):
    # 对数组执行Savitzky-Golay平滑 window_length=len(y)
    smoothed = savgol_filter(y, y.shape[0], polyorder=5)
    Diff = y - smoothed  # 差值，即ΔR
    # 绘制原始数据和平滑后的数据
    return smoothed, Diff


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

a11=[axs[0,0],'(a)',x,          y * 1000,           '-',    'FeGe',     "black",    '$\mu_{0}\mathrm{H(T)}$','$\mathrm{M(m emu})$']#磁化强度
a12=[axs[0,0],'(a)',x,          y * 1000,           '-',    'FeGe',     "black",    '$\mu_{0}\mathrm{H(T)}$','$\mathrm{R_{xx}(m\Omega})$']#电阻
a21=[axs[0,0],'',   x_inter,    y_smooth * 1000,    '--',   'Smoothed', "red",      "",""]#平滑
b11=[axs[0,1],'(b)',x_inter,    y_diff*1000000,     '-',    '',         'black',    '$\mu_{0}H\mathrm{(T)}$','$\mathrm{\Delta M(\mu emu)}$']#磁场Δ磁化强度
b12=[axs[0,1],'(b)',x_inter,    y_diff*1000000,     '-',    '',         'black',    '$\mu_{0}H\mathrm{(T)}$','$\mathrm{\Delta R_{xx}(\mu\Omega)}$']#磁场Δ电阻
c11=[axs[1,0],'(c)',x_inv,      y_diff*1000000,     '-',    '',         'black',    '$1/\mu_{0}H\mathrm{(T^{-1})}$','$\mathrm{\Delta M(\mu emu)}$']#1/磁场Δ磁化强度
c12=[axs[1,0],'(c)',x_inv,      y_diff*1000000,     '-',    '',         'black',    '$1/\mu_{0}H\mathrm{(T^{-1})}$','$\mathrm{\Delta R_{xx}(\mu\Omega)}$']#1/磁场Δ电阻
d11=[axs[1,1],'(d)',freq,       amp*1000,           '-',    '',         'black',    'Frequency(T)','FFT Amp. (arb.units)',[0,3000],[0,6]]#频率幅值
#请遵循以下格式，【图层，指标，横轴数据，纵轴数据，曲线的样式，曲线的标签（为空不处理），曲线的颜色，横坐标标签（为空不处理），纵坐标标签（为空不处理），x轴范围（可选），y轴范围（可选）】
plots=[a11,a21,b11,c11,d11]

for i in plots:
    i[0].text(-0.1, 1.0, i[1], transform=i[0].transAxes,
                va='top', ha='right', fontsize=16, fontweight='bold')
    if i[5]=="":
        i[0].plot(i[2],i[3],i[4],linewidth=2.0, color=i[6])
    else:
        i[0].plot(i[2],i[3],i[4],label=i[5],linewidth=2.0, color=i[6])
        i[0].legend(prop={'size': 14})
    if i[7]!="":
        i[0].set_xlabel(i[7], fontsize=16)
    if i[8]!="":
        i[0].set_ylabel(i[8], fontsize=16)
    i[0].tick_params(labelsize=14)
    i[0].spines['top'].set_linewidth(1.5)
    i[0].spines['bottom'].set_linewidth(1.5)
    i[0].spines['left'].set_linewidth(1.5)
    i[0].spines['right'].set_linewidth(1.5)
    i[0].tick_params(direction='in', width=1.5)
    try:
        i[0].set_xlim(i[9])
    except Exception as error:
        pass
    try:
        i[0].set_ylim(i[10])
    except Exception as error:
        pass
plt.tight_layout()
plt.savefig('test.png', dpi=300, bbox_inches='tight')
plt.show()