import numpy as np

#读取身高数据
data = np.loadtxt('data.txt')
#读取性别数据
with open('gender.txt', 'r') as file:
    # 去除每行末尾的换行符
    gender = [line.strip() for line in file]

number_all = len(gender)
mask_m = [g == 'M' for g in gender]
number_m = sum(mask_m)
mask_f = [g == 'F' for g in gender]
number_f = sum(mask_f)
#真实男女比例
p_gt = number_m/number_all
#男生身高数据
import statistics
data_m = data[mask_m]
#男生身高均值
mu_1_gt = statistics.mean(data_m)
#男生身高方差
sigma_1_gt = statistics.stdev(data_m)
#女生身高数据
data_f = data[mask_f]
#女生身高均值
mu_2_gt = statistics.mean(data_f)
#女生身高方差
sigma_2_gt = statistics.stdev(data_f)

#define gaussian function
def gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2/(2 * sigma**2))

def P_m(x, p, mu_1, sigma_1, mu_2, sigma_2):
    #样本属于男生的可能性：
    # P(M|H) = P(M) * P(H|M)/(P(M) * P(H|M) + P(F) * P(H|F))
    # P(M) + P(F) = 1
    p_h_m = gaussian(x, mu_1, sigma_1)
    p_h_f    = gaussian(x, mu_2, sigma_2)
    p_m = p * p_h_m/(p * p_h_m + (1 - p) * p_h_f)
    return p_m

def main():
    #学生数量
    number = len(data)
    #初始化男女生比例 p:1-p
    p = 0.5
    #初始化男生身高均值、方差
    mu_1 = 170
    sigma_1 = 10
    #初始化女生身高均值、方差
    mu_2 = 160
    sigma_2 = 10
    #最大迭代次数
    iter_max = 500
    #判断是否收敛的阈值
    thr = 0.001

    #开始迭代
    for i in range(iter_max):
        #E
        #属于男生的后验概率
        p_m = P_m(data, p, mu_1, sigma_1, mu_2, sigma_2)
        #属于女生的后验概率
        p_f = 1 - p_m

        # M
        p_new = np.mean(p_m)
        mu_1_new = np.sum(p_m * data) / np.sum(p_m)
        mu_2_new = np.sum(p_f * data) / np.sum(p_f)
        sigma_1_new = np.sqrt(np.sum(p_m * (data - mu_1_new)**2) / np.sum(p_m))
        sigma_2_new = np.sqrt(np.sum(p_f * (data - mu_2_new)**2) / np.sum(p_f))

        # 判断是否收敛
        if abs(p_new - p) < thr and abs(mu_1_new - mu_1) < thr:
            break

        # 更新参数
        p, mu_1, mu_2, sigma_1, sigma_2 = p_new, mu_1_new, mu_2_new, sigma_1_new, sigma_2_new

    return p, 1 - p, mu_1, mu_2, sigma_1, sigma_2

if __name__ == '__main__':
    # 估计男女生比例、身高、方差
    p_male, p_female, mu_1, mu_2, sigma_1, sigma_2= main()

    # 输出结果
    print(f"男女生预测比例为{p_male:.2f}:{p_female:.2f}")
    print(f"男女生真实比例为{p_gt:.2f}:{(1 - p_gt):.2f}")
    print(f"男女生身高预测平均值为{mu_1:.2f}:{mu_2:.2f}")
    print(f"男女生身高真实平均值为{mu_1_gt:.2f}:{mu_2_gt:.2f}")
    print(f"男女生身高预测标准差为{sigma_1:.2f}:{sigma_2:.2f}")
    print(f"男女生身高真实标准差为{sigma_1_gt:.2f}:{sigma_2_gt:.2f}")