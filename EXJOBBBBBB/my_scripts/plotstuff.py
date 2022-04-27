import matplotlib.pyplot as plt
import numpy as np

def main():
    x = np.array([2126.06, 2221.77, 1966.13, 1934.09, 1895.32, 2019.39, 1859.84, 1951.62, 2019.27, 1842.26 ])
    #x = [1900, 2300, 2000, 2200, 2000, 2200, 2300, 2000, 2100, 2100]
    #y = [16,26,36,46]

    plt.hist(x, density=True, bins=15)  # density=False would make counts
    #plt.hist(y, density=True, bins=30)  # density=False would make counts
    
    print(x.std())
    print(x.mean())
    print(x.var())
    plt.xlabel('RMSE')
    plt.suptitle('LSTM accuracy')
    plt.show()

if __name__ == "__main__":
    main()