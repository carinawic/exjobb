import matplotlib.pyplot as plt
import numpy as np

def main():
    x = np.array([ 
        
        
        2518, 2964, 2833, 2482, 2118, 2466, 2068, 2261, 2548, 2659

])

    plt.hist(x, bins=15)  # density=False would make counts
    #plt.hist(y, density=True, bins=30)  # density=False would make counts
    
    print(x.std())
    print(x.mean())
    print(x.var())
    plt.xlabel('RMSE')
    plt.ylabel('frequency')
    plt.yticks([0,1,2])
    plt.suptitle('LSTM accuracy')
    plt.show()

if __name__ == "__main__":
    main()