import numpy as np
import pandas as pd


def main():
    ses = np.load('results/ses.npy')#[:20]
    read = np.load('results/read.npy')#[:20]
    write = np.load('results/write.npy')#[:20]

    for n, a in zip(['ses', 'read', 'write'], [ses, read, write]):
        for val in ['min', 'mean', 'max']:
            aa = eval(f'np.{val}(a, axis=0)')

            df = pd.DataFrame(aa)

            df.to_csv(f'results/{n}_{val}.csv')


if __name__ == '__main__':
    main()
