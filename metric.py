import os
from matplotlib import pyplot as plt
import numpy as np


def pnsr(original, prediction):
    mse = np.sum((original - prediction)**2)/original.size
    pnsr = 10*np.log10((255**2)/mse)
    return pnsr


if __name__ == '__main__':
    """
    Compute pnsr of a blend folder and save it in a text file
    """
    dataset = "celebA"
    if not os.path.exists("./Output/pnsr_{}/".format(dataset)):
        os.mkdir("./Output/pnsr_{}/".format(dataset))
    a_file = open("./Output/pnsr_{}/pnsr.txt".format(dataset), "w")

    list_pnsr = []
    for i in range(int(len(os.listdir('./Output/Blend/'))/4)):
        blend = plt.imread('./Output/Blend/Image_{}_blend.jpg'.format(i))
        original_image = plt.imread('./Output/Blend/Image_{}_original.jpg'.format(i))

        current_pnsr = pnsr(original_image, blend)

        list_pnsr.append(current_pnsr)
        a_file.write(str(current_pnsr)+'\n')
        print("pnsr {} = {}".format(i, list_pnsr[i]))

    mean_psnr = np.mean(np.asarray(list_pnsr))
    print("mean pnsr = {}".format(mean_psnr))
    a_file.write(str("mean:")+'\n')
    a_file.write(str(mean_psnr)+'\n')
    a_file.close()
