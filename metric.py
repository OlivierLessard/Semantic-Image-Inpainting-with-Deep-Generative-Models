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
    mask_type = "half"
    dataset_path = "./Output/pnsr_{}/".format(dataset)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    mask_type_path = os.path.join(dataset_path, mask_type)
    if not os.path.exists(mask_type_path):
        os.mkdir(mask_type_path)
    save_path = os.path.join(mask_type_path, "pnsr.txt".format(dataset))
    a_file = open(save_path, "w")

    list_pnsr = []
    blend_path = os.path.join("./Output/BLend/", mask_type+"/")
    for i in range(int(len(os.listdir(blend_path))/4)):
        read_path = os.path.join(blend_path, 'Image_{}_blend.jpg'.format(i))
        blend = plt.imread(read_path)
        read_path = os.path.join(blend_path, 'Image_{}_original.jpg'.format(i))
        original_image = plt.imread(read_path)

        current_pnsr = pnsr(original_image, blend)

        list_pnsr.append(current_pnsr)
        a_file.write(str(current_pnsr)+'\n')
        print("pnsr {} = {}".format(i, list_pnsr[i]))

    mean_psnr = np.mean(np.asarray(list_pnsr))
    print("mean pnsr = {}".format(mean_psnr))
    a_file.write(str("mean:")+'\n')
    a_file.write(str(mean_psnr)+'\n')
    a_file.close()
