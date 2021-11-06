import matplotlib.pyplot as plt
import numpy as np
import random


def draw_results(images, y_pred):
    fig, ax = plt.subplots(2,3, figsize=(50, 50))
    rnd_sample = random.sample(range(len(images)),6)

    ax[0][0].set_title('Imagen numero: 1', size = 50)
    ax[0][1].set_title('Imagen numero: 2', size = 50)
    ax[0][2].set_title('Imagen numero: 3', size = 50)
    ax[1][0].set_title('Imagen numero: 4', size = 50)
    ax[1][1].set_title('Imagen numero: 5', size = 50)
    ax[1][2].set_title('Imagen numero: 6', size = 50)

    ax[0][0].imshow(images[rnd_sample[0]])
    ax[0][0].imshow(y_pred[rnd_sample[0]], alpha=0.3)
    ax[0][1].imshow(images[rnd_sample[1]])
    ax[0][1].imshow(y_pred[rnd_sample[1]], alpha=0.3)
    ax[0][2].imshow(images[rnd_sample[2]])
    ax[0][2].imshow(y_pred[rnd_sample[2]], alpha=0.3)
    ax[1][0].imshow(images[rnd_sample[3]])
    ax[1][0].imshow(y_pred[rnd_sample[3]], alpha=0.3)
    ax[1][1].imshow(images[rnd_sample[4]])
    ax[1][1].imshow(y_pred[rnd_sample[4]], alpha=0.3)
    ax[1][2].imshow(images[rnd_sample[5]])
    ax[1][2].imshow(y_pred[rnd_sample[5]], alpha=0.3)

    fig.tight_layout()
    plt.show()






        
    