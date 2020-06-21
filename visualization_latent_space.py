import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import utils.models as m
import utils.datasets as d

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#load model
VAE = torch.load("models_and_losses/CNN-VAE_ReLU_NLL_lr-{1e-3,1e-4}_bs-512_epoch-200.pt")
DECODER = VAE.decoder

DECODER.to(DEVICE)

def predict(z):
    mu, sigma = DECODER(z.float().to(DEVICE))
    mu = mu.cpu().detach().numpy()[0]
    sigma = sigma.cpu().detach().numpy()[0]
    return mu, sigma

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.40)
#t = np.linspace(-4,4,1000)
t = np.arange(160)
# initial latent space
z0 = np.zeros(15)
delta_z = 0.01

m,s = predict(torch.tensor([[z0]]))

l, = plt.plot(t, m, lw=2)
fb = plt.fill_between(t, m-s, m+s, alpha=0.5)

#plt.ylim((-2,4))
ax.margins(x=0)

#axfreq = plt.axes([0.18, 0.1, 0.65, 0.03], facecolor=axcolor)
#axamp = plt.axes([0.18, 0.15, 0.65, 0.03], facecolor=axcolor)

Z_s = [Slider(plt.axes([0.18, 0.01+0.02*(14-i), 0.65, 0.01]), f'Z_{i}', -4, 4, valinit=z0[i], valstep=delta_z) for i in reversed(range(15))]

#sfreq = Slider(plt.axes([0.18, 0.1, 0.65, 0.03]), 'Freq', 0, 1, valinit=f0, valstep=delta_f)
#samp = Slider(plt.axes([0.18, 0.15, 0.65, 0.03]), 'Amp', 0.1, 10.0, valinit=a0)


def update(val):
    z_s = [Z.val for Z in Z_s]
    #freq = sfreq.val
    m,s = predict(torch.tensor([[z_s]]))
    #plt.ylim((np.min(m)-s-0.2,np.max(m)+s+0.2))
    l.set_ydata(m)
    #plt.fill_between(t, s-freq, s+freq, alpha=0.5)
    tmp = plt.fill_between(t, m-s, m+s, alpha=0.001)
    #path = fb.get_paths()[0]
    fb.get_paths()[0].vertices[:,0] = tmp.get_paths()[0].vertices[:,0]
    fb.get_paths()[0].vertices[:,1] = tmp.get_paths()[0].vertices[:,1]

    #fb = plt.fill_between(t, s_new-freq, s_new+freq, alpha=0.5)
    #l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

for i in range(15):
    Z_s[i].on_changed(update)
#samp.on_changed(update)



plt.show()
