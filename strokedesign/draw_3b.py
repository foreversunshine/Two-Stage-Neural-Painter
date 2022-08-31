from strokes_gen_3bspl import *
from gan_renderer import *

import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

step = 0

while step < 10:
    f = torch.FloatTensor(1, 15).uniform_().to(device)
    stroke = f[:, :12].view(-1).cpu().numpy()
    stroke = torch.from_numpy(draw3b(stroke, 64)).to(device)
    stroke = stroke.view(1, 64, 64, 1)
    tempx = f[:, -3:].view(1, 1, 1, 3)
    color_stroke = stroke * tempx
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    canvas = torch.zeros([1, 3, 64, 64]).to(device)
    canvas = color_stroke+1-stroke
    # canvas = canvas * (1 - color_stroke[0]) + stroke[0]
    canvas = np.transpose(canvas.cpu().numpy(), (0, 2, 3, 1))
    a = canvas[0]
    cv2.imwrite('outputstrokes3bspl/2/s' + str(step) + '.png', (a * 255).astype('uint8'))
    # cv2.imwrite('/home/gc/wq/ganstrokes_3bsc/1/s' + str(step) + '.png', (a * 255).astype('uint8'))
    step += 1
