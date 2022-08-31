import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import cv2
from PIL import Image

from canvas_tp import NeuralCanvas, NeuralCanvasStitched
from transforms import RandomRotate, Normalization, RandomCrop, RandomScale
from viz import *
from SDCGAN_sgmd import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# all 0 to 1
ACTIONS_TO_IDX = {
    'pressure': 0,
    'size': 1,
    'control_x': 2,
    'control_y': 3,
    'end_x': 4,
    'end_y': 5,
    'color_r': 6,
    'color_g': 7,
    'color_b': 8,
    'start_x': 9,
    'start_y': 10,
    'entry_pressure': 11,
    'pressure2': 12,
    'size2': 13,
    'control_x2': 14,
    'control_y2': 15,
    'end_x2': 16,
    'end_y2': 17,
    'color_r2': 18,
    'color_g2': 19,
    'color_b2': 20,
    'start_x2': 21,
    'start_y2': 22,
    'entry_pressure2': 23,
    'entry_pressure3': 24,
    'pressure3': 25,
    'size3': 26,
    'control_x3': 27,
    'control_y3': 28,
    'end_x3': 29,
    'end_y3': 30,
    'color_r3': 31,
    'color_g3': 32,
    'color_b3': 33,
    'start_x3': 34,
    'start_y3': 35,
    'entry_pressure33': 36,
    'pressure23': 37,
    'size23': 38,
    'control_x23': 39,
    'control_y23': 40,
    'end_x23': 41,
    'end_y23': 42,
    'color_r23': 43,
    'color_g23': 44,
    'color_b23': 45,
    'start_x23': 46,
    'start_y23': 47,
    'entry_pressure23': 48,
    'entry_pressure333': 49,
}

def pad(img, H, W):
    b, c, h, w = img.shape
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = torch.cat([torch.zeros((b, c, pad_h, w), device=img.device), img,
                     torch.zeros((b, c, pad_h + remainder_h, w), device=img.device)], dim=-2)
    img = torch.cat([torch.zeros((b, c, H, pad_w), device=img.device), img,
                     torch.zeros((b, c, H, pad_w + remainder_w), device=img.device)], dim=-1)
    return img

inception_v1 = torch.hub.load('pytorch/vision:v0.9.0', 'googlenet', pretrained=True)
resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
vgg19 = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)

STROKES_PER_BLOCK = 3 #@param {type:"slider", min:1, max:15, step:1}
# REPEAT_CANVAS_HEIGHT = 4 #@param {type:"slider", min:1, max:30, step:1}
# REPEAT_CANVAS_WIDTH = 4 #@param {type:"slider", min:1, max:30, step:1}
#@markdown REPEAT_CANVAS_HEIGHT and REPEAT_CANVAS_WIDTH are important parameters to choose how many 64x64 canvases make up the height and width of the output image. Try matching them with your target's aspect ratio.
LAYER = "3B" #@param ["3A", "3B"]
LAYER_IDX = -12 if LAYER == "3A" else -13
#@markdown Which GoogleNet layer to use for content loss. Deeper layers (3B) result in more abstract results
STOCHASTIC = False #@param {type:"boolean"}
#@markdown Experimental. Adding uncertainty may (or may not) help produce more robust images.
NORMALIZE = True #@paAram {type:"boolean"}
LEARNING_RATE = 0.099 #@param {type: "number"}
IMAGE_NAME = 'images/flower1.jpg' #@param {type: "string"}
mask = 'mask/flower1.png'
print('STROKES_PER_BLOCK: {}'.format(STROKES_PER_BLOCK))
# print("REPEAT_CANVAS_HEIGHT", REPEAT_CANVAS_HEIGHT)
# print("REPEAT_CANVAS_WIDTH", REPEAT_CANVAS_WIDTH)
print('LAYER: {}'.format(LAYER))
print('STOCHASTIC: {}'.format(STOCHASTIC))
print('NORMALIZE: {}'.format(NORMALIZE))
print('LEARNING RATE: {}'.format(LEARNING_RATE))
print('IMAGE NAME: {}'.format(IMAGE_NAME))


neural_painter = Generator(len(ACTIONS_TO_IDX), 64, 3).to(device)
neural_painter.load_state_dict(torch.load('sgan/sdcgan50_4b2_fc_10.tar'))


# Normalization expected by GoogleNet (images scaled to (-1, 1))
normalizer = Normalization(torch.tensor([0.5, 0.5, 0.5]).to(device),
                           torch.tensor([0.5, 0.5, 0.5]).to(device))

# Define image augmentations
padder = nn.ConstantPad2d(12, 0.5)
rand_crop_8 = RandomCrop(8)
rand_scale = RandomScale([1 + (i - 5) / 50. for i in range(11)])
random_rotater = RandomRotate(angle=5, same_throughout_batch=True)
rand_crop_4 = RandomCrop(4)

# Content layer
# feature_extractor_vgg = nn.Sequential(*list(vgg19.children())[:1])
# feature_extractor_vgg.eval().to(device)
feature_extractor = nn.Sequential(*list(inception_v1.children())[:LAYER_IDX])
feature_extractor.eval().to(device)
feature_extractor_res = nn.Sequential(*list(resnet18.children())[:4])
feature_extractor_res.eval().to(device)

# Define canvas and action preprocessor

action_preprocessor = torch.sigmoid  # torch.sigmoid is the default action preprocessor
# action_preprocessor = nn.Sequential(*list(resnet18.children())[:9], nn.Linear(in_features=1024, out_features=560, bias=True))
canvas = []
for i in range(0, 4):
    canvas.append(NeuralCanvasStitched(neural_painter=neural_painter, overlap_px=32,
                              repeat_h=2**(i+1)-1, repeat_w=2**(i+1)-1,
                              strokes_per_block=STROKES_PER_BLOCK,
                              action_preprocessor=action_preprocessor))

# Load input image
image_o = Image.open(IMAGE_NAME)
loader = transforms.Compose([
    # transforms.Resize([canvas.final_canvas_h, canvas.final_canvas_w]),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
image_o = loader(image_o).unsqueeze(0)[:, :3, :, :].to(device, torch.float)
image_o = pad(image_o, 512, 512)
torchvision.utils.save_image(image_o, 'results/new_canvases_tp2_200/flower1.png')
# Load input image mask
mask = Image.open(mask)
loader = transforms.Compose([
    # transforms.Resize([canvas.final_canvas_h, canvas.final_canvas_w]),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
mask = loader(mask).unsqueeze(0)[:, :3, :, :].to(device, torch.float)
mask = pad(mask, 512, 512)
mask_bg = 1 - mask
torchvision.utils.save_image(mask, 'results/new_canvases_tp2_200/flower1_mask.png')
torchvision.utils.save_image(mask_bg, 'results/new_canvases_tp2_200/flower1_mask_bg.png')
# extract the edge of the foreground
img = cv2.imread(IMAGE_NAME)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
edges = torch.from_numpy(edges)
edges = torch.unsqueeze(edges, 0)
edges = torch.unsqueeze(edges, 0)
edges = pad(edges, 512, 512)
edges = edges.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# image = torch.cat((image, image, image), 1)
output_canvas_o = torch.ones(1, 3, 512, 512).to(device)
# output_canvas = output_canvas_o - (edges/255)
output_canvas = output_canvas_o
image = image_o*mask
image_bg = image_o*mask_bg

# loss_fn = torch.nn.SmoothL1Loss()
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.L1Loss()

# f = open('./loss/loss.txt', 'w')
n_pt = 100   # number of painting times
intermediate_canvases = []
intermediate_canvases_bg = []
intermediate_paint = []
for k in range(1, 4):
    output_canvas = F.interpolate(output_canvas, (64 * (2 ** k), 64 * (2 ** k)))
    temp_canvas = output_canvas
    image = F.interpolate(image_o, (64 * (2 ** k), 64 * (2 ** k)))
    actions = torch.FloatTensor(canvas[k].total_num_strokes, 1, len(ACTIONS_TO_IDX)).uniform_().to(device)
    optimizer = optim.Adam([actions.requires_grad_()], lr=LEARNING_RATE)
    for idx in range(n_pt+1):
        optimizer.zero_grad()
        if idx == n_pt:
            output_canvas, intermediate_canvase = canvas[k](actions, temp_canvas.detach(), True)
        else:
            output_canvas= canvas[k](actions, temp_canvas.detach())[0]
        # Everything else below is for calculating the loss function for intrinsic style transfer
        stacked_canvas = torch.cat([output_canvas, image])
        augmented_canvas = stacked_canvas
        # Pass through pretrained
        output_features = feature_extractor(augmented_canvas)
        output_features_res = feature_extractor_res(augmented_canvas)
        # output_features_vgg = feature_extractor_vgg(augmented_canvas)
        # cost = loss_fn(output_features[0], output_features[1])
        # cost = loss_fn(output_features_res[0], output_features_res[1])
        cost2 = loss_fn(stacked_canvas[0], stacked_canvas[1])
        cost = loss_fn(output_features[0], output_features[1]) * 0.1 + loss_fn(output_features_res[0], output_features_res[1]) * 0.9
        # cost = torch.pow(output_features[0] - output_features[1], 2).mean()  # L2 loss
        # cost = torch.abs(output_features[0] - output_features[1]).mean() * 0.8 + torch.abs(output_features_vgg[0] - output_features_vgg[1]).mean() * 0.2  # L1 loss
        # cost = torch.abs(output_features[0] - output_features[1]).mean()  # L1 loss
        # cost = cost + torch.abs(stacked_canvas[0] - stacked_canvas[1]).mean()  # pixel loss
#         f.write(str(idx)+"\t"+str(cost.item())+"\t"+str(cost2.item())+"\n")
        cost.backward()
        optimizer.step()
        if idx % 10 == 0:
            print(f'k {k}\tStep {idx}\tCost {cost.item()}\tCost2 {cost2.item()}')
            torchvision.utils.save_image(output_canvas, 'results/tp2_200_flower1_new/resa' + str(k) + '_' + str(idx) + '.png')
    intermediate_canvases.extend(intermediate_canvase)
for k in range(1, 4):
    output_canvas_o = F.interpolate(output_canvas_o, (64 * (2 ** k), 64 * (2 ** k)))
    temp_canvas = output_canvas_o
    image = F.interpolate(image_o, (64 * (2 ** k), 64 * (2 ** k)))
    actions = torch.FloatTensor(canvas[k].total_num_strokes, 1, len(ACTIONS_TO_IDX)).uniform_().to(device)
    optimizer = optim.Adam([actions.requires_grad_()], lr=LEARNING_RATE)
    for idx in range(n_pt+1):
        optimizer.zero_grad()
        if idx == n_pt:
            output_canvas_o, intermediate_canvase = canvas[k](actions, temp_canvas.detach(), True)
        else:
            output_canvas_o= canvas[k](actions, temp_canvas.detach())[0]
        # Everything else below is for calculating the loss function for intrinsic style transfer
        stacked_canvas = torch.cat([output_canvas_o, image])
        augmented_canvas = stacked_canvas
        # Pass through pretrained
        output_features = feature_extractor(augmented_canvas)
        output_features_res = feature_extractor_res(augmented_canvas)
        # output_features_vgg = feature_extractor_vgg(augmented_canvas)
        # cost = loss_fn(output_features[0], output_features[1])
        # cost = loss_fn(output_features_res[0], output_features_res[1])
        cost2 = loss_fn(stacked_canvas[0], stacked_canvas[1])
        cost = loss_fn(output_features[0], output_features[1]) * 0.1 + loss_fn(output_features_res[0], output_features_res[1]) * 0.9
        # cost = torch.pow(output_features[0] - output_features[1], 2).mean()  # L2 loss
        # cost = torch.abs(output_features[0] - output_features[1]).mean() * 0.8 + torch.abs(output_features_vgg[0] - output_features_vgg[1]).mean() * 0.2  # L1 loss
        # cost = torch.abs(output_features[0] - output_features[1]).mean()  # L1 loss
        # cost = cost + torch.abs(stacked_canvas[0] - stacked_canvas[1]).mean()  # pixel loss
#         f.write(str(idx)+"\t"+str(cost.item())+"\t"+str(cost2.item())+"\n")
        cost.backward()
        optimizer.step()
        if idx % 10 == 0:
            print(f'k {k}\tStep {idx}\tCost {cost.item()}\tCost2 {cost2.item()}')
            torchvision.utils.save_image(output_canvas_o, 'results/tp2_200_flower1_new/resb30' + str(k) + '_' + str(idx+n_pt) + '.png')
    intermediate_canvases_bg.extend(intermediate_canvase)
n = len(intermediate_canvases)
print("n=",n)

mask = mask.cpu()
mask_bg = mask_bg.cpu()
n1,n2,n3 = 0,0,0
s1 = STROKES_PER_BLOCK*9+1
s2 = STROKES_PER_BLOCK*49+1
s3 = STROKES_PER_BLOCK*225+1
n1 = s1
n2 = n1 + s2
n3 = n2 + s3
print("n1=",n1)
print("n2=",n2)
print("n3=",n3)
intermediate_canvases_out = []

for idx in range(n1):
    intermediate_canvases[idx] = intermediate_canvases[idx]*mask
    intermediate_canvases[idx] = intermediate_canvases[idx] + mask_bg
    intermediate_canvases_bg[idx] = intermediate_canvases_bg[idx]*mask_bg
    intermediate_canvases_bg[idx] = intermediate_canvases_bg[idx] + intermediate_canvases[n1-1]*mask 
    intermediate_canvases_out.append(intermediate_canvases[idx])
   
for idx in range(n1):
    intermediate_canvases_out.append(intermediate_canvases_bg[idx])
   
for idx in range(n1, n2):
    intermediate_canvases[idx] = intermediate_canvases[idx]*mask
    intermediate_canvases[idx] = intermediate_canvases[idx] + intermediate_canvases_bg[n1-1]*mask_bg
    intermediate_canvases_bg[idx] = intermediate_canvases_bg[idx]*mask_bg
    intermediate_canvases_bg[idx] = intermediate_canvases_bg[idx] + intermediate_canvases[n2-1]*mask 
    intermediate_canvases_out.append(intermediate_canvases[idx])
 
for idx in range(n1, n2):
    intermediate_canvases_out.append(intermediate_canvases_bg[idx])    
    
for idx in range(n2, n3):
    intermediate_canvases[idx] = intermediate_canvases[idx]*mask
    intermediate_canvases[idx] = intermediate_canvases[idx] + intermediate_canvases_bg[n2-1]*mask_bg
    intermediate_canvases_bg[idx] = intermediate_canvases_bg[idx]*mask_bg
    intermediate_canvases_bg[idx] = intermediate_canvases_bg[idx] + intermediate_canvases[n3-1]*mask 
    intermediate_canvases_out.append(intermediate_canvases[idx])
 
for idx in range(n2, n3):
    intermediate_canvases_out.append(intermediate_canvases_bg[idx])  
    

print(len(intermediate_canvases_out))
m = len(intermediate_canvases_out)
for idx in range(m):
    torchvision.utils.save_image(intermediate_canvases_out[idx], 'results/stroke_canvases_layer_flower_3b02/' + str(idx) + '.png')
# intermediate_canvases.extend(intermediate_canvases_bg)
animate_strokes_on_canvas(intermediate_canvases_out, image_o, "results/layer_200_flower4b2fc_new.mp4", skip_every_n=1)
