import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import os
import argparse
from tqdm import tqdm

# ----------------- ARGUMENT PARSER -----------------
parser = argparse.ArgumentParser(description="Neural Style Transfer")
parser.add_argument("--content", type=str, default="images/content.jpg", help="Path to content image")
parser.add_argument("--style", type=str, default="images/style.jpg", help="Path to style image")
parser.add_argument("--output", type=str, default="output/result.jpg", help="Path to save output image")
parser.add_argument("--size", type=int, default=512, help="Resize size for images")
parser.add_argument("--steps", type=int, default=300, help="Number of optimization steps")
args = parser.parse_args()

# ----------------- CONFIG -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = args.size

# ----------------- UTILS -----------------
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])
unloader = transforms.ToPILImage()

def image_loader(path):
    image = Image.open(path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# ----------------- LOAD IMAGES -----------------
content_img = image_loader(args.content)
style_img = image_loader(args.style)

assert content_img.size() == style_img.size(), "Content and style images must be same size"

# ----------------- MODEL & FEATURES -----------------
cnn = models.vgg19(pretrained=True).features.to(device).eval()
for param in cnn.parameters():
    param.requires_grad = False

content_layers = ['21']
style_layers = ['0','5','10','19','28']

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

def get_features(x):
    content_feats = {}
    style_feats = {}
    cur = x
    for name, layer in cnn._modules.items():
        cur = layer(cur)
        if name in content_layers:
            content_feats[name] = cur
        if name in style_layers:
            style_feats[name] = cur
    return content_feats, style_feats

content_feats, _ = get_features(content_img)
_, style_feats = get_features(style_img)
style_grams = {layer: gram_matrix(style_feats[layer]) for layer in style_feats}

# ----------------- INPUT IMAGE -----------------
input_img = content_img.clone().requires_grad_(True)

# ----------------- LOSS WEIGHTS -----------------
content_weight = 1e4
style_weight = 1e2

optimizer = optim.LBFGS([input_img])

# ----------------- OPTIMIZATION -----------------
print("Optimizing…")
run = [0]
while run[0] <= args.steps:
    def closure():
        input_img.data.clamp_(0,1)
        optimizer.zero_grad()
        c_feats, s_feats = get_features(input_img)
        content_loss = content_weight * nn.MSELoss()(c_feats[content_layers[0]], content_feats[content_layers[0]])
        style_loss = 0
        for layer in style_layers:
            g = gram_matrix(s_feats[layer])
            style_loss += nn.MSELoss()(g, style_grams[layer])
        style_loss *= style_weight
        loss = content_loss + style_loss
        loss.backward()
        if run[0] % 50 == 0:
            print(f"Iteration {run[0]}: content {content_loss.item():.2f}, style {style_loss.item():.2f}")
        run[0] += 1
        return loss
    optimizer.step(closure)

# ----------------- SAVE OUTPUT -----------------
input_img.data.clamp_(0,1)
if not os.path.exists(os.path.dirname(args.output)):
    os.makedirs(os.path.dirname(args.output))
unloader(input_img.squeeze(0).cpu()).save(args.output)
print(f"Output saved at {args.output}")
