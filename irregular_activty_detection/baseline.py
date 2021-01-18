###############################################################################
# Libraries
###############################################################################
import os
import glob
from PIL import Image
import json
import numpy as np

import torch
from models import resnet
from preprocess_data import scale_crop

import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
###############################################################################


###############################################################################
# Functions
###############################################################################
print('Load functions...')
class Options:
    def __init__(self):
        self.input_type = 'rgb'
        self.sample_duration = 16
        self.batch_size = 1
        self.n_threads = 1
        self.sample_size = 112

        # For models
        self.model = 'resnet'
        self.model_depth = 50
        self.n_classes = 5
        self.n_input_channels = 3
        self.resnet_shortcut = 8
        self.conv1_t_size = 7
        self.conv1_t_stride = 1
        self.no_max_pool = False
        self.resnet_widen_factor = 1.0

        self.arch = '{}-{}'.format(self.model, self.model_depth)

def get_clip(clip_dir, opt):
    frames = os.listdir(clip_dir)

    imgs = []

    start_idx = np.random.randint(0, len(frames) - opt.sample_duration)

    for idx, frame in enumerate(frames):
        if idx < start_idx or idx >= start_idx + opt.sample_duration:
            continue

        img_path = os.path.join(clip_dir, frame)
        img = Image.open(img_path)
        imgs.append(img.copy())
        img.close()

        # img = Image.open(img_path)
        # imgs.append(img)

    clip = scale_crop(imgs, 0, opt)
    clip = torch.unsqueeze(clip, 0)

    return clip

def generate_model(opt):
    model = resnet.generate_model(model_depth=opt.model_depth,
                                  n_classes=opt.n_classes,
                                  n_input_channels=opt.n_input_channels,
                                  shortcut_type=opt.resnet_shortcut,
                                  conv1_t_size=opt.conv1_t_size,
                                  conv1_t_stride=opt.conv1_t_stride,
                                  no_max_pool=opt.no_max_pool,
                                  widen_factor=opt.resnet_widen_factor)

    return model

def load_model(model_path, arch, model):
    print('loading checkpoint {} model'.format(model_path))
    checkpoint = torch.load(model_path, map_location='cuda:0')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model
###############################################################################



###############################################################################
# Parameters
###############################################################################
print('Load parameters and set model...')
model_path = 'baseline/baseline.pth'
train_path = 'dataset/train'
test_path = 'dataset/test'
submit_file = 'submit.json'

opt = Options()
model = generate_model(opt)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
###############################################################################



###############################################################################
# Create Dataset
###############################################################################
print("Create datasets...")
X = []
Y = []
folders = glob.glob(train_path + '/*')
for val, folder in enumerate(folders):
    subfolders = glob.glob(folder + '/*')
    for subfolder in subfolders:
        Y.append(val)
        X.append(get_clip(subfolder, opt))

X = torch.tensor(np.concatenate(X, axis=0), dtype=torch.float32)
Y = torch.tensor(Y)
dataset = TensorDataset(X, Y)
train_dataset, val_dataset = random_split(dataset, [300, 75])
train_loader = DataLoader(dataset=train_dataset, batch_size=15)
val_loader = DataLoader(dataset=val_dataset, batch_size=15)
###############################################################################



###############################################################################
# Train
###############################################################################
print("Training...")
losses = []
val_losses = []
for epoch in range(1, 21):
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss)
    print(f'Epoch: {epoch}, Loss: {running_loss:.5f}')

torch.save(model.state_dict(), model_path)

###############################################################################



###############################################################################
# Create Predictions
###############################################################################
print("Making predictions...")
opt = Options()
model = generate_model(opt)
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
model.to(device)
clips = os.listdir(test_path)

result = dict()

with torch.no_grad():
    model.eval()

    for clip in clips:
        clip_dir = os.path.join(test_path, clip)
        input = get_clip(clip_dir, opt)
        input = input.to(device)

        outputs = model(input)
        outputs = F.softmax(outputs, dim=1).cpu()
        sorted_scores, label = torch.topk(outputs, k=1)

        result[clip] = label.squeeze().item()

with open(submit_file, "w") as json_file:
    json.dump(result, json_file)
###############################################################################