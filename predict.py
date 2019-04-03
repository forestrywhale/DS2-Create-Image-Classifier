# Imports here
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import argparse

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()
parser.add_argument('imagepwd', help='Image path.')
parser.add_argument('checkpointpwd', help='Saved checkpoint file path.', default = 'checkpoint.pth')
parser.add_argument('--top_k', help='Top predicted categories.', default = 5)
parser.add_argument('--category_names', help='Mapping flower names.', default = None)
parser.add_argument('--gpu', help='Option for GPU computing.', default = True)
args = parser.parse_args()

def load_previous_model(filename, lr = 0.001):
    
    loaddata = torch.load(filename)
    #print(loaddata.keys())
    #print(loaddata['class_to_idx'])
    
    model = models.vgg16(pretrained=True)
    model.classifier = loaddata['classifier']
    model.load_state_dict(loaddata['state_dict'])
    model.class_to_idx = loaddata['class_to_idx']
    if args.gpu:
        model.to('cuda')
    else:
        model.to('cpu')

    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    return model

newmodel = load_previous_model(args.checkpointpwd)

# here is image preprocessing 
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # open the image
    im = Image.open(image)
    im.show()

    # obtain figure size
    width, height = im.size
    #print(width, height )
    
    # center crop
    new_width = min(width, height)
    new_height = min(width, height)
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    
    im = im.crop((left, top, right, bottom))
    
    # resize
    im = im.resize((224,224))
    np_im = np.array(im)
    #print(np_im[0][0])
    
    # normalization
    means = np.asarray([0.485, 0.456, 0.406])
    stds = np.asarray([0.229, 0.224, 0.225])

    # normalize between 0-1
    norm_im = np_im/255.0
    #print(norm_im)
    
    # normalize with standard normal distribution
    norm_im = (norm_im-means)/stds
    #print(norm_im)
    
    # final transpose
    final_im = norm_im.transpose((2,0,1))

    return final_im

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # to the right dimention
    imagenp = process_image(image_path)
    # to a tensor
    imagetensor = torch.from_numpy(imagenp)
    imagetensor.unsqueeze_(0)
    imagein = imagetensor.cuda().float()
    
    model.eval()
    
    with torch.no_grad():

        score = model.forward(imagein)
        prob, idxs = torch.topk(score, topk)
        #print(score, prob, idxs )
        
        # calc probability
        prob = torch.exp(prob)
        
        return prob, idxs
    
prob, names = predict(args.imagepwd,newmodel, args.top_k)

# change indices to class names
# import mapping
import json
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
   
    y_names = []
    yn = names[0].cpu().numpy()
    for item in yn:
        flowername = cat_to_name[str(item)]
        y_names.append(flowername)
    names = y_names

print(prob)
print(names)