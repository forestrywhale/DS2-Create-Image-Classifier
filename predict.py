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

def load_previous_model(filename):
    
    loaddata = torch.load(filename)

    model_name = loaddata['model_name']
    #print(model_name)
    if model_name == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
        
    model.input_size = loaddata['input_size']
    model.output_size = loaddata['output_size']
    model.hidden_layers = loaddata['hidden_layers']
    model.batch_size = loaddata['batch_size']
    model.learning_rate = loaddata['learning_rate']
    model.classifier = loaddata['classifier']
    model.load_state_dict(loaddata['state_dict'])
    model.epoch = loaddata["epoch"]
    model.optimizer = loaddata['optimizer']
    model.class_to_idx = loaddata['class_to_idx']
    model.to('cuda')

    optimizer = optim.Adam(model.classifier.parameters(), model.learning_rate)
    
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

# Reverse the dict
idx_to_class = {val: key for key, val in newmodel.class_to_idx.items()}
# Get the correct indices
top_classes = [idx_to_class[each] for each in names[0].cpu().numpy()]
# get the corresponding names
import json
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[each] for each in top_classes]

print(prob.cpu().numpy())
print(names.cpu().numpy())