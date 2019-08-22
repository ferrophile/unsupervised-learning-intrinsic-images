import os
import numpy as np
from skimage import io
from skimage.transform import resize

import torch
from torch.autograd import Variable

from models import saw_utils
from models.pix2pix_model import Pix2PixModel
from options.decomp_options import DecompOptions

opt = DecompOptions().parse()
model = Pix2PixModel(opt)

img_path = opt.decomp_image
img_fn = os.path.splitext(os.path.split(img_path)[-1])[0]
saw_img = np.asarray(io.imread(img_path)).astype(float) / 255.0

original_h, original_w = saw_img.shape[0], saw_img.shape[1]
saw_img = saw_utils.resize_img_arr(saw_img)
saw_img = np.transpose(saw_img, (2, 0, 1))
input_ = torch.from_numpy(saw_img).unsqueeze(0).contiguous().float()
input_images = Variable(input_.cuda() , requires_grad = False)

prediction_S, prediction_R, rgb_s = model.netG.forward(input_images)

prediction_Sr = torch.exp(prediction_S)
prediction_S_np = prediction_Sr.data[0,0,:,:].cpu().numpy()
prediction_S_np = resize(prediction_S_np, (original_h, original_w), order=1, preserve_range=True)

prediction_S_np = np.clip(prediction_S_np * 0.5, 0, 1)
io.imsave('images/{}_ps.png'.format(img_fn), prediction_S_np)

prediction_Rr = torch.exp(prediction_R)
prediction_R_np = np.transpose(prediction_Rr.data[0,:,:,:].cpu().numpy(), (1, 2, 0))
prediction_R_np = resize(prediction_R_np, (original_h, original_w), order=1, preserve_range=True)
io.imsave('images/{}_pr.png'.format(img_fn), prediction_R_np)
