import cv2
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap
import os

'''
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install --ignore-installed imageio
pip install insightface==0.2.1 onnxruntime moviepy

1. Face detection and align model
https://onedrive.live.com/?authkey=%21ADJ0aAOSsc90neY&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215837&parId=4A83B6B633B029CC%215834&action=locate
-> insightface_func/models/antelope/*.onnx

2. Face parsing model
https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?usp=sharing
-> parsing_model/checkpoint/79999_iter.pth

3. Face recognition model and SimSwap pretrained model
https://drive.google.com/drive/folders/1jV6_0FIMPC53FZ2HzZNJZGMe55bbu17R
-> arcface_model/arcface_checkpoint.tar
-> checkpoints/people/latest_net_G.pth
'''

transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.pic_a_path = 'demo_file/yoon.jpg' # source
    opt.video_path = 'demo_file/moon.mp4' # target

    opt.use_mask = True
    opt.Arc_path = 'arcface_model/arcface_checkpoint.tar'

    opt.output_path = os.path.join('output', '%s_%s.mp4' % (os.path.basename(opt.pic_a_path), os.path.basename(opt.video_path)))

    model = create_model(opt)
    model.eval()

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    with torch.no_grad():
        img_a_whole = cv2.imread(opt.pic_a_path)
        img_a_align_crop, _ = app.get(img_a_whole, opt.crop_size)

        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
        img_a = transformer_Arcface(img_a_align_crop_pil)

        img_id = img_a.unsqueeze(0)
        img_id = img_id.cuda()

        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample) # [1, 512]
        latend_id = F.normalize(latend_id, p=2, dim=1)

        video_swap(opt.video_path, latend_id, model, app, opt.output_path, temp_results_dir=opt.temp_path,\
            no_simswaplogo=True, use_mask=opt.use_mask, crop_size=opt.crop_size)
