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
from glob import glob

transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.pic_a_path = glob('demo_file/karina*') # source
    opt.video_path = 'demo_file/moon2.mp4' # target

    opt.use_mask = True
    opt.Arc_path = 'arcface_model/arcface_checkpoint.tar'

    pic_a_str = os.path.basename(opt.pic_a_path[0])
    if len(opt.pic_a_path) >= 2:
        pic_a_str = ''
        for p in opt.pic_a_path:
            pic_a_str += os.path.splitext(os.path.basename(p))[0] + '_'

    opt.output_path = os.path.join('output', '%s_%s.mp4' % (pic_a_str, os.path.basename(opt.video_path)))

    model = create_model(opt)
    model.eval()

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    with torch.no_grad():
        latend_ids = torch.Tensor([]).cuda()
    
        for path in opt.pic_a_path:
            img_a_whole = cv2.imread(path)
            img_a_align_crop, _ = app.get(img_a_whole, opt.crop_size)

            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
            img_a = transformer_Arcface(img_a_align_crop_pil)

            img_id = img_a.unsqueeze(0)
            img_id = img_id.cuda()

            img_id_downsample = F.interpolate(img_id, size=(112, 112))
            latend_id = model.netArc(img_id_downsample) # [1, 512]

            latend_ids = torch.cat((latend_ids, latend_id))

        # if latend_ids.size(0) >= 2:
        latend_ids = torch.mean(latend_ids, dim=0, keepdim=True)
        latend_ids = F.normalize(latend_ids, p=2, dim=1)

        video_swap(opt.video_path, latend_ids, model, app, opt.output_path, temp_results_dir=opt.temp_path,\
            no_simswaplogo=True, use_mask=opt.use_mask, crop_size=opt.crop_size)
