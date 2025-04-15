import os
import time
import argparse

from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms

from models.DehazeXL.decoders.decoder import DehazeXL


def unnormalize(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'Using device: {device}')

    model = DehazeXL().to(device)

    if os.path.isfile(args.model_path):
        model_path = [args.model_path]
    else:
        model_path = os.listdir(args.model_path)

    cloud_img_list = []
    folder_flag = 0
    if not os.path.exists(args.test_img):
        raise FileExistsError("Please input path of test img.")
    if os.path.isfile(args.test_img):
        cloud_img_list.append(args.test_img)
    else:
        folder_flag = 1
        cloud_img_list = os.listdir(args.test_img)

    for i, m in tqdm(enumerate(model_path)):
        m = m if os.path.exists(m) else os.path.join(args.model_path, m)
        print("{}/{} - Loading model: {}".format(i, len(model_path), m))
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(m, map_location=torch.device('cpu'), weights_only=True))
        else:
            model.load_state_dict(torch.load(m, weights_only=True))
        if args.fp16:
            model.half()
            print("FP16 mode has been turned on")
        model.eval()
        if args.test_img is None:
            raise FileExistsError("Please input path of test img.")
        start_time = time.time()
        for path in cloud_img_list:
            os.makedirs(os.path.join(args.save_dir, f'{os.path.basename(path)}'), exist_ok=True)
            save_dir = os.path.join(args.save_dir, f'{os.path.basename(path)}')
            print(save_dir)
            if folder_flag == 0:
                cloud_img = Image.open(path).convert('RGB')
            else:
                cloud_img = Image.open(os.path.join(args.test_img, path)).convert('RGB')
            if args.uni_size_w > 0 and args.uni_size_h > 0:
                cloud_img = cloud_img.resize((args.uni_size_w, args.uni_size_h))
            transforms_list = [
                transforms.ToTensor()
            ]
            if args.nor:
                transforms_list.append(transforms.Normalize([0.48694769, 0.53490934, 0.45040282],
                                                            [0.16451647, 0.15308164, 0.15011494]))
            transform = transforms.Compose(transforms_list)
            print("Cloud removing...")
            with torch.no_grad():
                input_tensor = transform(cloud_img).unsqueeze(0).to(device)
                if args.fp16:
                    input_tensor = input_tensor.half()
                out = model(input_tensor)
                if args.nor:
                    out = unnormalize(out,
                                      [0.48694769, 0.53490934, 0.45040282],
                                      [0.16451647, 0.15308164, 0.15011494])
                res = transforms.ToPILImage()(out.clamp(0, 1).squeeze(0))
            print("Saving result...")
            res.save(os.path.join(save_dir, f'{os.path.basename(m)}_{os.path.basename(path)}'))
        end_time = time.time()
        avg_time = (end_time - start_time) / len(cloud_img_list)
        print("Done. Avg Time:{}".format(avg_time))


if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser(description='Test a model.')
    parser.add_argument('--test_img', type=str, default=r'./datasets/8KDehaze_mini/cloud_L1/16s14e09_c50y15_0.png',
                        help='Path to dataset')
    parser.add_argument('--model_path', type=str, default=r'./checkpoints/conv_best1.pth',
                        help='Path to pretrained model')
    parser.add_argument('--save_dir', type=str, default=r'./res',
                        help='Path to save predict results')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--uni_size_w', type=int, default=4096,
                        help='Unified input image size w')
    parser.add_argument('--uni_size_h', type=int, default=4096,
                        help='Unified input image size h')
    parser.add_argument('--nor', action='store_true',
                        help='Normalize input image')
    parser.add_argument('--fp16', action='store_true',
                        help='Loading models and data in FP16 format')
    main(parser.parse_args())
