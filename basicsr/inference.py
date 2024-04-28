import argparse
import cv2
import glob
import numpy as np
import os
import torch
import tqdm
from skimage import img_as_ubyte
from natsort import natsorted
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from models.archs.dark_arch import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        
        default=  # noqa: E251
        'experiments/Enhancement_DarkNet_lol/models/net_g_3000.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='Enhancement/Datasets/test/Lol/input', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/Enhancement_test', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = DarkNet()
    model.load_state_dict(torch.load(args.model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)
    print("Model Info: ",model)
    print("Number of Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    os.makedirs(args.output, exist_ok=True)
    image_files = natsorted(glob.glob(os.path.join(args.input, '*')))

    # check if the list is empty
    if not image_files:
        print(f"No images found in the input directory: {args.input}")
    else:
        print(f"Found {len(image_files)} images in the input directory: {args.input}")
    with torch.no_grad():
        for filepath in tqdm.tqdm(image_files):
            # print(file_)
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
            input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).cuda()

            # Pad the input if not_multiple_of 4
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+4)//4)*4, ((w+4)//4)*4
            padh = H-h if h%4!=0 else 0
            padw = W-w if w%4!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            restored = model(input_)
            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:,:,:h,:w]

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])

            filename = os.path.split(filepath)[-1]
            cv2.imwrite(os.path.join(args.output, filename),cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))


            
    # for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '')))):
    #     imgname = os.path.splitext(os.path.basename(path))[0]
    #     print('Testing', idx, imgname)
    #     # read image
    #     img = cv2.imread(path, cv2.IMREAD_COLOR)
    #     if img is None:
    #         print(f"Failed to load image at {path}")
    #         continue
    #     img = img.astype(np.float32) / 255.
    #     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    #     img = img.unsqueeze(0).to(device)
    #     # inference
    #     try:
    #         with torch.no_grad():
    #             output = model(img)
    #     except Exception as error:
    #         print('Error', error, imgname)
    #     else:
    #         # save image
    #         output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    #         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    #         output = (output * 255.0).round().astype(np.uint8)
    #         cv2.imwrite(os.path.join(args.output, f'{imgname}Example.png'), output)


if __name__ == '__main__':
    main()