import os
import time
import einops
import sys
import cv2
import utils as ut
import torch
import torch.optim as optim
from argparse import ArgumentParser
from model import SlotAttentionAutoEncoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE = ", DEVICE)


def infer_single(model, resultsPath=None):
    if args.dataset == "DAVIS":
        h = 480
        w = 854
    elif args.dataset == "ieemoo":
        h = 480
        w = 640

    with torch.no_grad():
        t = time.time()
        print(' --> running inference')
        all_f = os.listdir(test_video)
        all_f.sort()
        index = 0
        for cur_f in all_f:
            cur_img = os.path.join(test_video, cur_f)
            print(' --> running inference, index = %s' % index)

            if DEVICE == "cuda":
                flows = flows.float().to(DEVICE)  # b t c h w
            else:
                flows = flows.float()

            # run inference
            flows = einops.rearrange(flows, 'b t c h w -> (b t) c h w')
            recon_image, recons, masks, _ = model(flows)  # t s 1 h w
            masks = einops.rearrange(masks, '(b t) s c h w -> b t s c h w', t=4)
            for i in range(masks.size()[0]):
                save_masks, mean_mask = ut.ensemble_hungarian_iou_noGT(masks[i], h, w)

                for mi, save_mask in enumerate(save_masks):
                    fgp = fgap[mi][0]
                    save_mask = einops.rearrange(save_mask, 'c h w -> h w c')
                    os.makedirs(os.path.join(resultsPath, category, fgp), exist_ok=True)
                    cv2.imwrite(os.path.join(resultsPath, category, fgp, index), (save_mask * 255.))
                mean_mask = einops.rearrange(mean_mask, 'c h w -> h w c')
                os.makedirs(os.path.join(resultsPath, category, 'mean'), exist_ok=True)
                cv2.imwrite(os.path.join(resultsPath, category, 'mean', index), (mean_mask * 255.))
        print(' --> inference, time {}'.format(time.time()-t))


def main(args):
    # step1: get raft feature
    # test_path = "../results/data"
    # os.system("python ../raft/run_inference.py --data_path=%s" % test_path)

    # step2: do motion group infer
    lr = args.lr
    num_slots = args.num_slots
    iters = args.num_iterations
    resume_path = args.resume_path
    resolution = (128, 224)
    in_out_channels = 3 if args.flow_to_rgb else 2
    use_flow = False if args.flow_to_rgb else True

    resultsPath = os.path.join('../results/', args.resume_path.split('/')[-1])
    os.makedirs(resultsPath, exist_ok=True)

    print("initialize model ")
    model = SlotAttentionAutoEncoder(resolution=resolution,
                                     num_slots=num_slots,
                                     in_out_channels=in_out_channels,
                                     iters=iters)
    if DEVICE == "cuda":
        print("model to ", DEVICE)
        model.to(DEVICE)

    print("initialize training")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    it = 0
    if resume_path:
        print('resuming from checkpoint')
        if DEVICE == "cuda":
            checkpoint = torch.load(resume_path)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()
    else:
        print('no checkpouint found')
        sys.exit(0)

    print('======> start inference {}, {}, use {}.'.format(args.dataset, args.verbose, DEVICE))
    # evaluate on single image
    infer_single(model, resultsPath=resultsPath)


if __name__ == "__main__":
    parser = ArgumentParser()
    #optimization
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_train_steps', type=int, default=5e9)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--decay_steps', type=int, default=8e4)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    #settings
    parser.add_argument('--dataset', type=str, default='ieemoo', choices=['DAVIS', 'MoCA', 'FBMS', 'STv2', 'ieemoo'])
    parser.add_argument('--with_rgb', action='store_true')
    parser.add_argument('--flow_to_rgb', action='store_true')
    parser.add_argument('--inference', action='store_true')
    #architecture
    parser.add_argument('--num_slots', type=int, default=2)
    parser.add_argument('--num_iterations', type=int, default=5)
    #misc
    parser.add_argument('--verbose', type=str, default=None)
    parser.add_argument('--resume_path', type=str, default="../models/ckpt_fbms.pth")
    args = parser.parse_args()
    args.inference = True
    main(args)

