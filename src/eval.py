import os
import time
import einops
import sys
import cv2
import numpy as np
import utils as ut
import config as cg
import torch
import torchvision
import torch.optim as optim
from argparse import ArgumentParser
from model import SlotAttentionAutoEncoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE = ", DEVICE)


def eval_nolabel(val_loader, model, moca, use_flow, it, resultsPath=None, writer=None, train=False):
    with torch.no_grad():
        ious = {}
        single_step_ious = {}
        t = time.time()
        print(' --> running inference')
        for idx, val_sample in enumerate(val_loader):
            flows, gt, meta, fgap = val_sample
            if DEVICE == "cuda":
                flows = flows.float().to(DEVICE)  # b t c h w
                gt = gt.float().to(DEVICE)  # b c h w
            else:
                flows = flows.float()
                gt = gt.float()
            category, index = meta[0][0], meta[1][0]

            if category not in ious.keys():
                ious[category] = []
            if category not in single_step_ious.keys():
                single_step_ious[category] = []
            # run inference
            flows = einops.rearrange(flows, 'b t c h w -> (b t) c h w')
            print("eval run model ")
            recon_image, recons, masks, _ = model(flows)  # t s 1 h w
            masks = einops.rearrange(masks, '(b t) s c h w -> b t s c h w', t=4)

            if train:
                if np.random.random() > 0.95:
                    grid = ut.convert_for_vis(flows.unsqueeze(1), use_flow=use_flow)
                    grid_ri = ut.convert_for_vis(recon_image, use_flow=use_flow).unsqueeze(1)
                    grid_r = ut.convert_for_vis(recons, use_flow=use_flow)
                    grid_m = (255*torch.cat([masks, masks, masks], dim=3)).type(torch.ByteTensor)
                    grid_m = einops.rearrange(grid_m, 'b t s c h w -> (b t) s c h w')
                    grid_all = torch.cat([grid, grid_ri, grid_m, grid_r], dim=1)
                    nrow = grid_all.size()[1]
                    grid_all = einops.rearrange(grid_all, 'b s c h w -> (b s) c h w')
                    grid_all = torchvision.utils.make_grid(grid_all, nrow=nrow)
                    writer.add_image('val/images', grid_all, it+idx)

        frameious = sum(ious.values(), [])
        single_step_frameious = sum(single_step_ious.values(), [])
        frame_mean_iou = sum(frameious) / len(frameious)
        frame_single_step_mean_iou = sum(single_step_frameious) / len(single_step_frameious)
        print(' --> inference, time {}'.format(time.time()-t),
            'acc = {}'.format(np.round(frame_mean_iou, 4)),
            'single_step_acc = {}'.format(np.round(frame_single_step_mean_iou, 4)))
        if train:
            writer.add_scalar('IOU/val_mean', frame_mean_iou, it)
            writer.add_scalar('IOU/val_single_step_mean', frame_single_step_mean_iou, it)
            return frame_mean_iou


def eval(val_loader, model, moca, use_flow, it, resultsPath=None, writer=None, train=False):
    with torch.no_grad():
        ious = {}
        single_step_ious = {}
        t = time.time()
        print(' --> running inference')
        for idx, val_sample in enumerate(val_loader):
            flows, gt, meta, fgap = val_sample
            if DEVICE == "cuda":
                flows = flows.float().to(DEVICE)  # b t c h w
                gt = gt.float().to(DEVICE)  # b c h w
            else:
                flows = flows.float()
                gt = gt.float()
            category, index = meta[0][0], meta[1][0]

            if category not in ious.keys():
                ious[category] = []
            if category not in single_step_ious.keys():
                single_step_ious[category] = []
            # run inference
            flows = einops.rearrange(flows, 'b t c h w -> (b t) c h w')
            print("eval run model ")
            recon_image, recons, masks, _ = model(flows)  # t s 1 h w
            masks = einops.rearrange(masks, '(b t) s c h w -> b t s c h w', t=4)
            for i in range(masks.size()[0]):
                save_masks, mean_mask, iou, single_step_iou = ut.ensemble_hungarian_iou(masks[i], gt[i:i+1], moca)  # t 1 h w

                #append iou
                single_step_ious[category].append(single_step_iou)
                ious[category].append(iou)
                if not train:
                    for mi, save_mask in enumerate(save_masks):
                        fgp = fgap[mi][0]
                        save_mask = einops.rearrange(save_mask, 'c h w -> h w c')
                        os.makedirs(os.path.join(resultsPath, category, fgp), exist_ok=True)
                        cv2.imwrite(os.path.join(resultsPath, category, fgp, index), (save_mask * 255.))
                    mean_mask = einops.rearrange(mean_mask, 'c h w -> h w c')
                    os.makedirs(os.path.join(resultsPath, category, 'mean'), exist_ok=True)
                    cv2.imwrite(os.path.join(resultsPath, category, 'mean', index), (mean_mask * 255.))

            if train:
                if np.random.random() > 0.95:
                    grid = ut.convert_for_vis(flows.unsqueeze(1), use_flow=use_flow)
                    grid_ri = ut.convert_for_vis(recon_image, use_flow=use_flow).unsqueeze(1)
                    grid_r = ut.convert_for_vis(recons, use_flow=use_flow)
                    grid_m = (255*torch.cat([masks, masks, masks], dim=3)).type(torch.ByteTensor)
                    grid_m = einops.rearrange(grid_m, 'b t s c h w -> (b t) s c h w')
                    grid_all = torch.cat([grid, grid_ri, grid_m, grid_r], dim=1)
                    nrow = grid_all.size()[1]
                    grid_all = einops.rearrange(grid_all, 'b s c h w -> (b s) c h w')
                    grid_all = torchvision.utils.make_grid(grid_all, nrow=nrow)
                    writer.add_image('val/images', grid_all, it+idx)

        frameious = sum(ious.values(), [])
        single_step_frameious = sum(single_step_ious.values(), [])
        frame_mean_iou = sum(frameious) / len(frameious)
        frame_single_step_mean_iou = sum(single_step_frameious) / len(single_step_frameious)
        print(' --> inference, time {}'.format(time.time()-t), 
            'acc = {}'.format(np.round(frame_mean_iou, 4)),
            'single_step_acc = {}'.format(np.round(frame_single_step_mean_iou, 4)))
        if train:
            writer.add_scalar('IOU/val_mean', frame_mean_iou, it)
            writer.add_scalar('IOU/val_single_step_mean', frame_single_step_mean_iou, it)
            return frame_mean_iou


def main(args):
    lr = args.lr
    num_slots = args.num_slots
    iters = args.num_iterations
    resume_path = args.resume_path
    batch_size = 1
    args.resolution = (128, 224)

    print("setup log and model path, initialize tensorboard")
    [logPath, modelPath, resultsPath] = cg.setup_path(args)

    print("initialize dataloader")
    trn_dataset, val_dataset, resolution, in_out_channels, use_flow, loss_scale, ent_scale, cons_scale = cg.setup_dataset(args)
    val_loader = ut.FastDataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
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
        it = checkpoint['iteration']
        model.eval()
    else:
        print('no checkpouint found')
        sys.exit(0)

    if args.dataset == "MoCA": 
        moca = True
    else:
        moca = False

    print('======> start inference {}, {}, use {}.'.format(args.dataset, args.verbose, DEVICE))
    # evaluate on validation set
    eval(val_loader, model, moca, use_flow, it, resultsPath=resultsPath, train=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    #optimization
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_train_steps', type=int, default=5e9)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--decay_steps', type=int, default=8e4)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    #settings
    parser.add_argument('--dataset', type=str, default='DAVIS', choices=['DAVIS', 'MoCA', 'FBMS', 'STv2'])
    parser.add_argument('--with_rgb', action='store_true')
    parser.add_argument('--flow_to_rgb', action='store_true')
    parser.add_argument('--inference', action='store_true')
    #architecture
    parser.add_argument('--num_slots', type=int, default=2)
    parser.add_argument('--num_iterations', type=int, default=5)
    #misc
    parser.add_argument('--verbose', type=str, default=None)
    parser.add_argument('--resume_path', type=str, default="../models/ckpt_davis.pth")
    args = parser.parse_args()
    args.inference = True
    main(args)


'''
inference, time 102.64219832420349 acc = [0.5485] single_step_acc = [0.5551]
'''