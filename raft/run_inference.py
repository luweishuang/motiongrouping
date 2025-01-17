import os
import glob as gb
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../models/ckpt_fbms.pth")
    args = parser.parse_args()
    # data_path = "/data/pfc/motionGrouping/DAVIS2016"
    # data_path = "/data/pfc/motionGrouping/ieemoo"
    data_path = args.data_path
    gap = [1, 2]
    reverse = [0, 1]
    rgbpath = os.path.join(data_path, 'JPEGImages')  # path to the dataset
    folder = gb.glob(os.path.join(rgbpath, '*'))

    cnt = 0
    for r in reverse:
        for g in gap:
            for f in folder:
                # print('===> Runing {}, gap {}'.format(f, g))
                mode = 'raft-things.pth'  # model
                if r==1:
                    raw_outroot = data_path + '/Flows_gap-{}/'.format(g)  # where to raw flow
                    outroot = data_path + '/FlowImages_gap-{}/'.format(g)  # where to save the image flow
                elif r==0:
                    raw_outroot = data_path + '/Flows_gap{}/'.format(g)   # where to raw flow
                    outroot = data_path + '/FlowImages_gap{}/'.format(g)   # where to save the image flow
                print("python predict.py "
                        "--gap {} --model {} --path {} "
                        "--outroot {} --reverse {} --raw_outroot {}".format(g, mode, f, outroot, r, raw_outroot))
                os.system("python predict.py "
                        "--gap {} --model {} --path {} "
                        "--outroot {} --reverse {} --raw_outroot {}".format(g, mode, f, outroot, r, raw_outroot))
                cnt += 1
    print(cnt)
