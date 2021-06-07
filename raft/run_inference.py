import os
import glob as gb

data_path = "/data/pfc/motionGrouping/DAVIS2016"
gap = [1, 2]
reverse = [0, 1]
rgbpath = os.path.join(data_path, 'JPEGImages/1080p')  # path to the dataset
folder = gb.glob(os.path.join(rgbpath, '*'))

for r in reverse:
  for g in gap:
    for f in folder:
      print('===> Runing {}, gap {}'.format(f, g))
      mode = 'raft-things.pth'  # model
      if r==1:
        raw_outroot = data_path + '/Flows_gap-{}/'.format(g)  # where to raw flow
        outroot = data_path + '/FlowImages_gap-{}/'.format(g)  # where to save the image flow
      elif r==0:
        raw_outroot = data_path + '/Flows_gap{}/'.format(g)   # where to raw flow
        outroot = data_path + '/FlowImages_gap{}/'.format(g)   # where to save the image flow
      os.system("python predict.py "
                "--gap {} --model {} --path {} "
                "--outroot {} --reverse {} --raw_outroot {}".format(g, mode, f, outroot, r, raw_outroot))
