import os

# src_dir = "/data/pfc/motionGrouping/ieemoo/JPEGImages"
src_dir = "/data/motionGrouping/data/ieemoo/JPEGImages"
dst_dir = src_dir + "_new"
os.makedirs(dst_dir, exist_ok=True)
for sub in os.listdir(src_dir):
    sub_path = os.path.join(src_dir, sub)
    sub_path_dst = os.path.join(dst_dir, sub)
    os.makedirs(sub_path_dst, exist_ok=True)
    for cur_f in os.listdir(sub_path):
        cur_img = os.path.join(sub_path, cur_f)
        cur_img_dst = os.path.join(sub_path_dst, cur_f.replace(".png", ".jpg"))
        os.system("convert %s %s" % (cur_img, cur_img_dst))
