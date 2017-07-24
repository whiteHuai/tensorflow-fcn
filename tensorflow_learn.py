# import os
# import scipy as scp
# import scipy.misc
#
import numpy as np
import PIL.Image as im
import os
import os.path
import tensorflow as tf
#
# img1 = scp.misc.imread("tabby_cat.png")
# images = tf.placeholder("float")
# feed_dict = {images: img1}
# batch_images = tf.expand_dims(images, 0)
# red, green, blue = tf.split(batch_images, 3, 3)
# sess=tf.Session()
# a=sess.run(batch_images,feed_dict=feed_dict)
# print(a)

# import numpy as np
# import scipy.io as scio
#
# dataFile='coast_arnat59.mat'
# data=scio.loadmat(dataFile)
# print(data)

# def uint82bin(n, count=8):
#     """returns the binary of integer n, count refers to amount of bits"""
#     return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])
#
# def labelcolormap(N):
#     cmap = np.zeros((N, 3), dtype = np.uint8)
#     for i in range(N):
#         r = 0
#         g = 0
#         b = 0
#         id = i
#         for j in range(7):
#             str_id = uint82bin(id)
#             r = r ^ ( np.uint8(str_id[-1]) << (7-j))
#             g = g ^ ( np.uint8(str_id[-2]) << (7-j))
#             b = b ^ ( np.uint8(str_id[-3]) << (7-j))
#             id = id >> 3
#         cmap[i, 0] = r
#         cmap[i, 1] = g
#         cmap[i, 2] = b
#     return cmap
#
# a=labelcolormap(20)
# print(a)


############################################制作FCN标签
d_pix_num={(0,0,0):0,(128,0,0):1,(0,128,0):2,(128,128,0):3,(0,0,128):4,
           (128,0,128):5,(0,128,128):6,(128,128,128):7,(64,0,0):8,
           (192, 0, 0): 9,(64,128,0):10,(192,128,0):11,(64,0,128):12,
           (192, 0, 128): 13,(64,128,128):14,(192,128,128):15,(0,64,0):16,
           (128, 64, 0): 17,(0,192,0):18,(128,192,0):19,(0,64,128):20,
           (224,224,192):0}


def conver2num(img):
    n_img=np.array(img)
    img_shape=n_img.shape
    new_img=np.zeros((img_shape[0],img_shape[1]),dtype=int)
    for h in range(img_shape[0]):
        for w in range(img_shape[1]):
            ind=tuple(n_img[h][w])
            pi=d_pix_num[ind]
            new_img[h][w]=pi
    return new_img






rootdir='/home/white/下载/VOC2012/SegmentationClass'
list1=os.listdir(rootdir)
for i in range(0,len(list1)):
    path=os.path.join(rootdir,list1[i])
    if os.path.isfile(path):
        print(list1[i])
        img=im.open(path)
        img=img.convert('RGB')
        im_ar = conver2num(img)
        temp_path=list1[i][:-3]+'txt'
        np.savetxt('/home/white/下载/VOC2012/SegTXT/'+temp_path,im_ar,fmt='%d')



#################################################################
# im1=im.open('/home/white/下载/VOC2012/SegmentationClass/2007_000032.png')
# im1=im1.convert('RGB')
# # n_im1=np.array(im1)
# # im1_pix=n_im1[200][22]
# # pix=im1.getpixel((22,200))
# # print(pix)
# # im1.save('imm1.jpg','jpeg')
# im_ar=conver2num(im1)
# np.savetxt('/home/white/下载/VOC2012/SegTXT/')
# # print(im_ar)



