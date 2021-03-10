import random
import numpy as np

def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches):
    if (N_patches%full_imgs.shape[0] != 0):
        print("Program exit: please enter a multiple of num_image train")
        print("N_patches: ", N_patches)
        print("Total images train: ", full_imgs.shape[0])
        exit()
    assert (len(full_imgs.shape)==3 and len(full_masks.shape)==3)  #3D arrays
    assert (full_imgs.shape[1] == full_masks.shape[1] and full_imgs.shape[2] == full_masks.shape[2])
    patches = np.empty((N_patches,patch_h,patch_w))
    patches_masks = np.empty((N_patches,patch_h,patch_w))
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
    print("patches per full image: " +str(patch_per_img))
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        k=0
        while k <patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))

            patch = full_imgs[i,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_mask = full_masks[i,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot]=patch
            patches_masks[iter_tot]=patch_mask
            iter_tot +=1   #total
            k+=1  #per full_img
    return patches, patches_masks