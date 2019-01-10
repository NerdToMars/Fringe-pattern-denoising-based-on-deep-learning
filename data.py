from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.preprocessing import image as KI

def data_generator(dataset,shuffle=True,target_size = (8,256,256),batch_size=1):
    '''
    dataset: Array of all the data, each item is the string of the absolute folder path
    target_size: the shape of sample images(number, width, height), 8 means there are 8 images captured in 1 experiment.
    '''
    image_stack_ids = np.copy(dataset)
    stack_index = -1
    b = 0  # batch item index
    error_count = 0
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            stack_index = (stack_index + 1) % len(image_stack_ids)
            if shuffle and stack_index == 0:
                np.random.shuffle(image_stack_ids)
                
            stack_id = image_stack_ids[stack_index]
            all_in_focus_image = KI.load_img(stack_id+'/AllInFocus.bmp',target_size=target_size[1:3],grayscale=True)
            all_in_focus_image = KI.img_to_array(all_in_focus_image)
            if b == 0:
                batch_image_stack = np.zeros(
                (batch_size,target_size[0],)+all_in_focus_image.shape, dtype=all_in_focus_image.dtype)
                batch_depth = np.zeros(
                (batch_size,)+all_in_focus_image.shape,dtype=all_in_focus_image.dtype)
            
            #load image stack and depth add to batch
            for i in range(target_size[0]):
                np_img = KI.img_to_array(
                    KI.load_img(stack_id+str(i+1)+'.bmp',target_size=target_size[1:3],grayscale=True))
#                 print(batch_image_stack.shape)
#                 print(b,i)
                batch_image_stack[b][i] = np_img
                
            np_depth = KI.img_to_array(
                    KI.load_img(stack_id+'/groud.bmp',target_size=target_size[1:3],grayscale=True))
            batch_depth[b] = np_depth
            b += 1
            if b >= batch_size:
                b = 0
                yield batch_image_stack, batch_depth  
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            print(dataset[stack_index])
            error_count += 1
            if error_count > 5:
                raise
                
def data_feed(data_dir,order=True,target_size=(8,256,256)):
    '''
    load a single data
    '''
    all_list = os.listdir(data_dir)
    
    image_idx = []
    for file_name in all_list:
        start_idx = file_name.find('-')
        end_idx = file_name.find('.')
        file_idx = int(file_name[start_idx:end_idx])
        image_idx.append(file_idx)
    
    if order:
        image_idx.sort()

    img_ki = KI.load_img(data_dir+'1'+str(image_idx[0])+'.bmp',target_size=target_size[1:3],grayscale=True)
    img_ki_array = KI.img_to_array(img_ki)
    image_stack = np.zeros((target_size[0],)+img_ki_array.shape,dtype=np.uint8)
    for idx,item in enumerate(image_idx):
        image_stack[idx] = KI.img_to_array(
            KI.load_img(data_dir+'1'+str(item)+'.bmp',target_size=target_size[1:3],grayscale=True))
    
    return image_stack


       


