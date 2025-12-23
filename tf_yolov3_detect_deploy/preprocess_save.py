import os
import cv2
import numpy as np

TEST_SET = ['val']
INPUT_IMAGE_PATH = 'test_sample/'
OUTPUT_DATA_PATH = 'preprocessed_data/' 

if not os.path.exists(OUTPUT_DATA_PATH):
    os.makedirs(OUTPUT_DATA_PATH)
for folder in TEST_SET:
    if not os.path.exists(os.path.join(OUTPUT_DATA_PATH, folder)):
        os.makedirs(os.path.join(OUTPUT_DATA_PATH, folder))

def letterbox_image(image, size):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h,w,3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image

def pre_process(image, model_image_size):
    image = image[...,::-1] # BGR to RGB
    image_h, image_w, _ = image.shape
    if model_image_size != (None, None):
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    return image_data

if __name__ == "__main__":
    print("start...")
    
    MODEL_SIZE = (608, 608) 

    for folder in TEST_SET:
        folder_path = os.path.join(INPUT_IMAGE_PATH, folder)
        files = os.listdir(folder_path)
        
        for file_name in files:
            if not file_name.endswith('.jpg'): continue
            
            img_path = os.path.join(folder_path, file_name)
            image = cv2.imread(img_path)
            
            if image is None: continue
            
            image_data = pre_process(image, MODEL_SIZE)
            
            save_name = file_name.replace('.jpg', '.npy')
            save_path = os.path.join(OUTPUT_DATA_PATH, folder, save_name)
            
            np.save(save_path, image_data)
            print(f"Saved: {save_path}")

    print("done!")