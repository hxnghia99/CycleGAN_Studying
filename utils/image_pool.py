#=======================================================================================#
#                                                                                       #
#   File name   : image_pool.py                                                         #
#   Author      : hxnghia99                                                             #
#   Created date: May 19th, 2023                                                        #
#   GitHub      : https://github.com/hxnghia99/CycleGAN_Styding                         #
#   Description : define a class to create buffer to save images                        #
#                                                                                       #
#=======================================================================================#

import random
import torch


class ImagePool():
    """This class implements an image buffer that stores previously generated images for updating discriminators (not only use the latest generators)"""

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int)     : the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool

        Parameters:
            images: the latest generated images from the generator

        Returns:
            - Images from the buffer.

        By 50%, the buffer will return input images.
        By 50%, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
