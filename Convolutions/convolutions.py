import numpy as np


def basic_convolution(image, kernel, verbose=False):
    
    kernel = np.flipud(np.fliplr(kernel))
    # Get the kernel dimensions and a unit for finding relative position
    # E.g the top right of a 5x5 kernel will equate to (2,2) so the travel size is 2
    k_h = len(kernel)
    k_w = len(kernel[0])
    i_h = len(image)
    i_w = len(image[0])
    
    travel_h = k_h // 2
    travel_w = k_w // 2
    
    image_out = np.ones_like(image)

    for i in range(travel_h, i_h - travel_h):
        for j in range(travel_w, i_w - travel_w):
            
            accumulator = 0
            
            for x in range(k_h):
                for y in range(k_w):
                    
                    try:
                        accumulator += image[i - travel_h + x][j - travel_w + y] * kernel[x][y]
                    except:
                        continue
            
            image_out[i][j] = accumulator

    return image_out


def extended_convolution(image, kernel, verbose=False):
    
    kernel = np.flipud(np.fliplr(kernel))
    # E.g the top right of a 5x5 kernel will equate to (2,2) so the travel size is 2
    image_height = len(image)
    image_width = len(image[0])
    kernel_size = len(kernel)
    travel = kernel_size // 2
    image_out = np.ones_like(image)

    for i in range(len(image)):
        for j in range(len(image[i])):
            accumulator = 0
            
            
            for x in range(-travel, travel + 1):
                for y in range(-travel,travel + 1):
                    
                    try:
                        
                        # Bottom right corner
                        if (i+x) >= image_height and (j+y) >= image_width:
                            accumulator += image[image_height-1][image_width-1] * kernel[x + travel][y + travel]

                        # Bottom left and top right corner
                        elif (((i+x) >= image_height and (j+y) < 0) or 
                              ((i+x) == image_height-1 and (j+y) < 0) or
                              ((i+x) >= image_height and (j+y) == 0)):
                            accumulator += image[image_height-1][0] * kernel[x + travel][y + travel]
                        elif (((i+x) < 0 and (j+y) >= image_width) or
                              ((i+x) < 0 and (j+y) == image_width-1) or 
                              ((i+x) == 0 and (j+y) >= image_width)):
                            accumulator += image[0][image_width-1] * kernel[x + travel][y + travel]

                        # Rigth / bottom edges
                        elif (i+x) >= image_height and (j+y) < image_width:
                            accumulator += image[image_height-1][j+y] * kernel[x + travel][y + travel]
                        elif (i+x) < image_height and (j+y) >= image_width:
                            accumulator += image[i+x][image_width-1] * kernel[x + travel][y + travel]

                        # Top left corner and top / left edges
                        elif (i+x) < 0 and (j+y) < 0:
                            accumulator += image[0][0] * kernel[x + travel][y + travel]
                        elif (i+x) < 0 and (j+y) >= 0:
                            accumulator += image[0][j+y] * kernel[x + travel][y + travel]
                        elif (i+x) >= 0 and (j+y) < 0:
                            accumulator += image[i+x][0] * kernel[x + travel][y + travel]
                            
                        # Everything else
                        else:
                            accumulator += image[i+x][j+y] * kernel[x + travel][y + travel]
                            
                    except:
                        continue
                        
            image_out[i][j] = accumulator
            
    return image_out


def fft_convolution(image,kernel):

    travel = len(kernel) // 2

    image_height = len(image)
    image_width = len(image[0])
    
    size = (image_height - len(kernel), image_width - len(kernel[0]))  # total amount of padding
    f_kernel = np.pad(kernel, (((size[0]+1)//2, size[0]//2), ((size[1]+1)//2, size[1]//2)))
    f_kernel = np.fft.ifftshift(f_kernel)

    product = np.real(np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(f_kernel)))

    kernel = np.flipud(np.fliplr(kernel))

    for i in range(image_height):
        for j in range(image_width):
            if ((i < travel + 1 or i > image_height - (travel + 1)) or
                (j < travel + 1 or j > image_width - (travel + 1))):

                accumulator = 0

                for x in range(-travel, travel + 1):
                    for y in range(-travel,travel + 1):
                    
                        try:
                        
                        # Bottom right corner
                            if (i+x) >= image_height and (j+y) >= image_width:
                                accumulator += image[image_height-1][image_width-1] * kernel[x + travel][y + travel]

                        # Bottom left and top right corner
                            elif (((i+x) >= image_height and (j+y) < 0) or 
                                  ((i+x) == image_height-1 and (j+y) < 0) or
                                ((i+x) >= image_height and (j+y) == 0)):
                                accumulator += image[image_height-1][0] * kernel[x + travel][y + travel]
                            elif (((i+x) < 0 and (j+y) >= image_width) or
                                  ((i+x) < 0 and (j+y) == image_width-1) or 
                                ((i+x) == 0 and (j+y) >= image_width)):
                                accumulator += image[0][image_width-1] * kernel[x + travel][y + travel]

                        # Ritgh / bottom edges
                            elif (i+x) >= image_height and (j+y) < image_width:
                                accumulator += image[image_height-1][j+y] * kernel[x + travel][y + travel]
                            elif (i+x) < image_height and (j+y) >= image_width:
                                accumulator += image[i+x][image_width-1] * kernel[x + travel][y + travel]

                        # Top left corner and top / left edges
                            elif (i+x) < 0 and (j+y) < 0:
                                accumulator += image[0][0] * kernel[x + travel][y + travel]
                            elif (i+x) < 0 and (j+y) >= 0:
                                accumulator += image[0][j+y] * kernel[x + travel][y + travel]
                            elif (i+x) >= 0 and (j+y) < 0:
                                accumulator += image[i+x][0] * kernel[x + travel][y + travel]
                            
                        # Everything else
                            else:
                                accumulator += image[i+x][j+y] * kernel[x + travel][y + travel]
                            
                        except:
                            continue
                        
                product[i][j] = accumulator

    return product
