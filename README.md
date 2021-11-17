Name: Catherine Gai

SID: 3034712396

Email: catherine_gai@berkeley.edu

Link to project report website: https://inst.eecs.berkeley.edu/~cs194-26/fa21/upload/files/proj2/cs194-26-aay/catherine_gai_proj2/Project%2002.html

This folder contains four functional python files: "filter.py", "frequency.py", "align_image_code.py", "hybrid_image_starter.py", "stack.py", "blending.py". 

The folder also contains extra image files: "big_sur.jpeg","rainbow.jpeg" (for image sharpening); "angjoo.jpeg", "efros.jpeg", "wolf.jpeg", "lion.jpeg" (for image hybrid); "pizza1.jpeg", "pizza2.jpeg", "left_squirrel.jpeg", "right_squirrel.jpeg" (for image blending). 



**filter.py:**

This python file contains functions and other python commands adequate to generate all required images for Part 1. 

* derivative(*params*): calculates the x-derivative and y-derivative of a given image. 

To stimulate image-generation process, run ` python filter.py`. This command will result in seven images. Sequentially they are: x-derivative of the original image, y-derivative of the original image, blurred image, x-derivative of the blurred image by applying Gaussian first then the derivative, y-derivative of the blurred image by applying Gaussian first then the derivative, x-derivative of the blurred image by directly applying derivative of Gaussian filter, y-derivative of the blurred image by directly applying derivative of Gaussian filter. 



**frequency.py:**

This python file contains functions for image sharpening. 

* sharpen(*params*): sharpens the given image with respect to specified Gaussian filter and weight. 
* main(): sharpens "taj.jpeg", "big_sur.jpeg". First blurs then sharpens "rainbow.jpeg". 

To stimulate image-generation process, run `python frequency.py`. This command will generate five images. Sequentially they are: sharpened "taj.jpeg", sharpened "big_sur.jpeg", blurred "rainbow.jpeg", sharpened "rainbow.jpeg" after blurring. 



**align_image_code.py:**

The code is provided by the staff to facilitate image hybrid. 



**hybrid_image_starter.py:**

This python file contains functions that perform image hybrid. 

* hybrid_image_bw(*params*): performs image hybrid on two gray-scale images, Image 1 is turned to low-frequency version to be visible only from faraway. Image 2 si turned to high-frequency verison to be visible only from near. The function also shows frequency distribution of filtered low-frequency and high-frequency images. 
* Hybrid_image_color(*params*): performs image hybrid on two colored images, Image 1 is turned to low-frequency version to be visible only from faraway. Image 2 si turned to high-frequency verison to be visible only from near. 
* hybrid_cat(): performs image hybrid on "DerekPicture.jpeg" and "nutmeg.jpeg". The function outputs both gray-scale results and colored results. It also shows the frequency distribution of the hybrid image. The hybrid image shows a cat while viewing from near and Derek while viewing from faraway.
* hybrid_efros(): performs image hybrid on "efros.jpeg" and "kanazawa.jpeg". The function outputs both gray-scale results. It also shows the frequency distribution of the hybrid image. The hybrid image shows Kanazawa while viewing from near and Efros while viewing from faraway. 
* hybrid_wolf(): performs image hybrid on "lion.jpeg" and "wolf.jpeg". The function outputs both gray-scale results. It also shows frequency distribution of the two original images, those of the filtered images, and that of the hybrid image. The hybrid image shows a lion while viewing from near and a wolf while viewing from faraway. 

To stimulate image-generation process, run `python hybrid_image_starter.py`. 



**stack.py:**

This function contains functions that generates Gaussian stack and Laplacian stack. 

* gaussian_stack(*params*): generates Gaussian stack of an image, given the number of layers of that Gaussian stack and a factor that determines the size of Gaussian kernels at each level. 
* laplacian_stack(*params*): generates Laplacian stack of an image, given the number of layers of that Laplacian stack and a factor that determines the size of Gaussian kernels at each level. Calls gaussian_stack(*params*), and use its returned results to calculate corresponding Laplacian stack. 
* main(): Calculates the 6-layer Gaussian stack and 6-layer Laplacian stack of "oraple.jpeg". 

To stimulate image-generation process, run `python stack.py`. 



**blending.py:**

This function contains functions that blends two images together. 

* blending(*params*): given two images, a blending mask, laplacian_factor that is used to create Laplacian stack for those input images, gaussian_factor that is used to create Gaussian stack for the mask, output the blended result. 
* blending_apple_color(): blends colored "apple.jpeg" and "orange.jpeg" together. 
* blending_apple_gray(): blends gray_scale "apple.jpeg" and "orange.jpeg" together. 
* blending_pizza(): blends colored "pizza1.jpeg" and "pizza2.jpeg" together. 
* blending_squirrels(): blends colored "left_squirrel.jpeg" and "right_squirrel.jpeg" together. This function uses irregular mask that is a combination of a square and a circle. 

To stimulate image-generation process, run `python blending.py`. 

