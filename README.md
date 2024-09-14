This was my project from software engineering course.

I needed to create application that will do bone segmentation from 2D dicom images.

For that I had dataset of 350 sequential bone images that I'm unable to share due to patient confidentiality.
There were multiple paths to choose for solving this problem but I went on the way with neural network.

First part of learning neural network to predict bone is to generate masks for training pictures.
Making masks manually one per one was out of question, so I started exploring dicom images and I found out that
pixel values of MRI and CT scans are usually in range from -1000 to +3000, and that bone pixel values are after +2500.
So I ended with creating automatic mask generator.
It probably traded precision but for this project was more than enough.

Original images were 512x512 and could be cropped to 256x256 resoulution because bone is in the center. So I got rid of the excess pixels.
Cropping improved performance a lot because there were 4x less pixels to compute.

For neural network I chosed U-net architecture because it's common for this type of tasks.

I tried different optimizers, batch sizes, epochs, number of filters and metrics.
The ones left in code gave me the best results.
