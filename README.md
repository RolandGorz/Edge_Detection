kernel.cu is the source code.

Program highlights:
Utilizied coalesced texture memory access to achieve high bandwith as well as limited the number of memory transfers to 2 
to allow for smooth real time video recording for both the original image and the morphological operation. Is also capable
of playing a video and performing the laplacian operation on it in real time. Also the example video I provide does utilize
slow motion effects, so it looks like the program is performing slowly but it is not.

How to use program:

Run the executable name laplacian.exe and press the button corresponding to the type of input that is wanted to be provided. A popup window will appear
that lets you choose the input file. Real time input does not have an input file, but a camera is needed to be connected as well as recognized as the
computer's default camera. Also please make sure to wait a few seconds when using real time because the camera's fps must first be calculated. This will
take at most 8 seconds. There will also be a popup window for where to save the output if saving is ticked in the checkbox.

GPU used:

NVIDIA GeForce 1070

Performance:

Average runtime for a 1920x1080 image = 1ms
Megapixels per second = (1920 * 1080 * (1000/ 1)) / 1000000 = 2073.6 MPixels/s
