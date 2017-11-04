#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "opencv2/opencv.hpp"
#include <Windows.h>
#include <time.h>

using namespace cv;
using namespace std;

//texture that stores the input image data

texture<uchar, 2, cudaReadModeElementType> src;

//bools that keep track if the user wants to save the outputs or an error occurred.

bool saveimage;
bool savevideo;
bool record;
bool failedOutput;
bool nocam;
bool fpsfail;

/*5x5 disk structuring element = {0, 1, 1, 1, 0},
								 {1, 1, 1, 1, 1},
								 {1, 1, 1, 1, 1},
								 {1, 1, 1, 1, 1},
								 {0, 1, 1, 1, 0}*/

__global__ void laplacian_texture(uchar *dev_lap, int rows, int cols) {
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y;

	uchar max = 0;
	uchar min = 255;

	if (tidx >= cols || tidy >= rows) {
		return;
	}

	//loop through the 25 elements that the structuring element covers and keep track of the maximum and minimum value;

	for (int i = tidy - 2; i <= tidy + 2; i++) {
		for (int j = tidx - 6; j <= tidx + 6; j += 3) {
			if (i < 0 || i >= rows || j < 0 || j >= cols || ((i == tidy - 2) && (j == tidx - 6)) || ((i == tidy - 2) && (j == tidx + 6)) || ((i == tidy + 2) && (j == tidx - 6)) || ((i == tidy + 2) && (j == tidx + 6))) {
				continue;
			}

			uchar current = tex2D(src, j, i);

			if (current > max) {
				max = current;
			}
			if (current < min) {
				min = current;
			}
		}
	}

	//perform the laplacian at the current pixel

	uchar original = tex2D(src, tidx, tidy);

	if ((max - original) < (original - min)) {
		dev_lap[tidy * cols + tidx] = 0;
	}
	else {
		dev_lap[tidy * cols + tidx] = (max - original) - (original - min);
	}
}

__global__ void laplacian_simple(uchar *dev_data, uchar *dev_lap, int total_pixels, int cols) {

	//threadID provides every thread that runs on the GPU an individual value. Every thread works on a pixel in each color channel.

	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	uchar max = 0;
	uchar min = 255;

	//Since the arrays are 1 dimensional the right_edge and left_edge make sure a pixel from a seperate row is not accessed.

	int right_edge = 0;
	int left_edge = 0;

	/*If the image has more pixels than total threads running on the GPU then the thread also works on the next pixel that
	would have been missed*/

	for (threadID; threadID < total_pixels; threadID += blockDim.x * gridDim.x) {
		for (int row = threadID - (2 * cols); row <= threadID + (2 * cols); row += cols) {
			right_edge = cols * ((row / cols) + 1);
			left_edge = cols * (row / cols);
			for (int pos = row - 6; pos <= row + 6; pos+=3) {
				if (row < 0 || row >= total_pixels || pos < left_edge || pos >= right_edge || ((row == threadID - (2 * cols)) && (pos == row - 6)) || ((row == threadID - (2 * cols)) && (pos == row + 6)) || ((row == threadID + (2 * cols)) && (pos == row - 6)) || ((row == threadID + (2 * cols)) && (pos == row + 6))) {
					continue;
				}
				//Calculates the maximum and minimum within the area that the structuring element covers at the current pixel.

				uchar current = dev_data[pos];

				if (current > max) {
					max = current;
				}
				if (current < min) {
					min = current;
				}
			}
		}
		/*Calculates the dilation - the erosion of the current pixel to get the laplacian.
		If the dilation is less than the erosion then the pixel is set to 0 to prevent an overflow*/

		uchar original = dev_data[threadID];

		if ((max - original) < (original - min)) {
			dev_lap[threadID] = 0;
		}
		else {
			dev_lap[threadID] = (max - original) - (original - min);
		}

		//Reset the maximum and minimum storage for the next pixel

		max = 0;
		min = 255;
	}
}

//Used when the user inputs a video file but does not want to save the output
void videoNoSave() {

	//code to make the open file dialog box appear

	OPENFILENAME ofn;       // common dialog box structure
	char szFile[520];       // buffer for file name
	HWND hwnd = NULL;       // owner window
	HANDLE hf;              // file handle

							// Initialize OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFile = szFile;
	// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
	// use the contents of szFile to initialize itself.
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "*.avi, *.divx\0*.avi;*.divx;\0\0*\0\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = ".";
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	// Display the Open dialog box. 

	if (GetOpenFileName(&ofn) == TRUE)
		hf = CreateFile(ofn.lpstrFile,
			GENERIC_READ,
			0,
			(LPSECURITY_ATTRIBUTES)NULL,
			OPEN_EXISTING,
			FILE_ATTRIBUTE_NORMAL,
			(HANDLE)NULL);

	if (strlen(ofn.lpstrFile) == 0) {
		return;
	}

	for (int i = 0, int j = 0; i <= strlen(ofn.lpstrFile); i++, j++) {
		if (ofn.lpstrFile[i] == '\\') {
			ofn.lpstrFile[i] = '/';
		}
	}

	//close the handle because the open file dialog box had a handle on the file which would not allow videocapture to read it

	CloseHandle(hf);

	VideoCapture cap(ofn.lpstrFile);

	double fps = cap.get(CV_CAP_PROP_FPS);

	Mat frame;
	Mat lap_frame;

	namedWindow("Laplacian", 1);
	namedWindow("Original", 1);
	HWND LAPhwnd = (HWND)cvGetWindowHandle("Laplacian");
	HWND ORIhwnd = (HWND)cvGetWindowHandle("Original");

	cudaArray *dev_data;

	uchar *dev_lap;

	dim3 gridsize, blocksize;

	/*Clamp address mode means that if a value that is outside of the texture array is accessed then instead of 
	seg faulting the nearest value along the endge is looked at. This is great for this program because the elements
	along that would already be part of the structuring element*/

	src.addressMode[0] = cudaAddressModeClamp;
	src.addressMode[1] = cudaAddressModeClamp;

	if (cap.isOpened() && IsWindowVisible(LAPhwnd)) {
		//malloc and calculate constants here to refrain from taking up time during the video loop.
		cap >> frame;
		lap_frame = frame.clone();

		blocksize.x = 32;
		blocksize.y = 32;
		gridsize.x = ceil(float(3 * frame.cols) / blocksize.x);
		gridsize.y = ceil(float(frame.rows) / blocksize.y);

		cudaMallocArray(&dev_data, &src.channelDesc, 3 * frame.cols, frame.rows);

		cudaMalloc((void**)&dev_lap, 3 * frame.rows * frame.cols * sizeof(uchar));
	}

	int size = 3 * frame.cols * frame.rows * sizeof(uchar);

	while (cap.isOpened() && IsWindowVisible(LAPhwnd)) {

		//Allow the user to close the original video, but keep playing the morphological operation.
		//If the user closes the laplacian video then close the rest of the windows as well.

		if (IsWindowVisible(ORIhwnd)) {
			imshow("Original", frame);
		}

		cudaMemcpyToArray(dev_data, 0, 0, frame.data, size, cudaMemcpyHostToDevice);

		cudaBindTextureToArray(src, dev_data, src.channelDesc);

		laplacian_texture << <gridsize, blocksize >> >(dev_lap, frame.rows, 3 * frame.cols);

		cudaMemcpy(lap_frame.data, dev_lap, size, cudaMemcpyDeviceToHost);

		imshow("Laplacian", lap_frame);
		waitKey(1000 / fps);
		cap >> frame; // get a new frame from camera
		
		//If we reached the end of the video then clean up.

		if (frame.empty()) {
			destroyAllWindows();
			break;
		}
	}

	//If the laplacian window was closed then close the original as well

	if (IsWindowVisible(ORIhwnd)) {
		destroyAllWindows();
	}

	cudaUnbindTexture(src);

	cudaFree(dev_data);
	cudaFree(dev_lap);
	cap.release();
}

//Very similar to video without save except for the fact that this one has saving involved

void videoSave() {
	OPENFILENAME ofn;       // common dialog box structure
	char szFile[520];       // buffer for file name
	HWND hwnd = NULL;       // owner window
	HANDLE hf;              // file handle

							// Initialize OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFile = szFile;
	// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
	// use the contents of szFile to initialize itself.
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "*.avi, *.divx\0*.avi;*.divx;\0\0*\0\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = ".";
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	// Display the Open dialog box. 

	if (GetOpenFileName(&ofn) == TRUE)
		hf = CreateFile(ofn.lpstrFile,
			GENERIC_READ,
			0,
			(LPSECURITY_ATTRIBUTES)NULL,
			OPEN_EXISTING,
			FILE_ATTRIBUTE_NORMAL,
			(HANDLE)NULL);

	if (strlen(ofn.lpstrFile) == 0) {
		return;
	}

	for (int i = 0, int j = 0; i <= strlen(ofn.lpstrFile); i++, j++) {
		if (ofn.lpstrFile[i] == '\\') {
			ofn.lpstrFile[i] = '/';
		}
	}

	CloseHandle(hf);

	VideoCapture cap(ofn.lpstrFile);
	Mat frame;
	Mat lap_frame;

	OPENFILENAME sfn;
	char syFile[520];
	ZeroMemory(&sfn, sizeof(sfn));
	sfn.lStructSize = sizeof(sfn);
	sfn.hwndOwner = NULL;
	sfn.lpstrFile = syFile;
	sfn.lpstrFile[0] = '\0';
	sfn.nMaxFile = sizeof(syFile);
	sfn.lpstrFilter = "*.avi\0*.avi;\0\0*\0";
	sfn.nFilterIndex = 1;
	sfn.lpstrFileTitle = NULL;
	sfn.nMaxFileTitle = 0;
	sfn.lpstrInitialDir = ".";
	sfn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_EXPLORER | OFN_ENABLEHOOK;
	sfn.lpstrDefExt = "avi";

	if (GetSaveFileName(&sfn) != true)
	{
		//do nothing
	}
	else {
		for (int i = 0, int j = 0; i <= strlen(sfn.lpstrFile); i++, j++) {
			if (sfn.lpstrFile[i] == '\\') {
				sfn.lpstrFile[i] = '/';
			}
		}
		remove(sfn.lpstrFile);

		double fps = cap.get(CV_CAP_PROP_FPS);

		VideoWriter output_cap(sfn.lpstrFile, -1, fps, Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT)));

		if (!output_cap.isOpened())
		{
			failedOutput = true;
			return;
		}

		namedWindow("Laplacian", 1);
		namedWindow("Original", 1);
		HWND LAPhwnd = (HWND)cvGetWindowHandle("Laplacian");
		HWND ORIhwnd = (HWND)cvGetWindowHandle("Original");

		cudaArray *dev_data;

		uchar *dev_lap;

		dim3 gridsize, blocksize;

		src.addressMode[0] = cudaAddressModeClamp;
		src.addressMode[1] = cudaAddressModeClamp;

		if (cap.isOpened() && IsWindowVisible(LAPhwnd)) {
			cap >> frame;
			lap_frame = frame.clone();

			blocksize.x = 32;
			blocksize.y = 32;
			gridsize.x = ceil(float(3 * frame.cols) / blocksize.x);
			gridsize.y = ceil(float(frame.rows) / blocksize.y);

			cudaMallocArray(&dev_data, &src.channelDesc, 3 * frame.cols, frame.rows);

			cudaMalloc((void**)&dev_lap, 3 * frame.rows * frame.cols * sizeof(uchar));
		}

		int size = 3 * frame.cols * frame.rows * sizeof(uchar);

		while (cap.isOpened() && IsWindowVisible(LAPhwnd)) {

			if (IsWindowVisible(ORIhwnd)) {
				imshow("Original", frame);
			}

			cudaMemcpyToArray(dev_data, 0, 0, frame.data, size, cudaMemcpyHostToDevice);

			cudaBindTextureToArray(src, dev_data, src.channelDesc);

			laplacian_texture << <gridsize, blocksize >> >(dev_lap, frame.rows, 3 * frame.cols);

			cudaMemcpy(lap_frame.data, dev_lap, size, cudaMemcpyDeviceToHost);

			imshow("Laplacian", lap_frame);
			output_cap.write(lap_frame);
			waitKey(1000 / fps);
			cap >> frame; // get a new frame from camera
			if (frame.empty()) {
				destroyAllWindows();
				break;
			}
		}

		if (IsWindowVisible(ORIhwnd)) {
			destroyAllWindows();
		}

		cudaUnbindTexture(src);

		cudaFree(dev_data);
		cudaFree(dev_lap);
	}
	cap.release();
}

void camera_feed_nosave() {
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) {
		nocam = true;
		cout << "Failed to find default camera" << endl;
		return;
	}
	//Let the user set camera resolution
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	//Find the camera fps here

	int num_frames = 120;
	time_t start, end;
	Mat frame;
	Mat lap_frame;

	cap >> frame;
	if (frame.empty()) {
		nocam = true;
		cout << "Failed to find default camera" << endl;
		return;
	}

	time(&start);
	for (int i = 0; i < num_frames; i++) {
		cap >> frame;
	}
	time(&end);
	double seconds = difftime(end, start);

	if (seconds == 0) {
		cout << "Error with camera. Failed to calculate fps" << endl;
		return;
	}

	double fps = num_frames / seconds;

	cout << fps << endl;

	namedWindow("Laplacian", 1);
	namedWindow("Original", 1);
	HWND LAPhwnd = (HWND)cvGetWindowHandle("Laplacian");
	HWND ORIhwnd = (HWND)cvGetWindowHandle("Original");

	cudaArray *dev_data;

	uchar *dev_lap;

	dim3 gridsize, blocksize;

	src.addressMode[0] = cudaAddressModeClamp;
	src.addressMode[1] = cudaAddressModeClamp;

	if (IsWindowVisible(LAPhwnd)) {
		cap >> frame;
		lap_frame = frame.clone();

		blocksize.x = 32;
		blocksize.y = 32;
		gridsize.x = ceil(float(3 * frame.cols) / blocksize.x);
		gridsize.y = ceil(float(frame.rows) / blocksize.y);

		cudaMallocArray(&dev_data, &src.channelDesc, 3 * frame.cols, frame.rows);

		cudaMalloc((void**)&dev_lap, 3 * frame.rows * frame.cols * sizeof(uchar));
	}

	int size = 3 * frame.cols * frame.rows * sizeof(uchar);

	while (IsWindowVisible(LAPhwnd)) {
		if (IsWindowVisible(ORIhwnd)) {
			imshow("Original", frame);
		}

		cudaMemcpyToArray(dev_data, 0, 0, frame.data, size, cudaMemcpyHostToDevice);

		cudaBindTextureToArray(src, dev_data, src.channelDesc);

		laplacian_texture << <gridsize, blocksize >> >(dev_lap, frame.rows, 3 * frame.cols);

		cudaMemcpy(lap_frame.data, dev_lap, size, cudaMemcpyDeviceToHost);

		imshow("Laplacian", lap_frame);
		waitKey(1000 / fps);
		cap >> frame; // get a new frame from camera
	}

	if (IsWindowVisible(ORIhwnd)) {
		destroyAllWindows();
	}

	cudaUnbindTexture(src);

	cudaFree(dev_data);
	cudaFree(dev_lap);
	cap.release();
}

void camera_feed_save() {
	VideoCapture cap(0);// open the default camera
	if (!cap.isOpened()) {
		nocam = true;
		cout << "Failed to find default camera" << endl;
		return;
	}
	//Let the user set camera resolution
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	
	OPENFILENAME sfn;
	char syFile[520];
	ZeroMemory(&sfn, sizeof(sfn));
	sfn.lStructSize = sizeof(sfn);
	sfn.hwndOwner = NULL;
	sfn.lpstrFile = syFile;
	sfn.lpstrFile[0] = '\0';
	sfn.nMaxFile = sizeof(syFile);
	sfn.lpstrFilter = "*.avi\0*.avi;\0\0*\0";
	sfn.nFilterIndex = 1;
	sfn.lpstrFileTitle = NULL;
	sfn.nMaxFileTitle = 0;
	sfn.lpstrInitialDir = ".";
	sfn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_EXPLORER | OFN_ENABLEHOOK;
	sfn.lpstrDefExt = "avi";

	//Find the camera fps here

	int num_frames = 120;
	time_t start, end;
	Mat frame;
	Mat lap_frame;

	cap >> frame;
	if (frame.empty()) {
		nocam = true;
		cout << "Failed to find default camera" << endl;
		return;
	}

	time(&start);
	for (int i = 0; i < num_frames; i++) {
		cap >> frame;
	}
	time(&end);
	double seconds = difftime(end, start);

	if (seconds == 0) {
		fpsfail = true;
		cout << "Error with camera. Failed to calculate fps" << endl;
		return;
	}

	double fps = num_frames / seconds;

	cout << fps << endl;

	if (GetSaveFileName(&sfn) != true)
	{
		//do nothing
	}
	else {
		for (int i = 0, int j = 0; i <= strlen(sfn.lpstrFile); i++, j++) {
			if (sfn.lpstrFile[i] == '\\') {
				sfn.lpstrFile[i] = '/';
			}
		}
		remove(sfn.lpstrFile);

		//cap.get(CV_CAP_PROP_FPS) is used for input videos not webcam.
		VideoWriter output_cap(sfn.lpstrFile, -1, fps, Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
		if (!output_cap.isOpened())
		{
			failedOutput = true;
			return;
		}

		namedWindow("Laplacian", 1);
		namedWindow("Original", 1);
		HWND LAPhwnd = (HWND)cvGetWindowHandle("Laplacian");
		HWND ORIhwnd = (HWND)cvGetWindowHandle("Original");

		cudaArray *dev_data;

		uchar *dev_lap;

		dim3 gridsize, blocksize;

		src.addressMode[0] = cudaAddressModeClamp;
		src.addressMode[1] = cudaAddressModeClamp;

		if (IsWindowVisible(LAPhwnd)) {
			cap >> frame;
			lap_frame = frame.clone();

			blocksize.x = 32;
			blocksize.y = 32;
			gridsize.x = ceil(float(3 * frame.cols) / blocksize.x);
			gridsize.y = ceil(float(frame.rows) / blocksize.y);

			cudaMallocArray(&dev_data, &src.channelDesc, 3 * frame.cols, frame.rows);

			cudaMalloc((void**)&dev_lap, 3 * frame.rows * frame.cols * sizeof(uchar));
		}

		int size = 3 * frame.cols * frame.rows * sizeof(uchar);

		while (IsWindowVisible(LAPhwnd)) {
			if (IsWindowVisible(ORIhwnd)) {
				imshow("Original", frame);
			}

			cudaMemcpyToArray(dev_data, 0, 0, frame.data, size, cudaMemcpyHostToDevice);

			cudaBindTextureToArray(src, dev_data, src.channelDesc);

			laplacian_texture << <gridsize, blocksize >> >(dev_lap, frame.rows, 3 * frame.cols);

			cudaMemcpy(lap_frame.data, dev_lap, size, cudaMemcpyDeviceToHost);

			imshow("Laplacian", lap_frame);
			output_cap.write(lap_frame);
			waitKey(1000 / fps);
			cap >> frame; // get a new frame from camera
		}

		if (IsWindowVisible(ORIhwnd)) {
			destroyAllWindows();
		}

		cudaUnbindTexture(src);

		cudaFree(dev_data);
		cudaFree(dev_lap);
	}
	cap.release();
}

void image_texture() {

	//Read the filename that the user wishes to enter and keep asking for user input until a file can be opened or the user quits

	OPENFILENAME ofn;       // common dialog box structure
	char szFile[520];       // buffer for file name
	HWND hwnd = NULL;       // owner window
	HANDLE hf;              // file handle

							// Initialize OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFile = szFile;
	// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
	// use the contents of szFile to initialize itself.
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "*.jpg, *.png, *.bmp, *.dib, *.jpeg, *.jpe, *.jfif, *.tif, *.tiff\0*.jpg;*.png;*.bmp;*.dib;*.jpeg;*.jpe;*.jfif;*.tif;*.tiff\0\0*\0\0\0\0\0\0\0\0\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = ".";
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	// Display the Open dialog box. 

	if (GetOpenFileName(&ofn) == TRUE)
		hf = CreateFile(ofn.lpstrFile,
			GENERIC_READ,
			0,
			(LPSECURITY_ATTRIBUTES)NULL,
			OPEN_EXISTING,
			FILE_ATTRIBUTE_NORMAL,
			(HANDLE)NULL);

	if (strlen(ofn.lpstrFile) == 0) {
		return;
	}

	for (int i = 0, int j = 0; i <= strlen(ofn.lpstrFile); i++, j++) {
		if (ofn.lpstrFile[i] == '\\') {
			ofn.lpstrFile[i] = '/';
		}
	}

	CloseHandle(hf);

	Mat image = imread(ofn.lpstrFile, 1);

	namedWindow("INPUT", CV_WINDOW_KEEPRATIO);
	imshow("INPUT", image);

	uchar *dev_lap;

	cudaMalloc((void**)&dev_lap, 3 * image.rows * image.cols * sizeof(uchar));

	cudaArray *dev_data;

	cudaMallocArray(&dev_data, &src.channelDesc, 3 * image.cols, image.rows);

	cudaMemcpyToArray(dev_data, 0, 0, image.data, 3 * image.cols * image.rows * sizeof(uchar), cudaMemcpyHostToDevice);

	cudaBindTextureToArray(src, dev_data, src.channelDesc);

	dim3 gridsize, blocksize;
	blocksize.x = 32;
	blocksize.y = 32;
	gridsize.x = ceil(float(3 * image.cols) / blocksize.x);
	gridsize.y = ceil(float(image.rows) / blocksize.y);

	laplacian_texture <<<gridsize, blocksize>>>(dev_lap, image.rows, 3 * image.cols);

	cudaMemcpy(image.data, dev_lap, 3 * image.rows * image.cols * sizeof(uchar), cudaMemcpyDeviceToHost);

	cudaUnbindTexture(src);

	cudaFree(dev_data);
	cudaFree(dev_lap);

	namedWindow("OUTPUT", CV_WINDOW_KEEPRATIO);
	imshow("OUTPUT", image);

	if (saveimage) {
		OPENFILENAME sfn;
		char syFile[520];
		ZeroMemory(&sfn, sizeof(sfn));
		sfn.lStructSize = sizeof(sfn);
		sfn.hwndOwner = NULL;
		sfn.lpstrFile = syFile;
		sfn.lpstrFile[0] = '\0';
		sfn.nMaxFile = sizeof(syFile);
		sfn.lpstrFilter = "*.jpg, *.png, *.bmp, *.dib, *.jpeg, *.jpe, *.jfif, *.tif, *.tiff\0*.jpg;*.png;*.bmp;*.dib;*.jpeg;*.jpe;*.jfif;*.tif;*.tiff\0\0*\0\0\0\0\0\0\0\0\0";
		sfn.nFilterIndex = 1;
		sfn.lpstrFileTitle = NULL;
		sfn.nMaxFileTitle = 0;
		sfn.lpstrInitialDir = ".";
		sfn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_EXPLORER | OFN_ENABLEHOOK;
		sfn.lpstrDefExt = "jpg";

		if (GetSaveFileName(&sfn) != true)
		{
			//do nothing
		}
		else {
			for (int i = 0, int j = 0; i <= strlen(sfn.lpstrFile); i++, j++) {
				if (sfn.lpstrFile[i] == '\\') {
					sfn.lpstrFile[i] = '/';
				}
			}
			imwrite(sfn.lpstrFile, image);
		}
	}
	waitKey(0);
	return;
}

void image_simple() {

	//Read the filename that the user wishes to enter and keep asking for user input until a file can be opened or the user quits

	OPENFILENAME ofn;       // common dialog box structure
	char szFile[520];       // buffer for file name
	HWND hwnd = NULL;       // owner window
	HANDLE hf;              // file handle

							// Initialize OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFile = szFile;
	// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
	// use the contents of szFile to initialize itself.
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "Supported Image Files\0*.jpg;*.png;*.bmp;*.dib;*.jpeg;*.jpe;*.jfif;*.tif;*.tiff\0ALL FILES\0*\0\0\0\0\0\0\0\0\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	// Display the Open dialog box. 

	if (GetOpenFileName(&ofn) == TRUE)
		hf = CreateFile(ofn.lpstrFile,
			GENERIC_READ,
			0,
			(LPSECURITY_ATTRIBUTES)NULL,
			OPEN_EXISTING,
			FILE_ATTRIBUTE_NORMAL,
			(HANDLE)NULL);

	if (strlen(ofn.lpstrFile) == 0) {
		return;
	}

	for (int i = 0, int j = 0; i <= strlen(ofn.lpstrFile); i++, j++) {
		if (ofn.lpstrFile[i] == '\\') {
			ofn.lpstrFile[i] = '/';
		}
	}

	CloseHandle(hf);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	Mat image = imread(ofn.lpstrFile, 1);

	namedWindow("INPUT", CV_WINDOW_KEEPRATIO);
	imshow("INPUT", image);

	/*Split the image into the 3 image channels Blue, Green, and Red respectively. This makes 3 arrays that contain the intensity values
	of each image channel. These arrays are then allocated and passed to the GPU. LapB contains the intensity values after the algorithm
	for computing the laplacian completes.*/

	uchar *dev_data;
	uchar *dev_lap;

	cudaMalloc((void**)&dev_data, image.rows * image.cols * 3 * sizeof(uchar));
	cudaMalloc((void**)&dev_lap, image.rows * image.cols * 3 * sizeof(uchar));

	cudaMemcpy(dev_data, image.data, image.rows * image.cols * 3 * sizeof(uchar), cudaMemcpyHostToDevice);

	/*Call the CUDA kernel with a grid size of 512 that each one will be run on a Streaming Multiprocessor with each
	Multiprocessor running 1024 threads*/

	laplacian_simple << <512, 1024 >> >(dev_data, dev_lap, 3 * image.rows * image.cols, 3 * image.cols);
	//Transfer the lapB array from the device to the host

	cudaMemcpy(image.data, dev_lap, image.rows * image.cols * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << milliseconds << endl;

	cudaFree(dev_data);
	cudaFree(dev_lap);

	//Merge the 3 seperate channel arrays into one array for the output image. Display the input and output image.

	namedWindow("OUTPUT", CV_WINDOW_KEEPRATIO);
	imshow("OUTPUT", image);

	OPENFILENAME sfn;
	char syFile[520];
	ZeroMemory(&sfn, sizeof(sfn));
	sfn.lStructSize = sizeof(sfn);
	sfn.hwndOwner = NULL;
	sfn.lpstrFile = syFile;
	sfn.lpstrFile[0] = '\0';
	sfn.nMaxFile = sizeof(syFile);
	sfn.lpstrFilter = "*.jpg, *.png, *.bmp, *.dib, *.jpeg, *.jpe, *.jfif, *.tif, *.tiff\0*.jpg;*.png;*.bmp;*.dib;*.jpeg;*.jpe;*.jfif;*.tif;*.tiff\0\0*\0\0\0\0\0\0\0\0\0";
	sfn.nFilterIndex = 1;
	sfn.lpstrFileTitle = NULL;
	sfn.nMaxFileTitle = 0;
	sfn.lpstrInitialDir = ".";
	sfn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_EXPLORER | OFN_ENABLEHOOK;
	sfn.lpstrDefExt = "jpg";

	if (GetSaveFileName(&sfn) != true)
	{
		cout << "Saving file canceled, closing program in 10 secconds." << endl;
	}
	else {
		for (int i = 0, int j = 0; i <= strlen(sfn.lpstrFile); i++, j++) {
			if (sfn.lpstrFile[i] == '\\') {
				sfn.lpstrFile[i] = '/';
			}
		}
		imwrite(sfn.lpstrFile, image);
	}
	waitKey(0);
	return;
}

/* This is where all the input to the window goes to */
HWND button1;
HWND check1;
HWND button2;
HWND check2;
HWND button3;
HWND check3;
char input[520];
HWND edit;
HWND text;
LRESULT CALLBACK WndProc(HWND hwnd, UINT Message, WPARAM wParam, LPARAM lParam) {
	switch (Message) {
		
		case WM_CREATE: {

			text = CreateWindow(TEXT("STATIC"), TEXT("Laplacian Morphological Operation"), 
				WS_VISIBLE | WS_CHILD, 
				190, 10, 
				400, 25, 
				hwnd, (HMENU) NULL, NULL, NULL);

			HFONT text_change = CreateFont(20, 0, 0, 0, FW_DONTCARE, FALSE, FALSE, FALSE, ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, "Times New Roman");
			SendMessage(text, WM_SETFONT, WPARAM(text_change), TRUE);

			//GetWindowText(edit, input, 260);

			text = CreateWindow(TEXT("STATIC"), TEXT("Live input needs a few seconds to calculate the camera's FPS. Please wait after selecting."),
				WS_VISIBLE | WS_CHILD,
				190, 255,
				400, 50,
				hwnd, (HMENU)NULL, NULL, NULL);

			button1 = CreateWindow(TEXT("BUTTON"), TEXT("Image Input"),
				WS_VISIBLE | WS_CHILD,
				10, 50,
				150, 50,
				hwnd, (HMENU) 1, NULL, NULL);

			text_change = CreateFont(30, 10, 0, 0, FW_DONTCARE, FALSE, FALSE, FALSE, ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, "Arial");
			SendMessage(button1, WM_SETFONT, WPARAM(text_change), TRUE);

			check1 = CreateWindow(TEXT("button"), TEXT("Save Image"),
				WS_VISIBLE | WS_CHILD | BS_CHECKBOX,
				20, 100, 
				100, 20,
				hwnd, (HMENU)2, ((LPCREATESTRUCT)lParam)->hInstance, NULL);
			CheckDlgButton(hwnd, 2, BST_CHECKED);

			button2 = CreateWindow(TEXT("BUTTON"), TEXT("Video Input"),
				WS_VISIBLE | WS_CHILD,
				10, 150,
				150, 50,
				hwnd, (HMENU)3, NULL, NULL);

			check2 = CreateWindow(TEXT("button"), TEXT("Save Video"),
				WS_VISIBLE | WS_CHILD | BS_CHECKBOX,
				20, 200,
				95, 20,
				hwnd, (HMENU)4, ((LPCREATESTRUCT)lParam)->hInstance, NULL);
			CheckDlgButton(hwnd, 4, BST_UNCHECKED);

			text_change = CreateFont(30, 10, 0, 0, FW_DONTCARE, FALSE, FALSE, FALSE, ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, "Arial");
			SendMessage(button2, WM_SETFONT, WPARAM(text_change), TRUE);

			button3 = CreateWindow(TEXT("BUTTON"), TEXT("Live Input"),
				WS_VISIBLE | WS_CHILD,
				10, 250,
				150, 50,
				hwnd, (HMENU)5, NULL, NULL);

			check3 = CreateWindow(TEXT("button"), TEXT("Record Video"),
				WS_VISIBLE | WS_CHILD | BS_CHECKBOX,
				20, 300,
				105, 20,
				hwnd, (HMENU)6, ((LPCREATESTRUCT)lParam)->hInstance, NULL);
			CheckDlgButton(hwnd, 6, BST_UNCHECKED);

			text_change = CreateFont(30, 10, 0, 0, FW_DONTCARE, FALSE, FALSE, FALSE, ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, "Arial");
			SendMessage(button3, WM_SETFONT, WPARAM(text_change), TRUE);

			break;
		}
		case WM_COMMAND: {

			if (LOWORD(wParam) == 1) {
				/*GetWindowText(edit, input, 260);
				MessageBox(hwnd, input, "title for popup", MB_ICONINFORMATION);*/
				if (IsDlgButtonChecked(hwnd, 2)) {
					saveimage = true;
				}
				else {
					saveimage = false;
				}
				EnableWindow(button1, false);
				EnableWindow(check1, false);
				EnableWindow(button2, false);
				EnableWindow(check2, false);
				EnableWindow(button3, false);
				EnableWindow(check3, false);
				image_texture();
				EnableWindow(button1, true);
				EnableWindow(check1, true);
				EnableWindow(button2, true);
				EnableWindow(check2, true);
				EnableWindow(button3, true);
				EnableWindow(check3, true);
			}

			if (LOWORD(wParam) == 2) {
				BOOL checked = IsDlgButtonChecked(hwnd, 2);
				if (checked) {
					CheckDlgButton(hwnd, 2, BST_UNCHECKED);
				}
				else {
					CheckDlgButton(hwnd, 2, BST_CHECKED);
				}
			}

			if (LOWORD(wParam) == 3) {
				/*GetWindowText(edit, input, 260);
				MessageBox(hwnd, input, "title for popup", MB_ICONINFORMATION);*/
				if (IsDlgButtonChecked(hwnd, 4)) {
					savevideo = true;
				}
				else {
					savevideo = false;
				}
				EnableWindow(button1, false);
				EnableWindow(check1, false);
				EnableWindow(button2, false);
				EnableWindow(check2, false);
				EnableWindow(button3, false);
				EnableWindow(check3, false);
				if (savevideo) {
					videoSave();
				}
				else {
					videoNoSave();
				}
				if (failedOutput) {
					MessageBox(hwnd, "Output video could not be opened use different compression option", "Error", MB_ICONINFORMATION);
				}
				failedOutput = false;
				EnableWindow(button1, true);
				EnableWindow(check1, true);
				EnableWindow(button2, true);
				EnableWindow(check2, true);
				EnableWindow(button3, true);
				EnableWindow(check3, true);
			}

			if (LOWORD(wParam) == 4) {
				BOOL checked = IsDlgButtonChecked(hwnd, 4);
				if (checked) {
					CheckDlgButton(hwnd, 4, BST_UNCHECKED);
				}
				else {
					CheckDlgButton(hwnd, 4, BST_CHECKED);
				}
			}

			if (LOWORD(wParam) == 5) {
				/*GetWindowText(edit, input, 260);
				MessageBox(hwnd, input, "title for popup", MB_ICONINFORMATION);*/
				if (IsDlgButtonChecked(hwnd, 6)) {
					record = true;
				}
				else {
					record = false;
				}
				EnableWindow(button1, false);
				EnableWindow(check1, false);
				EnableWindow(button2, false);
				EnableWindow(check2, false);
				EnableWindow(button3, false);
				EnableWindow(check3, false);
				if (record) {
					camera_feed_save();
				}
				else {
					camera_feed_nosave();
				}
				if (failedOutput) {
					MessageBox(hwnd, "Output video could not be opened use different compression option", "Error", MB_ICONINFORMATION);
				}
				if (nocam) {
					MessageBox(hwnd, "Failed to find default camera", "Error", MB_ICONINFORMATION);
				}
				if (fpsfail) {
					MessageBox(hwnd, "Error with camera. Failed to calculate fps", "Error", MB_ICONINFORMATION);
				}
				failedOutput = false;
				nocam = false;
				fpsfail = false;
				EnableWindow(button1, true);
				EnableWindow(check1, true);
				EnableWindow(button2, true);
				EnableWindow(check2, true);
				EnableWindow(button3, true);
				EnableWindow(check3, true);
			}

			if (LOWORD(wParam) == 6) {
				BOOL checked = IsDlgButtonChecked(hwnd, 6);
				if (checked) {
					CheckDlgButton(hwnd, 6, BST_UNCHECKED);
				}
				else {
					CheckDlgButton(hwnd, 6, BST_CHECKED);
				}
			}

			break;
		}

		/* Upon destruction, tell the main thread to stop */
		case WM_DESTROY: {
			PostQuitMessage(0);
			break;
		}

					 /* All other messages (a lot of them) are processed using default procedures */
		default:
			return DefWindowProc(hwnd, Message, wParam, lParam);
	}
	return 0;
}

/* The 'main' function of Win32 GUI programs: this is where execution starts */
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	WNDCLASSEX wc; /* A properties struct of our window */
	HWND hwnd; /* A 'HANDLE', hence the H, or a pointer to our window */
	MSG msg; /* A temporary location for all messages */

			 /* zero out the struct and set the stuff we want to modify */
	memset(&wc, 0, sizeof(wc));
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.lpfnWndProc = WndProc; /* This is where we will send messages to */
	wc.hInstance = hInstance;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);

	/* White, COLOR_WINDOW is just a #define for a system color, try Ctrl+Clicking it */
	//wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wc.hbrBackground = GetSysColorBrush(COLOR_3DFACE);
	wc.lpszClassName = "WindowClass";

	wc.hIcon = (HICON)LoadImage( // returns a HANDLE so we have to cast to HICON
		NULL,             // hInstance must be NULL when loading from a file
		"lapIcon.ico",   // the icon file name
		IMAGE_ICON,       // specifies that the file is an icon
		0,                // width of the image (we'll specify default later on)
		0,                // height of the image
		LR_LOADFROMFILE |  // we want to load a file (as opposed to a resource)
		LR_DEFAULTSIZE |   // default metrics based on the type (IMAGE_ICON, 32x32)
		LR_SHARED         // let the system release the handle when it's no longer used
		);
	wc.hIconSm = LoadIcon(NULL, NULL); /* use the name "A" to use the project icon */

	if (!RegisterClassEx(&wc)) {
		MessageBox(NULL, "Window Registration Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
		return 0;
	}

	hwnd = CreateWindowEx(WS_EX_CLIENTEDGE, "WindowClass", "CUDA Laplacian", WS_VISIBLE | WS_SYSMENU,
		CW_USEDEFAULT, /* x */
		CW_USEDEFAULT, /* y */
		640, /* width */
		480, /* height */
		NULL, NULL, hInstance, NULL);

	if (hwnd == NULL) {
		MessageBox(NULL, "Window Creation Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
		return 0;
	}

	/*
	This is the heart of our program where all input is processed and
	sent to WndProc. Note that GetMessage blocks code flow until it receives something, so
	this loop will not produce unreasonably high CPU usage
	*/
	while (GetMessage(&msg, NULL, 0, 0) > 0) { /* If no error is received... */
		TranslateMessage(&msg); /* Translate key codes to chars if present */
		DispatchMessage(&msg); /* Send it to WndProc */
	}
	return msg.wParam;
}