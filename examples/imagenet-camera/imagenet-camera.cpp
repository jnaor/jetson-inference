/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaNormalize.h"
#include "cudaFont.h"
#include "imageNet.h"


#define DEFAULT_CAMERA -1		// -1 for onboard CSI camera, or change to index of /dev/video V4L2 camera (>=0)	
#define DEFAULT_CAMERA_WIDTH 1280	// default camera width is 1280 pixels, change this if you want a different size
#define DEFAULT_CAMERA_HEIGHT 720	// default camera height is 720 pixels, change this is you want a different size

		
bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main( int argc, char** argv )
{
	printf("imagenet-camera\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	

	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA);
	
	if( !camera )
	{
		printf("\nimagenet-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\nimagenet-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create imageNet
	 */
	imageNet* net = imageNet::Create(argc, argv);
	
	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}


	/*
	 * create display window and overlay font
	 */
	glDisplay* display = glDisplay::Create();
	cudaFont*  font    = cudaFont::Create();
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\nimagenet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\nimagenet-camera:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	
	while( !signal_recieved )
	{
		float* imgRGBA = NULL;
		
		// get the latest frame
		if( !camera->CaptureRGBA(&imgRGBA, 1000) )
			printf("\nimagenet-camera:  failed to capture frame\n");

		// classify image
		const int img_class = net->Classify(imgRGBA, camera->GetWidth(), camera->GetHeight(), &confidence);
	
		if( img_class >= 0 )
		{
			printf("imagenet-camera:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, net->GetClassDesc(img_class));	

			if( font != NULL )
			{
				char str[256];
				sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
	
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 0, 0, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			}
		}	

		// update display
		if( display != NULL )
		{
			display->RenderOnce((float*)imgRGBA, camera->GetWidth(), camera->GetHeight());

			// update status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | %s | %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->GetNetworkName(), precisionTypeToStr(net->GetPrecision()), display->GetFPS());
			display->SetTitle(str);	

			// check if the user quit
			if( display->IsClosed() )
				signal_recieved = true;
		}
	}
	
	
	/*
	 * destroy resources
	 */
	printf("\nimagenet-camera:  shutting down...\n");
	
	SAFE_DELETE(camera);
	SAFE_DELETE(display);
	SAFE_DELETE(net);
	
	printf("imagenet-camera:  shutdown complete.\n");
	return 0;
}
