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

#include <iostream>
#include <stdexcept>
#include <sstream>

// TODO: delete this
// #include <fstream>
// std::ofstream error_log("/home/wizzy/Desktop/drek_main.txt");


#include "cudaMappedMemory.h"
#include "imageIO.h"
#include "segNet.h"

#include <signal.h>
#include <opencv2/opencv.hpp>


class Segmenter
{
public:
	Segmenter(const uint32_t width, const uint32_t height) : m_width(width), m_height(height)
	{ 

		// arguments for network
		char* argv[] = {"--network=fcn-resnet18-sun"};
		
		/*
		* create segmentation network
		*/
		m_net = segNet::Create(2, argv);

		// for compund error messages
		std::ostringstream error_message;

		if( !m_net )
		{
			throw std::runtime_error("segnet-camera:   failed to initialize imageNet");
		}

		// set alpha blending value for classes that don't explicitly already have an alpha	
		m_net->SetOverlayAlpha(120.0f);

		if( !cudaAllocMapped((void**)&m_imgOverlay, width * height * sizeof(float) * 4) )
		{
			error_message << "segnet-camera:  failed to allocate CUDA memory for overlay image " <<  
					width << ", " << height;
			throw std::runtime_error(error_message.str());
		}

		if( !cudaAllocMapped((void**)&m_imgMask, width/2 * height/2 * sizeof(float) * 4) )
		{
			throw std::runtime_error("segnet-camera:  failed to allocate CUDA memory for mask image");
		}
	}

	// destructor
	~Segmenter()
	{
		// wait for the GPU to finish		
		CUDA(cudaDeviceSynchronize());

		// print out timing info
		m_net->PrintProfilerTimes();

		// release network resources
		SAFE_DELETE(m_net);
	}

	// processing
	bool process(cv::Mat rgb, cv::Mat depth)
	{
		// to hold pointer to image in CPU/GPU shared memory
		float* imgCPU = 0;

		// to hold pointer in CUDA memory
		float* imgCUDA  = 0;

		// subtract mean from image (or not)
		float4 mean_pixel = make_float4(0, 0, 0, 0);

		// upload to gpu
		uploadImageToGPU(rgb.data, m_width, m_height, 3, true, (void**) &imgCPU, 
						(void**) &imgCUDA, (void*) &mean_pixel);

		// process the segmentation network
		if( !m_net->Process(imgCUDA, m_width, m_height) )
		{
			std::cerr << "segnet-console:  failed to process segmentation" << std::endl;
		}
		
		// generate overlay
		if( !m_net->Overlay(m_imgOverlay, m_width, m_height, segNet::FILTER_POINT) )
		{
			std::cerr << "segnet-console:  failed to process segmentation overlay" << std::endl;
		}

		// generate mask
		if( !m_net->Mask(m_imgMask, m_width/2, m_height/2, segNet::FILTER_POINT) )
		{
			std::cerr << "segnet-console:  failed to process segmentation mask" << std::endl;
		}	
	
	}

private:
	// width of images to be processed
	const uint32_t m_width;

	// height of images to be processed
	const uint32_t m_height;

	// handle to segmentation network
	segNet* m_net;

	// segmentation overlay output buffer
	float* m_imgOverlay = NULL;
	
	// segmentation overlay output mask buffer 
	float* m_imgMask    = NULL;
};



int main()
{

	cv::Mat rgb = cv::imread("/home/wizzy/Desktop/rgb.png");
	cv::Mat depth = cv::imread("/home/wizzy/Desktop/depth.png");
	
	Segmenter segmenter(rgb.cols, rgb.rows);
	
	return segmenter.process(rgb, depth);
}

