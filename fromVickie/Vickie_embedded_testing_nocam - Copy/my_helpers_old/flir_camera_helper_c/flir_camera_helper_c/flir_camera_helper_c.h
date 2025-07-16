#pragma once
#define DLLEXPORT extern "C" __declspec(dllexport)

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include "windows.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

SystemPtr system_gl;
CameraList camList;
CameraPtr pCam;
ImagePtr pResultImage;

double timestamp_seconds_start;

DLLEXPORT void reset_timestamp(void);

DLLEXPORT bool init(void);
DLLEXPORT bool cleanup(void);
DLLEXPORT bool open_cam(unsigned int serial_number, unsigned int videomode, unsigned int xoffset, unsigned int yoffset, unsigned int width, unsigned int height, double gain, double shutter);
DLLEXPORT bool close_cam(void);
DLLEXPORT double get_image(unsigned char *image_data, size_t size);
DLLEXPORT double get_gain();
DLLEXPORT bool set_gain(double value);
DLLEXPORT double get_brightness();
DLLEXPORT bool set_brightness(double value);
DLLEXPORT double get_shutter();
DLLEXPORT bool set_shutter(double value);