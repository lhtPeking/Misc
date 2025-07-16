#include <windows.h>
#include "flir_camera_helper_c.h"
#include <iostream>
#include <math.h>

DLLEXPORT bool init(void) {
	
	// Find the camera and connect
	system_gl = System::GetInstance();
	
	// Retrieve list of cameras from the system
	camList = system_gl->GetCameras();

	return true;
}

DLLEXPORT bool cleanup(void) {
	
	camList.Clear();

	// Release system
	pCam = NULL;

	system_gl->ReleaseInstance();

	return true;
}

DLLEXPORT bool open_cam(unsigned int serial_number, unsigned int videomode, unsigned int xoffset, unsigned int yoffset, unsigned int width, unsigned int height, double gain, double shutter) {

	cout << "opening camera; serial number: " << serial_number << ", videomode: " << videomode << ", xoffset: " << xoffset << ", yoffset: "  << yoffset << ", width: " << width << ", height: " << height << ", gain: " << gain << ", shutter: " << shutter << endl;

	try {
		pCam = camList.GetBySerial(to_string(serial_number));
	} catch (Spinnaker::Exception &e) {
		cout << "Error: " << e.what() << endl;
		return false;
	}
	
	try {
		pCam->Init();
	} catch (Spinnaker::Exception &e) {
		cout << "Error: " << e.what() << endl;
		return false;
	}

	try {

		INodeMap & nodeMap = pCam->GetNodeMap();

		CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
		CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
		int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();
		ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

		CEnumerationPtr ptrVideoMode = nodeMap.GetNode("VideoMode");
		ptrVideoMode->SetIntValue(videomode);

		// setup the buffer properties
		INodeMap & nodeMapTLStream = pCam->GetTLStreamNodeMap();
		CEnumerationPtr StreamNode = nodeMapTLStream.GetNode("StreamBufferHandlingMode");
		StreamNode->SetIntValue(3); // newest, overwrite
	} catch (Spinnaker::Exception &e) {
		cout << "Error: " << e.what() << endl;
		return false;
	}
	
	try {
		pCam->ChunkSelector.SetValue(ChunkSelectorEnums::ChunkSelector_Timestamp);
		pCam->ChunkEnable.SetValue(true);

		pCam->ChunkSelector.SetValue(ChunkSelectorEnums::ChunkSelector_OffsetX);
		pCam->ChunkEnable.SetValue(true);

		pCam->ChunkSelector.SetValue(ChunkSelectorEnums::ChunkSelector_OffsetY);
		pCam->ChunkEnable.SetValue(true);

		// frame id chunk does not work

		pCam->ChunkModeActive.SetValue(true);
	} catch (Spinnaker::Exception &e) {
		cout << "Error: " << e.what() << endl;
		return false;
	}

	// set the roi on the camera (don't change anymore)
	// if this is incorrect, crashes, check this!
	if (videomode == 0) {
		
		try {
			pCam->OffsetX.SetValue(xoffset, false);
			pCam->OffsetY.SetValue(yoffset, false);
			pCam->Width.SetValue(width, false);
			pCam->Height.SetValue(height, false);
		}
		catch (Spinnaker::Exception &e) {
			cout << "Error: " << e.what() << endl;
			return false;
		}
	}


	set_gain(gain);
	set_shutter(shutter);

	// and start the image stream
	try {
		pCam->BeginAcquisition();
	} catch (Spinnaker::Exception &e) {
		cout << "Error: " << e.what() << endl;
		return false;
	}

	timestamp_seconds_start = -1; // reset the timestamp to zero, dont know how to do this on the camera

	return true;
}

DLLEXPORT void reset_timestamp(void) {
	timestamp_seconds_start = -1;
}

DLLEXPORT bool close_cam(void) {

	pCam->EndAcquisition();
	pCam->DeInit();
	
	return true;
}

DLLEXPORT double get_image(unsigned char *image_data, size_t size) {

	if (pCam == NULL) return -1;
	
	ImagePtr pResultImage = pCam->GetNextImage();

	ChunkData chunkData = pResultImage->GetChunkData();

	memcpy(image_data, pResultImage->GetData(), size);

	pResultImage->Release();

	double timestamp_seconds_now = chunkData.GetTimestamp() / (1000 * 1000 * 1000.);
	if (timestamp_seconds_start == -1) timestamp_seconds_start = timestamp_seconds_now;

	return timestamp_seconds_now - timestamp_seconds_start;
}

DLLEXPORT double get_gain() {

	if (pCam == NULL) return 0;

	return pCam->Gain.GetValue();
}

DLLEXPORT bool set_gain(double value) {

	if (pCam == NULL) return false;

	pCam->GainAuto.SetValue(Spinnaker::GainAutoEnums::GainAuto_Off);

	try {
		pCam->Gain.SetValue(value);
	} catch (Spinnaker::Exception &e) {
		cout << "Error: " << e.what() << endl;
		return false;
	}

	return true;
}

DLLEXPORT double get_brightness() { return 0; }
DLLEXPORT bool set_brightness(double value) { return true; }

DLLEXPORT double get_shutter() {

	if (pCam == NULL) return 0;

	return pCam->ExposureTime.GetValue() / 1000.;  // make us in ms
}

DLLEXPORT bool set_shutter(double value) {

	if (pCam == NULL) return false;

	pCam->ExposureAuto.SetValue(Spinnaker::ExposureAutoEnums::ExposureAuto_Off);
	pCam->ExposureMode.SetValue(Spinnaker::ExposureModeEnums::ExposureMode_Timed);

	try {
		pCam->ExposureTime.SetValue(value * 1000); // in make ms in us
	} catch (Spinnaker::Exception &e) {
		cout << "Error: " << e.what() << endl;
		return false;
	}

	return true;
}