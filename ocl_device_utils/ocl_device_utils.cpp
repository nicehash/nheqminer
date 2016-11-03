#include "ocl_device_utils.h"

#include <iostream>
#include <stdexcept>
#include <utility>
#include <algorithm>

using namespace std;
using namespace cl;


bool ocl_device_utils::_hasQueried = false;
std::vector<std::string> ocl_device_utils::_platformNames;
std::vector<PrintInfo> ocl_device_utils::_devicesPlatformsDevices;

vector<Platform> ocl_device_utils::getPlatforms() {
	vector<Platform> platforms;
	try {
		Platform::get(&platforms);
	}
	catch (Error const& err) {
#if defined(CL_PLATFORM_NOT_FOUND_KHR)
		if (err.err() == CL_PLATFORM_NOT_FOUND_KHR)
			cout << "No OpenCL platforms found" << endl;
		else
#endif
			throw err;
	}
	return platforms;
}

void ocl_device_utils::print_opencl_devices() {
	ocl_device_utils::QueryDevices();
	ocl_device_utils::PrintDevices();
}

vector<Device> ocl_device_utils::getDevices(vector<Platform> const& _platforms, unsigned _platformId) {
	vector<Device> devices;
	try {
		_platforms[_platformId].getDevices(/*CL_DEVICE_TYPE_CPU| */CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, &devices);
	}
	catch (Error const& err) {
		// if simply no devices found return empty vector
		if (err.err() != CL_DEVICE_NOT_FOUND)
			throw err;
	}
	return devices;
}

string ocl_device_utils::StringnNullTerminatorFix(const string& str) {
	return string(str.c_str(), strlen(str.c_str()));
}

bool ocl_device_utils::QueryDevices() {
	if (!_hasQueried) {
		_hasQueried = true;
		try {
			// get platforms
			auto platforms = getPlatforms();
			if (platforms.empty()) {
				cout << "No OpenCL platforms found" << endl;
				return false;
			}
			else {
				for (auto i_pId = 0u; i_pId < platforms.size(); ++i_pId) {
					string platformName = StringnNullTerminatorFix(platforms[i_pId].getInfo<CL_PLATFORM_NAME>());
					if (std::find(_platformNames.begin(), _platformNames.end(), platformName) == _platformNames.end()) {
						PrintInfo current;
						_platformNames.push_back(platformName);
						// new
						current.PlatformName = platformName;
						current.PlatformNum = i_pId;

						auto clDevs = getDevices(platforms, i_pId);
						for (auto i_devId = 0u; i_devId < clDevs.size(); ++i_devId) {
							OpenCLDevice curDevice;
							curDevice.DeviceID = i_devId;
							curDevice._CL_DEVICE_NAME = StringnNullTerminatorFix(clDevs[i_devId].getInfo<CL_DEVICE_NAME>());
							switch (clDevs[i_devId].getInfo<CL_DEVICE_TYPE>()) {
							case CL_DEVICE_TYPE_CPU:
								curDevice._CL_DEVICE_TYPE = "CPU";
								break;
							case CL_DEVICE_TYPE_GPU:
								curDevice._CL_DEVICE_TYPE = "GPU";
								break;
							case CL_DEVICE_TYPE_ACCELERATOR:
								curDevice._CL_DEVICE_TYPE = "ACCELERATOR";
								break;
							default:
								curDevice._CL_DEVICE_TYPE = "DEFAULT";
								break;
							}


							curDevice._CL_DEVICE_GLOBAL_MEM_SIZE = clDevs[i_devId].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
							curDevice._CL_DEVICE_VENDOR = StringnNullTerminatorFix(clDevs[i_devId].getInfo<CL_DEVICE_VENDOR>());
							curDevice._CL_DEVICE_VERSION = StringnNullTerminatorFix(clDevs[i_devId].getInfo<CL_DEVICE_VERSION>());
							curDevice._CL_DRIVER_VERSION = StringnNullTerminatorFix(clDevs[i_devId].getInfo<CL_DRIVER_VERSION>());

							current.Devices.push_back(curDevice);
						}
						_devicesPlatformsDevices.push_back(current);
					}
				}
			}
		}
		catch (exception &ex) {
			// TODO
			cout << "ocl_device_utils::QueryDevices() exception: " << ex.what() << endl;
			return false;
		}
		return true;
	}
	
	return false;
}

int ocl_device_utils::GetCountForPlatform(int platformID) {
	for (const auto &platInfo : _devicesPlatformsDevices)
	{
		if (platformID == platInfo.PlatformNum) {
			return platInfo.Devices.size();
		}
	}
	return 0;
}

void ocl_device_utils::PrintDevices() {
	int allDevsCount = 0;
	for (const auto &platInfo : _devicesPlatformsDevices) {
		allDevsCount += platInfo.Devices.size();
	}
	cout << "Number of OpenCL devices found: " << allDevsCount << endl;
	{
		int devPlatformsComma = _devicesPlatformsDevices.size();
		for (const auto &platInfo : _devicesPlatformsDevices) {
			cout << "\tPlatform: " << platInfo.PlatformName << " | " << "PlatformNum: " << platInfo.PlatformNum << endl;
			cout << "\t\tDevices: " << endl;
			// device print
			int devComma = platInfo.Devices.size();
			for (const auto &dev : platInfo.Devices) {
				cout << "\t\t\t#" << dev.DeviceID << " " << dev._CL_DEVICE_NAME << " | " << dev._CL_DEVICE_TYPE << endl;
			}
		}
	}
}