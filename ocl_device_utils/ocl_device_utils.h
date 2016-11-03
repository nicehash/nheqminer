#pragma once

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include "cl_ext.hpp"
#include <map>
#include <vector>
#include "OpenCLDevice.h"


struct PrintInfo {
	std::string PlatformName;
	int PlatformNum;
	std::vector<OpenCLDevice> Devices;
};

class ocl_device_utils {
public:
	static bool QueryDevices();
	static void PrintDevices();
	static int GetCountForPlatform(int platformID);
	static void print_opencl_devices();

private:
	static std::vector<cl::Device> getDevices(std::vector<cl::Platform> const& _platforms, unsigned _platformId);
	static std::vector<cl::Platform> getPlatforms();

	static bool _hasQueried;
	static std::vector<std::string> _platformNames;
	static std::vector<PrintInfo> _devicesPlatformsDevices;

	static std::string StringnNullTerminatorFix(const std::string& str);
};