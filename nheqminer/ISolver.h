#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

enum class SolverType {
	CPU = 0,
	CUDA,
	OPENCL
};

class ISolver
{
public:
	//ISolver() { }
	//virtual ~ISolver() { }
	virtual void start() = 0;
	virtual void stop() = 0;

	virtual void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef) = 0;

	virtual std::string getdevinfo() = 0;
	virtual std::string getname() = 0;
	virtual SolverType GetType() const = 0;
};

