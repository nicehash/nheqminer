#pragma once

#include <iostream>

#include "arith_uint256.h"
#include "ISolver.h"

template<typename StaticInterface>
class Solver : public ISolver
{
protected:
	const SolverType _type;
	StaticInterface * const _context = nullptr;	
public:
	Solver(StaticInterface *contex, SolverType type) : _context(contex), _type(type){}
	virtual ~Solver() {
		// the solver owns the context should delete it
		if (_context != nullptr) {
			delete _context;
		}
	}

	virtual void start() override {
		StaticInterface::start(*_context);
	}

	virtual void stop() override {
		StaticInterface::stop(*_context);
	}

	virtual void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef) override {
		StaticInterface::solve(
			tequihash_header,
			tequihash_header_len,
			nonce,
			nonce_len,
			cancelf,
			solutionf,
			hashdonef,
			*_context);
	}

	virtual void solve_verus(CBlockHeader &bh,
		arith_uint256 &target,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef) override {
			std::cout << std::endl << "\tERROR: Calling solve_verus on non-Verus solver" << std::endl;
		#ifndef _WIN32
			sleep(1);
		#else
			_sleep(1000);
		#endif // !_WIN32


	}

	virtual std::string getdevinfo() override {
		return _context->getdevinfo();
	}

	virtual std::string getname() override {
		return _context->getname();
	}

	virtual SolverType GetType() const override {
		return _type;
	}
};

// we make it a separate solver with the same interface, rather than a test for minimum overhead
template<typename StaticInterface>
class SolverVerus : public ISolver
{
protected:
	const SolverType _type;
	StaticInterface * const _context = nullptr;	
public:
	SolverVerus(StaticInterface *contex, SolverType type) : _context(contex), _type(type){}
	virtual ~SolverVerus() {
		// the solver owns the context should delete it
		if (_context != nullptr) {
			delete _context;
		}
	}

	virtual void start() override {
		StaticInterface::start(*_context);
	}

	virtual void stop() override {
		StaticInterface::stop(*_context);
	}

	virtual void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef) override {
			std::cout << std::endl << "\tERROR: Calling solve instead of solve_verus on Verus solver" << std::endl;
			#ifndef _WIN32
						sleep(1);
			#else
						_sleep(1000);
			#endif // !_WIN32
	}

	virtual void solve_verus(CBlockHeader &bh, 
		arith_uint256 &target,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef) override {
		StaticInterface::solve_verus(
			bh,
			target,
			cancelf,
			solutionf,
			hashdonef,
			*_context);
	}

	virtual std::string getdevinfo() override {
		return _context->getdevinfo();
	}

	virtual std::string getname() override {
		return _context->getname();
	}

	virtual SolverType GetType() const override {
		return _type;
	}
};
