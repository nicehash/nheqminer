#pragma once

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