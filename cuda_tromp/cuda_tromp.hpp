#ifdef WIN32

#ifdef _LIB
#define DLL_CUDA_TROMP __declspec(dllexport)
#else
#define DLL_CUDA_TROMP
#endif

#ifdef CUDA_75
#define SOLVER_NAME cuda_tromp_75
#else
#define SOLVER_NAME cuda_tromp
#endif

struct eq_cuda_context;

struct DLL_CUDA_TROMP SOLVER_NAME
{
	int threadsperblock;
	int blocks;
	int device_id;
	eq_cuda_context* context;

	SOLVER_NAME(int platf_id, int dev_id);

	std::string getdevinfo();

	static int getcount();

	static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version);

	static void start(SOLVER_NAME& device_context);

	static void stop(SOLVER_NAME& device_context);

	static void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		SOLVER_NAME& device_context);

	std::string getname() { return "CUDA-TROMP"; }

private:
	std::string m_gpu_name;
	std::string m_version;
	int m_sm_count;
};

#ifndef _LIB
#undef SOLVER_NAME
#define SOLVER_NAME cuda_tromp_75

struct DLL_CUDA_TROMP SOLVER_NAME
{
	int threadsperblock;
	int blocks;
	int device_id;
	eq_cuda_context* context;

	SOLVER_NAME(int platf_id, int dev_id);

	std::string getdevinfo();

	static int getcount();

	static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version);

	static void start(SOLVER_NAME& device_context);

	static void stop(SOLVER_NAME& device_context);

	static void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		SOLVER_NAME& device_context);

	std::string getname() { return "CUDA-TROMP-75"; }

private:
	std::string m_gpu_name;
	std::string m_version;
	int m_sm_count;
};

#endif

#else // WIN32

// TODO fix this

#ifdef _LIB
#define DLL_CUDA_TROMP __declspec(dllexport)
#else
#define DLL_CUDA_TROMP
#endif

#ifdef CUDA_75
#define SOLVER_NAME cuda_tromp_75
#else
#define SOLVER_NAME cuda_tromp
#endif

struct eq_cuda_context;

struct DLL_CUDA_TROMP SOLVER_NAME
{
    int threadsperblock;
    int blocks;
    int device_id;
    eq_cuda_context* context;

    SOLVER_NAME(int platf_id, int dev_id);

    std::string getdevinfo();

    static int getcount();

    static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version);

    static void start(SOLVER_NAME& device_context);

    static void stop(SOLVER_NAME& device_context);

    static void solve(const char *tequihash_header,
        unsigned int tequihash_header_len,
        const char* nonce,
        unsigned int nonce_len,
        std::function<bool()> cancelf,
        std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
        std::function<void(void)> hashdonef,
        SOLVER_NAME& device_context);

    std::string getname() { return "CUDA-TROMP"; }

private:
    std::string m_gpu_name;
    std::string m_version;
    int m_sm_count;
};

#endif
