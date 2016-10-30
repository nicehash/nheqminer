#ifdef WIN32

#ifdef _USRDLL
#define DLL_CPU_TROMP __declspec(dllexport)
#else
#define DLL_CPU_TROMP
#endif

#if !defined(_USRDLL) || (defined(_USRDLL) && defined(__AVX__))
#define CPU_TROMP cpu_tromp_avx
#define CPU_TROMP_NAME "CPU-TROMP-AVX"

struct DLL_CPU_TROMP CPU_TROMP
{
    std::string getdevinfo() { return ""; }

    static void start(CPU_TROMP& device_context);

    static void stop(CPU_TROMP& device_context);

    static void solve(const char *tequihash_header,
        unsigned int tequihash_header_len,
        const char* nonce,
        unsigned int nonce_len,
        std::function<bool()> cancelf,
        std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
        std::function<void(void)> hashdonef,
        CPU_TROMP& device_context);

    std::string getname() { return CPU_TROMP_NAME; }

    int use_opt;
};

#endif

#if !defined(_USRDLL)
#undef CPU_TROMP
#undef CPU_TROMP_NAME
#endif

#if !defined(_USRDLL) || (defined(_USRDLL) && !defined(__AVX__))
#define CPU_TROMP cpu_tromp_sse2
#define CPU_TROMP_NAME "CPU-TROMP-SSE2"

struct DLL_CPU_TROMP CPU_TROMP
{
	std::string getdevinfo() { return ""; }

	static void start(CPU_TROMP& device_context);

	static void stop(CPU_TROMP& device_context);

	static void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		CPU_TROMP& device_context);

	std::string getname() { return CPU_TROMP_NAME; }

	int use_opt;
};

#endif

using cpu_tromp = CPU_TROMP;

#else // WIN32

#define CPU_TROMP cpu_tromp
#if defined(__AVX__)
#define CPU_TROMP_NAME "CPU-TROMP-AVX"
#elif defined(__SSE2__)
#define CPU_TROMP_NAME "CPU-TROMP-SSE2"
#endif

struct CPU_TROMP
{
    std::string getdevinfo() { return ""; }

    static void start(CPU_TROMP& device_context);

    static void stop(CPU_TROMP& device_context);

    static void solve(const char *tequihash_header,
        unsigned int tequihash_header_len,
        const char* nonce,
        unsigned int nonce_len,
        std::function<bool()> cancelf,
        std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
        std::function<void(void)> hashdonef,
        CPU_TROMP& device_context);

    std::string getname() { return CPU_TROMP_NAME; }

    int use_opt;
};

#endif
