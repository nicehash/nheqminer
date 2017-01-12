#pragma once

#define INTERVAL_SECONDS 15 // 15 seconds

class Speed
{
	int m_interval;

	using time_point = std::chrono::high_resolution_clock::time_point;

	time_point m_start;

	std::vector<time_point> m_buffer_hashes;
	std::vector<time_point> m_buffer_solutions;
	std::vector<time_point> m_buffer_shares;
	std::vector<time_point> m_buffer_shares_ok;

	std::mutex m_mutex_hashes;
	std::mutex m_mutex_solutions;
	std::mutex m_mutex_shares;
	std::mutex m_mutex_shares_ok;

	void Add(std::vector<time_point>& buffer, std::mutex& mutex);
	double Get(std::vector<time_point>& buffer, std::mutex& mutex);

public:
	Speed(int interval);
	virtual ~Speed();

	void AddHash();
	void AddSolution();
	void AddShare();
	void AddShareOK();
	double GetHashSpeed();
	double GetSolutionSpeed();
	double GetShareSpeed();
	double GetShareOKSpeed();

	void Reset();
};

extern Speed speed;