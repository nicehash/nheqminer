#include <iostream>
#include <chrono>
#include <vector>
#include <mutex>

#include "speed.hpp"


Speed::Speed(int interval) 
	: m_interval(interval), m_start(std::chrono::high_resolution_clock::now()) {}
Speed::~Speed() { }

void Speed::Add(std::vector<time_point>& buffer, std::mutex& mutex)
{
	mutex.lock();
	buffer.push_back(std::chrono::high_resolution_clock::now());
	mutex.unlock();
}

double Speed::Get(std::vector<time_point>& buffer, std::mutex& mutex)
{
	time_point now = std::chrono::high_resolution_clock::now();
	time_point past = now - std::chrono::seconds(m_interval);
	double interval = (double)m_interval;
	if (past < m_start)
	{
		interval = ((double)std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start).count()) / 1000;
		past = m_start;
	}
	size_t total = 0;

	mutex.lock();
	for (std::vector<time_point>::iterator it = buffer.begin(); it != buffer.end();)
	{
		if ((*it) < past)
		{
			it = buffer.erase(it);
		}
		else
		{
			++total;
			++it;
		}
	}
	mutex.unlock();

	return (double)total / (double)interval;
}

void Speed::AddHash()
{
	Add(m_buffer_hashes, m_mutex_hashes);
}

double Speed::GetHashSpeed()
{
	return Get(m_buffer_hashes, m_mutex_hashes);
}

void Speed::AddSolution()
{
	Add(m_buffer_solutions, m_mutex_solutions);
}

double Speed::GetSolutionSpeed()
{
	return Get(m_buffer_solutions, m_mutex_solutions);
}

void Speed::AddShare()
{
	Add(m_buffer_shares, m_mutex_shares);
}

double Speed::GetShareSpeed()
{
	return Get(m_buffer_shares, m_mutex_shares);
}

void Speed::AddShareOK()
{
	Add(m_buffer_shares_ok, m_mutex_shares_ok);
}


double Speed::GetShareOKSpeed()
{
	return Get(m_buffer_shares_ok, m_mutex_shares_ok);
}

void Speed::Reset()
{
	m_mutex_hashes.lock();
	m_buffer_hashes.clear();
	m_mutex_hashes.unlock();

	m_mutex_solutions.lock();
	m_buffer_solutions.clear();
	m_mutex_solutions.unlock();

	m_mutex_shares.lock();
	m_buffer_shares.clear();
	m_mutex_shares.unlock();

	m_mutex_shares_ok.lock();
	m_buffer_shares_ok.clear();
	m_mutex_shares_ok.unlock();

	m_start = std::chrono::high_resolution_clock::now();
}


Speed speed(INTERVAL_SECONDS);
