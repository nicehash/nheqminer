#pragma once
// Copyright (c) 2016 Genoil <jw@meneer.net>
// Copyright (c) 2016 Jack Grigg <jack@z.cash>
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "libstratum/ZcashStratum.h"

#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <mutex>
#include <thread>
#include <atomic>

#include "json/json_spirit_value.h"

using namespace std;
using namespace boost::asio;
using boost::asio::ip::tcp;
using namespace json_spirit;

#ifndef _MSC_VER
#define CONSOLE_COLORS
#endif

#ifndef CONSOLE_COLORS
#define CL_N    ""
#define CL_RED  ""
#define CL_GRN  ""
#define CL_YLW  ""
#define CL_BLU  ""
#define CL_MAG  ""
#define CL_CYN  ""

#define CL_BLK  "" /* black */
#define CL_RD2  "" /* red */
#define CL_GR2  "" /* green */
#define CL_YL2  "" /* dark yellow */
#define CL_BL2  "" /* blue */
#define CL_MA2  "" /* magenta */
#define CL_CY2  "" /* cyan */
#define CL_SIL  "" /* gray */

#define CL_GRY  "" /* dark gray */
#define CL_LRD  "" /* light red */
#define CL_LGR  "" /* light green */
#define CL_LYL  "" /* tooltips */
#define CL_LBL  "" /* light blue */
#define CL_LMA  "" /* light magenta */
#define CL_LCY  "" /* light cyan */

#define CL_WHT  "" /* white */
#else
#define CL_N    "\x1B[0m"
#define CL_RED  "\x1B[31m"
#define CL_GRN  "\x1B[32m"
#define CL_YLW  "\x1B[33m"
#define CL_BLU  "\x1B[34m"
#define CL_MAG  "\x1B[35m"
#define CL_CYN  "\x1B[36m"

#define CL_BLK  "\x1B[22;30m" /* black */
#define CL_RD2  "\x1B[22;31m" /* red */
#define CL_GR2  "\x1B[22;32m" /* green */
#define CL_YL2  "\x1B[22;33m" /* dark yellow */
#define CL_BL2  "\x1B[22;34m" /* blue */
#define CL_MA2  "\x1B[22;35m" /* magenta */
#define CL_CY2  "\x1B[22;36m" /* cyan */
#define CL_SIL  "\x1B[22;37m" /* gray */

#ifdef WIN32
#define CL_GRY  "\x1B[01;30m" /* dark gray */
#else
#define CL_GRY  "\x1B[90m"    /* dark gray selectable in putty */
#endif
#define CL_LRD  "\x1B[01;31m" /* light red */
#define CL_LGR  "\x1B[01;32m" /* light green */
#define CL_LYL  "\x1B[01;33m" /* tooltips */
#define CL_LBL  "\x1B[01;34m" /* light blue */
#define CL_LMA  "\x1B[01;35m" /* light magenta */
#define CL_LCY  "\x1B[01;36m" /* light cyan */

#define CL_WHT  "\x1B[01;37m" /* white */
#endif


typedef struct {
        string host;
        string port;
        string user;
        string pass;
} cred_t;

template <typename Miner, typename Job, typename Solution>
class StratumClient
{
public:
	StratumClient(std::shared_ptr<boost::asio::io_service> io_s, Miner * m,
                  string const & host, string const & port,
                  string const & user, string const & pass,
                  int const & retries, int const & worktimeout);
    ~StratumClient() { }

    void setFailover(string const & host, string const & port);
    void setFailover(string const & host, string const & port,
                     string const & user, string const & pass);

    bool isRunning() { return m_running; }
    bool isConnected() { return m_connected && m_authorized; }
    bool current() { return p_current; }
    bool submit(const Solution* solution, const std::string& jobid);
    void reconnect();
    void disconnect();

private:
    void startWorking();
    void workLoop();
    void connect();

    void work_timeout_handler(const boost::system::error_code& ec);

    void processReponse(const Object& responseObject);

    cred_t * p_active;
    cred_t m_primary;
    cred_t m_failover;

    bool m_authorized;
    bool m_connected;
    bool m_running = true;

    int    m_retries = 0;
    int    m_maxRetries;
    int m_worktimeout = 60;

    string m_response;

    Miner * p_miner;
    std::mutex x_current;
    Job * p_current;
    Job * p_previous;

    bool m_stale = false;

    std::unique_ptr<std::thread> m_work;

    std::shared_ptr<boost::asio::io_service> m_io_service;
    tcp::socket m_socket;

    boost::asio::streambuf m_requestBuffer;
    boost::asio::streambuf m_responseBuffer;

    boost::asio::deadline_timer * p_worktimer;

    string m_nextJobTarget;

	std::atomic_int m_share_id;

	unsigned char o_index;
};


// ZcashStratumClient
typedef StratumClient<ZcashMiner, ZcashJob, EquihashSolution> ZcashStratumClient;