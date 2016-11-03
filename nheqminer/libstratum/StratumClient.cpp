// Copyright (c) 2016 Genoil <jw@meneer.net>
// Copyright (c) 2016 Jack Grigg <jack@z.cash>
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "StratumClient.h"
#include "version.h"
#include "streams.h"
//#include "util.h"

#include "utilstrencodings.h"

#include "json/json_spirit_reader_template.h"
#include "json/json_spirit_utils.h"

using boost::asio::ip::tcp;
using namespace json_spirit;

#include <boost/log/trivial.hpp>

#define BOOST_LOG_CUSTOM(sev) BOOST_LOG_TRIVIAL(sev) << "stratum | "


template <typename Miner, typename Job, typename Solution>
StratumClient<Miner, Job, Solution>::StratumClient(
		std::shared_ptr<boost::asio::io_service> io_s, Miner * m,
        string const & host, string const & port,
        string const & user, string const & pass,
        int const & retries, int const & worktimeout)
    : m_socket(*io_s)
{
	m_io_service = io_s;

    m_primary.host = host;
    m_primary.port = port;
    m_primary.user = user;
    m_primary.pass = pass;

    p_active = &m_primary;

    m_authorized = false;
    m_connected = false;
    m_maxRetries = retries;
    m_worktimeout = worktimeout;

    p_miner = m;
    p_current = nullptr;
    p_previous = nullptr;
    p_worktimer = nullptr;

    startWorking();
}

template <typename Miner, typename Job, typename Solution>
void StratumClient<Miner, Job, Solution>::setFailover(
        string const & host, string const & port)
{
    setFailover(host, port, p_active->user, p_active->pass);
}

template <typename Miner, typename Job, typename Solution>
void StratumClient<Miner, Job, Solution>::setFailover(
        string const & host, string const & port,
        string const & user, string const & pass)
{
    m_failover.host = host;
    m_failover.port = port;
    m_failover.user = user;
    m_failover.pass = pass;
}

template <typename Miner, typename Job, typename Solution>
void StratumClient<Miner, Job, Solution>::startWorking()
{
    m_work.reset(new std::thread([&]() {
        this->workLoop();
    }));
}

template <typename Miner, typename Job, typename Solution>
void StratumClient<Miner, Job, Solution>::workLoop()
{
	if (!p_miner->isMining()) {
		BOOST_LOG_CUSTOM(info) << "Starting miner";
		p_miner->start();
	}

    while (m_running) {
        try {
            if (!m_connected) {
                //m_io_service.run();
                //boost::thread t(boost::bind(&boost::asio::io_service::run, &m_io_service));
                connect();

            }
            read_until(m_socket, m_responseBuffer, "\n");
            std::istream is(&m_responseBuffer);
            std::string response;
            getline(is, response);

			BOOST_LOG_CUSTOM(trace) << "Received: " << response;

            if (!response.empty() && response.front() == '{' && response.back() == '}') {
                Value valResponse;
                if (read_string(response, valResponse) && valResponse.type() == obj_type) {
                    const Object& responseObject = valResponse.get_obj();
                    if (!responseObject.empty()) {
                        processReponse(responseObject);
                        m_response = response;
                    } else {
                        //LogS("[WARN] Response was empty\n");
                    }
                } else {
                    //LogS("[WARN] Parse response failed\n");
                }
            } else {
                //LogS("[WARN] Discarding incomplete response\n");
            }
        } catch (std::exception const& _e) {
			BOOST_LOG_CUSTOM(warning) << _e.what();
            reconnect();
        }
    }
}


template <typename Miner, typename Job, typename Solution>
void StratumClient<Miner, Job, Solution>::connect()
{
	BOOST_LOG_CUSTOM(info) << "Connecting to stratum server " << p_active->host << ":" << p_active->port;

    tcp::resolver r(*m_io_service);
    tcp::resolver::query q(p_active->host, p_active->port);
    tcp::resolver::iterator endpoint_iterator = r.resolve(q);
    tcp::resolver::iterator end;

    boost::system::error_code error = boost::asio::error::host_not_found;
    while (error && endpoint_iterator != end) {
        m_socket.close();
        m_socket.connect(*endpoint_iterator++, error);
    }
    if (error) {
		BOOST_LOG_CUSTOM(error) << "Could not connect to stratum server " <<
			p_active->host << ":" << p_active->port << ", " << error.message();
        reconnect();
    } else {
		BOOST_LOG_CUSTOM(info) << "Connected!";
        m_connected = true;
		std::stringstream ss;
		ss << "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\""
			<< p_miner->userAgent() << "\", null,\""
			<< p_active->host << "\",\""
			<< p_active->port << "\"]}\n";
		std::string sss = ss.str();
        std::ostream os(&m_requestBuffer);
		os << sss;
		BOOST_LOG_CUSTOM(trace) << "Sending: " << sss;
        write(m_socket, m_requestBuffer);

		m_share_id = 4;
    }
}

template <typename Miner, typename Job, typename Solution>
void StratumClient<Miner, Job, Solution>::reconnect()
{
	/*if (p_miner->isMining()) {
		BOOST_LOG_CUSTOM(info) << "Stopping miner";
		p_miner->stop();
	}*/
	p_miner->setJob(nullptr);

    if (p_worktimer) {
        p_worktimer->cancel();
        p_worktimer = nullptr;
    }

    //m_io_service.reset();
    //m_socket.close(); // leads to crashes on Linux
    m_authorized = false;
    m_connected = false;

    if (!m_failover.host.empty()) {
        m_retries++;

        if (m_retries > m_maxRetries) {
            if (m_failover.host == "exit") {
                disconnect();
                return;
            } else if (p_active == &m_primary) {
                p_active = &m_failover;
            } else {
                p_active = &m_primary;
            }
            m_retries = 0;
        }
    }

	BOOST_LOG_CUSTOM(info) << "Reconnecting in 3 seconds...";
    boost::asio::deadline_timer timer(*m_io_service, boost::posix_time::seconds(3));
    timer.wait();
}

template <typename Miner, typename Job, typename Solution>
void StratumClient<Miner, Job, Solution>::disconnect()
{
    if (!m_connected) return;
	BOOST_LOG_CUSTOM(info) << "Disconnecting";
    m_connected = false;
    m_running = false;
    if (p_miner->isMining()) {
		BOOST_LOG_CUSTOM(info) << "Stopping miner";
        p_miner->stop();
    }
    m_socket.close();
    //m_io_service.stop();
    if (m_work) {
        m_work->join();
        m_work.reset();
    }
}

template <typename Miner, typename Job, typename Solution>
void StratumClient<Miner, Job, Solution>::processReponse(const Object& responseObject)
{
    const Value& valError = find_value(responseObject, "error");
    if (valError.type() == array_type) {
        const Array& error = valError.get_array();
        string msg;
        if (error.size() > 0 && error[1].type() == str_type) {
            msg = error[1].get_str();
        } else {
            msg = "Unknown error";
        }
        //LogS("%s\n", msg);
    }
    std::ostream os(&m_requestBuffer);
	std::stringstream ss;
    const Value& valId = find_value(responseObject, "id");
    int id = 0;
    if (valId.type() == int_type) {
        id = valId.get_int();
    }
    Value valRes;
    bool accepted = false;
    switch (id) {
	case 0:
	{
		const Value& valMethod = find_value(responseObject, "method");
		string method = "";
		if (valMethod.type() == str_type) {
			method = valMethod.get_str();
		}

		if (method == "mining.notify") {
			const Value& valParams = find_value(responseObject, "params");
			if (valParams.type() == array_type) {
				const Array& params = valParams.get_array();
				Job* workOrder = p_miner->parseJob(params);

				if (workOrder)
				{
					if (!workOrder->clean)
					{
						BOOST_LOG_CUSTOM(info) << CL_CYN "Ignoring non-clean job #" << workOrder->jobId() << CL_N;;
						break;
					}

					BOOST_LOG_CUSTOM(info) << CL_CYN "Received new job #" << workOrder->jobId() << CL_N;
					workOrder->setTarget(m_nextJobTarget);

					if (!(p_current && *workOrder == *p_current)) {
						//x_current.lock();
						//if (p_worktimer)
						//    p_worktimer->cancel();

						if (p_previous) {
							delete p_previous;
						}
						p_previous = p_current;
						p_current = workOrder;

						p_miner->setJob(p_current);
						//x_current.unlock();
						//p_worktimer = new boost::asio::deadline_timer(m_io_service, boost::posix_time::seconds(m_worktimeout));
						//p_worktimer->async_wait(boost::bind(&StratumClient::work_timeout_handler, this, boost::asio::placeholders::error));
					}
				}
			}
		}
		else if (method == "mining.set_target") {
			const Value& valParams = find_value(responseObject, "params");
			if (valParams.type() == array_type) {
				const Array& params = valParams.get_array();
				m_nextJobTarget = params[0].get_str();
				BOOST_LOG_CUSTOM(info) << CL_MAG "Target set to " << m_nextJobTarget << CL_N;
			}
		}
		else if (method == "mining.set_extranonce") {
			const Value& valParams = find_value(responseObject, "params");
			if (valParams.type() == array_type) {
				const Array& params = valParams.get_array();
				p_miner->setServerNonce(params[0].get_str());
			}
		}
		else if (method == "client.reconnect") {
			const Value& valParams = find_value(responseObject, "params");
			if (valParams.type() == array_type) {
				const Array& params = valParams.get_array();
				if (params.size() > 1) {
					p_active->host = params[0].get_str();
					p_active->port = params[1].get_str();
				}
				// TODO: Handle wait time
				BOOST_LOG_CUSTOM(info) << "Reconnection requested";
				reconnect();
			}
		}
		break;
	}
    case 1:
        valRes = find_value(responseObject, "result");
        if (valRes.type() == array_type) {
			BOOST_LOG_CUSTOM(info) << "Subscribed to stratum server";
			const Array& result = valRes.get_array();
            // Ignore session ID for now.
            p_miner->setServerNonce(result[1].get_str());
			ss << "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\""
			   << p_active->user << "\",\"" << p_active->pass << "\"]}\n";
			std::string sss = ss.str();
			os << sss;
			BOOST_LOG_CUSTOM(trace) << "Sending: " << sss;
            write(m_socket, m_requestBuffer);
        }
        break;
	case 2:
	{
		valRes = find_value(responseObject, "result");
		m_authorized = false;
		if (valRes.type() == bool_type) {
			m_authorized = valRes.get_bool();
		}
		if (!m_authorized) {
			BOOST_LOG_CUSTOM(error) << "Worker not authorized: " << p_active->user;
			disconnect();
			return;
		}
		BOOST_LOG_CUSTOM(info) << "Authorized worker " << p_active->user;

		ss << "{\"id\":3,\"method\":\"mining.extranonce.subscribe\",\"params\":[]}\n";
		std::string sss = ss.str();
		os << sss;
		BOOST_LOG_CUSTOM(trace) << "Sending: " << sss;
		write(m_socket, m_requestBuffer);

		break;
	}
    case 3:
        // nothing to do...
        break;
    default:
        valRes = find_value(responseObject, "result");
        if (valRes.type() == bool_type) {
            accepted = valRes.get_bool();
        }
        if (accepted) {
			BOOST_LOG_CUSTOM(info) << CL_GRN "Accepted share #" << id << CL_N;
            p_miner->acceptedSolution(m_stale);
        } else {
			valRes = find_value(responseObject, "error");
			std::string reason = "unknown";
			if (valRes.type() == array_type)
			{
				const Array& params = valRes.get_array();
				if (params.size() > 1 && params[1].type() == str_type)
					reason = params[1].get_str();
			}
			BOOST_LOG_CUSTOM(warning) << CL_RED "Rejected share #" << id << CL_N " (" << reason << ")";
            p_miner->rejectedSolution(m_stale);
        }
        break;
    
    }
}

template <typename Miner, typename Job, typename Solution>
void StratumClient<Miner, Job, Solution>::work_timeout_handler(
        const boost::system::error_code& ec)
{
    if (!ec) {
        //LogS("No new work received in %d seconds.\n", m_worktimeout);
        reconnect();
    }
}

template <typename Miner, typename Job, typename Solution>
bool StratumClient<Miner, Job, Solution>::submit(const Solution* solution, const std::string& jobid)
{
	int id = std::atomic_fetch_add(&m_share_id, 1);
	BOOST_LOG_CUSTOM(info) << "Submitting share #" << id << ", nonce " << solution->toString().substr(0, 64 - solution->nonce1size);

	CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
	ss << solution->nonce;
	ss << solution->solution;
	std::string strHex = HexStr(ss.begin(), ss.end());

	std::stringstream stream;
	stream << "{\"id\":" << id << ",\"method\":\"mining.submit\",\"params\":[\"";
	stream << p_active->user;
	stream << "\",\"" << jobid;
	stream << "\",\"" << solution->time;
	stream << "\",\"" << strHex.substr(solution->nonce1size, 64 - solution->nonce1size);
	stream << "\",\"" << strHex.substr(64);
	stream << "\"]}\n";
	std::string json = stream.str();
	std::ostream os(&m_requestBuffer);
	os << json;
	BOOST_LOG_CUSTOM(trace) << "Sending: " << json;
	write(m_socket, m_requestBuffer);

	return true;
}

// XMP
template class StratumClient<ZMinerAVXCUDA80_XMP, ZcashJob, EquihashSolution>;
template class StratumClient<ZMinerSSE2CUDA80_XMP, ZcashJob, EquihashSolution>;
template class StratumClient<ZMinerAVXCUDA75_XMP, ZcashJob, EquihashSolution>;
template class StratumClient<ZMinerSSE2CUDA75_XMP, ZcashJob, EquihashSolution>;
// Silentarmy
template class StratumClient<ZMinerAVXCUDA80_SA, ZcashJob, EquihashSolution>;
template class StratumClient<ZMinerSSE2CUDA80_SA, ZcashJob, EquihashSolution>;
template class StratumClient<ZMinerAVXCUDA75_SA, ZcashJob, EquihashSolution>;
template class StratumClient<ZMinerSSE2CUDA75_SA, ZcashJob, EquihashSolution>;