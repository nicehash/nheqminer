#include <iostream>
#include <chrono>
#include <mutex>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/log/trivial.hpp>

#define BOOST_LOG_CUSTOM(sev) BOOST_LOG_TRIVIAL(sev) << "api | "

#include "api.hpp"
#include "speed.hpp"


API::API(std::shared_ptr<boost::asio::io_service> io_service)
	: m_io_service(io_service), m_acceptor(*io_service), m_socket(*io_service)
{
}


API::~API()
{
	m_acceptor.close();
}


bool API::start(int local_port)
{
	boost::asio::ip::tcp::endpoint endp(boost::asio::ip::address_v4::from_string("127.0.0.1"), local_port);
	m_acceptor.open(endp.protocol());
	m_acceptor.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
	boost::system::error_code ec;
	m_acceptor.bind(endp, ec);

	if (ec)
	{
		BOOST_LOG_CUSTOM(info) << "Failed to bind local port " << local_port << " err: " << ec;
		return false;
	}

	m_acceptor.listen();
	do_accept();

	BOOST_LOG_CUSTOM(info) << "Listening on port " << local_port;

	return true;
}


bool API::poll()
{
	try
	{
		boost::system::error_code ec;
		if (m_io_service->poll(ec) > 0) return true;
		else return false;
	}
	catch (std::exception& ex)
	{
		BOOST_LOG_CUSTOM(error) << ex.what();
		return false;
	}
}


void API::do_accept()
{
	m_acceptor.async_accept(m_socket,
		[this](boost::system::error_code ec)
	{
		// Check whether the server was stopped by a signal before this
		// completion handler had a chance to run.
		if (!m_acceptor.is_open())
		{
			return;
		}

		if (!ec)
		{
			BOOST_LOG_CUSTOM(debug) << "Accepted " << m_socket.remote_endpoint();

			std::shared_ptr<Client> c(new Client(std::move(m_socket)));
			c->Start();
		}

		do_accept();
	});
}


Client::Client(boost::asio::ip::tcp::socket socket)
	: m_socket(std::move(socket))
{
}


Client::~Client()
{
}


void Client::Start()
{
	boost::asio::async_read_until(m_socket, m_response_buffer, "\n",
		boost::bind(&Client::ReadResponse, shared_from_this(),
		boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
}


void Client::ReadResponse(const boost::system::error_code& ec, std::size_t bytes_transferred)
{
	if (!ec && bytes_transferred)
	{
		std::istream is(&m_response_buffer);
		std::string line;
		std::getline(is, line);

		if (line.back() == '\r') line.pop_back();

		BOOST_LOG_CUSTOM(trace) << "Received: " << line;

		if (Parse(line))
			Start();
	}
	else if (ec)
	{
		BOOST_LOG_CUSTOM(debug) << "Connection lost";
	}
}


bool Client::Parse(const std::string& request)
{
	std::stringstream ss;
	ss << "{\"method\":\"" << request << "\",\"result\":";

	if (request == "status")
	{
		BOOST_LOG_CUSTOM(debug) << "Responding to status request";

		double allshares = speed.GetShareSpeed() * 60;
		double accepted = speed.GetShareOKSpeed() * 60;

		ss << "{\"interval_seconds\":" << INTERVAL_SECONDS << ",";
		ss << "\"speed_ips\":" << speed.GetHashSpeed() << ",";
		ss << "\"speed_sps\":" << speed.GetSolutionSpeed() << ",";
		ss << "\"accepted_per_minute\":" << accepted << ",";
		ss << "\"rejected_per_minute\":" << (allshares - accepted);
		ss << "},\"error\":null}";
	}
	else
	{
		BOOST_LOG_CUSTOM(debug) << "Invalid request: " << request;

		ss << "false,\"error\":\"Invalid request.\"}";
	}

	ss << "\n";
	std::string out = ss.str();

	BOOST_LOG_CUSTOM(trace) << "Sending: " << out;

	std::vector<boost::asio::const_buffer> buffers;
	buffers.push_back(boost::asio::buffer(out));

	boost::system::error_code ec;
	m_socket.send(buffers, 0, ec);
	if (ec)
	{
		BOOST_LOG_CUSTOM(debug) << ec;
		return false;
	}

	return true;
}
