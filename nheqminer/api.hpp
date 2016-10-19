#pragma once

class API
{
	std::shared_ptr<boost::asio::io_service> m_io_service;
	boost::asio::ip::tcp::acceptor m_acceptor;
	boost::asio::ip::tcp::socket m_socket;

	void do_accept();

public:
	API(std::shared_ptr<boost::asio::io_service> io_service);
	virtual ~API();

	bool start(int local_port);
	bool poll();
};


class Client : public std::enable_shared_from_this<Client>
{
	boost::asio::ip::tcp::socket m_socket;
	boost::asio::streambuf m_response_buffer;

	void ReadResponse(const boost::system::error_code& ec, std::size_t bytes_transferred);
	bool Parse(const std::string& request);

public:
	Client(boost::asio::ip::tcp::socket socket);
	virtual ~Client();

	void Start();
};