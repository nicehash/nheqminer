#include <iostream>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

namespace logging = boost::log;
namespace keywords = boost::log::keywords;


namespace boost {
	BOOST_LOG_OPEN_NAMESPACE
	namespace aux {

		template< typename CharT, typename ArgsT >
		shared_ptr<
			sinks::synchronous_sink<
			sinks::basic_text_ostream_backend< CharT >
			>
		> add_console_log2(boost::log::core_ptr cptr, std::basic_ostream< CharT >& strm, ArgsT const& args)
		{
			shared_ptr< std::basic_ostream< CharT > > pStream(&strm, boost::null_deleter());

			typedef sinks::basic_text_ostream_backend< CharT > backend_t;
			shared_ptr< backend_t > pBackend = boost::make_shared< backend_t >();

			pBackend->add_stream(pStream);
			pBackend->auto_flush(args[keywords::auto_flush | false]);

			typedef sinks::synchronous_sink< backend_t > sink_t;
			shared_ptr< sink_t > pSink = boost::make_shared< sink_t >(pBackend);

			aux::setup_filter(*pSink, args,
				typename is_void< typename parameter::binding< ArgsT, keywords::tag::filter, void >::type >::type());

			aux::setup_formatter(*pSink, args,
				typename is_void< typename parameter::binding< ArgsT, keywords::tag::format, void >::type >::type());

			cptr->add_sink(pSink);

			return pSink;
		}

	}
	BOOST_LOG_CLOSE_NAMESPACE
}

#ifndef _DEBUG
__declspec(dllexport)
#endif
void init_logging(boost::log::core_ptr cptr, int level)
{
	std::cout << "Setting log level to " << level << std::endl;

	cptr->set_filter
	(
		logging::trivial::severity >= level
	);

	logging::aux::add_console_log2
	(
		cptr,
		std::cout,
		keywords::format = "[%TimeStamp%][%ThreadID%]: %Message%"
	);

	cptr->add_global_attribute("TimeStamp", boost::log::attributes::local_clock());
	cptr->add_global_attribute("ThreadID", boost::log::attributes::current_thread_id());
}