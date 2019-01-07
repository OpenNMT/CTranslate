#pragma once

#include <cstddef>
#include <cstdint>

#ifdef WITH_BOOST_LOG
#  include "onmt/Logger.h"
#  define ONMT_LOG_STREAM_SEV(s, l) BOOST_LOG_SEV(onmt::Logger::lg(), l) << s
#else
#  define ONMT_LOG_STREAM_SEV(s, l) ((void)0)
#endif

namespace onmt
{

  inline void *align( std::size_t alignment, std::size_t size,
                      void *&ptr, std::size_t &space ) {
    // Copyright 2014 David Krauss
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57350
    std::uintptr_t pn = reinterpret_cast< std::uintptr_t >( ptr );
    std::uintptr_t aligned = ( pn + alignment - 1 ) & - alignment;
    std::size_t padding = aligned - pn;
    if ( space < size + padding ) return nullptr;
    space -= padding;
    return ptr = reinterpret_cast< void * >( aligned );
  }

}
