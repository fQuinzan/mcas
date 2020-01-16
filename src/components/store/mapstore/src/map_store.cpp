/*
  Copyright [2017-2019] [IBM Corporation]
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
#include <api/kvstore_itf.h>
#include <city.h>
#include <common/exceptions.h>
#include <common/rwlock.h>
#include <common/cycles.h>
#include <common/utils.h>
#include <fcntl.h>
#include <nupm/allocator_ra.h>
#include <nupm/rc_alloc_lb.h>
#include <stdio.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cerrno>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <tbb/scalable_allocator.h>
#pragma GCC diagnostic pop

#define ENABLE_TIMESTAMPS
#define DEFAULT_ALIGNMENT 8
#define SINGLE_THREADED
#define NUMA_ZONE 0 /* treat memory as a single zone, although it may not be */
#define MIN_POOL GB(1)

#include "map_store.h"

using namespace Component;
using namespace Common;

struct Value_type {
  void * _ptr;
  size_t _length;
  Common::RWLock * _value_lock; /*< read write lock */
#ifdef ENABLE_TIMESTAMPS
  tsc_time_t _tsc;
#endif
};

  
class Key_hash;

using aac_t = nupm::allocator_adaptor<char, nupm::Rca_LB>;
using string_t = std::basic_string<char, std::char_traits<char>, aac_t>;
using aam_t = nupm::allocator_adaptor<std::pair<string_t, Value_type>, nupm::Rca_LB>;
using map_t = std::unordered_map<string_t, Value_type, Key_hash,
                                 std::equal_to<string_t>, aam_t>;
using aal_t = nupm::allocator_adaptor<Common::RWLock, nupm::Rca_LB>;

static size_t choose_alignment(size_t size)
{
  if((size >= 4096) && (size % 4096 == 0)) return 4096;
  if((size >= 64) && (size % 64 == 0)) return 64;
  if((size >= 16) && (size % 16 == 0)) return 16;
  if((size >= 8) && (size % 8 == 0)) return 8;
  if((size >= 4) && (size % 4 == 0)) return 4;
  return 1;
}

class Key_hash {
public:
  size_t operator()(string_t const &k) const {
    return CityHash64(k.c_str(), k.length());
  }
};

/* time stamp datum */
static epoch_time_t epoch_at_power_on; /*< epoch time when local tsc == 0 */
static long long    ticks_per_second;

static epoch_time_t tsc_to_epoch(tsc_time_t tsc) {
  return epoch_at_power_on + (tsc / ticks_per_second);
}

static tsc_time_t epoch_to_tsc(epoch_time_t epoch) {
  if(epoch < epoch_at_power_on)
    throw std::range_error("epoch earlier than power-on epoch");
  return (epoch - epoch_at_power_on) * ticks_per_second;
}

static void __attribute__((constructor)) initialize_time_datum()
{
  /* establish time datum */
  ticks_per_second = std::llround(get_rdtsc_frequency_mhz()) * 1000000LL;
  time_t epoch;
  time(&epoch);
  tsc_time_t tsc = rdtsc();
  epoch_at_power_on = epoch - (tsc / ticks_per_second);
}


static void * allocate_region_memory(size_t alignment, size_t size)
{
  assert(size > 0);
  
  void *p = mmap(nullptr,
                 size, 
                 PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_SHARED | MAP_LOCKED,
                 0, /* file */
                 0 /* offset */);

  if (p == nullptr || p == reinterpret_cast<void*>(-1)) {
    perror("");
    throw General_exception("mmap failed on DRAM for region allocation (alignment=%lu, size=%lu)",
                            alignment, size);
  }

  if(madvise(p, size, MADV_DONTFORK) != 0)
    throw General_exception("madvise 'don't fork' failed unexpectedly (%p %lu)", p, size);

  memset(p, 0, size);
  PMAJOR("allocated memory region (%p,%lu)", p, size);
  return p; 
}


class Pool_handle {
private:
  static constexpr unsigned _debug_level = 0;

  class Iterator {    
  public:
    explicit Iterator(const Pool_handle * pool) : _pool(pool) {
      if(pool == nullptr) throw Logic_exception("bad iterator ctor param");
      _mark = pool->writes();
      _iter = pool->_map.begin();
      _end = pool->_map.end();
    }

    bool is_end() const { return _iter == _end; }
    bool check_mark(uint32_t writes) const { return _mark == writes; }

    const Pool_handle *   _pool;
    uint32_t              _mark;
    map_t::const_iterator _iter;
    map_t::const_iterator _end;
  };
  
public:
  Pool_handle(size_t nsize)
    : _nsize(nsize < MIN_POOL ? MIN_POOL : nsize), // * 1024 > MIN_POOL ? nsize * 1024 : MIN_POOL),
      _tmp({allocate_region_memory(MB(2), _nsize), _nsize}),
      _map({(_lb.add_managed_region(_tmp.iov_base, _nsize, NUMA_ZONE), aam_t(_lb))})
  {
    _regions.push_back(_tmp);
    if(_debug_level)
      PLOG("Map_store: added memory region (%p,%lu)",_tmp.iov_base, _tmp.iov_len);
  }

  ~Pool_handle() {
    // FIX DAWN-295
    //    for(auto r : _regions)
    //      release_region_memory(r.iov_base, r.iov_len);
  }
  
  size_t               _nsize; /*< order important */
  ::iovec              _tmp;
  std::vector<::iovec> _regions;
  std::string          _name;
  nupm::Rca_LB         _lb;
  map_t                _map; /*< hash table based map */
  Common::RWLock       _map_lock; /*< read write lock */
  unsigned int         _flags;
  std::set<Iterator*>  _iterators;
  
private:
  /* 
     We use this counter to see if new writes have come in
     during an iteration.  This is essentially an optmistic
     locking strategy. 
  */
  uint32_t             _writes __attribute__((aligned(4)));
  
  inline void write_touch() { _writes++; }  
  inline uint32_t writes() const { return _writes; }
  
  aac_t aac{_lb};
  aal_t aal{_lb};

public:
  status_t put(const std::string &key, const void *value,
               const size_t value_len, unsigned int flags);

  status_t get(const std::string &key, void *&out_value, size_t &out_value_len);

  status_t get_direct(const std::string &key, void *out_value,
                      size_t &out_value_len);

  status_t get_attribute(const IKVStore::pool_t pool,
                         const IKVStore::Attribute attr,
                         std::vector<uint64_t> &out_attr,
                         const std::string *key);

  status_t resize_value(const std::string &key,
                        const size_t new_size,
                        const size_t alignment);

  status_t lock(const std::string &key,
                IKVStore::lock_type_t type,
                void *&out_value,
                size_t &out_value_len,
                IKVStore::key_t& out_key);

  void unlock(IKVStore::key_t key_handle);

  status_t erase(const std::string &key);

  size_t count();

  status_t map(std::function<int(const void * key,
                                 const size_t key_len,
                                 const void * value,
                                 const size_t value_len)> function);

  status_t map(std::function<int(const void* key,
                                 const size_t key_len,
                                 const void* value,
                                 const size_t value_len,
                                 const tsc_time_t timestamp)> function,
               const epoch_time_t t_begin,
               const epoch_time_t t_end);

  status_t map_keys(std::function<int(const std::string &key)> function);

  status_t get_pool_regions(std::vector<::iovec> &out_regions);

  status_t grow_pool(const size_t increment_size, size_t &reconfigured_size);

  status_t free_pool_memory(const void *addr, const size_t size = 0);
  
  status_t allocate_pool_memory(const size_t size,
                                const size_t alignment,
                                void *&out_addr);

  IKVStore::pool_iterator_t open_pool_iterator();

  status_t deref_pool_iterator(IKVStore::pool_iterator_t iter,
                               const epoch_time_t t_begin,
                               const epoch_time_t t_end,
                               IKVStore::pool_reference_t& ref,
                               bool& time_match,
                               bool increment = true);
  
  status_t close_pool_iterator(IKVStore::pool_iterator_t iter);

};

struct Pool_session {
  Pool_session(Pool_handle *ph) : pool(ph) {}
  bool check() const { return canary == 0x45450101; }
  Pool_handle *pool;
  const unsigned canary = 0x45450101;
};

struct tls_cache_t {
  Pool_session *session;
};

std::mutex                                     _pool_sessions_lock;
std::set<Pool_session *>                       _pool_sessions;
std::unordered_map<std::string, Pool_handle *> _pools; /*< existing pools */
static __thread tls_cache_t tls_cache = {nullptr};

using Std_lock_guard = std::lock_guard<std::mutex>;

Pool_session *get_session(const IKVStore::pool_t pid) {
  auto session = reinterpret_cast<Pool_session *>(pid);

  if (session != tls_cache.session) {
    Std_lock_guard g(_pool_sessions_lock);

    if (_pool_sessions.count(session) == 0) return nullptr;

    tls_cache.session = session;
  }

  assert(session);
  return session;
}

status_t Pool_handle::put(const std::string &key,
                          const void *value,
                          const size_t value_len,
                          unsigned int flags) {
  if (!value || !value_len || value_len > _nsize) {
    PWRN("Map_store: invalid parameters (value=%p, value_len=%lu)", value,
         value_len);
    return E_INVAL;
  }

#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock, RWLock_guard::WRITE);
#endif

  write_touch(); /* this could be early, but over-conservative is ok */
  
  string_t k(key.data(), key.length(), aac);

  auto i = _map.find(k);
  if (i != _map.end()) {
    if (flags & IKVStore::FLAGS_DONT_STOMP) {
      PWRN("put refuses to stomp (%s)", key.c_str());
      return IKVStore::E_KEY_EXISTS;
    }

    auto &p = i->second;
    
    if (p._length == value_len) {
      memcpy(p._ptr, value, value_len);
    }
    else {
      /* different size, reallocate */
      try {
        _lb.free(p._ptr, NUMA_ZONE, p._length);
      }
      catch(...) {
        return E_FAIL;
      }

      p._ptr = _lb.alloc(value_len,
                         NUMA_ZONE, choose_alignment(value_len));
      
      memcpy(p._ptr, value, value_len);
      /* update entry */
      i->second._length = value_len;
      i->second._ptr = p._ptr;
    }
#ifdef ENABLE_TIMESTAMPS
    wmb();
    i->second._tsc = rdtsc(); /* update time stamp */
#endif
  }
  else {
    auto buffer = _lb.alloc(value_len,
                            NUMA_ZONE, choose_alignment(value_len));

    memcpy(buffer, value, value_len);
    Common::RWLock * p = new (aal.allocate(1, DEFAULT_ALIGNMENT)) Common::RWLock();
#ifdef ENABLE_TIMESTAMPS
    _map.emplace(k, Value_type{buffer, value_len, p, rdtsc()});
#else
    _map.emplace(k, Value_type{buffer, value_len, p});
#endif
  }

  return S_OK;
}

status_t Pool_handle::get(const std::string &key,
                          void *&out_value,
                          size_t &out_value_len) {
  if (_debug_level)
    PLOG("Map_store: get(%s,%p,%lu)", key.c_str(), out_value, out_value_len);

#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock);
#endif
  string_t k(key.c_str(), aac);
  auto i = _map.find(k);

  if (i == _map.end()) return IKVStore::E_KEY_NOT_FOUND;

  out_value_len = i->second._length;

  /* we are supposed to use plain malloc for return result
     or change free_memory ? */
  //  out_value = _lb.alloc(out_value_len, NUMA_ZONE, choose_alignment(out_value_len));
  out_value = malloc(out_value_len);
  memcpy(out_value, i->second._ptr, i->second._length);

  return S_OK;
}

status_t Pool_handle::get_direct(const std::string &key,
                                 void *out_value,
                                 size_t &out_value_len) {
  if (_debug_level) PLOG("Map_store GET: key=(%s) ", key.c_str());

  if (out_value == nullptr || out_value_len == 0)
    throw API_exception("invalid parameter");

#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock);
#endif
  string_t k(key.c_str(), aac);
  auto i = _map.find(k);

  if (i == _map.end()) {
    if (_debug_level) PERR("Map_store: error key not found");
    return IKVStore::E_KEY_NOT_FOUND;
  }

  if (out_value_len < i->second._length) {
    if (_debug_level) PERR("Map_store: error insufficient buffer");

    return E_INSUFFICIENT_BUFFER;
  }

  out_value_len = i->second._length; /* update length */
  memcpy(out_value, i->second._ptr, i->second._length);

  return S_OK;
}

status_t Pool_handle::get_attribute(const IKVStore::pool_t, // pool
                                    const IKVStore::Attribute attr,
                                    std::vector<uint64_t> &out_attr,
                                    const std::string *key) {
  switch (attr) {
  case IKVStore::Attribute::VALUE_LEN: {
    if (key == nullptr) return E_INVALID_ARG;
    out_attr.clear();
#ifndef SINGLE_THREADED
    RWLock_guard guard(map_lock);
#endif
    string_t k(key->c_str(), aac);
    auto i = _map.find(k);
    if (i == _map.end()) return IKVStore::E_KEY_NOT_FOUND;
    out_attr.push_back(i->second._length);
    break;
  }
#ifdef ENABLE_TIMESTAMPS
  case IKVStore::Attribute::WRITE_EPOCH_TIME: {
    out_attr.clear();
#ifndef SINGLE_THREADED
    RWLock_guard guard(map_lock);
#endif
    string_t k(key->c_str(), aac);    
    auto i = _map.find(k);
    if (i == _map.end()) return IKVStore::E_KEY_NOT_FOUND;
    out_attr.push_back(tsc_to_epoch(i->second._tsc));
    break;
  }
#endif
  case IKVStore::Attribute::COUNT: {
    out_attr.push_back(_map.size());
    break;
  }
  default:
    return E_INVALID_ARG;
  }

  return S_OK;
}

status_t Pool_handle::lock(const std::string &key,
                           IKVStore::lock_type_t type,
                           void *&out_value,
                           size_t &out_value_len,
                           IKVStore::key_t& out_key)
{
    
  void *buffer = nullptr;
  string_t k(key.c_str(), aac);
  bool created = false;

  try {
    auto i = _map.find(k);

    if(_debug_level)
      PLOG("Map_store: looking for key:(%s) %lu", key.c_str(), key.length());

    if (i == _map.end()) {

      write_touch();
      
      /* lock API has semantics of create on demand */
      if (out_value_len == 0) {
        out_key = IKVStore::KEY_NONE;
        return E_INVAL;
      }

      buffer = _lb.alloc(out_value_len, NUMA_ZONE, choose_alignment(out_value_len));

      if (buffer == nullptr)
        throw General_exception("Pool_handle::lock on-demand create allocate_memory failed (len=%lu)",
                                out_value_len);
      created = true;

      if(_debug_level)
        PLOG("Map_store: creating on demand key=(%s) len=%lu",
             key.c_str(),
             out_value_len);

      Common::RWLock * p = new (aal.allocate(1, DEFAULT_ALIGNMENT)) Common::RWLock();
      _map.emplace(k, Value_type{buffer, out_value_len, p, 0});
    }
  }
  catch (...) {
    out_key = IKVStore::KEY_NONE;
    return E_INVAL;
  }

  if(_debug_level)
    PLOG("Map_store: got key");
  
  if (type == IKVStore::STORE_LOCK_READ) {
    if(_map[k]._value_lock->read_trylock() != 0) {
      if(_debug_level)
        PWRN("Map_store: key (%s) unable to take read lock", key.c_str());
      
      out_key = IKVStore::KEY_NONE;
      return E_INVAL;
    }
  }
  else if (type == IKVStore::STORE_LOCK_WRITE) {
    
    write_touch();
    
    if(_map[k]._value_lock->write_trylock() != 0) {
      if(_debug_level)
        PWRN("Map_store: key (%s) unable to take write lock", key.c_str());
      
      out_key = IKVStore::KEY_NONE;
      return E_INVAL;
    }
  }
  else throw API_exception("invalid lock type");

  out_value = _map[k]._ptr;
  out_value_len = _map[k]._length;

#ifdef ENABLE_TIMESTAMPS    
  _map[k]._tsc = rdtsc(); /* update time stamp */
#endif
  
  out_key = reinterpret_cast<IKVStore::key_t>(_map[k]._value_lock);
  
  return created ? S_OK_CREATED : S_OK;
}

void Pool_handle::unlock(IKVStore::key_t key_handle) {
  /* TODO: how do we know key_handle is valid? */
  if(reinterpret_cast<Common::RWLock *>(key_handle)->unlock() != 0)
    throw Logic_exception("Pool_handle::unlock Common::RWLock::unlock failed unexpectedly");
}

status_t Pool_handle::erase(const std::string &key) {
#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock, RWLock_guard::WRITE);
#endif
  string_t k(key.c_str(), aac);
  auto i = _map.find(k);

  if (i == _map.end()) return IKVStore::E_KEY_NOT_FOUND;

  write_touch();
  _map.erase(i);
  
  try {
    _lb.free(i->second._ptr, NUMA_ZONE, i->second._length);
  }
  catch(...) {
    return E_FAIL;
  }
  aal.deallocate(i->second._value_lock, 1, DEFAULT_ALIGNMENT);

  return S_OK;
}

size_t Pool_handle::count() {
#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock);
#endif
  return _map.size();
}

status_t Pool_handle::map(std::function<int(const void * key,
                                            const size_t key_len,
                                            const void * value,
                                            const size_t value_len)> function)
{
#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock);
#endif

  for (auto &pair : _map) {
    auto val = pair.second;
    function(pair.first.c_str(), pair.first.length(), val._ptr, val._length);
  }

  return S_OK;
}

status_t Pool_handle::map(std::function<int(const void* key,
                                            const size_t key_len,
                                            const void* value,
                                            const size_t value_len,
                                            const tsc_time_t timestamp)> function,
                          const epoch_time_t t_begin,
                          const epoch_time_t t_end)
{
#ifdef ENABLE_TIMESTAMPS

#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock);
#endif

  auto begin_tsc = (t_begin == 0) ? 0 : epoch_to_tsc(t_begin);
  auto end_tsc = (t_end == 0) ? 0 : epoch_to_tsc(t_end);

  for (auto &pair : _map) {
    auto val = pair.second;
    if(val._tsc >= begin_tsc && (end_tsc == 0 || val._tsc <= end_tsc)) {
      if(function(pair.first.c_str(),
                  pair.first.length(),
                  val._ptr,
                  val._length,
                  val._tsc) < 0) {
        return S_MORE;
      }
    }
  }

  return S_OK;
#else
  return E_NOT_SUPPORTED;
#endif
}


status_t Pool_handle::map_keys(std::function<int(const std::string &key)> function) {
#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock);
#endif

  for (auto &pair : _map) function(std::string(pair.first.c_str()));

  return S_OK;
}

status_t Pool_handle::resize_value(const std::string &key,
                                   const size_t new_size,
                                   const size_t alignment) {
  if (new_size == 0) return E_INVAL;

#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock);
#endif

  auto i = _map.find(string_t(key.c_str(), aac));

  if (i == _map.end()) return IKVStore::E_KEY_NOT_FOUND;
  if (i->second._length == new_size) return E_INVAL;

  write_touch();
  
  /* perform resize */
  auto buffer = _lb.alloc(new_size, NUMA_ZONE, alignment);

  /* lock KV-pair */
  void *out_value;
  size_t out_value_len;
  IKVStore::key_t out_key;
  status_t s = lock(key,
                    IKVStore::STORE_LOCK_WRITE,
                    out_value,
                    out_value_len,
                    out_key);

  if (out_key == IKVStore::KEY_NONE) return E_INVAL;

  size_t size_to_copy = std::min<size_t>(new_size, boost::numeric_cast<size_t>(i->second._length));

  memcpy(buffer, i->second._ptr, size_to_copy);

  /* free previous memory */
  try {
    _lb.free(i->second._ptr, NUMA_ZONE, i->second._length);
  }
  catch(...) {
    return E_FAIL;
  }

  i->second._ptr = buffer;
  i->second._length = new_size;

  /* release lock */
  unlock(out_key);
  return s;
}

status_t Pool_handle::get_pool_regions(std::vector<::iovec> &out_regions) {
  if (_regions.empty()) {
    return E_INVAL;
  }
  for (auto region : _regions)
    out_regions.push_back(region); 
  return S_OK;
}

status_t Pool_handle::grow_pool(const size_t increment_size,
                                size_t &reconfigured_size) {
  if (increment_size <= 0) {
    return E_INVAL;
  }
  reconfigured_size = _nsize + increment_size;
  void *new_region = allocate_region_memory(DEFAULT_ALIGNMENT, increment_size);
  _lb.add_managed_region(new_region, increment_size, NUMA_ZONE);
  _regions.push_back({new_region, increment_size});
  _nsize = reconfigured_size;
  return S_OK;
}

status_t Pool_handle::free_pool_memory(const void *addr, const size_t size) {
  if (!addr || _regions.empty()) {
    return E_INVAL;
  }
  try {
    _lb.free(const_cast<void *>(addr), NUMA_ZONE, size);
  }
  catch(...) {
    return E_FAIL;
  }
  /* the regions are not freed */
  return S_OK;
}

status_t Pool_handle::allocate_pool_memory(const size_t size,
                                           const size_t alignment,
                                           void *&out_addr) {
  if (size == 0 || size > _nsize || _regions.empty()) {
    return E_INVAL;
  }

  size_t ssize = size < 8UL ? 8UL : size;

  try {
    /* we can't fully support alignment choice */
    out_addr = _lb.alloc(ssize, NUMA_ZONE, (alignment > 0) && (size % alignment == 0) ? alignment : choose_alignment(ssize));
  }
  catch(...) {
    return E_INVAL;
  }
  return S_OK;
}


IKVStore::pool_iterator_t Pool_handle::open_pool_iterator()
{
  auto i = new Iterator(this);
  _iterators.insert(i);
  return reinterpret_cast<IKVStore::pool_iterator_t>(i);
}

status_t Pool_handle::deref_pool_iterator(IKVStore::pool_iterator_t iter,
#ifdef ENABLE_TIMESTAMPS  
                                          const epoch_time_t t_begin,
                                          const epoch_time_t t_end,
#else
                                          const epoch_time_t ,
                                          const epoch_time_t ,                                          
#endif
                                          IKVStore::pool_reference_t& ref,
                                          bool& time_match,
                                          bool increment)
{

  auto i = reinterpret_cast<Iterator*>(iter);
  if(_iterators.count(i) != 1) return E_INVAL;

  if(i->is_end()) return E_OUT_OF_BOUNDS;
  if(!i->check_mark(_writes)) return E_ITERATOR_DISTURBED;

#ifdef ENABLE_TIMESTAMPS  
  auto begin_tsc = (t_begin == 0) ? 0 : epoch_to_tsc(t_begin);
  auto end_tsc = (t_end == 0) ? 0 : epoch_to_tsc(t_end);
#endif

  auto r = i->_iter;
  ref.key = r->first.data();
  ref.key_len = r->first.length();
  ref.value = r->second._ptr;
  ref.value_len = r->second._length;
#ifdef ENABLE_TIMESTAMPS  
  ref.timestamp = r->second._tsc;
  time_match = (ref.timestamp >= begin_tsc && (end_tsc == 0 || ref.timestamp <= end_tsc));
#endif

  if(increment) {
    try {
      i->_iter++;
    }
    catch(...) {
      return E_ITERATOR_DISTURBED;
    }
  }
  
  return S_OK;
}
  
status_t Pool_handle::close_pool_iterator(IKVStore::pool_iterator_t iter)
{
  auto i = reinterpret_cast<Iterator*>(iter);
  if(iter == nullptr || _iterators.erase(i) != 1) return E_INVAL;
  delete i;
  return S_OK;
}


/** Main class */

Map_store::Map_store(const std::string&, const std::string &)
{
}

Map_store::~Map_store() {
  // TODO: seg faults?
#if 0
  Std_lock_guard g(_pool_sessions_lock);

  for(auto& s : _pool_sessions)
    delete s;

  for(auto& p : _pools)
    delete p.second;
#endif
}

IKVStore::pool_t Map_store::create_pool(const std::string &name,
                                        const size_t nsize, unsigned int flags,
                                        uint64_t /*args*/) {
  if (flags & IKVStore::FLAGS_READ_ONLY)
    throw API_exception("read only create_pool not supported on map-store component");

  const auto handle = new Pool_handle(nsize);
  Pool_session *session = nullptr;
  handle->_name = name;
  handle->_flags = flags;

  // handle->_map.reserve(nsize);
  {
    Std_lock_guard g(_pool_sessions_lock);

    if (flags & IKVStore::FLAGS_CREATE_ONLY) {
      if (_pools.find(handle->_name) != _pools.end()) {
        delete handle;
        return POOL_ERROR;
      }
    }
    session = new Pool_session{handle};
    _pools[handle->_name] = handle;
    PLOG("Map_store: adding new session (%p)", session);
    _pool_sessions.insert(session); /* create a session too */
  }

  if (_debug_level)
    PLOG("Map_store: created pool OK: %s", handle->_name.c_str());

  assert(session);
  return reinterpret_cast<IKVStore::pool_t>(session);
}

IKVStore::pool_t Map_store::open_pool(const std::string &name,
                                      unsigned int /*flags*/) {
  const std::string &key = name;

  Pool_handle *ph = nullptr;
  /* see if a pool exists that matches the key */
  for (auto &h : _pools) {
    PLOG("pool:%s", h.first.c_str());
    if (h.first == key) {
      ph = h.second;
      break;
    }
  }

  if (ph == nullptr) return Component::IKVStore::POOL_ERROR;

  auto new_session = new Pool_session(ph);
  if (_debug_level) PLOG("Map_store: opened pool(%p)", new_session);
  _pool_sessions.insert(new_session);

  return reinterpret_cast<IKVStore::pool_t>(new_session);
}

status_t Map_store::close_pool(const pool_t pid) {
  if (_debug_level) PLOG("Map_store: close_pool(%p)", reinterpret_cast<void *>(pid));

  auto session = get_session(pid);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  tls_cache.session = nullptr;
  Std_lock_guard g(_pool_sessions_lock);
  delete session;
  _pool_sessions.erase(session);
  if (_debug_level) PLOG("Map_store: erased session %p", session);

  return S_OK;
}

status_t Map_store::delete_pool(const std::string &poolname) {
  Std_lock_guard g(_pool_sessions_lock);

  // return S_OK;
  Pool_handle *ph = nullptr;
  /* see if a pool exists that matches the poolname */
  for (auto &h : _pools) {
    if (h.first == poolname) {
      ph = h.second;
      break;
    }
  }

  if (ph == nullptr) {
    PWRN("Map_store: delete_pool (%s) pool not found", poolname.c_str());
    return E_POOL_NOT_FOUND;
  }

  for (auto &s : _pool_sessions) {
    if (s->pool->_name == poolname) {
      PWRN(
           "Map_store: delete_pool (%s) pool delete failed because pool still "
           "open (%p)",
           poolname.c_str(), s);
      return E_ALREADY_OPEN;
    }
  }

  /* delete pool too */
  if (_pools.find(poolname) == _pools.end())
    throw Logic_exception("unable to delete pool session");

  _pools.erase(poolname);
  delete ph;
  return S_OK;
}

status_t Map_store::put(IKVStore::pool_t pid, const std::string &key,
                        const void *value, size_t value_len,
                        unsigned int flags) {
  auto session = get_session(pid);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  return session->pool->put(key, value, value_len, flags);
}

status_t Map_store::get(const pool_t pid, const std::string &key,
                        void *&out_value, size_t &out_value_len) {
  auto session = get_session(pid);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  return session->pool->get(key, out_value, out_value_len);
}

status_t Map_store::get_direct(const pool_t pid, const std::string &key,
                               void *out_value, size_t &out_value_len,
                               Component::IKVStore::memory_handle_t /*handle*/) {
  auto session = get_session(pid);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  return session->pool->get_direct(key, out_value, out_value_len);
}

status_t Map_store::put_direct(const pool_t pid, const std::string &key,
                               const void *value, const size_t value_len,
                               memory_handle_t /*memory_handle*/,
                               unsigned int flags) {
  return Map_store::put(pid, key, value, value_len, flags);
}

status_t Map_store::resize_value(const pool_t pool,
                                 const std::string &key,
                                 const size_t new_size,
                                 const size_t alignment) {
  auto session = get_session(pool);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  return session->pool->resize_value(key, new_size, alignment);
}

status_t Map_store::get_attribute(const pool_t pool,
                                  const IKVStore::Attribute attr,
                                  std::vector<uint64_t> &out_attr,
                                  const std::string *key) {
  auto session = get_session(pool);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  return session->pool->get_attribute(pool, attr, out_attr, key);
}

status_t Map_store::lock(const pool_t pid, const std::string &key,
                         lock_type_t type, void *&out_value,
                         size_t &out_value_len, IKVStore::key_t &out_key) {
  auto session = get_session(pid);
  if (!session) {
    out_key = IKVStore::KEY_NONE;
    return E_FAIL;
  }

  if (_debug_level) PLOG("Map_store: lock(%s)", key.c_str());

  try {
    return session->pool->lock(key, type, out_value, out_value_len, out_key);
  }
  catch (...) {
    return E_FAIL;
  }
}

status_t Map_store::unlock(const pool_t pid, key_t key_handle) {
  auto session = get_session(pid);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  session->pool->unlock(key_handle);
  return S_OK;
}

status_t Map_store::erase(const pool_t pid, const std::string &key) {
  auto session = get_session(pid);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  return session->pool->erase(key);
}

size_t Map_store::count(const pool_t pid) {
  auto session = get_session(pid);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  return session->pool->count();
}

status_t Map_store::free_memory(void *p) {
  return free_memory(p);
}

void Map_store::debug(const pool_t, unsigned, uint64_t) {}

int Map_store::get_capability(Capability cap) const {
  switch (cap) {
  case Capability::POOL_DELETE_CHECK:
    return 1;
  case Capability::POOL_THREAD_SAFE:
    return 1;
  case Capability::RWLOCK_PER_POOL:
    return 1;
#ifdef ENABLE_TIMESTAMPS
  case Capability::WRITE_TIMESTAMPS:
    return 1;
#endif    
  default:
    return -1;
  }
}

status_t Map_store::map(const IKVStore::pool_t pool,
                        std::function<int(const void * key,
                                          const size_t key_len,
                                          const void * value,
                                          const size_t value_len)>
                        function) {
  auto session = get_session(pool);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  return session->pool->map(function);
}

status_t Map_store::map(const pool_t pool,
                        std::function<int(const void* key,
                                          const size_t key_len,
                                          const void* value,
                                          const size_t value_len,
                                          const tsc_time_t timestamp)> function,
                        const epoch_time_t t_begin,
                        const epoch_time_t t_end)
{
  auto session = get_session(pool);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  return session->pool->map(function, t_begin, t_end);
}


status_t Map_store::map_keys(const IKVStore::pool_t pool,
                             std::function<int(const std::string &key)> function) {
  auto session = get_session(pool);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  return session->pool->map_keys(function);
}

status_t Map_store::get_pool_regions(const pool_t pool,
                                     std::vector<::iovec> &out_regions) {
  auto session = get_session(pool);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;
  return session->pool->get_pool_regions(out_regions);
}
status_t Map_store::grow_pool(const pool_t pool, const size_t increment_size,
                              size_t &reconfigured_size) {
  auto session = get_session(pool);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;
  return session->pool->grow_pool(increment_size, reconfigured_size);
}

status_t Map_store::free_pool_memory(const pool_t pool, const void *addr,
                                     const size_t size) {
  auto session = get_session(pool);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;

  return session->pool->free_pool_memory(addr, size);
}

status_t Map_store::allocate_pool_memory(const pool_t pool,
                                         const size_t size,
                                         const size_t alignment,
                                         void *&out_addr) {
  auto session = get_session(pool);
  if (!session) return IKVStore::E_POOL_NOT_FOUND;
  return session->pool->allocate_pool_memory(size, alignment > size ? size : alignment, out_addr);
}

IKVStore::pool_iterator_t Map_store::open_pool_iterator(const pool_t pool)
{
  auto session = get_session(pool);
  if (!session) return nullptr;
  auto i = session->pool->open_pool_iterator();
  return i;
}

status_t Map_store::deref_pool_iterator(const pool_t pool,
                                        IKVStore::pool_iterator_t iter,
                                        const epoch_time_t t_begin,
                                        const epoch_time_t t_end,
                                        pool_reference_t& ref,
                                        bool& time_match,
                                        bool increment)
{
  auto session = get_session(pool);
  if (!session) return E_INVAL;
  return session->pool->deref_pool_iterator(iter,
                                            t_begin,
                                            t_end,
                                            ref,
                                            time_match,
                                            increment);
}

status_t Map_store::close_pool_iterator(const pool_t pool,
                                        IKVStore::pool_iterator_t iter)
{
  auto session = get_session(pool);
  if (!session) return E_INVAL;
  return session->pool->close_pool_iterator(iter);  
}


/**
 * Factory entry point
 *
 */
extern "C" void *factory_createInstance(Component::uuid_t &component_id) {
  if (component_id == Map_store_factory::component_id()) {
    return static_cast<void *>(new Map_store_factory());
  } else
    return NULL;
}
