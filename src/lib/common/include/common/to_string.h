/*
   Copyright [2021] [IBM Corporation]
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

#ifndef _MCAS_COMMON_TO_STRING_
#define _MCAS_COMMON_TO_STRING_

#include <sstream>

/* Note:
 *   Requires C++17, which may in turn require Boost 1.66.0 to fix a bug
 *   in icl/type_traits/type_to_string.hpp partial specialization.
 */
namespace common
{
  /* Convert stream arguments to a string */
  template <typename... Args>
    std::string to_string(Args&&... args)
    {
      std::ostringstream s;
      (s << ... << args);
      return s.str();
    }
}
#endif