/*
   Copyright [2017-2021] [IBM Corporation]
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


#ifndef _MCAS_HSTORE_PERMISSION_ERROR_H
#define _MCAS_HSTORE_PERMISSION_ERROR_H

#include "access.h"
#include <stdexcept>

namespace impl
{
	struct permission_error
		: public std::runtime_error
	{
		explicit permission_error(std::size_t ix, unsigned have, unsigned need);
	};
}

#endif