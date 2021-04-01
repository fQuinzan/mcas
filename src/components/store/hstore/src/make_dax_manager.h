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


#ifndef MCAS_HSTORE_MAKE_DAX_MANAGER_H
#define MCAS_HSTORE_MAKE_DAX_MANAGER_H

#include <nupm/dax_manager_abstract.h>

#include <common/string_view.h>
#include <common/logfwd.h>
#include <memory>

std::unique_ptr<nupm::dax_manager_abstract> make_dax_manager(
	const common::log_source &ls_
	, common::string_view dax_map
	, bool force_reset = false
);

#endif