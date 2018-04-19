#ifndef FUNC_HPP
#define FUNC_HPP

#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <numeric>
#include <algorithm>


#include "Date.hpp"
#include "tools.hpp"

extern std::unordered_map< std::string, size_t >   skill_to_id, company_to_id, position_to_id;
extern std::vector< std::string >                  id_to_skill, id_to_company, id_to_position;

void read_all(wjy::Date data_date);
void create_tensor();

//for test
void test();


#endif
