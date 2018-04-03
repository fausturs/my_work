#ifndef FUNC_HPP
#define FUNC_HPP

#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <numeric>
#include <algorithm>
#include "sparse_tensor.hpp"

extern std::unordered_map< std::string, size_t >   skill_to_id, company_to_id, position_to_id;
extern std::vector< std::string >                  id_to_skill, id_to_company, id_to_position;


std::vector< std::string > split(const std::string& st, const std::unordered_set< char >& spliter = {','});

wjy::sparse_tensor<double, 3> create_tensor(const std::string path = "../data/position_cut.txt");

// vector sum
template <class Input_it1, class Input_it2, class T>
void add_to(Input_it1 first1, Input_it1 last1, Input_it2 first2, T rate = 1)
{
    for (;first1!=last1;first1++,first2++)
        *first1 += (*first2)*rate;
}

// split the first input vector to k group, each group have n element
// calculate each group's inner product with the second input vector
// output a k length vector
// this function can be used in some case of tensor multiply with vector
template <class Input_it1, class Input_it2, class Output_it1, class T>
void inner_product_n(Input_it1 first1, Input_it1 last1, Input_it2 first2, size_t n, Output_it1 d_first, T value)
{
    for (;first1!=last1;first1+=n)
        *(d_first++) = std::inner_product(first1, first1+n, first2, value);
}

template <class T>
void print_vector(T& v, std::ostream& myout = std::cout)
{
    for (auto & x : v) myout<<x<<" ";
    myout<<std::endl;
}
template <class Input_it>
void print_vector(Input_it first, Input_it last, std::ostream& myout = std::cout)
{
    for (;first!=last; first++) myout<<(*first)<<" ";
    myout<<std::endl;
}


template <class T, class... Types>
void format_print(std::ostream& myout, T first)
{
    myout<<std::left<<std::setw(16)<<first<<std::endl;
}
template <class T, class... Types>
void format_print(std::ostream& myout, T first, Types... args)
{
    myout<<std::left<<std::setw(16)<<first<<" ";
    format_print(myout, args...);
}

//std::vector< std::string > split(const std::string& st, const std::unordered_set< std::string >& spliter = {","});
//
//std::unordered_set< std::string > read_skill_list(const std::string& path = "../data/skill_list.txt");
//std::unordered_map< std::string, int > read_demand_level(const std::string& path = "../data/demand_level.txt");
//
//std::list< std::pair<std::string, int> > parse_jd_demand(const std::vector< std::string >& jd,const std::unordered_set< std::string >& skill_list, const std::unordered_map< std::string, int >& demand_level);
//
//


#endif
