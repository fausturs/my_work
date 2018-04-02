#ifndef FUNC_HPP
#define FUNC_HPP

#include <iostream>
#include <iomanip>
#include <string>
#include <numeric>
#include <algorithm>

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

std::wstring string_to_wstring(const std::string &);
std::string wstring_to_string(const std::wstring &);







#endif
