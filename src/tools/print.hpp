//
//  print.hpp
//  cpp_test
//
//  Created by wjy on 2018/9/17.
//  Copyright © 2018年 wjy. All rights reserved.
//

#ifndef print_hpp
#define print_hpp

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <set>
#include <list>
#include <map>
#include <initializer_list>
#include <complex>
#include <array>
#include <valarray>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <type_traits>


namespace wjy {
    
    
    template < typename T >
    void my_print(std::ostream&, const T&);
    template < typename T >
    void my_print(std::ostream&, const std::complex<T>&);
    template < typename T >
    void my_print(std::ostream&, const std::vector<T>&);
    template < typename T >
    void my_print(std::ostream&, const std::list<T>&);
    template < typename T , std::size_t N>
    void my_print(std::ostream&, const std::array<T, N>&);
    template < typename T >
    void my_print(std::ostream&, const std::initializer_list<T>&);
    template < typename T >
    void my_print(std::ostream&, const std::set<T>&);
    template < typename T >
    void my_print(std::ostream&, const std::unordered_set<T>&);
    template < typename T1, typename T2 >
    void my_print(std::ostream&, const std::pair<T1, T2>&);
    template < typename K, typename V >
    void my_print(std::ostream&, const std::map<K, V>&);
    template < typename K, typename V >
    void my_print(std::ostream&, const std::unordered_map<K, V>&);
    template < typename... T >
    void my_print(std::ostream&, const std::tuple<T...>&);
    template < class T >
    void print_list(std::ostream&, const T&, char, char, const std::string&);
    template < typename K, typename V >
    void print_map(std::ostream&, const std::map<K, V>&, const std::string&);
    template < typename... T, std::size_t... IS >
    void print_tuple(std::ostream&, const std::tuple<T...>&, const std::string&, const std::index_sequence<IS...>&);
    
    template< typename T = std::string>
    void print(std::ostream& os, const T& a = "")
    {
        my_print(os, a);
    }
    
    template< typename Head, typename... T >
    void print(std::ostream& os, const Head& a, const T&... t)
    {
        my_print(os, a);
        print(os, t...);
    }
    
    template< typename... T >
    void print(const T&... t)
    {
        print(std::cout, t...);
    }
    
    template< typename... T >
    void println(const T&... t)
    {
        print(std::cout, t...);
        std::cout<<std::endl;
    }
    
    template< typename... T >
    void println(std::ostream& os, const T&... t)
    {
        print(os, t...);
        os<<std::endl;
    }
    
    
    
    template < typename T >
    void my_print(std::ostream& os, const T& a){
        os<<a;
    }
    
    template < typename T >
    void my_print(std::ostream& os, const std::complex<T>& c){
        os<<c.real()<<"+"<<c.imag()<<"i";
    }
    
    
    template < typename T >
    void my_print(std::ostream& os, const std::vector<T>& v){
        print_list(os, v, '[', ']', " ");
    }
    
    template < typename T >
    void my_print(std::ostream& os, const std::list<T>& l){
        print_list(os, l, '[', ']', " ");
    }
    
    template < typename T , std::size_t N>
    void my_print(std::ostream& os, const std::array<T, N>& a){
        print_list(os, a, '[', ']', " ");
    }
    
    template < typename T >
    void my_print(std::ostream& os, const std::initializer_list<T>& il){
        print_list(os, il, '[', ']', " ");
    }
    
    template < typename T >
    void my_print(std::ostream& os, const std::set<T>& s){
        print_list(os, s, '{', '}', " ");
    }
    
    template < typename T >
    void my_print(std::ostream& os, const std::unordered_set<T>& us){
        print_list(os, us, '{', '}', " ");
    }
    
    template < typename T1, typename T2 >
    void my_print(std::ostream& os, const std::pair<T1, T2>& p){
        os<<'<';
        my_print(os, p.first);
        os<<", ";
        my_print(os, p.second);
        os<<'>';
    }
    
    template < typename K, typename V >
    void my_print(std::ostream& os, const std::map<K, V>& m){
        print_map(os, m, " ");
    }
    
    template < typename K, typename V >
    void my_print(std::ostream& os, const std::unordered_map<K, V>& um){
        print_map(os, um, " ");
    }
    
    template < typename... T >
    void my_print(std::ostream& os, const std::tuple<T...>& t){
        os<<'(';
        print_tuple(os, t, " ", std::index_sequence_for<T...>{});
        os<<')';
    }
    
    
    template < class T>
    void print_list(std::ostream& os, const T& a, char begin, char end, const std::string& s)
    {
        os<<begin;
        std::string spliter = "";
        for(const auto & ai : a)
        {
            os<<spliter;
            spliter = s;
            my_print(os, ai);
        }
        os<<end;
    }
    
    template < typename... T, std::size_t... IS >
    void print_tuple(std::ostream& os, const std::tuple<T...>& t, const std::string& s, const std::index_sequence<IS...>&)
    {
        (void)std::initializer_list<int>{ ( os<<(IS==0?"":s), my_print(os, std::get<IS>(t)),  0 )...};
    }
    
    template < typename K, typename V >
    void print_map(std::ostream& os, const std::map<K, V>& m, const std::string& s)
    {
        os<<'{';
        std::string spliter = "";
        for (const auto & p : m)
        {
            os<<spliter;
            spliter = s;
            os<<'{';
            my_print(os, p.first);
            os<<": ";
            my_print(os, p.second);
            os<<'}';
        }
        os<<'}';
    }

}

#endif /* print_hpp */
