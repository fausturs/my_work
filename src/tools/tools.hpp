#ifndef tools_hpp
#define tools_hpp

#include <iostream>
#include <iomanip>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <vector>
#include <random>
#include <array>
#include <numeric>
#include <algorithm>
#include <functional>
#include <cxxabi.h>

namespace wjy {
    
    std::vector< std::string > split(const std::string& st, const std::unordered_set< char >& spliter = {','});

    template <size_t w, class T, class... Types>
    void format_print(std::ostream& myout, T first)
    {
        myout<<std::left<<std::setw(w)<<first<<std::endl;
    }
    template <size_t w, class T, class... Types>
    void format_print(std::ostream& myout, T first, Types... args)
    {
        myout<<std::left<<std::setw(w)<<first;
        format_print<w>(myout, args...);
    }
    // vector sum
    template <class Input_it1, class Input_it2, class T>
    void vector_sum(Input_it1 first1, Input_it1 last1, Input_it2 first2, T ratio = 1)
    {
        for (;first1!=last1;first1++,first2++)
            *first1 += (*first2)*ratio;
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
    
    //  tensor_first    : the first iterator of tensor
    //  d               : the tensor is dth-order
    //  dims_first      : each order's dim of tensor
    //  k               : multply the vector from which order, k \in [0, d-1]
    //  vector_first    : the first iterator of vector, the length of it is *(dims_first + k);
    //  out_first       : output iterator
    template<class T, class Input_it1, class Input_it2, class Input_it3, class Output_it>
    void tensor_multiply_vector(Input_it1 tensor_first, size_t d, Input_it2 dims_first, size_t k, Input_it3 vector_first, Output_it out_first)
    {
        // B_{pre, suf} = \sum_{i_k} A_{pre, i_k, suf} * v_{i_k};
        size_t pre_size = 1, suf_size = 1;
        for (size_t i=0; i<k ;i++,dims_first++) pre_size*=(*dims_first);
        size_t dim_k = *(dims_first++);
        for (size_t i=k+1; i<d ;i++,dims_first++) suf_size *= (*dims_first);
        std::vector< T > temp(dim_k);
        for (size_t pre=0; pre<pre_size; pre++)
        {
            for (size_t suf=0; suf<suf_size; suf++)
            {
                size_t index = pre*(suf_size*dim_k) + suf;
                for (size_t i=0; i<dim_k; i++,index+=suf_size) temp[i] = *(tensor_first+index);
                *(out_first++) = std::inner_product(temp.begin(), temp.end(), vector_first, static_cast<T>(0) );
            }
        }
    }
    
    template <typename T, size_t k>
    std::array<T, k> a_fill_array(T x)
    {
        std::array<T, k> a;
        for (auto & a_i : a) a_i = x;
        return a;
    }
    
    
    //  use like get_type_name< some_type >()
    //  or get_type_name< decltype(some_value) >()
    template <class T> class Show_type{};
    template <typename T>
    std::string get_type_name()
    {
        int status = -1;
        auto name = typeid(Show_type< T >).name();
        char* clearName = abi::__cxa_demangle(name, NULL, NULL, &status);
        const char* const demangledName = (status==0) ? clearName : name;
        std::string ret_val(demangledName);
        free(clearName);
        ret_val = ret_val.substr(15, ret_val.size()-16);
        return ret_val;
    }

    //  knn
    //  space complexity O(n), time complexity O(n^2logn)
    template <typename T>
    std::vector<size_t> knn(const std::vector<T>& data, size_t k, size_t category_num, std::function< double(const T&, const T&) > distance, size_t iterate_num=200, int random_seed=1)
    {
        size_t n = data.size();
        std::vector<size_t> category(n, 0), new_category(n, 0);
        std::set< std::pair< double, size_t>, std::greater<std::pair< double, size_t>> > all_distance;
        //  initialize
        size_t rd = (random_seed==-1)?std::random_device{}():random_seed;
        std::mt19937 mt(rd);
        std::uniform_int_distribution<size_t> uid(0, category_num-1);
        for (auto & c : category) c = uid(mt);
        //  iterate
        double dis;
        for (size_t iter=0; iter<iterate_num; iter++)
        {
            for (size_t i=0; i<n; i++)
            {
                all_distance.clear();
                for (size_t j=0; j<n; j++)
                {
                    if (i!=j) dis = distance(data[i], data[j]);
                    all_distance.insert( std::make_pair(dis, j) );
                }
                //  k nearest neighbor vote
                std::vector<size_t> vote(category_num, 0);
                auto it = all_distance.begin();
                for (size_t j=0; it!=all_distance.end() && j<k; it++,j++)
                    vote[ category[it->second] ]++;
                new_category[i] = std::distance(vote.begin(), std::max_element(vote.begin(), vote.end()));
            }
            std::swap(category, new_category);
        }
        return category;
    }
}







#endif
