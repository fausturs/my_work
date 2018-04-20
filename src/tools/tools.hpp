#ifndef tools_hpp
#define tools_hpp

#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
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
    
    
    
    template <class T> class Show_type{};
    //想法是把要显示的类型作为T参数传递进函数
    //为了避免typeid操作符吧const reference volatile三种修饰符去掉， 这里用了个空模板类把T包了一层
    //这样typeid就没办法去掉尖括号里面的修饰符
    //最后返回字符串的时候，记得把外面包的Show_type类型去掉就可以了
    //调用方式就是    get_type_name< some_type >()
    //或者是         get_type_name< decltype(some_value) >()
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
}

#endif
