#ifndef linear_algebra_functions_hpp
#define linear_algebra_functions_hpp

#include <iostream>

namespace wjy{
    
    //  vector sum
    //  x1 += x2*ratio;
    template <class Input_it1, class Input_it2, class T>
    void vector_sum(Input_it1 first1, Input_it1 last1, Input_it2 first2, T ratio = 1)
    {
        for (;first1!=last1;first1++,first2++)
            *first1 += (*first2)*ratio;
    }

    //  vector sum
    //  x3 = x1*alpha + x2*beta
    template <class Input_it1, class Input_it2, class Output_it, class T>
    void vector_sum(Input_it1 first1, T alpha, Input_it2 first2, T beta, size_t n, Output_it out_first)
    {
        for (size_t i=0; i<n; i++)
            *(out_first++) = *(first1++) * alpha + *(first2++) * beta;
    }

}

#endif