//
//  linear_regression.hpp
//  cpp_test
//
//  Created by wjy on 2018/4/17.
//  Copyright © 2018年 wjy. All rights reserved.
//

#ifndef linear_regression_hpp
#define linear_regression_hpp

#include "predictor.hpp"

namespace wjy {
    //  \min_x \|Ax-b\|^2
    template< typename T >
    class linear_regression final : public predictor< T, std::vector<T> >{
        std::vector<T> A, b;
        void load_A_b(const std::string& file_path);
        using predictor<T, std::vector<T> >::parameters;

        virtual std::vector<T> gradient(const std::vector<T>&) const override;
        virtual T loss(const std::vector<T>&) const override;
        virtual T _predict(const std::vector<T>&, const std::vector<T>&) const override;
        
        std::vector<T> A_times_vector(const std::vector<T>&) const;
    public:
        linear_regression() = default;
        //  constructor for train
        linear_regression(const std::string& file_path, size_t mini_batch_num=1);
        linear_regression(const std::vector<T>& A, const std::vector<T>& b, size_t mini_batch_num=1);
        // constructor only for pridect
        explicit linear_regression(const std::vector<T>& v);

        //  movable
        linear_regression(linear_regression&&) = default;
        linear_regression& operator=(linear_regression&&) = default;

        
        virtual ~linear_regression() override = default;
    };

    template <typename T>
    void linear_regression<T>::load_A_b(const std::string &file_path)
    {
        std::ifstream myin(file_path);
        assert(myin);
        size_t rows, cols;
        myin>>rows>>cols;
        this->parameters_num = cols;
        A.resize(rows*cols);
        std::copy_n(std::istream_iterator<T>(myin), rows*cols, A.begin());
        b.resize(rows);
        std::copy_n(std::istream_iterator<T>(myin), rows, b.begin());
    }

    template <typename T>
    linear_regression<T>::linear_regression(const std::string& file_path, size_t mini_batch_num)
    :predictor< T, std::vector<T> >(mini_batch_num)
    {
        load_A_b(file_path);
    }

    template <typename T>
    linear_regression<T>::linear_regression(const std::vector<T>& A, const std::vector<T>& b, size_t mini_batch_num)
    :predictor< T, std::vector<T> >(mini_batch_num)
    {
        this->A = A;
        this->b = b;
        this->parameters_num = A.size()/b.size();
    }

//    template <typename T>
//    linear_regression<T>::linear_regression(const std::string& model_path)
//    {
//        std::ifstream myin(model_path);
//        this->load_parameters(myin);
//    }

    template <typename T>
    linear_regression<T>::linear_regression(const std::vector<T>& v)
    {
        this->parameters = v;
        this->parameters_num = this->parameters.size();
    }

    template< typename T >
    std::vector<T> linear_regression<T>::gradient(const std::vector<T>& para) const
    {
        //  gradient = 2A^\top (Ax-b)
        size_t rows = b.size(), cols = this->parameters_num;
        std::vector<T> g(cols, 0), temp(rows, 0);
        auto v = A_times_vector(para);                          //v = Ax
        wjy::vector_sum(v.begin(), v.end(), b.begin(), -1);     //v = Ax - b
        for (size_t i=0; i<cols; i++)
        {
            //  temp is the ith row of A^\top
            for (size_t j=0; j<rows; j++) temp[j] = A[i + j*cols];
            g[i] = 2 * std::inner_product(v.begin(), v.end(), temp.begin(), static_cast<T>(0));
        }
        return g;
    }

    template< typename T >
    std::vector<T> linear_regression<T>::A_times_vector(const std::vector<T>& para) const
    {
        size_t rows = b.size(), cols = this->parameters_num;
        std::vector<T> v(rows);
        auto it = A.cbegin();
        for (size_t i=0; i<rows; i++,it+=cols)
            v[i] = std::inner_product(it, it+cols, para.begin(), static_cast<T>(0));
        return v;
    }

    template< typename T >
    T linear_regression<T>::loss(const std::vector<T>& para) const
    {
        auto v = A_times_vector(para);                                          //v = Ax
        wjy::vector_sum(v.begin(), v.end(), b.begin(), static_cast<T>(-1));     //v = Ax - b
        return std::inner_product(v.begin(), v.end(), v.begin(), static_cast<T>(0));
    }

    template< typename T >
    T linear_regression<T>::_predict(const std::vector<T>& para, const std::vector<T>& a) const
    {
        T pred = 0;
        pred = std::inner_product(para.begin(), para.end(), a.begin(), pred);
        return pred;
    }
}

#endif /* linear_regression_hpp */
