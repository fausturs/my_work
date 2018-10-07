#ifndef my_model_4_hpp
#define my_model_4_hpp

#include <iostream>
#include <map>

#include "pairwise_interaction_tensor_factorization.hpp"

namespace wjy{
    // 将skill做了模糊聚类，再加上时间纬度，再加上对公司做的聚类。
    // 通过控制超参可以实现仅加其中几种
    // 

    template<typename T, size_t kth_order>
    class my_model_4 : public pairwise_interaction_tensor_factorization<T, kth_order>{
        using base_class = pairwise_interaction_tensor_factorization<T, kth_order>;
        using base_class::tensor_A;
        using base_class::matrixes_size;
        using base_class::parameters_num;
        using base_class::batchs;

        virtual std::vector<T> gradient(const std::vector<T>&) const override;
        virtual std::vector<T> mini_batch_stochastic_gradient(const std::vector<T>&, size_t) const override;
        virtual T loss(const std::vector<T>&) const override;

        // 公司聚类的参数与函数
        T lambda_1;
        size_t company_category_num;
        std::vector<size_t> company_category_map;
        void init_company_category_map(std::vector<size_t>&&);
        void gradient_1(const std::vector<T>& para, std::vector<T>& g) const;
        void mini_batch_stochastic_gradient_1(const std::vector<T>&para, size_t batch_i, std::vector<T>& g) const;
        void loss_1(const std::vector<T>& para, T& l) const;

        //  时间因素的参数与函数
        T lambda_2;
        std::vector< std::vector<T> > old_parameters;
        std::vector<T> gradient_of_old_parameters(const std::vector<T>&) const;
        void gradient_2(const std::vector<T>& para, std::vector<T>& g) const;
        void mini_batch_stochastic_gradient_2(const std::vector<T>&para, size_t batch_i, std::vector<T>& g) const;
        void loss_2(const std::vector<T>& para, T& l) const;

        //  技能聚类的参数与函数
        T lambda_3;
        size_t skill_category_num;
        std::map< size_t, std::vector< std::pair<size_t, T> > > skill_category_map;
        void init_skill_category_map(std::map< size_t, std::vector< std::pair<size_t, T> > >&&);
        void gradient_3(const std::vector<T>& para, std::vector<T>& g) const;
        void mini_batch_stochastic_gradient_3(const std::vector<T>&para, size_t batch_i, std::vector<T>& g) const;
        void loss_3(const std::vector<T>& para, T& l) const;

    public:
        my_model_4() = default;

        my_model_4(
            // PITF parameter
            sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, size_t mini_batch_num,
            // 公司聚类的参数
            T lambda_1, std::vector<size_t> company_category_map,
            // 时间因素的参数
            T lambda_2, std::vector< std::vector<T> > old_parameters,
            // 技能聚类的参数
            T lambda_3, std::map< size_t, std::vector< std::pair<size_t, T> > > skill_category_map
        );

        void test() override;
        virtual  ~my_model_4() override = default;

    }; // class my_model_4
    
    // 1
    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::init_company_category_map(std::vector<size_t>&& company_category_map)
    {
        if (std::abs(lambda_1)<0.00000001) return;
        this->company_category_map = std::move(company_category_map);
        //  discretization
        auto v = company_category_map;
        std::sort(v.begin(), v.end());
        auto it_end = std::unique(v.begin(), v.end());
        this->company_category_num = std::distance(v.begin(), it_end);
        for (auto & x : this->company_category_map)
        {
            auto it = std::lower_bound(v.begin(), it_end, x);
            x = std::distance(v.begin(), it);
        }
    }
    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::gradient_1(const std::vector<T>& para, std::vector<T>& g) const
    {
        if (std::abs(lambda_1)<0.00000001) return;
        size_t vector_length = matrixes_size.front().second;
        size_t category_parameters_offset = 0;
        for (auto & m : matrixes_size) category_parameters_offset += m.first*m.second*(kth_order-1);
        std::vector<T> temp(vector_length, 0);
        // 因为company在tensor_A的第一维 所以这里i=0
        size_t i = 0;
        for (size_t j=1; j<kth_order; j++)
        {
            size_t offset_ij = matrixes_size[i].first*vector_length*(j-1);
            for (size_t k=0; k<matrixes_size[i].first; k++)
            {
                //
                size_t offset1 = offset_ij + k*vector_length;
                size_t offset2 = category_parameters_offset + company_category_num*vector_length*(j-1) + vector_length*company_category_map[k];
                auto it1 = para.begin() + offset1;
                auto it2 = para.begin() + offset2;
                vector_sum(it1, static_cast<T>(2), it2, static_cast<T>(-2), vector_length, temp.begin());
                //  update gradient
                auto g_it1 = g.begin() + offset1;
                auto g_it2 = g.begin() + offset2;
                vector_sum(g_it1, g_it1+vector_length, temp.begin(), lambda_1);
                vector_sum(g_it2, g_it2+vector_length, temp.begin(), -lambda_1);
            }
        }
    }
    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::mini_batch_stochastic_gradient_1(const std::vector<T>&para, size_t batch_i, std::vector<T>& g) const
    {
        if (std::abs(lambda_1)<0.00000001) return;
        size_t n = tensor_A.size();
        //  这一行的时间按复杂度是O(m)
        size_t m = std::distance(batchs[batch_i], batchs[batch_i+1]);
        size_t vector_length = matrixes_size.front().second;
        size_t category_parameters_offset = 0;
        for (auto & m : matrixes_size) category_parameters_offset += m.first*m.second*(kth_order-1);
        std::vector<T> temp(vector_length, 0);
        size_t i = 0;
        for (size_t j=1; j<kth_order; j++)
        {
            size_t offset_ij = matrixes_size[i].first*vector_length*(j-1);
            for (size_t k=0; k<matrixes_size[i].first; k++)
            {
                //
                size_t offset1 = offset_ij + k*vector_length;
                size_t offset2 = category_parameters_offset + company_category_num*vector_length*(j-1) + vector_length*company_category_map[k];
                auto it1 = para.begin() + offset1;
                auto it2 = para.begin() + offset2;
                vector_sum(it1, static_cast<T>(2), it2, static_cast<T>(-2), vector_length, temp.begin());
                //  update gradient
                auto g_it1 = g.begin() + offset1;
                auto g_it2 = g.begin() + offset2;
                //  唯一和上面那个函数gradient_1不一样的地方在这里后面的系数
                vector_sum(g_it1, g_it1+vector_length, temp.begin(), lambda_1*m/n);
                vector_sum(g_it2, g_it2+vector_length, temp.begin(), -lambda_1*m/n);
            }
        }
    }
    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::loss_1(const std::vector<T>& para, T& l) const
    {
        if (std::abs(lambda_1)<0.00000001) return;
        T l1 = 0;
        size_t vector_length = matrixes_size.front().second;
        size_t category_parameters_offset = 0;
        for (auto & m : matrixes_size) category_parameters_offset += m.first*m.second*(kth_order-1);
        size_t i = 0;
        for (size_t j=1; j<kth_order; j++)
        {
            size_t offset_ij = matrixes_size[i].first*vector_length*(j-1);
            for (size_t k=0; k<matrixes_size[i].first; k++)
            {
                auto it1 = para.begin() + offset_ij + k*vector_length;
                auto it2 = para.begin() + category_parameters_offset + company_category_num*vector_length*(j-1) + vector_length*company_category_map[k];
                l1 = std::inner_product(it1, it1+vector_length, it2, l1);
            }
        }
        l += l1;
    }

    // 2
    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::gradient_2(const std::vector<T>& para, std::vector<T>& g) const
    {
        if (std::abs(lambda_2)<0.00000001) return;
        std::vector<T> temp_g(parameters_num, 0), temp(parameters_num, 0);
        size_t n = old_parameters.size();  
        for (size_t i=0; i<n; i++)
        {
            T ratio = lambda_2/(n-i);
            vector_sum(para.begin(), static_cast<T>(2), old_parameters[i].begin(), static_cast<T>(-2), parameters_num, temp.begin());
            vector_sum(temp_g.begin(), temp_g.end(), temp.begin(), ratio);
        }
        vector_sum(g.begin(), g.end(), temp.begin(), static_cast<T>(1));
    }
    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::mini_batch_stochastic_gradient_2(const std::vector<T>&para, size_t batch_i, std::vector<T>& g) const
    {
        if (std::abs(lambda_2)<0.00000001) return;
        std::vector<T> temp_g(parameters_num, 0), temp(parameters_num, 0);
        size_t n = old_parameters.size();  
        for (size_t i=0; i<n; i++)
        {
            T ratio = lambda_2/(n-i);
            vector_sum(para.begin(), static_cast<T>(2), old_parameters[i].begin(), static_cast<T>(-2), parameters_num, temp.begin());
            vector_sum(temp_g.begin(), temp_g.end(), temp.begin(), ratio);
        }
        //  这一行的时间按复杂度是O(m)
        size_t m = std::distance(batchs[batch_i], batchs[batch_i+1]);
        vector_sum(g.begin(), g.end(), temp.begin(), static_cast<T>(m)/tensor_A.size());
    }
    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::loss_2(const std::vector<T>& para, T& l) const
    {
        if (std::abs(lambda_2)<0.00000001) return;
        T l2 = 0;
        std::vector<T> temp(parameters_num);
        size_t n = old_parameters.size();
        for (size_t i=0; i<n; i++)
        {
            //
            vector_sum(para.begin(), static_cast<T>(2), old_parameters[i].begin(), static_cast<T>(-2), parameters_num, temp.begin());
            T l_i = lambda_2 * std::inner_product(temp.begin(), temp.end(), temp.begin(), static_cast<T>(0));
            l_i /= n-i;
            l2 += l_i;
        }
        l += l2;
    }
    
    //3
    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::init_skill_category_map(std::map< size_t, std::vector< std::pair<size_t, T> > >&& skill_category_map)
    {
        if (std::abs(lambda_3)<0.00000001) return;
        this->skill_category_map = skill_category_map;
        //  discretization
        std::vector<size_t> v; 
        for (auto & m_v : this->skill_category_map)
            for( auto & p : m_v.second )
                v.push_back(p.first);
        std::sort(v.begin(), v.end());
        auto it_end = std::unique(v.begin(), v.end());
        this->skill_category_num = std::distance(v.begin(), it_end);
        for (auto & m_v : this->skill_category_map)
            for( auto & p : m_v.second )
            {
                auto it = std::lower_bound(v.begin(), it_end, p.first);
                p.first = std::distance(v.begin(), it);
            }
    }

    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::gradient_3(const std::vector<T>& para, std::vector<T>& g) const
    {
        if (std::abs(lambda_3)<0.00000001) return;
        size_t vector_length = matrixes_size.front().second;
        std::vector<T> temp(vector_length, 0);
        size_t category_parameters_offset = 0, offset_i = 0;
        for (auto & m : matrixes_size) category_parameters_offset += m.first*m.second*(kth_order-1);
        category_parameters_offset += company_category_num * vector_length * (kth_order-1);
        // magic number 2
        for (size_t i=0; i<2; i++) offset_i += matrixes_size[i].first * matrixes_size[i].second * (kth_order-1);
        size_t i = 2;
        for (size_t j=1; j<kth_order; j++)
        {
            size_t offset_ij = offset_i + matrixes_size[i].first*vector_length*(j-1);
            for (size_t k=0; k<matrixes_size[i].first; k++)
            {
                if (skill_category_map.count(k)==0) continue;
                //
                size_t offset1 = offset_ij + k*vector_length;
                auto it1 = para.begin() + offset1;
                auto g_it1 = g.begin() + offset1;

                for (auto & p : skill_category_map.at(k))
                {
                    auto category_k = p.first;
                    auto rate       = p.second;
                    size_t offset2  = category_parameters_offset + skill_category_num*vector_length*(j-1) + vector_length*category_k;
                    auto it2 = para.begin() + offset2;
                    vector_sum(it1, static_cast<T>(2 * rate), it2, static_cast<T>(-2*rate), vector_length, temp.begin());
                    //  update gradient
                    auto g_it2 = g.begin() + offset2;
                    vector_sum(g_it1, g_it1+vector_length, temp.begin(), lambda_3);
                    vector_sum(g_it2, g_it2+vector_length, temp.begin(), -lambda_3);
                }
            }
        }
    }
    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::mini_batch_stochastic_gradient_3(const std::vector<T>&para, size_t batch_i, std::vector<T>& g) const
    {
        if (std::abs(lambda_3)<0.00000001) return;
         size_t n = tensor_A.size();
        //  这一行的时间按复杂度是O(m)
        size_t m = std::distance(batchs[batch_i], batchs[batch_i+1]);
        size_t vector_length = matrixes_size.front().second;
        std::vector<T> temp(vector_length, 0);
        size_t category_parameters_offset = 0, offset_i = 0;
        for (auto & m : matrixes_size) category_parameters_offset += m.first*m.second*(kth_order-1);
        category_parameters_offset += company_category_num * vector_length * (kth_order-1);
        // magic number 2
        for (size_t i=0; i<2; i++) offset_i += matrixes_size[i].first * matrixes_size[i].second * (kth_order-1);
        size_t i = 2;
        for (size_t j=1; j<kth_order; j++)
        {
            size_t offset_ij = offset_i + matrixes_size[i].first*vector_length*(j-1);
            for (size_t k=0; k<matrixes_size[i].first; k++)
            {
                if (skill_category_map.count(k)==0) continue;
                //
                size_t offset1 = offset_ij + k*vector_length;
                auto it1 = para.begin() + offset1;
                auto g_it1 = g.begin() + offset1;

                for (auto & p : skill_category_map.at(k))
                {
                    auto category_k = p.first;
                    auto rate       = p.second;
                    size_t offset2  = category_parameters_offset + skill_category_num*vector_length*(j-1) + vector_length*category_k;
                    auto it2 = para.begin() + offset2;
                    vector_sum(it1, static_cast<T>(2 * rate), it2, static_cast<T>(-2*rate), vector_length, temp.begin());
                    //  update gradient
                    auto g_it2 = g.begin() + offset2;
                    //  与gradient_3的区别在于系数
                    vector_sum(g_it1, g_it1+vector_length, temp.begin(), lambda_3*m/n);
                    vector_sum(g_it2, g_it2+vector_length, temp.begin(), -lambda_3*m/n);
                }
            }
        }
    }
    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::loss_3(const std::vector<T>& para, T& l) const
    {
        if (std::abs(lambda_3)<0.00000001) return;
        T l_3=0;
        size_t vector_length = matrixes_size.front().second;
        size_t category_parameters_offset = 0, offset_i = 0;
        for (auto & m : matrixes_size) category_parameters_offset += m.first*m.second*(kth_order-1);
        category_parameters_offset += company_category_num * vector_length * (kth_order-1);
        // magic number 2
        for (size_t i=0; i<2; i++) offset_i += matrixes_size[i].first * matrixes_size[i].second * (kth_order-1);
        size_t i = 2;
        for (size_t j=1; j<kth_order; j++)
        {
            size_t offset_ij = offset_i + matrixes_size[i].first*vector_length*(j-1);
            for (size_t k=0; k<matrixes_size[i].first; k++)
            {
                auto it1 = para.begin() + offset_ij + k*vector_length;
                if (skill_category_map.count(k)==0) continue;
                for (auto & p : skill_category_map.at(k))
                {
                    auto category_k = p.first;
                    auto rate       = p.second;
                    auto it2        = para.begin() + category_parameters_offset + skill_category_num*vector_length*(j-1) + vector_length*category_k;
                    T temp_l = std::inner_product(it1, it1+vector_length, it2, static_cast<T>(0));
                    l_3 += temp_l * rate;
                }
            }
        }
    }

    template<typename T, size_t kth_order>
    std::vector<T> my_model_4<T, kth_order>::gradient(const std::vector<T>& para) const
    {
        auto g = base_class::gradient(para);
        // 1
        gradient_1(para, g);
        // 2
        gradient_2(para, g);
        // 3
        gradient_3(para, g);

        return g;
    }
    template<typename T, size_t kth_order>
    std::vector<T> my_model_4<T, kth_order>::mini_batch_stochastic_gradient(const std::vector<T>& para, size_t batch_i ) const
    {
        auto g = base_class::mini_batch_stochastic_gradient(para, batch_i);
        // 1
        mini_batch_stochastic_gradient_1(para, batch_i, g);
        // 2
        mini_batch_stochastic_gradient_2(para, batch_i, g);
        // 3
        mini_batch_stochastic_gradient_3(para, batch_i, g); 
        return g;
    }
    template<typename T, size_t kth_order>
    T my_model_4<T, kth_order>::loss(const std::vector<T>& para) const
    {
        auto l = base_class::loss(para);
        // 1
        loss_1(para, l);
        // 2
        loss_2(para, l);
        // 3
        loss_3(para, l);

        return l;
    }



    template<typename T, size_t kth_order>
    my_model_4<T, kth_order>::my_model_4(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, size_t mini_batch_num, T lambda_1, std::vector<size_t> company_category_map, T lambda_2, std::vector< std::vector<T> > old_parameters,T lambda_3, std::map< size_t, std::vector< std::pair<size_t, T> > > skill_category_map)
    :base_class(std::move(tensor_A), parameters_rank, lambda, mini_batch_num)
    ,lambda_1(lambda_1), company_category_num(0)
    ,lambda_2(lambda_2)
    ,lambda_3(lambda_3), skill_category_num(0)
    {
        // 1
        init_company_category_map(std::move(company_category_map));
        parameters_num += parameters_rank*company_category_num*(kth_order-1);

        // 2
        this->old_parameters = std::move(old_parameters);

        // 3
        init_skill_category_map(std::move(skill_category_map));
        parameters_num += parameters_rank*skill_category_num*(kth_order-1);
    }

    template<typename T, size_t kth_order>
    void my_model_4<T, kth_order>::test()
    {
        std::cout<<"This is class my_model_4."<<std::endl;
    }
} // namespace wjy


#endif // file my_model_4_hpp