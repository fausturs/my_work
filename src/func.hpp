#ifndef FUNC_HPP
#define FUNC_HPP

#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <numeric>
#include <algorithm>
#include <functional>


#include "Date.hpp"
#include "tools.hpp"
#include "predictor.hpp"
#include "sparse_tensor.hpp"

extern std::unordered_map< std::string, size_t >   skill_to_id, company_to_id, position_to_id;
extern std::vector< std::string >                  id_to_skill, id_to_company, id_to_position;

void read_all(wjy::Date data_date);
void create_tensor();


template <typename T, size_t kth_order>
std::vector< size_t > k_means_for_tensors(const std::vector< wjy::sparse_tensor<T, kth_order> >& tensors, size_t k, size_t iterate_num)
{
    //  init
    size_t n = tensors.size();
    std::vector< wjy::sparse_tensor<T, kth_order> > centers(k);
    std::vector< size_t > centers_size(k, 0), category(n);
    for (size_t i=0; i<n; i++) category[i] = i%k;   
    //  distance of two tensors
    std::vector<T> norm(n, 0);
    for (size_t i=0; i<n; i++)
    {
        auto & tensor = tensors[i];
        for (auto & entry : tensor) norm[i] += entry.second*entry.second;
        norm[i] = std::sqrt(norm[i]);
    }
    using V = const wjy::sparse_tensor<T, kth_order> &;
    auto distance = [](V x, V y, T norm_x, T norm_y){
        T dis = 0;
        //  assume x.size()<y.size();
        for (auto & entry : x)
            if (y.count(entry.first)!=0) dis += entry.second * y.at(entry.first);
        // return dis/(norm_x*norm_y);
        return std::sqrt(norm_x*norm_x + norm_y*norm_y - 2*dis);
    };
    // return category;
    //  iterate
    for (size_t iter=0; iter<iterate_num; iter++)
    {
        std::cout<<"iterate : "<<iter<<std::endl;
        //  calculate new centers
        for (auto & center : centers) center.clear();
        centers_size = std::vector<size_t>(k, 0);
        for (size_t i=0; i<n; i++)
        {
            auto & center = centers[ category[i] ];
            centers_size[ category[i] ]++;
            for (auto & entry : tensors[i]) center[ entry.first ] += entry.second;
        }
        std::vector<T> centers_norm(k, 0);
        for (size_t i=0; i<k ;i++)
        {
            for (auto & entry : centers[i])
            {
                entry.second /= static_cast<T>(centers_size[i]);
                centers_norm[i] += entry.second*entry.second;
            }
            centers_norm[i] = std::sqrt(centers_norm[i]);
        }
        //  update nearest center
        for (size_t i=0; i<n; i++)
        {
            T dis = -1;
            for (size_t j=0; j<k; j++)
            {
                T new_dis = distance(tensors[i], centers[j], norm[i], centers_norm[j]);
                if (dis == -1 || new_dis < dis) 
                {
                    category[i] = j;
                    dis = new_dis;
                }
            }
        }
    }
    return category;
}



//          evaluation functions
template <typename D, typename T, typename V>
T mean_square_error(const D& d, const wjy::predictor<T, V >& predictor)
{
    T MSE = 0;
    for (auto & entry : d)
    {
        auto & index = entry.first;
        auto & value = entry.second;
        T temp = (value - predictor( index ));
        MSE += temp*temp;
    }
    return MSE / d.size();
}

template <typename D, typename T, typename V>
T mean_absolute_percentage_error(const D& d, const wjy::predictor<T, V >& predictor)
{
    T MAPE = 0;
    for (auto & entry : d)
    {
        auto & index = entry.first;
        auto & value = entry.second;
        T temp = std::abs( (value - predictor( index ))/value );
        MAPE += temp;
    }
    return MAPE * 100 / d.size();
}

template <typename D, typename T, typename V>
T r_square(const D& d, const wjy::predictor<T, V>& predictor)
{
    T rs = 0, mean = 0, ss_tot = 0, ss_res = 0;
    for (auto & entry : d)
    {
        auto & index = entry.first;
        auto & value = entry.second;
        mean += value;
        T temp = (value - predictor( index ));
        ss_res += temp*temp;
    }
    mean /= d.size();
    for (auto & entry : d) ss_tot += (entry.second - mean)*(entry.second - mean);
    return 1-(ss_res/ss_tot);
}

template <typename D, typename T, typename V>
std::vector< std::pair<std::string, T> > all_evaluations(const D& d, const wjy::predictor<T, V >& predictor)
{
    std::vector< std::pair<std::string, T> > ans;
    ans.emplace_back("MSE", mean_square_error(d, predictor));
    ans.emplace_back("MAPE", mean_absolute_percentage_error(d, predictor));
    ans.emplace_back("R-Square", r_square(d, predictor));
    return ans;
}

template <typename D, typename T, typename V>
T mean_square_error_1(const std::vector<D>& d, const std::vector<wjy::predictor<T, V >*> predictor)
{
    T MSE = 0;
    double num = 0;
    for (int i=0; i<d.size(); i++)
    {
        num += d[i].size();
        for (auto & entry : d[i])
        {
            auto & index = entry.first;
            auto & value = entry.second;
            T temp = (value - predictor[i]->operator()( index ));
            MSE += temp*temp;
        }
    }
    return MSE / num;
}



template <typename D, typename T, typename V>
T mean_absolute_percentage_error_1(const std::vector<D>& d, const std::vector<wjy::predictor<T, V >*> predictor)
{
    T MAPE = 0;
    double num = 0;
    for (int i=0; i<d.size(); i++)
    {
        num+=d[i].size();
        for (auto & entry : d[i])
        {
            auto & index = entry.first;
            auto & value = entry.second;
            T temp = std::abs( (value - predictor[i]->operator()( index ))/value );
            MAPE += temp;
        }
    }
    return MAPE * 100 / num;
}

template <typename D, typename T, typename V>
T r_square_1(const std::vector<D>& d, const std::vector<wjy::predictor<T, V >*> predictor)
{
    T rs = 0, mean = 0, ss_tot = 0, ss_res = 0;
    double num = 0;
    for (int i=0; i<d.size(); i++)
    {
        num+=d[i].size();
        for (auto & entry : d[i])
        {
            auto & index = entry.first;
            auto & value = entry.second;
            mean += value;
            T temp = (value -predictor[i]->operator()( index ));
            ss_res += temp*temp;
        }
    }
    mean /= num;
    for (int i=0; i<d.size(); i++)
        for (auto & entry : d[i]) 
            ss_tot += (entry.second - mean)*(entry.second - mean);
    return 1-(ss_res/ss_tot);
}

template <typename D, typename T, typename V>
std::vector< std::pair<std::string, T> > all_evaluations_1(const std::vector<D>& d, const std::vector<wjy::predictor<T, V >*>& predictor)
{
    std::vector< std::pair<std::string, T> > ans;
    ans.emplace_back("MSE", mean_square_error_1(d, predictor));
    ans.emplace_back("MAPE", mean_absolute_percentage_error_1(d, predictor));
    ans.emplace_back("R-Square", r_square_1(d, predictor));
    return ans;
}

//  generate negative entries
template<typename T>
wjy::sparse_tensor<T, 3> generate_negative_entries(const wjy::sparse_tensor<T, 3>& t, int n = 5)
{
    wjy::sparse_tensor<T, 3> negative_tensor;
    auto dims = wjy::dims_of_sparse_tensor(t);
    auto t1 = wjy::flatten_sparse_tensor(t, 0);
    auto skill_popularity = wjy::flatten_sparse_tensor(t1, 0);
    auto position_popularity = wjy::flatten_sparse_tensor(t1, 1);
    T sum_s = 0, sum_p = 0;
    for (auto & pop : skill_popularity) sum_s += pop.second;
    for (auto & pop : position_popularity) sum_p += pop.second;
    std::mt19937 mt(1);
    std::uniform_int_distribution<size_t> uid(0, dims[0]-1);
    std::uniform_real_distribution<double> urd(0, 1);

    for (size_t position=0; position<dims[1]; position++)
    {
        T pop = position_popularity[{position}];
        double p_p = 1 / ( 1 + std::exp(-pop/sum_p*5000) );
        for (size_t skill=0; skill<dims[2]; skill++)
        {
            if (t1.count({position, skill})!=0) continue;
            pop = skill_popularity[{skill}];
            double p_s = 1 / ( 1 + std::exp(-pop/sum_s*5000) );
            double p = (p_s-0.5) * (p_p-0.5);
            if (urd(mt)>p) continue;
            wjy::sparse_tensor_index<3> index;
            for (int i=0; i<n; i++)
            {
                index[0]=uid(mt); index[1]=position; index[2]=skill;
                negative_tensor[index] = 0;
            }
        }
    }
    return negative_tensor;
}

//for test
void test();


#endif
