#ifndef SPARSE_TENSOR_HPP
#define SPARSE_TENSOR_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <cassert>

namespace wjy {
    
    template<size_t kth_order>
    using sparse_tensor_index = std::array<size_t, kth_order>;

    // hash function
    template<size_t kth_order>
    class sparse_tensor_index_hash{
        std::hash<std::string> string_hash;
    public:
        size_t operator()(const sparse_tensor_index<kth_order>& indexes) const
        {
            std::string st_index;
            for (auto i : indexes) st_index += std::to_string(i)+",";
            return string_hash(st_index);
        }
    };
    
    template <class T, size_t kth_order>
    using sparse_tensor = std::unordered_map< sparse_tensor_index<kth_order> , T, sparse_tensor_index_hash<kth_order> >;

    template <class T, size_t kth_order>
    void save_sparse_tensor(const sparse_tensor<T, kth_order>& t, const std::string& path)
    {
        std::ofstream myout(path);
        assert(myout);
        myout<<t.size()<<std::endl;
        for (auto & ele : t)
        {
            for (auto i : ele.first) myout<<i<<" ";
            myout<<ele.second<<std::endl;
        }
        myout.close();
    }

    template <class T, size_t kth_order>
    void load_sparse_tensor(sparse_tensor<T, kth_order>& t, const std::string& path)
    {
        std::ifstream myin(path);
        assert(myin);

        sparse_tensor_index<kth_order> indexes;
        T value;
        size_t n;
        t.clear();
        myin>>n;
        for (size_t k=0; k<n; k++)
        {
            for (size_t i=0; i<kth_order; i++) myin>>indexes[i];
            myin>>value;
            t[ indexes ] = value;
        }
        myin.close();
    }
    
}

#endif
