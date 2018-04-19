#include "tools.hpp"

std::vector< std::string > wjy::split(const std::string& st, const std::unordered_set< char >& spliter)
{
    if (spliter.empty()) return {st};
    size_t position=-1;
    std::vector<std::string> ans;
    for (size_t i = 0; i<st.size(); i++ )
        if (spliter.find(st[i]) != spliter.end())
        {
            ans.push_back(st.substr(position+1, i-position-1));
            position = i;
        }
    ans.push_back(st.substr(position+1));
    return ans;
}
