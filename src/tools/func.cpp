#include "func.hpp"

#include <iostream>
#include <fstream>
#include <locale>
#include <codecvt>
#include <string>
#include <list>
#include <cassert>

#include "MySQL.hpp"

std::unordered_map< std::string, size_t >   skill_to_id, company_to_id, position_to_id;
std::vector< std::string >                  id_to_skill, id_to_company, id_to_position;

std::unordered_map< std::string, size_t >   demand_level;
std::unordered_map< int, std::pair<std::string, std::string> >  jdid_company_position;
//
void read_all();
std::pair< std::string, std::string > get_company_position(int id);
std::list< std::pair<std::string, int> > parse_jd_demand(const std::vector< std::string >& jd, const std::unordered_map< std::string, size_t >& skill_list, const std::unordered_map< std::string, size_t >& demand_level);

wjy::sparse_tensor<double, 3> create_tensor(const std::string& path)
{
    std::ifstream myin(path);
    assert(myin);
    read_all();
    
    wjy::sparse_tensor<double, 3> tensor;
    wjy::sparse_tensor_index<3> index = {0, 0, 0};
    std::string st_jd;
    while (std::getline(myin, st_jd))
    {
		//std::cout<<st_jd<<"\n";
        auto job_description = split(st_jd, {',','/'});
        int id = std::atoi( job_description[0].c_str() );
		//std::cout<<"idddddddddddddddddddddddddddddd "<<id<<"\n";
        std::pair<std::string, std::string> company_position;
		if (jdid_company_position.find(id) == jdid_company_position.end()) continue;
		company_position = jdid_company_position[id];
        // these two keys must in the map.
        index[0] = company_to_id[company_position.first];
        index[1] = position_to_id[company_position.second];
		
        auto demand_level_list = parse_jd_demand(job_description, skill_to_id, demand_level);
		//std::cout<<"id "<<id<<" size "<<demand_level_list.size()<<"\n";
        for (auto& skill_level : demand_level_list)
        {
			//std::cout<<skill_level.first<<" "<<skill_level.second<<"\n";
            index[2] = skill_to_id[skill_level.first];
            tensor[ index ] = skill_level.second;
        }
		//if (id > 30) break;
    }
	myin.close();
    return tensor;
}




/****************************************************************************************/
/*                      private functions&var for this file                             */
/****************************************************************************************/

wjy::MySQL sql("127.0.0.1", "wjy", "", "lagou_data");

std::vector< std::string > split(const std::string& st, const std::unordered_set< char >& spliter)
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

// it's a (something, something_id) pair list. something_id helps to build the tensor.
void read_some_list(const std::string& path, std::unordered_map< std::string, size_t >& some_to_id, std::vector< std::string >& id_to_some)
{
    std::ifstream myin(path);
    assert(myin);
    some_to_id.clear();
    id_to_some.clear();
    
    std::string something;
    size_t n = 0;
    while (std::getline(myin, something))
    {
		if (some_to_id.find(something) != some_to_id.end()) continue;
        some_to_id[something] = n++;
        id_to_some.push_back(something);
    }
	myin.close();
}

void read_skill_list(const std::string& path = "../data/skill_list.txt")
{
    return read_some_list(path, skill_to_id, id_to_skill);
}

void read_company_list(const std::string& path = "../data/company_list.txt")
{
    return read_some_list(path, company_to_id, id_to_company);
}

void read_position_list(const std::string& path = "../data/position_list.txt")
{
    return read_some_list(path, position_to_id, id_to_position);
}

void read_demand_level(const std::string& path = "../data/demand_level.txt")
{
    std::ifstream myin(path);
    assert(myin);
    demand_level.clear();
    std::string word;
    int level;
    while (myin>>word>>level) demand_level[word] = level;
	myin.close();
}

void read_jdid_company_position(const std::string& path = "../data/jdid_company_position.txt")
{
    std::ifstream myin(path);
    assert(myin);
    jdid_company_position.clear();
    std::string st;
    while(std::getline(myin, st))
	{
		auto sts = split(st, {'?'});
		int id = std::atoi(sts[0].c_str());
		assert(id>0 && sts.size()==3);
		jdid_company_position[id] = std::make_pair(sts[1], sts[2]);
	}
	myin.close();
}

void read_all()
{
    read_skill_list();
    read_company_list();
    read_position_list();
    read_demand_level();
	read_jdid_company_position();
}


// input a jd, output all skills and it's demand.
std::list< std::pair<std::string, int> > parse_jd_demand(const std::vector< std::string >& jd, const std::unordered_map< std::string, size_t >& skill_list, const std::unordered_map< std::string, size_t >& demand_level)
{
    std::list< std::pair<std::string, int> > skill_level_list;
	std::unordered_set< std::string > skills;
    auto i = jd.size(), j = i;
    do{
        i--;
        auto skill = skill_list.find(jd[i]);
        if (skill == skill_list.end() || skills.find(skill->first)!=skills.end()) continue;
        if (j>i) // two skills may have the same level of demand
            for (j=i-1; j>0; j--)
			{
                if (demand_level.find(jd[j]) != demand_level.end()) break;
			}
		if (j==0) break;
        skill_level_list.push_back( std::make_pair(jd[i], demand_level.at(jd[j])) );
		skills.insert(skill->first);
    }while (i!=1);// jd[0] is the id for this jd
    return skill_level_list;
}

/*
std::pair< std::string, std::string > get_company_position(int id)
{
	std::cout<<id<<std::endl;
    if (!sql) sql.connect();
    std::string q = "select compname,position from clean_positions where id = \'"+ std::to_string(id) + "\';";
    sql.query(q);
    auto rslt = sql.get_result_table();
    return std::make_pair( rslt[0][0], rslt[0][1] );
}
*/


//for test
void test()
{
	read_all();
	//create_tensor();
}




