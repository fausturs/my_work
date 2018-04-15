//
//  Date.hpp
//  test_c++
//
//  Created by wjy on 16/7/7.
//  Copyright © 2016年 wjy. All rights reserved.
//

#ifndef Date_hpp
#define Date_hpp

//#include <stdio.h>
#include <string>

namespace wjy {
    

    class Date
    {
        int year;
        int month;
        int day;
        
        int week;
        int days_from_start_date;
        
        const static int month_max_days[];
        //start_date is the date which this class count start
        //to some reason this date only can be some year's January 1
        const static Date start_date;
        
        //this constructor is for initialize the start_date
        Date(int year,int month,int day,int week);
        
        bool is_leap_year(int year) const;
        int calc_days_from_start_date() const;
    public:
        //default date is today
        Date();
        Date(int year,int month,int day);
        Date(const Date&) = default;
        Date& operator=(const Date&) = default;
        Date(Date&&) = default;
        Date& operator=(Date&&) = default;
        
        Date next_day() const;
        Date next_n_days(int n) const;
        
        Date before_day() const;
        Date before_n_days(int n) const;
        //if the new date is legal,return true and set the date to new date
        //else return false, and do nothing
        bool set_date(int year,int month,int day);

        int get_year() const;
        int get_month() const;
        int get_day() const;
        int get_week() const;
        //cover the date to string, split by a char -- spliter which default is '/'
        //2014 7 12 will cover to  "2014/07/12"
        std::string to_string(const std::string spliter = "/") const;
        
        bool is_leap_year() const;
        
    //    friend function about compare operator
        friend bool operator <(const Date& D1,const Date& D2);
        friend bool operator <=(const Date& D1,const Date& D2);
        friend bool operator >(const Date& D1,const Date& D2);
        friend bool operator >=(const Date& D1,const Date& D2);
        
        friend bool operator ==(const Date& D1,const Date& D2);
        friend bool operator !=(const Date& D1,const Date& D2);
    //     get the days between these two date
        friend int operator -(const Date& D1,const Date& D2);
        
    //  get today
        static Date today();
    };

    bool operator <(const Date& D1,const Date& D2);
    bool operator <=(const Date& D1,const Date& D2);
    bool operator >(const Date& D1,const Date& D2);
    bool operator >=(const Date& D1,const Date& D2);

    bool operator ==(const Date& D1,const Date& D2);
    bool operator !=(const Date& D1,const Date& D2);

    int operator -(const Date& D1,const Date& D2);

    
}

#endif /* Date_hpp */
