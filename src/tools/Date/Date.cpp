//
//  Date.cpp
//  test_c++
//
//  Created by wjy on 16/7/7.
//  Copyright © 2016年 wjy. All rights reserved.
//

#include "Date.hpp"
#include <sstream>
#include <time.h>
#include <stdio.h>

using namespace wjy;

//                                  00,01,02,03,04,05,06,07,08,09,10,11,12
const int Date::month_max_days[] = {-1,31,28,31,30,31,30,31,31,30,31,30,31};
const Date Date::start_date(1970,1,1,4);
//this constructor is only for construct the start_date
Date::Date(int year,int month,int day,int week)
{
    this->year = year;
    this->month = month;
    this->day = day;
    this->week = week;
    this->days_from_start_date = 0;
}

Date::Date()
{
    (*this) = today();
}

Date::Date(int year,int month,int day)
{
    if (!set_date(year, month, day))
    {
        (*this) = start_date;
    }
}

bool Date::is_leap_year(int year) const
{
    if (year < 0) return false;
    return ((year%100 != 0)&&(year%4 == 0)) || (year%400 == 0);
}
int Date::calc_days_from_start_date() const
{
    int ans = 0;
    for (int i = start_date.year;i<year;i++)
    {
        ans += 365;
        if (is_leap_year(i)) ans++;
    }
    for (int i = 1;i<month;i++)
    {
        ans += month_max_days[i];
    }
    if (2<month && is_leap_year(year)) ans++;
    ans += (day -1);
    return ans;
}

Date Date::next_day() const
{
    Date temp(*this);
    temp.days_from_start_date++;
    temp.week++;
    if (temp.week == 8) temp.week = 1;

    //in the same month
    if ((day < month_max_days[month]) || (is_leap_year(year)&&(month==2)&&(day==28)))
    {
        temp.day++;
        return temp;
    }
    //in the same year
    temp.day = 1;
    if (month<12)
    {
        temp.month++;
        return temp;
    }
    //in the next year
    temp.month = 1;
    temp.year++;
    return temp;
}
Date Date::next_n_days(int n) const
{
    Date temp(start_date);
    n = n + days_from_start_date;
    if (n<0) return start_date;
    //caculate the week and days_from_start_date
    temp.days_from_start_date = n;
    temp.week = (start_date.week + temp.days_from_start_date)%7;
    if (temp.week == 0) temp.week = 7;
    //in witch year
    int i = start_date.year;
    int days_of_year;
    while (true)
    {
        days_of_year = 365 + is_leap_year(i);
        if (n<days_of_year) break;
        n -= days_of_year;
        i++;
    }
    temp.year = i;
    //in witch month
    int days_of_month;
    for (i = 1;i<=12;i++)
    {
        days_of_month = month_max_days[i] + (is_leap_year(temp.year)&&(i == 2));
        if (n < days_of_month) break;
        n -= days_of_month;
    }
    temp.month = i;
    //day
    temp.day = n+1;
    return temp;
}

Date Date::before_day() const
{
    Date temp(*this);
    if ((--temp.days_from_start_date)<0) return start_date;
    temp.week--;
    if (temp.week == 0) temp.week=7;
    if (day != 1)
    {
        temp.day--;
    }else if (month != 1)
    {
        temp.month--;
        temp.day = month_max_days[temp.month];
        if (is_leap_year(temp.year) && (temp.month == 2)) temp.day = 29;
    }else
    {
        temp.month = 12;
        temp.day = 31;
        temp.year--;
    }
    
    return temp;
}
Date Date::before_n_days(int n) const
{
    //here has a problem
    //is n days before is earlier than the start_date, it will doesn't work
    //and return start_date
    return next_n_days(-n);
}

bool Date::set_date(int year, int month, int day)
{
    if (month<1 || month>12) return false;
    //if the date want be set is legal
    if ( (month_max_days[month] >= day) || (is_leap_year(year)&&(month==2)&&(day==29)) )
    {
        this->year = year;
        this->month = month;
        this->day = day;
        
        days_from_start_date = calc_days_from_start_date();
        //here we have already know that 1970/01/01 is Thursday, so here use 4
        week = (start_date.week+days_from_start_date)%7;
        if (week == 0) week = 7;
        return true;
    }
    return false;
}

int Date::get_year() const
{
    return year;
}
int Date::get_month() const
{
    return month;
}
int Date::get_day() const
{
    return day;
}
int Date::get_week() const
{
    return week;
}

std::string Date::to_string(const std::string spliter) const
{
    std::ostringstream myout;
    myout<<year<<spliter;
    if (month<10) myout<<0;
    myout<<month<<spliter;
    if (day<10) myout<<0;
    myout<<day;
    return myout.str();
}

bool Date::is_leap_year() const
{
    return is_leap_year(year);
}

bool wjy::operator <(const Date& D1,const Date& D2)
{
    return  (D1.days_from_start_date<D2.days_from_start_date);
}
bool wjy::operator <=(const Date& D1,const Date& D2)
{
    return  (D1.days_from_start_date<=D2.days_from_start_date);
}
bool wjy::operator >(const Date& D1,const Date& D2)
{
    return (D1.days_from_start_date>D2.days_from_start_date);
}
bool wjy::operator >=(const Date& D1,const Date& D2)
{
    return (D1.days_from_start_date>=D2.days_from_start_date);
}

bool wjy::operator ==(const Date& D1,const Date& D2)
{
    return (D1.days_from_start_date==D2.days_from_start_date);
}
bool wjy::operator !=(const Date& D1,const Date& D2)
{
    return (D1.days_from_start_date!=D2.days_from_start_date);
}

int wjy::operator -(const Date& D1,const Date& D2)
{
    return D1.days_from_start_date - D2.days_from_start_date;
}

Date Date::today()
{
    Date temp(start_date);
    time_t now_time = time(0);
    tm* local_time = localtime(&now_time);
    temp.set_date(local_time->tm_year + 1900, local_time->tm_mon + 1, local_time->tm_mday);
    return temp;
}
