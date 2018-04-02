
#include "func.hpp"
#include <locale>
#include <codecvt>
#include <string>

#include "MySQL.hpp"


std::wstring_convert< std::codecvt_utf8_utf16<wchar_t> > converter;

std::wstring string_to_wstring(const std::string & narrow_utf8_source_string)
{
    return converter.from_bytes(narrow_utf8_source_string);
}
std::string wstring_to_string(const std::wstring & wide_utf16_source_string)
{
    return converter.to_bytes(wide_utf16_source_string);
}
