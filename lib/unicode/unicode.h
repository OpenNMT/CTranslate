#pragma once

#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>

typedef uint32_t unicode_code_point_t;
typedef std::map<unicode_code_point_t, std::vector<unicode_code_point_t> > map_of_list_t;
typedef std::unordered_map<unicode_code_point_t, unicode_code_point_t> map_unicode;

std::string cp_to_utf8(unicode_code_point_t u);
unicode_code_point_t utf8_to_cp(const unsigned char* s, unsigned int &l);

std::vector<std::string> split_utf8(const std::string& str, const std::string& sep);
void explode_utf8(std::string str,
                  std::vector<std::string>& chars,
                  std::vector<unicode_code_point_t>& code_points);

enum _type_letter
{
  _letter_other,
  _letter_lower,
  _letter_upper
};

bool is_separator(unicode_code_point_t u);
bool is_letter(unicode_code_point_t u, _type_letter &tl);
bool is_number(unicode_code_point_t u);

unicode_code_point_t get_upper(unicode_code_point_t u);
unicode_code_point_t get_lower(unicode_code_point_t u);
