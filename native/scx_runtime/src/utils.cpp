#include <string>
#include <vector>
#include <sstream>

std::vector<std::string> split_lines(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream iss(s);
  std::string line;
  while (std::getline(iss, line)) out.push_back(line);
  return out;
}
