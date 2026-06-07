#include "dds_stream.h"

#include <algorithm>

bool DDSStream::open(const std::filesystem::path& path) {
  close();
  path_ = path;
  file.open(path_, std::ios::binary);
  if (!file.is_open()) {
    fileSize_ = 0;
    return false;
  }

  file.seekg(0, std::ios::end);
  fileSize_ = static_cast<uint64_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  return true;
}

bool DDSStream::open(const wchar_t* path) {
  if (!path) return false;
  return open(std::filesystem::path(path));
}

bool DDSStream::open(const char* path) {
  if (!path) return false;
  return open(std::filesystem::path(path));
}

bool DDSStream::read(uint64_t offset, void* dst, uint64_t size) {
  if (!file.is_open() || !dst) return false;
  if (offset > fileSize_) return false;
  if (size == 0) return true;
  if (offset + size > fileSize_) return false;

  file.clear();
  file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(size));
  return file.good() || static_cast<uint64_t>(file.gcount()) == size;
}

DDSPage DDSStream::make_page(uint64_t offset, uint64_t tileBytes) const {
  DDSPage page{};
  if (offset >= fileSize_) {
    return page;
  }
  page.offset = offset;
  page.size = std::min(tileBytes, fileSize_ - offset);
  return page;
}

std::vector<DDSPage> DDSStream::make_tiles(uint64_t tileBytes) const {
  std::vector<DDSPage> pages;
  if (tileBytes == 0 || fileSize_ == 0) {
    return pages;
  }

  for (uint64_t offset = 0; offset < fileSize_; offset += tileBytes) {
    pages.push_back(make_page(offset, tileBytes));
  }
  return pages;
}

void DDSStream::close() {
  if (file.is_open()) {
    file.close();
  }
  file.clear();
  fileSize_ = 0;
  path_.clear();
}
