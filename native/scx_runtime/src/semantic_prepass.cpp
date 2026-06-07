#include "semantic_prepass.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          out << "\\u00";
          const char* hex = "0123456789abcdef";
          out << hex[(ch >> 4) & 0x0f] << hex[ch & 0x0f];
        } else {
          out << ch;
        }
        break;
    }
  }
  return out.str();
}

static std::string quote_arg(const fs::path& value) {
  return "\"" + value.string() + "\"";
}

static fs::path find_repo_root() {
  fs::path cur = fs::current_path();
  for (int i = 0; i < 12; ++i) {
    if (fs::exists(cur / "native" / "semantic_kernel_cpp" / "build" / "Release" / "semantic_kernel_cli.exe")) {
      return cur;
    }
    if (!cur.has_parent_path()) {
      break;
    }
    fs::path parent = cur.parent_path();
    if (parent == cur) {
      break;
    }
    cur = parent;
  }
  return {};
}

static fs::path resolve_semantic_kernel_cli() {
  const char* env = std::getenv("SEMANTIC_KERNEL_CLI");
  if (env && *env && fs::exists(env)) {
    return fs::path(env);
  }
  fs::path root = find_repo_root();
  if (!root.empty()) {
    fs::path cli = root / "native" / "semantic_kernel_cpp" / "build" / "Release" / "semantic_kernel_cli.exe";
    if (fs::exists(cli)) {
      return cli;
    }
  }
  return {};
}

static fs::path prepass_dir() {
  fs::path dir = fs::temp_directory_path() / "kuhul_scx_semantic_prepass";
  fs::create_directories(dir);
  return dir;
}

static uint64_t stable_hash(const std::string& value) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char ch : value) {
    h ^= ch;
    h *= 1099511628211ull;
  }
  return h;
}

SemanticPrepassResult semantic_first_stop(const std::string& prompt,
                                          const std::string& stage) {
  SemanticPrepassResult result;
  fs::path cli = resolve_semantic_kernel_cli();
  result.semantic_kernel_cli = cli.string();
  if (cli.empty()) {
    return result;
  }

  fs::path dir = prepass_dir();
  const uint64_t h = stable_hash(stage + "\n" + prompt);
  fs::path input = dir / ("prompt_" + std::to_string(h) + ".xml");
  fs::path report = dir / ("prompt_" + std::to_string(h) + ".reader.json");

  std::ofstream out(input, std::ios::binary);
  if (!out.is_open()) {
    return result;
  }

  out << "<numatics-prompt stage=\"" << json_escape(stage) << "\">\n"
      << "  <fold domain=\"inference/semantic-first-stop\"/>\n"
      << "  <policy permission=\"reader_required\"/>\n"
      << "  <semantic-grams><![CDATA[\n"
      << "prompt.semantic.topology\n"
      << "dds.inference.route\n"
      << "moe.shader.dispatch\n"
      << "kxc.compiler.lowering\n"
      << "fpga.fabric.route\n"
      << "]]></semantic-grams>\n"
      << "  <prompt><![CDATA[\n"
      << prompt << "\n"
      << "]]></prompt>\n"
      << "</numatics-prompt>\n";
  out.close();

  std::string command = "\"" + quote_arg(cli) + " read_topology " + quote_arg(input) + " " + quote_arg(report) + "\"";
  int code = std::system(command.c_str());

  result.input_path = input.string();
  result.report_path = report.string();
  result.exit_code = code;
  result.ok = (code == 0 && fs::exists(report));
  return result;
}
