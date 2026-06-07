#include "runtime.h"
#include "dds_stream.h"
#include "kbc1.h"
#include "manifest_loader.h"
#include <d3d12.h>
#include <d3d11.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include <windows.h>

// Returns the directory containing scx_runtime.exe, resolved at runtime.
static std::filesystem::path exe_dir() {
  wchar_t buf[MAX_PATH];
  GetModuleFileNameW(nullptr, buf, MAX_PATH);
  return std::filesystem::path(buf).parent_path();
}

// Returns the .shards root: exe lives at .shards/native/scx_runtime/build/Release/
// So go up 4 levels: Release -> build -> scx_runtime -> native -> .shards
static std::filesystem::path shards_root() {
  return exe_dir().parent_path().parent_path().parent_path().parent_path();
}

static std::filesystem::path find_upward(std::filesystem::path start, const std::filesystem::path& suffix) {
  for (int i = 0; i < 12; ++i) {
    auto candidate = start / suffix;
    if (std::filesystem::exists(candidate)) return candidate;
    if (!start.has_parent_path()) break;
    auto parent = start.parent_path();
    if (parent == start) break;
    start = parent;
  }
  return {};
}

static std::string gpu_arch() {
  if (const char* env = std::getenv("KUHUL_GPU_ARCH")) {
    if (*env) return env;
  }
#if defined(_M_ARM64) || defined(__aarch64__)
  return "arm64";
#elif defined(_WIN64)
  return "x64";
#else
  return "x86";
#endif
}

static std::filesystem::path gpu_root() {
  if (const char* env = std::getenv("KUHUL_GPU_ROOT")) {
    if (*env && std::filesystem::exists(env)) return std::filesystem::path(env);
  }

  auto exe = exe_dir();
  std::vector<std::filesystem::path> candidates = {
    exe.parent_path() / "gpu",
    exe.parent_path().parent_path() / "gpu",
    exe.parent_path().parent_path().parent_path() / "gpu",
    find_upward(exe, "native/dotnet/PowerShell-LLM/Models/native/gpu"),
    find_upward(std::filesystem::current_path(), "native/dotnet/PowerShell-LLM/Models/native/gpu")
  };
  for (const auto& candidate : candidates) {
    if (!candidate.empty() && std::filesystem::exists(candidate / "inc" / "dxcapi.h")) {
      return candidate;
    }
  }
  return {};
}

static std::filesystem::path gpu_bin_dir() {
  auto root = gpu_root();
  if (root.empty()) return {};
  auto arch = gpu_arch();
  auto direct = root / "bin" / arch;
  if (std::filesystem::exists(direct)) return direct;
  return root / "bin" / "x64";
}

static std::filesystem::path scx_runtime_root() {
  auto exe = exe_dir();
  std::vector<std::filesystem::path> candidates = {
    exe.parent_path(),
    exe.parent_path().parent_path(),
    find_upward(exe, "native/dotnet/PowerShell-LLM/Models/native/scx_runtime"),
    find_upward(std::filesystem::current_path(), "native/dotnet/PowerShell-LLM/Models/native/scx_runtime")
  };
  for (const auto& candidate : candidates) {
    if (!candidate.empty() && std::filesystem::exists(candidate / "shaders")) {
      return candidate;
    }
  }
  return {};
}

static void prepend_gpu_dll_path() {
  auto bin = gpu_bin_dir();
  if (bin.empty()) return;
  SetDllDirectoryW(bin.wstring().c_str());
}

using namespace Microsoft::WRL;

struct D3DContext {
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> computeQ;
  ComPtr<ID3D12CommandQueue> copyQ;
  ComPtr<ID3D12CommandAllocator> allocCompute;
  ComPtr<ID3D12GraphicsCommandList> listCompute;
  ComPtr<ID3D12Fence> fenceCompute;
  uint64_t fenceValue = 0;

  ComPtr<ID3D12RootSignature> rootSig;
  ComPtr<ID3D12PipelineState> psoMoeRoute;
  ComPtr<ID3D12DescriptorHeap> srvHeap;
  UINT srvSize = 0;
  ComPtr<ID3D12Resource> bufToken;
  ComPtr<ID3D12Resource> bufRouter;
  ComPtr<ID3D12Resource> bufTopk;
};

struct D3D11Context {
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  ComPtr<ID3D11ComputeShader> csMoe;
  bool ready = false;
};
static D3D11Context g_ctx11;
static bool g_gpu_ready = false;
static D3DContext g_ctx;
static bool ensure_d3d11();

static ComPtr<ID3DBlob> compile_shader(const std::filesystem::path& path, const char* entry, const char* target){
  if(!std::filesystem::exists(path)){
    std::cerr << "Shader file not found: " << path.string() << "\n";
    return nullptr;
  }
  UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_ENABLE_BACKWARDS_COMPATIBILITY;
#if defined(_DEBUG)
  flags |= D3DCOMPILE_DEBUG;
#endif
  ComPtr<ID3DBlob> blob, err;
  HRESULT hr = D3DCompileFromFile(path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, target, flags, 0, &blob, &err);
  if(FAILED(hr)){
    std::cerr << "D3DCompileFromFile failed (" << std::hex << hr << ") for " << path.string() << "\n";
    if(err) std::cerr << "D3DCompile error: " << (char*)err->GetBufferPointer() << "\n";
    return nullptr;
  }
  return blob;
}

static ComPtr<ID3DBlob> load_shader_or_compile(const std::vector<std::filesystem::path>& searchPaths, const char* entry, const char* target){
  // 1) Try CSO siblings
  for(const auto& p : searchPaths){
    auto cso = p;
    cso.replace_extension(".cso");
    if(std::filesystem::exists(cso)){
      std::ifstream f(cso, std::ios::binary);
      if(f){
        f.seekg(0, std::ios::end);
        size_t sz = (size_t)f.tellg();
        f.seekg(0, std::ios::beg);
        ComPtr<ID3DBlob> blob;
        if(SUCCEEDED(D3DCreateBlob(sz, &blob))){
          f.read(reinterpret_cast<char*>(blob->GetBufferPointer()), sz);
          if(f.good()) {
            std::cerr << "Loaded CSO: " << cso.string() << " (" << sz << " bytes)\n";
            return blob;
          }
        }
      }
    }
  }
  // 2) Compile first existing HLSL
  for(const auto& p : searchPaths){
    if(std::filesystem::exists(p)){
      auto blob = compile_shader(p, entry, target);
      if(blob) return blob;
    } else {
      std::cerr << "Shader path not found: " << p.string() << "\n";
    }
  }
  return nullptr;
}

static bool create_root_and_pso(){
  if(g_ctx.rootSig && g_ctx.psoMoeRoute) return true;
  D3D12_DESCRIPTOR_RANGE ranges[3] = {};
  ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[0].NumDescriptors = 1; ranges[0].BaseShaderRegister = 0;
  ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[1].NumDescriptors = 1; ranges[1].BaseShaderRegister = 1;
  ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[2].NumDescriptors = 1; ranges[2].BaseShaderRegister = 0;

  D3D12_ROOT_PARAMETER params[2] = {};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
  params[0].Descriptor.ShaderRegister = 0;
  params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[1].DescriptorTable.NumDescriptorRanges = 3;
  params[1].DescriptorTable.pDescriptorRanges = ranges;
  params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
  rsDesc.NumParameters = 2;
  rsDesc.pParameters = params;
  rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

  ComPtr<ID3DBlob> sigBlob, errBlob;
  D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = { D3D_ROOT_SIGNATURE_VERSION_1_0 };
  if(FAILED(g_ctx.device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData)))){
    std::cerr << "Root signature feature unsupported\n";
    return false;
  }
  const D3D_ROOT_SIGNATURE_VERSION rsVersion = featureData.HighestVersion >= D3D_ROOT_SIGNATURE_VERSION_1_0
      ? D3D_ROOT_SIGNATURE_VERSION_1_0
      : D3D_ROOT_SIGNATURE_VERSION_1;

  if(FAILED(D3D12SerializeRootSignature(&rsDesc, rsVersion, &sigBlob, &errBlob))){
    if(errBlob) std::cerr << "RootSig error: " << (char*)errBlob->GetBufferPointer() << "\n";
    return false;
  }
  HRESULT hr_rs = g_ctx.device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&g_ctx.rootSig));
  if(FAILED(hr_rs)){
    std::cerr << "CreateRootSignature failed hr=0x" << std::hex << hr_rs << std::dec << "\n";
    return false;
  }

  // DXBC-first: prefer native FXC-compiled CSO, then HLSL.
  auto root = shards_root();
  std::vector<std::filesystem::path> search = {
    root / "native/scx_runtime/shaders/moe_route_warp.cso",
    root / "native/scx_runtime/build/Release/moe_route_warp.cso",
    root / "native/scx_runtime/shaders/moe_route_warp.hlsl",
    // CWD-relative fallbacks for legacy invocation
    std::filesystem::absolute("native/scx_runtime/shaders/moe_route_warp.cso"),
    std::filesystem::absolute("gpu/bin/x64/moe_route_warp.cso"),
  };
  auto cs = load_shader_or_compile(search, "main", "cs_5_0");
  if(!cs) {
    std::cerr << "Failed to load/compile moe_route_warp shader (no CSO or HLSL compiled)\n";
    return false;
  }

  D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
  psoDesc.pRootSignature = g_ctx.rootSig.Get();
  psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
  HRESULT hr = g_ctx.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_ctx.psoMoeRoute));
  if(FAILED(hr)){
    std::cerr << "CreateComputePipelineState failed hr=0x" << std::hex << hr << std::dec << "\n";
    return false;
  }

  return true;
}

static bool create_runtime_buffers(){
  if(g_ctx.srvHeap) return true;
  D3D12_DESCRIPTOR_HEAP_DESC hd = {};
  hd.NumDescriptors = 3;
  hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  if(FAILED(g_ctx.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_ctx.srvHeap)))) return false;
  g_ctx.srvSize = g_ctx.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

  auto makeUpload = [&](UINT64 bytes, ComPtr<ID3D12Resource>& out)->bool{
    D3D12_HEAP_PROPERTIES props = {}; props.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; desc.SampleDesc.Count = 1;
    return SUCCEEDED(g_ctx.device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&out)));
  };

  makeUpload(sizeof(float)*4, g_ctx.bufToken);
  makeUpload(sizeof(float)*64*4, g_ctx.bufRouter);
  makeUpload(sizeof(uint32_t)*2, g_ctx.bufTopk);

  auto base = g_ctx.srvHeap->GetCPUDescriptorHandleForHeapStart();
  auto next = [&](int idx){ D3D12_CPU_DESCRIPTOR_HANDLE h = base; h.ptr += idx * g_ctx.srvSize; return h; };

  D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
  srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv.Buffer.FirstElement = 0; srv.Buffer.NumElements = 4; srv.Buffer.StructureByteStride = sizeof(float); srv.Format = DXGI_FORMAT_UNKNOWN;
  g_ctx.device->CreateShaderResourceView(g_ctx.bufToken.Get(), &srv, next(0));

  srv.Buffer.NumElements = 64*4;
  g_ctx.device->CreateShaderResourceView(g_ctx.bufRouter.Get(), &srv, next(1));

  D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
  uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  uav.Buffer.FirstElement = 0; uav.Buffer.NumElements = 2; uav.Buffer.StructureByteStride = sizeof(uint32_t);
  g_ctx.device->CreateUnorderedAccessView(g_ctx.bufTopk.Get(), nullptr, &uav, next(2));

  return true;
}

bool init_gpu_if_available() {
  prepend_gpu_dll_path();
  ComPtr<IDXGIFactory4> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return false;
  UINT adapterIndex = 0;
  if(const char* env = std::getenv("GPU_ADAPTER")){
    adapterIndex = static_cast<UINT>(std::atoi(env));
  }
  ComPtr<IDXGIAdapter1> adapter;
  if (FAILED(factory->EnumAdapters1(adapterIndex, &adapter))) return false;
  D3DContext ctx;

  // Try descending feature levels; if only FL11.x is available, skip D3D12 and use D3D11.
  D3D_FEATURE_LEVEL levels[] = {
    D3D_FEATURE_LEVEL_12_1,
    D3D_FEATURE_LEVEL_12_0,
    D3D_FEATURE_LEVEL_11_1,
    D3D_FEATURE_LEVEL_11_0
  };
  bool created = false;
  for (auto fl : levels) {
    if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), fl, IID_PPV_ARGS(&ctx.device)))) {
      created = true;
      if(fl < D3D_FEATURE_LEVEL_12_0){
        std::cerr << "D3D12 device is FL11.x; using D3D11 compute path instead\n";
        g_gpu_ready = false;
        return ensure_d3d11();
      }
      break;
    }
  }
  if (!created) return ensure_d3d11();

  // Compute queue
  D3D12_COMMAND_QUEUE_DESC qdesc = {};
  qdesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  ctx.device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&ctx.computeQ));

  // Copy queue
  D3D12_COMMAND_QUEUE_DESC cdesc = {};
  cdesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
  ctx.device->CreateCommandQueue(&cdesc, IID_PPV_ARGS(&ctx.copyQ));

  ctx.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&ctx.allocCompute));
  ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, ctx.allocCompute.Get(), nullptr, IID_PPV_ARGS(&ctx.listCompute));
  ctx.listCompute->Close();

  ctx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx.fenceCompute));

  g_ctx = ctx;
  g_gpu_ready = true;
  return true;
}

// Minimal D3D11 fallback for MoE route shader
static bool ensure_d3d11(){
  if(g_ctx11.ready) return true;
  UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
  flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
  D3D_FEATURE_LEVEL levels[] = {
    D3D_FEATURE_LEVEL_11_1,
    D3D_FEATURE_LEVEL_11_0,
    D3D_FEATURE_LEVEL_10_1
  };
  D3D_FEATURE_LEVEL outLevel;
  ComPtr<ID3D11Device> dev;
  ComPtr<ID3D11DeviceContext> ctx;
  HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels, _countof(levels), D3D11_SDK_VERSION, &dev, &outLevel, &ctx);
  if(FAILED(hr)){
    hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, flags, levels, _countof(levels), D3D11_SDK_VERSION, &dev, &outLevel, &ctx);
    if(FAILED(hr)){
      std::cerr << "D3D11CreateDevice failed hr=0x" << std::hex << hr << std::dec << "\n";
      return false;
    }
  }
  auto root = shards_root();
  auto scxRoot = scx_runtime_root();
  auto gpuBin = gpu_bin_dir();
  std::vector<std::filesystem::path> search = {
    scxRoot / "shaders/moe_route_warp.cso",
    scxRoot / "build/Release/moe_route_warp.cso",
    scxRoot / "shaders/moe_route_warp.hlsl",
    exe_dir() / "moe_route_warp.cso",
    gpuBin / "moe_route_warp.cso",
    root / "native/scx_runtime/shaders/moe_route_warp.cso",    // DXBC from FXC preferred
    root / "native/scx_runtime/build/Release/moe_route_warp.cso",
    root / "native/scx_runtime/shaders/moe_route_warp.hlsl",
    // CWD-relative fallbacks for legacy invocation
    std::filesystem::absolute("native/scx_runtime/shaders/moe_route_warp.cso"),
    std::filesystem::absolute("gpu/bin/x64/moe_route_warp.cso"),
    std::filesystem::absolute("gpu/bin/x86/moe_route_warp.cso"),
  };
  auto csBlob = load_shader_or_compile(search, "main", "cs_5_0");
  if(!csBlob){
    std::cerr << "D3D11 fallback: shader load/compile failed\n";
    return false;
  }
  ComPtr<ID3D11ComputeShader> cs;
  hr = dev->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &cs);
  if(FAILED(hr)){
    std::cerr << "D3D11 CreateComputeShader failed hr=0x" << std::hex << hr << std::dec << "\n";
    return false;
  }
  g_ctx11.device = dev;
  g_ctx11.context = ctx;
  g_ctx11.csMoe = cs;
  g_ctx11.ready = true;
  std::cerr << "D3D11 fallback active (feature level 0x" << std::hex << outLevel << std::dec << ")\n";
  return true;
}

void run_d3d12(const KBC1_Program& p, const ManifestInfo* manifest) {
  if (!g_gpu_ready) {
    std::cerr << "GPU not initialized\n";
    if(ensure_d3d11()){
      g_ctx11.context->CSSetShader(g_ctx11.csMoe.Get(), nullptr, 0);
      g_ctx11.context->Dispatch(1,1,1);
    }
    return;
  }
  if(!create_root_and_pso()){
    std::cerr << "Failed to create PSO, attempting D3D11 fallback...\n";
    if(ensure_d3d11()){
      g_ctx11.context->CSSetShader(g_ctx11.csMoe.Get(), nullptr, 0);
      g_ctx11.context->Dispatch(1,1,1);
    }
    return;
  }
  create_runtime_buffers();

  // Minimal engine: iterate KBC1 ops and log; real op-specific pipelines can
  // be bound as the scaffold is extended.
  g_ctx.allocCompute->Reset();
  g_ctx.listCompute->Reset(g_ctx.allocCompute.Get(), nullptr);

  for(const auto& inst : p.inst){
    switch(inst.op){
      case OP_MOE_ROUTE:
      case OP_MOE_DISPATCH:
      case OP_MOE_COMBINE:
        g_ctx.listCompute->SetPipelineState(g_ctx.psoMoeRoute.Get());
        g_ctx.listCompute->SetComputeRootSignature(g_ctx.rootSig.Get());
        {
          ID3D12DescriptorHeap* heaps[] = { g_ctx.srvHeap.Get() };
          g_ctx.listCompute->SetDescriptorHeaps(1, heaps);
          auto gpu = g_ctx.srvHeap->GetGPUDescriptorHandleForHeapStart();
          g_ctx.listCompute->SetComputeRootDescriptorTable(1, gpu);
        }
        g_ctx.listCompute->Dispatch(1,1,1);
        break;
      default:
        break;
    }
  }

  g_ctx.listCompute->Close();
  ID3D12CommandList* lists[] = { g_ctx.listCompute.Get() };
  g_ctx.computeQ->ExecuteCommandLists(1, lists);
  g_ctx.computeQ->Signal(g_ctx.fenceCompute.Get(), ++g_ctx.fenceValue);
}
