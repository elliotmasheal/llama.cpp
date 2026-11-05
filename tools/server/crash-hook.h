// crash-hook.h — process-wide crash recorder for llama-server (C-1).
// Include ONCE in the translation unit that owns main(), then call
// crash_hook_install() as the first statement of main().
//
// Captures: first-chance+second-chance exceptions (VEH/UEF) and SIGABRT.
// Records: exception code, faulting address, module base+name of RIP, and up to
// 32 stack addresses (resolved offline against the build PDB). Writes to
// E:\fastLlama\perf-logs\crash-<timestamp>.log using only Win32 minimal APIs
// (CreateFile/WriteFile) — no malloc, no locks, no Vulkan state, safe inside a
// corrupted-process handler. Write failures are silent by design.
#pragma once
#include <windows.h>
#include <dbghelp.h>
#include <psapi.h>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <new>

#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "psapi.lib")

namespace crash_hook {

static constexpr int MAX_FRAMES = 32;
static char g_log_path[MAX_PATH] = "E:\\fastLlama\\perf-logs\\crash.log";
static char g_scratch[8192];
static volatile LONG g_in_handler = 0;
static volatile LONG g_logged = 0;

static void write_log(const char * data, DWORD len) {
    HANDLE h = CreateFileA(g_log_path, FILE_APPEND_DATA, FILE_SHARE_READ, nullptr,
                           OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { return; }   // silent: log target unavailable
    DWORD written = 0;
    WriteFile(h, data, len, &written, nullptr);
    CloseHandle(h);
}

static void append_line(const char * line) {
    // bounded copy into the static scratch, single write
    int n = 0;
    while (line[n] && n < (int) sizeof(g_scratch) - 2) { g_scratch[n] = line[n]; n++; }
    g_scratch[n++] = '\n';
    write_log(g_scratch, (DWORD) n);
}

static void record_context(const char * kind, EXCEPTION_POINTERS * ep) {
    if (InterlockedExchange(&g_in_handler, 1)) { return; }   // re-entry guard
    if (InterlockedExchange(&g_logged, 1)) { g_in_handler = 0; return; }   // one log per process

    SYSTEMTIME st;
    GetLocalTime(&st);
    // regenerate timestamped filename
    char path[MAX_PATH];
    const char * slash = strrchr(g_log_path, '\\');
    if (slash) {
        int base = (int) (slash - g_log_path + 1);
        memcpy(path, g_log_path, base);
        snprintf(path + base, MAX_PATH - base, "crash-%04d%02d%02d-%02d%02d%02d.log",
                 st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    } else {
        memcpy(path, g_log_path, MAX_PATH);
    }
    memcpy(g_log_path, path, MAX_PATH);

    int n = snprintf(g_scratch, sizeof(g_scratch),
        "=== crash ===\nkind: %s\ncode: 0x%08X\naddr: %p\nthread: %lu\ntime: %04d-%02d-%02d %02d:%02d:%02d\n",
        kind,
        ep ? (unsigned) ep->ExceptionRecord->ExceptionCode : 0,
        ep ? ep->ExceptionRecord->ExceptionAddress : nullptr,
        GetCurrentThreadId(),
        st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);

    void * rip = ep ? ep->ExceptionRecord->ExceptionAddress : nullptr;
    // module lookup for RIP
    MEMORY_BASIC_INFORMATION mbi{};
    if (rip && VirtualQuery(rip, &mbi, sizeof(mbi))) {
        char modname[MAX_PATH] = "(unknown)";
        if (GetModuleFileNameExA(GetCurrentProcess(), (HMODULE) mbi.AllocationBase, modname, MAX_PATH) == 0) {
            snprintf(modname, MAX_PATH, "mod@%p", mbi.AllocationBase);
        }
        n += snprintf(g_scratch + n, sizeof(g_scratch) - n,
            "module: %s (base=%p, rip_off=+0x%llx)\n",
            modname, mbi.AllocationBase, (unsigned long long) ((uintptr_t) rip - (uintptr_t) mbi.AllocationBase));
    }

    // stack walk: prefer dbghelp StackWalk64; fall back to raw RSP scan
    HANDLE thr = GetCurrentThread();
    CONTEXT * c = ep ? ep->ContextRecord : nullptr;
    if (c && ep) {
        STACKFRAME64 frame{};
        frame.AddrPC.Offset = c->Rip; frame.AddrPC.Mode = AddrModeFlat;
        frame.AddrFrame.Offset = c->Rbp; frame.AddrFrame.Mode = AddrModeFlat;
        frame.AddrStack.Offset = c->Rsp; frame.AddrStack.Mode = AddrModeFlat;
        static bool sym_ready = false;
        if (!sym_ready) {
            sym_ready = SymInitialize(GetCurrentProcess(), nullptr, TRUE);
        }
        n += snprintf(g_scratch + n, sizeof(g_scratch) - n, "stack (rip chain):\n");
        for (int f = 0; f < MAX_FRAMES; ++f) {
            STACKFRAME64 f2 = frame;
            if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, GetCurrentProcess(), thr, &f2, c,
                             nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) break;
            if (f2.AddrPC.Offset == 0) break;
            char sym_line[256] = "";
            union { unsigned char b[sizeof(SYMBOL_INFO) + 128]; SYMBOL_INFO s; } sb{};
            sb.s.SizeOfStruct = sizeof(SYMBOL_INFO);
            sb.s.MaxNameLen = 127;
            DWORD64 disp = 0;
            if (SymFromAddr(GetCurrentProcess(), f2.AddrPC.Offset, &disp, &sb.s)) {
                snprintf(sym_line, sizeof(sym_line), "  %s +0x%llx", sb.s.Name, (unsigned long long) disp);
            }
            MEMORY_BASIC_INFORMATION m{};
            char mod[64] = "";
            if (VirtualQuery((void *) f2.AddrPC.Offset, &m, sizeof(m))) {
                snprintf(mod, sizeof(mod), " modoff=+0x%llx", (unsigned long long) (f2.AddrPC.Offset - (uintptr_t) m.AllocationBase));
            }
            n += snprintf(g_scratch + n, sizeof(g_scratch) - n, "  #%02d rip=%p%s %s\n",
                          f, (void *) f2.AddrPC.Offset, mod, sym_line);
            frame = f2;
        }
    } else {
        n += snprintf(g_scratch + n, sizeof(g_scratch) - n, "stack: no context (abort/fastfail path)\n");
    }

    write_log(g_scratch, (DWORD) n);
    g_in_handler = 0;
}

static LONG WINAPI veh_handler(EXCEPTION_POINTERS * ep) {
    // only log the interesting first-chance codes once; everything else passes
    if (ep && ep->ExceptionRecord) {
        DWORD code = ep->ExceptionRecord->ExceptionCode;
        if (code == 0xC0000005 || code == 0xC0000409 || code == 0xC00000FD ||
            code == 0x80000003 || (code & 0xFFFF0000) == 0xE0000000) {
            record_context("first-chance", ep);
        }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

static LONG WINAPI uef_handler(EXCEPTION_POINTERS * ep) {
    record_context("unhandled", ep);
    return EXCEPTION_CONTINUE_SEARCH;
}

static void sigabrt_handler(int) {
    record_context("abort/SIGABRT", nullptr);
    // re-raise so the CRT fast-fail path proceeds unchanged
    signal(SIGABRT, SIG_DFL);
    raise(SIGABRT);
}

static void crash_hook_install() {
    // timestamped default path
    SYSTEMTIME st;
    GetLocalTime(&st);
    const char * slash = strrchr(g_log_path, '\\');
    char path[MAX_PATH];
    if (slash) {
        int base = (int) (slash - g_log_path + 1);
        memcpy(path, g_log_path, base);
        snprintf(path + base, MAX_PATH - base, "crash-%04d%02d%02d-%02d%02d%02d.log",
                 st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
        memcpy(g_log_path, path, MAX_PATH);
    }
    // pre-warm dbghelp symbol resolution for this process
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    AddVectoredExceptionHandler(1, veh_handler);
    SetUnhandledExceptionFilter(uef_handler);
    signal(SIGABRT, sigabrt_handler);
}

} // namespace crash_hook
