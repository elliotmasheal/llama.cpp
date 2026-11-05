#include "crash-hook.h"

int llama_server(int argc, char ** argv);

int main(int argc, char ** argv) {
    crash_hook::crash_hook_install();
    return llama_server(argc, argv);
}
