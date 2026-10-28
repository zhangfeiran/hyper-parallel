/* Copyright 2026 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static int capture_fd = -1;
static atomic_int captured = 0;
static atomic_long optimizer_step = 0;
static _Atomic(struct sigaction *) previous[NSIG];
static void (*next_abort)(void);
/* LD_PRELOAD reserves this slot before threads start; fatal paths must not allocate TLS. */
static _Thread_local volatile sig_atomic_t handling_fatal __attribute__((tls_model("initial-exec")));

static void write_text(const char *text)
{
    size_t remaining = strlen(text);
    while (remaining != 0) {
        ssize_t count = write(capture_fd, text, remaining);
        if (count <= 0) {
            return;
        }
        text += count;
        remaining -= (size_t)count;
    }
}

static void write_number(uint64_t value)
{
    char digits[32];
    size_t index = sizeof(digits);
    digits[--index] = '\0';
    do {
        digits[--index] = (char)('0' + value % 10);
        value /= 10;
    } while (value != 0);
    write_text(digits + index);
}

static void write_signed(int64_t value)
{
    if (value < 0) {
        write_text("-");
        write_number((uint64_t)(-(value + 1)) + 1);
    } else {
        write_number((uint64_t)value);
    }
}

int64_t __mulvdi3(int64_t left, int64_t right)
{
    int64_t result;
    if (__builtin_mul_overflow(left, right, &result)) {
        if (capture_fd >= 0) {
            write_text("DSV41_NATIVE_INTEGER_OVERFLOW operation=__mulvdi3 lhs=");
            write_signed(left);
            write_text(" rhs=");
            write_signed(right);
            write_text("\n");
        }
        abort();
    }
    return result;
}

static void dump_native(int signum, const char *origin)
{
    if (capture_fd < 0 || atomic_exchange(&captured, 1) != 0) {
        return;
    }
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    write_text("DSV41_NATIVE_STACK_BEGIN signal=");
    write_number((uint64_t)signum);
    write_text(" pid=");
    write_number((uint64_t)getpid());
    write_text(" tid=");
    write_number((uint64_t)syscall(SYS_gettid));
    write_text(" optimizer_step=");
    write_number((uint64_t)atomic_load(&optimizer_step));
    write_text(" realtime_seconds=");
    write_number((uint64_t)now.tv_sec);
    write_text(" nanoseconds=");
    write_number((uint64_t)now.tv_nsec);
    write_text(" origin=");
    write_text(origin);
    write_text("\n");
    void *frames[256];
    int count = backtrace(frames, 256);
    backtrace_symbols_fd(frames, count, capture_fd);
    write_text("DSV41_NATIVE_STACK_END frames=");
    write_number((uint64_t)count);
    write_text("\n");
}

__attribute__((noreturn)) static void terminate_with_signal(int signum)
{
    struct sigaction action = {.sa_handler = SIG_DFL};
    sigemptyset(&action.sa_mask);
    sigaction(signum, &action, NULL);
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, signum);
    sigprocmask(SIG_UNBLOCK, &mask, NULL);
    syscall(SYS_tgkill, getpid(), syscall(SYS_gettid), signum);
    _exit(128 + signum);
}

static void capture_signal(int signum, siginfo_t *info, void *context)
{
    /* A chained handler can restore this handler and re-raise the same signal. */
    if (handling_fatal != 0) {
        terminate_with_signal(signum);
    }
    handling_fatal = 1;
    dump_native(signum, "signal");
    struct sigaction *old = atomic_load(&previous[signum]);
    if (old != NULL && old->sa_handler != SIG_DFL && old->sa_handler != SIG_IGN) {
        if ((old->sa_flags & SA_RESETHAND) != 0) {
            atomic_store(&previous[signum], NULL);
            struct sigaction action = {.sa_handler = SIG_DFL};
            sigemptyset(&action.sa_mask);
            sigaction(signum, &action, NULL);
        }
        if ((old->sa_flags & SA_SIGINFO) != 0) {
            old->sa_sigaction(signum, info, context);
        } else {
            old->sa_handler(signum);
        }
        /* Fatal handlers must not resume execution or leave a pending forwarding cycle. */
        terminate_with_signal(signum);
    }
    if (old != NULL && old->sa_handler == SIG_IGN) {
        handling_fatal = 0;
        return;
    }
    terminate_with_signal(signum);
}

int dsv41_capture_install(void)
{
    if (capture_fd < 0) {
        return -1;
    }
    const int signals[] = {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE};
    for (size_t index = 0; index < sizeof(signals) / sizeof(signals[0]); ++index) {
        int signum = signals[index];
        struct sigaction current;
        if (sigaction(signum, NULL, &current) != 0) {
            return -1;
        }
        if ((current.sa_flags & SA_SIGINFO) != 0 && current.sa_sigaction == capture_signal) {
            continue;
        }
        /* Keep immutable handler snapshots alive while a fatal signal can read them. */
        struct sigaction *saved = malloc(sizeof(*saved));
        if (saved == NULL) {
            return -1;
        }
        *saved = current;
        atomic_store(&previous[signum], saved);
        struct sigaction action = {.sa_sigaction = capture_signal,
                                   .sa_flags = SA_SIGINFO | SA_ONSTACK |
                                               (current.sa_flags & (SA_RESTART | SA_NODEFER))};
        action.sa_mask = current.sa_mask;
        if (sigaction(signum, &action, NULL) != 0) {
            return -1;
        }
    }
    return 0;
}

void dsv41_capture_set_step(long value)
{
    atomic_store(&optimizer_step, value);
}

__attribute__((constructor)) static void initialize_capture(void)
{
    const char *path = getenv("DSV41_NATIVE_CAPTURE_LOG");
    next_abort = dlsym(RTLD_NEXT, "abort");
    if (path == NULL) {
        return;
    }
    capture_fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (capture_fd < 0) {
        perror("Cannot open native capture log");
        _exit(190);
    }
    /* Resolve the unwinder before a fatal path can encounter allocator corruption. */
    void *warmup[8];
    backtrace(warmup, 8);
    if (dsv41_capture_install() != 0) {
        write_text("ERROR: Cannot install native capture handlers\n");
        _exit(190);
    }
}

__attribute__((noreturn)) void abort(void)
{
    dump_native(SIGABRT, "abort");
    if (next_abort != NULL) {
        next_abort();
    }
    struct sigaction action = {.sa_handler = SIG_DFL};
    sigemptyset(&action.sa_mask);
    sigaction(SIGABRT, &action, NULL);
    syscall(SYS_tgkill, getpid(), syscall(SYS_gettid), SIGABRT);
    _exit(134);
}
