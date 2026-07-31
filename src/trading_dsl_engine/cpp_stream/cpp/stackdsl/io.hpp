#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace stackdsl {

class MMapFile {
public:
    MMapFile(const char* path, bool writable, std::size_t create_size = 0) {
        const int flags = writable ? (O_RDWR | (create_size ? O_CREAT : 0)) : O_RDONLY;
        fd_ = ::open(path, flags, 0644);
        if (fd_ < 0) throw std::runtime_error(std::string("open: ") + std::strerror(errno));
        if (create_size) {
            if (::ftruncate(fd_, static_cast<off_t>(create_size)) != 0) throw std::runtime_error(std::string("ftruncate: ") + std::strerror(errno));
            size_ = create_size;
        } else {
            struct stat st{};
            if (::fstat(fd_, &st) != 0) throw std::runtime_error(std::string("fstat: ") + std::strerror(errno));
            size_ = static_cast<std::size_t>(st.st_size);
        }
        const int prot = PROT_READ | (writable ? PROT_WRITE : 0);
        data_ = ::mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) throw std::runtime_error(std::string("mmap: ") + std::strerror(errno));
    }

    ~MMapFile() {
        if (data_ && data_ != MAP_FAILED) ::munmap(data_, size_);
        if (fd_ >= 0) ::close(fd_);
    }

    MMapFile(const MMapFile&) = delete;
    MMapFile& operator=(const MMapFile&) = delete;

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    void advise_sequential() noexcept {
#ifdef POSIX_FADV_SEQUENTIAL
        ::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
#ifdef MADV_SEQUENTIAL
        ::madvise(data_, size_, MADV_SEQUENTIAL);
#endif
    }

    void request_writeback(std::size_t offset, std::size_t length) noexcept {
#if defined(__linux__) && defined(SYNC_FILE_RANGE_WRITE)
        ::sync_file_range(fd_, static_cast<off64_t>(offset), static_cast<off64_t>(length), SYNC_FILE_RANGE_WRITE);
#else
        (void)offset;
        (void)length;
#endif
    }

private:
    int fd_ = -1;
    void* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace stackdsl
