#include "core/allocator.h"
#include "core/common.h"
#include <ostream>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        used += size;

        // Extend in the last of memory.
        if (peak >= size + last_addr_) {
            free_blocks_[last_addr_]  = size;
            last_addr_               += size;
            return last_addr_ - size;
        }

        // Find in previous blocks.
        size_t cursor = 0;
        for (const auto& [offset, block_size] : free_blocks_) {
            if (cursor + size <= offset) {
                free_blocks_[cursor] = size;
                return cursor;
            }
            cursor = offset + block_size;
        }

        // Extend the memory.
        peak                      = last_addr_ + size;
        free_blocks_[last_addr_]  = size;
        last_addr_               += size;
        return last_addr_ - size;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================

        // | ---- block 1 ---- | ---- block 2 ---- |
        // --- | addr size | ----------------------
        auto iter = free_blocks_.upper_bound(addr);
        IT_ASSERT(iter != free_blocks_.begin());
        --iter;
        auto [offset, block_size] = *iter;
        free_blocks_.erase(iter);

        // block is splitted
        if (addr != offset || size != block_size) {
            IT_ASSERT(addr + size <= offset + block_size);
            // remove the front part of the block
            if (addr == offset) {
                free_blocks_[addr + size] = block_size - size;
                // remove the end part of the block
            } else if (addr + size == offset + block_size) {
                free_blocks_[offset] = addr - offset;

                // Removed block is the last block.
                if (addr + size == last_addr_) {
                    last_addr_ = addr;
                }
                // block is split into two parts
            } else {
                free_blocks_[offset]      = addr - offset;
                free_blocks_[addr + size] = offset + block_size - addr - size;
            }
        } else {
            // Removed block is the last block.
            if (offset + block_size == last_addr_) {
                last_addr_ = offset;
            }
        }
        used -= size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
