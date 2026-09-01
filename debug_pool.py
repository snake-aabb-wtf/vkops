# -*- coding: utf-8 -*-
"""debug_pool.py — isolate descriptor pool / set allocation behaviour."""
import ctypes
import sys

import vk
from vk import (H, VulkanDevice, ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                ST_DESCRIPTOR_POOL_CREATE_INFO, ST_DESCRIPTOR_SET_ALLOCATE_INFO,
                DESCRIPTOR_TYPE_STORAGE_BUFFER, SHADER_STAGE_COMPUTE,
                VkDescriptorSetLayoutBinding, VkDescriptorSetLayoutCreateInfo,
                VkDescriptorPoolSize, VkDescriptorPoolCreateInfo,
                VkDescriptorSetAllocateInfo)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

dev = VulkanDevice()
print("device:", dev.info())
print("sizeof PoolCreateInfo:", ctypes.sizeof(vk.VkDescriptorPoolCreateInfo),
      " SetAllocateInfo:", ctypes.sizeof(vk.VkDescriptorSetAllocateInfo),
      " PoolSize:", ctypes.sizeof(vk.VkDescriptorPoolSize))

F = dev._bind
d = dev.device


def make_layout(nbind):
    bindings = (VkDescriptorSetLayoutBinding * nbind)()
    for i in range(nbind):
        bindings[i].binding = i
        bindings[i].descriptorType = DESCRIPTOR_TYPE_STORAGE_BUFFER
        bindings[i].descriptorCount = 1
        bindings[i].stageFlags = SHADER_STAGE_COMPUTE
    slci = VkDescriptorSetLayoutCreateInfo()
    slci.sType = ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    slci.bindingCount = nbind
    slci.pBindings = bindings
    lay = H()
    r = F["vkCreateDescriptorSetLayout"][0](d, ctypes.byref(slci), None, ctypes.byref(lay))
    print(f"layout({nbind} bindings) rc={r}")
    return lay


def try_pool(lay, max_sets, per_type, nbind, tag):
    psi = VkDescriptorPoolSize()
    psi.type = DESCRIPTOR_TYPE_STORAGE_BUFFER
    psi.descriptorCount = per_type
    dpci = VkDescriptorPoolCreateInfo()
    dpci.sType = ST_DESCRIPTOR_POOL_CREATE_INFO
    dpci.maxSets = max_sets
    dpci.poolSizeCount = 1
    dpci.pPoolSizes = ctypes.pointer(psi)
    pool = H()
    r = F["vkCreateDescriptorPool"][0](d, ctypes.byref(dpci), None, ctypes.byref(pool))
    print(f"[{tag}] create pool(maxSets={max_sets}, desc={per_type}) rc={r} pool={pool.value}")
    if r != 0:
        return
    layouts = (H * 1)(lay)
    dsai = VkDescriptorSetAllocateInfo()
    dsai.sType = ST_DESCRIPTOR_SET_ALLOCATE_INFO
    dsai.descriptorPool = pool
    dsai.descriptorSetLayoutCount = 1
    dsai.pSetLayouts = layouts
    s = H()
    r = F["vkAllocateDescriptorSets"][0](d, ctypes.byref(dsai), ctypes.byref(s))
    print(f"[{tag}] allocate set rc={r} set={s.value}")


lay4 = make_layout(4)
lay1 = make_layout(1)
try_pool(lay4, 256, 2048, 4, "A")
try_pool(lay4, 16, 128, 4, "B")
try_pool(lay1, 16, 16, 1, "C")
try_pool(lay4, 4, 16, 4, "D")
print("done")
