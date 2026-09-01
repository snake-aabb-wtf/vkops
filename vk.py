# -*- coding: utf-8 -*-
"""vk.py — Minimal Vulkan compute core via ctypes, zero external dependencies.

All enum values/struct layouts verified against Vulkan SDK 1.4.350.0 vulkan_core.h
(pulled from the target machine). Supports: instance/device init, memory
allocation (host-visible persistent-mapped buffers), GLSL->SPIR-V via glslc,
compute pipelines with storage buffers + push constants, single or batched
dispatch with memory barriers between jobs.
"""
import ctypes
import glob
import hashlib
import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np

H = ctypes.c_void_p  # generic Vulkan handle

# ---- enum values (verified against vulkan_core.h 1.4.350.0) ----
API_VERSION_1_3 = (1 << 22) | (3 << 12)

ST_APPLICATION_INFO = 0
ST_INSTANCE_CREATE_INFO = 1
ST_PHYSICAL_DEVICE_FEATURES_2 = 1000059000
ST_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES = 1000082000
ST_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES = 1000083000
ST_DEVICE_QUEUE_CREATE_INFO = 2
ST_DEVICE_CREATE_INFO = 3
ST_SUBMIT_INFO = 4
ST_MEMORY_ALLOCATE_INFO = 5
ST_BUFFER_CREATE_INFO = 12
ST_PIPELINE_SHADER_STAGE_CREATE_INFO = 18
ST_COMPUTE_PIPELINE_CREATE_INFO = 29
ST_PIPELINE_LAYOUT_CREATE_INFO = 30
ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32
ST_DESCRIPTOR_POOL_CREATE_INFO = 33
ST_DESCRIPTOR_SET_ALLOCATE_INFO = 34
ST_WRITE_DESCRIPTOR_SET = 35
ST_COMMAND_POOL_CREATE_INFO = 39
ST_COMMAND_BUFFER_ALLOCATE_INFO = 40
ST_COMMAND_BUFFER_BEGIN_INFO = 42
ST_BUFFER_MEMORY_BARRIER = 44
ST_MEMORY_BARRIER = 46

BUFFER_USAGE_TRANSFER_SRC = 0x1
BUFFER_USAGE_TRANSFER_DST = 0x2
BUFFER_USAGE_STORAGE_BUFFER = 0x20

MEMORY_DEVICE_LOCAL = 0x1
MEMORY_HOST_VISIBLE = 0x2
MEMORY_HOST_COHERENT = 0x4
MEMORY_HOST_CACHED = 0x8

QUEUE_GRAPHICS = 0x1
QUEUE_COMPUTE = 0x2

DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
SHADER_STAGE_COMPUTE = 0x20

PIPELINE_BIND_POINT_COMPUTE = 1
COMMAND_BUFFER_LEVEL_PRIMARY = 0
COMMAND_POOL_RESET_COMMAND_BUFFER = 0x2
COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT = 0x1

PHYS_DEVICE_TYPE_INTEGRATED = 1
PHYS_DEVICE_TYPE_DISCRETE = 2
PHYS_DEVICE_TYPE_CPU = 4

PIPELINE_STAGE_COMPUTE = 0x800
PIPELINE_STAGE_TRANSFER = 0x1000
ACCESS_SHADER_READ = 0x20
ACCESS_SHADER_WRITE = 0x40
ACCESS_MEMORY_READ = 0x8000
ACCESS_MEMORY_WRITE = 0x10000

VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF
SHARING_MODE_EXCLUSIVE = 0


class VulkanError(Exception):
    pass


def _check(result, what):
    if result != 0:
        raise VulkanError(f"{what} failed: VkResult {result}")


# ---- structures ----
class VkApplicationInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32), ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32), ("apiVersion", ctypes.c_uint32)]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
                ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
                ("enabledExtensionCount", ctypes.c_uint32), ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p))]


class VkDeviceQueueCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32),
                ("queueFamilyIndex", ctypes.c_uint32), ("queueCount", ctypes.c_uint32),
                ("pQueuePriorities", ctypes.POINTER(ctypes.c_float))]


class VkDeviceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32),
                ("queueCreateInfoCount", ctypes.c_uint32), ("pQueueCreateInfos", ctypes.POINTER(VkDeviceQueueCreateInfo)),
                ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
                ("enabledExtensionCount", ctypes.c_uint32), ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
                ("pEnabledFeatures", H)]


class VkMemoryAllocateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("allocationSize", ctypes.c_uint64),
                ("memoryTypeIndex", ctypes.c_uint32)]


class VkBufferCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32),
                ("size", ctypes.c_uint64), ("usage", ctypes.c_uint32), ("sharingMode", ctypes.c_uint32),
                ("queueFamilyIndexCount", ctypes.c_uint32), ("pQueueFamilyIndices", ctypes.POINTER(ctypes.c_uint32))]


class VkMemoryRequirements(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("alignment", ctypes.c_uint64), ("memoryTypeBits", ctypes.c_uint32)]


class VkMemoryType(ctypes.Structure):
    _fields_ = [("propertyFlags", ctypes.c_uint32), ("heapIndex", ctypes.c_uint32)]


class VkMemoryHeap(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("flags", ctypes.c_uint32)]


class VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [("memoryTypeCount", ctypes.c_uint32), ("memoryTypes", VkMemoryType * 32),
                ("memoryHeapCount", ctypes.c_uint32), ("memoryHeaps", VkMemoryHeap * 16)]


class VkQueueFamilyProperties(ctypes.Structure):
    _fields_ = [("queueFlags", ctypes.c_uint32), ("queueCount", ctypes.c_uint32),
                ("timestampValidBits", ctypes.c_uint32), ("minImageTransferGranularity", ctypes.c_uint32 * 3)]


class VkDescriptorSetLayoutBinding(ctypes.Structure):
    _fields_ = [("binding", ctypes.c_uint32), ("descriptorType", ctypes.c_uint32),
                ("descriptorCount", ctypes.c_uint32), ("stageFlags", ctypes.c_uint32), ("pImmutableSamplers", H)]


class VkDescriptorSetLayoutCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32),
                ("bindingCount", ctypes.c_uint32), ("pBindings", ctypes.POINTER(VkDescriptorSetLayoutBinding))]


class VkDescriptorPoolSize(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("descriptorCount", ctypes.c_uint32)]


class VkDescriptorPoolCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32),
                ("maxSets", ctypes.c_uint32), ("poolSizeCount", ctypes.c_uint32),
                ("pPoolSizes", ctypes.POINTER(VkDescriptorPoolSize))]


class VkDescriptorSetAllocateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("descriptorPool", H),
                ("descriptorSetLayoutCount", ctypes.c_uint32), ("pSetLayouts", ctypes.POINTER(H))]


class VkDescriptorBufferInfo(ctypes.Structure):
    _fields_ = [("buffer", H), ("offset", ctypes.c_uint64), ("range", ctypes.c_uint64)]


class VkWriteDescriptorSet(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("dstSet", H), ("dstBinding", ctypes.c_uint32),
                ("dstArrayElement", ctypes.c_uint32), ("descriptorCount", ctypes.c_uint32),
                ("descriptorType", ctypes.c_uint32), ("pImageInfo", H),
                ("pBufferInfo", ctypes.POINTER(VkDescriptorBufferInfo)), ("pTexelBufferView", H)]


class VkPushConstantRange(ctypes.Structure):
    _fields_ = [("stageFlags", ctypes.c_uint32), ("offset", ctypes.c_uint32), ("size", ctypes.c_uint32)]


class VkPipelineLayoutCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32),
                ("setLayoutCount", ctypes.c_uint32), ("pSetLayouts", ctypes.POINTER(H)),
                ("pushConstantRangeCount", ctypes.c_uint32), ("pPushConstantRanges", ctypes.POINTER(VkPushConstantRange))]


class VkPipelineShaderStageCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32),
                ("stage", ctypes.c_uint32), ("module", H), ("pName", ctypes.c_char_p), ("pSpecializationInfo", H)]


class VkComputePipelineCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32),
                ("stage", VkPipelineShaderStageCreateInfo), ("layout", H),
                ("basePipelineHandle", H), ("basePipelineIndex", ctypes.c_int32)]


class VkCommandPoolCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32), ("queueFamilyIndex", ctypes.c_uint32)]


class VkCommandBufferAllocateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("commandPool", H),
                ("level", ctypes.c_uint32), ("commandBufferCount", ctypes.c_uint32)]


class VkCommandBufferBeginInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32), ("pInheritanceInfo", H)]


class VkMemoryBarrier(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H),
                ("srcAccessMask", ctypes.c_uint32), ("dstAccessMask", ctypes.c_uint32)]


class VkPhysicalDeviceFeatures2(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("features", ctypes.c_uint32 * 64)]


class VkPhysicalDeviceShaderFloat16Int8Features(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H),
                ("shaderFloat16", ctypes.c_uint32), ("shaderInt8", ctypes.c_uint32)]


class VkPhysicalDevice16BitStorageFeatures(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H),
                ("storageBuffer16BitAccess", ctypes.c_uint32),
                ("uniformAndStorageBuffer16BitAccess", ctypes.c_uint32),
                ("storagePushConstant16", ctypes.c_uint32),
                ("storageInputOutput16", ctypes.c_uint32)]


class VkSubmitInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H),
                ("waitSemaphoreCount", ctypes.c_uint32), ("pWaitSemaphores", H),
                ("pWaitDstStageMask", H), ("commandBufferCount", ctypes.c_uint32),
                ("pCommandBuffers", ctypes.POINTER(H)), ("signalSemaphoreCount", ctypes.c_uint32),
                ("pSignalSemaphores", H)]


def _find_vulkan_dll():
    candidates = [shutil.which("vulkan-1.dll")]
    candidates += glob.glob(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                                         "System32", "vulkan-1.dll"))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise VulkanError("vulkan-1.dll not found; is a Vulkan driver installed?")


def _find_glslc():
    p = shutil.which("glslc")
    if p:
        return p
    for c in glob.glob(r"C:\VulkanSDK\*\Bin\glslc.exe"):
        return c
    return None


class Buffer:
    """Host-visible persistent-mapped storage buffer (ideal for APUs)."""

    def __init__(self, dev, size):
        self.dev = dev
        self.size = int(size)
        bc = VkBufferCreateInfo()
        bc.sType = ST_BUFFER_CREATE_INFO
        bc.size = self.size
        bc.usage = BUFFER_USAGE_STORAGE_BUFFER | BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST
        bc.sharingMode = SHARING_MODE_EXCLUSIVE
        self.handle = H()
        _check(dev.call("vkCreateBuffer", dev.device, ctypes.byref(bc), None, ctypes.byref(self.handle)), "vkCreateBuffer")
        req = VkMemoryRequirements()
        dev.call("vkGetBufferMemoryRequirements", dev.device, self.handle, ctypes.byref(req))
        self.mem = H()
        ma = VkMemoryAllocateInfo()
        ma.sType = ST_MEMORY_ALLOCATE_INFO
        ma.allocationSize = req.size
        ma.memoryTypeIndex = dev.pick_memory_type(req.memoryTypeBits)
        _check(dev.call("vkAllocateMemory", dev.device, ctypes.byref(ma), None, ctypes.byref(self.mem)), "vkAllocateMemory")
        _check(dev.call("vkBindBufferMemory", dev.device, self.handle, self.mem, 0), "vkBindBufferMemory")
        self._ptr = H()
        _check(dev.call("vkMapMemory", dev.device, self.mem, 0, VK_WHOLE_SIZE, 0, ctypes.byref(self._ptr)), "vkMapMemory")

    def np(self, dtype=np.float32, count=None, offset=0):
        dt = np.dtype(dtype)
        if count is None:
            count = (self.size - offset) // dt.itemsize
        raw = (ctypes.c_ubyte * (self.size - offset)).from_address(self._ptr.value)
        return np.frombuffer(raw, dtype=dt, count=count, offset=offset)

    def upload(self, arr):
        arr = np.ascontiguousarray(arr)
        v = self.np(arr.dtype, count=arr.size)
        v[:] = arr.ravel()

    def download_to(self, arr):
        arr = np.ascontiguousarray(arr)
        v = self.np(arr.dtype, count=arr.size)
        arr.ravel()[:] = v

    def free(self):
        d = self.dev
        if getattr(self, "mem", None) is not None:
            d.call("vkDestroyBuffer", d.device, self.handle, None)
            d.call("vkFreeMemory", d.device, self.mem, None)
            self.mem = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class Pipeline:
    def __init__(self, dev, handle, set_layout, layout, num_bindings):
        self.dev = dev
        self.handle = handle
        self.set_layout = set_layout
        self.layout = layout
        self.num_bindings = num_bindings


class VulkanDevice:
    """Headless Vulkan compute device (single compute queue)."""

    def __init__(self, prefer_integrated=True):
        self.dll = ctypes.WinDLL(_find_vulkan_dll())
        d = self.dll
        F = self._bind = {}
        F["vkCreateInstance"] = (d.vkCreateInstance, ctypes.c_int, [ctypes.POINTER(VkInstanceCreateInfo), H, ctypes.POINTER(H)])
        F["vkEnumeratePhysicalDevices"] = (d.vkEnumeratePhysicalDevices, ctypes.c_int, [H, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(H)])
        F["vkGetPhysicalDeviceProperties"] = (d.vkGetPhysicalDeviceProperties, None, [H, H])  # dest as raw pointer
        F["vkGetPhysicalDeviceFeatures2"] = (d.vkGetPhysicalDeviceFeatures2, None, [H, H])    # pNext-chained, raw
        F["vkGetPhysicalDeviceMemoryProperties"] = (d.vkGetPhysicalDeviceMemoryProperties, None, [H, ctypes.POINTER(VkPhysicalDeviceMemoryProperties)])
        F["vkGetPhysicalDeviceQueueFamilyProperties"] = (d.vkGetPhysicalDeviceQueueFamilyProperties, None, [H, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(VkQueueFamilyProperties)])
        F["vkCreateDevice"] = (d.vkCreateDevice, ctypes.c_int, [H, ctypes.POINTER(VkDeviceCreateInfo), H, ctypes.POINTER(H)])
        F["vkGetDeviceQueue"] = (d.vkGetDeviceQueue, None, [H, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(H)])
        F["vkDestroyInstance"] = (d.vkDestroyInstance, None, [H, H])
        F["vkDestroyDevice"] = (d.vkDestroyDevice, None, [H, H])
        # device-level
        F["vkAllocateMemory"] = (d.vkAllocateMemory, ctypes.c_int, [H, ctypes.POINTER(VkMemoryAllocateInfo), H, ctypes.POINTER(H)])
        F["vkFreeMemory"] = (d.vkFreeMemory, None, [H, H, H])
        F["vkMapMemory"] = (d.vkMapMemory, ctypes.c_int, [H, H, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(H)])
        F["vkCreateBuffer"] = (d.vkCreateBuffer, ctypes.c_int, [H, ctypes.POINTER(VkBufferCreateInfo), H, ctypes.POINTER(H)])
        F["vkDestroyBuffer"] = (d.vkDestroyBuffer, None, [H, H, H])
        F["vkGetBufferMemoryRequirements"] = (d.vkGetBufferMemoryRequirements, None, [H, H, ctypes.POINTER(VkMemoryRequirements)])
        F["vkBindBufferMemory"] = (d.vkBindBufferMemory, ctypes.c_int, [H, H, H, ctypes.c_uint64])
        F["vkCreateShaderModule"] = (d.vkCreateShaderModule, ctypes.c_int, [H, ctypes.POINTER(VkShaderModuleCreateInfoStub), H, ctypes.POINTER(H)])
        F["vkDestroyShaderModule"] = (d.vkDestroyShaderModule, None, [H, H, H])
        F["vkCreateDescriptorSetLayout"] = (d.vkCreateDescriptorSetLayout, ctypes.c_int, [H, ctypes.POINTER(VkDescriptorSetLayoutCreateInfo), H, ctypes.POINTER(H)])
        F["vkCreateDescriptorPool"] = (d.vkCreateDescriptorPool, ctypes.c_int, [H, ctypes.POINTER(VkDescriptorPoolCreateInfo), H, ctypes.POINTER(H)])
        F["vkResetDescriptorPool"] = (d.vkResetDescriptorPool, ctypes.c_int, [H, H, ctypes.c_uint32])
        F["vkAllocateDescriptorSets"] = (d.vkAllocateDescriptorSets, ctypes.c_int, [H, ctypes.POINTER(VkDescriptorSetAllocateInfo), ctypes.POINTER(H)])
        F["vkUpdateDescriptorSets"] = (d.vkUpdateDescriptorSets, None, [H, ctypes.c_uint32, ctypes.POINTER(VkWriteDescriptorSet), ctypes.c_uint32, H])
        F["vkCreatePipelineLayout"] = (d.vkCreatePipelineLayout, ctypes.c_int, [H, ctypes.POINTER(VkPipelineLayoutCreateInfo), H, ctypes.POINTER(H)])
        F["vkCreateComputePipelines"] = (d.vkCreateComputePipelines, ctypes.c_int, [H, H, ctypes.c_uint32, ctypes.POINTER(VkComputePipelineCreateInfo), H, ctypes.POINTER(H)])
        F["vkDestroyPipeline"] = (d.vkDestroyPipeline, None, [H, H, H])
        F["vkDestroyPipelineLayout"] = (d.vkDestroyPipelineLayout, None, [H, H, H])
        F["vkDestroyDescriptorSetLayout"] = (d.vkDestroyDescriptorSetLayout, None, [H, H, H])
        F["vkDestroyDescriptorPool"] = (d.vkDestroyDescriptorPool, None, [H, H, H])
        F["vkCreateCommandPool"] = (d.vkCreateCommandPool, ctypes.c_int, [H, ctypes.POINTER(VkCommandPoolCreateInfo), H, ctypes.POINTER(H)])
        F["vkAllocateCommandBuffers"] = (d.vkAllocateCommandBuffers, ctypes.c_int, [H, ctypes.POINTER(VkCommandBufferAllocateInfo), ctypes.POINTER(H)])
        F["vkResetCommandBuffer"] = (d.vkResetCommandBuffer, ctypes.c_int, [H, ctypes.c_uint32])
        F["vkBeginCommandBuffer"] = (d.vkBeginCommandBuffer, ctypes.c_int, [H, ctypes.POINTER(VkCommandBufferBeginInfo)])
        F["vkEndCommandBuffer"] = (d.vkEndCommandBuffer, ctypes.c_int, [H])
        F["vkCmdBindPipeline"] = (d.vkCmdBindPipeline, None, [H, ctypes.c_uint32, H])
        F["vkCmdBindDescriptorSets"] = (d.vkCmdBindDescriptorSets, None, [H, ctypes.c_uint32, H, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(H), ctypes.c_uint32, H])
        F["vkCmdPushConstants"] = (d.vkCmdPushConstants, None, [H, H, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, H])
        F["vkCmdDispatch"] = (d.vkCmdDispatch, None, [H, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32])
        F["vkCmdPipelineBarrier"] = (d.vkCmdPipelineBarrier, None, [H, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, H, ctypes.c_uint32, H, ctypes.c_uint32, ctypes.POINTER(VkMemoryBarrier)])
        F["vkQueueSubmit"] = (d.vkQueueSubmit, ctypes.c_int, [H, ctypes.c_uint32, ctypes.POINTER(VkSubmitInfo), H])
        F["vkQueueWaitIdle"] = (d.vkQueueWaitIdle, ctypes.c_int, [H])
        F["vkDeviceWaitIdle"] = (d.vkDeviceWaitIdle, ctypes.c_int, [H])

        # actually attach signatures (stored tuples are (func, restype, argtypes))
        for _name, (_f, _rt, _at) in F.items():
            _f.restype = _rt
            _f.argtypes = _at

        self._create_instance_and_device(prefer_integrated)
        self._glslc = _find_glslc()
        self._spv_cache_dir = os.path.join(tempfile.gettempdir(), "vkops_spv")
        os.makedirs(self._spv_cache_dir, exist_ok=True)
        self._sets_cache = {}
        self._pools = []
        self._pool_sets_left = 0
        self._current_pool = None
        self._set_cache = {}

    # -- init -------------------------------------------------------------
    def _create_instance_and_device(self, prefer_integrated):
        F = self._bind
        app = VkApplicationInfo()
        app.sType = ST_APPLICATION_INFO
        app.pApplicationName = b"vkops"
        ici = VkInstanceCreateInfo()
        ici.sType = ST_INSTANCE_CREATE_INFO
        ici.pApplicationInfo = ctypes.pointer(app)
        layers = []
        if os.environ.get("VKOPS_VALIDATE"):
            layers.append(b"VK_LAYER_KHRONOS_validation")
        if layers:
            pp = (ctypes.c_char_p * len(layers))(*layers)
            ici.enabledLayerCount = len(layers)
            ici.ppEnabledLayerNames = pp
        self.instance = H()
        # request the newest API we can; fall back on very old loaders
        err = None
        for api in (API_VERSION_1_3, (1 << 22) | (2 << 12), (1 << 22) | (1 << 12)):
            app.apiVersion = api
            err = F["vkCreateInstance"][0](ctypes.byref(ici), None, ctypes.byref(self.instance))
            if err == 0:
                break
        _check(err, "vkCreateInstance")

        n = ctypes.c_uint32(0)
        _check(F["vkEnumeratePhysicalDevices"][0](self.instance, ctypes.byref(n), None), "vkEnumeratePhysicalDevices")
        pdevs = (H * n.value)()
        _check(F["vkEnumeratePhysicalDevices"][0](self.instance, ctypes.byref(n), pdevs), "vkEnumeratePhysicalDevices")

        best, fallback = None, None
        self.physical_device = None
        self.device_name = "?"
        self.device_type = None
        for i in range(n.value):
            pd = pdevs[i]
            buf = ctypes.create_string_buffer(4096)
            F["vkGetPhysicalDeviceProperties"][0](pd, ctypes.cast(buf, H))
            b = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint32))
            dev_type = b[4]
            api_version = b[0]
            name = buf.raw[20:20 + 256].split(b"\x00")[0].decode("utf-8", "replace")
            info = (pd, dev_type, name, api_version)
            if fallback is None:
                fallback = info
            if dev_type == PHYS_DEVICE_TYPE_INTEGRATED:
                best = info
                break
            if dev_type == PHYS_DEVICE_TYPE_DISCRETE and best is None:
                best = info
        info = best or fallback
        if info is None:
            raise VulkanError("no Vulkan physical device found")
        self.physical_device, self.device_type, self.device_name, self.api_version = info
        if prefer_integrated and best is None:
            pass  # keep fallback anyway

        # memory types
        memprops = VkPhysicalDeviceMemoryProperties()
        F["vkGetPhysicalDeviceMemoryProperties"][0](self.physical_device, ctypes.byref(memprops))
        self.memprops = memprops
        # Prefer cached system-heap memory: on APUs all RAM is unified, and the
        # DEVICE_LOCAL heap is a tiny carveout that the display also eats into.
        self.memtype_default = self._choose_type(
            required=MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            prefer=[MEMORY_HOST_CACHED, MEMORY_DEVICE_LOCAL])

        # queue family: first with compute bit
        qn = ctypes.c_uint32(0)
        F["vkGetPhysicalDeviceQueueFamilyProperties"][0](self.physical_device, ctypes.byref(qn), None)
        qprops = (VkQueueFamilyProperties * qn.value)()
        F["vkGetPhysicalDeviceQueueFamilyProperties"][0](self.physical_device, ctypes.byref(qn), qprops)
        self.queue_family = None
        for i in range(qn.value):
            if qprops[i].queueFlags & QUEUE_COMPUTE:
                self.queue_family = i
                self.queue_family_props = qprops[i]
                break
        if self.queue_family is None:
            raise VulkanError("no compute-capable queue family")

        qci = VkDeviceQueueCreateInfo()
        qci.sType = ST_DEVICE_QUEUE_CREATE_INFO
        qci.queueFamilyIndex = self.queue_family
        qci.queueCount = 1
        prio = (ctypes.c_float * 1)(1.0)
        qci.pQueuePriorities = prio
        dci = VkDeviceCreateInfo()
        dci.sType = ST_DEVICE_CREATE_INFO
        dci.queueCreateInfoCount = 1
        dci.pQueueCreateInfos = ctypes.pointer(qci)

        # query & enable FP16 (packed math) + 16-bit storage buffer access
        f16q = VkPhysicalDeviceShaderFloat16Int8Features()
        f16q.sType = ST_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES
        b16q = VkPhysicalDevice16BitStorageFeatures()
        b16q.sType = ST_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES
        f2q = VkPhysicalDeviceFeatures2()
        f2q.sType = ST_PHYSICAL_DEVICE_FEATURES_2
        f2q.pNext = ctypes.addressof(f16q)
        f16q.pNext = ctypes.addressof(b16q)
        self.call("vkGetPhysicalDeviceFeatures2", self.physical_device, ctypes.byref(f2q))
        api_maj = self.api_version >> 22
        api_min = (self.api_version >> 12) & 0x3FF
        # feature structs are core since Vulkan 1.2; on older devices we would
        # need the VK_KHR_* extension names enabled, so require 1.2 for FP16.
        self.fp16_enabled = (api_maj, api_min) >= (1, 2) and \
            bool(f16q.shaderFloat16) and bool(b16q.storageBuffer16BitAccess)
        self._fp16_dev_structs = None
        if self.fp16_enabled:
            f16e = VkPhysicalDeviceShaderFloat16Int8Features()
            f16e.sType = ST_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES
            f16e.shaderFloat16 = 1
            f16e.shaderInt8 = int(bool(f16q.shaderInt8))
            b16e = VkPhysicalDevice16BitStorageFeatures()
            b16e.sType = ST_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES
            b16e.storageBuffer16BitAccess = 1
            b16e.uniformAndStorageBuffer16BitAccess = 1
            f16e.pNext = ctypes.addressof(b16e)
            self._fp16_dev_structs = (f16e, b16e)
            dci.pNext = ctypes.addressof(f16e)

        self.device = H()
        _check(F["vkCreateDevice"][0](self.physical_device, ctypes.byref(dci), None, ctypes.byref(self.device)), "vkCreateDevice")
        self.queue = H()
        F["vkGetDeviceQueue"][0](self.device, self.queue_family, 0, ctypes.byref(self.queue))

        cpi = VkCommandPoolCreateInfo()
        cpi.sType = ST_COMMAND_POOL_CREATE_INFO
        cpi.flags = COMMAND_POOL_RESET_COMMAND_BUFFER
        cpi.queueFamilyIndex = self.queue_family
        self.cmd_pool = H()
        _check(F["vkCreateCommandPool"][0](self.device, ctypes.byref(cpi), None, ctypes.byref(self.cmd_pool)), "vkCreateCommandPool")
        cai = VkCommandBufferAllocateInfo()
        cai.sType = ST_COMMAND_BUFFER_ALLOCATE_INFO
        cai.commandPool = self.cmd_pool
        cai.level = COMMAND_BUFFER_LEVEL_PRIMARY
        cai.commandBufferCount = 1
        self.cmd = H()
        _check(F["vkAllocateCommandBuffers"][0](self.device, ctypes.byref(cai), ctypes.byref(self.cmd)), "vkAllocateCommandBuffers")

    def _choose_type(self, required, prefer):
        """Pick memory type index containing `required` bits, maximizing preferred bits (in order)."""
        best_idx, best_score = None, -1
        for i in range(self.memprops.memoryTypeCount):
            t = self.memprops.memoryTypes[i]
            if (t.propertyFlags & required) != required:
                continue
            score = 0
            for bit in prefer:
                if t.propertyFlags & bit:
                    score += 1
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx is None:
            raise VulkanError(f"no memory type with required bits {required:#x}")
        return best_idx

    def pick_memory_type(self, type_bits):
        """Within allowed bits, prefer the default host-visible coherent type."""
        if (type_bits >> self.memtype_default) & 1:
            return self.memtype_default
        for i in range(self.memprops.memoryTypeCount):
            if (type_bits >> i) & 1:
                return i
        raise VulkanError("no compatible memory type")

    def alloc(self, nbytes):
        return Buffer(self, nbytes)

    def call(self, name, *args):
        f, _rt, _at = self._bind[name]
        return f(*args)

    # -- shaders / pipelines ----------------------------------------------
    def compile_spirv(self, glsl_src):
        if not self._glslc:
            raise VulkanError("glslc not found (Vulkan SDK required on this machine)")
        api_maj = self.api_version >> 22
        api_min = (self.api_version >> 12) & 0x3FF
        if (api_maj, api_min) >= (1, 3):
            target_env = "vulkan1.3"
        elif (api_maj, api_min) >= (1, 2):
            target_env = "vulkan1.2"
        else:
            target_env = "vulkan1.1"
        # cache key includes the target env: same source, different SPIR-V per API version
        key = hashlib.sha1((target_env + "|" + glsl_src).encode()).hexdigest()[:16]
        out = os.path.join(self._spv_cache_dir, key + ".spv")
        if not os.path.exists(out):
            src = out + ".comp"
            with open(src, "w", encoding="utf-8") as f:
                f.write(glsl_src)
            r = subprocess.run([self._glslc, "-O", f"--target-env={target_env}", "-o", out, src],
                               capture_output=True)
            if r.returncode != 0:
                raise VulkanError("glslc error:\n" + r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace"))
        with open(out, "rb") as f:
            return f.read()

    def make_pipeline(self, glsl_src, num_bindings, push_size=32):
        spv = self.compile_spirv(glsl_src)
        F = self._bind
        # shader module
        code = (ctypes.c_uint32 * (len(spv) // 4)).from_buffer_copy(spv)
        stub = VkShaderModuleCreateInfoStub()
        stub.sType = ST_SHADER_MODULE_CREATE_INFO
        stub.codeSize = len(spv)
        stub.pCode = ctypes.cast(code, H)
        module = H()
        _check(F["vkCreateShaderModule"][0](self.device, ctypes.byref(stub), None, ctypes.byref(module)), "vkCreateShaderModule")
        # set layout
        bindings = (VkDescriptorSetLayoutBinding * num_bindings)()
        for i in range(num_bindings):
            bindings[i].binding = i
            bindings[i].descriptorType = DESCRIPTOR_TYPE_STORAGE_BUFFER
            bindings[i].descriptorCount = 1
            bindings[i].stageFlags = SHADER_STAGE_COMPUTE
        slci = VkDescriptorSetLayoutCreateInfo()
        slci.sType = ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
        slci.bindingCount = num_bindings
        slci.pBindings = bindings
        set_layout = H()
        _check(F["vkCreateDescriptorSetLayout"][0](self.device, ctypes.byref(slci), None, ctypes.byref(set_layout)), "vkCreateDescriptorSetLayout")
        # pipeline layout + push constants
        pcr = VkPushConstantRange()
        pcr.stageFlags = SHADER_STAGE_COMPUTE
        pcr.offset = 0
        pcr.size = push_size
        plci = VkPipelineLayoutCreateInfo()
        plci.sType = ST_PIPELINE_LAYOUT_CREATE_INFO
        plci.setLayoutCount = 1
        plci.pSetLayouts = ctypes.pointer(set_layout)
        plci.pushConstantRangeCount = 1
        plci.pPushConstantRanges = ctypes.pointer(pcr)
        layout = H()
        _check(F["vkCreatePipelineLayout"][0](self.device, ctypes.byref(plci), None, ctypes.byref(layout)), "vkCreatePipelineLayout")
        # compute pipeline
        stage = VkPipelineShaderStageCreateInfo()
        stage.sType = ST_PIPELINE_SHADER_STAGE_CREATE_INFO
        stage.stage = SHADER_STAGE_COMPUTE  # VkShaderStageFlagBits bit value (0x20)
        stage.module = module
        stage.pName = b"main"
        cpci = VkComputePipelineCreateInfo()
        cpci.sType = ST_COMPUTE_PIPELINE_CREATE_INFO
        cpci.stage = stage
        cpci.layout = layout
        cpci.basePipelineIndex = -1
        pipeline = H()
        _check(F["vkCreateComputePipelines"][0](self.device, None, 1, ctypes.byref(cpci), None, ctypes.byref(pipeline)), "vkCreateComputePipelines")
        F["vkDestroyShaderModule"][0](self.device, module, None)
        return Pipeline(self, pipeline, set_layout, layout, num_bindings)

    def _get_set(self, pipeline, buffers):
        key = (id(pipeline), tuple(b.handle.value for b in buffers))
        s = self._set_cache.get(key)
        if s is not None:
            return s
        if self._pool_sets_left == 0:
            psi = VkDescriptorPoolSize()
            psi.type = DESCRIPTOR_TYPE_STORAGE_BUFFER
            psi.descriptorCount = 256 * 8
            dpci = VkDescriptorPoolCreateInfo()
            dpci.sType = ST_DESCRIPTOR_POOL_CREATE_INFO
            dpci.maxSets = 256
            dpci.poolSizeCount = 1
            dpci.pPoolSizes = ctypes.pointer(psi)
            pool = H()
            _check(self._bind["vkCreateDescriptorPool"][0](self.device, ctypes.byref(dpci), None, ctypes.byref(pool)), "vkCreateDescriptorPool")
            self._pools.append(pool)
            self._current_pool = pool
            self._pool_sets_left = 256
        dsai = VkDescriptorSetAllocateInfo()
        dsai.sType = ST_DESCRIPTOR_SET_ALLOCATE_INFO
        dsai.descriptorPool = self._current_pool
        dsai.descriptorSetLayoutCount = 1
        layouts = (H * 1)(pipeline.set_layout)  # descriptor SET layout, not pipeline layout
        dsai.pSetLayouts = layouts
        s = H()
        _check(self._bind["vkAllocateDescriptorSets"][0](self.device, ctypes.byref(dsai), ctypes.byref(s)), "vkAllocateDescriptorSets")
        self._pool_sets_left -= 1
        infos = (VkDescriptorBufferInfo * len(buffers))()
        for i, b in enumerate(buffers):
            infos[i].buffer = b.handle
            infos[i].offset = 0
            infos[i].range = VK_WHOLE_SIZE
        writes = (VkWriteDescriptorSet * len(buffers))()
        for i, b in enumerate(buffers):
            writes[i].sType = ST_WRITE_DESCRIPTOR_SET
            writes[i].dstSet = s
            writes[i].dstBinding = i
            writes[i].descriptorCount = 1
            writes[i].descriptorType = DESCRIPTOR_TYPE_STORAGE_BUFFER
            writes[i].pBufferInfo = ctypes.cast(
                ctypes.byref(infos, ctypes.sizeof(VkDescriptorBufferInfo) * i),
                ctypes.POINTER(VkDescriptorBufferInfo))
        self._bind["vkUpdateDescriptorSets"][0](self.device, len(buffers), writes, 0, None)
        self._set_cache[key] = s
        return s

    # -- dispatch ----------------------------------------------------------
    def submit_jobs(self, jobs):
        """jobs: list of dicts {pipeline, buffers, push (bytes, <=32), gx, gy, gz}.
        Records all into one command buffer with memory barriers between, submits once."""
        F = self._bind
        d = self.device
        self._bind["vkResetCommandBuffer"][0](self.cmd, 0)
        bi = VkCommandBufferBeginInfo()
        bi.sType = ST_COMMAND_BUFFER_BEGIN_INFO
        bi.flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT
        _check(F["vkBeginCommandBuffer"][0](self.cmd, ctypes.byref(bi)), "vkBeginCommandBuffer")
        first = True
        for job in jobs:
            if not first:
                mb = VkMemoryBarrier()
                mb.sType = ST_MEMORY_BARRIER
                mb.srcAccessMask = ACCESS_SHADER_WRITE
                mb.dstAccessMask = ACCESS_SHADER_READ | ACCESS_SHADER_WRITE
                F["vkCmdPipelineBarrier"][0](self.cmd, PIPELINE_STAGE_COMPUTE, PIPELINE_STAGE_COMPUTE, 0,
                                             1, ctypes.byref(mb), 0, None, 0, None)
            s = self._get_set(job["pipeline"], job["buffers"])
            F["vkCmdBindPipeline"][0](self.cmd, PIPELINE_BIND_POINT_COMPUTE, job["pipeline"].handle)
            F["vkCmdBindDescriptorSets"][0](self.cmd, PIPELINE_BIND_POINT_COMPUTE, job["pipeline"].layout,
                                            0, 1, ctypes.pointer(s), 0, None)
            pb = job.get("push") or b"\x00" * 32
            pbuf = (ctypes.c_ubyte * len(pb)).from_buffer_copy(pb)
            F["vkCmdPushConstants"][0](self.cmd, job["pipeline"].layout, SHADER_STAGE_COMPUTE, 0, len(pb), pbuf)
            F["vkCmdDispatch"][0](self.cmd, job["gx"], job.get("gy", 1), job.get("gz", 1))
            first = False
        _check(F["vkEndCommandBuffer"][0](self.cmd), "vkEndCommandBuffer")
        si = VkSubmitInfo()
        si.sType = ST_SUBMIT_INFO
        si.commandBufferCount = 1
        si.pCommandBuffers = ctypes.pointer(self.cmd)
        _check(F["vkQueueSubmit"][0](self.queue, 1, ctypes.byref(si), None), "vkQueueSubmit")
        _check(F["vkQueueWaitIdle"][0](self.queue), "vkQueueWaitIdle")

    def info(self):
        api_maj = self.api_version >> 22
        api_min = (self.api_version >> 12) & 0x3FF
        return (f"{self.device_name} (api={api_maj}.{api_min}, type={self.device_type}, "
                f"queue_family={self.queue_family}, mem_type={self.memtype_default}, "
                f"fp16={self.fp16_enabled})")


# minimal struct for shader module creation (separate sType value space is fine — same enum)
ST_SHADER_MODULE_CREATE_INFO = 16


class VkShaderModuleCreateInfoStub(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", H), ("flags", ctypes.c_uint32),
                ("codeSize", ctypes.c_uint64), ("pCode", H)]
