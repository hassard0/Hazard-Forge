from conan import ConanFile
from conan.tools.cmake import cmake_layout


class HazardForge(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("sdl/3.2.6")
        self.requires("vk-bootstrap/1.4.350")
        self.requires("vulkan-memory-allocator/3.3.0")
        self.requires("vulkan-headers/1.4.350.0")
        self.requires("vulkan-loader/1.4.350.0")
        # Khronos validation layer (Slice AS oracle / Slice AT permanent gate). NOT a link-time
        # dependency: it provides VkLayer_khronos_validation.{dll,json}, loaded at RUNTIME when
        # VK_LAYER_PATH points at this package's bin dir and VK_LAYER_KHRONOS_validation is enabled.
        # Pulled here so scripts/verify.ps1's Vulkan-validation smoke gate can actually load the layer
        # (it is NOT installed system-wide on the bench box; without this the oracle is silently
        # inactive and a validation regression would slip through unnoticed).
        self.requires("vulkan-validationlayers/1.4.350.0")

    def layout(self):
        cmake_layout(self)
