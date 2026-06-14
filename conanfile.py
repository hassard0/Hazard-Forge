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

    def layout(self):
        cmake_layout(self)
