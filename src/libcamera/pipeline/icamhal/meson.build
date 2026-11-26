# SPDX-License-Identifier: CC0-1.0

libcamhal_dep = dependency('libcamhal', required : true)

icamhal_sources = files([
    'icamhal.cpp',
])

# libcamera releases before 0.4 name the list libcamera_sources.
if is_variable('libcamera_internal_sources')
    libcamera_internal_sources += icamhal_sources
else
    libcamera_sources += icamhal_sources
endif

libcamera_deps += [libcamhal_dep]
