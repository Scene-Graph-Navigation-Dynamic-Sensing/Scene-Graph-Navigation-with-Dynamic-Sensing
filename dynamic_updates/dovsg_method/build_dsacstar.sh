#!/usr/bin/env bash
# Builds ACE's `dsacstar` extension (RANSAC PnP for ACE poses) into
# DovSG/third_party/dsacstar_ext (read by run_dovsg_update.py in the
# scene_graph env) and into DoVSG's .venv.
# Upstream's ace/dsacstar/setup.py expects OpenCV from a conda env; there is none
# on this machine, so a minimal static OpenCV (core, imgproc, features2d, flann,
# calib3d) is built from source into DovSG/third_party/opencv_min first.
set -euo pipefail
DOVSG_ROOT=${DOVSG_ROOT:-/home/ravindu/SceneGraphNav/external/DovSG}
O=$DOVSG_ROOT/third_party/opencv_min
WORK=$(mktemp -d)

if [ ! -f "$O/lib/libopencv_calib3d.a" ]; then
    curl -sL https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz | tar xz -C "$WORK"
    cmake -S "$WORK/opencv-4.10.0" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_LIST=core,imgproc,features2d,flann,calib3d \
        -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF -DWITH_OPENEXR=OFF \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_ZLIB=ON \
        -DCMAKE_INSTALL_PREFIX="$O"
    make -C "$WORK/build" -j16 install
fi

cp "$DOVSG_ROOT"/ace/dsacstar/*.cpp "$DOVSG_ROOT"/ace/dsacstar/*.h "$WORK/"
cat > "$WORK/setup.py" <<EOF
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
O = "$O"
libs = [f"{O}/lib/libopencv_{m}.a" for m in ("calib3d", "features2d", "flann", "imgproc", "core")] \
    + [f"{O}/lib/opencv4/3rdparty/libzlib.a"]
setup(name="dsacstar", ext_modules=[CppExtension(
    name="dsacstar", sources=["dsacstar.cpp", "thread_rand.cpp"], include_dirs=[f"{O}/include/opencv4"],
    extra_compile_args=["-fopenmp"], extra_link_args=libs + ["-fopenmp", "-ldl", "-lpthread"])],
    cmdclass={"build_ext": BuildExtension})
EOF
(cd "$WORK" && "$DOVSG_ROOT/.venv/bin/python" setup.py build_ext --inplace)
mkdir -p "$DOVSG_ROOT/third_party/dsacstar_ext"
cp "$WORK"/dsacstar*.so "$DOVSG_ROOT/third_party/dsacstar_ext/"
cp "$WORK"/dsacstar*.so "$DOVSG_ROOT/.venv/lib/python3.10/site-packages/"
"$DOVSG_ROOT/.venv/bin/python" -c "import torch, dsacstar; print('dsacstar ok')"
