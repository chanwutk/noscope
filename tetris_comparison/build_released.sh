#!/bin/sh
# Build the pinned NoScope executable in the legacy CUDA image.
set -eu
cd "$(dirname "$0")"
source_dir=$(cd "$1" && pwd)
build_dir=$2
mkdir -p "$build_dir"
build_dir=$(cd "$build_dir" && pwd)
[ "$(git -C "$source_dir" rev-parse HEAD)" = 1b4a96364a79924f5c127bd21f8379b909d86f46 ]
for patch in "$PWD/speedhax-replay.patch" "$PWD/speedhax-no-darknet.patch"; do
    if ! git -C "$source_dir" apply --reverse --check "$patch" 2>/dev/null; then
        git -C "$source_dir" apply --check "$patch"
        git -C "$source_dir" apply "$patch"
    fi
done
# Use HTTPS for the same Bazel archive. The upstream SHA-256 stays in force.
sed -i 's@http://www.sqlite.org/2017/sqlite-amalgamation-3200000.zip@https://www.sqlite.org/2017/sqlite-amalgamation-3200000.zip@' \
    "$source_dir/tensorflow/workspace.bzl"
# GCC already searches /usr/local/include. Bazel rejects the upstream machine's absolute -I paths.
sed -i '\@-I/lfs/1/ddkang/local/include@d; s@-L/lfs/1/ddkang/local/lib@-L/usr/local/lib@g' \
    "$source_dir/tensorflow/noscope/BUILD"
docker run --rm -v "$source_dir:/speedhax" -v "$build_dir:/build" -w /speedhax \
    --entrypoint sh noscope-released-inference -ec '
        export TF_NEED_CUDA=1 TF_CUDA_VERSION=8.0 CUDA_TOOLKIT_PATH=/usr/local/cuda
        export TF_CUDNN_VERSION=6 CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
        export TF_CUDA_COMPUTE_CAPABILITIES=6.0 TF_CUDA_CLANG=0
        yes "" | ./configure
        bazel --output_user_root=/build/bazel build --jobs=16 --config=cuda -c opt \
            --copt=-mavx2 //tensorflow/noscope:noscope
        cp -L bazel-bin/tensorflow/noscope/noscope /build/noscope
    '
