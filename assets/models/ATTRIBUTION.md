# Test asset attribution

## box.fbx

- **Source:** the [assimp](https://github.com/assimp/assimp) project's test-fixture corpus,
  `test/models/FBX/box.fbx`
  (https://raw.githubusercontent.com/assimp/assimp/master/test/models/FBX/box.fbx).
- **License:** assimp is distributed under the **3-clause BSD license**, which covers its bundled
  test models. See https://github.com/assimp/assimp/blob/master/LICENSE.
- **Contents:** a binary "Kaydara FBX Binary" file, **FBX version 7400**, exported by the Autodesk
  FBX SDK 2017.1. It contains a single `Geometry` (a cube): 24 per-face-corner vertices
  (`Vertices`, a RAW/uncompressed `double` array) and `PolygonVertexIndex` (a RAW `int32` array of
  12 triangles, each polygon delimited by a negative last index). Both geometry arrays use
  `encoding == 0` (uncompressed); the importer's clean-room zlib INFLATE path (`encoding == 1`,
  which most production FBX exporters use) is exercised separately by a round-trip vector in
  `tests/fbx_loader_test.cpp`.
- **Used by:** `tests/fbx_loader_test.cpp` (golden test for `engine/asset/fbx_loader.h`, issue #15 —
  the binary FBX mesh importer), via the `HF_BOX_FBX` compile definition in `tests/CMakeLists.txt`.
