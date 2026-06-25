// Unit test for the UMG-Designer WIDGET-TREE EDITOR view + edit data models (engine/editor/
// widget_editor_data.h + engine/editor/widget_edit_ops.h, issue #30 — the GUI half of the UMG-class UI
// framework). Pure CPU (hf_core), ASan-eligible like the other pure tests, NO image / NO render-bake.
//
// SELF-CONTAINED: the scaffolding (check() + HF_TEST_MAIN_INIT) mirrors seq_editor_test.cpp /
// flow_editor_test.cpp / profiler_view_test.cpp so this compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/widget_editor_test.cpp` on the Mac — the cheap cross-platform
// proof. The WHOLE view is INTEGER (the hierarchy rows at a fixed depth*indent grid, the layout-preview
// boxes = the integer SolveLayout rects scaled by an integer ratio, the inspector = the raw int32 Style
// fields) so the view — and hence DigestWidgetEditorView (FNV-1a-64) over it — is bit-identical run-to-run
// AND platform-to-platform (MSVC == Apple clang). The golden is a PINNED FNV-1a-64 value IN the test.
//
// The tree is hf::ui::MakeLayoutShowcase(): the FIXED 7-widget UMG fixture (root vertical stack -> header /
// body / footer; body [horizontal stack] -> left / right; header -> title). So: 7 hierarchy rows, 7
// layout-preview boxes, and (for a valid selection) kInspCount inspector fields.
//
// What this pins:
//   (a) row count == widget count (7) + a hand-checked pre-order order (root, header, title, body, left,
//       right, footer) + depths;
//   (b) box count == widget count (7) + the root box == the scaled viewport;
//   (c) the inspector carries the selected (root) widget's Style fields (kInspCount rows, root pad==8);
//   (d) DigestWidgetEditorView(view) == a hard-pinned uint64 (the cross-platform proof);
//   (e) re-building the view is bit-identical (deterministic / replay-stable);
//   (f) an AddChildWidget edit grows the rows/boxes + changes the digest DETERMINISTICALLY, and the edit is
//       itself replay-stable; a SetWidgetStyleProp edit changes the inspector + the digest; DeleteWidget
//       removes the subtree + compacts indices (rows/boxes shrink) + keeps the tree valid.

#include "editor/widget_editor_data.h"
#include "editor/widget_edit_ops.h"
#include "ui/widget.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::editor;
using hf::ui::Tree;
using hf::ui::Style;
using hf::ui::Rect;
using hf::ui::WidgetId;
using hf::ui::MakeLayoutShowcase;
using hf::ui::ChildCount;
using hf::ui::kStackH;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

int main() {
    HF_TEST_MAIN_INIT();

    const Tree tree = MakeLayoutShowcase();
    const WidgetEditorLayout L;                  // default fixed grid
    const Rect viewport{ 0, 0, 1280, 720 };
    const WidgetId selected = tree.root;         // select the root for the inspector

    const WidgetEditorView view = BuildWidgetEditorView(tree, viewport, selected, L);

    std::printf("widget-editor: widgets=%zu rows=%zu boxes=%zu inspector=%zu selected=%u\n",
                tree.widgets.size(), view.rows.size(), view.boxes.size(),
                view.inspector.size(), view.selected);

    // ---- (a) row count == widget count + pre-order order + depths. ------------------------------------
    check(view.rows.size() == tree.widgets.size() && view.rows.size() == 7,
          "widget-editor: one hierarchy row per widget (7 rows)");
    if (view.rows.size() == 7) {
        // MakeLayoutShowcase ids: 0 root, 1 header, 2 body, 3 footer, 4 left, 5 right, 6 title.
        // Pre-order via firstChild/nextSibling: root(0) -> header(1) -> title(6) -> body(2) -> left(4)
        //   -> right(5) -> footer(3).
        const WidgetId expectIds[7]  = { 0, 1, 6, 2, 4, 5, 3 };
        const int      expectDepth[7] = { 0, 1, 2, 1, 2, 2, 1 };
        bool order = true, depths = true;
        for (int i = 0; i < 7; ++i) {
            if (view.rows[i].id    != expectIds[i])    order  = false;
            if (view.rows[i].depth != expectDepth[i]) depths = false;
        }
        check(order,  "widget-editor: rows are pre-order (root,header,title,body,left,right,footer)");
        check(depths, "widget-editor: row depths follow the nesting (0,1,2,1,2,2,1)");
        check(view.rows[0].selected && !view.rows[1].selected,
              "widget-editor: the selected (root) row is flagged, others are not");
        check(view.rows[0].x == L.hierX && view.rows[0].y == L.hierY,
              "widget-editor: root row at (hierX, hierY)");
        check(view.rows[2].x == L.hierX + 2 * L.indentW,
              "widget-editor: depth-2 row (title) indented 2 levels");
        check(view.rows[1].y == L.hierY + 1 * L.rowH,
              "widget-editor: row 1 one rowH below row 0");
    }

    // ---- (b) box count == widget count + the root box == the scaled viewport. -------------------------
    check(view.boxes.size() == tree.widgets.size() && view.boxes.size() == 7,
          "widget-editor: one layout-preview box per widget (7 boxes)");
    if (view.boxes.size() == 7) {
        // Boxes are in widget-INDEX order (box[0] == root). The root's SolveLayout rect IS the viewport,
        // scaled by previewNum/previewDen (1/2) and offset by (previewX, previewY).
        const WidgetLayoutBox& rb = view.boxes[0];
        check(rb.id == 0 && rb.selected,
              "widget-editor: box[0] is the root + flagged selected");
        check(rb.x == L.previewX + (viewport.x * L.previewNum) / L.previewDen &&
              rb.y == L.previewY + (viewport.y * L.previewNum) / L.previewDen &&
              rb.w == (viewport.w * L.previewNum + L.previewDen / 2) / L.previewDen &&
              rb.h == (viewport.h * L.previewNum + L.previewDen / 2) / L.previewDen,
              "widget-editor: root box == the viewport scaled by previewNum/previewDen + offset");
    }

    // ---- (c) the inspector carries the selected (root) widget's Style. --------------------------------
    check(view.inspector.size() == static_cast<std::size_t>(kInspCount),
          "widget-editor: inspector has kInspCount Style rows for the selection");
    if (view.inspector.size() == static_cast<std::size_t>(kInspCount)) {
        // The root's Style: pad{L,T,R,B}=8, everything else 0, flexWeight=1, flags=0, kind=0.
        const Style& rs = tree.widgets[tree.root].style;
        check(view.inspector[kInspWidth].prop  == kInspWidth  &&
              view.inspector[kInspWidth].value == rs.width,
              "widget-editor: inspector[0] is width with the root's value");
        check(view.inspector[kInspPadL].value == rs.padL && rs.padL == 8,
              "widget-editor: inspector reports the root padL == 8");
        check(view.inspector[kInspFlexWeight].value == static_cast<int32_t>(rs.flexWeight),
              "widget-editor: inspector reports the root flexWeight");
    }

    // ---- (d) PINNED DIGEST — the cross-platform make-or-break (identical on MSVC + clang). ------------
    const uint64_t digest = DigestWidgetEditorView(view);
    std::printf("widget-editor: view digest = 0x%016llx\n", (unsigned long long)digest);
    const uint64_t kPinnedDigest = 0xc208ea031ced1608ull;  // PINNED on first run (MSVC == clang)
    check(digest == kPinnedDigest,
          "widget-editor: DigestWidgetEditorView(view) == pinned uint64 (the cross-platform proof)");

    // ---- (e) REPLAY-STABLE — re-building the same view reproduces the digest. -------------------------
    check(DigestWidgetEditorView(BuildWidgetEditorView(tree, viewport, selected, L)) == digest,
          "widget-editor: re-building the view is bit-identical (deterministic)");

    // ---- (f1) an AddChildWidget edit grows the rows/boxes + changes the digest deterministically. -----
    // Add a fixed-size text child (kind 1) under the body (id 2). This adds a hierarchy row + a layout box
    // and reshapes the layout -> the view digest MUST change, reproducibly.
    {
        Tree edited = MakeLayoutShowcase();
        Style childS; childS.width = 40; childS.height = 20;
        childS.flags = hf::ui::kFixedW | hf::ui::kFixedH;
        const WidgetId added = AddChildWidget(edited, /*parent body=*/2, childS, /*kind text=*/1);
        check(added == 7 && edited.widgets.size() == 8 && ChildCount(edited, 2) == 3,
              "widget-editor: AddChildWidget appends id 7 under body (body now 3 children)");

        const WidgetEditorView ev = BuildWidgetEditorView(edited, viewport, selected, L);
        const uint64_t editedDigest = DigestWidgetEditorView(ev);
        std::printf("widget-editor: after AddChild rows=%zu boxes=%zu digest=0x%016llx\n",
                    ev.rows.size(), ev.boxes.size(), (unsigned long long)editedDigest);
        check(ev.rows.size() == 8 && ev.boxes.size() == 8,
              "widget-editor: AddChildWidget grows rows+boxes by one (7 -> 8)");
        check(editedDigest != digest, "widget-editor: AddChildWidget changes the view digest");
        // Replay-stable: re-applying the SAME edit to a fresh tree reproduces the edited digest.
        Tree edited2 = MakeLayoutShowcase();
        AddChildWidget(edited2, 2, childS, 1);
        check(DigestWidgetEditorView(BuildWidgetEditorView(edited2, viewport, selected, L)) == editedDigest,
              "widget-editor: the AddChildWidget edit is replay-stable (same digest)");
    }

    // ---- (f2) a SetWidgetStyleProp edit changes the inspector + the digest. ----------------------------
    {
        Tree edited = MakeLayoutShowcase();
        // Bump the root's padL from 8 -> 24; the inspector row + the layout (root content box) change.
        SetWidgetStyleProp(edited, edited.root, kWSPPadL, 24);
        check(edited.widgets[edited.root].style.padL == 24,
              "widget-editor: SetWidgetStyleProp writes root padL = 24");
        const WidgetEditorView ev = BuildWidgetEditorView(edited, viewport, selected, L);
        check(ev.inspector[kInspPadL].value == 24,
              "widget-editor: the inspector reflects the new padL");
        check(DigestWidgetEditorView(ev) != digest,
              "widget-editor: SetWidgetStyleProp changes the view digest");
        // A negative flexWeight clamps to 0 (the uint32 SetProp discipline).
        SetWidgetStyleProp(edited, 4 /*left*/, kWSPFlexWeight, -5);
        check(edited.widgets[4].style.flexWeight == 0u,
              "widget-editor: SetWidgetStyleProp clamps a negative flexWeight to 0");
    }

    // ---- (f3) DeleteWidget removes the subtree, compacts indices, keeps the tree valid. ---------------
    {
        Tree edited = MakeLayoutShowcase();
        // Delete the body (id 2), whose subtree is body+left+right (3 widgets). 7 -> 4 widgets.
        const bool ok = DeleteWidget(edited, 2);
        check(ok && edited.widgets.size() == 4,
              "widget-editor: DeleteWidget(body) removes body+left+right (7 -> 4 widgets)");
        // The survivors: root, header, title, footer. The root now has 2 children (header, footer).
        check(ChildCount(edited, edited.root) == 2,
              "widget-editor: after deleting body the root has 2 children (header, footer)");
        // Indices stayed valid (every parent/child link in range or kNoWidget) — exercise the view builder.
        const WidgetEditorView ev = BuildWidgetEditorView(edited, viewport, edited.root, L);
        check(ev.rows.size() == 4 && ev.boxes.size() == 4,
              "widget-editor: the compacted tree yields 4 rows + 4 boxes");
        check(DigestWidgetEditorView(ev) != digest,
              "widget-editor: DeleteWidget changes the view digest");
        // Out-of-range delete is a no-op false.
        check(!DeleteWidget(edited, 999), "widget-editor: DeleteWidget out-of-range returns false");
    }

    if (g_fail == 0) std::printf("ALL PASS\n");
    else             std::printf("%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
