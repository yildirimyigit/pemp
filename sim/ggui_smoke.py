"""GGUI / Vulkan smoke test -- run this FIRST on any candidate machine before the
full GGUI rollout.  It inits taichi exactly like FluidLab (CUDA compute) and opens
an OFFSCREEN GGUI window, renders one frame, and writes /tmp/ggui_smoke.png.

  ~/sw/anaconda3/envs/fluidlab/bin/python sim/ggui_smoke.py

PASS  -> prints "GGUI OFFSCREEN OK" + saves the png  => the full GGUI render will work:
          python sim/fluidlab_mixing_pemp_test.py  --run <dir> --render ggui
          python sim/fluidlab_mixing_bare_test.py  --run <dir> --render ggui
FAIL  -> taichi logs the picked device's Vulkan version, then core-dumps in VMA if
          that device reports apiVersion > 1.3.  Look at the "supports Vulkan ...
          version 1.x.y" line: you need <= 1.3 on the NVIDIA (compute) GPU.  If it's
          1.4, this machine won't do GGUI with taichi 1.1.0 -- try an older-driver box.
"""
import numpy as np
import taichi as ti

ti.init(arch=ti.cuda)  # FluidLab uses arch=gpu (cuda) for compute; GGUI uses Vulkan

try:
    win = ti.ui.Window("smoke", (256, 256), show_window=False)  # offscreen
    canvas = win.get_canvas(); canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    cam = ti.ui.Camera(); cam.position(0.5, 1.2, 2.0); cam.lookat(0.5, 0.5, 0.5); cam.fov(40)
    scene.set_camera(cam)
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    pts = ti.Vector.field(3, ti.f32, shape=(2000,))
    pts.from_numpy((np.random.rand(2000, 3) * 0.4 + 0.3).astype(np.float32))
    scene.particles(pts, color=(0.3, 0.17, 0.07), radius=0.01)
    canvas.scene(scene)
    img = np.rot90(win.get_image_buffer_as_numpy())[:, :, :3]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.imsave("/tmp/ggui_smoke.png", (img * 255).astype(np.uint8))
    print("GGUI OFFSCREEN OK -> /tmp/ggui_smoke.png  (this machine can do --render ggui)")
except Exception as e:
    print("GGUI FAILED:", type(e).__name__, str(e)[:300])
    print("If taichi core-dumped before this line, the picked Vulkan device is > 1.3.")
