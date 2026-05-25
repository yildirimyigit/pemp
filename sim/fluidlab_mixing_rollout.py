"""Shared FluidLab Mixing rollout + video, used by the PEMP/CNMP test scripts
(fluidlab_mixing_pemp_test.py, fluidlab_mixing_bare_test.py).

Build ONE env per process and reset() between trajectories -- taichi accumulates
GPU fields across gym.make calls, so remaking the env per frequency would OOM.

Two renderers:
  * matplotlib (default) -- top-down particle scatter, Vulkan-free, works anywhere.
  * GGUI (--render ggui)  -- FluidLab's native 3D renderer (cup + stirrer + lights).
    Needs a Vulkan device reporting apiVersion <= 1.3 (taichi 1.1.0's VMA asserts on
    1.4).  Validate the machine first with sim/ggui_smoke.py.
The behavior is identical either way; only the picture differs.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ACT_LIMIT = 0.007  # MixingEnv.action_range

# Mixing GGUI camera (from the commented block in fluidlab/envs/mixing_env.py)
GGUI_CAM = dict(
    res=(960, 960),
    camera_pos=(-0.15, 2.82, 2.5),
    camera_lookat=(0.5, 0.5, 0.5),
    fov=30,
    lights=[{"pos": (0.5, 1.5, 0.5), "color": (0.5, 0.5, 0.5)},
            {"pos": (0.5, 1.5, 1.5), "color": (0.5, 0.5, 0.5)}],
)


def make_env():
    """Build the Mixing env once.  Skips the GL renderer by pretending we're
    'on server' so build_env() won't try to instantiate it (we set up our own
    renderer, or render from particle state)."""
    import fluidlab.utils.misc as _misc
    _misc.is_on_server = lambda: True

    import gym
    import fluidlab.envs  # noqa: F401  (registers the gym ids)

    env = gym.make("Mixing-v0", seed=0, loss=False)
    env.reset()
    print("[sim] Mixing env built; action_dim =", env.action_space.shape[0])
    return env


def set_start_offset(env, dx=0.0, dz=0.0):
    """After env.reset(), relocate the stirrer's start by (dx, 0, dz) in cup coords.

    The stir actions are velocities integrated from this start, so shifting the start
    shifts where the swirl forms -- a phase-independent spatial randomness source that
    leaves the action trajectory (and thus training) untouched.  The sim's cylinder
    boundary clamps the stirrer, so large offsets just saturate at the cup wall.
    Call AFTER env.reset() (reset restores the default start) and BEFORE stepping.
    """
    if dx == 0.0 and dz == 0.0:
        return
    ag = env.taichi_env.agent
    s = [np.array(e).copy() for e in ag.get_state(0)]
    s[0][0] += dx
    s[0][2] += dz
    ag.set_state(0, s)


def _save(fig, update, n_frames, out_path, fps):
    """Render an animation to mp4 (ffmpeg) with a .gif fallback."""
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000.0 / fps, blit=False)
    try:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
        saved = out_path
    except Exception as e:
        saved = os.path.splitext(out_path)[0] + ".gif"
        print(f"[render] ffmpeg unavailable ({e.__class__.__name__}); writing gif instead")
        anim.save(saved, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[render] wrote {saved}")
    return saved


def rollout_and_render(env, actions, out_path, stride=2, fps=25, subsample=5, title="",
                       start_offset=(0.0, 0.0)):
    """Reset the env, step `actions`, render the latte top-down with matplotlib."""
    from fluidlab.configs.macros import COFFEE_VIS

    env.reset()
    set_start_offset(env, *start_offset)
    te = env.taichi_env
    mat = te.simulator.particles_i.mat.to_numpy()
    sub = np.arange(0, mat.shape[0], subsample)
    is_cof = mat[sub] == COFFEE_VIS
    coffee_rgb, milk_rgb = np.array([0.29, 0.17, 0.07]), np.array([0.95, 0.92, 0.85])
    colors = np.where(is_cof[:, None], coffee_rgb, milk_rgb)

    states = []
    for t in range(len(actions)):
        te.step(np.clip(actions[t], -ACT_LIMIT, ACT_LIMIT))
        if t % stride == 0:
            states.append(te.get_state()["state"]["x"][sub].copy())
    print(f"[sim] stepped {len(actions)} actions, captured {len(states)} frames")

    fig = plt.figure(figsize=(5, 5), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0.05, 0.95)
    ax.set_aspect("equal"); ax.axis("off")
    order0 = np.argsort(states[0][:, 1])
    scat = ax.scatter(states[0][order0, 0], states[0][order0, 2],
                      s=3.0, c=colors[order0], linewidths=0)
    txt = ax.set_title("")

    def update(i):
        x = states[i]
        order = np.argsort(x[:, 1])  # low particles first, surface drawn on top
        scat.set_offsets(np.c_[x[order, 0], x[order, 2]])
        scat.set_facecolors(colors[order])
        txt.set_text(f"{title}  step {i * stride}")
        return scat, txt

    return _save(fig, update, len(states), out_path, fps)


def rollout_and_render_ggui(env, actions, out_path, stride=2, fps=25, title="",
                            start_offset=(0.0, 0.0)):
    """Reset the env, step `actions`, render with FluidLab's native GGUI renderer.

    Requires a Vulkan device with apiVersion <= 1.3 (else taichi 1.1.0 core-dumps in
    VMA at window init).  Set up the renderer once per env and reuse it.
    """
    te = env.taichi_env
    if te.renderer is None:
        from fluidlab.fluidengine.renderers.ggui_renderer import GGUIRenderer
        te.renderer = GGUIRenderer(mode="rgb_array", **GGUI_CAM)
        te.renderer.build(te.simulator, te.particles)

    env.reset()
    set_start_offset(env, *start_offset)
    frames = []
    for t in range(len(actions)):
        te.step(np.clip(actions[t], -ACT_LIMIT, ACT_LIMIT))
        if t % stride == 0:
            frames.append(te.render(mode="rgb_array"))
    print(f"[sim] stepped {len(actions)} actions, captured {len(frames)} GGUI frames")

    h, w = frames[0].shape[:2]
    fig = plt.figure(figsize=(w / 160, h / 160), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    im = ax.imshow(frames[0])
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  color="black", fontsize=11)

    def update(i):
        im.set_data(frames[i])
        txt.set_text(f"{title}  step {i * stride}")
        return im, txt

    return _save(fig, update, len(frames), out_path, fps)
