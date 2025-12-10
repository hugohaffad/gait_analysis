from config import *

import numpy as np
import matplotlib.pyplot as plt
import moveck_bridge_btk as btk

output_dir = REP
output_dir.mkdir(parents=True, exist_ok=True)

joints = ["Hip", "Knee", "Ankle"]

def sort_events(events):
    for side in events:
        for evt in ["HS", "TO"]:
            frames = np.array(events[side][evt]["frame"])
            times  = np.array(events[side][evt]["time"])

            order = np.argsort(frames)

            events[side][evt]["frame"] = frames[order].tolist()
            events[side][evt]["time"]  = times[order].tolist()

    return events

def normalize_cycle(signal, start_frame, end_frame, n_points=101):
    cycle = signal[start_frame:end_frame+1]
    n = len(cycle)

    x_old = np.linspace(0, 100, n)
    x_new = np.linspace(0, 100, n_points)

    cycle_norm = np.interp(x_new, x_old, cycle)

    return x_new, cycle_norm

def angle(v):
    theta = np.degrees(np.unwrap(np.arctan2(v[:, 1], v[:, 2]), period=2*np.pi))
    return theta

def kinematics_analysis(file, joints):

    h = btk.btkReadAcquisition(str(file))

    # ---- Markers ---- #
    markers, markersInfo = btk.btkGetMarkers(h)
    freq = btk.btkGetPointFrequency(h)

    # ---- Segments ---- #
    left_pelvis = markers["EIAS_G"] - markers["HANCHE_G"]
    right_pelvis = markers["EIAS_D"] - markers["HANCHE_D"]
    left_thigh = markers["GENOU_G"] - markers["HANCHE_G"]
    right_thigh = markers["GENOU_D"] - markers["HANCHE_D"]
    left_leg = markers["GENOU_G"] - markers["CHEVILLE_G"]
    right_leg = markers["GENOU_D"] - markers["CHEVILLE_D"]
    left_foot = markers["PIED_G"] - markers["CHEVILLE_G"]
    right_foot = markers["PIED_D"] - markers["CHEVILLE_D"]

    # ---- Angles ---- #
    left_hip = (180 - (angle(left_thigh) - angle(left_pelvis))) % 360
    left_hip = np.where(left_hip > 180, left_hip - 360, left_hip)
    right_hip = (180 - (angle(right_thigh) - angle(right_pelvis))) % 360
    right_hip = np.where(right_hip > 180, right_hip - 360, right_hip)

    left_knee = (180 - (angle(left_thigh) - angle(left_leg))) % 360
    left_knee = np.where(left_knee > 180, left_knee - 360, left_knee)
    right_knee = (180 - (angle(right_thigh) - angle(right_leg))) % 360
    right_knee = np.where(right_knee > 180, right_knee - 360, right_knee)

    left_ankle = 90 - (angle(left_foot) - angle(left_leg))
    left_ankle = np.where(left_ankle > 180, left_ankle - 360, left_ankle)
    right_ankle = 90 - (angle(right_foot) - angle(right_leg))
    right_ankle = np.where(right_ankle > 180, right_ankle - 360, right_ankle)

    angles = {
        "Hip": {"left": left_hip, "right": right_hip},
        "Knee": {"left": left_knee, "right": right_knee},
        "Ankle": {"left": left_ankle, "right": right_ankle},
    }

    # ---- Forceplates ---- #
    forceplates, forceplatesInfo = btk.btkGetForcePlatforms(h)
    grw = btk.btkGetGroundReactionWrenches(h)
    fp_freq = btk.btkGetAnalogFrequency(h)
    active_forceplates = []

    # ---- Events ---- #
    events = {
        "left": {"HS": {"frame": [], "time": []},
                 "TO": {"frame": [], "time": []}},
        "right": {"HS": {"frame": [], "time": []},
                  "TO": {"frame": [], "time": []}},
    }

    for i in range(len(forceplates)):
        if np.count_nonzero(grw[i]["F"]) == 0:
            continue
        else:
            active_forceplates.append(i)

    for i in active_forceplates:
        GRFz = grw[i]["F"][:, 2]

        HS_idx = np.where(GRFz > 5)[0][0]
        TO_idx = np.where(GRFz > 5)[0][-1]

        HS_time = HS_idx / fp_freq
        TO_time = TO_idx / fp_freq

        HS_frame = int(np.round(HS_time * freq))
        TO_frame = int(np.round(TO_time * freq))

        if markers["CHEVILLE_D"][HS_frame, 2] < markers["CHEVILLE_G"][HS_frame, 2]:
            side = "right"
        else:
            side = "left"

        events[side]["HS"]["frame"].append(HS_frame)
        events[side]["HS"]["time"].append(HS_time)
        events[side]["TO"]["frame"].append(TO_frame)
        events[side]["TO"]["time"].append(TO_time)

    events = sort_events(events)

    if len(events["right"]["HS"]["frame"]) < 2 or len(events["left"]["HS"]["frame"]) < 2:
        raise ValueError(
            f"Pas assez de HS dans {file} : "
            f"right={len(events['right']['HS']['frame'])}, "
            f"left={len(events['left']['HS']['frame'])}"
        )

    start_R, end_R = events["right"]["HS"]["frame"][0], events["right"]["HS"]["frame"][1]
    start_L, end_L = events["left"]["HS"]["frame"][0], events["left"]["HS"]["frame"][1]

    # ---- Normalisation des cycles ---- #
    angles_norm = {"right": {}, "left": {}}
    gc = None

    for joint in joints:
        # droite
        sig_R = angles[joint]["right"]
        gc_tmp, sig_R_norm = normalize_cycle(sig_R, start_R, end_R)

        # gauche
        sig_L = angles[joint]["left"]
        _, sig_L_norm = normalize_cycle(sig_L, start_L, end_L)

        angles_norm["right"][joint] = sig_R_norm
        angles_norm["left"][joint]  = sig_L_norm

        if gc is None:
            gc = gc_tmp   # on fixe le vecteur 0–100 % une fois pour toutes

    results = {
        "gc": gc,
        "right": angles_norm["right"],  # dict joint -> courbe normalisée
        "left":  angles_norm["left"],
    }

    return results

def compute_group_curves(dir_path, joints):
    all_curves = {joint: {"right": [], "left": []} for joint in joints}
    gc = None

    for file in dir_path.iterdir():
        if file.suffix.lower() != ".c3d":
            continue  # au cas où il y ait autre chose dans le dossier

        res = kinematics_analysis(file, joints=joints)

        if gc is None:
            gc = res["gc"]

        for joint in joints:
            all_curves[joint]["right"].append(res["right"][joint])
            all_curves[joint]["left"].append(res["left"][joint])

    return gc, all_curves

def plot_group_curves(gc, all_curves, joints, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = axes.flatten()

    for i, joint in enumerate(joints):
        ax = axes[i]

        for side, color, label in [("right", "tab:blue", "Right"),
                                   ("left",  "tab:orange", "Left")]:

            data = np.vstack(all_curves[joint][side])
            mean_curve = data.mean(axis=0)
            std_curve  = data.std(axis=0)

            ax.plot(gc, mean_curve, label=label, linewidth=2)
            ax.fill_between(gc,
                            mean_curve - std_curve,
                            mean_curve + std_curve,
                            alpha=0.2)

        ax.set_title(joint, fontsize=10, fontweight="bold")
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("Angle (°)")
        ax.set_xlim(0, 100)
        ax.grid(True)

    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(str(output_dir / filename), dpi=300)
    plt.close()

# ---- Healthy ---- #
gc_hea, curves_hea = compute_group_curves(HEA, joints)
plot_group_curves(gc_hea, curves_hea, joints,
                  title="Healthy gait",
                  filename="joint_angles_healthy.png")

# ---- Impaired ---- #
gc_imp, curves_imp = compute_group_curves(IMP, joints)
plot_group_curves(gc_imp, curves_imp, joints,
                  title="Impaired gait",
                  filename="joint_angles_impaired.png")