

def angle_between(u, v):
    num = np.dot(u, v)
    den = np.linalg.norm(u) * np.linalg.norm(v)
    cos_theta = num / den
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.rad2deg(np.arccos(cos_theta))

ST_parameters = {
    "gauche": {
        "step_length": [],
        "stride_length": [],
        "cycle_time": [],
        "cycle_frame": []
    },
    "droite": {
        "step_length": [],
        "stride_length": [],
        "cycle_time": [],
        "cycle_frame": []
    }
}

# Détermination d'un cycle de marche
plt.figure()
for side in ["gauche", "droite"]:
    start_frame, end_frame = np.sort(events[side]["HS"]["frame"])
    cycle_frame = end_frame + 1 - start_frame

    ST_parameters[side]["cycle_frame"].append(cycle_frame)
    ST_parameters[side]["cycle_time"].append(cycle_frame/freq)

    print(f"Durée du cycle de marche {side} : {cycle_frame/freq:.2f} s")

    print(start_frame, end_frame)