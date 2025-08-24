import math

def is_between(val, low, high):
    return low <= val <= high

# A helper function to check if the user is in a 'start' or 'end' position
def get_state(angle, up_threshold, down_threshold):
    if angle >= up_threshold:
        return "up"
    elif angle <= down_threshold:
        return "down"
    return "mid"

def check_squat(angles, state, reps):
    left_knee_angle = angles.get("left_knee_angle", 180)
    right_knee_angle = angles.get("right_knee_angle", 180)
    knee_angle = min(left_knee_angle, right_knee_angle)

    left_hip_angle = angles.get("left_hip_angle", 180)
    right_hip_angle = angles.get("right_hip_angle", 180)
    hip_angle = min(left_hip_angle, right_hip_angle)

    # Posture check for squats: back straight (torso angle) and knees tracking over toes (hip angle)
    torso_angle = angles.get("torso_angle", 180)
    posture_correct = is_between(torso_angle, 150, 190) and is_between(hip_angle, 70, 180) and is_between(knee_angle, 60, 180)

    # Repetition logic for squats
    if state == "up" and knee_angle < 100 and hip_angle < 100:
        state = "down"
    elif state == "down" and knee_angle > 160 and hip_angle > 160:
        reps += 1
        state = "up"

    return posture_correct, state, reps

def check_pushup(angles, state, reps):
    left_elbow_angle = angles.get("left_elbow_angle", 180)
    right_elbow_angle = angles.get("right_elbow_angle", 180)
    elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
    
    torso_angle = angles.get("torso_angle", 180)
    hip_angle = (angles.get("left_hip_angle", 180) + angles.get("right_hip_angle", 180)) / 2

    # Posture check: body should be in a straight line (low variation in hip, torso angles)
    posture_correct = is_between(hip_angle, 160, 190) and is_between(torso_angle, 160, 190)

    # Repetition logic for pushups
    if state == "up" and elbow_angle < 90:
        state = "down"
    elif state == "down" and elbow_angle > 160:
        reps += 1
        state = "up"

    return posture_correct, state, reps

def check_hammer_curl(angles, state, reps):
    left_elbow_angle = angles.get("left_elbow_angle", 180)
    right_elbow_angle = angles.get("right_elbow_angle", 180)
    elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
    
    # Check if shoulders are stable (staying at a consistent angle)
    left_shoulder_angle = angles.get("left_shoulder_angle", 0)
    right_shoulder_angle = angles.get("right_shoulder_angle", 0)
    posture_correct = is_between(left_shoulder_angle, 10, 40) and is_between(right_shoulder_angle, 10, 40)

    # Repetition logic for hammer curls
    if state == "up" and elbow_angle < 60:
        state = "down"
    elif state == "down" and elbow_angle > 150:
        reps += 1
        state = "up"

    return posture_correct, state, reps

def check_deadlift(angles, state, reps):
    left_hip_angle = angles.get("left_hip_angle", 180)
    right_hip_angle = angles.get("right_hip_angle", 180)
    hip_angle = (left_hip_angle + right_hip_angle) / 2

    torso_angle = angles.get("torso_angle", 180)
    
    # Posture check: A straight back is key. Hip and torso angles should not drop excessively.
    posture_correct = is_between(torso_angle, 160, 190) and is_between(hip_angle, 70, 180)

    # Repetition logic for deadlifts
    if state == "up" and hip_angle < 100:
        state = "down"
    elif state == "down" and hip_angle > 160:
        reps += 1
        state = "up"

    return posture_correct, state, reps

def check_lateral_raise(angles, state, reps):
    left_shoulder_angle = angles.get("left_shoulder_angle", 0)
    right_shoulder_angle = angles.get("right_shoulder_angle", 0)
    avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2

    left_elbow_angle = angles.get("left_elbow_angle", 180)
    right_elbow_angle = angles.get("right_elbow_angle", 180)
    
    # Posture check: Elbows should be mostly straight, and shoulders shouldn't go too high
    posture_correct = is_between(left_elbow_angle, 150, 180) and is_between(right_elbow_angle, 150, 180) and avg_shoulder_angle < 100

    # Repetition logic for lateral raises
    if state == "down" and avg_shoulder_angle > 70:
        state = "up"
    elif state == "up" and avg_shoulder_angle < 20:
        reps += 1
        state = "down"
    
    return posture_correct, state, reps

def check_russian_twist(angles, state, reps):
    # This logic needs to be more complex to track side-to-side movement
    torso_angle = angles.get("torso_angle", 90)
    left_shoulder_angle = angles.get("left_shoulder_angle", 0)
    right_shoulder_angle = angles.get("right_shoulder_angle", 0)
    
    # The hip angle should remain relatively stable and bent to signify a seated position.
    hip_angle = (angles.get("left_hip_angle", 180) + angles.get("right_hip_angle", 180)) / 2

    # Posture check: torso should be leaned back at a consistent angle (not changing during the rep)
    posture_correct = is_between(hip_angle, 70, 110)

    # State tracking: "center" -> "left" -> "center" -> "right" -> "center"
    if state == "start":
        if abs(left_shoulder_angle - right_shoulder_angle) > 30: # Detect initial twist
            state = "twisted" if left_shoulder_angle > right_shoulder_angle else "twisted"
    elif state == "twisted":
        if abs(left_shoulder_angle - right_shoulder_angle) < 10: # Detect return to center
            reps += 0.5
            state = "start"

    return posture_correct, state, reps

def check_shoulder_press(angles, state, reps):
    left_elbow_angle = angles.get("left_elbow_angle", 180)
    right_elbow_angle = angles.get("right_elbow_angle", 180)
    avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

    torso_angle = angles.get("torso_angle", 180)
    
    # Posture check: Core should be stable (no significant forward/backward lean)
    posture_correct = is_between(torso_angle, 160, 190)

    # Repetition logic for shoulder press
    if state == "down" and avg_elbow_angle > 160:
        state = "up"
    elif state == "up" and avg_elbow_angle < 90:
        reps += 1
        state = "down"

    return posture_correct, state, reps

def check_plank(angles, state, reps):
    hip_angle = (angles.get("left_hip_angle", 180) + angles.get("right_hip_angle", 180)) / 2
    
    # Posture check: A straight line from shoulders to ankles is ideal (hip angle near 180)
    posture_correct = is_between(hip_angle, 160, 190)

    # Reps are not counted for plank; this is a static hold.
    # The 'state' can remain 'up' to signify a hold.
    return posture_correct, state, reps

def check_bench_press(angles, state, reps):
    elbow_angle = (angles.get("left_elbow_angle", 180) + angles.get("right_elbow_angle", 180)) / 2
    
    left_shoulder_angle = angles.get("left_shoulder_angle", 0)
    right_shoulder_angle = angles.get("right_shoulder_angle", 0)

    # Posture check: shoulders should be stable (not shrugging) and the press should be controlled
    # This assumes a good form, a better check would involve the hip-to-shoulder angle
    posture_correct = is_between(elbow_angle, 80, 180) and is_between(left_shoulder_angle, 10, 40) and is_between(right_shoulder_angle, 10, 40)
    
    # Repetition logic for bench press
    if state == "up" and elbow_angle < 100:
        state = "down"
    elif state == "down" and elbow_angle > 160:
        reps += 1
        state = "up"

    return posture_correct, state, reps

# --- Dispatcher ---
EXERCISE_RULES = {
    "squat": check_squat,
    "push-up": check_pushup,
    "hammer curl": check_hammer_curl,
    "deadlift": check_deadlift,
    "lateral raise": check_lateral_raise,
    "russian twist": check_russian_twist,
    "shoulder press": check_shoulder_press,
    "plank": check_plank,
    "bench press": check_bench_press,
}