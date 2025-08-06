import numpy as np

def create_training_data_for_preferences(segments, preferences):
    """
    Prepare training data using full segments and provided pairwise preferences.
    
    Args:
        segments: List of segments, each a list of (state, action, reward, done) tuples.
        preferences: List of (seg_i, seg_j, label) tuples, where label=1 means seg_i is better.
    
    Returns:
        training_obs: List of (traj_i, traj_j) tuples, each traj of shape (T', 1, 5, 5).
        training_labels: List of binary labels (1 if traj_i better, 0 if traj_j better).
        times: List of (time_i, time_j) tuples, where time_i, time_j are lists of indices.
        actions: List of (actions_i, actions_j) tuples, each of shape (T', 5).
    """
    max_traj_length = 0
    training_obs = []
    training_labels = []
    times = []
    actions = []
    
    for seg_i_idx, seg_j_idx, label in preferences:
        # Get segments
        seg_i = segments[seg_i_idx]
        seg_j = segments[seg_j_idx]
        
        # Extract full trajectories (all steps, ignoring done)
        traj_i = np.array([seg_i[i][0] for i in range(len(seg_i))])  # Shape: (T_i', 1, 5, 5)
        traj_j = np.array([seg_j[i][0] for i in range(len(seg_j))])  # Shape: (T_j', 1, 5, 5)
        actions_i = np.array([seg_i[i][1] for i in range(len(seg_i))])  # Shape: (T_i', 5)
        actions_j = np.array([seg_j[i][1] for i in range(len(seg_j))])  # Shape: (T_j', 5)
        
        # Store data
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)  # 0 if seg_i better, 1 if seg_j better
        times.append((list(range(len(seg_i))), list(range(len(seg_j)))))  # All indices
        actions.append((actions_i, actions_j))
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
    
    print("maximum traj length", max_traj_length)
    return training_obs, training_labels, times, actions

def process_preferences(preferences, segments):
    """
    Process preferences by comparing segment rewards and creating debugged preferences.
    
    Args:
        preferences: List of tuples (seg_i_idx, seg_j_idx, label)
        segments: List of segments containing reward data
    
    Returns:
        List of debugged preferences as tuples (seg_i_idx, seg_j_idx, 1)
    """
    debugged_preferences = []
    for seg_i_idx, seg_j_idx, label in preferences:
        # Get segments
        seg_i = segments[seg_i_idx]
        seg_j = segments[seg_j_idx]

        # Extract rewards
        rewards_i = [r for _, _, r, _ in seg_i]  # Shape: List of scalars
        rewards_j = [r for _, _, r, _ in seg_j]  # Shape: List of scalars
        
        # Compare sum of rewards
        if np.sum(rewards_i) == np.sum(rewards_j):
            print("equal")
            print(np.sum(rewards_i), " ", np.sum(rewards_j))
        elif np.sum(rewards_i) > np.sum(rewards_j):
            debugged_preferences.append((seg_i_idx, seg_j_idx, 0))
        elif np.sum(rewards_i) < np.sum(rewards_j):
            debugged_preferences.append((seg_j_idx, seg_i_idx, 0))
    
    return debugged_preferences

def create_training_data_for_corrections(corrections):
    """
    Prepare training data using correction feedback pairs.
    
    Args:
        corrections: List of (worse_segment, better_segment) tuples, where each segment is a list of (state, action, reward, done) tuples.
    
    Returns:
        training_obs: List of (traj_i, traj_j) tuples, where traj_i is from better_segment, traj_j from worse_segment, each traj of shape (T', 1, 5, 5).
        training_labels: List of binary labels (1 meaning traj_i is better).
        times: List of (time_i, time_j) tuples, where time_i, time_j are lists of indices.
        actions: List of (actions_i, actions_j) tuples, each of shape (T', 5).
    """
    max_traj_length = 0
    training_obs = []
    training_labels = []
    times = []
    actions = []
    
    for worse_seg, better_seg in corrections:
  
        traj_i = np.array([better_seg[i][0] for i in range(len(better_seg))])  # better: Shape: (T_i', 1, 5, 5)
        traj_j = np.array([worse_seg[i][0] for i in range(len(worse_seg))])  # worse: Shape: (T_j', 1, 5, 5)
        actions_i = np.array([better_seg[i][1] for i in range(len(better_seg))])  # Shape: (T_i', 5)
        actions_j = np.array([worse_seg[i][1] for i in range(len(worse_seg))])  # Shape: (T_j', 5)
        
        # Store data
        training_obs.append((traj_i, traj_j))
        training_labels.append(0)  # 0 since traj_i (better) is preferred
        times.append((list(range(len(better_seg))), list(range(len(worse_seg)))))  # All indices
        actions.append((actions_i, actions_j))
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
    
    print("maximum traj length", max_traj_length)
    return training_obs, training_labels, times, actions