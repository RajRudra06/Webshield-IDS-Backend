def calculate_reward(predicted_label, actual_label, confidence):
    
    correct = (predicted_label == actual_label)
    is_threat_actual = (actual_label != 'benign')
    is_threat_predicted = (predicted_label != 'benign')
    
    if correct:
        # Base reward for correct prediction
        reward = 1.0
        
        # Bonus for high confidence correct predictions
        if confidence > 0.85:
            reward += 0.5
        elif confidence > 0.70:
            reward += 0.3
        
        # Extra bonus for correctly detecting threats
        if is_threat_actual:
            reward += 0.3
        
    else:
        # Base penalty for incorrect prediction
        reward = -1.0
        
        # Heavy penalty for false negatives (missing threats)
        if is_threat_actual and not is_threat_predicted:
            reward -= 0.5  # Critical mistake
        
        # Moderate penalty for false positives
        elif not is_threat_actual and is_threat_predicted:
            reward -= 0.2  # Annoying but less critical
        
        # Penalty for low confidence wrong predictions
        if confidence < 0.6:
            reward -= 0.1  # Should have been uncertain
    
    return reward