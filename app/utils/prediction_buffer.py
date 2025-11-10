from collections import deque
from datetime import datetime
import uuid

class PredictionBuffer:
    """
    Store recent predictions for feedback learning
    """
    
    def __init__(self, maxlen=1000):
        self.buffer = deque(maxlen=maxlen)
        self.lookup = {}  # For fast ID lookup
    
    def add(self, state_key, action, prediction, confidence, url, probabilities):
        """Add a prediction to the buffer"""
        request_id = str(uuid.uuid4())
        
        record = {
            'id': request_id,
            'state': state_key,
            'action': action,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities,
            'url': url,
            'timestamp': datetime.now().isoformat()
        }
        
        self.buffer.append(record)
        self.lookup[request_id] = record
        
        return request_id
    
    def get(self, request_id):
        """Retrieve a prediction by ID"""
        return self.lookup.get(request_id)
    
    def size(self):
        """Get current buffer size"""
        return len(self.buffer)
    
    def clear_old(self):
        """Clear old entries to maintain lookup dict"""
        # Keep lookup dict in sync with deque
        current_ids = {record['id'] for record in self.buffer}
        self.lookup = {k: v for k, v in self.lookup.items() if k in current_ids}


# Global prediction buffer
prediction_buffer = PredictionBuffer()