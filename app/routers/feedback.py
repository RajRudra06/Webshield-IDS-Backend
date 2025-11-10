from fastapi import APIRouter, Body
from ..models import Feedback

router = APIRouter()
_FEEDBACKS = []

@router.post("/", response_model=dict)
def submit_feedback(data: Feedback = Body(...)):
    _FEEDBACKS.append(data.dict())
    return {"status": "received", "count": len(_FEEDBACKS)}


# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from ..utils.rl_Implementation.rl_agent import rl_agent
# from ..utils.rl_Implementation.prediction_buffer import prediction_buffer
# from ..utils.rl_Implementation.rl_reward_calculator import calculate_reward

# router = APIRouter()


# class FeedbackRequest(BaseModel):
#     request_id: str
#     actual_label: str  # 'benign', 'defacement', 'malware', 'phishing'


# @router.post("/submit")
# def submit_feedback(feedback: FeedbackRequest):
#     """
#     Submit ground truth feedback for RL learning
#     """
#     try:
#         # 1. Find the prediction using request_id
#         record = prediction_buffer.get(feedback.request_id)
        
#         if not record:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Request ID '{feedback.request_id}' not found. It may have expired."
#             )
        
#         # 2. Validate actual label
#         valid_labels = ['benign', 'defacement', 'malware', 'phishing']
#         if feedback.actual_label not in valid_labels:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid label. Must be one of: {valid_labels}"
#             )
        
#         # 3. Calculate reward (was it correct?)
#         reward = calculate_reward(
#             predicted_label=record['prediction'],
#             actual_label=feedback.actual_label,
#             confidence=record['confidence']
#         )
        
#         # 4. RL AGENT LEARNS from this feedback
#         rl_agent.update(
#             state_key=record['state'],
#             action=record['action'],
#             reward=reward,
#             next_state_key=record['state']
#         )
        
#         # 5. Save the learned Q-values periodically
#         if prediction_buffer.size() % 10 == 0:
#             rl_agent.save()
#             prediction_buffer.clear_old()
        
#         # 6. Return learning details
#         correct = (record['prediction'] == feedback.actual_label)
        
#         return {
#             "success": True,
#             "message": "Feedback received and RL agent updated",
#             "learning_details": {
#                 "request_id": feedback.request_id,
#                 "predicted": record['prediction'],
#                 "actual": feedback.actual_label,
#                 "correct": correct,
#                 "reward_given": round(reward, 2),
#                 "action_taken": rl_agent.action_names[record['action']],
#                 "state_explored": str(record['state'])
#             },
#             "agent_stats": {
#                 "total_states_learned": len(rl_agent.q_table),
#                 "buffer_size": prediction_buffer.size()
#             }
#         }
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"⚠️  Error in feedback submission: {e}")
#         raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")


# @router.get("/stats")
# def get_rl_stats():
#     """
#     Get RL agent learning statistics - CRITICAL FOR DEMONSTRATION!
#     """
#     try:
#         stats = rl_agent.get_stats()
#         sample_q = rl_agent.get_q_values_sample(n=5)
        
#         return {
#             "agent_configuration": stats,
#             "learning_progress": {
#                 "total_states_explored": len(rl_agent.q_table),
#                 "predictions_buffered": prediction_buffer.size()
#             },
#             "sample_q_values": sample_q,
#             "interpretation": {
#                 "epsilon": "Exploration rate - higher means more random actions",
#                 "states_explored": "Number of unique URL patterns encountered",
#                 "q_values": "Expected rewards for each action in given state (higher = better)"
#             }
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# @router.post("/reset")
# def reset_rl_agent():
#     """
#     Reset RL agent (optional - for testing/demo purposes)
#     """
#     try:
#         rl_agent.q_table.clear()
#         rl_agent.save()
        
#         return {
#             "success": True,
#             "message": "RL agent has been reset to initial state"
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
# # ```

# # ---

# # ## **🎯 Why You NEED the `/stats` Endpoint:**

# # ### **Without `/stats`:**
# # ❌ You submit feedback but **can't prove learning happened**
# # ❌ Can't show Q-values updating
# # ❌ Can't demonstrate RL is working
# # ❌ **Your presentation will fail!**

# # ### **With `/stats`:**
# # ✅ Shows Q-values updating in real-time
# # ✅ Shows number of states explored
# # ✅ **PROVES** reinforcement learning is happening
# # ✅ Makes your demo convincing

# # ---

# ## **📊 In Swagger UI You'll See:**

# ### **After adding complete code:**
# # ```
# # POST /feedback/submit    ← Submit ground truth
# # GET  /feedback/stats     ← SHOW LEARNING (critical!)
# # POST /feedback/reset     ← Reset for fresh demo